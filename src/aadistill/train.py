"""Stage 3 recovery trainer: offline KD/SFT over packed Stage 2 blocks.

One trainer covers the recovery sub-stages (AGENTS.md 4.5) through config:
`trainable_patterns` selects which parameters train (sub-stage 1 freezes
attention/embeddings and recovers FFN + norms; sub-stage 4 trains "all"),
and `loss` mixes masked next-token CE with on-the-fly teacher KD (forward KL
on the teacher's full-vocab distribution, computed by running the teacher on
the same packed blocks — no cached logits, per the 2026-07-21 mixture
decision). `kd_scope` chooses whether KD applies at every prediction
position ("all", dense signal including context tokens) or only where the
CE mask is on ("assistant").

Reproducibility contract:
- Block order is an infinite deterministic stream: epoch e's permutation is
  derived from (seed, e) alone, and a run's position in the stream is just
  `step * blocks_per_step`, so resume needs no dataloader state.
- Checkpoints hold the student (save_pretrained, runtime-loadable) plus
  optimizer state, step counter, RNG state, and the config hash; `restore`
  refuses a checkpoint written under a different config or freeze set.
- The training log is append-only jsonl (AGENTS.md 3.7); resumed runs keep
  appending to the same file.

Numerics: master weights in the configured dtype (float32 for real runs per
the Stage 3 decision record), optional bf16 autocast for compute; CE and KD
are always reduced in float32. KD softmaxes are chunked over positions to
bound the float32 peak at large vocab.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import encode_sample, load_split, pack_blocks
from .manifest import sha256_json

KD_SCOPES = ("all", "assistant")


def validate_train_config(cfg: dict) -> None:
    """Fail loudly on a missing or mistyped config field (AGENTS.md 2.3)."""

    def need(d, key, types, ctx=""):
        if key not in d:
            raise ValueError(f"config missing {ctx}{key!r}")
        if types is not None and not isinstance(d[key], types):
            raise ValueError(f"config field {ctx}{key!r} has wrong type")
        return d[key]

    for key in ("stage", "run_name", "student_path", "data_dir", "out_dir"):
        need(cfg, key, str)
    need(cfg, "block_len", int)
    need(cfg, "seed", int)
    if need(cfg, "dtype", str) not in ("float32", "bfloat16"):
        raise ValueError(f"unsupported dtype {cfg['dtype']!r}")
    need(cfg, "device", str)
    need(cfg, "groups", (list, type(None)))
    teacher = need(cfg, "teacher", (dict, type(None)))
    if teacher is not None:
        for key in ("model_id", "revision", "dtype"):
            need(teacher, key, str, "teacher.")
    patterns = need(cfg, "trainable_patterns", (list, str))
    if isinstance(patterns, str) and patterns != "all":
        raise ValueError("trainable_patterns must be 'all' or a list of regexes")
    loss = need(cfg, "loss", dict)
    for key in ("ce_weight", "kd_weight", "kd_temperature"):
        need(loss, key, (int, float), "loss.")
    if need(loss, "kd_scope", str, "loss.") not in KD_SCOPES:
        raise ValueError(f"loss.kd_scope must be one of {KD_SCOPES}")
    optim = need(cfg, "optim", dict)
    for key in ("lr", "weight_decay", "eps", "grad_clip"):
        need(optim, key, (int, float), "optim.")
    if len(need(optim, "betas", list, "optim.")) != 2:
        raise ValueError("optim.betas must have two entries")
    sched = need(cfg, "schedule", dict)
    for key in ("total_steps", "warmup_steps"):
        need(sched, key, int, "schedule.")
    need(sched, "min_lr_frac", (int, float), "schedule.")
    batch = need(cfg, "batch", dict)
    for key in ("blocks_per_step", "micro_blocks"):
        if need(batch, key, int, "batch.") < 1:
            raise ValueError(f"batch.{key} must be >= 1")
    ck = need(cfg, "checkpoint", dict)
    for key in ("save_every", "keep_last"):
        need(ck, key, int, "checkpoint.")
    iv = need(cfg, "intervals", dict)
    for key in ("log_every", "eval_every", "eval_blocks"):
        need(iv, key, int, "intervals.")


def build_blocks(tokenizer, data_dir, split, block_len, groups=None):
    """Encode one split of the Stage 2 mixture into packed training blocks.

    Packs per group (a block never straddles groups, keeping attribution)
    and returns (input_ids [N, L], loss_mask [N, L], block_groups, stats).
    """
    loaded = load_split(data_dir, split)
    if groups is not None:
        missing = [g for g in groups if g not in loaded]
        if missing:
            raise ValueError(f"groups {missing} not present in {split} split")
        loaded = {g: loaded[g] for g in groups}
    ids_parts, mask_parts, block_groups = [], [], []
    stats = {}
    for group in sorted(loaded):
        encoded = [encode_sample(tokenizer, s) for s in loaded[group]]
        ids, mask, dropped = pack_blocks(encoded, block_len)
        stats[group] = {
            "samples": len(encoded),
            "blocks": int(ids.shape[0]),
            "dropped_tail_tokens": dropped,
        }
        ids_parts.append(ids)
        mask_parts.append(mask)
        block_groups += [group] * int(ids.shape[0])
    input_ids = torch.cat(ids_parts)
    loss_mask = torch.cat(mask_parts)
    if input_ids.shape[0] == 0:
        raise ValueError(f"{split} split produced no blocks at block_len={block_len}")
    return input_ids, loss_mask, block_groups, stats


def prediction_mask(loss_mask: torch.Tensor, scope: str) -> torch.Tensor:
    """Boolean [B, T-1] mask of prediction positions for KD."""
    if scope == "all":
        return torch.ones_like(loss_mask[:, 1:])
    if scope == "assistant":
        return loss_mask[:, 1:].clone()
    raise ValueError(f"unknown kd scope {scope!r}")


def masked_ce(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor):
    """Summed next-token CE over positions whose target token is trainable.

    Returns (sum_loss, n_targets); sum_loss keeps the graph and is reduced
    in float32.
    """
    pos = loss_mask[:, 1:]
    count = int(pos.sum())
    if count == 0:
        return logits.sum() * 0.0, 0
    sel = logits[:, :-1][pos]
    targets = input_ids[:, 1:][pos]
    return F.cross_entropy(sel.float(), targets, reduction="sum"), count


def kd_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float = 1.0,
    chunk: int = 512,
):
    """Summed tau^2 * KL(teacher_tau || student_tau) over prediction positions.

    Chunked over positions so the float32 softmax peak stays bounded at
    large vocab. Returns (sum_loss, n_positions).
    """
    sp = student_logits[:, :-1][pos_mask]
    tp = teacher_logits[:, :-1][pos_mask]
    count = int(sp.shape[0])
    if count == 0:
        return student_logits.sum() * 0.0, 0
    total = None
    for i in range(0, count, chunk):
        s = torch.log_softmax(sp[i : i + chunk].float() / temperature, dim=-1)
        t = torch.log_softmax(tp[i : i + chunk].float() / temperature, dim=-1)
        kl = (t.exp() * (t - s)).sum()
        total = kl if total is None else total + kl
    return total * (temperature * temperature), count


def select_trainable(model, patterns) -> dict:
    """Set requires_grad per parameter name; 'all' or a list of regexes."""
    if patterns == "all":
        names = []
        for name, param in model.named_parameters():
            param.requires_grad_(True)
            names.append(name)
    else:
        regexes = [re.compile(p) for p in patterns]
        names = []
        for name, param in model.named_parameters():
            keep = any(r.search(name) for r in regexes)
            param.requires_grad_(keep)
            if keep:
                names.append(name)
        if not names:
            raise ValueError(f"no parameters match trainable_patterns {patterns}")
    return {
        "trainable_names": names,
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_lr_frac: float) -> float:
    """Linear warmup to 1.0, then cosine decay to min_lr_frac at total_steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def epoch_permutation(n_blocks: int, seed: int, epoch: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n_blocks, generator=g)


def stream_block_indices(n_blocks: int, seed: int, start: int, count: int) -> list[int]:
    """Slice [start, start+count) of the infinite deterministic block stream.

    The stream is epoch 0's permutation, then epoch 1's, ... — a pure
    function of (n_blocks, seed), so any position can be re-derived exactly
    from the global consumed-block counter alone (exact resume).
    """
    epoch, pos = divmod(start, n_blocks)
    perm = epoch_permutation(n_blocks, seed, epoch)
    out = []
    for _ in range(count):
        out.append(int(perm[pos]))
        pos += 1
        if pos == n_blocks:
            epoch, pos = epoch + 1, 0
            perm = epoch_permutation(n_blocks, seed, epoch)
    return out


class JsonlLogger:
    """Append-only jsonl event log (AGENTS.md 3.7). Never overwrites."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields) -> dict:
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **{k: v for k, v in fields.items() if v is not None},
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


class Trainer:
    """Config-driven recovery trainer with exact resume.

    `train_blocks` / `val_blocks` are (input_ids, loss_mask) CPU tensor
    pairs from `build_blocks` (or synthetic ones in tests); batches move to
    `device` per microbatch.
    """

    def __init__(
        self,
        cfg: dict,
        student,
        train_blocks,
        val_blocks=None,
        teacher=None,
        device: str = "cpu",
        out_dir: str | Path | None = None,
        logger: JsonlLogger | None = None,
    ):
        validate_train_config(cfg)
        loss_cfg = cfg["loss"]
        if loss_cfg["kd_weight"] > 0 and teacher is None:
            raise ValueError("loss.kd_weight > 0 requires a teacher model")
        if loss_cfg["ce_weight"] <= 0 and loss_cfg["kd_weight"] <= 0:
            raise ValueError("at least one of ce_weight / kd_weight must be > 0")
        self.cfg = cfg
        self.config_sha = sha256_json(cfg)
        self.device = device
        self.out_dir = Path(out_dir) if out_dir is not None else Path(cfg["out_dir"])
        self.logger = logger or JsonlLogger(self.out_dir / "train_log.jsonl")

        self.student = student.to(device)
        self.student.config.use_cache = False
        if cfg.get("gradient_checkpointing"):
            self.student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.teacher = None
        if teacher is not None:
            self.teacher = teacher.to(device).eval()
            self.teacher.config.use_cache = False
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        self.freeze_report = select_trainable(self.student, cfg["trainable_patterns"])
        self.params = [p for _, p in self.student.named_parameters() if p.requires_grad]
        opt_cfg = cfg["optim"]
        self.opt = torch.optim.AdamW(
            self.params,
            lr=opt_cfg["lr"],
            betas=tuple(opt_cfg["betas"]),
            eps=opt_cfg["eps"],
            weight_decay=opt_cfg["weight_decay"],
        )

        self.train_ids, self.train_mask = train_blocks
        if self.train_ids.shape[0] == 0:
            raise ValueError("no training blocks")
        if val_blocks is not None:
            val_ids, val_mask = val_blocks
            # Fixed shuffle so a truncated eval (eval_blocks < all) still
            # samples across groups instead of the first groups alphabetically.
            perm = epoch_permutation(val_ids.shape[0], cfg["seed"] + 777, 0)
            self.val_ids, self.val_mask = val_ids[perm], val_mask[perm]
        else:
            self.val_ids = self.val_mask = None
        self.step = 0

    def _autocast(self):
        if self.cfg.get("autocast_bf16"):
            dev_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            return torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
        return nullcontext()

    def _micro_losses(self, ids: torch.Tensor, mask: torch.Tensor):
        """Forward one microbatch; returns (ce_sum, ce_n, kd_sum, kd_n)."""
        loss_cfg = self.cfg["loss"]
        with self._autocast():
            logits = self.student(ids).logits
        ce_sum, ce_n = masked_ce(logits, ids, mask)
        kd_sum, kd_n = torch.zeros((), device=ids.device), 0
        if self.teacher is not None and loss_cfg["kd_weight"] > 0:
            with torch.no_grad(), self._autocast():
                t_logits = self.teacher(ids).logits
            kd_sum, kd_n = kd_forward_kl(
                logits,
                t_logits,
                prediction_mask(mask, loss_cfg["kd_scope"]),
                loss_cfg["kd_temperature"],
            )
        return ce_sum, ce_n, kd_sum, kd_n

    def step_once(self) -> dict:
        """One optimizer step over blocks_per_step blocks (grad accumulation)."""
        cfg = self.cfg
        bps = cfg["batch"]["blocks_per_step"]
        micro = cfg["batch"]["micro_blocks"]
        loss_cfg, sched = cfg["loss"], cfg["schedule"]
        idxs = stream_block_indices(
            self.train_ids.shape[0], cfg["seed"], self.step * bps, bps
        )
        ids, mask = self.train_ids[idxs], self.train_mask[idxs]
        # Normalizers are known from the masks alone, so microbatch losses can
        # be scaled exactly before backward (sum over micro = true step mean).
        ce_total = int(mask[:, 1:].sum()) if loss_cfg["ce_weight"] > 0 else 0
        kd_total = (
            int(prediction_mask(mask, loss_cfg["kd_scope"]).sum())
            if self.teacher is not None and loss_cfg["kd_weight"] > 0
            else 0
        )
        lr = cfg["optim"]["lr"] * lr_factor(
            self.step, sched["total_steps"], sched["warmup_steps"], sched["min_lr_frac"]
        )
        for group in self.opt.param_groups:
            group["lr"] = lr

        self.student.train()
        self.opt.zero_grad(set_to_none=True)
        ce_acc = kd_acc = 0.0
        for i in range(0, bps, micro):
            mids = ids[i : i + micro].to(self.device)
            mmask = mask[i : i + micro].to(self.device)
            ce_sum, _, kd_sum, _ = self._micro_losses(mids, mmask)
            loss = torch.zeros((), device=self.device)
            if ce_total > 0:
                loss = loss + loss_cfg["ce_weight"] * ce_sum / ce_total
            if kd_total > 0:
                loss = loss + loss_cfg["kd_weight"] * kd_sum / kd_total
            if loss.requires_grad:
                loss.backward()
            ce_acc += float(ce_sum.detach())
            kd_acc += float(kd_sum.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.params, cfg["optim"]["grad_clip"]
        )
        self.opt.step()
        self.step += 1

        ce_mean = ce_acc / ce_total if ce_total else None
        kd_mean = kd_acc / kd_total if kd_total else None
        total = sum(
            w * m
            for w, m in (
                (loss_cfg["ce_weight"], ce_mean),
                (loss_cfg["kd_weight"], kd_mean),
            )
            if m is not None
        )
        return {
            "step": self.step,
            "loss": round(total, 6),
            "ce": round(ce_mean, 6) if ce_mean is not None else None,
            "kd": round(kd_mean, 6) if kd_mean is not None else None,
            "lr": lr,
            "grad_norm": round(float(grad_norm), 4),
            "ce_targets": ce_total,
            "kd_positions": kd_total,
        }

    @torch.no_grad()
    def evaluate(self, max_blocks: int | None = None) -> dict:
        """CE (assistant targets) and KD metrics over the fixed val order."""
        if self.val_ids is None:
            raise ValueError("trainer has no validation blocks")
        n = self.val_ids.shape[0]
        if max_blocks:
            n = min(n, max_blocks)
        micro = self.cfg["batch"]["micro_blocks"]
        was_training = self.student.training
        self.student.eval()
        ce_s = kd_s = 0.0
        ce_n = kd_n = 0
        for i in range(0, n, micro):
            ids = self.val_ids[i : i + micro].to(self.device)
            mask = self.val_mask[i : i + micro].to(self.device)
            ce_sum, cn, kd_sum, kn = self._micro_losses(ids, mask)
            ce_s += float(ce_sum)
            ce_n += cn
            kd_s += float(kd_sum)
            kd_n += kn
        if was_training:
            self.student.train()
        out = {"val_blocks": n}
        if ce_n:
            out["val_ce"] = round(ce_s / ce_n, 6)
            out["val_ppl"] = round(math.exp(min(ce_s / ce_n, 30.0)), 4)
        if kd_n:
            out["val_kd"] = round(kd_s / kd_n, 6)
        return out

    def save_checkpoint(self) -> Path:
        tag = f"step_{self.step:06d}"
        ckpt_dir = self.out_dir / "checkpoints" / tag
        self.student.save_pretrained(ckpt_dir / "model")
        torch.save(
            {
                "step": self.step,
                "optimizer": self.opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "config_sha256": self.config_sha,
                "trainable_names": self.freeze_report["trainable_names"],
            },
            ckpt_dir / "trainer_state.pt",
        )
        (self.out_dir / "checkpoints" / "latest.txt").write_text(tag + "\n")
        keep = self.cfg["checkpoint"]["keep_last"]
        if keep > 0:
            stale = sorted((self.out_dir / "checkpoints").glob("step_*"))[:-keep]
            for d in stale:
                shutil.rmtree(d)
        self.logger.log("checkpoint_saved", step=self.step, path=str(ckpt_dir))
        return ckpt_dir

    def restore(self, ckpt_dir: str | Path) -> None:
        """Resume optimizer/counters from a checkpoint written by this config.

        The caller loads the student weights from ``<ckpt_dir>/model`` before
        constructing the Trainer; this restores everything else.
        """
        state = torch.load(Path(ckpt_dir) / "trainer_state.pt", weights_only=True)
        if state["config_sha256"] != self.config_sha:
            raise ValueError(
                "checkpoint was written under a different config "
                f"({state['config_sha256'][:12]} != {self.config_sha[:12]}); "
                "refusing to resume"
            )
        if list(state["trainable_names"]) != self.freeze_report["trainable_names"]:
            raise ValueError("checkpoint freeze set differs from current config")
        self.opt.load_state_dict(state["optimizer"])
        self.step = int(state["step"])
        torch.set_rng_state(state["torch_rng_state"])
        self.logger.log("resume_loaded", step=self.step, checkpoint=str(ckpt_dir))

    def run(self) -> dict:
        """Train to total_steps with periodic logging/eval/checkpointing."""
        cfg = self.cfg
        total = cfg["schedule"]["total_steps"]
        iv, ck = cfg["intervals"], cfg["checkpoint"]
        bps = cfg["batch"]["blocks_per_step"]
        block_len = int(self.train_ids.shape[1])
        eval_blocks = iv["eval_blocks"] or None
        t_start = time.time()

        if self.step == 0:
            self.logger.log(
                "run_start",
                run_name=cfg["run_name"],
                config_sha256=self.config_sha,
                total_steps=total,
                train_blocks=int(self.train_ids.shape[0]),
                val_blocks=int(self.val_ids.shape[0]) if self.val_ids is not None else 0,
                **{
                    k: self.freeze_report[k]
                    for k in ("trainable_params", "total_params")
                },
            )
            if self.val_ids is not None:
                ev = self.evaluate(eval_blocks)
                self.logger.log("eval_result", step=0, **ev)
                print(f"eval step 0: {ev}", flush=True)

        while self.step < total:
            t0 = time.time()
            m = self.step_once()
            m["seconds"] = round(time.time() - t0, 2)
            m["tokens_seen"] = self.step * bps * block_len
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                m["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if iv["log_every"] and self.step % iv["log_every"] == 0:
                self.logger.log("train_step", **m)
                parts = [f"step {self.step}/{total}", f"loss {m['loss']:.4f}"]
                if m["ce"] is not None:
                    parts.append(f"ce {m['ce']:.4f}")
                if m["kd"] is not None:
                    parts.append(f"kd {m['kd']:.4f}")
                parts += [f"lr {m['lr']:.2e}", f"{m['seconds']}s"]
                print("  ".join(parts), flush=True)
            at_end = self.step >= total
            if (
                iv["eval_every"]
                and self.val_ids is not None
                and self.step % iv["eval_every"] == 0
                and not at_end
            ):
                ev = self.evaluate(eval_blocks)
                self.logger.log("eval_result", step=self.step, **ev)
                print(f"eval step {self.step}: {ev}", flush=True)
            if ck["save_every"] and self.step % ck["save_every"] == 0 and not at_end:
                self.save_checkpoint()

        final_eval = None
        if self.val_ids is not None:
            final_eval = self.evaluate(eval_blocks)
            self.logger.log("eval_result", step=self.step, **final_eval)
            print(f"eval step {self.step} (final): {final_eval}", flush=True)
        ckpt_dir = self.save_checkpoint()
        self.logger.log(
            "run_end", steps=self.step, seconds=round(time.time() - t_start, 1)
        )
        return {
            "steps": self.step,
            "final_eval": final_eval,
            "checkpoint": str(ckpt_dir),
        }
