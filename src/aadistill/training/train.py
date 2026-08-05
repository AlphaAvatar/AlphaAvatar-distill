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

An optional `lora` config adds low-rank adapters to selected linear modules
(Experiment 3 arm A2: attention q/k/v/o) while their base matrices stay frozen.
The saved `model/` directory is always the *merged*, adapter-free Hugging Face
checkpoint, so every arm is evaluated through the same inference architecture;
see `lora.py` for the design and the exact-resume argument.

Reproducibility contract:
- Block order is an infinite deterministic stream: epoch e's permutation is
  derived from (seed, e) alone, and a run's position in the stream is just
  `step * blocks_per_step`, so resume needs no dataloader state.
- The LR schedule is a pure function of the step counter, so there is no
  separate scheduler state to save either.
- Checkpoints hold the student (save_pretrained, runtime-loadable) plus
  optimizer state, step counter, consumed-block position, RNG state, the
  config hash and — for a LoRA run — the adapter config and the unmerged base
  and LoRA tensors; `restore` refuses a checkpoint written under a different
  config, freeze set or adapter spec.
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

from ..data.dataset import best_fit_blocks, encode_sample, load_split, pack_blocks
from ..infrastructure.manifest import sha256_json
from .lora import (
    LoRAConfig,
    apply_lora,
    lora_and_base_tensors,
    lora_report,
    load_lora_and_base_,
    merged_state_dict,
)

KD_SCOPES = ("all", "assistant", "all_no_think")
PACKINGS = ("concat", "best_fit", "ladder")
LORA_SUFFIXES = (".lora_A", ".lora_B")


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
    # Optional so the four logged runs' configs stay valid; absent means the
    # concat path they actually ran.
    if cfg.get("packing", "concat") not in PACKINGS:
        raise ValueError(f"config field 'packing' must be one of {PACKINGS}")
    # `"ladder"` trains on a pre-packed token ladder: `data_dir` is the pack and
    # `rung` selects the prefix. Nothing is re-encoded or re-packed, so the rung
    # trained is the rung the gate measured.
    if cfg.get("packing") == "ladder":
        need(cfg, "rung", int)
        if cfg.get("groups") is not None:
            raise ValueError("packing 'ladder' cannot subset groups; the pack "
                             "already fixes the mixture")
        if cfg.get("extra_val") is not None:
            raise ValueError("packing 'ladder' does not support extra_val")
        val_blocks = cfg.get("val_blocks", 16)
        if not isinstance(val_blocks, int) or val_blocks < 1:
            raise ValueError("config field 'val_blocks' must be a positive int")
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
    # LoRA tensors are optimized by the *same* single AdamW group, learning rate,
    # schedule and weight-decay semantics as every other trainable parameter.
    # There is deliberately no separate LoRA learning rate or parameter group:
    # A2 isolates low-rank attention parameterization, not adapter tuning, and a
    # second optimizer setting would be a second variable.
    if cfg.get("lora") is not None:
        if not isinstance(cfg["lora"], dict):
            raise ValueError("config field 'lora' must be a dict")
        LoRAConfig.from_dict(cfg["lora"])          # raises on a bad adapter spec
    for field in ("lora_lr", "lora_weight_decay", "no_decay_patterns"):
        if field in optim:
            raise ValueError(
                f"optim.{field} is not supported: LoRA and full-rank parameters "
                "share one optimizer group with identical settings")
    sched = need(cfg, "schedule", dict)
    for key in ("total_steps", "warmup_steps"):
        need(sched, key, int, "schedule.")
    need(sched, "min_lr_frac", (int, float), "schedule.")
    batch = need(cfg, "batch", dict)
    for key in ("blocks_per_step", "micro_blocks"):
        if need(batch, key, int, "batch.") < 1:
            raise ValueError(f"batch.{key} must be >= 1")
    if "truncate_padding" in batch and not isinstance(batch["truncate_padding"], bool):
        raise ValueError("batch.truncate_padding must be a bool")
    ck = need(cfg, "checkpoint", dict)
    for key in ("save_every", "keep_last"):
        need(ck, key, int, "checkpoint.")
    iv = need(cfg, "intervals", dict)
    for key in ("log_every", "eval_every", "eval_blocks"):
        need(iv, key, int, "intervals.")
    extra = cfg.get("extra_val")
    if extra is not None:
        if not isinstance(extra, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra.items()
        ):
            raise ValueError("extra_val must map val-set names to data dirs")
        if "val" in extra:
            raise ValueError("extra_val name 'val' collides with the primary val")


def build_blocks(
    tokenizer, data_dir, split, block_len, groups=None, packing="concat", seed=0
):
    """Encode one split of the Stage 2 mixture into packed training blocks.

    Packs per group (a block never straddles groups, keeping attribution)
    and returns (input_ids [N, L], loss_mask [N, L], block_groups, stats).

    `packing` selects the block-building strategy:

    * `"concat"` — concatenate-then-cut (`pack_blocks`). Samples straddle block
      boundaries. This is the data path of the four logged Stage 3 runs, so it
      stays the default: changing it would silently make those runs
      irreproducible from their configs.
    * `"best_fit"` — length-aware bin packing (`best_fit_blocks`). No sample is
      split across a boundary; residual capacity is padded and masked out.

    `seed` only affects `"best_fit"`, whose placement order is seeded so block
    contents are a pure function of (seed, block_len).
    """
    if packing not in PACKINGS:
        raise ValueError(f"packing must be one of {PACKINGS}, got {packing!r}")
    loaded = load_split(data_dir, split)
    if groups is not None:
        missing = [g for g in groups if g not in loaded]
        if missing:
            raise ValueError(f"groups {missing} not present in {split} split")
        loaded = {g: loaded[g] for g in groups}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if packing == "best_fit" and pad_id is None:
        raise ValueError("best_fit packing needs a pad or eos token id")
    ids_parts, mask_parts, content_parts, block_groups = [], [], [], []
    stats = {}
    for group in sorted(loaded):
        encoded = [encode_sample(tokenizer, s) for s in loaded[group]]
        if packing == "best_fit":
            ids, mask, content, group_stats = best_fit_blocks(
                encoded, block_len, pad_id=pad_id, seed=seed,
                return_content_mask=True,
            )
            content_parts.append(content)
        else:
            ids, mask, dropped = pack_blocks(encoded, block_len)
            group_stats = {
                "samples": len(encoded),
                "blocks": int(ids.shape[0]),
                "dropped_tail_tokens": dropped,
            }
        stats[group] = group_stats
        ids_parts.append(ids)
        mask_parts.append(mask)
        block_groups += [group] * int(ids.shape[0])
    input_ids = torch.cat(ids_parts)
    loss_mask = torch.cat(mask_parts)
    # None under concat packing, which has no padding to exclude.
    content_mask = torch.cat(content_parts) if content_parts else None
    if input_ids.shape[0] == 0:
        raise ValueError(f"{split} split produced no blocks at block_len={block_len}")
    return input_ids, loss_mask, block_groups, stats, content_mask


def think_span_mask(
    input_ids: torch.Tensor, open_id: int, close_id: int
) -> torch.Tensor:
    """Boolean [B, T], True on tokens inside a think span, `</think>` included.

    Marks the region `<think> … </think>`. Under the empty-think rendering that
    is only 3-4 tokens per sample, but they are the tokens the teacher and the
    CE target disagree about most sharply (experiment log 2026-07-28).

    Computed from ids rather than tracked through packing so it is correct
    regardless of how blocks were built, and so a block holding several packed
    samples gets every one of its spans marked.
    """
    opens = (input_ids == open_id).int()
    closes = (input_ids == close_id).int()
    # Open contributes from its own position; close ends the run *after* itself,
    # so the closing tag is added back explicitly.
    inside = (opens.cumsum(1) - closes.cumsum(1)) > 0
    return inside | closes.bool()


def prediction_mask(
    loss_mask: torch.Tensor,
    scope: str,
    content_mask: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    think_ids: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Boolean [B, T-1] mask of prediction positions for KD.

    `content_mask` marks real (non-padding) tokens. Scope `"all"` means every
    *real* position: under padded packing the pad run is not a prediction the
    student should be matched to, and counting it would also inflate the KD
    normalizer. Absent a content mask (concat packing never pads), `"all"` is
    every position, which is what the four logged runs trained on.

    `"all_no_think"` is `"all"` minus the template-inserted think span. The
    reason it exists: with empty-think targets the teacher is forced through a
    protocol it would never produce, and at `</think>` it puts p≈0 on the very
    token CE demands — so KD there transmits a contradiction rather than
    knowledge, at 2× CE's per-position weight
    (`logs/EXPERIMENTS.md`). This scope
    removes the contested positions and leaves the rest of KD untouched.
    """
    if scope == "assistant":
        return loss_mask[:, 1:].clone()
    if scope not in ("all", "all_no_think"):
        raise ValueError(f"unknown kd scope {scope!r}")

    if content_mask is None:
        keep = torch.ones_like(loss_mask, dtype=torch.bool)
    else:
        keep = content_mask.clone()
    if scope == "all_no_think":
        if input_ids is None or think_ids is None:
            raise ValueError(
                "kd_scope 'all_no_think' needs input_ids and think_ids; "
                "the caller must resolve <think>/</think> from the tokenizer"
            )
        keep = keep & ~think_span_mask(input_ids, *think_ids)
    return keep[:, 1:]


def nonpad_extent(content_mask: torch.Tensor) -> int:
    """Positions that must be forwarded for a microbatch: 1 + the last real one.

    Padding is a contiguous *suffix* of every packed block (`pack_sessions`
    appends the pad run after the last real token), and attention is causal, so
    no real token's hidden state depends on a position at or beyond this extent.
    Everything past it therefore contributes nothing to CE, KD or the gradient
    and does not need to be forwarded.

    The contiguity that licenses this is **checked, not assumed** — a future
    packer that interleaved padding would silently invalidate the optimization,
    so a violating block raises instead of being quietly mis-trained.
    """
    if content_mask.dtype != torch.bool:
        content_mask = content_mask.bool()
    counts = content_mask.sum(dim=1)
    positions = torch.arange(content_mask.shape[1], device=content_mask.device)
    if not torch.equal(content_mask, positions[None, :] < counts[:, None]):
        raise ValueError(
            "padding is not a contiguous suffix of every block: real tokens must "
            "form a prefix for suffix truncation to be sound. Set "
            "batch.truncate_padding=false, or fix the packer.")
    return int(counts.max())


def _unpack_blocks(blocks):
    """Accept (ids, loss_mask) or (ids, loss_mask, content_mask)."""
    if len(blocks) == 2:
        return blocks[0], blocks[1], None
    if len(blocks) == 3:
        return blocks
    raise ValueError("block tuples must be (ids, mask) or (ids, mask, content)")


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


def select_trainable(model, patterns, lora_modules=None) -> dict:
    """Set requires_grad per parameter name; 'all' or a list of regexes.

    LoRA parameters are **not** governed by `trainable_patterns`: an adapter
    exists only to be trained, and making it depend on a regex written for the
    base model invites a silent no-op arm. They are excluded from the regex
    sweep, forced trainable, and counted separately so a run can always state
    how much of its capacity was full-rank and how much was low-rank.
    """
    lora_modules = lora_modules or {}
    is_lora = {f"{m}{suffix}" for m in lora_modules for suffix in LORA_SUFFIXES}
    full_names, lora_names = [], []
    for name, param in model.named_parameters():
        if name in is_lora:
            param.requires_grad_(True)
            lora_names.append(name)
            continue
        keep = patterns == "all" or any(
            re.search(p, name) for p in patterns)
        param.requires_grad_(keep)
        if keep:
            full_names.append(name)
    if not full_names and patterns != "all":
        raise ValueError(f"no parameters match trainable_patterns {patterns}")
    full = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and n not in is_lora)
    lora = sum(p.numel() for n, p in model.named_parameters() if n in is_lora)
    return {
        "trainable_names": full_names + lora_names,
        "full_rank_trainable_names": full_names,
        "lora_trainable_names": lora_names,
        "trainable_params": full + lora,
        "full_rank_trainable_params": full,
        "lora_trainable_params": lora,
        # The base model's parameter count, i.e. what the merged deployable
        # checkpoint holds. LoRA tensors are excluded because they do not
        # survive the merge.
        "total_params": sum(p.numel() for n, p in model.named_parameters()
                            if n not in is_lora),
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
        extra_val_blocks: dict | None = None,
        think_ids: tuple[int, int] | None = None,
    ):
        validate_train_config(cfg)
        loss_cfg = cfg["loss"]
        if loss_cfg["kd_weight"] > 0 and teacher is None:
            raise ValueError("loss.kd_weight > 0 requires a teacher model")
        # Fail here rather than at the first backward: a run that silently fell
        # back to plain "all" would look like the treatment arm and quietly
        # invalidate the comparison it exists to make.
        if loss_cfg["kd_scope"] == "all_no_think" and think_ids is None:
            raise ValueError(
                "loss.kd_scope 'all_no_think' requires think_ids=(open, close); "
                "resolve them from the tokenizer and pass them to Trainer"
            )
        self.think_ids = think_ids
        if loss_cfg["ce_weight"] <= 0 and loss_cfg["kd_weight"] <= 0:
            raise ValueError("at least one of ce_weight / kd_weight must be > 0")
        self.cfg = cfg
        self.config_sha = sha256_json(cfg)
        # Skip the padding suffix in every forward. It changes no normalizer, no
        # logical block length and no supervised-token count — only the positions
        # actually pushed through the models.
        #
        # Default **off**, deliberately. The two paths agree mathematically but
        # not bitwise: a shorter sequence reorders float32 reductions, and Adam
        # amplifies the resulting ~1e-8 gradient differences on near-zero-gradient
        # components. Defaulting on would silently change what a previously
        # logged config computes, which is exactly what P4 forbids. Opting in per
        # config also changes that config's hash, so the manifest records which
        # path a run took instead of leaving it to the code version.
        self.truncate_padding = bool(cfg["batch"].get("truncate_padding", False))
        # Token accounting, kept as three distinct quantities so a saving is
        # never confused with a change in how much supervision the run saw.
        self._exec_logical = 0         # cumulative: blocks x logical block length
        self._exec_positions = 0       # cumulative: positions actually forwarded
        self._exec_nonpad = 0          # cumulative: real (non-pad) tokens forwarded
        self._step_positions = 0       # same two, for the current step only
        self._step_nonpad = 0
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

        # LoRA is applied *before* the freeze sweep and the optimizer is built,
        # so the adapter is part of the trainable set from step 0 and never
        # bolted onto a running optimizer.
        self.lora_cfg = (
            LoRAConfig.from_dict(cfg["lora"]) if cfg.get("lora") else None)
        self.lora_modules = (
            apply_lora(self.student, self.lora_cfg) if self.lora_cfg else {})

        self.freeze_report = select_trainable(
            self.student, cfg["trainable_patterns"], self.lora_modules)
        if self.lora_cfg:
            self.freeze_report.update(lora_report(self.lora_modules, self.lora_cfg))
        # One group, one learning rate, one weight decay — for full-rank and
        # LoRA parameters alike. Identical in shape and settings to every run
        # before Experiment 3, so A2 differs from A1 only by the adapter itself.
        self.params = [p for _, p in self.student.named_parameters() if p.requires_grad]
        opt_cfg = cfg["optim"]
        self.opt = torch.optim.AdamW(
            self.params,
            lr=opt_cfg["lr"],
            betas=tuple(opt_cfg["betas"]),
            eps=opt_cfg["eps"],
            weight_decay=opt_cfg["weight_decay"],
        )

        # Block tuples are (ids, loss_mask) or (ids, loss_mask, content_mask);
        # the content mask is present only under padded (best-fit) packing.
        self.train_ids, self.train_mask, self.train_content = _unpack_blocks(
            train_blocks
        )
        if self.train_ids.shape[0] == 0:
            raise ValueError("no training blocks")
        if val_blocks is not None:
            val_ids, val_mask, val_content = _unpack_blocks(val_blocks)
            # Fixed shuffle so a truncated eval (eval_blocks < all) still
            # samples across groups instead of the first groups alphabetically.
            perm = epoch_permutation(val_ids.shape[0], cfg["seed"] + 777, 0)
            self.val_ids, self.val_mask = val_ids[perm], val_mask[perm]
            self.val_content = None if val_content is None else val_content[perm]
        else:
            self.val_ids = self.val_mask = self.val_content = None
        # Named secondary val sets (e.g. the frozen val_v0 alongside a v1
        # mixture's own val split); evaluated and logged next to the primary.
        self.extra_vals: dict[str, tuple] = {}
        for name, blocks in (extra_val_blocks or {}).items():
            ids, mask, content = _unpack_blocks(blocks)
            perm = epoch_permutation(ids.shape[0], cfg["seed"] + 777, 0)
            self.extra_vals[name] = (
                ids[perm], mask[perm], None if content is None else content[perm]
            )
        if self.extra_vals and self.val_ids is None:
            raise ValueError("extra_val_blocks given but no primary val_blocks")
        self.step = 0

    def _autocast(self):
        if self.cfg.get("autocast_bf16"):
            dev_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            return torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
        return nullcontext()

    def _micro_losses(
        self, ids: torch.Tensor, mask: torch.Tensor, content: torch.Tensor | None = None
    ):
        """Forward one microbatch; returns (ce_sum, ce_n, kd_sum, kd_n).

        With `batch.truncate_padding` the microbatch is sliced to its non-pad
        extent before either model runs. Every sequence-aligned tensor is sliced
        together — `input_ids`, the CE mask and the content mask — so positions
        stay aligned; there is no separate attention mask or label tensor in this
        trainer (targets are `input_ids` shifted, and padding is excluded by the
        masks rather than by an attention mask).

        The logical block length is unchanged: `mask`/`content` past the extent
        are all False, so the CE and KD normalizers computed by the caller from
        the *full* masks are the same numbers this path produces. Only the work
        disappears.
        """
        loss_cfg = self.cfg["loss"]
        n_blocks, logical_len = ids.shape[0], ids.shape[1]
        nonpad = int(content.sum()) if content is not None else n_blocks * logical_len
        if self.truncate_padding and content is not None:
            extent = nonpad_extent(content)
            if extent < logical_len:
                ids = ids[:, :extent]
                mask = mask[:, :extent]
                content = content[:, :extent]
        # Accounted in both paths so the two are directly comparable.
        executed = int(ids.shape[0] * ids.shape[1])
        self._exec_logical += n_blocks * logical_len
        self._exec_positions += executed
        self._exec_nonpad += nonpad
        self._step_positions += executed
        self._step_nonpad += nonpad
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
                prediction_mask(mask, loss_cfg["kd_scope"], content,
                                input_ids=ids, think_ids=self.think_ids),
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
        content = None if self.train_content is None else self.train_content[idxs]
        # Normalizers are known from the masks alone, so microbatch losses can
        # be scaled exactly before backward (sum over micro = true step mean).
        ce_total = int(mask[:, 1:].sum()) if loss_cfg["ce_weight"] > 0 else 0
        kd_total = (
            int(prediction_mask(mask, loss_cfg["kd_scope"], content,
                                input_ids=ids, think_ids=self.think_ids).sum())
            if self.teacher is not None and loss_cfg["kd_weight"] > 0
            else 0
        )
        # A pure function of `step`, which is why the checkpoint needs no
        # separate scheduler state.
        lr = cfg["optim"]["lr"] * lr_factor(
            self.step, sched["total_steps"], sched["warmup_steps"], sched["min_lr_frac"]
        )
        for group in self.opt.param_groups:
            group["lr"] = lr

        self.student.train()
        self.opt.zero_grad(set_to_none=True)
        self._step_positions = self._step_nonpad = 0
        ce_acc = kd_acc = 0.0
        for i in range(0, bps, micro):
            mids = ids[i : i + micro].to(self.device)
            mmask = mask[i : i + micro].to(self.device)
            mcontent = (
                None if content is None else content[i : i + micro].to(self.device)
            )
            ce_sum, _, kd_sum, _ = self._micro_losses(mids, mmask, mcontent)
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
            # Three deliberately distinct quantities. `logical_block_tokens` is
            # the block capacity the run is defined in and never changes;
            # `executed_nonpad_tokens` is the real tokens the models saw;
            # `supervised_tokens` is what CE trained on. Truncation moves only
            # the positions actually forwarded, which is reported separately as
            # `executed_positions` — it equals `executed_nonpad_tokens` when
            # micro_blocks == 1 and exceeds it when a microbatch mixes lengths.
            "logical_block_tokens": bps * self.train_ids.shape[1],
            "executed_positions": self._step_positions,
            "executed_nonpad_tokens": self._step_nonpad,
            "supervised_tokens": ce_total,
            "truncate_padding": self.truncate_padding,
        }

    @torch.no_grad()
    def _eval_blocks(
        self, val_ids, val_mask, max_blocks: int | None, val_content=None
    ) -> dict:
        """CE (assistant targets) and KD metrics over a fixed block order."""
        n = val_ids.shape[0]
        if max_blocks:
            n = min(n, max_blocks)
        micro = self.cfg["batch"]["micro_blocks"]
        was_training = self.student.training
        self.student.eval()
        ce_s = kd_s = 0.0
        ce_n = kd_n = 0
        for i in range(0, n, micro):
            ids = val_ids[i : i + micro].to(self.device)
            mask = val_mask[i : i + micro].to(self.device)
            content = (
                None if val_content is None
                else val_content[i : i + micro].to(self.device)
            )
            ce_sum, cn, kd_sum, kn = self._micro_losses(ids, mask, content)
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

    def evaluate(self, max_blocks: int | None = None) -> dict:
        """Metrics over the primary val set (see _eval_blocks)."""
        if self.val_ids is None:
            raise ValueError("trainer has no validation blocks")
        return self._eval_blocks(
            self.val_ids, self.val_mask, max_blocks, self.val_content
        )

    def _eval_and_log(self, eval_blocks, suffix: str = "") -> dict:
        """Evaluate primary + named extra val sets, log one event each."""
        ev = self.evaluate(eval_blocks)
        self.logger.log("eval_result", step=self.step, val_set="val", **ev)
        print(f"eval step {self.step}{suffix}: {ev}", flush=True)
        for name, (ids, mask, content) in self.extra_vals.items():
            ex = self._eval_blocks(ids, mask, eval_blocks, content)
            self.logger.log("eval_result", step=self.step, val_set=name, **ex)
            print(f"eval step {self.step}{suffix} [{name}]: {ex}", flush=True)
        return ev

    def consumed_blocks(self) -> int:
        """Position in the infinite block stream. Resume needs no loader state."""
        return self.step * self.cfg["batch"]["blocks_per_step"]

    def save_checkpoint(self) -> Path:
        tag = f"step_{self.step:06d}"
        ckpt_dir = self.out_dir / "checkpoints" / tag
        if self.lora_modules:
            # `model/` is ALWAYS the deployable artifact: a plain Hugging Face
            # checkpoint with the delta folded into q/k/v/o and no LoRA keys.
            # Every arm of the experiment is then evaluated through the same
            # inference architecture, and no evaluation path can accidentally
            # score the un-adapted base model.
            self.student.save_pretrained(
                ckpt_dir / "model",
                state_dict=merged_state_dict(self.student, self.lora_modules),
            )
            from safetensors.torch import save_file

            save_file(lora_and_base_tensors(self.lora_modules),
                      str(ckpt_dir / "lora_state.safetensors"))
            (ckpt_dir / "checkpoint_meta.json").write_text(json.dumps({
                "step": self.step,
                "consumed_blocks": self.consumed_blocks(),
                "model_dir_is_merged": True,
                "config_sha256": self.config_sha,
                **lora_report(self.lora_modules, self.lora_cfg),
            }, indent=1))
        else:
            self.student.save_pretrained(ckpt_dir / "model")
        torch.save(
            {
                "step": self.step,
                "consumed_blocks": self.consumed_blocks(),
                "optimizer": self.opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "config_sha256": self.config_sha,
                "trainable_names": self.freeze_report["trainable_names"],
                "lora_config": self.lora_cfg.to_dict() if self.lora_cfg else None,
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

        For a LoRA run ``model/`` holds the *merged* weights, so the frozen base
        attention matrices and the LoRA tensors are read back from
        ``lora_state.safetensors`` and written over them. They are stored, not
        recovered by subtracting the delta: ``(w + d) - d`` is not exactly ``w``.
        """
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "trainer_state.pt", weights_only=True)
        if state["config_sha256"] != self.config_sha:
            raise ValueError(
                "checkpoint was written under a different config "
                f"({state['config_sha256'][:12]} != {self.config_sha[:12]}); "
                "refusing to resume"
            )
        if list(state["trainable_names"]) != self.freeze_report["trainable_names"]:
            raise ValueError("checkpoint freeze set differs from current config")
        saved_lora = state.get("lora_config")
        current_lora = self.lora_cfg.to_dict() if self.lora_cfg else None
        if saved_lora != current_lora:
            raise ValueError("checkpoint LoRA config differs from current config")
        if self.lora_modules:
            from safetensors.torch import load_file

            path = ckpt_dir / "lora_state.safetensors"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} missing; a LoRA run cannot resume from merged weights")
            load_lora_and_base_(self.lora_modules, load_file(str(path)))
        self.opt.load_state_dict(state["optimizer"])
        self.step = int(state["step"])
        expected = state.get("consumed_blocks")
        if expected is not None and expected != self.consumed_blocks():
            raise ValueError(
                f"consumed-block position {self.consumed_blocks()} does not match "
                f"the checkpoint's {expected}; batch config changed")
        torch.set_rng_state(state["torch_rng_state"])
        self.logger.log("resume_loaded", step=self.step, checkpoint=str(ckpt_dir),
                        consumed_blocks=self.consumed_blocks())

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
                extra_val_blocks={
                    name: int(blocks[0].shape[0])
                    for name, blocks in self.extra_vals.items()
                },
                **{
                    k: self.freeze_report[k]
                    for k in ("trainable_params", "full_rank_trainable_params",
                              "lora_trainable_params", "total_params")
                },
                lora_config=self.lora_cfg.to_dict() if self.lora_cfg else None,
            )
            if self.val_ids is not None:
                self._eval_and_log(eval_blocks)

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
                self._eval_and_log(eval_blocks)
            if ck["save_every"] and self.step % ck["save_every"] == 0 and not at_end:
                self.save_checkpoint()

        final_eval = None
        if self.val_ids is not None:
            final_eval = self._eval_and_log(eval_blocks, suffix=" (final)")
        ckpt_dir = self.save_checkpoint()
        self.logger.log(
            "run_end", steps=self.step, seconds=round(time.time() - t_start, 1)
        )
        return {
            "steps": self.step,
            "final_eval": final_eval,
            "checkpoint": str(ckpt_dir),
        }
