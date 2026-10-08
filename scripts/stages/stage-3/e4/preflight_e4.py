#!/usr/bin/env python
"""Experiment 4 preflight: prove the arms on the REAL student before renting a GPU.

    PYTHONPATH=src python scripts/training/preflight_e4.py \
        --out artifacts/audit/e4_preflight.json

Every check here is one that would otherwise surface only after money had been
spent, and every one has a specific way of failing silently:

* the Stage 1 fork point still hashes to the pinned value;
* the 1.60M rung is a **strict superset** of the 0.86M rung on actual token ids,
  so this is a scale experiment and not a different dataset;
* both rungs validate on the **same** held-out tail, or their CE is not
  comparable;
* the trainable policy on the real 596M geometry is full-rank attention + FFN +
  all norms, with embeddings and `lm_head` frozen — i.e. none of Experiment 3's
  freeze policy or LoRA survived into this recipe;
* the block stream is deterministic and reproduces from the consumed-block
  counter alone, so resume is exact;
* a real optimizer step on real weights produces finite loss and gradients.

No checkpoint is written and no run directory is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aadistill.data.ladder import ladder_blocks  # noqa: E402
from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.training.train import (  # noqa: E402
    Trainer, select_trainable, stream_block_indices, validate_train_config,
)

INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
INIT_SHA = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"
PACK = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
CONFIGS = {
    "P2-0.86M-sa": "configs/stage3/p2/p2_ceheavy_sa.json",
    "P2-0.86M-sb": "configs/stage3/p2/p2_ceheavy_sb.json",
    "E4-P2-1.60M-sa": "configs/stage3/e4/e4_p2_r1600k_sa.json",
    "E4-P2-1.60M-sb": "configs/stage3/e4/e4_p2_r1600k_sb.json",
}
ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


def group_of(name: str) -> str:
    if ".mlp." in name:
        return "ffn"
    if any(f".self_attn.{p}." in name for p in ATTN_PROJ):
        return "attn_proj"
    if ".self_attn.q_norm" in name or ".self_attn.k_norm" in name:
        return "attn_norm"
    if "layernorm" in name:
        return "decoder_norm"
    if name.startswith("model.norm."):
        return "final_norm"
    return "embedding_or_head"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts/audit/e4_preflight.json")
    ap.add_argument("--skip-step", action="store_true",
                    help="skip the real-weights optimizer-step check")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM

    failures: list[str] = []

    def require(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    init_sha = sha256_file(INIT / "model.safetensors")
    require(init_sha == INIT_SHA, f"Stage 1 init hash mismatch: {init_sha}")

    # ---- rung nesting, on real token ids -------------------------------
    small, _, s_stats = ladder_blocks(PACK, 860000, n_val=16)
    large, _, l_stats = ladder_blocks(PACK, 1600000, n_val=16)
    n860 = s_stats["train_blocks"]
    nested_ids = bool(np.array_equal(small[0].numpy(), large[0][:n860].numpy()))
    nested_mask = bool(np.array_equal(small[1].numpy(), large[1][:n860].numpy()))
    require(nested_ids and nested_mask,
            "1.60M rung is NOT a strict superset of 0.86M")
    require(s_stats["val_block_indices"] == l_stats["val_block_indices"],
            "rungs validate on different held-out blocks; CE not comparable")
    require(l_stats["train_supervised_tokens"] == 1_600_353,
            f"1.60M supervised tokens {l_stats['train_supervised_tokens']}")
    require(s_stats["train_supervised_tokens"] == 864_750,
            f"0.86M supervised tokens {s_stats['train_supervised_tokens']}")

    rung = {
        "r0860k": {"blocks": n860,
                   "supervised": s_stats["train_supervised_tokens"],
                   "mix": s_stats["train_token_mix"]},
        "r1600k": {"blocks": l_stats["train_blocks"],
                   "supervised": l_stats["train_supervised_tokens"],
                   "mix": l_stats["train_token_mix"]},
        "strict_superset_input_ids": nested_ids,
        "strict_superset_ce_mask": nested_mask,
        "shared_val_blocks": s_stats["val_block_indices"],
        "val_disjoint_from_all_rungs": bool(
            s_stats["val_disjoint_from_all_rungs"]
            and l_stats["val_disjoint_from_all_rungs"]),
        "pack_sha256": {
            "blocks_npz": sha256_file(PACK / "blocks.npz"),
            "ladder_json": sha256_file(PACK / "ladder.json"),
            "audit_jsonl": sha256_file(PACK / "audit.jsonl"),
        },
    }

    # ---- per-arm policy on the real geometry ---------------------------
    arms = {}
    for alias, path in CONFIGS.items():
        cfg = json.loads((REPO_ROOT / path).read_text())
        validate_train_config(cfg)
        require("lora" not in cfg, f"{alias}: LoRA config present")
        for field in ("lora_lr", "lora_weight_decay", "no_decay_patterns"):
            require(field not in cfg["optim"], f"{alias}: optim.{field} present")
        require(cfg["loss"] == {"ce_weight": 1.0, "kd_weight": 0.25,
                                "kd_temperature": 1.0, "kd_scope": "all"},
                f"{alias}: objective is not CE 1.0 / KD 0.25 scope all")

        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(INIT))
        report = select_trainable(model, cfg["trainable_patterns"])
        by_group: dict[str, dict[str, int]] = {}
        for name, param in model.named_parameters():
            g = by_group.setdefault(group_of(name), {"trainable": 0, "frozen": 0})
            g["trainable" if param.requires_grad else "frozen"] += param.numel()
        for g in ("attn_proj", "ffn", "attn_norm", "decoder_norm", "final_norm"):
            require(by_group[g]["frozen"] == 0,
                    f"{alias}: {g} must be fully trainable")
        require(by_group["embedding_or_head"]["trainable"] == 0,
                f"{alias}: embeddings/lm_head must be frozen")
        require(report["lora_trainable_params"] == 0,
                f"{alias}: LoRA parameters present")
        require(report["trainable_params"] == 440_467_456,
                f"{alias}: trainable {report['trainable_params']:,} != 440,467,456")

        n_blocks = rung["r1600k" if cfg["rung"] == 1600000 else "r0860k"]["blocks"]
        seed, bps = cfg["seed"], cfg["batch"]["blocks_per_step"]
        stream = stream_block_indices(n_blocks, seed, 0, n_blocks)
        require(sorted(stream) == list(range(n_blocks)),
                f"{alias}: epoch 0 is not a permutation")
        mid = cfg["schedule"]["total_steps"] // 2 * bps
        require(stream_block_indices(n_blocks, seed, mid, 8)
                == stream_block_indices(n_blocks, seed, 0, mid + 8)[mid:mid + 8],
                f"{alias}: block stream is not resumable from the counter")

        arms[alias] = {
            "config": path,
            "config_sha256": sha256_json(cfg),
            "seed": seed,
            "rung": cfg["rung"],
            "total_steps": cfg["schedule"]["total_steps"],
            "warmup_steps": cfg["schedule"]["warmup_steps"],
            "blocks": n_blocks,
            "passes": round(cfg["schedule"]["total_steps"] * bps / n_blocks, 3),
            "trainable_params": report["trainable_params"],
            "total_params": report["total_params"],
            "by_group": {k: v for k, v in sorted(by_group.items())},
            "n_trainable_tensors": len(report["trainable_names"]),
        }
        del model

    # ---- one real optimizer step on real weights -----------------------
    step_check = {"skipped": bool(args.skip_step)}
    if not args.skip_step:
        cfg = json.loads(
            (REPO_ROOT / CONFIGS["E4-P2-1.60M-sa"]).read_text())
        # CE-only, on purpose and on the record: this check is about real
        # weights, real packed data and the freeze policy surviving a real
        # optimizer step, not about the KD term — which needs the 4B teacher
        # resident and is exercised by the test suite and by the run itself.
        toy = dict(cfg)
        toy["schedule"] = {**cfg["schedule"], "total_steps": 1}
        toy["block_len"] = 256
        toy["loss"] = {**cfg["loss"], "kd_weight": 0.0}
        student = AutoModelForCausalLM.from_pretrained(INIT, dtype=torch.float32)
        ids = large[0][:2, :256].clone()
        mask = large[1][:2, :256].clone().bool()
        content = torch.ones_like(mask)
        if not mask.any():
            mask[:, 10:20] = True
        trainer = Trainer(toy, student, (ids, mask, content),
                          (ids, mask, content), device="cpu")
        before = {n: p.detach().clone() for n, p in trainer.student.named_parameters()}
        m = trainer.step_once()
        moved = sum(1 for n, p in trainer.student.named_parameters()
                    if not torch.equal(p, before[n]))
        frozen_moved = [n for n, p in trainer.student.named_parameters()
                        if not p.requires_grad and not torch.equal(p, before[n])]
        step_check = {
            "skipped": False,
            "objective": "CE only (kd_weight forced to 0 for this check)",
            "why": ("exercises real weights, real packed 1.60M blocks and the "
                    "freeze policy under a real optimizer step; the KD term "
                    "needs the 4B teacher resident and is covered by the test "
                    "suite and by the run itself"),
            "loss": m["loss"], "grad_norm": m["grad_norm"],
            "finite_loss": bool(np.isfinite(m["loss"])),
            "finite_grad_norm": bool(np.isfinite(m["grad_norm"])),
            "tensors_moved": moved,
            "frozen_tensors_moved": frozen_moved,
            "supervised_tokens": m["supervised_tokens"],
        }
        require(step_check["finite_loss"] and step_check["finite_grad_norm"],
                "non-finite loss or gradient on the real weights")
        require(not frozen_moved, f"frozen tensors moved: {frozen_moved[:3]}")
        require(moved > 0, "no parameter moved during a real optimizer step")
        del trainer, student

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": "Experiment 4 preflight — P2-CE-heavy at the nested 1.60M rung",
        "device": "cpu",
        "stage1_init": {"path": str(INIT.relative_to(REPO_ROOT)),
                        "model_safetensors_sha256": init_sha},
        "rung": rung,
        "arms": arms,
        "real_weights_step": step_check,
        "failures": failures,
        "passed": not failures,
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"rung nesting: 0.86M ⊂ 1.60M on ids={nested_ids} mask={nested_mask}; "
          f"{rung['r0860k']['blocks']} → {rung['r1600k']['blocks']} blocks, "
          f"{rung['r0860k']['supervised']:,} → {rung['r1600k']['supervised']:,} tokens")
    for alias, a in arms.items():
        print(f"  {alias:16s} rung {a['rung']:>8} steps {a['total_steps']:>5} "
              f"warmup {a['warmup_steps']:>3} passes {a['passes']} "
              f"trainable {a['trainable_params']:,}")
    if not step_check["skipped"]:
        print(f"  real-weights step: loss {step_check['loss']} "
              f"grad_norm {step_check['grad_norm']} "
              f"moved {step_check['tensors_moved']} tensors, "
              f"frozen moved {len(step_check['frozen_tensors_moved'])}")
    print(f"wrote {args.out}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("PASS: preflight clean")


if __name__ == "__main__":
    main()
