#!/usr/bin/env python
"""Measure the real wall-clock speedup of `truncate_padding` on E5-C blocks.

    PYTHONPATH=src python scripts/training/benchmark_e5_throughput.py \
        --pack artifacts/stage3/e5_pack_c_sa \
        --student /workspace/ckpt/p2_ceheavy_sa \
        --teacher Qwen/Qwen3-4B-Thinking-2507@768f209d… \
        --out artifacts/audit/e5_throughput.json

Executed-position accounting says 3.18x fewer positions. That is **not** a
wall-clock claim: the median E5-C block holds 675 real tokens, and at that length
a 0.6B student and a 4B teacher underuse the GPU while launch overhead dominates.
The ratio is therefore measured, on the production `Trainer.step_once`, and used
only to update the cost model.

Method, and the choices that make the ratio trustworthy:

* **Identical model state for both arms.** The student is re-loaded from the same
  checkpoint before each condition, so neither measurement inherits the other's
  optimizer state or weights.
* **Identical batches, in the same order.** Both conditions consume the same
  block indices, drawn to span the natural E5-C length distribution rather than
  whichever blocks happen to sort first — timing only short blocks would flatter
  truncation, timing only full ones would hide it.
* **Forward and backward**, through `step_once`, which is what training actually
  costs. Optimizer updates are discarded: `lr = 0` and no checkpoint is written.
* **Warm-up before timing**, so CUDA autotuning and allocator growth are not
  charged to the first condition.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.training.train import Trainer  # noqa: E402


def spread_indices(real_lengths: np.ndarray, n: int) -> list[int]:
    """Block indices spanning the length distribution, deterministically.

    Sorting by real length and sampling evenly across the sorted order keeps the
    short, median and near-full blocks all represented in proportion, which is
    what makes the measured ratio applicable to the whole pack.
    """
    order = np.argsort(real_lengths, kind="stable")
    if n >= len(order):
        return [int(i) for i in order]
    picks = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [int(order[i]) for i in picks]


def build_cfg(block_len: int, truncate: bool, micro: int, bps: int) -> dict:
    return {
        "stage": "stage3_recovery", "run_name": "e5_throughput",
        "student_path": "unused", "data_dir": "unused", "groups": None,
        "teacher": {"model_id": "x", "revision": "y", "dtype": "bfloat16"},
        "block_len": block_len, "dtype": "float32", "autocast_bf16": True,
        "gradient_checkpointing": True, "device": "cuda", "seed": 11,
        "trainable_patterns": [
            r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.",
            r"\.mlp\.(gate_proj|up_proj|down_proj)\.",
            "input_layernorm", "post_attention_layernorm", r"model\.norm\."],
        "loss": {"ce_weight": 1.0, "kd_weight": 0.25,
                 "kd_temperature": 1.0, "kd_scope": "all"},
        # lr 0: the step runs in full, including backward, but changes nothing.
        "optim": {"lr": 0.0, "weight_decay": 0.0, "betas": [0.9, 0.95],
                  "eps": 1e-8, "grad_clip": 1.0},
        "schedule": {"total_steps": 10_000, "warmup_steps": 0, "min_lr_frac": 1.0},
        "batch": {"blocks_per_step": bps, "micro_blocks": micro,
                  "truncate_padding": truncate},
        "checkpoint": {"save_every": 0, "keep_last": 1},
        "intervals": {"log_every": 0, "eval_every": 0, "eval_blocks": 0},
        "out_dir": "/tmp/e5_throughput",
    }


def time_condition(truncate: bool, blocks, student_path: str, teacher, *,
                   steps: int, warmup: int, micro: int, bps: int, device: str) -> dict:
    from transformers import AutoModelForCausalLM

    ids, ce, content = blocks
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.float32)
    cfg = build_cfg(int(ids.shape[1]), truncate, micro, bps)
    trainer = Trainer(cfg, student, (ids, ce, content), None, teacher=teacher,
                      device=device)
    for _ in range(warmup):
        trainer.step_once()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    executed = nonpad = 0
    for _ in range(steps):
        m = trainer.step_once()
        executed += m["executed_positions"]
        nonpad += m["executed_nonpad_tokens"]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del trainer, student
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "truncate_padding": truncate, "steps": steps, "seconds": round(elapsed, 3),
        "sec_per_step": round(elapsed / steps, 4),
        "executed_positions": executed,
        "executed_positions_per_sec": round(executed / elapsed, 1),
        "nonpadding_tokens": nonpad,
        "nonpadding_tokens_per_sec": round(nonpad / elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--micro-blocks", type=int, default=1)
    ap.add_argument("--blocks-per-step", type=int, default=2)
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arrays = np.load(args.pack / "blocks.npz")
    real = arrays["content_mask"].sum(axis=1)
    need = (args.steps + args.warmup) * args.blocks_per_step
    picks = spread_indices(real, need)
    ids = torch.from_numpy(arrays["input_ids"][picks].astype(np.int64))
    ce = torch.from_numpy(arrays["ce_mask"][picks])
    content = torch.from_numpy(arrays["content_mask"][picks])
    sel_real = real[picks]
    print(f"benchmark blocks {len(picks)} | real length min/p50/max "
          f"{sel_real.min()}/{int(np.median(sel_real))}/{sel_real.max()} "
          f"| pack p50 {int(np.median(real))}", flush=True)

    tid, _, rev = args.teacher.partition("@")
    teacher = AutoModelForCausalLM.from_pretrained(
        tid, revision=rev or None, dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    results = {}
    for truncate in (False, True):
        label = "truncated" if truncate else "full_width"
        results[label] = time_condition(
            truncate, (ids, ce, content), args.student, teacher,
            steps=args.steps, warmup=args.warmup, micro=args.micro_blocks,
            bps=args.blocks_per_step, device=device)
        print(f"  {label:11s} {results[label]['sec_per_step']:.4f} s/step "
              f"({results[label]['executed_positions_per_sec']:,.0f} exec pos/s)",
              flush=True)

    full, trunc = results["full_width"], results["truncated"]
    speedup = full["sec_per_step"] / max(1e-9, trunc["sec_per_step"])
    bench_min = (full["seconds"] + trunc["seconds"]) / 60
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": "measure the real wall-clock speedup of truncate_padding on "
                   "E5-C blocks; used ONLY to update the cost model",
        "method": {
            "production_trainer_path": True, "forward_and_backward": True,
            "identical_batches_and_order": True,
            "student_reloaded_per_condition": True,
            "lr": 0.0, "checkpoint_written": False,
            "warmup_steps": args.warmup,
            "blocks_span_length_distribution": True,
        },
        "blocks": {"n": len(picks),
                   "real_len_min": int(sel_real.min()),
                   "real_len_p50": int(np.median(sel_real)),
                   "real_len_max": int(sel_real.max()),
                   "pack_real_len_p50": int(np.median(real))},
        "full_width": full, "truncated": trunc,
        "measured_wall_clock_speedup": round(speedup, 4),
        "position_reduction_for_reference": round(
            full["executed_positions"] / max(1, trunc["executed_positions"]), 4),
        "benchmark_minutes": round(bench_min, 2),
        "benchmark_cost_usd": round(bench_min / 60 * args.rate, 3),
        "caveat": ("the position reduction is an upper bound on the wall-clock "
                   "ratio, not a prediction of it; only the measured speedup "
                   "may be used in the cost model"),
        "hardware": hardware_report(), "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nmeasured wall-clock speedup {speedup:.3f}x "
          f"(positions {payload['position_reduction_for_reference']:.2f}x)")
    print(f"benchmark cost ${payload['benchmark_cost_usd']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
