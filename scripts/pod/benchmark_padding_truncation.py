#!/usr/bin/env python
"""Benchmark the full-width and padding-truncated training paths on one GPU.

    PYTHONPATH=src python scripts/pod/benchmark_padding_truncation.py \
        --pack artifacts/stage3/ladder_uniform_probe \
        --student artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
        --teacher Qwen/Qwen3-4B-Thinking-2507 \
        --out artifacts/audit/padding_truncation_benchmark.json

Measures rather than extrapolates. Padding percentage bounds the *linear* FLOP
saving but says nothing about attention (quadratic), kernel launch overhead at
short sequence lengths, or memory. Each padding regime is timed directly.

Method
------
* both paths run the identical config, batch, seed and optimizer on the same
  process and the same GPU, differing only in `batch.truncate_padding`;
* `--warmup` steps are discarded before timing, so autotuning, allocator growth
  and any JIT are not counted;
* `torch.cuda.synchronize()` at every timing boundary — without it the timer
  measures enqueue rate, not compute;
* peak memory is read from `torch.cuda.max_memory_allocated` with the counter
  reset per path;
* the student forward/backward and the teacher forward are timed separately with
  their own synchronizations, so the split can be reported rather than assumed.

Regimes are drawn from a real pack: the most padded blocks (the tool-calling
shape), the median, the densest, and a random draw representing the mixture a
real run actually consumes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.training.train import Trainer, nonpad_extent  # noqa: E402


@contextlib.contextmanager
def cuda_timer(store: list):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    store.append(time.perf_counter() - t0)


def instrument(trainer: Trainer):
    """Time the student fwd+bwd and the teacher forward separately."""
    student_t: list = []
    teacher_t: list = []
    real_student = trainer.student
    real_teacher = trainer.teacher

    class Timed(torch.nn.Module):
        def __init__(self, inner, store):
            super().__init__()
            self.inner, self.store = inner, store

        def forward(self, *a, **k):
            with cuda_timer(self.store):
                return self.inner(*a, **k)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    trainer.student = Timed(real_student, student_t)
    if real_teacher is not None:
        trainer.teacher = Timed(real_teacher, teacher_t)
    return student_t, teacher_t


def pick_regimes(arrays, n: int, seed: int = 0):
    fill = arrays["content_mask"].sum(axis=1)
    order = np.argsort(fill)
    rng = np.random.default_rng(seed)
    mid = len(order) // 2
    return {
        "heavy_pad": order[:n].tolist(),
        "median_pad": order[mid:mid + n].tolist(),
        "dense": order[-n:].tolist(),
        "random_mixture": rng.choice(len(order), size=n, replace=False).tolist(),
    }


def run(cfg, blocks, truncate, student_path, teacher_spec, device, steps, warmup):
    import copy
    from transformers import AutoModelForCausalLM
    cfg = copy.deepcopy(cfg)
    cfg["batch"]["truncate_padding"] = truncate

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    student = AutoModelForCausalLM.from_pretrained(
        student_path, dtype=torch.float32).to(device)
    student.gradient_checkpointing_enable()
    teacher = None
    if teacher_spec:
        model_id, _, rev = teacher_spec.partition("@")
        teacher = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev or None, dtype=torch.bfloat16).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    tr = Trainer(cfg, student, blocks, blocks, teacher=teacher, device=device)
    s_t, t_t = instrument(tr)

    for _ in range(warmup):
        tr.step_once()
    s_t.clear(); t_t.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    step_times, positions, nonpad = [], [], []
    for _ in range(steps):
        with cuda_timer(step_times):
            m = tr.step_once()
        positions.append(m["executed_positions"])
        nonpad.append(m["executed_nonpad_tokens"])
    peak = torch.cuda.max_memory_allocated() / 2**30

    del tr, student, teacher
    torch.cuda.empty_cache()
    return {
        "seconds_per_step_median": round(statistics.median(step_times), 5),
        "seconds_per_step_mean": round(statistics.fmean(step_times), 5),
        "seconds_per_step_min": round(min(step_times), 5),
        "peak_memory_gib": round(peak, 3),
        "executed_positions_per_step": int(statistics.fmean(positions)),
        "executed_nonpad_tokens_per_step": int(statistics.fmean(nonpad)),
        "executed_tokens_per_second": round(
            statistics.fmean(positions) / statistics.median(step_times), 1),
        "nonpad_tokens_per_second": round(
            statistics.fmean(nonpad) / statistics.median(step_times), 1),
        "student_fwd_seconds_per_step": (
            round(sum(s_t) / len(step_times), 5) if s_t else None),
        "teacher_fwd_seconds_per_step": (
            round(sum(t_t) / len(step_times), 5) if t_t else None),
        "n_steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default="")
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/stage3/e1/e1_r0860k_sa_pca.json")
    ap.add_argument("--blocks-per-regime", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires a GPU")
    device = "cuda"
    arrays = np.load(args.pack / "blocks.npz")
    base = json.loads(args.config.read_text())
    base["device"] = device

    results = {}
    for name, idxs in pick_regimes(arrays, args.blocks_per_regime).items():
        ids = torch.from_numpy(arrays["input_ids"][idxs].astype(np.int64))
        mask = torch.from_numpy(arrays["ce_mask"][idxs]).bool()
        content = torch.from_numpy(arrays["content_mask"][idxs]).bool()
        blocks = (ids, mask, content)
        fill = float(content.float().mean())
        entry = {"fill_fraction": round(fill, 4), "n_blocks": len(idxs)}
        for label, flag in (("full_width", False), ("truncated", True)):
            try:
                entry[label] = run(base, blocks, flag, args.student, args.teacher,
                                   device, args.steps, args.warmup)
            except torch.cuda.OutOfMemoryError:
                entry[label] = {"error": "cuda_out_of_memory"}
                torch.cuda.empty_cache()
        f, t = entry.get("full_width", {}), entry.get("truncated", {})
        if "seconds_per_step_median" in f and "seconds_per_step_median" in t:
            entry["speedup"] = round(
                f["seconds_per_step_median"] / t["seconds_per_step_median"], 3)
            entry["peak_memory_ratio"] = round(
                t["peak_memory_gib"] / f["peak_memory_gib"], 3)
        results[name] = entry
        print(f"\n=== {name}  fill={fill:.3f}")
        for label in ("full_width", "truncated"):
            e = entry.get(label, {})
            if "error" in e:
                print(f"  {label:11s} {e['error']}")
                continue
            print(f"  {label:11s} {e['seconds_per_step_median']:.4f} s/step   "
                  f"peak {e['peak_memory_gib']:.2f} GiB   "
                  f"exec {e['executed_positions_per_step']} tok   "
                  f"{e['executed_tokens_per_second']:.0f} tok/s   "
                  f"student_fwd {e['student_fwd_seconds_per_step']}  "
                  f"teacher_fwd {e['teacher_fwd_seconds_per_step']}")
        if "speedup" in entry:
            print(f"  speedup {entry['speedup']}x   "
                  f"peak-memory ratio {entry['peak_memory_ratio']}")

    props = torch.cuda.get_device_properties(0)
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "gpu": props.name,
        "gpu_memory_gib": round(props.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": str(args.config),
        "student": args.student,
        "teacher": args.teacher,
        "steps_timed": args.steps,
        "warmup_steps": args.warmup,
        "note": ("Timed with torch.cuda.synchronize() at every boundary and "
                 "after discarding warmup steps. Speedup is measured per padding "
                 "regime, not extrapolated from padding percentage."),
        "regimes": results,
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
