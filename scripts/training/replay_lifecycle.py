#!/usr/bin/env python3
"""Branch-B: does live allocation grow across *identical* optimizer steps?

    PYTHONPATH=src python scripts/training/replay_lifecycle.py \
        --config configs/stage3/e8b/e8b_dp_r1600k_sa.json \
        --steps 400 --out artifacts/audit/e8b_lifecycle_replay.json

The full-stream shape audit excluded the workload as the cause: the worst block in the
1,761-step stream is at **step 133**, inside the 200-step gate's window, and the region
where DC-sa died is *less* demanding than what the gate had already survived. So the
residual +0.14 GiB at step 310 and the ~step-900 OOM are unexplained by shapes.

This replays **one pinned worst-case payload** — the same blocks, the same
accumulation and microbatch structure, the same optimizer path — and asks whether
*instantaneous* live allocation at identical lifecycle boundaries increases. Previous
evidence used `max_memory_allocated()`, a process-lifetime running maximum that is
non-decreasing by construction and therefore cannot distinguish growth from having met
a worse block. Here every step resets peak stats and records
`memory_allocated()`/`memory_reserved()` at eight boundaries.

Two phases, because DC-sa failed *after* the run had crossed the checkpoint/eval region:

  A. `--steps` identical steps, clean, establishing whether the boundary baseline moves;
  B. an eval and a checkpoint save, then more identical steps, comparing the boundary
     baseline before and after those events.

Architecture-generic on purpose: nothing here hard-codes the 4B teacher, the 596M
target, layer counts, hidden/FFN/head sizes, or an operator order. It reads the config
it is given and pins whichever block index it is told to. A later 30B -> 4.xB study
should be able to run it unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402


def gib(n: float) -> float:
    return round(n / 1024 ** 3, 4)


def snap() -> tuple[float, float]:
    torch.cuda.synchronize()
    return gib(torch.cuda.memory_allocated()), gib(torch.cuda.memory_reserved())


def allocator_stats() -> dict:
    """Retry/split counters, where the build exposes them."""
    try:
        s = torch.cuda.memory_stats()
    except Exception:
        return {}
    keys = ("num_alloc_retries", "num_ooms",
            "allocated_bytes.all.peak", "reserved_bytes.all.peak",
            "segment.all.current", "inactive_split_bytes.all.current",
            "active_bytes.all.current")
    return {k: s[k] for k in keys if k in s}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=400,
                    help="identical steps in phase A; must cross the growth horizon "
                         "observed in the real run (+0.14 GiB by step 310)")
    ap.add_argument("--post-event-steps", type=int, default=60)
    ap.add_argument("--pin-step", type=int, default=133,
                    help="stream step whose blocks are replayed; 133 is the "
                         "worst joint transient in this pack, found by "
                         "audit_stream_shapes.py — not a magic number")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((REPO_ROOT / args.config).read_text()
                     if not Path(args.config).is_absolute()
                     else Path(args.config).read_text())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv), "config": args.config,
        "cuda_available": torch.cuda.is_available(),
        "metric_note": "instantaneous memory_allocated()/memory_reserved() at each "
                       "boundary; peak stats are reset per step, so nothing here is a "
                       "process-lifetime running maximum",
    }
    if not torch.cuda.is_available():
        report["status"] = "CPU-only: nothing measured. Run on the A100."
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(report["status"])
        return 0

    import os

    from transformers import AutoModelForCausalLM

    from aadistill.data.ladder import ladder_blocks
    from aadistill.models.teacher import DTYPES, load_teacher
    from aadistill.training import train as T

    report["allocator_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)")
    report["device"] = torch.cuda.get_device_name(0)
    report["capacity_gib"] = gib(torch.cuda.get_device_properties(0).total_memory)

    pack = REPO_ROOT / cfg["data_dir"]
    if not (pack / "ladder.json").is_file():
        pack = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
    train_b, val_b, _ = ladder_blocks(pack, cfg["rung"], n_val=cfg["val_blocks"])
    t = cfg["teacher"]
    teacher, _, _ = load_teacher(t["model_id"], t["revision"], dtype=t["dtype"],
                                 device="cuda")
    student = AutoModelForCausalLM.from_pretrained(
        REPO_ROOT / cfg["student_path"], dtype=DTYPES[cfg["dtype"]])
    trainer = T.Trainer(cfg, student, train_b, val_b, teacher=teacher, device="cuda")

    # Pin the payload: every step draws the SAME blocks, so any change in boundary
    # allocation cannot be a workload change. The real step_once, real accumulation,
    # real optimizer path are otherwise untouched.
    bps = cfg["batch"]["blocks_per_step"]
    n_blocks = int(trainer.train_ids.shape[0])
    pinned = T.stream_block_indices(n_blocks, cfg["seed"], args.pin_step * bps, bps)
    report["pinned_blocks"] = [int(i) for i in pinned]
    T.stream_block_indices = lambda *a, **k: list(pinned)

    # Boundary instrumentation by WRAPPING the real functions, never reimplementing
    # them. An earlier draft replaced `_micro_losses` wholesale and silently dropped
    # its `truncate_padding` trim, which changes the executed extent and therefore the
    # very thing being measured. Wrapping keeps the measured path identical to the
    # trained path.
    #
    # Order inside the real `_micro_losses` is: student forward -> masked_ce ->
    # teacher forward -> kd_forward_kl. So `before_ce` IS `after_student_forward`, and
    # `before_kd` IS `after_teacher_forward`; both names are recorded.
    boundaries: list[dict] = []
    cur: dict = {}
    orig_ce, orig_kd = T.masked_ce, T.kd_forward_kl

    def ce_probe(*a, **k):
        x, y = snap()
        cur["after_student_forward"] = cur["before_ce"] = [x, y]
        out = orig_ce(*a, **k)
        cur["after_ce"] = list(snap())
        return out

    def kd_probe(*a, **k):
        x, y = snap()
        cur["after_teacher_forward"] = cur["before_kd"] = [x, y]
        out = orig_kd(*a, **k)
        cur["after_kd"] = list(snap())
        return out

    T.masked_ce, T.kd_forward_kl = ce_probe, kd_probe

    def run_steps(n: int, phase: str) -> None:
        for _ in range(n):
            torch.cuda.reset_peak_memory_stats()
            cur.clear()
            a0, r0 = snap()
            m = trainer.step_once()
            a1, r1 = snap()
            trainer.opt.zero_grad(set_to_none=True)
            a2, r2 = snap()
            boundaries.append({
                "phase": phase, "step": int(trainer.step),
                "step_start": [a0, r0],
                **{k: v for k, v in cur.items()},
                "after_backward_and_optimizer": [a1, r1],
                "after_zero_grad": [a2, r2],
                "step_peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
                "step_peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                "loss": round(float(m.get("loss", 0.0)), 6)
                if isinstance(m, dict) else None,
            })

    run_steps(args.steps, "A_identical")
    report["phase_A_steps"] = args.steps

    # Phase B: the events the gate never ran. DC-sa died after crossing this region.
    ev = {}
    a, r = snap()
    ev["before_events"] = [a, r]
    try:
        trainer._eval_blocks(trainer.val_ids, trainer.val_mask,
                             cfg["intervals"]["eval_blocks"],
                             getattr(trainer, "val_content", None))
        a, r = snap()
        ev["after_eval"] = [a, r]
    except Exception as exc:
        ev["eval_error"] = f"{type(exc).__name__}: {exc}"
    try:
        trainer.save_checkpoint()
        a, r = snap()
        ev["after_checkpoint_save"] = [a, r]
    except Exception as exc:
        ev["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
    report["events"] = ev
    run_steps(args.post_event_steps, "B_after_events")

    # --- verdict ---------------------------------------------------------
    def series(phase: str, key: str) -> list[float]:
        return [b[key][0] for b in boundaries
                if b["phase"] == phase and key in b]

    verdict = {}
    for key in ("step_start", "after_zero_grad"):
        a = series("A_identical", key)
        if len(a) >= 20:
            head, tail = a[:10], a[-10:]
            verdict[key] = {
                "first_10_mean": round(sum(head) / len(head), 4),
                "last_10_mean": round(sum(tail) / len(tail), 4),
                "drift_gib": round(sum(tail) / len(tail) - sum(head) / len(head), 4),
                "min": min(a), "max": max(a), "distinct": len(set(a))}
    b_start = series("B_after_events", "step_start")
    a_start = series("A_identical", "step_start")
    if a_start and b_start:
        verdict["baseline_returns_after_events"] = {
            "phase_A_last_10_mean": round(sum(a_start[-10:]) / len(a_start[-10:]), 4),
            "phase_B_last_10_mean": round(sum(b_start[-10:]) / len(b_start[-10:]), 4),
            "delta_gib": round(sum(b_start[-10:]) / len(b_start[-10:])
                               - sum(a_start[-10:]) / len(a_start[-10:]), 4)}
    peaks = [b["step_peak_allocated_gib"] for b in boundaries]
    verdict["per_step_peak"] = {"min": min(peaks), "max": max(peaks),
                                "first": peaks[0], "last": peaks[-1]}
    verdict["allocator_stats_end"] = allocator_stats()
    report["verdict"] = verdict
    report["boundaries"] = boundaries

    out = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
