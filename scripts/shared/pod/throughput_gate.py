#!/usr/bin/env python
"""Phase 1 throughput gate — run after the FIRST D0 endpoint evaluation.

The Experiment 1 evaluation ran at **254.8 output tokens/s** aggregate on the two
0.86M PCA arms, ~10x below what a 0.6B model on an L40S should deliver. Three
execution-path defects were corrected. This gate decides, from the first D0
endpoint's own telemetry, whether the correction actually worked — **before the
second D0 endpoint runs and before either D1 training run starts**.

Preregistered stop conditions (maintainer, 2026-08-03). The gate FAILS if any
of these holds:

1. **throughput unchanged** — aggregate output tokens/s ≤ 306, i.e. still within
   20% of the 254.8 baseline;
2. **step time unchanged** — a comparable long-output wave still shows a median
   scheduler-step time ≥ 100 ms at an effective batch near 37. "Comparable" is
   defined here, not after the fact: output p50 ≥ 300 tokens and mean effective
   batch in [20, 60], which is the regime the baseline waves ran in;
3. **GPU starvation or another execution defect** — median in-wave GPU
   utilization below `--min-gpu-util`, or telemetry missing entirely when
   `--require-telemetry` is set.

On failure the caller must preserve partial output and telemetry, tear the pod
down safely, report actual cost, and stop. On success phase 1 continues without
further approval, under the unchanged $18.78 hard spending stop.

Exit status: 0 pass, 1 fail, 2 could not evaluate the gate.

    scripts/pod/throughput_gate.py --eval-dir artifacts/eval/e2/d0_sa \\
        --out artifacts/eval/e2/d0_sa_throughput_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASELINE_TOK_S = 254.8          # measured, E1 0.86M PCA arms, 209,850 tok / 823.5 s
THRESHOLD_TOK_S = 306.0         # baseline + 20%
STEP_MS_LIMIT = 100.0
COMPARABLE_P50_TOKENS = 300     # the baseline waves ran at output p50 306-768
COMPARABLE_BATCH = (20.0, 60.0)  # baseline mean effective batch ~37


def load_sets(eval_dir: Path) -> list[dict]:
    out = []
    for path in sorted(eval_dir.glob("*.json")):
        if path.name.endswith("_gate.json"):
            continue
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "throughput" in d:
            out.append({"file": path.name, **d})
    return out


def evaluate(sets: list[dict], min_gpu_util: float,
             require_telemetry: bool) -> dict:
    failures, notes = [], []

    tok = sum(s["throughput"]["output_tokens"] for s in sets)
    sec = sum(s["throughput"]["generation_wall_seconds"] for s in sets)
    agg = round(tok / sec, 1) if sec else None
    if agg is None:
        return {"verdict": "error", "reason": "no generation wall time recorded"}

    # --- condition 1: aggregate throughput ---------------------------------
    if agg <= THRESHOLD_TOK_S:
        failures.append(
            f"aggregate {agg} output tok/s is within 20% of the {BASELINE_TOK_S} "
            f"baseline (limit {THRESHOLD_TOK_S})")
    speedup = round(agg / BASELINE_TOK_S, 2)

    # --- condition 2: step time on comparable long-output waves ------------
    comparable = []
    for s in sets:
        t = s["throughput"]
        batch = t.get("effective_batch_size_mean")
        if (t.get("output_tokens_p50", 0) >= COMPARABLE_P50_TOKENS
                and batch is not None
                and COMPARABLE_BATCH[0] <= batch <= COMPARABLE_BATCH[1]):
            comparable.append(s)
    for s in comparable:
        ms = s["throughput"].get("step_ms_p50")
        if ms is None:
            notes.append(f"{s['file']}: comparable wave has no step-time median")
            continue
        if ms >= STEP_MS_LIMIT:
            failures.append(
                f"{s['file']}: median step {ms} ms at effective batch "
                f"{s['throughput']['effective_batch_size_mean']} — still at or "
                f"above the {STEP_MS_LIMIT} ms limit")
    if not comparable:
        notes.append(
            "no wave matched the comparable regime (output p50 >= "
            f"{COMPARABLE_P50_TOKENS}, effective batch in {COMPARABLE_BATCH}); "
            "condition 2 could not be evaluated on this checkpoint")

    # --- condition 3: GPU starvation / missing telemetry -------------------
    utils = [s["gpu"]["utilization_p50"] for s in sets
             if isinstance(s.get("gpu"), dict)
             and s["gpu"].get("utilization_p50") is not None]
    if utils:
        worst = min(utils)
        if worst < min_gpu_util:
            failures.append(
                f"median in-wave GPU utilization {worst}% is below "
                f"{min_gpu_util}% — the GPU is starved, so the bottleneck is "
                "still outside the model computation")
    else:
        msg = "no in-wave GPU utilization telemetry (run with --diagnostics)"
        (failures if require_telemetry else notes).append(msg)

    return {
        "verdict": "fail" if failures else "pass",
        "aggregate_output_tokens_per_second": agg,
        "baseline_output_tokens_per_second": BASELINE_TOK_S,
        "speedup_vs_baseline": speedup,
        "threshold_output_tokens_per_second": THRESHOLD_TOK_S,
        "total_output_tokens": tok,
        "total_generation_wall_seconds": round(sec, 1),
        "comparable_waves": [s["file"] for s in comparable],
        "gpu_utilization_p50_min": min(utils) if utils else None,
        "failures": failures,
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path,
                    help="the --out-dir of the first D0 endpoint evaluation")
    ap.add_argument("--min-gpu-util", type=float, default=40.0)
    ap.add_argument("--require-telemetry", action="store_true", default=True)
    ap.add_argument("--allow-missing-telemetry", dest="require_telemetry",
                    action="store_false")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sets = load_sets(args.eval_dir)
    if not sets:
        print(f"GATE ERROR: no instrumented result files in {args.eval_dir}",
              file=sys.stderr)
        return 2

    result = evaluate(sets, args.min_gpu_util, args.require_telemetry)
    result.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "eval_dir": str(args.eval_dir),
        "sets": {s["file"]: {k: s["throughput"].get(k) for k in (
            "input_tokens", "output_tokens", "output_tokens_p50",
            "output_tokens_p95", "output_tokens_max", "generation_wall_seconds",
            "output_tokens_per_second", "prompts_per_second", "scheduler_steps",
            "step_ms_p50", "step_ms_p95", "effective_batch_size_mean",
            "concurrency_max")} for s in sets},
        "engine": sets[0].get("engine"),
        "stop_reason_rates": {s["file"]: {
            "natural_termination": s.get("natural_termination_rate"),
            "degeneration": s.get("degeneration_rate"),
            "context_limit": s.get("context_limit_rate"),
            "right_censored": s.get("right_censored_rate"),
        } for s in sets},
    })
    out = args.out or (args.eval_dir / "throughput_gate.json")
    out.write_text(json.dumps(result, indent=1))

    print(f"aggregate {result['aggregate_output_tokens_per_second']} out-tok/s "
          f"({result['speedup_vs_baseline']}x the {BASELINE_TOK_S} baseline), "
          f"limit {THRESHOLD_TOK_S}")
    for name, t in result["sets"].items():
        print(f"  {name:26s} {t['output_tokens']:>8,} tok  "
              f"{t['generation_wall_seconds']:>7.1f} s  "
              f"{t['output_tokens_per_second']:>7.1f} tok/s  "
              f"p50 {t['output_tokens_p50']:>5}  "
              f"step {t['step_ms_p50']} ms  batch {t['effective_batch_size_mean']}")
    for n in result["notes"]:
        print(f"  note: {n}")
    for f in result["failures"]:
        print(f"  FAIL: {f}")
    print(f"\nVERDICT: {result['verdict'].upper()}  -> {out}")
    if result["verdict"] == "fail":
        print("\nPreserve partial output and telemetry, tear the pod down, "
              "report actual cost, and stop. Do NOT start the second D0 "
              "endpoint or either D1 training run.")
        return 1
    print("\nGate passed. Phase 1 continues without further approval, under the "
          "unchanged $18.78 hard spending stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
