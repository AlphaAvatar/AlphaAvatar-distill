#!/usr/bin/env python3
"""Record the measured optimization effects, and what they do to each cost cell.

    PYTHONPATH=src:scripts python \
        scripts/autoinit/write_c2_measured_optimization.py [--write]

**An ADJUSTMENT, not an observation.** The cost model's cells are pooled
per-expansion minutes from committed searches. This round did not run a search;
it benchmarked two COMPONENTS on a real L40S against the real teacher. So the
refresh is a transparent adjustment of those cells by measured component
savings, with every input named -- never a new pooled row, and never a ratio
applied to a whole cell.

The arithmetic, in one place:

    new_cell = old_cell
             - (state_eval_saving, capped at the observed state_eval phase)
             - (operator_saving, DEPTH only)

where

    state_eval_saving = suite_positions * (old_ms_per_position - new_ms_per_position)
    operator_saving   = observed_operator_minutes * (1 - 1/depth_speedup)

Deliberately NOT included: the reference-cache recompute waste. It is 36.1% of
DEPTH's forward passes and the largest known saving in the search, and this
round only INSTRUMENTED it -- the measurement showed nothing for
`empty_cache()` to reclaim on a card holding only the teacher, so no behaviour
changed and no saving may be claimed.

AUTHORIZES NOTHING and FUNDS NOTHING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

OUT = "logs/stages/stage-1/phase_c2/plans/phase_c2_measured_optimization.json"
SCHEMA = "aadistill.autoinit.c2_measured_optimization/v1"

#: --- MEASURED, on one L40S, 2026-09-18 -----------------------------------
#: Subrun p3: four real items from the frozen mixture, 2032 prediction
#: positions, against the real pinned 4B teacher in bf16. The old side includes
#: the `.float().cpu()` transfer, because that is what the old path did.
REDUCTION_OLD_MS_PER_POSITION = 1.8699
REDUCTION_NEW_MS_PER_POSITION = 0.0246
REDUCTION_RUN = "c2_full_search_perf_20260918_p3"
REDUCTION_DRIFT = 3.0319411605575326e-05

#: Subrun p1: 71 candidate evaluations over 3 real items, twice. Both sides
#: device-resident -- which p1 was, by the same defect that invalidated its
#: reduction measurement. That defect makes p1's DEPTH number the CORRECT
#: comparison for this candidate, since the DEPTH reduction was already
#: device-resident before this round and candidate 2 changes only how many
#: quantities it computes.
DEPTH_OLD_SECONDS = 10.0
DEPTH_NEW_SECONDS = 9.1
DEPTH_RUN = "c2_full_search_perf_20260918_p1"

#: The suite the state-eval phase actually runs, counted from its own items
#: file. An earlier draft of this arithmetic used the CALIBRATION mixture's
#: 59,763 positions, which is a different artifact -- 24% low.
STATE_EVAL_SUITE = "artifacts/stage1/state_eval_v1/items.jsonl"

PHASES = ("materialize_seconds", "identify_seconds", "canonical_reload_seconds",
          "validation_seconds", "state_evaluation_seconds")
TELEMETRY = (
    "logs/stages/stage-1/phase_b/runs/attempt5/search_telemetry.jsonl",
    "logs/stages/stage-1/phase_c2/runs/attempt4/evidence/telemetry.jsonl",
)


def suite_positions() -> int:
    rows = [json.loads(line) for line
            in (REPO_ROOT / STATE_EVAL_SUITE).read_text().splitlines()
            if line.strip()]
    return sum(int(r["n_prediction_positions"]) for r in rows)


def _phase(row: dict, key: str) -> float:
    value = row.get(key) or {}
    return (float(value.get("seconds", 0.0)) if isinstance(value, dict)
            else float(value or 0))


def worst_expansions() -> dict[str, dict]:
    """The MAX expansion per implementation, with its phase split.

    The cost model prices on the per-cell maximum, so a refresh has to adjust
    the components of that same expansion rather than an average.
    """
    worst: dict[str, dict] = {}
    for rel in TELEMETRY:
        for line in (REPO_ROOT / rel).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "operator_seconds" not in row:
                continue
            parts = {"operator": float(row["operator_seconds"]),
                     "parent_load": float(row["parent_load_seconds"])}
            for key in PHASES:
                parts[key.replace("_seconds", "")] = _phase(row, key)
            total = sum(parts.values())
            impl = row["impl_id"]
            if total > worst.get(impl, {}).get("total_seconds", 0.0):
                worst[impl] = {"total_seconds": total, "parts": parts,
                               "source": rel}
    return worst


def refreshed() -> dict:
    positions = suite_positions()
    per_position_saving_ms = (REDUCTION_OLD_MS_PER_POSITION
                              - REDUCTION_NEW_MS_PER_POSITION)
    state_eval_saving_s = positions * per_position_saving_ms / 1000.0
    depth_speedup = DEPTH_OLD_SECONDS / DEPTH_NEW_SECONDS

    cells = {}
    for impl, row in sorted(worst_expansions().items()):
        parts = row["parts"]
        observed_state_eval = parts["state_evaluation"]
        #: CAPPED at the observed phase. A saving larger than the phase it
        #: applies to would be claiming time the expansion never spent.
        applied_state_eval = min(state_eval_saving_s, observed_state_eval)
        applied_operator = 0.0
        if impl.startswith("depth.causal_kl"):
            applied_operator = parts["operator"] * (1 - 1 / depth_speedup)
        new_total = row["total_seconds"] - applied_state_eval - applied_operator
        cells[impl] = {
            "observed_total_minutes": round(row["total_seconds"] / 60, 4),
            "observed_state_evaluation_minutes": round(observed_state_eval / 60, 4),
            "observed_operator_minutes": round(parts["operator"] / 60, 4),
            "state_eval_saving_applied_minutes": round(applied_state_eval / 60, 4),
            "state_eval_saving_was_capped": (
                state_eval_saving_s > observed_state_eval),
            "operator_saving_applied_minutes": round(applied_operator / 60, 4),
            "refreshed_total_minutes": round(new_total / 60, 4),
            "reduction_factor": round(row["total_seconds"] / max(new_total, 1e-9), 3),
            "source": row["source"],
        }

    return {
        "schema": SCHEMA,
        "_contract": (
            "The measured effect of the 2026-09-18 performance round on each "
            "cost cell. An ADJUSTMENT of pooled per-expansion minutes by "
            "measured component savings, not a new pooled observation. Every "
            "input is named and the arithmetic is one formula. AUTHORIZES "
            "NOTHING and FUNDS NOTHING."),
        "measured": {
            "reduction": {
                "run": REDUCTION_RUN,
                "old_ms_per_position": REDUCTION_OLD_MS_PER_POSITION,
                "new_ms_per_position": REDUCTION_NEW_MS_PER_POSITION,
                "speedup": round(REDUCTION_OLD_MS_PER_POSITION
                                 / REDUCTION_NEW_MS_PER_POSITION, 2),
                "worst_relative_drift": REDUCTION_DRIFT,
                "_what_the_old_side_was": (
                    "the `.float().cpu()` transfer of both [T, ~152k] tensors "
                    "plus the host reduction, which is what StateEvaluator did"),
                "_equivalence": (
                    "RELATIVE drift 3.03e-05 on the pooled per-item KL of four "
                    "calibration items, with item ordering identical and top-1 "
                    "agreement exact. That is a kernel-level result and NOT a "
                    "decision-level one -- see _what_this_is_not below."),
                "_what_this_is_not": (
                    "This field claimed the drift was '257x below the search's "
                    "own 0.007782 decision threshold and just under float32's "
                    "sqrt(V)*eps floor'. Review corrected both halves. (1) "
                    "0.007782 is C2's pre-B numerical-SENSITIVITY DISCLOSURE "
                    "trigger -- the tightest gap between the frozen C "
                    "candidates on worst_domain -- and its own record states it "
                    "is not a noise bound, not a variance measurement and not "
                    "evidence of determinism. The Pareto decision epsilon is "
                    "1e-4 ABSOLUTE per objective. (2) Dividing an absolute gap "
                    "by a RELATIVE drift yields a ratio with no units, so the "
                    "quotient was not a safety factor. (3) sqrt(V)*eps is an "
                    "error-SCALE heuristic, not a hard floor below which "
                    "agreement is impossible. The decision-level evidence is "
                    "the ABSOLUTE drift on the ranked objectives over the "
                    "COMPLETE suite, certified separately in "
                    "logs/stages/stage-1/phase_c2/validations/"
                    "state-eval-certification/v1/."),
            },
            "depth_reduction": {
                "run": DEPTH_RUN,
                "old_seconds": DEPTH_OLD_SECONDS,
                "new_seconds": DEPTH_NEW_SECONDS,
                "speedup": round(depth_speedup, 3),
                "_both_sides_device_resident": (
                    "which is the correct comparison for this candidate: the "
                    "DEPTH reduction was already device-resident before this "
                    "round, so there is no transfer to remove -- that is "
                    "candidate 1's win and it is measured separately"),
                "_equivalence": (
                    "removal order [17, 18] from both variants, 71 candidate "
                    "evaluations each, identical per-round tables"),
            },
            "state_eval_suite": {
                "artifact": STATE_EVAL_SUITE,
                "prediction_positions": positions,
                "_not_the_calibration_mixture": (
                    "an earlier draft of this arithmetic used the calibration "
                    "mixture's 59,763 positions, a different artifact, 24% low"),
            },
            "state_eval_saving_minutes": round(state_eval_saving_s / 60, 4),
        },
        "cells": cells,
        "deliberately_not_claimed": {
            "reference_cache_recomputes": (
                "182,780 reference recomputes against 323,180 ablated forwards "
                "across the nineteen measured DEPTH expansions -- 36.1% of "
                "every forward pass, and the largest known saving in the "
                "search. This round INSTRUMENTED it and changed nothing: the "
                "L40S measurement found 0.013 GiB reclaimable by "
                "`empty_cache()` on a card holding only the teacher, with the "
                "67-item cache admitting in full, so allocator hoarding does "
                "not explain the historical 2.6-GiB-free observations. No "
                "behaviour changed, so no saving is claimed."),
            "candidate_3": (
                "not adopted. Caching the normalized reference needs 33.83 GiB "
                "against a 16.91 GiB binding constraint."),
        },
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    doc = refreshed()
    body = json.dumps(doc, indent=1) + "\n"
    m = doc["measured"]
    print(f"reduction   {m['reduction']['speedup']}x  "
          f"({m['reduction']['old_ms_per_position']} -> "
          f"{m['reduction']['new_ms_per_position']} ms/position), drift "
          f"{m['reduction']['worst_relative_drift']:.3e}")
    print(f"depth       {m['depth_reduction']['speedup']}x  "
          f"({m['depth_reduction']['old_seconds']} -> "
          f"{m['depth_reduction']['new_seconds']} s)")
    print(f"suite       {m['state_eval_suite']['prediction_positions']} positions "
          f"-> {m['state_eval_saving_minutes']} min saved per state_eval\n")
    print(f"{'implementation':42} {'was':>7} {'now':>7} {'factor':>7}")
    for impl, cell in sorted(doc["cells"].items(),
                             key=lambda kv: -kv[1]["observed_total_minutes"]):
        print(f"{impl:42} {cell['observed_total_minutes']:7.2f} "
              f"{cell['refreshed_total_minutes']:7.2f} "
              f"{cell['reduction_factor']:6.2f}x"
              + ("  [capped]" if cell["state_eval_saving_was_capped"] else ""))
    print("\nAUTHORIZES NOTHING.")

    if args.write:
        (REPO_ROOT / OUT).write_text(body)
        print(f"wrote {OUT}")
    elif (REPO_ROOT / OUT).is_file() and (REPO_ROOT / OUT).read_text() == body:
        print("unchanged")
    else:
        print("WOULD CHANGE — pass --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
