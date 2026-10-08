#!/usr/bin/env python3
"""Price the Phase-C2 baseline-completion session from Attempt-4's own measurements.

    PYTHONPATH=src:scripts python scripts/autoinit/price_c2_baseline_completion.py \
        --run-id attempt4 --stage-id 1 [--write]

The session being priced does one thing: rebuild the frozen baseline B through
its complete deterministic fixed path, measure it once on the frozen
`state_eval` suite, and compute the B->C comparison against the five candidate
measurements Attempt 4 already froze. It runs no beam search.

**Why this document exists at all.** Attempt 4's `baseline_rebuild_reserve` was
27.665 min and the rebuild did not finish. That reserve was derived from a
single historical wall-clock slice -- C1 attempt 18's stage C->D, 24.185 min --
used as an upper bound for work whose cost this repository had already measured
as variable. In the SAME session that the reserve failed, seven complete DEPTH
derivations were observed at 21.30 to 32.90 min, so two of them individually
exceeded the entire reserve for the four-operator path. A single favourable
observation is not a bound.

So every line below is derived from an observation of the same work on the same
hardware, and each one says which. Nothing is copied from the old reserve.

**The unit of DEPTH cost is evaluations, not minutes.** A complete
`depth.causal_kl_greedy_v1` derivation on this geometry is exactly 260
evaluations -- rounds 0..7 over 36,35,34,33,32,31,30,29 candidates -- observed
seven times out of seven. Minutes are that count divided by a rate, and the rate
varied 7.05 to 12.22 eval/min within one session. Pricing the count against the
SLOWEST observed rate is a bound; pricing an average duration is not.

The refused rebuild is the slowest observation and it is the one used: it ran at
7.05 eval/min with only 2.6 GiB of device memory free, which forced the
reference cache down to 11 of 67 items. Whether a session that runs no beam
first frees enough for the full 16.9 GiB cache is UNMEASURED, so no credit is
taken for it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402

SCHEMA = "aadistill.autoinit.c2_baseline_completion_pricing/v1"

#: The rate the L40S securePrice was accepted at for every C2 formal session.
#: Reported against, never changed here.
PRICE_PER_HOUR = 1.09

#: A complete DEPTH derivation on this geometry, from the search's own logs.
#: Asserted against the evidence rather than trusted.
DEPTH_EVALS_COMPLETE = 260

#: The convention every C2 total has used: a tenth of the expected path.
CONTINGENCY_FRACTION = 0.10

STEP_IMPLS = ("depth.causal_kl_greedy_v1", "ffn.activation_importance_v0",
              "width.global_pca_v0", "attention.activation_importance_v1")


def _seconds(record: dict, key: str) -> float:
    value = record.get(key)
    if isinstance(value, dict):
        return float(value.get("seconds", 0.0))
    return float(value or 0.0)


def observe(run_dir: Path) -> dict:
    """Every measurement this pricing rests on, read from Attempt 4's evidence."""
    telemetry = [json.loads(line) for line
                 in (run_dir / "evidence/telemetry.jsonl").read_text().splitlines()
                 if line.strip()]
    log = (run_dir / "evidence/driver_run.log").read_text().splitlines()
    session = json.loads((run_dir / "runtime/session.json").read_text())

    import re
    pattern = re.compile(
        r"round (\d+) candidate (\d+)/(\d+) \(layer \d+\) score \S+ . "
        r"(\d+) evals . ([\d.]+) min . ([\d.]+) eval/min")
    derivations, current = [], None
    for line in log:
        match = pattern.search(line)
        if not match:
            continue
        evals = int(match.group(4))
        if current is None or evals < current["evals"]:
            if current:
                derivations.append(current)
        current = {"evals": evals, "minutes": float(match.group(5)),
                   "rate": float(match.group(6))}
    if current:
        derivations.append(current)

    complete = [d for d in derivations if d["evals"] == DEPTH_EVALS_COMPLETE]
    partial = [d for d in derivations if d["evals"] != DEPTH_EVALS_COMPLETE]
    if not complete:
        raise SystemExit(
            "no complete DEPTH derivation is present in this run's log, so the "
            "cost of a complete one cannot be derived from it")

    per_impl = {}
    for impl in STEP_IMPLS:
        rows = [r for r in telemetry if r["impl_id"] == impl]
        if not rows:
            raise SystemExit(f"{impl} was never observed in this run's telemetry")
        per_impl[impl] = {
            "n_observations": len(rows),
            "operator_seconds_min": min(r["operator_seconds"] for r in rows),
            "operator_seconds_max": max(r["operator_seconds"] for r in rows),
        }

    overhead_keys = ("parent_load_seconds", "materialize_seconds",
                     "canonical_reload_seconds", "identify_seconds")
    return {
        "n_expansions": len(telemetry),
        "depth": {
            "complete_derivations": len(complete),
            "evals_per_complete_derivation": DEPTH_EVALS_COMPLETE,
            "minutes_min": min(d["minutes"] for d in complete),
            "minutes_max": max(d["minutes"] for d in complete),
            "minutes_mean": round(
                sum(d["minutes"] for d in complete) / len(complete), 3),
            "rate_min_eval_per_min": min(d["rate"] for d in derivations),
            "rate_max_eval_per_min": max(d["rate"] for d in derivations),
            "refused_rebuild": partial[-1] if partial else None,
        },
        "per_impl": per_impl,
        "per_step_overhead_seconds_max": sum(
            max(_seconds(r, key) for r in telemetry) for key in overhead_keys),
        "state_eval_seconds_max": max(
            _seconds(r, "state_evaluation_seconds") for r in telemetry),
        "session_cost_usd": session["cost"]["actual_usd"],
        "session_elapsed_minutes": session["cost"]["elapsed_minutes"],
    }


def price(observed: dict, *, setup_minutes: float, collect_minutes: float,
          teardown_minutes: float) -> dict:
    """One line per unit of work, each naming the observation that bounds it."""
    depth = observed["depth"]
    slowest = depth["rate_min_eval_per_min"]
    depth_bound = DEPTH_EVALS_COMPLETE / slowest
    #: Typical, not worst: the ceiling below is where conservatism belongs, and
    #: an "expected" line that quietly carries the maximum makes the two
    #: indistinguishable.
    depth_expected = depth["minutes_mean"]

    def op_minutes(impl: str) -> float:
        return observed["per_impl"][impl]["operator_seconds_max"] / 60.0

    overhead = observed["per_step_overhead_seconds_max"] * len(STEP_IMPLS) / 60.0
    state_eval = observed["state_eval_seconds_max"] / 60.0

    items = [
        {"item": "setup_and_staging", "minutes": setup_minutes, "basis": "measured",
         "source": ("Attempt 4's own setup on this image: SETUP_RC=0 in 3 min 13 s. "
                    "Attempt 3's was 2 min 33 s on the same image; the larger is used.")},
        {"item": "driver_startup_teacher_load_bookkeeping", "minutes": 1.5,
         "basis": "measured_bound",
         "source": ("Attempt 4's 44 expansion records account for 294.85 of the "
                    "295.89 min between stage A passing and the selection being "
                    "committed, so EVERYTHING outside per-expansion work -- the "
                    "teacher root load included -- cost 1.04 min across the whole "
                    "beam. Rounded up.")},
        {"item": "depth_full_derivation", "minutes": depth_bound, "basis": "derived",
         "expected_basis": ("the mean of the seven complete derivations observed "
                            "in Attempt 4; the bound beside it divides the eval "
                            "count by the slowest rate observed"),
         "source": (f"{DEPTH_EVALS_COMPLETE} evaluations, the exact count of a "
                    f"complete derivation on this geometry, observed "
                    f"{depth['complete_derivations']}/{depth['complete_derivations']} "
                    f"times, divided by {slowest} eval/min -- the SLOWEST rate "
                    "observed in Attempt 4, which is the rate the refused rebuild "
                    "itself ran at under 2.6 GiB of free device memory.")},
        {"item": "ffn", "minutes": op_minutes("ffn.activation_importance_v0"),
         "basis": "measured", "source": "max operator_seconds over Attempt 4's observations"},
        {"item": "residual_width", "minutes": op_minutes("width.global_pca_v0"),
         "basis": "measured", "source": "max operator_seconds over Attempt 4's observations"},
        {"item": "attention", "minutes": op_minutes("attention.activation_importance_v1"),
         "basis": "measured", "source": "max operator_seconds over Attempt 4's observations"},
        {"item": "materialization_reload_identity_x4", "minutes": overhead,
         "basis": "measured",
         "source": ("the per-step maxima of parent_load, materialize, "
                    "canonical_reload and identify, summed and applied to all four "
                    "steps of the path")},
        {"item": "state_eval_of_B_once", "minutes": state_eval, "basis": "measured",
         "source": "max state_evaluation_seconds over Attempt 4's 44 observations"},
        {"item": "b_to_c_comparison_post_processing", "minutes": 1.0,
         "basis": "allowance",
         "source": ("arithmetic over one B evaluation and five frozen C "
                    "evaluations, then one record written. No observation exists "
                    "because Attempt 4 never reached it; an explicit minute is "
                    "more honest than a derived fraction of nothing.")},
        {"item": "evidence_collection_and_sync", "minutes": collect_minutes,
         "basis": "measured",
         "source": ("Attempt 4 collected 7 files including the 28.9 MB journal, "
                    "archived, transferred and verified them in 2.27 min. This "
                    "session's artifacts are smaller.")},
        {"item": "teardown_and_provider_confirmation", "minutes": teardown_minutes,
         "basis": "measured", "source": "Attempt 4: pod deleted and confirmed gone in 11 s"},
    ]

    expected_items = []
    for entry in items:
        minutes = depth_expected if entry["item"] == "depth_full_derivation" else entry["minutes"]
        expected_items.append({**entry, "minutes": minutes})

    bound_path = sum(e["minutes"] for e in items)
    expected_path = sum(e["minutes"] for e in expected_items)
    contingency = bound_path * CONTINGENCY_FRACTION

    #: Sized to THIS session's artifacts rather than copied from a session that
    #: produced probe checkpoints. Attempt 4's whole collect-archive-transfer-
    #: verify-teardown path took 2.45 min for 29 MB; a completion session
    #: produces a state evaluation and a comparison record.
    recovery_reserve = 10.0

    hard_minutes = bound_path + contingency + recovery_reserve
    return {
        "line_items_bounding": items,
        "line_items_expected": expected_items,
        "totals": {
            "expected_minutes": round(expected_path, 3),
            "expected_usd": round(expected_path / 60.0 * PRICE_PER_HOUR, 4),
            "bounding_path_minutes": round(bound_path, 3),
            "contingency_fraction": CONTINGENCY_FRACTION,
            "contingency_minutes": round(contingency, 3),
            "artifact_recovery_reserve_minutes": recovery_reserve,
            "hard_ceiling_minutes": round(hard_minutes, 3),
            "hard_ceiling_usd": math.ceil(hard_minutes / 60.0 * PRICE_PER_HOUR * 1e4) / 1e4,
            "price_per_hour": PRICE_PER_HOUR,
            "_ceiling_rounds_up": ("a ceiling rounds UP or it under-authorizes the "
                                   "plan it prices"),
            "_the_ceiling_covers_the_whole_B_path": (
                "every operator of the frozen path, its materialization and "
                "identity checks, the single state_eval and the comparison are "
                "line items above. The reserve that failed in Attempt 4 was "
                "smaller than the DEPTH step alone at its observed worst."),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="attempt4")
    ap.add_argument("--stage-id", default="1")
    ap.add_argument("--experiment-id", default="phase_c2")
    ap.add_argument("--project-remaining-usd", type=float, default=None,
                    help=("if given, also report the maximum securePrice this "
                          "package could tolerate under it. Reporting only; the "
                          "accepted rate boundary is not changed here."))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    run_rel = rel_run_dir(args.experiment_id, args.run_id, args.stage_id)
    observed = observe(REPO / run_rel)
    priced = price(observed, setup_minutes=3.217, collect_minutes=2.267,
                   teardown_minutes=0.183)

    doc = {
        "schema": SCHEMA,
        "_what_this_prices": (
            "ONE Phase-C2 baseline-completion session: rebuild the frozen "
            "baseline B through its complete deterministic fixed path, measure it "
            "once on the frozen state_eval suite, compute the B->C comparison "
            "against the five candidate measurements Attempt 4 froze, collect and "
            "tear down. NO beam search, no new candidate, no remeasurement of any C."),
        "_does_not_reuse_the_failed_reserve": (
            "Attempt 4's baseline_rebuild_reserve of 27.665 min is not an input "
            "here and no line below is derived from it. It was a single "
            "historical wall-clock slice used as a bound for work the same "
            "session then observed taking 21.30 to 32.90 min."),
        "derived_from": {
            "run": run_rel,
            "telemetry": f"{run_rel}/evidence/telemetry.jsonl",
            "driver_log": f"{run_rel}/evidence/driver_run.log",
            "session_record": f"{run_rel}/runtime/session.json",
        },
        "observations": observed,
        **priced,
        "hardware": {
            "gpu": "NVIDIA L40S",
            "price_basis_usd_per_hour": PRICE_PER_HOUR,
            "_rate_boundary_unchanged": (
                "the accepted formal boundary remains securePrice <= $1.09/h. "
                "This document does not change it and no budget increase is "
                "requested by it."),
        },
        "authorizes": "nothing",
    }
    if args.project_remaining_usd is not None:
        hours = priced["totals"]["hard_ceiling_minutes"] / 60.0
        doc["rate_tolerance_informational"] = {
            "project_remaining_usd": args.project_remaining_usd,
            "hard_ceiling_hours": round(hours, 4),
            "max_tolerable_secure_price_usd_per_hour": round(
                args.project_remaining_usd / hours, 4),
            "_meaning": ("the highest hourly rate at which this session's own hard "
                         "ceiling would still fit the project's remaining balance. "
                         "Informational: the accepted boundary is unchanged and "
                         "remaining balance is not permission."),
        }
    doc["pricing_sha256"] = sha256_json({k: v for k, v in doc.items()
                                         if k != "pricing_sha256"})

    out = (REPO / "logs/stages/stage-1/phase_c2/plans"
           / "phase_c2_baseline_completion_pricing.json")
    if args.write:
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out.relative_to(REPO)}")
    totals = doc["totals"]
    print(f"  DEPTH bound       {totals['bounding_path_minutes']:.2f} min path, of which "
          f"DEPTH {DEPTH_EVALS_COMPLETE}/{observed['depth']['rate_min_eval_per_min']} "
          f"= {DEPTH_EVALS_COMPLETE / observed['depth']['rate_min_eval_per_min']:.2f} min")
    print(f"  expected          {totals['expected_minutes']:.2f} min  ${totals['expected_usd']:.4f}")
    print(f"  hard ceiling      {totals['hard_ceiling_minutes']:.2f} min  "
          f"${totals['hard_ceiling_usd']:.4f}  at ${PRICE_PER_HOUR}/h")
    if "rate_tolerance_informational" in doc:
        r = doc["rate_tolerance_informational"]
        print(f"  tolerable rate    <= ${r['max_tolerable_secure_price_usd_per_hour']:.4f}/h "
              f"under ${r['project_remaining_usd']:.4f} remaining (informational)")
    print(f"  pricing_sha256    {doc['pricing_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
