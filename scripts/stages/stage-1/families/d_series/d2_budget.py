#!/usr/bin/env python3
"""The executable budget for the steps before and through D2. Derived, not typed.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/families/d_series/d2_budget.py [--write]

Every figure comes from a committed owner:

* R0 -- its registered plan;
* the reference calibration session -- R0's measured setup components, D1's
  bounding arm-materialization minutes and a conservative compute allowance;
* the D2 search -- D1's PRICED search (the same space and support), with an
  explicit, provisional factor for the objective's unmeasured overhead;
* D2 behaviour -- D1's priced behavioural conventions (per-probe bounding
  minutes, overrun factor, session overhead, minutes per arm), applied to the
  shared protocol's field: K new candidates and NO B/q2 probes or arms.

The rollout-rescoring diagnostic is NOT in any hard ceiling: its cost is
measured by the reference calibration first, as the independent review
required. Its provisional figure is shown separately and labelled so.

AUTHORIZES NOTHING: every paid step needs its own grant at the live rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.d_series.d2_budget/v1"
RECORD = "logs/stages/stage-1/families/d_series/plans/d2_budget_v1.json"
D1_DESIGN = "logs/stages/stage-1/phase_d1/plans/d1_design.json"
R0_PLAN = ("logs/stages/stage-1/families/d_series/validations/prediction-rows-r0/"
           "v1/plan.json")

#: Container-disk billing, $/GB/month, as the launchers price it.
DISK_USD_PER_GB_MONTH = 0.10
TEARDOWN_RESERVE_USD = 0.30
#: The D2 objective's overhead is ESTIMATED at +3-7% of DEPTH and +5-10% of
#: state evaluation and not yet measured; the ceiling applies the top of the
#: estimate to every structural minute until the first D2 expansion's
#: `objective_seconds` replaces it.
OBJECTIVE_OVERHEAD_PROVISIONAL = 1.10
#: The rescoring diagnostic's estimate per probe, NOT a bound.
RESCORING_MINUTES_PER_PROBE_ESTIMATE = 5.0


def _read(rel: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / rel).read_text())


def _usd(minutes: float, rate: float, disk_gb: float) -> dict[str, float]:
    gpu = minutes / 60 * rate
    disk = disk_gb * DISK_USD_PER_GB_MONTH / (30 * 24 * 60) * minutes
    return {"minutes": round(minutes, 1), "gpu_usd": round(gpu, 2),
            "disk_usd": round(disk, 2), "all_in_usd": round(gpu + disk, 2)}


def build(k_candidates: int = 4) -> dict[str, Any]:
    chain = _read(D1_DESIGN)["budget"]["chain"]
    rate = float(chain["price_per_hour"])
    search = chain["sessions"]["search"]
    screening = chain["sessions"]["screening"]
    r0 = _read(R0_PLAN)["time_and_money"]

    #: R0: its plan, unchanged.
    r0_row = {"expected": {"all_in_usd": r0["expected"]["all_in_usd"],
                           "minutes": r0["expected"]["minutes"]},
              "hard": {"all_in_usd": r0["hard_ceiling_with_recompute_fallback"][
                           "all_in_usd"],
                       "minutes": r0["hard_ceiling_with_recompute_fallback"][
                           "minutes"]},
              "grant_proposed_usd": r0["proposed_grant"]["all_in_ceiling_usd"],
              "owner": R0_PLAN}

    #: Reference calibration: R0's measured setup components + one arm (B).
    per_arm = float(screening["minutes_per_arm_materialization"])
    setup_exp = sum(c[1] for c in r0["expected"]["components_minutes"][:2])
    setup_hard = sum(c[1] for c in r0["ceiling_cached_path"][
        "components_minutes"][:2])
    calib_exp = setup_exp + 21.3 + 1.0 + 1.0 + 1.0 + 1.0
    calib_hard = setup_hard + per_arm + 2.0 + 5.0 + 5.0 + 2.0
    calib = {"expected": _usd(calib_exp, rate, 60),
             "hard": _usd(calib_hard, rate, 60),
             "components": {
                 "setup_and_overhead": [setup_exp, setup_hard],
                 "b_materialization": [21.3, per_arm],
                 "teacher_fetch_and_load": [1.0, 2.0],
                 "teacher_and_b_forwards_147_items": [1.0, 5.0],
                 "rescoring_timing": [1.0, 5.0],
                 "teardown": [1.0, 2.0]},
             "_basis": ("setup from R0's plan (measured a2/a8 components); B "
                        "rebuild measured 1279.8 s (expected) and D1's 22 min "
                        "per arm (bound); forwards from state-evaluation "
                        "timings (80 items in ~11 s at the frozen suite) with a "
                        "5x allowance")}
    calib["grant_proposed_usd"] = round(calib["hard"]["all_in_usd"]
                                        + TEARDOWN_RESERVE_USD + 0.04, 2)

    #: D2 search: D1's priced search, objective overhead provisional.
    overhead = float(search["session_overhead_minutes"])
    struct_max = float(search["structural_minutes_max"])
    search_hard_min = overhead + struct_max * OBJECTIVE_OVERHEAD_PROVISIONAL
    d1_measured_min = 551.99
    search_exp_min = d1_measured_min * 1.05
    disk_search = float(search["container_disk_gb"])
    d2_search = {"expected": _usd(search_exp_min, rate, disk_search),
                 "hard": _usd(search_hard_min, rate, disk_search),
                 "_basis": (
                     "D1's priced search (same space, support and beam): "
                     f"session overhead {overhead} min + structural max "
                     f"{struct_max} min x {OBJECTIVE_OVERHEAD_PROVISIONAL} "
                     "(PROVISIONAL objective overhead). Expected: D1's "
                     "measured 551.99 min x 1.05. No scored_v1 saving is "
                     "assumed until R0 passes")}

    #: D2 behaviour on the shared protocol: K candidates, no B/q2 probes.
    probe_bound = float(screening["minutes_per_probe_bounding"])
    overrun = float(screening["overrun_factor"])
    sess_over = float(screening["session_overhead_minutes"])
    disk_beh = float(screening["container_disk_gb"])

    def behaviour(n_probes: int, n_arms: int) -> dict[str, Any]:
        fixed = sess_over + n_arms * per_arm
        probes = n_probes * probe_bound
        return {"n_probes": n_probes, "n_arms_materialized": n_arms,
                "expected": _usd(fixed + probes, rate, disk_beh),
                "hard": _usd(fixed + probes * overrun, rate, disk_beh),
                "rescoring_provisional_not_a_bound": _usd(
                    n_probes * RESCORING_MINUTES_PER_PROBE_ESTIMATE * overrun,
                    rate, disk_beh)}

    d2_screening = behaviour(2 * k_candidates, k_candidates)
    d2_confirmation = behaviour(3, 1)

    rows = {"r0": r0_row, "reference_calibration": calib,
            "d2_search": d2_search, "d2_screening": d2_screening,
            "d2_confirmation": d2_confirmation}
    exp_total = round(r0_row["expected"]["all_in_usd"]
                      + sum(rows[k]["expected"]["all_in_usd"]
                            for k in ("reference_calibration", "d2_search",
                                      "d2_screening", "d2_confirmation")), 2)
    hard_total = round(r0_row["hard"]["all_in_usd"]
                       + sum(rows[k]["hard"]["all_in_usd"]
                             for k in ("reference_calibration", "d2_search",
                                       "d2_screening", "d2_confirmation")), 2)
    sessions = 5
    return {
        "schema": SCHEMA,
        "_contract": ("Derived budget for R0, the reference calibration and D2 "
                      "(search + behaviour on d_series_shared_eval@v1). "
                      "AUTHORIZES NOTHING; each paid step needs its own grant "
                      "at the live rate."),
        "rate_usd_per_hour_basis": rate,
        "k_candidates": k_candidates,
        "k_status": ("K=4 PREPARED per the independent review; frozen only in "
                     "D2's preregistration"),
        "order": ["r0", "reference_calibration", "d2_search", "d2_screening",
                  "d2_confirmation"],
        "steps": rows,
        "totals": {"expected_usd": exp_total, "hard_ceiling_usd": hard_total,
                   "teardown_reserves_usd": round(sessions
                                                  * TEARDOWN_RESERVE_USD, 2),
                   "hard_plus_reserves_usd": round(
                       hard_total + sessions * TEARDOWN_RESERVE_USD, 2),
                   "rescoring_excluded": (
                       "the rollout-rescoring diagnostic adds about "
                       f"${d2_screening['rescoring_provisional_not_a_bound']['all_in_usd'] + d2_confirmation['rescoring_provisional_not_a_bound']['all_in_usd']:.2f} "
                       "at its ESTIMATE; it joins the ceilings only after the "
                       "reference calibration measures it")},
        "per_session_cap_usd": 30.0,
        "every_session_under_cap": all(
            rows[k]["hard"]["all_in_usd"] <= 30.0
            for k in ("reference_calibration", "d2_search", "d2_screening",
                      "d2_confirmation")),
        "authorizes": "nothing",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    body = json.dumps(build(), indent=1) + "\n"
    path = REPO_ROOT / RECORD
    if args.write:
        path.write_text(body)
        print(f"wrote {RECORD}")
        return 0
    same = path.is_file() and path.read_text() == body
    print(f"{RECORD}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
