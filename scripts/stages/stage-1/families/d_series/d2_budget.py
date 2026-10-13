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
#: state evaluation and not yet measured. Applied ONLY to the two paths the
#: objective touches -- the DEPTH scoring loop and the state evaluation --
#: as added seconds, until the reference calibration's stage L measures them
#: (`--calibration-session`). Expected at the middle, planning bound at the top.
OBJECTIVE_OVERHEAD_PROVISIONAL = 1.10
OBJECTIVE_OVERHEAD_EXPECTED = 1.05

#: D1's own search telemetry, 92/92 expansions, copied from the out-of-tree
#: store and verified against the hash D1's committed evidence manifest
#: records for it. The same space, support, beam, batch and L40S as D2.
D1_TELEMETRY = ("logs/stages/stage-1/families/d_series/analyses/"
                "d1_search_telemetry.jsonl")
D1_SEARCH_RUN = "logs/stages/stage-1/phase_d1/runs/d1_search_20261006_210210"
D1_TELEMETRY_IN_MANIFEST = "audit/autoinit_d1/search/telemetry.jsonl"
_PHASES = ("canonical_reload_seconds", "identify_seconds", "materialize_seconds",
           "state_evaluation_seconds", "validation_seconds")
DEPTH_IMPL = "depth.causal_kl_greedy_v1"


def d1_search_telemetry() -> list[dict[str, Any]]:
    """D1's telemetry rows, refused unless the bytes are the ones D1 recorded."""
    import hashlib

    raw = (REPO_ROOT / D1_TELEMETRY).read_bytes()
    manifest = _read(f"{D1_SEARCH_RUN}/evidence/manifest.json")
    want = {e["path"]: e["sha256"] for e in manifest["entries"]}[
        D1_TELEMETRY_IN_MANIFEST]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise ValueError(f"{D1_TELEMETRY} hashes to {got}, not the {want} D1's "
                         "evidence manifest records; it is not D1's telemetry")
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def telemetry_basis(rows: Sequence[dict[str, Any]], elapsed_minutes: float
                    ) -> dict[str, Any]:
    """Per operator and position (root / deeper): end-to-end cell max and mean.

    A cell is one expansion end to end -- operator, parent load and the five
    phases -- the quantity `CostModel` prices. Also the two paths the objective
    touches, separated: the DEPTH scoring loop and the state evaluation.
    """
    ids = {r["state_id"] for r in rows}
    roots = {r["parent_id"] for r in rows} - ids
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {sorted(roots)}")

    def cell(r):
        return (float(r["operator_seconds"]) + float(r["parent_load_seconds"])
                + sum(float((r.get(k) or {}).get("seconds") or 0) for k in _PHASES))

    cells: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        where = "root" if r["parent_id"] in roots else "deeper"
        cells.setdefault(r["impl_id"], {}).setdefault(where, []).append(cell(r) / 60)
    depth = [r for r in rows if r["impl_id"] == DEPTH_IMPL]
    loop = [float(r["operator_timing"]["item_seconds"]) for r in depth]
    cands = [int(r["operator_timing"]["candidate_subsets"]) for r in depth]
    se = [float((r.get("state_evaluation_seconds") or {}).get("seconds") or 0)
          for r in rows]
    total_cells = sum(cell(r) for r in rows) / 60
    return {
        "expansions": len(rows),
        "cells_minutes": {
            impl: {f"{w}_{stat}": round(fn(v), 4) for w, v in by.items()
                   for stat, fn in (("max", max),
                                    ("mean", lambda x: sum(x) / len(x)))}
            for impl, by in sorted(cells.items())},
        "counts": {impl: {w: len(v) for w, v in by.items()}
                   for impl, by in sorted(cells.items())},
        "depth": {"expansions": len(depth),
                  "candidates_per_expansion": sorted(set(cands)),
                  "scoring_loop_minutes": round(sum(loop) / 60, 2),
                  "loop_seconds_per_candidate_mean": round(
                      sum(loop) / max(1, sum(cands)), 4),
                  "reference_cache_modes": sorted(
                      {str((r.get("reference_cache") or {}).get("mode"))
                       for r in depth}),
                  "recomputed": sum(1 for r in depth if (
                      r.get("reference_cache") or {}).get("mode") == "recomputed")},
        "state_evaluation": {"evaluations": len(se),
                             "minutes": round(sum(se) / 60, 2),
                             "seconds_mean": round(sum(se) / len(se), 3),
                             "seconds_max": round(max(se), 3)},
        "cells_total_minutes": round(total_cells, 2),
        "session_minutes": elapsed_minutes,
        "outside_cells_minutes": round(elapsed_minutes - total_cells, 2),
    }


def objective_overhead(calibration_session: str | Path | None,
                       basis: dict[str, Any]) -> dict[str, Any]:
    """Seconds the D2 objective adds per DEPTH candidate and per evaluation.

    MEASURED from a reference-calibration session's stage L when one is given
    and both pairs completed on one cache mode; otherwise PROVISIONAL, from
    the estimate, and labelled so.
    """
    per_cand = basis["depth"]["loop_seconds_per_candidate_mean"]
    se_mean = basis["state_evaluation"]["seconds_mean"]
    provisional = {
        "source": "provisional -- an estimate, not a measurement",
        "depth_seconds_per_candidate": {
            "expected": round(per_cand * (OBJECTIVE_OVERHEAD_EXPECTED - 1), 4),
            "bound": round(per_cand * (OBJECTIVE_OVERHEAD_PROVISIONAL - 1), 4)},
        "state_eval_seconds_per_evaluation": {
            "expected": round(se_mean * (OBJECTIVE_OVERHEAD_EXPECTED - 1), 4),
            "bound": round(basis["state_evaluation"]["seconds_max"]
                           * (OBJECTIVE_OVERHEAD_PROVISIONAL - 1), 4)}}
    if calibration_session is None:
        return provisional
    doc = json.loads(Path(calibration_session).read_text())
    lc = doc.get("L_reference_calibration") or {}
    d, e = lc.get("timing_depth_one_round") or {}, lc.get("timing_state_eval") or {}
    usable = ("overhead_seconds_per_candidate" in d and d.get("same_cache_mode")
              and "overhead_seconds_per_evaluation" in e)
    if not usable:
        return {**provisional, "measurement_unusable": {
            "session": str(calibration_session),
            "depth": {k: d.get(k) for k in ("skipped", "failed",
                                            "same_cache_mode")},
            "state_eval": {k: e.get(k) for k in ("skipped", "failed")}}}
    dc = max(0.0, float(d["overhead_seconds_per_candidate"]))
    se = max(0.0, float(e["overhead_seconds_per_evaluation"]))
    return {"source": f"measured: {calibration_session} (stage L)",
            "depth_seconds_per_candidate": {"expected": dc, "bound": dc},
            "state_eval_seconds_per_evaluation": {"expected": se, "bound": se},
            "_one_measurement": ("one GPU-bound pair at the root; the per-"
                                 "candidate work the objective adds does not "
                                 "depend on the parent's depth")}


def priced_from_records(rate: float, disk_gb: int,
                        calibration_session: str | Path | None = None,
                        depth_cell: str = "records_max") -> dict[str, Any]:
    """D2's search priced by D1's machinery on a cost model rebuilt from D1's
    own search telemetry, plus the objective's added seconds.

    No second pricing formula: `stages.phase_d1.search_space.search_cost` on a
    `CostModel` whose DEPTH cell is D1's measured end-to-end MAX (the
    recompute-mode reality: 15 of D1's 16 DEPTH expansions recomputed the
    reference cache) and whose other cells keep the frozen, more conservative
    cross-host values (D1's own were 3-4x smaller on a fast host). Expected =
    D1's measured session + the objective's expected added minutes over D1's
    actual trajectory.

    ``depth_cell="cached_validated"`` prices DEPTH at D1's design cell for a
    CACHED reference (a8's measured invocation, end to end) instead -- valid
    only once the allocator-hoard fix in `_available_memory_bytes` is shown on
    a GPU to keep the cache under a hoarded pool (stage L records it).
    """
    import dataclasses

    from stages.phase_d1 import search_space as d1
    from stages.phase_d1 import write_d1_design as W

    outcome = _read(f"{D1_SEARCH_RUN}/closeout/outcome.json")
    elapsed = float(outcome["cost"]["elapsed_minutes"])
    basis = telemetry_basis(d1_search_telemetry(), elapsed)
    ov = objective_overhead(calibration_session, basis)
    cands = max(basis["depth"]["candidates_per_expansion"])
    W._ensure_the_frozen_operators_are_registered()
    frozen = d1.cost_model()
    minutes = {k: dict(v) for k, v in frozen.minutes.items()}
    measured = basis["cells_minutes"]

    def add(which: str) -> float:
        return (ov["state_eval_seconds_per_evaluation"][which] / 60)

    for impl, cell in minutes.items():
        if impl == DEPTH_IMPL:
            dep = (measured[DEPTH_IMPL] if depth_cell == "records_max" else
                   _read(D1_DESIGN)["budget"]["topk_production_basis"][
                       "depth_cell_after"])
            extra = cands * ov["depth_seconds_per_candidate"]["bound"] / 60
            cell.update({k: round(dep[k] + extra + add("bound"), 4)
                         for k in ("root_max", "deeper_max")})
            cell.update({"root_mean": cell["root_max"],
                         "deeper_mean": cell["deeper_max"]})
        else:
            cell.update({k: round(v + add("bound"), 4) for k, v in cell.items()})
    model = dataclasses.replace(frozen, minutes=minutes, source=(
        f"{frozen.source} -- DEPTH rebuilt from D1's search telemetry "
        f"({D1_TELEMETRY}), plus the D2 objective's added seconds "
        f"({ov['source']})"))
    with W._measured_cost_model(model):
        priced = d1.search_cost(price_per_hour=rate, container_disk_gb=disk_gb)
    exp_added = (basis["depth"]["expansions"] * cands
                 * ov["depth_seconds_per_candidate"]["expected"]
                 + basis["state_evaluation"]["evaluations"]
                 * ov["state_eval_seconds_per_evaluation"]["expected"]) / 60
    return {"basis": basis, "objective_overhead": ov, "depth_cell": depth_cell,
            "depth_cell_minutes": {k: minutes[DEPTH_IMPL][k]
                                   for k in ("root_max", "deeper_max")},
            "priced": {k: priced[k] for k in (
                "expansions_max", "structural_minutes_max",
                "session_overhead_minutes", "hard_ceiling_minutes",
                "hard_ceiling_usd")},
            "expected_minutes": round(elapsed + exp_added, 2),
            "expected_added_minutes": round(exp_added, 2)}
#: The rescoring diagnostic's estimate per probe, NOT a bound.
RESCORING_MINUTES_PER_PROBE_ESTIMATE = 5.0


def _read(rel: str) -> dict[str, Any]:
    return json.loads((REPO_ROOT / rel).read_text())


def _usd(minutes: float, rate: float, disk_gb: float) -> dict[str, float]:
    gpu = minutes / 60 * rate
    disk = disk_gb * DISK_USD_PER_GB_MONTH / (30 * 24 * 60) * minutes
    return {"minutes": round(minutes, 1), "gpu_usd": round(gpu, 2),
            "disk_usd": round(disk, 2), "all_in_usd": round(gpu + disk, 2)}


def build(k_candidates: int = 4,
          calibration_session: str | Path | None = None) -> dict[str, Any]:
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
    from stages.phase_d1.topk_adoption_driver import L_PAIR_BOUND_S

    calib_exp = setup_exp + 21.3 + 1.0 + 1.0 + 1.0 + 1.0
    calib_hard = setup_hard + per_arm + 2.0 + 5.0 + 5.0 + 2.0
    #: The objective-overhead timing pairs (stage L). Expected from a3 (state
    #: evaluation 10.4-11.1 s, three evaluations) and a8 (36 of 260 candidates
    #: at p50 4.50 s + ~30 s outside the loop, twice); bound = the driver's
    #: own per-pair gate, so the plan and the gate cannot disagree.
    timing_exp = 0.7 + 7.0
    timing_bound = sum(L_PAIR_BOUND_S.values()) / 60
    calib = {"expected": _usd(calib_exp + timing_exp, rate, 60),
             "hard": _usd(calib_hard, rate, 60),
             "hard_with_timing": _usd(calib_hard + timing_bound, rate, 60),
             "components": {
                 "setup_and_overhead": [setup_exp, setup_hard],
                 "b_materialization": [21.3, per_arm],
                 "teacher_fetch_and_load": [1.0, 2.0],
                 "teacher_and_b_forwards_147_items": [1.0, 5.0],
                 "rescoring_timing": [1.0, 5.0],
                 "teardown": [1.0, 2.0],
                 "objective_timing_state_eval_pair": [
                     0.7, L_PAIR_BOUND_S["state_eval"] / 60],
                 "objective_timing_depth_one_round_pair": [
                     7.0, L_PAIR_BOUND_S["depth_one_round"] / 60]},
             "timing_pairs_are": (
                 "deadline-gated and optional: the calibration record is "
                 "written first, and a pair starts only when the driver's "
                 "deadline still covers its whole bound. So `hard` -- the "
                 "must-complete work -- prices the session cap, and the pairs "
                 "run inside it whenever the work before them is not at its "
                 "maxima (expected: always)"),
             "_basis": ("setup from R0's plan (measured a2/a8 components); B "
                        "rebuild measured 1279.8 s (expected) and D1's 22 min "
                        "per arm (bound); forwards from state-evaluation "
                        "timings (80 items in ~11 s at the frozen suite) with a "
                        "5x allowance")}
    calib["grant_proposed_usd"] = round(calib["hard"]["all_in_usd"]
                                        + TEARDOWN_RESERVE_USD + 0.04, 2)
    calib["session_cap_proposed_usd"] = calib["hard"]["all_in_usd"]
    calib["alternative_guaranteeing_timing"] = {
        "session_cap_usd": calib["hard_with_timing"]["all_in_usd"],
        "grant_usd": round(calib["hard_with_timing"]["all_in_usd"]
                           + TEARDOWN_RESERVE_USD + 0.04, 2),
        "buys": ("both timing pairs even when every earlier component is at "
                 "its maximum; a maintainer choice, not taken by this record")}

    #: D2 search: priced from D1's own search records (see priced_from_records).
    disk_search = int(search["container_disk_gb"])
    rec = priced_from_records(rate, disk_search, calibration_session)
    cached = priced_from_records(rate, disk_search, calibration_session,
                                 depth_cell="cached_validated")
    search_exp_min = rec["expected_minutes"]
    search_hard_min = rec["priced"]["hard_ceiling_minutes"]
    d2_search = {
        "measured_basis": {
            "d1_search_actual": _usd(rec["basis"]["session_minutes"], rate, 0),
            "telemetry": D1_TELEMETRY,
            "telemetry_verified_against": (
                f"{D1_SEARCH_RUN}/evidence/manifest.json"),
            "source": (f"{D1_SEARCH_RUN}/closeout/outcome.json: 92/92 "
                       "expansions, same space, support, beam and L40S"),
            "components": rec["basis"]},
        "objective_overhead": rec["objective_overhead"],
        "expected": _usd(search_exp_min, rate, disk_search),
        "hard": _usd(search_hard_min, rate, disk_search),
        "depth_cell_minutes": rec["depth_cell_minutes"],
        "priced_by": rec["priced"],
        "hard_if_cache_fix_validated": _usd(
            cached["priced"]["hard_ceiling_minutes"], rate, disk_search),
        "cache_fix": {
            "what": ("`_available_memory_bytes` returns the allocator's "
                     "hoard (empty_cache) before reading the driver: D1's "
                     "search recomputed the reference in 15/16 DEPTH "
                     "expansions with 0.12 GiB free and 36 GiB reclaimable"),
            "validated_on_gpu": False,
            "validated_by": ("stage L's DEPTH pair, which runs under a "
                             "deliberately hoarded pool and records "
                             "reference_cache.mode; or D2's own first DEPTH "
                             "expansions"),
            "until_validated": ("the planning bound prices DEPTH at D1's "
                                "measured recompute-mode maxima"),
            "depth_cell_minutes_if_validated": cached["depth_cell_minutes"]},
        "previous_model_superseded": {
            "d1_priced_structural_max_minutes": float(
                search["structural_minutes_max"]),
            "d1_priced_session_overhead_minutes": float(
                search["session_overhead_minutes"]),
            "why": ("D1's design priced DEPTH from a8's one CACHED invocation "
                    "(1194 s); D1's own search then recomputed the reference "
                    "cache in 15 of 16 DEPTH expansions, up to 2486 s. The "
                    "records model takes DEPTH from that search, and applies "
                    "the objective's overhead to the two paths it touches "
                    "instead of x1.10 on every structural minute")},
        "assumptions_not_measured": (
            [] if rec["objective_overhead"]["source"].startswith("measured")
            else ["the objective's added seconds per DEPTH candidate and per "
                  "state evaluation: +5% of D1's mean (expected) and +10% "
                  "(planning bound) until stage L measures them"]) + [
            "D2's trajectory has D1's size (92 expansions, 16 DEPTH): the "
            "beam schedule fixes the expansion count, not the operator mix"],
        "hard_is": ("a PLANNING BOUND from D1's measured maxima, not a proven "
                    "bound on runtime. What bounds MONEY is the enforceable "
                    "stop below; the bound decides only whether the search is "
                    "expected to FINISH inside it"),
        "enforceable_stop": {
            "usd": None,
            "set_at": "the D2 search authorization, at the live rate",
            "mechanism": ("the launcher's session cap: the watchdog kills the "
                          "pod at the plan's hard-terminate minutes and the "
                          "driver's soft stop fails the search closed before "
                          "it, so spend stops at the cap whatever the runtime"),
            "proposal": "the planning bound, rounded up"},
        "_basis": ("expected = D1's measured 551.99 min + the objective's "
                   "expected added minutes over D1's trajectory; bound = "
                   "D1's search_cost on the records-rebuilt cost model")}

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


# --- the programme: D2, D3 and a fresh confirmation, against every limit ----

SCENARIO_RECORD = ("logs/stages/stage-1/families/d_series/plans/"
                   "d_series_programme_budget.json")


def d1_priced_confirmation(rate: float) -> dict[str, Any]:
    """A fresh independent confirmation priced as D1's: 2 arms x 3 seeds.

    The candidate AND a common control measured on the reserved battery, both
    trained there -- the shape D1's own confirmation had. How to obtain the
    control most cheaply is decided when a SHARED_GO exists; this prices the
    conservative case.
    """
    conf = _read(D1_DESIGN)["budget"]["chain"]["sessions"]["confirmation"]
    exp_min = float(conf["expected_minutes"])
    return {"expected": _usd(exp_min, rate, float(conf["container_disk_gb"])),
            "hard": {"all_in_usd": round(float(conf["hard_ceiling_usd"]), 2),
                     "minutes": float(conf["hard_ceiling_minutes"])},
            "_basis": "d1_design.json :: budget.chain.sessions.confirmation"}


def programme_scenarios(limits: dict[str, float],
                        budget: dict[str, Any] | None = None) -> dict[str, Any]:
    """D2, then D3, then one fresh confirmation, against the live limits.

    Hard figures carry each session's teardown reserve exactly once: R0's and
    the reference calibration's are inside their grants; every formal session
    adds TEARDOWN_RESERVE_USD here. D3 reuses the reference calibration's
    lambda, so it adds no calibration session; it is priced like D2.

    TWO SEARCH PRICES, because they differ by one GPU fact: DEPTH at D1's
    measured recompute-mode maxima (`records_max`, the price today) and DEPTH
    at the cached cell once the allocator-hoard fix is shown on a GPU
    (`cache_fix_validated`). Each is checked against the per-session ceiling.
    """
    budget = budget or build()
    steps = budget["steps"]
    rate = budget["rate_usd_per_hour_basis"]
    eng = {"r0": steps["r0"]["grant_proposed_usd"],
           "reference_calibration": steps["reference_calibration"][
               "grant_proposed_usd"]}
    eng_exp = (steps["r0"]["expected"]["all_in_usd"]
               + steps["reference_calibration"]["expected"]["all_in_usd"])
    engineering = {"expected": round(eng_exp, 2),
                   "hard_with_reserves": round(sum(eng.values()), 2)}
    fresh = d1_priced_confirmation(rate)
    fresh_block = {"expected": fresh["expected"]["all_in_usd"],
                   "hard_with_reserves": round(fresh["hard"]["all_in_usd"]
                                               + TEARDOWN_RESERVE_USD, 2),
                   "rescoring_estimate": 0.0}
    k2 = build(k_candidates=2)["steps"]["d2_screening"]["hard"]["all_in_usd"]
    k4 = steps["d2_screening"]["hard"]["all_in_usd"]
    search = steps["d2_search"]

    def under(search_hard_usd: float) -> dict[str, Any]:
        hard = {"d2_search": search_hard_usd,
                "d2_screening": steps["d2_screening"]["hard"]["all_in_usd"],
                "d2_confirmation": steps["d2_confirmation"]["hard"]["all_in_usd"]}
        block = {"expected": round(sum(steps[n]["expected"]["all_in_usd"]
                                       for n in hard), 2),
                 "hard_with_reserves": round(sum(v + TEARDOWN_RESERVE_USD
                                                 for v in hard.values()), 2),
                 "rescoring_estimate": round(sum(
                     steps[n].get("rescoring_provisional_not_a_bound",
                                  {}).get("all_in_usd", 0.0) for n in hard), 2)}

        def scenario(blocks):
            exp_ = round(engineering["expected"]
                         + sum(b["expected"] for _, b in blocks), 2)
            hard_ = round(engineering["hard_with_reserves"]
                          + sum(b["hard_with_reserves"] for _, b in blocks), 2)
            resc = round(sum(b["rescoring_estimate"] for _, b in blocks), 2)
            return {"blocks": ["engineering (R0 + reference calibration)"]
                    + [n for n, _ in blocks],
                    "expected_usd": exp_, "hard_with_reserves_usd": hard_,
                    "plus_rescoring_estimate_usd": resc,
                    "project_remaining_after_hard_usd": round(
                        limits["project_remaining_usd"] - hard_ - resc, 2),
                    "fits_project_at_hard": hard_ + resc
                    <= limits["project_remaining_usd"],
                    "fits_project_at_expected": exp_
                    <= limits["project_remaining_usd"]}

        needed = (block["hard_with_reserves"] + block["rescoring_estimate"]
                  + fresh_block["hard_with_reserves"])
        left = (limits["project_remaining_usd"]
                - engineering["hard_with_reserves"]
                - block["hard_with_reserves"] - block["rescoring_estimate"])
        return {
            "search_hard_usd": search_hard_usd,
            "search_fits_per_session_cap": (search_hard_usd
                                            <= limits["per_session_cap_usd"]),
            "d2_block": block, "d3_block": dict(block),
            "scenarios": {
                "A_d2": scenario([("d2", block)]),
                "B_d2_d3": scenario([("d2", block), ("d3", block)]),
                "C_d2_d3_fresh_confirmation": scenario(
                    [("d2", block), ("d3", block),
                     ("fresh_confirmation", fresh_block)])},
            "reservation_rule": {
                "left_after_d2_at_hard_usd": round(left, 2),
                "needed_for_d3_and_fresh_at_hard_usd": round(needed, 2),
                "satisfied": left >= needed,
                "shortfall_usd": round(max(0.0, needed - left), 2)}}

    return {
        "schema": "aadistill.d_series.programme_budget/v2",
        "_contract": ("A DATED planning snapshot of D2, D3 and one fresh "
                      "independent confirmation against the live limits from "
                      "the budget authority. Re-derive before every "
                      "authorization; it permits nothing."),
        "limits": dict(limits),
        "limits_owner": ("scripts/maintenance/consolidation/derive_budget.py "
                         "(the one budget authority)"),
        "engineering_grants": eng,
        "engineering_fits_allowance": (sum(eng.values())
                                       <= limits["engineering_remaining_usd"]),
        "engineering": engineering,
        "fresh_confirmation": fresh_block,
        "reservation_rule": (
            "authorize D2's formal sessions only if the project remainder "
            "after engineering and D2 at hard ceilings still covers D3 and one "
            "fresh confirmation at theirs"),
        "search_pricing": {
            "records_max": under(search["hard"]["all_in_usd"]),
            "cache_fix_validated": under(
                search["hard_if_cache_fix_validated"]["all_in_usd"])},
        "search_pricing_in_force": (
            "cache_fix_validated" if search["cache_fix"]["validated_on_gpu"]
            else "records_max"),
        "levers_if_short": {
            "validate_the_reference_cache_fix": {
                "saves_hard_usd_per_search": round(
                    search["hard"]["all_in_usd"]
                    - search["hard_if_cache_fix_validated"]["all_in_usd"], 2),
                "how": ("the reference calibration's stage L DEPTH pair, "
                        "already in its plan, at no extra session")},
            "measure_the_objective_overhead": (
                "stage L's two timing pairs replace the provisional "
                "+5%/+10% on the DEPTH loop and the state evaluation"),
            "d3_at_k2": {"saves_hard_usd": round(k4 - k2, 2),
                         "cost": "narrower structural coverage for D3"},
            "scored_v1_after_r0": ("about 8% of DEPTH time if R0 passes; "
                                   "unmeasured"),
            "actual_spend_releases_headroom": (
                "hard ceilings are worst cases; D1's sessions spent "
                "$10.03 of a $21.49 search ceiling and $10.34 of a $15.20 "
                "confirmation ceiling. Re-deriving after each session uses "
                "the actual spend"),
            "cap_increase": "a maintainer decision, never an autonomous one",
        },
        "authorizes": "nothing",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--scenarios", action="store_true",
                    help="also write the dated programme snapshot, reading "
                         "the live limits from the budget authority")
    ap.add_argument("--calibration-session", default=None,
                    help="a reference-calibration session's adoption.json: "
                         "its stage-L timing pairs replace the provisional "
                         "objective overhead")
    args = ap.parse_args(argv)
    budget = build(calibration_session=args.calibration_session)
    if args.scenarios:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "derive_budget", REPO_ROOT / "scripts/maintenance/consolidation/"
                                         "derive_budget.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        live = mod.derive(REPO_ROOT)
        limits = {"project_remaining_usd": float(live["project"]["remaining_usd"]),
                  "project_cap_usd": float(live["project"]["cap_usd"]),
                  "formal_remaining_usd": float(live["formal"]["remaining_usd"]),
                  "engineering_remaining_usd": float(
                      live["engineering"]["remaining_usd"]),
                  "per_session_cap_usd": float(live["per_session_ceiling_usd"])}
        snap = programme_scenarios(limits, budget)
        out = REPO_ROOT / SCENARIO_RECORD
        out.write_text(json.dumps(snap, indent=1) + "\n")
        print(f"wrote {SCENARIO_RECORD}")
    body = json.dumps(budget, indent=1) + "\n"
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
