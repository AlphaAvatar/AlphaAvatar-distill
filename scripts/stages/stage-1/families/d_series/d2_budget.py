#!/usr/bin/env python3
"""The executable budget for the steps before and through D2. Derived, not typed.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/families/d_series/d2_budget.py [--write]

Every figure comes from a committed owner:

* R0 -- its registered plan;
* the reference calibration session -- R0's measured setup components, D1's
  bounding arm-materialization minutes and a conservative compute allowance;
* the D2 search -- D1's PRICED search (the same space and support), plus the
  redesigned D2's added work (contract revision 6), every term of which is
  PROVISIONAL until a GPU validation measures it: stage L measured the
  SUPERSEDED student-uncertainty objective, whose per-position work the new D2
  does not do, so that timing prices D3 (which shares it) and never D2;
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
R0_CAMPAIGN = ("logs/stages/stage-1/families/d_series/validations/"
               "prediction-rows-r0/v1/campaign.json")
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
    dist = [float(r["operator_timing"]["distortion_seconds"])
            / int(r["operator_timing"]["candidate_subsets"]) for r in depth]
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
                  "loop_seconds_per_candidate_max": round(
                      max(l / c for l, c in zip(loop, cands)), 4),
                  #: The KL reduction per candidate, UNATTRIBUTED (no sync):
                  #: it carries the forward's async tail, so it over-states.
                  "distortion_seconds_per_candidate": {
                      "mean": round(sum(dist) / len(dist), 4),
                      "max": round(max(dist), 4)},
                  "profiles": len({r["profile"] for r in depth}),
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


#: The reference calibration's campaign: the ONE place a stage-L session is
#: booked, with its evidence file and that file's sha256. The price reads the
#: evidence through it and nowhere else -- no CLI switch, no field to edit.
CALIBRATION_CAMPAIGN = ("logs/stages/stage-1/families/d_series/validations/"
                        "reference-calibration/v1/campaign.json")
#: A single measured overhead is a measurement, not a bound: the planning
#: bound carries this variation margin on it. The watchdog cap is the
#: enforceable stop either way.
MEASURED_OVERHEAD_BOUND_FACTOR = 1.5


def calibration_evidence(repo_root: str | Path | None = None) -> dict[str, Any]:
    """The stage-L GPU evidence, VERIFIED, or the reason it cannot be used.

    Usable only when a subrun is booked in the calibration campaign with its
    evidence path and sha256, the bytes hash to it, the session COMPLETED stage
    L, and it ran the objective, rules and incumbent this programme registers.
    Anything else leaves the price on D1's records, labelled why.
    """
    from stages.d_series import incumbent as INC
    from stages.d_series import objectives as O

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    out: dict[str, Any] = {"usable": False, "cache_fix_validated": False,
                           "campaign": CALIBRATION_CAMPAIGN}
    try:
        campaign = json.loads((root / CALIBRATION_CAMPAIGN).read_text())
    except (OSError, ValueError) as exc:
        return {**out, "why": f"campaign unreadable: {exc}"}
    booked = [r for r in campaign.get("subruns") or []
              if r.get("evidence_session") and r.get("evidence_session_sha256")]
    if not booked:
        return {**out, "why": "no stage-L session is booked with evidence"}
    row = booked[-1]
    path = root / row["evidence_session"]
    out.update({"subrun_id": row.get("subrun_id"),
                "evidence_session": row["evidence_session"],
                "evidence_session_sha256": row["evidence_session_sha256"]})
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {**out, "why": f"evidence unreadable: {exc}"}
    import hashlib
    if hashlib.sha256(raw).hexdigest() != row["evidence_session_sha256"]:
        return {**out, "why": "the evidence does not hash to what the campaign "
                              "booked"}
    doc = json.loads(raw)
    lc = doc.get("L_reference_calibration") or {}
    prem = doc.get("L_premises") or {}
    inc = INC.standing_incumbent()
    checks = {
        "session_complete": doc.get("status") == "COMPLETE",
        "stage_l_ran": bool(lc.get("calibration")),
        #: Stage L measured the 2026-10-13 D2, SUPERSEDED on 2026-10-14. The
        #: check binds the evidence to what it measured; its objective timing
        #: then prices D3 (the same student Top-K entropy work), never the
        #: redesigned D2 -- see `teacher_confidence_overhead`.
        "stage_l_objective_is_the_superseded_d2": prem.get("d2_objective_hash")
        == O.D2_STUDENT_UNCERTAINTY_SUPERSEDED.objective_hash,
        "lambda_rule": prem.get("lambda_rule_hash") == O.lambda_rule()["rule_hash"],
        "threshold_rule": prem.get("threshold_rule_hash")
        == O.diagnostic_threshold_rule()["rule_hash"],
        "incumbent": ((lc.get("B") or {}).get("identity") or {}).get(
            "artifact_digest") == inc["artifact_digest"],
    }
    depth = lc.get("timing_depth_one_round") or {}
    state = lc.get("timing_state_eval") or {}
    usable = all(checks.values())
    #: Round 0's candidate count at the root, as measured: the KL-only arm's
    #: one-round invocation. The redesigned D2's parent-KL diagnostic runs on
    #: exactly the round-0 candidates of every DEPTH expansion.
    round0 = (((depth.get("arms") or {}).get("kl_only") or {})
              .get("candidate_subsets"))
    return {**out, "checks": checks, "usable": usable,
            "cache_fix_validated": bool(usable and depth.get(
                "cache_fix_validated") is True),
            "depth": {k: depth.get(k) for k in (
                "overhead_seconds_per_candidate", "same_cache_mode",
                "cache_fix_validated", "skipped", "failed")},
            "round0_candidates_measured": round0,
            "objective_measured": "the superseded D2 (student Top-K entropy)",
            "state_eval": {k: state.get(k) for k in (
                "overhead_seconds_per_evaluation", "skipped", "failed")},
            "d2_scored_rows_equivalence": (lc.get("d2_scored_rows_equivalence")
                                           or {}).get("verdict"),
            "rescoring": ((lc.get("calibration") or {}).get(
                "rescoring_cost_measured") or {}),
            "why": None if usable else "an identity or completion check failed"}


#: The REDESIGNED D2's own GPU validation (a proposal until granted). Its
#: campaign books the session the way the calibration campaign does; the
#: scored-rows verdict and the measured overheads of THIS objective are read
#: from it and nowhere else.
D2_VALIDATION_CAMPAIGN = ("logs/stages/stage-1/families/d_series/validations/"
                          "d2-gpu-validation/v1/campaign.json")
D2_VALIDATION_PLAN = ("logs/stages/stage-1/families/d_series/validations/"
                      "d2-gpu-validation/v1/plan.json")


def d2_validation_evidence(repo_root: str | Path | None = None
                           ) -> dict[str, Any]:
    """The D2 GPU validation's ADOPTABLE evidence, or why there is none.

    Usable only when a session is booked with its evidence sha256, the bytes
    hash to it, and `d2_gpu_validation.adopt` accepts the record: COMPLETE,
    the objective, contract, preregistration and plan identities of this
    tree, verdicts recomputed from its measurements equal to the recorded
    ones, and validation PASS. A failed or incomplete run stays booked and
    billed and is never adopted. Stage L's verdicts belong to the superseded
    objective and never count.
    """
    import hashlib

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    out: dict[str, Any] = {"usable": False, "campaign": D2_VALIDATION_CAMPAIGN,
                           "scored_rows_equivalence": None}
    try:
        campaign = json.loads((root / D2_VALIDATION_CAMPAIGN).read_text())
    except (OSError, ValueError):
        return {**out, "why": "the D2 GPU validation has not run (no campaign)"}
    booked = [r for r in campaign.get("subruns") or []
              if r.get("evidence_session") and r.get("evidence_session_sha256")]
    if not booked:
        return {**out, "why": "no D2 GPU validation session is booked"}
    row = booked[-1]
    try:
        raw = (root / row["evidence_session"]).read_bytes()
    except OSError as exc:
        return {**out, "why": f"evidence unreadable: {exc}"}
    if hashlib.sha256(raw).hexdigest() != row["evidence_session_sha256"]:
        return {**out, "why": "the evidence does not hash to what was booked"}
    doc = json.loads(raw)
    #: ADOPTION is stage V's own rule (`d2_gpu_validation.adopt`): verdicts
    #: recomputed from the measurements, identities against the committed
    #: records of THIS tree, the run COMPLETE, validation PASS. The
    #: scored-rows verdict exists only for an adoptable run.
    from stages.phase_d2 import d2_gpu_validation as V

    verdict = V.adopt(doc, REPO_ROOT)
    return {**out, **verdict,
            "evidence_session": row["evidence_session"],
            "evidence_session_sha256": row["evidence_session_sha256"],
            "why": None if verdict["usable"] else "; ".join(verdict["reasons"])}


def objective_overhead(evidence: dict[str, Any] | None,
                       basis: dict[str, Any]) -> dict[str, Any]:
    """Seconds a STUDENT-CONFIDENCE objective adds per DEPTH candidate and
    per evaluation: the student's Top-K entropy at every scored position.

    Stage L measured it on the superseded D2; D3 does the same per-position
    work, so this prices D3. MEASURED when the verified stage-L evidence
    carries both timings on one cache mode -- expected = the measurement,
    bound = the measurement x MEASURED_OVERHEAD_BOUND_FACTOR, because one
    timing is not a bound. Otherwise PROVISIONAL, from the estimate, and
    labelled so.
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
    if not evidence or not evidence.get("usable"):
        return provisional
    d, e = evidence.get("depth") or {}, evidence.get("state_eval") or {}
    if not (d.get("overhead_seconds_per_candidate") is not None
            and d.get("same_cache_mode")
            and e.get("overhead_seconds_per_evaluation") is not None):
        return {**provisional, "measurement_unusable": {"depth": d,
                                                        "state_eval": e}}
    dc = max(0.0, float(d["overhead_seconds_per_candidate"]))
    se = max(0.0, float(e["overhead_seconds_per_evaluation"]))
    f = MEASURED_OVERHEAD_BOUND_FACTOR
    return {"source": (f"measured once: {evidence['evidence_session']} (stage "
                       "L); the bound adds a variation margin"),
            "depth_seconds_per_candidate": {"expected": dc,
                                            "bound": round(dc * f, 4)},
            "state_eval_seconds_per_evaluation": {"expected": se,
                                                  "bound": round(se * f, 4)},
            "bound_factor_on_one_measurement": f}


#: THE REDESIGNED D2's ADDED WORK, every term PROVISIONAL (not measured).
#: The new objective computes no student entropy; what it adds over D1 is
#: (1) the root reference store, built once per calibration profile -- one
#: root forward, a full-vocabulary entropy and a sketch over the mixture,
#: bounded at this many DEPTH candidate evaluations; (2) the round-0
#: parent-KL diagnostic, one extra sketch-KL reduction per round-0 candidate
#: of every DEPTH expansion, bounded by D1's per-candidate reduction time
#: (unattributed, so it carries the forward's async tail: conservative) and
#: expected at half of it; (3) c_T x KL and stored lookups in DEPTH and state
#: evaluation, and the weighted activation statistics in FFN/WIDTH/ATTENTION,
#: priced at the generic provisional +5% / +10% on the paths they touch.
ROOT_STORE_CANDIDATE_EQUIVALENTS = {"expected": 2.0, "bound": 5.0}
PARENT_KL_DIAGNOSTIC_FRACTION = {"expected": 0.5, "bound": 1.0}


def teacher_confidence_overhead(evidence: dict[str, Any] | None,
                                basis: dict[str, Any]) -> dict[str, Any]:
    """The redesigned D2's added seconds, labelled PROVISIONAL throughout.

    Never reads stage L's objective timing: that measured another objective.
    Reads from stage L only the round-0 candidate count it measured (falling
    back, conservatively, to every candidate of an expansion).
    """
    per_cand = basis["depth"]["loop_seconds_per_candidate_mean"]
    per_cand_max = basis["depth"]["loop_seconds_per_candidate_max"]
    dist = basis["depth"]["distortion_seconds_per_candidate"]
    se_mean = basis["state_evaluation"]["seconds_mean"]
    round0 = (evidence or {}).get("round0_candidates_measured")
    round0_src = ("stage L's measured root round" if round0 else
                  "UNKNOWN -- every candidate of an expansion, conservatively")
    round0 = int(round0 or max(basis["depth"]["candidates_per_expansion"]))
    profiles = int(basis["depth"]["profiles"])
    e, b = OBJECTIVE_OVERHEAD_EXPECTED - 1, OBJECTIVE_OVERHEAD_PROVISIONAL - 1
    return {
        "source": ("provisional -- the redesigned D2 is not measured; stage L "
                   "measured the superseded objective, which prices D3 only"),
        "depth_seconds_per_candidate": {
            "expected": round(per_cand * e, 4),
            "bound": round(per_cand * b, 4)},
        "state_eval_seconds_per_evaluation": {
            "expected": round(se_mean * e, 4),
            "bound": round(basis["state_evaluation"]["seconds_max"] * b, 4)},
        "depth_seconds_per_expansion": {
            "what": "the round-0 parent-KL diagnostic",
            "round0_candidates": round0, "round0_source": round0_src,
            "expected": round(round0 * dist["mean"]
                              * PARENT_KL_DIAGNOSTIC_FRACTION["expected"], 2),
            "bound": round(round0 * dist["max"]
                           * PARENT_KL_DIAGNOSTIC_FRACTION["bound"], 2)},
        "operator_cell_factor": {
            "what": ("the weighted activation statistics of the non-DEPTH "
                     "operators, on their whole end-to-end cells"),
            "expected": OBJECTIVE_OVERHEAD_EXPECTED,
            "bound": OBJECTIVE_OVERHEAD_PROVISIONAL},
        "one_time_minutes": {
            "what": "root reference store builds, one per calibration profile",
            "profiles": profiles,
            "expected": round(profiles * per_cand
                              * ROOT_STORE_CANDIDATE_EQUIVALENTS["expected"]
                              / 60, 2),
            "bound": round(profiles * per_cand_max
                           * ROOT_STORE_CANDIDATE_EQUIVALENTS["bound"] / 60, 2)},
    }


def priced_from_records(rate: float, disk_gb: int,
                        evidence: dict[str, Any] | None = None,
                        depth_cell: str = "records_max",
                        overhead: Any = None) -> dict[str, Any]:
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
    #: Which objective's added work: the redesigned D2's (default) or a
    #: student-confidence objective's (D3, `objective_overhead`).
    ov = (overhead or teacher_confidence_overhead)(evidence, basis)
    cands = max(basis["depth"]["candidates_per_expansion"])
    per_exp = ov.get("depth_seconds_per_expansion") or {}
    cell_factor = ov.get("operator_cell_factor") or {}
    one_time = ov.get("one_time_minutes") or {}
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
            extra = (cands * ov["depth_seconds_per_candidate"]["bound"]
                     + float(per_exp.get("bound", 0.0))) / 60
            cell.update({k: round(dep[k] + extra + add("bound"), 4)
                         for k in ("root_max", "deeper_max")})
            cell.update({"root_mean": cell["root_max"],
                         "deeper_mean": cell["deeper_max"]})
        else:
            f = float(cell_factor.get("bound", 1.0))
            cell.update({k: round(v * f + add("bound"), 4)
                         for k, v in cell.items()})
    model = dataclasses.replace(frozen, minutes=minutes, source=(
        f"{frozen.source} -- DEPTH rebuilt from D1's search telemetry "
        f"({D1_TELEMETRY}), plus the D2 objective's added seconds "
        f"({ov['source']})"))
    with W._measured_cost_model(model):
        priced = d1.search_cost(price_per_hour=rate, container_disk_gb=disk_gb)
    priced = {k: priced[k] for k in (
        "expansions_max", "structural_minutes_max",
        "session_overhead_minutes", "hard_ceiling_minutes", "hard_ceiling_usd")}
    #: A ONE-TIME term (the root stores) joins the hard ceiling once, outside
    #: the per-expansion cells, and its dollars are re-derived with it.
    if one_time:
        priced["one_time_minutes"] = float(one_time["bound"])
        priced["hard_ceiling_minutes"] = round(
            priced["hard_ceiling_minutes"] + float(one_time["bound"]), 2)
        priced["hard_ceiling_usd"] = _usd(priced["hard_ceiling_minutes"],
                                          rate, disk_gb)["all_in_usd"]
    non_depth_minutes = sum(
        n * v for impl, by in basis["counts"].items() if impl != DEPTH_IMPL
        for w, n in by.items()
        for v in [basis["cells_minutes"][impl][f"{w}_mean"]])
    exp_added = ((basis["depth"]["expansions"] * cands
                  * ov["depth_seconds_per_candidate"]["expected"]
                  + basis["depth"]["expansions"]
                  * float(per_exp.get("expected", 0.0))
                  + basis["state_evaluation"]["evaluations"]
                  * ov["state_eval_seconds_per_evaluation"]["expected"]) / 60
                 + non_depth_minutes
                 * (float(cell_factor.get("expected", 1.0)) - 1.0)
                 + float(one_time.get("expected", 0.0)))
    return {"basis": basis, "objective_overhead": ov, "depth_cell": depth_cell,
            "depth_cell_minutes": {k: minutes[DEPTH_IMPL][k]
                                   for k in ("root_max", "deeper_max")},
            "priced": priced,
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
    from stages.phase_d1.topk_adoption_driver import (
        L_CORE_STEPS, L_STEP_BOUND_S,
    )

    calib_exp = setup_exp + 21.3 + 1.0 + 1.0 + 1.0 + 1.0
    calib_hard = setup_hard + per_arm + 2.0 + 5.0 + 5.0 + 2.0
    #: Stage L's evidence steps. Expected from a3 (state evaluation 10.4-11.1 s)
    #: and a8 (36 of 260 candidates at p50 4.50 s + ~30 s outside the loop),
    #: a recomputed candidate at D1's 9.56 s; bounds = the driver's own
    #: per-step gate, so the plan and the gate cannot disagree.
    step_exp = {"state_eval": 0.7, "d2_rows_state_eval": 0.4,
                "depth_one_round": 7.0, "depth_recompute_equivalence": 6.2,
                "d2_rows_depth": 3.5}
    core_bound = sum(L_STEP_BOUND_S[k] for k in L_CORE_STEPS) / 60
    extended_bound = sum(v for k, v in L_STEP_BOUND_S.items()
                         if k not in L_CORE_STEPS) / 60
    calib = {"expected": _usd(calib_exp + sum(step_exp.values()), rate, 60),
             "hard": _usd(calib_hard, rate, 60),
             "hard_with_timing": _usd(calib_hard + core_bound, rate, 60),
             "hard_with_every_step": _usd(calib_hard + core_bound
                                          + extended_bound, rate, 60),
             "components": {
                 "setup_and_overhead": [setup_exp, setup_hard],
                 "b_materialization": [21.3, per_arm],
                 "teacher_fetch_and_load": [1.0, 2.0],
                 "teacher_and_b_forwards_147_items": [1.0, 5.0],
                 "rescoring_timing": [1.0, 5.0],
                 "teardown": [1.0, 2.0],
                 **{f"stage_L_{k}": [step_exp[k], round(v / 60, 2)]
                    for k, v in L_STEP_BOUND_S.items()}},
             "steps_are": (
                 "deadline-gated: the calibration record is written first; a "
                 "step starts only when the driver's deadline covers its whole "
                 "bound, and an extended step also the core bounds still to "
                 "come. The session cap prices the must-complete work plus the "
                 "two CORE steps at their bounds; the extended steps (the "
                 "cached-vs-recomputed and D2 scored-rows equivalences) run "
                 "inside the same cap whenever earlier work beats its bound "
                 "(expected: always)"),
             "core_steps": list(L_CORE_STEPS),
             "_basis": ("setup from R0's plan (measured a2/a8 components); B "
                        "rebuild measured 1279.8 s (expected) and D1's 22 min "
                        "per arm (bound); forwards from state-evaluation "
                        "timings (80 items in ~11 s at the frozen suite) with a "
                        "5x allowance")}
    #: THE GRANTED SHAPE (maintainer, post-48fcc638 review): the cap covers
    #: the must-complete work and both core steps at their bounds.
    calib["session_cap_proposed_usd"] = calib["hard_with_timing"]["all_in_usd"]
    calib["grant_proposed_usd"] = round(calib["session_cap_proposed_usd"]
                                        + TEARDOWN_RESERVE_USD + 0.04, 2)

    #: D2 search: priced from D1's own search records (see priced_from_records).
    disk_search = int(search["container_disk_gb"])
    evidence = calibration_evidence()
    records = priced_from_records(rate, disk_search, evidence)
    cached = priced_from_records(rate, disk_search, evidence,
                                 depth_cell="cached_validated")
    #: THE PRICE IN FORCE follows the verified evidence and nothing else.
    in_force = ("cache_fix_validated" if evidence["cache_fix_validated"]
                else "records_max")
    rec = cached if in_force == "cache_fix_validated" else records
    #: D3's search, for the programme view only: the same machinery with the
    #: STUDENT-confidence overhead stage L measured (on the superseded D2,
    #: whose per-position work D3 shares).
    d3_rec = priced_from_records(rate, disk_search, evidence,
                                 depth_cell=("cached_validated"
                                             if in_force == "cache_fix_validated"
                                             else "records_max"),
                                 overhead=objective_overhead)
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
        "pricing_in_force": in_force,
        "container_disk_gb": disk_search,
        "hard_records_max": _usd(records["priced"]["hard_ceiling_minutes"],
                                 rate, disk_search),
        "hard_if_cache_fix_validated": _usd(
            cached["priced"]["hard_ceiling_minutes"], rate, disk_search),
        "calibration_evidence": evidence,
        "cache_fix": {
            "what": ("`_available_memory_bytes` returns the allocator's "
                     "hoard (empty_cache) before reading the driver: D1's "
                     "search recomputed the reference in 15/16 DEPTH "
                     "expansions with 0.12 GiB free and 36 GiB reclaimable"),
            "validated_on_gpu": evidence["cache_fix_validated"],
            "_validated_on_gpu_is": ("DERIVED from the booked, hash-verified "
                                     "stage-L evidence above; never set by "
                                     "hand"),
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
        "assumptions_not_measured": [
            ("EVERY added term of the redesigned D2 (contract revision 6): "
             "c_T x KL in DEPTH and state evaluation at +5% (expected) / +10% "
             "(bound) of D1's paths; the round-0 parent-KL diagnostic at "
             f"{PARENT_KL_DIAGNOSTIC_FRACTION} of D1's per-candidate reduction; "
             "the weighted statistics at +5% / +10% of the FFN/WIDTH/ATTENTION "
             "cells; the root stores at "
             f"{ROOT_STORE_CANDIDATE_EQUIVALENTS} DEPTH-candidate evaluations "
             "per profile. Stage L's objective timing measured the SUPERSEDED "
             "objective and is not used here; the D2 GPU validation measures "
             "these before the search is priced from them"),
            "D2's trajectory has D1's size (92 expansions, 16 DEPTH): the "
            "beam schedule fixes the expansion count, not the operator mix",
            ("the reference cache fix (validated by stage L for D1's parent "
             "reference) also holds the redesigned DEPTH, whose parent "
             "reference is now the round-0 diagnostic and whose root "
             "reference is a resident store; the driver's cache guard still "
             "stops the search on the first recomputing expansion")],
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
        "_basis": ("expected = D1's measured 551.99 min + the redesigned "
                   "objective's expected added minutes over D1's trajectory; "
                   "bound = D1's search_cost on the records-rebuilt cost "
                   "model plus the one-time root stores"),
        "d3_search_for_programme_view": {
            "expected": _usd(d3_rec["expected_minutes"], rate, disk_search),
            "hard": _usd(d3_rec["priced"]["hard_ceiling_minutes"], rate,
                         disk_search),
            "objective_overhead": d3_rec["objective_overhead"],
            "_basis": ("D3 does the student Top-K entropy work stage L "
                       "measured on the superseded D2; priced with that "
                       "measurement and its one-measurement margin")}}

    #: D2 behaviour on the shared protocol: K candidates, no B/q2 probes.
    probe_bound = float(screening["minutes_per_probe_bounding"])
    overrun = float(screening["overrun_factor"])
    sess_over = float(screening["session_overhead_minutes"])
    disk_beh = float(screening["container_disk_gb"])

    #: THE ROLLOUT RESCORING: measured by stage L when its evidence verified
    #: (projected per probe from real-model per-token time), else the
    #: estimate. Measured, it JOINS the ceilings with the one-measurement
    #: margin; estimated, it stays outside them and is reported apart.
    resc = (evidence.get("rescoring") or {}) if evidence["usable"] else {}
    resc_measured = resc.get("projected_minutes_per_probe")

    def behaviour(n_probes: int, n_arms: int) -> dict[str, Any]:
        fixed = sess_over + n_arms * per_arm
        probes = n_probes * probe_bound
        if resc_measured is not None:
            r_exp = n_probes * float(resc_measured)
            r_hard = r_exp * MEASURED_OVERHEAD_BOUND_FACTOR * overrun
            return {"n_probes": n_probes, "n_arms_materialized": n_arms,
                    "expected": _usd(fixed + probes + r_exp, rate, disk_beh),
                    "hard": _usd(fixed + probes * overrun + r_hard, rate,
                                 disk_beh),
                    "rescoring": {"source": "measured once (stage L)",
                                  "minutes_per_probe": float(resc_measured),
                                  "in_ceilings": True,
                                  "hard": _usd(r_hard, rate, disk_beh)},
                    "rescoring_provisional_not_a_bound": _usd(0.0, rate,
                                                              disk_beh)}
        return {"n_probes": n_probes, "n_arms_materialized": n_arms,
                "expected": _usd(fixed + probes, rate, disk_beh),
                "hard": _usd(fixed + probes * overrun, rate, disk_beh),
                "rescoring": {"source": "estimate", "in_ceilings": False,
                              "minutes_per_probe":
                                  RESCORING_MINUTES_PER_PROBE_ESTIMATE},
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
                   "rescoring": (
                       "MEASURED by stage L and inside the behavioural "
                       "ceilings with the one-measurement margin"
                       if resc_measured is not None else
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
    #: RESERVES RELEASED AGAINST ACTUAL BILLS. The live limits are derived
    #: from recorded spend, so a CLOSED campaign's cost is already inside
    #: `project_remaining_usd`; subtracting its grant again would count it
    #: twice. An open campaign still holds its unspent grant.
    outstanding, eng_state = {}, {}
    for name, rel in (("r0", R0_CAMPAIGN), ("reference_calibration",
                                             CALIBRATION_CAMPAIGN)):
        camp = _read(rel)
        booked = round(sum(float(r.get("cost_usd") or 0)
                           for r in camp.get("subruns") or []), 4)
        closed = str(camp.get("status", "")).startswith("COMPLETE and CLOSED")
        outstanding[name] = 0.0 if closed else round(eng[name] - booked, 4)
        eng_state[name] = {"grant_usd": eng[name], "booked_usd": booked,
                           "closed": closed,
                           "outstanding_usd": outstanding[name]}
    eng_exp = sum(
        0.0 if eng_state[n]["closed"] else steps[n]["expected"]["all_in_usd"]
        for n in eng)
    #: THE REDESIGNED D2's GPU VALIDATION, a PROPOSAL until granted: its
    #: proposed grant is held as outstanding engineering, so no scenario
    #: spends money the validation needs. Read from its plan record.
    plan_path = REPO_ROOT / D2_VALIDATION_PLAN
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        grant = float(plan["totals"]["proposed_grant_usd"])
        eng["d2_gpu_validation"] = grant
        outstanding["d2_gpu_validation"] = grant
        eng_state["d2_gpu_validation"] = {
            "grant_usd": grant, "booked_usd": 0.0, "closed": False,
            "outstanding_usd": grant, "status": plan["status"],
            "owner": D2_VALIDATION_PLAN}
        eng_exp += float(plan["totals"]["expected"]["all_in_usd"])
    engineering = {"expected": round(eng_exp, 2),
                   "hard_with_reserves": round(sum(outstanding.values()), 2),
                   "campaigns": eng_state,
                   "_closed_campaigns_are": ("already inside the live limits "
                                             "through their recorded cost")}
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

        #: D3 is priced with ITS OWN search (the student-confidence overhead
        #: stage L measured), at the same DEPTH cell as D2's price under
        #: consideration, and D2's behavioural blocks (the same protocol).
        d3_search_hard = (search["d3_search_for_programme_view"]["hard"]
                          ["all_in_usd"] + (search_hard_usd
                                            - search["hard"]["all_in_usd"]))
        d3_hard = dict(hard, d2_search=d3_search_hard)
        d3_block = {"expected": round(
                        search["d3_search_for_programme_view"]["expected"][
                            "all_in_usd"]
                        + sum(steps[n]["expected"]["all_in_usd"]
                              for n in ("d2_screening", "d2_confirmation")), 2),
                    "hard_with_reserves": round(sum(
                        v + TEARDOWN_RESERVE_USD for v in d3_hard.values()), 2),
                    "rescoring_estimate": block["rescoring_estimate"],
                    "search_hard_usd": round(d3_search_hard, 2)}

        def scenario(blocks):
            remaining = limits["project_remaining_usd"]
            exp_ = round(engineering["expected"]
                         + sum(b["expected"] for _, b in blocks), 2)
            hard_ = round(engineering["hard_with_reserves"]
                          + sum(b["hard_with_reserves"] for _, b in blocks), 2)
            resc = round(sum(b["rescoring_estimate"] for _, b in blocks), 2)
            #: THREE DIFFERENT KINDS OF NUMBER, never summed under one name:
            #: (1) hard ceilings + teardown reserves -- bounds the launchers
            #: enforce; (2) the rescoring diagnostic -- an UNMEASURED estimate;
            #: (3) their sum -- a conservative planning view, not a ceiling.
            return {"blocks": ["engineering (R0 + reference calibration + the "
                       "proposed D2 GPU validation)"]
                    + [n for n, _ in blocks],
                    "expected_usd": exp_,
                    "hard_with_reserves_usd": hard_,
                    "unverified_diagnostic_estimate_usd": resc,
                    "conservative_planning_usd": round(hard_ + resc, 2),
                    "project_remaining_after_hard_usd": round(
                        remaining - hard_, 2),
                    "project_remaining_after_conservative_usd": round(
                        remaining - hard_ - resc, 2),
                    "fits_project_at_hard": hard_ <= remaining,
                    "fits_project_conservative": hard_ + resc <= remaining,
                    "fits_project_at_expected": exp_ <= remaining}

        needed = (d3_block["hard_with_reserves"]
                  + fresh_block["hard_with_reserves"])
        left = (limits["project_remaining_usd"]
                - engineering["hard_with_reserves"]
                - block["hard_with_reserves"])
        estimates = block["rescoring_estimate"] + d3_block["rescoring_estimate"]
        return {
            "search_hard_usd": search_hard_usd,
            "search_fits_per_session_cap": (search_hard_usd
                                            <= limits["per_session_cap_usd"]),
            "d2_block": block, "d3_block": d3_block,
            "scenarios": {
                "A_d2": scenario([("d2", block)]),
                "B_d2_d3": scenario([("d2", block), ("d3", d3_block)]),
                "C_d2_d3_fresh_confirmation": scenario(
                    [("d2", block), ("d3", d3_block),
                     ("fresh_confirmation", fresh_block)])},
            "reservation_rule": {
                "left_after_d2_at_hard_usd": round(left, 2),
                "needed_for_d3_and_fresh_at_hard_usd": round(needed, 2),
                "satisfied": left >= needed,
                "shortfall_usd": round(max(0.0, needed - left), 2),
                "conservative_with_unverified_estimates": {
                    "estimates_usd": round(estimates, 2),
                    "satisfied": left >= needed + estimates,
                    "shortfall_usd": round(max(0.0, needed + estimates - left),
                                           2)}}}

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
        "_engineering_grants_are": ("R0 and the reference calibration "
                                    "(granted, closed) and the D2 GPU "
                                    "validation (PROPOSED, held)"),
        "engineering_fits_allowance": (engineering["hard_with_reserves"]
                                       <= limits["engineering_remaining_usd"]),
        "engineering": engineering,
        "fresh_confirmation": fresh_block,
        "reservation_rule": (
            "authorize D2's formal sessions only if the project remainder "
            "after engineering and D2 at hard ceilings still covers D3 and one "
            "fresh confirmation at theirs"),
        "search_pricing": {
            "records_max": under(search["hard_records_max"]["all_in_usd"]),
            "cache_fix_validated": under(
                search["hard_if_cache_fix_validated"]["all_in_usd"])},
        "search_pricing_in_force": search["pricing_in_force"],
        "levers_if_short": {
            "validate_the_reference_cache_fix": {
                "saves_hard_usd_per_search": round(
                    search["hard_records_max"]["all_in_usd"]
                    - search["hard_if_cache_fix_validated"]["all_in_usd"], 2),
                "how": ("the reference calibration's stage L DEPTH pair, "
                        "already in its plan, at no extra session")},
            "measure_the_objective_overhead": (
                "the D2 GPU validation (a proposal) replaces every "
                "provisional term of the redesigned D2's price -- the "
                "+5%/+10% on DEPTH, state evaluation and the statistics "
                "operators, the round-0 parent-KL diagnostic and the root "
                "stores -- with measurements; stage L's timing priced the "
                "superseded objective and now prices D3 only"),
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
    args = ap.parse_args(argv)
    #: No evidence switch on the command line: the price reads the stage-L
    #: session booked in the calibration campaign, verified by its sha256.
    budget = build()
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
