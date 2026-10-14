#!/usr/bin/env python3
"""The redesigned D2's minimal GPU validation: a PROPOSAL, priced, not granted.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/d2_gpu_validation_plan.py [--write]

Preregistration v2 (contract revision 6) changes two things D1's and stage L's
GPU evidence never exercised: DEPTH's reference (parent -> root store) and the
frozen teacher's full-vocabulary confidence weighting the KL and the
FFN/WIDTH/ATTENTION statistics. CPU tests pin the arithmetic on a toy; what
only a GPU on the real teacher can answer is listed below, step by step, with
what each step measures, why the CPU cannot, and its minutes -- every figure
derived from a committed measurement (stage L's booked session, R0's plan),
never typed.

AUTHORIZES NOTHING. A session needs its own maintainer grant; this record is
the proposal that grant would price.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.phase_d2.gpu_validation_plan/v1"
RECORD = "logs/stages/stage-1/phase_d2/plans/d2_gpu_validation_plan.json"
#: A step's planning bound over its stage-L-derived expectation, for a step
#: whose measured analogue ran on another objective or another parent.
STEP_BOUND_FACTOR = 1.5
#: The root store and the validity analysis are bounded at this many DEPTH
#: candidate evaluations per profile (one root forward, a full-vocabulary
#: entropy and a sketch over the mixture): the d2_budget figure, one owner.


def _stage_l() -> dict[str, Any]:
    from stages.d_series import d2_budget as BUD

    ev = BUD.calibration_evidence(REPO_ROOT)
    if not ev["usable"]:
        raise RuntimeError(f"stage L's evidence is not usable: {ev['why']}")
    doc = json.loads((REPO_ROOT / ev["evidence_session"]).read_text())
    lc = doc["L_reference_calibration"]
    depth = lc["timing_depth_one_round"]["arms"]["kl_only"]
    state = [r["seconds"] for r in lc["timing_state_eval"]["runs"]
             if r["arm"].startswith("kl_only")]
    return {"evidence_session": ev["evidence_session"],
            "depth_round_seconds": float(depth["operator_invocation_seconds"]),
            "round0_candidates": int(depth["candidate_subsets"]),
            "state_eval_seconds": max(state),
            "b_rebuild_seconds": float(lc["B"]["seconds"])}


def _root_operator_seconds(BUD) -> dict[str, float]:
    """D1's measured root invocations of the three statistics operators."""
    rows = BUD.d1_search_telemetry()
    ids = {r["state_id"] for r in rows}
    roots = {r["parent_id"] for r in rows} - ids
    out: dict[str, float] = {}
    for r in rows:
        if r["parent_id"] in roots and r["impl_id"] != BUD.DEPTH_IMPL:
            out[r["impl_id"]] = max(out.get(r["impl_id"], 0.0),
                                    float(r["operator_seconds"]))
    return dict(sorted(out.items()))


def build() -> dict[str, Any]:
    from stages.d_series import d2_budget as BUD
    from stages.d_series import objectives as O

    L = _stage_l()
    budget = BUD.build()
    calib = budget["steps"]["reference_calibration"]["components"]
    rate = float(budget["rate_usd_per_hour_basis"])
    f = STEP_BOUND_FACTOR
    per_cand = L["depth_round_seconds"] / L["round0_candidates"] / 60
    store = BUD.ROOT_STORE_CANDIDATE_EQUIVALENTS
    depth_round = L["depth_round_seconds"] / 60
    se = L["state_eval_seconds"] / 60
    stats_ops = _root_operator_seconds(BUD)
    #: Two arms (weighted, unweighted) per statistics operator, each at D1's
    #: measured root invocation on the same teacher and L40S.
    stats_minutes = 2 * sum(stats_ops.values()) / 60
    #: (id, what, why the CPU cannot, expected minutes, bound minutes, gate)
    steps = [
        ("V0_setup", "pod setup, environment, teacher fetch and load",
         "infrastructure", *calib["setup_and_overhead"], None),
        ("V1_root_stores_and_validity",
         "build the root reference store on BOTH calibration profiles at "
         "micro-batch 1 AND 3 (four builds), time them, compare the stores, "
         "and run teacher_confidence.validity_report on each profile",
         "the real teacher's full-vocabulary entropy distribution is the "
         "question; a toy has none",
         round(4 * per_cand * store["expected"], 2),
         round(4 * per_cand * store["bound"] * f, 2),
         "stores agree bsz1 vs bsz3 within R0's state tolerance (1e-5 abs on "
         "confidence); validity report written"),
        ("V2_b_rebuild_and_kl_relation",
         "rebuild B exactly as stage L did, forward it over both profiles and "
         "record its per-position root KL; validity_report WITH that KL "
         "(Spearman(c_T, KL), weighted vs unweighted shares by span)",
         "the c_T-KL relation needs a real compressed student; B is the "
         "incumbent, the scale every D-series number is read against",
         *calib["b_materialization"], "B's four identities verified"),
        ("V3_weighted_statistics",
         "FFN, WIDTH and ATTENTION at the root, each with full-scope "
         "(weighted) and narrow-scope (unweighted) statistics: time both, "
         "record the selection overlap (kept neurons, projection subspace "
         "overlap, kept heads)",
         "how far the teacher's confidence moves a REAL selection is the "
         "full-scope effect size; the toy's is meaningless",
         round(stats_minutes, 2), round(stats_minutes * f, 2),
         "both arms complete; overlap and timing recorded"),
        ("V4_depth_root_vs_parent",
         "one DEPTH round on the FFN child from V3 (a non-root parent) under "
         "the D2 objective at all_v1: root-reference table, round-0 parent-KL "
         "diagnostic, objective and diagnostic seconds, both cache modes",
         "the reference change acts only below the root; its cost and the "
         "root-vs-parent ranking difference exist only on real activations",
         round(depth_round, 2), round(depth_round * f, 2),
         "both references cached; the table and the parent ranking recorded"),
        ("V5_depth_scored_rows",
         "the same DEPTH round at scored_v1", "the D2 scored_v1 equivalence "
         "for THIS objective (stage L's verdict covers the superseded one)",
         round(depth_round, 2), round(depth_round * f, 2),
         "R0's tolerances: max relative 1e-5, same choice"),
        ("V6_state_evaluation",
         "B under the D2 objective: all_v1 vs scored_v1, bsz1 vs bsz3, and "
         "KL-only for the overhead",
         "the per-evaluation overhead and the scored-rows equivalence of THIS "
         "objective on the real suite",
         round(5 * se, 2), round(5 * se * f, 2),
         "R0's tolerance 1e-5 on every state.objective.* metric"),
        ("V7_teardown", "evidence fetch and teardown", "infrastructure",
         1.0, 2.0, None),
    ]
    rows = [{"id": i, "what": w, "why_gpu": y, "expected_minutes": e,
             "bound_minutes": b, "gate": g} for i, w, y, e, b, g in steps]
    exp = sum(r["expected_minutes"] for r in rows)
    hard = sum(r["bound_minutes"] for r in rows)
    expected = BUD._usd(exp, rate, 60)
    bound = BUD._usd(hard, rate, 60)
    return {
        "schema": SCHEMA,
        "status": "PROPOSAL -- not granted, not run",
        "objective": O.D2_OBJECTIVE.qualified_id,
        "objective_hash": O.D2_OBJECTIVE.objective_hash,
        "hardware": {"gpu": "1x NVIDIA L40S (48 GB)", "why": (
            "the GPU stage L and R0 ran on: their timings are this plan's "
            "basis, and the 4B teacher plus a child and the stores fit"),
            "rate_usd_per_hour_basis": rate, "container_disk_gb": 60},
        "basis": {"stage_l": L, "step_bound_factor": f,
                  "d1_root_operator_seconds_max": stats_ops,
                  "root_store_candidate_equivalents": store,
                  "setup_and_b_rebuild": ("d2_budget's reference-calibration "
                                          "components (R0's measured setup, "
                                          "stage L's measured B rebuild)")},
        "steps": rows,
        "totals": {"expected": expected, "bound": bound,
                   "teardown_reserve_usd": BUD.TEARDOWN_RESERVE_USD,
                   "proposed_session_cap_usd": bound["all_in_usd"],
                   "proposed_grant_usd": round(bound["all_in_usd"]
                                               + BUD.TEARDOWN_RESERVE_USD, 2)},
        "deliverables": [
            "the root store's identities, content digests and build seconds "
            "(both profiles, bsz1 and bsz3)",
            "the teacher-confidence validity report per profile, with B's KL",
            "full- vs narrow-scope selection overlap and timing per operator",
            "DEPTH root-vs-parent ranking on a non-root parent, with its cost",
            "scored_v1 equivalence verdicts for THIS objective (DEPTH, state)",
            ("measured replacements for every PROVISIONAL term of the D2 "
             "search price (d2_budget.teacher_confidence_overhead)")],
        "does_not": ["run a D2 search", "train", "score behaviour",
                     "change any definition: results are descriptive"],
        "evidence_contract": (
            "booked in logs/stages/stage-1/families/d_series/validations/"
            "d2-gpu-validation/v1/campaign.json with the evidence file's "
            "sha256; d2_budget.d2_validation_evidence reads it and nothing "
            "else, and the D2 authorization's scored_v1 gate reads that"),
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        print(f"wrote {RECORD}")
        return 0
    same = path.is_file() and path.read_text() == body
    print(f"{RECORD}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
