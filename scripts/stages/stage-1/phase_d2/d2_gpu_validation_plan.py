#!/usr/bin/env python3
"""The redesigned D2's GPU validation: plan, PROPOSED grant and campaign.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/d2_gpu_validation_plan.py [--write]

Writes, under ``validations/d2-gpu-validation/v1/``:

* ``plan.json`` -- REGISTERED BEFORE ANY SESSION: the steps, what each
  measures and why only a GPU can, the numeric acceptance protocol in its
  three classes (owned by `d2_gpu_validation.py`, copied here and checked equal
  by the stage's `$0` premises), the step bounds the driver gates on, and the
  minutes and dollars -- every figure derived from a committed measurement
  (stage L's booked session, R0's plan, D1's search telemetry), never typed;
* ``authorization.proposed.json`` -- the grant this plan would need. A
  PROPOSAL: the launcher reads ``authorization.json``, which does not exist
  until the maintainer grants, so nothing here can create a resource;
* ``campaign.json`` -- the cumulative ledger, created once and never
  overwritten (it books subruns).

Grant shape follows stage L's precedent: the session cap covers setup and
every CORE step at its bound; the one EXTENDED step (V5, which decides only
`scored_v1`) runs inside the same cap whenever earlier work beats its bound.
AUTHORIZES NOTHING.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.d_series.validation_plan/v1"
#: A step's planning bound over its derived expectation, for a step whose
#: measured analogue ran on another objective or another parent.
STEP_BOUND_FACTOR = 1.5
#: Python-side analysis per profile (validity report, drift, token decode):
#: CPU work over ~1e5 positions, unmeasured, allowed for explicitly.
ANALYSIS_MINUTES_PER_PROFILE = {"expected": 0.25, "bound": 1.0}
#: WIDTH's overlap diagnostic: two float64 normal-equation solves.
WIDTH_OVERLAP_MINUTES = {"expected": 0.25, "bound": 1.0}
BRANCH = "exec/d1-behavioural-readiness"
LAUNCHER = "scripts/stages/stage-1/phase_d1/topk_adoption_launch.sh"


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
    from stages.phase_d2 import d2_gpu_validation as V
    from stages.phase_d2 import d2_session as S

    L = _stage_l()
    budget = BUD.build()
    calib = budget["steps"]["reference_calibration"]["components"]
    rate = float(budget["rate_usd_per_hour_basis"])
    f = STEP_BOUND_FACTOR
    per_cand = L["depth_round_seconds"] / L["round0_candidates"]
    store = BUD.ROOT_STORE_CANDIDATE_EQUIVALENTS
    a = ANALYSIS_MINUTES_PER_PROFILE
    n_prof = 2
    basis = BUD.telemetry_basis(
        BUD.d1_search_telemetry(),
        float(BUD._read(f"{BUD.D1_SEARCH_RUN}/closeout/outcome.json")["cost"][
            "elapsed_minutes"]))
    diag = BUD.teacher_confidence_overhead(BUD.calibration_evidence(),
                                           basis)["depth_seconds_per_expansion"]
    stats_ops = _root_operator_seconds(BUD)
    stats_s = 2 * sum(stats_ops.values())
    depth_exp = (L["depth_round_seconds"] + diag["expected"]) / 60
    depth_bound = (L["depth_round_seconds"] + diag["bound"]) / 60 * f
    se = L["state_eval_seconds"] / 60
    m = lambda x: round(x, 2)                                    # noqa: E731
    steps = [
        ("V0_setup", "pod setup, environment, teacher fetch and load, the "
         "stage-V $0 premises on the pod", "infrastructure",
         *calib["setup_and_overhead"], "core", None),
        ("V1_root_stores_and_validity",
         "the root store of BOTH profiles at the formal bsz=3 and at bsz=1; "
         "arithmetic on identical teacher logits; the teacher-confidence "
         "validity report per profile",
         "the real teacher's full-vocabulary entropy is the question",
         m(2 * n_prof * per_cand * store["expected"] / 60 + n_prof * a["expected"]),
         m(2 * n_prof * per_cand * store["bound"] / 60 * f + n_prof * a["bound"]),
         "core", "arithmetic (entropy, confidence); integrity (identities)"),
        ("V2_b_rebuild_and_kl_relation",
         "rebuild B as stage L did; B's per-position root KL on both profiles; "
         "the c x KL reduction arithmetic on identical (KL, c); the validity "
         "report with B's KL",
         "the c_T-KL relation needs a real compressed student",
         m(calib["b_materialization"][0] + n_prof * per_cand / 60
           + n_prof * a["expected"]),
         m(calib["b_materialization"][1] + n_prof * per_cand / 60 * 2 * f
           + n_prof * a["bound"]),
         "core", "arithmetic (reduction); integrity (B identities)"),
        ("V6_state_evaluation",
         "B under KL-only and the D2 objective at bsz=3 all_v1, the D2 "
         "objective at bsz=3 scored_v1 and at bsz=1 all_v1",
         "the overhead, the raw-KL control and the scored_v1 equivalence of "
         "THIS objective on the real suite",
         m(5 * se), m(5 * se * f), "core",
         "arithmetic (raw-KL control); row equivalence (state)"),
        ("V3_weighted_statistics",
         "FFN, WIDTH and ATTENTION at the root, full scope (weighted) vs "
         "narrow scope (unweighted): timing and selection overlap",
         "how far the teacher's confidence moves a REAL selection",
         m(stats_s / 60 + WIDTH_OVERLAP_MINUTES["expected"]),
         m(stats_s / 60 * f + WIDTH_OVERLAP_MINUTES["bound"]), "core",
         "diagnostic only"),
        ("V4_depth_root_vs_parent",
         "one DEPTH round on V3's full-scope FFN child (a non-root parent) "
         "under the D2 objective at all_v1: parent KL, unweighted root KL and "
         "weighted root KL of the same candidates, their rankings, the "
         "confidence IQR/ESS, both cache modes",
         "the reference change acts only below the root",
         m(depth_exp), m(depth_bound), "core",
         "integrity (root reference, table consistency); rankings diagnostic"),
        ("V5_depth_scored_rows", "the same DEPTH round at scored_v1",
         "THIS objective's scored_v1 equivalence (stage L's covers the "
         "superseded one)", m(depth_exp), m(depth_bound), "extended",
         "row equivalence (DEPTH)"),
        ("V7_teardown", "evidence fetch and teardown", "infrastructure",
         1.0, 2.0, "core", None),
    ]
    rows = [{"id": i, "what": w, "why_gpu": y, "expected_minutes": e,
             "bound_minutes": b, "class": c, "gates": g}
            for i, w, y, e, b, c, g in steps]
    exp = sum(r["expected_minutes"] for r in rows)
    core = sum(r["bound_minutes"] for r in rows if r["class"] == "core")
    every = sum(r["bound_minutes"] for r in rows)
    cap = BUD._usd(core, rate, 60)
    #: The driver gates each step on its bound in SECONDS, rounded up.
    step_bounds = {r["id"]: int(math.ceil(r["bound_minutes"] * 60))
                   for r in rows if r["id"] in V.STEP_ORDER}
    session_cap = math.ceil(cap["all_in_usd"] * 100) / 100
    prereg = S.preregistration(REPO_ROOT)
    contract = json.loads((REPO_ROOT / O.CONTRACT).read_text())
    return {
        "schema": SCHEMA,
        "validation_id": "d2_gpu_validation",
        "version": 1,
        "status": ("PROPOSAL -- registered, not granted, not run "
                   "(authorization.proposed.json)"),
        "_contract": (
            "REGISTERED BEFORE ANY SESSION. Stage V of topk_adoption_driver.py "
            "(phase_d2/d2_gpu_validation.py) on ONE L40S, on the launcher, "
            "remote payload and campaign mechanism stage L and R0 ran on. "
            "Engineering validation of the redesigned D2 objective only: no D2 "
            "search, no recovery training, no behavioural probe. AUTHORIZES "
            "NOTHING by itself: authorization.json does, and it does not exist "
            "until the maintainer grants."),
        "objective": O.D2_OBJECTIVE.qualified_id,
        "objective_hash": O.D2_OBJECTIVE.objective_hash,
        "contract_hash": contract["contract_hash"],
        "preregistration_hash": prereg["preregistration_hash"],
        "formal_topology": {"micro_batch_size": 3,
                            "calibration_batch_packing": "length_sorted_v1",
                            "_frozen": ("D2's search runs only this; every "
                                        "gate below is measured under it")},
        "execution_order": list(V.STEP_ORDER),
        "steps": rows,
        "step_bounds_s": step_bounds,
        "acceptance": {
            "arithmetic": {
                "gate": True, "tolerances": dict(V.ARITHMETIC_TOLERANCES),
                "what": ("one set of inputs, two computations: float32 chunked "
                         "entropy vs float64 on the SAME teacher logits, chunk "
                         "512 vs 64, the confidence mapping, the c x KL item "
                         "mean vs float64 on the SAME (KL, c), the raw-KL "
                         "control of one state evaluation"),
                "justification": V.__doc__.split("1. **ARITHMETIC**")[1]
                .split("2. **")[0].strip()},
            "execution_drift": {
                "gate": False, "metrics": ["bsz1_vs_bsz3_root_store",
                                           "bsz1_vs_bsz3_state_evaluation"],
                "why_not_gated": (
                    "a different micro-batch is a different GEMM shape on BF16 "
                    "and so a different computation; R0's 1e-5 compares two "
                    "paths under ONE batch shape and does not transfer. D2's "
                    "topology is frozen at bsz=3, so this drift cannot reach a "
                    "D2 result: REPORTED, never gated")},
            "row_equivalence": {
                "gate": "decides scored_v1 eligibility only",
                "tolerances": dict(V.ROW_EQUIVALENCE_TOLERANCES),
                "what": ("all_v1 vs scored_v1 at the formal bsz=3: one DEPTH "
                         "round (V4 vs V5) and the state evaluation (V6)"),
                "source": "R0's registered tolerances, unchanged"},
            "integrity": {
                "gate": True,
                "checks": ["the teacher binding and B's four identities",
                           "each root store's identity equals its "
                           "declaration at bsz=1 and bsz=3",
                           "DEPTH reads the root reference",
                           "every DEPTH decision score equals the recorded "
                           "weighted objective, and every round-0 candidate "
                           "carries its parent KL"]},
            "diagnostics": {
                "gate": False,
                "execution": list(V.EXECUTION_DIAGNOSTICS),
                "science": list(V.SCIENCE_DIAGNOSTICS),
                "rule": ("recorded as measured -- including a narrow dynamic "
                         "range of c_T or an unchanged ranking -- and never a "
                         "reason to change a definition, a normalization, the "
                         "scope or a tolerance after the fact")},
            "verdicts": ("arithmetic and integrity PASS -> validation PASS; "
                         "scored_rows_equivalence PASS -> D2 may run "
                         "scored_v1, otherwise all_v1"),
        },
        "basis": {"stage_l": L, "step_bound_factor": f,
                  "d1_root_operator_seconds_max": stats_ops,
                  "parent_kl_diagnostic_seconds_per_round": {
                      k: diag[k] for k in ("expected", "bound",
                                           "round0_candidates")},
                  "root_store_candidate_equivalents": store,
                  "analysis_minutes_per_profile": a,
                  "setup_and_b_rebuild": ("d2_budget's reference-calibration "
                                          "components (R0's measured setup, "
                                          "stage L's measured B rebuild)")},
        "hardware": {"gpu": "1x NVIDIA L40S (48 GB)", "why": (
            "the GPU stage L and R0 ran on: their timings are this plan's "
            "basis, and the 4B teacher plus B, one child and the stores fit"),
            "rate_usd_per_hour_basis": rate, "container_disk_gb": 60},
        "totals": {
            "expected": BUD._usd(exp, rate, 60),
            "core_bound": cap, "every_step_bound": BUD._usd(every, rate, 60),
            "teardown_reserve_usd": BUD.TEARDOWN_RESERVE_USD,
            "proposed_session_cap_usd": session_cap,
            "proposed_grant_usd": round(session_cap
                                        + BUD.TEARDOWN_RESERVE_USD, 2),
            "_grant_shape": ("stage L's: the cap covers setup and every core "
                             "step at its bound; the extended V5 runs inside "
                             "it whenever earlier work beats its bound")},
        "deliverables": [
            "root stores: identities, content digests, build seconds",
            "the teacher-confidence validity report per profile, with B's KL",
            "arithmetic, integrity and scored-rows verdicts",
            "full- vs narrow-scope selection overlap per operator",
            "DEPTH: parent KL, unweighted and weighted root KL of the same "
            "candidates on a non-root parent, with rankings",
            "measured replacements for the PROVISIONAL terms of the D2 search "
            "price"],
        "does_not": ["run a D2 search", "train", "score behaviour",
                     "change any definition: results are descriptive"],
        "evidence": {
            "on_the_pod": ("adoption.json (stage V under V_d2_validation), "
                           "flushed after every step and written on every "
                           "exit path; journal.jsonl"),
            "fetched_before_teardown": "JSON and logs only (the launcher)",
            "booking": ("scripts/shared/pod/book_engineering_subrun.py copies "
                        "the small evidence into runs/<run>/, books cost.json's "
                        "all-in cost and the evidence sha256 in campaign.json; "
                        "derive_budget.py then counts it, and "
                        "d2_budget.d2_validation_evidence reads it")},
        "authorizes": "nothing",
    }


def proposed_authorization(plan: dict[str, Any], plan_sha256: str
                           ) -> dict[str, Any]:
    from stages.phase_d2 import d2_gpu_validation as V

    t = plan["totals"]
    return {
        "schema": "aadistill.engineering_authorization/v1",
        "authorization_id": "d_series_d2_gpu_validation",
        "book": "ENGINEERING",
        "status": "PROPOSED -- NOT GRANTED",
        "all_in_ceiling_usd": t["proposed_grant_usd"],
        "session_cap_usd": t["proposed_session_cap_usd"],
        "_ceiling_composition": (
            f"setup and every core step at its bound "
            f"(${t['core_bound']['all_in_usd']}, {t['core_bound']['minutes']} "
            f"min at ${plan['hardware']['rate_usd_per_hour_basis']}/h incl. "
            f"disk), rounded up to the cent, + the "
            f"${t['teardown_reserve_usd']} teardown reserve"),
        "rate_basis_usd_per_hour": plan["hardware"]["rate_usd_per_hour_basis"],
        "rate_rule": ("re-quoted live at launch; the launcher derives the "
                      "session ceiling and timeout from the live rate and "
                      "refuses if the cap no longer buys the work"),
        "covers": ["stage V of topk_adoption_driver.py on one L40S pod: the "
                   "steps of plan.json, nothing else"],
        "does_not_cover": [
            "a D2 search, recovery training or a behavioural probe",
            "a second pod for a failed step without a recorded engineering "
            "cause", "any spend above the all-in ceiling, cumulative across "
            "subruns"],
        "plan": V.PLAN, "plan_sha256": plan_sha256,
        "campaign": V.CAMPAIGN,
        "command": (
            f"ADOPT_CAMPAIGN={V.CAMPAIGN} ADOPT_COMMIT=<this grant's commit> "
            f"ADOPT_BRANCH={BRANCH} ADOPT_STAGES=V ADOPT_PREDICTION_ROWS=all_v1 "
            f"SESSION_CAP={t['proposed_session_cap_usd']:.2f} "
            f"bash {LAUNCHER} d2val_<utc>"),
        "activation": (
            "the maintainer's grant is recorded by writing this record as "
            "authorization.json with status GRANTED, granted_utc and "
            "granted_by; until then engineering_campaign_budget.py cannot "
            "read a ceiling and the launcher creates nothing"),
        "on_ordinary_failure": ("repair and retry inside this ceiling (AGENTS.md "
                                "P12.1); a deterministic paid failure is never "
                                "retried unchanged"),
        "authorizes": "nothing -- a proposal",
    }


def campaign() -> dict[str, Any]:
    from stages.phase_d2 import d2_gpu_validation as V

    return {
        "schema": "aadistill.engineering_campaign/v1",
        "campaign_id": "d_series_d2_gpu_validation",
        "_contract": ("Cumulative cost of the D2 GPU validation across every "
                      "resource and subrun. A rerun does NOT reset it."),
        "status": ("PROPOSED -- not granted: authorization.json does not "
                   "exist, so engineering_campaign_budget.py refuses and the "
                   "launcher creates nothing"),
        "authorization": V.AUTHORIZATION,
        "package_id": "phase_c1.execution_package.2026-09-11",
        "_package_id": ("the GPU ENGINEERING book, as for R0 and the reference "
                        "calibration"),
        "plan": V.PLAN,
        "teardown_reserve_usd": 0.3,
        "inherited_spend": [],
        "subruns": [],
        "booked_usd": 0.0,
        "authorizes": "nothing by itself",
    }


#: Fields of campaign.json this builder owns; everything else is the ledger's.
CAMPAIGN_FIXED = ("schema", "campaign_id", "authorization", "package_id",
                  "plan", "teardown_reserve_usd")


def main(argv: Sequence[str] | None = None) -> int:
    from stages.phase_d2 import d2_gpu_validation as V

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    plan = build()
    plan_body = json.dumps(plan, indent=1) + "\n"
    sha = hashlib.sha256(plan_body.encode()).hexdigest()
    auth_body = json.dumps(proposed_authorization(plan, sha), indent=1) + "\n"
    camp_path = REPO_ROOT / V.CAMPAIGN
    bodies = {V.PLAN: plan_body, V.AUTHORIZATION_PROPOSED: auth_body}
    stale = 0
    for rel, body in bodies.items():
        path = REPO_ROOT / rel
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            print(f"wrote {rel}")
        else:
            same = path.is_file() and path.read_text() == body
            stale += not same
            print(f"{rel}: {'current' if same else 'STALE (run with --write)'}")
    want = campaign()
    if camp_path.is_file():
        have = json.loads(camp_path.read_text())
        drift = [k for k in CAMPAIGN_FIXED if have.get(k) != want[k]]
        print(f"{V.CAMPAIGN}: " + (f"fixed fields drifted {drift}" if drift
                                   else "ledger present; fixed fields current"))
        stale += bool(drift)
    elif args.write:
        camp_path.write_text(json.dumps(want, indent=1) + "\n")
        print(f"wrote {V.CAMPAIGN}")
    else:
        print(f"{V.CAMPAIGN}: MISSING (run with --write)")
        stale += 1
    return 0 if args.write or not stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
