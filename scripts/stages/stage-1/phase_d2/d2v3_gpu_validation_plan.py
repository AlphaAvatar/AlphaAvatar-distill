#!/usr/bin/env python3
"""D2 version 3's minimal GPU validation: plan, PROPOSED grant and campaign.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d2/d2v3_gpu_validation_plan.py [--write]

REVISION 2 (the stage module's ``VALIDATION_DIR``). Revision 1's records under
``v1/`` -- plan, grant, campaign and its FAIL evidence -- are frozen: this
builder no longer writes them, and a revision-1 session adopts nothing here.

Writes, under the revision's directory:

* ``plan.json`` -- REGISTERED BEFORE ANY SESSION: the steps of stage W
  (`d2v3_gpu_validation.py`), what each measures and why only a GPU can, the
  numeric acceptance protocol (owned by the stage module, copied here and
  checked equal by its `$0` premises), the step bounds the driver gates on,
  and the minutes and dollars;
* ``authorization.proposed.json`` -- the grant this plan would need. A
  PROPOSAL: the launcher reads ``authorization.json``, which does not exist
  until the maintainer grants, so nothing here can create a resource;
* ``campaign.json`` -- the cumulative ledger, created once and never
  overwritten (it books subruns).

**Every bound is derived, never typed.** A step that repeats Stage V's work
takes Stage V's MEASURED duration (its adopted evidence) times the plan's
factor; version 3's new work is priced by stated models on measured sizes --
the union terms from `d2_budget.union_kl_overhead`, the float32 head GEMM and
the float64 oracle by FLOP counts at stated rates. The sample is fixed here
(16 items, 3 candidates) and never shrunk to fit a bound.

Reuses Stage V's mechanisms (launcher, acquisition, device handoff, cache
guard, teardown, booking, adoption) and skips the B rebuild. AUTHORIZES
NOTHING.
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

SCHEMA = "aadistill.d2v3_gpu_validation_plan/v2"
REVISION = 2
#: Run ids of this revision: d2v3r2val_<utc>, never revision 1's prefix.
RUN_PREFIX = f"d2v3r{REVISION}val"
#: A step's planning bound over its derived expectation (Stage V's factor).
STEP_BOUND_FACTOR = 1.5
BRANCH = "exec/d1-behavioural-readiness"
#: THE COMMIT A GRANT RUNS: the pod checks out exactly this. Typed because a
#: record cannot contain the commit that carries it; the tests check that
#: every path the session executes or reads is identical between it and the
#: current tree (`EXECUTED_PATHS`, minus `EXECUTED_PATH_EXCEPTIONS`).
EXECUTABLE_COMMIT = "7ef64f3b92d0a243474563dbd151954d316a16fd"
EXECUTED_PATHS = (
    "src", "scripts", "configs",
    "logs/stages/stage-1/families/d_series/validations/d2v3-gpu-validation/v2/"
    "plan.json",
    "logs/stages/stage-1/families/d_series/plans/"
    "d2_d3_objective_contract_v1.json",
    "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v3.json",
    "logs/stages/stage-1/phase_c1/plans/teacher_binding.json",
)
#: Inside those, what may differ: this builder (it names the commit), tests,
#: and the DEV-BOX-ONLY tools that run before or after the pod -- never on it.
EXECUTED_PATH_EXCEPTIONS = (
    "scripts/stages/stage-1/phase_d2/d2v3_gpu_validation_plan.py",
    "scripts/stages/stage-1/phase_d2/d2val_acquire.sh",
    "scripts/shared/pod/book_engineering_subrun.py",
    #: Prices from booked evidence after the session; stage W never imports
    #: them.
    "scripts/stages/stage-1/families/d_series/d2_budget.py",
    "scripts/stages/stage-1/phase_d2/d2_authorization.py",
    "scripts/maintenance/",
    "/tests/",
)
LAUNCHER = "scripts/stages/stage-1/phase_d1/topk_adoption_launch.sh"
ACQUIRE = "scripts/stages/stage-1/phase_d2/d2val_acquire.sh"

#: The models for version 3's NEW work, stated so every factor is checkable.
COST_MODEL = {
    "float32_gemm_flops_per_second": {"expected": 40e12, "bound": 20e12},
    "_float32_gemm_is": ("the float32 head product [rows, H] x [H, V]: under "
                         "half / a quarter of the L40S's 91.6 TFLOPS FP32"),
    "float64_flops_per_second": {"expected": 1.0e12, "bound": 0.5e12},
    "_float64_is": ("the oracle's float64 elementwise work: the L40S runs "
                    "FP64 at 1/64 of FP32 (1.43 TFLOPS peak)"),
    "float64_flops_per_element": {"teacher": {"expected": 30, "bound": 60},
                                  "candidate": {"expected": 40, "bound": 80}},
    "_per_element_is": ("one exponential (~20-25 DP ops) plus the log-softmax, "
                        "the KL term and three masked sums per candidate; the "
                        "teacher's pass is shared by every candidate"),
    "forward_seconds_per_item_factor": {"expected": 1.5, "bound": 4.0},
    "_forward_is": ("Stage V's measured DEPTH forward per item at bsz=3, "
                    "scaled for a bsz=1 forward of one item"),
    "python_seconds_per_item": {"expected": 1.0, "bound": 3.0},
    "_python_is": ("per sampled item: sketches, quantiles, transfers and "
                   "bookkeeping outside the kernels -- unmeasured, allowed "
                   "for explicitly"),
    "analysis_minutes_per_profile": {"expected": 0.25, "bound": 1.0},
    "_analysis_is": ("Stage V's allowance for the CPU-side validity report "
                     "per profile"),
}


def _adopted() -> dict[str, Any]:
    from stages.d_series import d2_budget as BUD

    v = BUD.adopted_validation()
    if not v:
        raise RuntimeError("version 2's Stage V is not adopted; this plan's "
                           "bounds rest on its measured step durations")
    return v


def revision_block() -> dict[str, Any]:
    """What revision 2 changes against revision 1, and what it does not."""
    from stages.phase_d2 import d2v3_gpu_validation as W

    v1 = REPO_ROOT / W.SUPERSEDED_VALIDATION_DIR
    return {
        "number": REVISION,
        "decided_by": ("maintainer, post-Stage-W independent review and "
                       "corrective authorization, 2026-10-14"),
        "supersedes": {
            "validation_dir": W.SUPERSEDED_VALIDATION_DIR,
            "plan_sha256": hashlib.sha256(
                (v1 / "plan.json").read_bytes()).hexdigest(),
            "campaign": f"{W.SUPERSEDED_VALIDATION_DIR}/campaign.json",
            "run": "d2v3val_20261014_170824",
            "verdict": "validation FAIL (arithmetic: kl_bounds)",
            "status": ("kept as is -- evidence, plan, grant and campaign "
                       "unchanged; adopts nothing; its spend stays booked "
                       "in its own campaign")},
        "changed": [
            "the KL arithmetic gate: both partition KLs against float64 from "
            "one error budget; the float64 order; float32 crossings within "
            "the combined bound, per side, with per-position records "
            "(d2v3_gpu_validation.kl_arithmetic) -- replacing revision 1's "
            "4 eps |log Z_teacher| slack",
            "the teacher-confidence validity report states the definition "
            "and normalizer the store declares (schema v2); revision 1's "
            "printed text is corrected by an erratum, its numbers unchanged",
            "the W3 price gains the same-input attribution pass"],
        "unchanged": [
            "the objective, support, contract and preregistration v3 (DRAFT) "
            "identities", "the steps W1-W6, the sample and the formal "
            "topology", "every other registered tolerance",
            "the scientific decision rubric"],
    }


def build() -> dict[str, Any]:
    from aadistill.infrastructure.manifest import sha256_json
    from stages.d_series import d2_budget as BUD
    from stages.d_series import objectives as O
    from stages.phase_d2 import d2_session as S
    from stages.phase_d2 import d2_gpu_validation as V
    from stages.phase_d2 import d2v3_gpu_validation as W

    v = _adopted()
    rec = v["record"]
    budget = BUD.build()
    rate = float(budget["rate_usd_per_hour_basis"])
    stage_v_plan = json.loads((REPO_ROOT / BUD.D2_VALIDATION_PLAN).read_text())
    disk = int(stage_v_plan["hardware"]["container_disk_gb"])
    basis = BUD.telemetry_basis(
        BUD.d1_search_telemetry(),
        float(BUD._read(f"{BUD.D1_SEARCH_RUN}/closeout/outcome.json")["cost"][
            "elapsed_minutes"]))
    #: PROVISIONAL, always (w_validation=False): this is the REGISTERED plan,
    #: and a measured re-pricing after its own session must not move it.
    ov = BUD.union_kl_overhead(BUD.calibration_evidence(), basis, v,
                               w_validation=False)
    added = ov["union_terms"]["added"]
    geo = ov["union_terms"]["geometry"]
    f = STEP_BOUND_FACTOR
    m = COST_MODEL
    vocab, hidden = geo["vocab_size"], geo["hidden_size"]

    def wall(step: str) -> float:
        return float(rec[step]["wall_seconds"])

    t4 = rec["V4_depth_root_vs_parent"]["timing"]
    cands = int(t4["candidate_subsets"])
    per_cand = added["depth_seconds_per_candidate"]
    per_eval = added["state_eval_seconds_per_evaluation"]
    rows_all = geo["depth_rows_per_candidate"]["all_v1"]
    scored = geo["depth_rows_per_candidate"]["scored_v1"] / rows_all
    stores = rec["V1_root_stores_and_validity"]["profiles"]
    rows_per_item = (sum(int(p["decision"]["positions"])
                         for p in stores.values())
                     / sum(int(p["decision"]["items_total"])
                           for p in stores.values()))
    profile_items = max(int(p["decision"]["items_total"])
                        for p in stores.values())
    build_s = max(float(p["builds"]["bsz3_formal"]["seconds"])
                  for p in stores.values())
    build_rows = max(int(p["decision"]["positions"]) for p in stores.values())
    forward_per_item = (float(t4["ablated_seconds"])
                        / max(1, int(t4["ablated_items"])))
    ffn = rec["V3_weighted_statistics"]["operators"][
        "ffn.activation_importance_v0"]["arms"]
    ffn_s = (float(ffn["full_scope"]["seconds"])
             + float(ffn["narrow_scope"]["seconds"]))

    def gemm_s(rows: float, which: str) -> float:
        return rows * vocab * hidden * 2 / m["float32_gemm_flops_per_second"][
            which]

    def union_row_s(which: str) -> float:
        return per_cand[which] / rows_all

    # --- W3: the sample, priced from its parts --------------------------------
    n_items = 2 * W.NUMERICS_SAMPLE["items_per_profile"]
    n_cands = W.NUMERICS_SAMPLE["bypass_candidates"]
    sample_rows = n_items * rows_per_item

    def w3_seconds(which: str) -> dict[str, float]:
        fl = m["float64_flops_per_element"]
        f64 = (sample_rows * vocab
               * (fl["teacher"][which] + (n_cands + 1) * fl["candidate"][which])
               / m["float64_flops_per_second"][which])
        fwd = (n_items * (n_cands + 1) + profile_items) * forward_per_item * \
            m["forward_seconds_per_item_factor"][which]
        parts = {
            "float64_oracle": f64,
            "head_gemm": gemm_s(sample_rows, which),
            "forwards": fwd,
            "union_reductions": (sample_rows * n_cands
                                 + profile_items * rows_per_item)
            * union_row_s(which),
            "component_benchmark": 6 * 3 * union_row_s(which) * rows_per_item
            * 2,
            "python": n_items * m["python_seconds_per_item"][which],
            "validity_report": 60 * m["analysis_minutes_per_profile"][which]}
        parts["total"] = sum(parts.values())
        return parts

    w3e, w3b = w3_seconds("expected"), w3_seconds("bound")
    #: Revision 2's attribution pass: one more reduction per sampled
    #: candidate row on head-logit inputs, and one sketch per sampled row
    #: (priced at the store build's per-row rate, forward included).
    for which, parts in (("expected", w3e), ("bound", w3b)):
        parts["same_input_attribution"] = (
            sample_rows * n_cands * union_row_s(which)
            + sample_rows * build_s / build_rows)
        parts["total"] = sum(v_ for k_, v_ in parts.items() if k_ != "total")
    #: W6: the suite store, built at every version-3 evaluator's priming.
    suite_rows = float(geo["state_suite_rows"])
    suite_store = {w: build_s * suite_rows / build_rows
                   + gemm_s(suite_rows, w) for w in ("expected", "bound")}
    union_evals = 3
    m_ = lambda x: round(x / 60, 2)                              # noqa: E731
    v0 = next(r for r in stage_v_plan["steps"] if r["id"] == "V0_setup")
    one = added["one_time_minutes"]
    steps = [
        ("W0_setup", "pod setup, environment, teacher fetch and load, the $0 "
         "premises on the pod", "infrastructure",
         v0["expected_minutes"], v0["bound_minutes"], None),
        ("W1_union_stores_and_confidence",
         "the union root store of BOTH profiles at the formal bsz=3 and at "
         "bsz=1: identity against the declaration (support, definitions, the "
         "executed build), build seconds and peak memory; the K+1 confidence "
         "against float64 on the same head logits; the teacher-confidence "
         "validity report (distribution, IQR, ESS, spans, domains, tokens); "
         "bsz=1 vs 3 drift",
         "the REAL teacher's Top-200 + tail confidence is the question",
         m_(wall("V1_root_stores_and_validity") + 60 * one["expected"]),
         m_(wall("V1_root_stores_and_validity") * f * 2
            + 60 * one["bound"]),
         "arithmetic (confidence); integrity (store identity)"),
        ("W2_ffn_child_weighted",
         "FFN at the root under version 3, full scope (weighted by the K+1 "
         "confidence) and narrow scope: the parent of W3-W6, the weighted "
         "denominator check and the selection overlap",
         "a real compressed parent, and the new weights on a real selection",
         m_(ffn_s), m_(ffn_s * f),
         "integrity (denominator = weight sum)"),
        ("W3_union_numerics",
         f"on {n_items} items ({W.NUMERICS_SAMPLE['items_per_profile']} per "
         f"profile) and {n_cands} bypass candidates of W2's child: BOTH "
         "float32 partition KLs (teacher Top-K + tail, union) against a "
         "float64 oracle on the float32-head teacher distribution, each from "
         "the registered error budget; the float64 order KL_T <= KL_U <= "
         "KL_full; float32 crossings of it against the combined bound, per "
         "side, with per-position records; a same-input attribution pass; "
         "retrieval against the full head; union size, overlap and tails; "
         "bf16 ties and the tie spill; the float32 head against the native "
         "bf16 forward; component timings; the child's union KL over the "
         "first profile for the confidence's weighted contributions",
         f"real {vocab:,}-entry bf16 tails and ties on the real teacher",
         m_(w3e["total"]), m_(w3b["total"] * f),
         "arithmetic (both KLs vs float64; float64 order; crossings within "
         "the combined bound; retrieval; reduction)"),
        ("W4_depth_union_all_v1",
         "one DEPTH round on W2's child under version 3 at all_v1 with the "
         "SYNCHRONIZED phase split: the objective's seconds per candidate, "
         "store hits, the rankings by weighted union KL, unweighted union KL, "
         "teacher Top-K + tail KL and parent KL, union statistics",
         "the per-candidate price term is the question",
         m_(wall("V4_depth_root_vs_parent") + cands * per_cand["expected"]),
         m_(wall("V4_depth_root_vs_parent") * f
            + cands * per_cand["bound"]),
         "integrity (root store with retrieval, table consistency)"),
        ("W5_depth_union_scored_v1", "the same DEPTH round at scored_v1",
         "THIS objective's scored_v1 equivalence, which decides the price "
         "lever",
         m_(wall("V5_depth_scored_rows")
            + cands * per_cand["expected"] * scored),
         m_(wall("V5_depth_scored_rows") * f
            + cands * per_cand["bound"] * scored),
         "row equivalence (DEPTH)"),
        ("W6_state_evaluation",
         "W2's child under version 3 at bsz=3 all_v1, bsz=3 scored_v1 and "
         "bsz=1 all_v1, each with its suite store primed and timed and the "
         "teacher's forwards during evaluation counted; and KL-only on the "
         "historical reference_topk_tail_v1 support as a DIAGNOSTIC beside "
         "the union KL (never the raw-KL control)",
         "the per-evaluation price term and the raw-KL control on the suite",
         m_(wall("V6_state_evaluation")
            + union_evals * suite_store["expected"]
            + 4 * per_eval["expected"]),
         m_(wall("V6_state_evaluation") * f * 2
            + union_evals * suite_store["bound"] * f
            + 4 * per_eval["bound"]),
         "arithmetic (raw-KL control); integrity (no teacher forward per "
         "evaluation); row equivalence (state)"),
        ("W7_teardown", "evidence fetch and teardown", "infrastructure",
         1.0, 2.0, None),
    ]
    rows = [{"id": i, "what": w_, "why_gpu": y, "expected_minutes": e,
             "bound_minutes": b, "class": ("core" if i in W.STEP_ORDER
                                           else "infrastructure"), "gates": g}
            for i, w_, y, e, b, g in steps]
    exp = sum(r["expected_minutes"] for r in rows)
    bound = sum(r["bound_minutes"] for r in rows)
    cap = BUD._usd(bound, rate, disk)
    session_cap = math.ceil(cap["all_in_usd"] * 100) / 100
    step_bounds = {r["id"]: int(math.ceil(r["bound_minutes"] * 60))
                   for r in rows if r["id"] in W.STEP_ORDER}
    prereg = S.preregistration(REPO_ROOT)
    contract = json.loads((REPO_ROOT / O.CONTRACT).read_text())
    science = {
        "schema": SCHEMA,
        "validation_id": "d2v3_gpu_validation",
        "version": REVISION,
        "revision": revision_block(),
        "_contract": (
            "REGISTERED BEFORE ANY SESSION. Stage W of topk_adoption_driver.py "
            "(phase_d2/d2v3_gpu_validation.py) on ONE L40S, on the launcher, "
            "remote payload and campaign mechanism Stage V ran on. Engineering "
            "validation of D2 version 3's numeric definitions only: no D2 "
            "search, no recovery training, no behavioural probe, no B "
            "rebuild. It binds the DRAFT preregistration v3 by its exact "
            "hash, which is not an approval of it. AUTHORIZES NOTHING."),
        "objective": O.D2_OBJECTIVE.qualified_id,
        "objective_hash": O.D2_OBJECTIVE.objective_hash,
        "support": O.D2_SUPPORT.as_dict(),
        "contract_hash": contract["contract_hash"],
        "preregistration_hash": prereg["preregistration_hash"],
        "preregistration_status": prereg["status"],
        "formal_topology": {"micro_batch_size": 3,
                            "calibration_batch_packing": "length_sorted_v1"},
        "stage_v_reuse": {
            "campaign": BUD.D2_VALIDATION_CAMPAIGN,
            "run": "d2val_20261014_103301",
            "reused_mechanisms": [
                f"{LAUNCHER} (live quote, campaign-derived ceiling, watchdog, "
                "teardown on every exit path, evidence fetched before "
                "teardown)",
                f"{ACQUIRE} (one-pod acquisition loop, capacity refusals "
                "booked at $0; parametrized by validation and stage)",
                "the CUDA device handoff and the executed-commit binding",
                "the reference-cache guard under a deliberately hoarded pool",
                "book_engineering_subrun.py booking with teardown evidence",
                "adoption by verdicts RECOMPUTED from the measurements"],
            "reused_measurements": [
                "version 2's per-step durations, as this plan's bound basis",
                "the weighted-statistics mechanism for WIDTH and ATTENTION "
                "(unchanged code; version 3 changes the weight VALUES, and W2 "
                "re-checks FFN's denominator under them)"],
            "not_reused": [
                "any verdict on the union KL, the retrieval, the "
                "deterministic Top-K or the K+1 confidence",
                "version 2's scored_v1 equivalence (another objective)",
                "version 2's teacher-confidence distribution"]},
        "execution_order": list(W.STEP_ORDER),
        "steps": rows,
        "step_bounds_s": step_bounds,
        "numerics_sample": {**W.NUMERICS_SAMPLE,
                            "candidate_depth_fractions":
                                list(W.CANDIDATE_FRACTIONS),
                            "rows": round(sample_rows),
                            "rule": ("every len/n-th item of each profile, "
                                     "in the mixture's own order")},
        "cost_model": {"plan": COST_MODEL,
                       "union_terms": {k: added[k] for k in added},
                       "w3_seconds": {"expected": {k: round(x, 2) for k, x in
                                                   w3e.items()},
                                      "bound": {k: round(x, 2) for k, x in
                                                w3b.items()}},
                       "suite_store_seconds": {k: round(x, 2) for k, x in
                                               suite_store.items()},
                       "geometry": geo,
                       "stage_v_basis": {
                           "evidence": v["evidence_session"],
                           "wall_seconds": {s_: wall(s_) for s_ in (
                               "V1_root_stores_and_validity",
                               "V4_depth_root_vs_parent",
                               "V5_depth_scored_rows",
                               "V6_state_evaluation")},
                           "ffn_full_plus_narrow_seconds": ffn_s,
                           "forward_seconds_per_item": forward_per_item}},
        "deadline_protection": {
            "step_gate": ("a step starts only when the driver's deadline "
                          "covers its whole registered bound (step_bounds_s)"),
            "inside_steps": ("W3 checks the deadline before every sampled item "
                             "and every contribution item, W6 before every "
                             "evaluation; DEPTH checks it per candidate"),
            "on_failure": ("a failed or deadline-stopped step keeps what it "
                           "measured beside its traceback; adoption.json is "
                           "flushed after every step and written on every "
                           "exit path; the launcher fetches it before "
                           "teardown")},
        "acceptance": {
            "arithmetic": {
                "gate": True, "tolerances": dict(W.ARITHMETIC_TOLERANCES),
                "what": ("the float32 computation against float64: the "
                         "teacher Top-K + tail KL and the union KL, each per "
                         "position |d| <= abs + rel x KL64, against a float64 "
                         "oracle on the float32-head teacher distribution and "
                         "the same candidate logits; the float64 order "
                         "KL_T <= KL_U <= KL_full at the float64 slack; "
                         "float32 crossings of that order within the combined "
                         "bound (count beyond it: 0); retrieval against the "
                         "full head's log-softmax; the K+1 confidence against "
                         "float64 on the same sketch; the c x KL item mean; "
                         "the raw-KL control inside ONE version-3 "
                         "evaluation"),
                "kl_error_budget": dict(W.KL_ERROR_BUDGET),
                "recorded_per_position": (
                    "per side: crossing counts, the largest crossing with its "
                    "float64 values, beyond-bound counts; per partition: "
                    "error distributions and error/tolerance histograms over "
                    "every position; full records of every crossing and "
                    "every position above half a tolerance (capped, the cap "
                    "stated); a same-input attribution separating float32 "
                    "rounding from the store's teacher-input path; anomalies "
                    "= float64 violations + crossings beyond the bound + "
                    "finiteness mismatches"),
                "teacher_distribution": ("the float32 head distribution the "
                                         "root store defines -- never the "
                                         "native bf16 logits")},
            "integrity": {
                "gate": True, "tolerances": dict(W.INTEGRITY_TOLERANCES),
                "checks": ["the teacher binding",
                           "each union store's identity equals its "
                           "declaration, including the executed build",
                           "the full-scope FFN arm is weighted by this "
                           "session's union store and divides by the weight "
                           "sum; the narrow arm is unweighted",
                           "DEPTH reads the ROOT union store with retrieval; "
                           "every decision score equals the recorded "
                           "objective",
                           "every version-3 evaluation reads its union store "
                           "and forwards the teacher zero times"]},
            "row_equivalence": {
                "gate": "decides scored_v1 eligibility only",
                "tolerances": dict(W.ROW_EQUIVALENCE_TOLERANCES),
                "what": "W4 vs W5 (DEPTH) and W6 (state) at the formal bsz=3"},
            "not_gated": {
                "execution": list(W.EXECUTION_DIAGNOSTICS),
                "science": list(W.SCIENCE_DIAGNOSTICS),
                "rule": ("recorded as measured -- including a narrow K+1 "
                         "confidence range, an unchanged ranking or a small "
                         "union effect -- and never a reason to change K, the "
                         "mapping, the tail formula, the support or a "
                         "tolerance after the fact")},
            "raw_kl_control_rule": (
                "state.objective.raw_kl.* against state.teacher_kl.* of the "
                "SAME version-3 evaluation; a KL on another support (the "
                "historical reference_topk_tail_v1 diagnostic) is REFUSED as "
                "a control; require_objective_support is never bypassed to "
                "build one"),
            "verdicts": ("arithmetic and integrity PASS -> validation PASS; "
                         "scored_rows_equivalence PASS -> D2 may run "
                         "scored_v1, otherwise all_v1"),
            "adoption": ("d2v3_gpu_validation.adopt: COMPLETE, identities "
                         "equal to the committed records, verdicts recomputed "
                         "from the measurements equal to the recorded ones, "
                         "the executed commit the granted one, validation "
                         "PASS")},
        #: REGISTERED NOW, before any version-3 GPU number exists, so the
        #: recommendation cannot be fitted to the result. It decides nothing:
        #: the maintainer takes the scientific decision on preregistration v3.
        "scientific_decision_rubric": {
            "status": ("a RECOMMENDATION rubric for the maintainer's decision "
                       "on preregistration v3; never a code gate, never a "
                       "reason to change K, the mapping, the tail formula or "
                       "the support"),
            "reference_point": ("version 2 on the same teacher and mixtures "
                                "(Stage V): c_T IQR 0.042-0.055, ESS fraction "
                                ">= 0.994, weighted vs unweighted root-KL "
                                "DEPTH ranking rho 0.999 with the same choice, "
                                "weighted objective 96-97% of the raw KL"),
            "signals": {
                "weighting": ("W1: the K+1 c_T IQR and ESS fraction over "
                              "scored positions per profile; W3: the weighted "
                              "share against the unweighted share by span, "
                              "domain and critical token; W4: weighted vs "
                              "unweighted union-KL ranking (rho, same choice, "
                              "top-3)"),
                "support": ("W3: union / full and teacher-Top-K / full KL "
                            "ratios, union size and overlap; W4: union vs "
                            "teacher-Top-K ranking (rho, same choice)"),
                "reference": ("W4: parent vs root rankings (version 2 already "
                              "showed rho 0.816 and a different choice)")},
            "GO": ("validation PASS, and at least one of: the weighted and "
                   "unweighted union-KL DEPTH choices differ; their ranking "
                   "rho < 0.95; the K+1 c_T IQR at least twice version 2's "
                   "(>= 0.11) on both profiles. Then version 3 tests more "
                   "than the root reference, and the full search is "
                   "recommended for the maintainer's approval"),
            "HOLD": ("validation PASS but the weighting is as inert as "
                     "version 2's: IQR <= 0.06 on both profiles, ESS fraction "
                     ">= 0.99, weighted vs unweighted rho >= 0.99 with the "
                     "same choice. Then a version-3 search would mainly test "
                     "the root reference and the union support; the "
                     "recommendation is an explicit maintainer decision "
                     "between running it with that claim boundary and "
                     "stopping D2 -- NOT a further redefinition of the "
                     "confidence"),
            "between": ("any other PASS: reported as measured with both "
                        "readings; the maintainer decides"),
            "NO-GO": ("validation FAIL on arithmetic or integrity that a "
                      "root-caused engineering repair cannot clear: version 3 "
                      "is not measurable as defined"),
            "never": ["tuning K, the mapping, a temperature, a mask or the "
                      "tail formula on these numbers",
                      "citing the stand-in checkpoint's confidence range as "
                      "the teacher's"]},
        "hardware": {"gpu": stage_v_plan["hardware"]["gpu"],
                     "rate_usd_per_hour_basis": rate,
                     "container_disk_gb": disk,
                     "peak_memory_estimate_bytes": int(
                         rec["V4_depth_root_vs_parent"]["peak_memory_bytes"]
                         + budget["steps"]["d2_search"]["memory"][
                             "store_bytes_total"]
                         + sum(budget["steps"]["d2_search"]["memory"][
                             "transient_bytes_per_block"].values())),
                     "_peak_is": ("Stage V's measured V4 peak plus every "
                                  "union store and one block's transients; "
                                  "the float64 oracle holds at most ~19 "
                                  "[64, V] float64 blocks (~1.5 GB)")},
        "deliverables": [
            "an adoption record with recomputed verdicts (arithmetic, "
            "integrity, row equivalence)",
            "the REAL teacher's K+1 confidence distribution, IQR, ESS and "
            "weighted contributions by span, domain and critical token",
            "MEASURED union terms per DEPTH candidate and per state "
            "evaluation, replacing the provisional ones in d2_budget",
            "the union-vs-full ratios, union sizes and tails on real teacher "
            "logits; weighted-vs-unweighted rankings",
            "the float32-head vs native-bf16 protocol difference"],
        "does_not": ["run a D2 search, recovery training or a behavioural "
                     "probe", "rebuild B", "touch D1, D3 or their records",
                     "re-judge, rewrite or adopt revision 1's evidence",
                     "repeat Stage V's version-2 measurements",
                     "change K, the mapping, the tail formula or a tolerance "
                     "on its results"],
        "evidence": {
            "on_the_pod": ("adoption.json (stage W under W_d2v3_validation), "
                           "flushed after every step and written on every "
                           "exit path; journal.jsonl"),
            "fetched_before_teardown": "JSON and logs only (the launcher)",
            "booking": ("book_engineering_subrun.py into this campaign, "
                        "append-only, teardown evidence hashed; "
                        "d2_budget.d2v3_validation_evidence adopts it or "
                        "not")},
    }
    return {
        **science,
        "plan_hash": sha256_json(science),
        "status": "PROPOSAL -- registered, not granted, not run",
        "totals": {
            "expected": BUD._usd(exp, rate, disk),
            "bound": cap,
            "teardown_reserve_usd": BUD.TEARDOWN_RESERVE_USD,
            "proposed_session_cap_usd": session_cap,
            "proposed_grant_usd": round(session_cap
                                        + BUD.TEARDOWN_RESERVE_USD, 2),
            "_grant_shape": ("the cap covers setup and every step at its "
                             "bound, rounded up to the cent; the grant adds "
                             "the teardown reserve")},
        "authorizes": "nothing",
    }


def proposed_authorization(plan: dict[str, Any], plan_sha256: str
                           ) -> dict[str, Any]:
    from stages.phase_d2 import d2v3_gpu_validation as W

    t = plan["totals"]
    cap = t["proposed_session_cap_usd"]
    return {
        "schema": "aadistill.engineering_authorization/v1",
        "authorization_id": f"d_series_d2v3_gpu_validation_r{REVISION}",
        "book": "ENGINEERING",
        "status": "PROPOSED -- NOT GRANTED",
        "all_in_ceiling_usd": t["proposed_grant_usd"],
        "session_cap_usd": cap,
        "_ceiling_composition": (
            f"setup and every step at its bound (${t['bound']['all_in_usd']}, "
            f"{t['bound']['minutes']} min at "
            f"${plan['hardware']['rate_usd_per_hour_basis']}/h incl. disk), "
            f"rounded up to the cent, + the ${t['teardown_reserve_usd']} "
            "teardown reserve"),
        "rate_basis_usd_per_hour": plan["hardware"]["rate_usd_per_hour_basis"],
        "rate_rule": ("re-quoted live at launch; the launcher derives the "
                      "session ceiling and timeout from the live rate and "
                      "refuses if the cap no longer buys the work"),
        "covers": ["stage W (revision 2) of topk_adoption_driver.py on one "
                   "L40S pod: the steps of plan.json, nothing else"],
        "relation_to_revision_1": (
            "a NEW grant with its own campaign; revision 1's $0.2538 stays "
            "booked in its closed campaign and inside the project's spend, "
            "and its unspent remainder is not carried over"),
        "does_not_cover": [
            "a D2 search, recovery training or a behavioural probe",
            "a second pod for a failed step without a recorded engineering "
            "cause", "any spend above the all-in ceiling, cumulative across "
            "subruns", "approval of preregistration v3"],
        "plan": W.PLAN, "plan_sha256": plan_sha256,
        "campaign": W.CAMPAIGN,
        "executable_commit": EXECUTABLE_COMMIT,
        "_executable_commit_is": (
            "the commit the pod checks out; every executed path and record is "
            "identical between it and the commit carrying this proposal "
            "(tested: test_d2v3_gpu_validation.py)"),
        "branch": BRANCH,
        "command": (
            f"ACQ_VALIDATION_DIR={W.VALIDATION_DIR} ACQ_STAGES=W "
            f"ACQ_RUN_PREFIX={RUN_PREFIX} bash {ACQUIRE}"),
        "launcher_command": (
            f"ADOPT_CAMPAIGN={W.CAMPAIGN} ADOPT_COMMIT={EXECUTABLE_COMMIT} "
            f"ADOPT_BRANCH={BRANCH} ADOPT_STAGES=W ADOPT_PREDICTION_ROWS=all_v1 "
            f"SESSION_CAP={cap:.2f} bash {LAUNCHER} {RUN_PREFIX}_<utc>"),
        "activation": (
            "the maintainer's grant is recorded by writing this record as "
            "authorization.json with status GRANTED, granted_utc, granted_by "
            "and the live pre-launch checks; until then "
            "engineering_campaign_budget.py cannot read a ceiling and the "
            "launcher creates nothing"),
        "on_ordinary_failure": ("repair and retry inside this ceiling (AGENTS.md "
                                "P12.1); a deterministic paid failure is never "
                                "retried unchanged"),
        "authorizes": "nothing -- a proposal",
    }


def campaign() -> dict[str, Any]:
    from stages.phase_d2 import d2v3_gpu_validation as W

    return {
        "schema": "aadistill.engineering_campaign/v1",
        "campaign_id": f"d_series_d2v3_gpu_validation_r{REVISION}",
        "_contract": ("Cumulative cost of D2 version 3's GPU validation, "
                      f"revision {REVISION}, across every resource and subrun. "
                      "A rerun does NOT reset it; revision 1's spend is in "
                      "its own closed campaign."),
        "status": ("PROPOSED -- not granted: authorization.json does not "
                   "exist, so engineering_campaign_budget.py refuses and the "
                   "launcher creates nothing"),
        "authorization": W.AUTHORIZATION,
        "package_id": "phase_c1.execution_package.2026-09-11",
        "_package_id": ("the GPU ENGINEERING book, as for R0, the reference "
                        "calibration and Stage V"),
        "plan": W.PLAN,
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
    from stages.phase_d2 import d2v3_gpu_validation as W

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    plan = build()
    plan_body = json.dumps(plan, indent=1, ensure_ascii=False) + "\n"
    sha = hashlib.sha256(plan_body.encode()).hexdigest()
    auth_body = json.dumps(proposed_authorization(plan, sha), indent=1) + "\n"
    camp_path = REPO_ROOT / W.CAMPAIGN
    bodies = {W.PLAN: plan_body, W.AUTHORIZATION_PROPOSED: auth_body}
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
        print(f"{W.CAMPAIGN}: " + (f"fixed fields drifted {drift}" if drift
                                   else "ledger present; fixed fields current"))
        stale += bool(drift)
    elif args.write:
        camp_path.write_text(json.dumps(want, indent=1) + "\n")
        print(f"wrote {W.CAMPAIGN}")
    else:
        print(f"{W.CAMPAIGN}: MISSING (run with --write)")
        stale += 1
    return 0 if args.write or not stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
