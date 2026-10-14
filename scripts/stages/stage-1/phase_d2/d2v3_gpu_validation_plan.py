#!/usr/bin/env python3
"""D2 version 3's minimal GPU validation: the PROPOSAL record, derived.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d2/d2v3_gpu_validation_plan.py [--write]

Version 3 changes three numeric definitions that only a GPU, the real
teacher and its bf16 kernels can validate: the union Top-K KL with exact
retrieval, the deterministic Top-K on real bf16 ties, and the teacher's
Top-200 + tail confidence. Everything else a GPU validation needs was already
exercised by version 2's Stage V on the paid pod and is REUSED, not repeated:
the launcher and acquisition loop, the CUDA device handoff, the cache guard
under a hoarded pool, teardown confirmation, campaign booking and adoption by
recomputed verdicts (`STAGE_V_REUSE`).

So this plan is the smallest set of steps that answers the new questions:

* no B rebuild (Stage V's dominant 21 minutes): the candidates are the
  teacher's own compressed variants -- an FFN child under version 3's
  weighted statistics, and DEPTH's bypassed candidates of it -- which are
  real compressed distributions and cost one operator invocation;
* one DEPTH round at all_v1 and the same round at scored_v1 (the scored_v1
  equivalence of THIS objective, which also decides the price lever);
* the state evaluator's union store and its per-evaluation cost;
* the union numerics against the FULL teacher logits on a sample.

Bounds come from Stage V's MEASURED step durations with the plan's factor,
plus version 3's union terms from `d2_budget.union_kl_overhead` at their
bounds -- never typed minutes. AUTHORIZES NOTHING: it is a proposal, and the
driver stage it names is not implemented yet (`prerequisites`).
"""
from __future__ import annotations

import argparse
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

SCHEMA = "aadistill.d2v3_gpu_validation_plan/v1"
VALIDATION_DIR = ("logs/stages/stage-1/families/d_series/validations/"
                  "d2v3-gpu-validation/v1")
PLAN = f"{VALIDATION_DIR}/plan.json"
#: The plan's margin on a measured step duration (Stage V's own factor).
STEP_BOUND_FACTOR = 1.5
#: The union numerics sample: items per profile and bypass candidates.
NUMERICS_SAMPLE = {"items_per_profile": 8, "bypass_candidates": 3}
#: The float64 full-vocabulary oracle's cost per row, in union reductions.
ORACLE_ROW_FACTOR = 4.0

#: What version 2's Stage V already exercised on the paid GPU, reused as is.
STAGE_V_REUSE = {
    "campaign": ("logs/stages/stage-1/families/d_series/validations/"
                 "d2-gpu-validation/v1/campaign.json"),
    "run": "d2val_20261014_103301",
    "reused_mechanisms": [
        "scripts/stages/stage-1/phase_d1/topk_adoption_launch.sh (launcher, "
        "remote payload, evidence fetch)",
        "scripts/stages/stage-1/phase_d2/d2val_acquire.sh (one-pod "
        "acquisition loop with capacity-refusal booking at $0)",
        "the CUDA device handoff (place_for_execution, "
        "require_execution_devices) and the executed-commit binding",
        "the reference-cache guard under a deliberately hoarded pool",
        "provider-confirmed teardown and scripts/shared/pod/"
        "book_engineering_subrun.py booking",
        "adoption by verdicts RECOMPUTED from the measurements"],
    "reused_measurements": [
        "version 2's per-step durations (this plan's bound basis)",
        "the weighted-statistics mechanism (denominators equal to the weight "
        "sum, full vs narrow scope timing): version 3 changes the weight "
        "VALUES, not the path"],
    "not_reused": [
        "any verdict on the union KL, the retrieval, the deterministic "
        "Top-K or the K+1 confidence",
        "version 2's scored_v1 equivalence (a different objective)",
        "version 2's teacher-confidence distribution"],
}


def _adopted() -> dict[str, Any]:
    from stages.d_series import d2_budget as BUD

    v = BUD.adopted_validation()
    if not v:
        raise RuntimeError("version 2's Stage V is not adopted; this plan's "
                           "bounds rest on its measured step durations")
    return v


def build() -> dict[str, Any]:
    from aadistill.infrastructure.manifest import sha256_json
    from stages.d_series import d2_budget as BUD
    from stages.d_series import objectives as O
    from stages.phase_d2 import d2_session as S

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
    ov = BUD.union_kl_overhead(BUD.calibration_evidence(), basis, v)
    added = ov["union_terms"]["added"]
    geo = ov["union_terms"]["geometry"]
    f = STEP_BOUND_FACTOR

    def wall(step: str) -> float:
        return float(rec[step]["wall_seconds"]) / 60

    t4 = rec["V4_depth_root_vs_parent"]["timing"]
    cands = int(t4["candidate_subsets"])
    per_cand = added["depth_seconds_per_candidate"]
    per_eval = added["state_eval_seconds_per_evaluation"]
    scored = geo["depth_rows_per_candidate"]["scored_v1"] / \
        geo["depth_rows_per_candidate"]["all_v1"]
    ffn = rec["V3_weighted_statistics"]["operators"][
        "ffn.activation_importance_v0"]["arms"]
    ffn_min = (float(ffn["full_scope"]["seconds"])
               + float(ffn["narrow_scope"]["seconds"])) / 60
    n_items = 2 * NUMERICS_SAMPLE["items_per_profile"]
    #: The numerics sample, from MEASURED sizes: the stores' rows per item,
    #: DEPTH's measured forward seconds per item, and the union terms per
    #: row; the float64 full-vocabulary oracle (two float64 log-softmaxes and
    #: two partition sums per row) at ORACLE_ROW_FACTOR union reductions.
    stores = rec["V1_root_stores_and_validity"]["profiles"].values()
    rows_per_item = (sum(int(p["decision"]["positions"]) for p in stores)
                     / sum(int(p["decision"]["items_total"]) for p in stores))
    forward_per_item = (float(t4["ablated_seconds"])
                        / max(1, int(t4["ablated_items"])))
    rows_all = geo["depth_rows_per_candidate"]["all_v1"]

    def numerics_minutes(which: str) -> float:
        per_row = per_cand[which] / rows_all
        return (NUMERICS_SAMPLE["bypass_candidates"] * n_items
                * (forward_per_item
                   + rows_per_item * per_row * ORACLE_ROW_FACTOR)) / 60
    m = lambda x: round(x, 2)                                    # noqa: E731
    v0 = next(r for r in stage_v_plan["steps"] if r["id"] == "V0_setup")
    steps = [
        ("W0_setup", "pod setup, environment, teacher fetch and load, the $0 "
         "premises on the pod", "infrastructure",
         v0["expected_minutes"], v0["bound_minutes"], "core", None),
        ("W1_union_stores_and_confidence",
         "the union root store of BOTH profiles at the formal bsz=3 and at "
         "bsz=1: identity against the declaration, build seconds and bytes; "
         "float32 head logits against the model's own bf16 logits; the K+1 "
         "confidence against float64 on identical sketches; the "
         "teacher-confidence validity report (distribution, IQR, ESS, spans, "
         "domains, token types)",
         "the real teacher's bf16 hidden states and its Top-200 + tail "
         "confidence are the question",
         m(wall("V1_root_stores_and_validity") + added["one_time_minutes"][
             "expected"]),
         m(wall("V1_root_stores_and_validity") * f * 2
           + added["one_time_minutes"]["bound"]),
         "core", "arithmetic (confidence); integrity (store identity)"),
        ("W2_ffn_child_weighted",
         "FFN at the root under version 3, full scope (weighted by the K+1 "
         "confidence) and narrow scope: the parent of W4/W5, the weighted "
         "denominator check and the selection overlap",
         "a real compressed parent, and the new weights on a real selection",
         m(ffn_min), m(ffn_min * f), "core",
         "integrity (denominator = weight sum); overlap diagnostic"),
        ("W3_union_numerics",
         f"on {n_items} items ({NUMERICS_SAMPLE['items_per_profile']} per "
         f"profile) and {NUMERICS_SAMPLE['bypass_candidates']} bypass "
         "candidates of W2's child: KL_{S_T+tail} <= KL_U <= KL_full against "
         "the FULL teacher logits (float64 oracle), retrieval against the "
         "full head, union size and overlap, tail masses, bf16 tie "
         "frequency, torch.topk-vs-deterministic set differences",
         "real 151,936-entry bf16 tails and ties; the CPU check ran on a "
         "stand-in checkpoint",
         m(numerics_minutes("expected")),
         m(numerics_minutes("bound") * f),
         "core", "arithmetic (union KL vs float64; retrieval; bounds)"),
        ("W4_depth_union_all_v1",
         "one DEPTH round on W2's child under version 3 at all_v1 with "
         "synchronized telemetry: the union terms' seconds per candidate, the "
         "rankings by weighted union KL, unweighted union KL, teacher Top-K "
         "+ tail KL and parent KL, the cache modes",
         "the per-candidate price term is the question",
         m(wall("V4_depth_root_vs_parent") + cands * per_cand["expected"] / 60),
         m(wall("V4_depth_root_vs_parent") * f
           + cands * per_cand["bound"] / 60),
         "core", "integrity (root store with retrieval, table consistency)"),
        ("W5_depth_union_scored_v1", "the same DEPTH round at scored_v1",
         "THIS objective's scored_v1 equivalence, which decides the price "
         "lever",
         m(wall("V5_depth_scored_rows")
           + cands * per_cand["expected"] * scored / 60),
         m(wall("V5_depth_scored_rows") * f
           + cands * per_cand["bound"] * scored / 60),
         "core", "row equivalence (DEPTH)"),
        ("W6_state_evaluation",
         "W2's child under version 3 at bsz=3 all_v1, bsz=3 scored_v1 and "
         "bsz=1 all_v1, and KL-only on the historical support: the "
         "per-evaluation seconds, the raw-KL control, the scored_v1 "
         "equivalence, the suite store",
         "the per-evaluation price term on the real suite",
         m(wall("V6_state_evaluation") + 4 * per_eval["expected"] / 60),
         m(wall("V6_state_evaluation") * f * 2
           + 4 * per_eval["bound"] / 60),
         "core", "arithmetic (raw-KL control); row equivalence (state)"),
        ("W7_teardown", "evidence fetch and teardown", "infrastructure",
         1.0, 2.0, "core", None),
    ]
    rows = [{"id": i, "what": w, "why_gpu": y, "expected_minutes": e,
             "bound_minutes": b, "class": c, "gates": g}
            for i, w, y, e, b, c, g in steps]
    exp = sum(r["expected_minutes"] for r in rows)
    bound = sum(r["bound_minutes"] for r in rows)
    cap = BUD._usd(bound, rate, disk)
    session_cap = math.ceil(cap["all_in_usd"] * 100) / 100
    prereg = S.preregistration(REPO_ROOT)
    contract = json.loads((REPO_ROOT / O.CONTRACT).read_text())
    science = {
        "schema": SCHEMA,
        "validation_id": "d2v3_gpu_validation",
        "version": 1,
        "objective": O.D2_OBJECTIVE.qualified_id,
        "objective_hash": O.D2_OBJECTIVE.objective_hash,
        "support": O.D2_SUPPORT.as_dict(),
        "contract_hash": contract["contract_hash"],
        "preregistration_hash": prereg["preregistration_hash"],
        "formal_topology": {"micro_batch_size": 3,
                            "calibration_batch_packing": "length_sorted_v1"},
        "stage_v_reuse": STAGE_V_REUSE,
        "steps": rows,
        "numerics_sample": dict(NUMERICS_SAMPLE),
        "acceptance": {
            "arithmetic": {
                "gate": True,
                "tolerances": {
                    "union_kl_vs_float64_abs_nats": 1e-4,
                    "union_kl_vs_float64_rel": 1e-4,
                    "confidence_vs_float64_abs": 1e-5,
                    "retrieval_vs_full_head_abs_nats": 2e-4,
                    "bound_violations_beyond_float32_slack": 0},
                "what": ("one set of inputs, two computations: the float32 "
                         "union reduction against a float64 oracle on the "
                         "SAME logits; retrieval against the full head's "
                         "log-softmax at the same rows; the K+1 confidence "
                         "against float64 on the same sketch"),
                "basis": ("the $0 CPU check on real logits measured union "
                          "error max 1.6e-5, retrieval max 6e-5 and no bound "
                          "violation; the tolerances leave room for GPU "
                          "summation order, never for a definition error")},
            "integrity": {
                "gate": True,
                "checks": ["each union store matches its declared identity "
                           "(support with K and tie-break, definitions, "
                           "build)",
                           "DEPTH reads the ROOT store with retrieval; every "
                           "decision score equals the recorded objective",
                           "the weighted FFN denominator equals the weight "
                           "sum",
                           "the state evaluator forwards no teacher per "
                           "evaluation"]},
            "row_equivalence": {
                "gate": True,
                "tolerances": {"depth_max_relative": 1e-5,
                               "depth_same_choice": True,
                               "state_max_abs": 1e-5},
                "decides": "whether D2's search may run scored_v1"},
            "not_gated": ["bsz=1 vs bsz=3 drift of the stores and metrics",
                          "union-vs-full approximation ratios and ranking "
                          "stability", "tie frequencies",
                          "the K+1 confidence's dynamic range"],
        },
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
                                  "union store and one block's transients")},
        "deliverables": [
            "an adoption record with recomputed verdicts (arithmetic, "
            "integrity, row equivalence)",
            "MEASURED union terms per DEPTH candidate and per state "
            "evaluation, replacing the provisional ones in d2_budget",
            "the K+1 confidence's distribution on the real teacher",
            "the union-vs-full ratios on real teacher logits"],
        "does_not": ["run a D2 search, recovery training or a behavioural "
                     "probe", "rebuild B", "touch D1, D3 or their records",
                     "repeat Stage V's version-2 measurements"],
        "prerequisites": [
            "the driver stage implementing W1-W6 (as Stage V's "
            "d2_gpu_validation.py implements V1-V6), rehearsed at $0 on the "
            "CPU with a tiny model before any grant",
            "maintainer approval of preregistration v3 and of this "
            "proposal's grant"],
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
                                        + BUD.TEARDOWN_RESERVE_USD, 2)},
        "authorizes": "nothing",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    body = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    path = REPO_ROOT / PLAN
    if a.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        print(f"wrote {PLAN}")
        return 0
    same = path.is_file() and path.read_text() == body
    print(f"{PLAN}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
