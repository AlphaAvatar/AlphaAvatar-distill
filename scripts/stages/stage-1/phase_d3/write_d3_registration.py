#!/usr/bin/env python3
"""Write the new D3's REGISTRATION: the design every D3 session is checked against.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/write_d3_registration.py [--write]

Registered BEFORE any D3 structural or behavioural outcome exists (maintainer
decision 2026-10-16): the control, the single-factor treatment and its derived
identities, the structural outcome rule, and -- for the conditional three-probe
rung -- the seeds, battery, recovery, comparators, what is reported and the
EXPLORATORY classification rule. Everything numeric is DERIVED here from its
owner (the control plan and record, the shared protocol, the core), never
typed. Authorizes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "aadistill.phase_d3.registration/v1"
#: The D1 confirmation seeds and battery, as the maintainer named them; the
#: writer requires the shared protocol's confirmation rung to carry exactly
#: these, so the registration can never silently re-bind another set.
CONFIRMATION_SEEDS = (396880939, 766527341, 1004399678)
BATTERY_ROLE = "d1_confirmation"
RECIPE_ID = "e1_p1_kd_heavy@0.86M"
#: The cancelled experiment this one replaces, by identity.
CANCELLED_D3 = {
    "objective_hash": ("7b2545389874b445628ac46ddef32bebee92622150560c1bd899"
                       "dffb75033636"),
    "objective_id": "objective.d3_kl_minus_student_confidence@v2",
    "status": "CANCELLED -- NEVER EXECUTED (maintainer decision 2026-10-16)",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    from aadistill.infrastructure.manifest import sha256_json
    from aadistill.initialization.scoring.kl_reference import (
        KL_REFERENCE_KEY, ROOT_REFERENCE_V1,
    )
    from stages.d_series import shared_evaluation as SE
    from stages.phase_d3 import d3_design as D

    leaf = D.control_leaf(repo_root)
    _, plan_sha = D.control_plan(repo_root)
    control_path = repo_root / D.CONTROL_RECORD
    control = json.loads(control_path.read_text())
    ctrl_spec = D.control_spec(leaf)
    probe_spec = D.treatment_spec(leaf)
    derived = D.operator_config_hashes(probe_spec, repo_root)
    at = D.treated_index(leaf)
    treated_hash = derived[at]["config_hash"]
    #: Every UNtreated step's derived input must still be q2's recorded one.
    moved = [r for r in derived if r["step"] != at
             and r["config_hash"] != r["pinned"]]
    if moved:
        raise D.D3DesignError(f"untreated steps' configs moved: {moved}")
    if treated_hash == leaf.steps[at].expected_config_hash:
        raise D.D3DesignError("the declared reference did not fork the DEPTH "
                              "config; the treatment would share q2's identity")
    spec = D.treatment_spec(leaf, treated_config_hash=treated_hash)
    contract = D.treatment_contract(ctrl_spec, spec)
    execution = D.execution()

    protocol = SE._read(SE.RECORD, repo_root)
    rung = protocol["rungs"]["shared_confirmation"]
    if tuple(rung["seeds"]) != CONFIRMATION_SEEDS:
        raise D.D3DesignError(
            f"the shared confirmation rung's seeds are {rung['seeds']}, not the "
            f"maintainer's {list(CONFIRMATION_SEEDS)}")
    if rung["reuses_battery"]["role"] != BATTERY_ROLE:
        raise D.D3DesignError("the confirmation rung does not reuse "
                              f"{BATTERY_ROLE}")
    if protocol["recovery"]["recipe_id"] != RECIPE_ID:
        raise D.D3DesignError("the shared recovery recipe is not "
                              f"{RECIPE_ID}")
    units = {(u["arm"], u["seed"]): u["probe_id"]
             for u in rung["reference_units"]}
    comparators = {
        arm: [units[(arm, s)] for s in CONFIRMATION_SEEDS]
        for arm in ("q2", "B")}
    stats = protocol["statistics"]

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": D.EXPERIMENT_ID,
        "name": "Fixed-q2 DEPTH Teacher-Reference Ablation",
        "version": 1,
        "status": ("REGISTERED -- $0 readiness only. No paid structural or "
                   "behavioural session is authorized by this record."),
        "decided_by": ("maintainer decision 2026-10-16 (the D3 definition); "
                       "logs/budget/decisions.md 2026-10-16"),
        "question": (
            "Given the exact D1-q2 structural initialization path, does "
            "choosing DEPTH layers by minimizing KL against the original "
            "Teacher, instead of the FFN-compressed Parent, produce a better "
            "initialization for downstream capability recovery?"),
        "replaces": CANCELLED_D3,
        "identity_is_distinct_from_the_cancelled_d3": True,
        "control": {
            "name": "D1-q2", "state_id": D.CONTROL_STATE_ID,
            "artifact_digest": D.CONTROL_ARTIFACT_DIGEST,
            "path_label": leaf.path_label,
            "control_path_hash": ctrl_spec.spec_hash,
            "replay_plan": D.CONTROL_PLAN, "replay_plan_sha256": plan_sha,
            "control_record": D.CONTROL_RECORD,
            "control_record_sha256": _sha256(control_path),
            "steps": [{"index": s.index, "kind": s.kind, "impl_id": s.impl_id,
                       "profile_id": s.profile_id,
                       "expected_artifact_digest": s.expected_artifact_digest,
                       "expected_config_hash": s.expected_config_hash}
                      for s in leaf.steps],
            "depth_removal_order": control["depth"]["removal_order"],
            "depth_kept_layers": control["depth"]["kept_layers"],
            "final_identities": control["final"],
        },
        "treatment": {
            "path_id": D.TREATMENT_PATH_ID,
            "treatment_path_hash": spec.spec_hash,
            "device": spec.device,
            "treated_step": at,
            "declared_config": {KL_REFERENCE_KEY: ROOT_REFERENCE_V1},
            "treated_step_config_hash": treated_hash,
            "derived_config_hashes": [
                {k: r[k] for k in ("step", "impl_id", "profile_id",
                                   "config_hash")} for r in derived],
            "contract": contract,
            "objective_control": "KL(P || C(S)) -- P: the 36-layer parent "
                                 "after q2's frozen FFN step",
            "objective_treatment": "KL(T || C(S)) -- T: the original, "
                                   "uncompressed teacher",
            "kl": "forward KL on the REFERENCE-defined Top-200 + aggregate "
                  "tail (reference_topk_tail_v1, K=200), both arms",
            "unchanged": [
                "scored positions (positions.supervised_target_v1@v1)",
                "per-item mean, sub-type then domain-balanced aggregation",
                "calibration data (calib.reasoning_heavy@v2)",
                "greedy removal, candidate subsets, 36 -> 28 geometry",
                "deterministic tie-break to the lower layer index",
                "D1's state evaluator (not used to select; no search)",
                "FFN, RESIDUAL_WIDTH and ATTENTION local selection rules",
                "operator order, profiles, seed, target architecture"],
            "not_introduced_from_d2": [
                "teacher-confidence weighting", "student-confidence reward",
                "reference_student_union_topk_tail_v1",
                "confidence-weighted activation statistics",
                "D2's beam-ranking objective",
                "D2's structural path or selected initializations"],
            "reference_support_note": contract["_reference_support"],
            "teacher_reference": {
                "built_from": "the original teacher, before any compression, "
                              "while the root is held (materialize_fixed_path)",
                "root_identity": D.root_identity(leaf),
                "profile": leaf.steps[at].profile_id,
                "rows": "every real prediction position; DEPTH's position "
                        "policy selects among them",
                "execution": {"micro_batch_size": execution.micro_batch_size,
                              "calibration_batch_packing":
                                  execution.calibration_batch_packing,
                              "prediction_rows": execution.prediction_rows},
                "fails_closed_on": [
                    "a missing store", "another mixture or item content",
                    "another profile", "another Top-K support",
                    "another root", "another execution protocol",
                    "a store with no execution binding"]},
            "parent_reference_diagnostics": (
                "every candidate of every round, from the same candidate "
                "logits, after the decision value; never read by the "
                "selection"),
            "downstream": ("RESIDUAL_WIDTH and ATTENTION are recomputed on the "
                           "actual new parent; q2's downstream digests are "
                           "never forced onto changed weights"),
        },
        "structural_outcomes": {
            "rule": "stages.phase_d3.d3_design.classify_structure",
            D.NO_STRUCTURAL_CHANGE: (
                "the treatment keeps q2's layers and every downstream step "
                "reproduces q2's bytes (all four final identities equal): "
                "preserve the evidence, NO recovery training"),
            D.STRUCTURAL_CHANGE: (
                "the treatment keeps a different layer set and its initial "
                "checkpoint is new: propose the separately authorized "
                "three-probe behavioural rung"),
            D.UNEXPLAINED_BYTE_DIFFERENCE: (
                "the same layers and different bytes: diagnose the "
                "materialization; nothing is called a scientific effect"),
            "lower_teacher_kl_is_not_a_result": True,
        },
        "behavioural_rung": {
            "conditional_on": D.STRUCTURAL_CHANGE,
            "separately_authorized": True,
            "n_probes": len(CONFIRMATION_SEEDS),
            "seeds": list(CONFIRMATION_SEEDS),
            "seeds_owner": rung["seeds_owner"],
            "battery": {"role": BATTERY_ROLE, **{k: rung["reuses_battery"][k]
                        for k in ("family_content_id", "item_ids_sha256",
                                  "n_prompts", "n_scorable")}},
            "recovery": {k: protocol["recovery"][k] for k in (
                "recipe_id", "recipe_digest", "data_digest",
                "trainer_source_digest", "training_runtime_digest",
                "training_image_digest")},
            "evaluation": {k: rung[k] for k in (
                "evaluation_protocol_hash", "generation_protocol_fingerprint",
                "scoring_digest")},
            "shared_protocol_hash": protocol["protocol_hash"],
            "start_from": ("the new D3 INITIAL checkpoint, never q2's "
                           "recovered weights"),
            "comparators": {
                "q2": {"role": "primary method comparison",
                       "units": comparators["q2"]},
                "B": {"role": "standing scientific anchor",
                      "units": comparators["B"]}},
            "comparators_are_imported": (
                "historical q2 and B units, verified by identity and "
                "per-sample hash; never retrained, regenerated or re-evaluated"),
            "screening_seeds": (
                "NOT part of this experiment. No rule for adding them is "
                "registered, so they are not added -- in particular not in "
                "response to any D3 outcome"),
            "reported": [
                "per-seed and pooled correct_overall", "paired per-sample "
                "counts (candidate-only / comparator-only / both)",
                "usable_rollout_rate and every component", "correct_given_usable",
                "per-stratum (set and domain) correctness and usable rollout",
                "natural termination, degeneration, context-limit hits",
                "generated and reasoning lengths"],
            "exploratory_classification": {
                "status": ("EXPLORATORY. The battery and the q2/B results were "
                           "observed before this design; nothing here is a "
                           "fresh independent confirmation"),
                "comparison": "D3 - q2 on correct_overall (primary method "
                              "comparison); D3 - B reported the same way as "
                              "the anchor",
                "interval": ("two-sided 95% stratified prompt-cluster "
                             "bootstrap of the prompt-mean seed-mean paired "
                             "difference (stages.phase_c1.isolation), seeds "
                             "as fixed blocks -- DESCRIPTIVE"),
                "bootstrap_iterations": int(stats["bootstrap_iterations"]),
                "bootstrap_seeds": {
                    cmp: SE.bootstrap_seed(protocol["protocol_hash"],
                                           D.EXPERIMENT_ID,
                                           f"exploratory_vs_{cmp}")
                    for cmp in ("q2", "B")},
                "labels": {
                    "BEHAVIOURAL_IMPROVEMENT": (
                        "the interval's lower end is above 0 AND at least 2 of "
                        "3 seed differences are positive"),
                    "BEHAVIOURAL_REGRESSION": (
                        "the interval's upper end is below 0 AND at least 2 "
                        "of 3 seed differences are negative"),
                    "INCONCLUSIVE": "otherwise"},
                "usable_rollout": ("reported with the same descriptive "
                                   "interval, separately; it never changes "
                                   "the correctness label"),
                "alpha": ("NONE. The cancelled D3's alpha allocation in the "
                          "shared protocol's family split is not inherited, "
                          "and no interval here is a test"),
                "never": ["SHARED_GO", "a verdict contract of the cancelled D3",
                          "automatic promotion",
                          "a claim of fluid-reasoning superiority"],
            },
        },
        "claim_boundary": [
            "a structural-method comparison on the existing mixed-domain "
            "protocol, not a measurement of fluid reasoning (AGENTS.md P10.2)",
            "the behavioural comparisons are exploratory on an observed "
            "battery, not a fresh independent confirmation",
            "selecting an official initialization method needs appropriately "
            "independent evidence",
            "if the batteries cannot separate the two methods, both are "
            "carried as candidates for a future capability-aligned evaluation",
            "it isolates the DEPTH reference within q2's path, not every "
            "difference between the full D1 and D2 searches"],
        "pricing": "logs/stages/stage-1/phase_d3/plans/d3_budget_v1.json",
        "authorizes": "nothing",
    }
    doc["registration_hash"] = sha256_json(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    from stages.phase_d3 import d3_design as D

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    body = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    out = REPO_ROOT / D.REGISTRATION
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        print(f"wrote {D.REGISTRATION}")
        return 0
    same = out.is_file() and out.read_text() == body
    print(f"{D.REGISTRATION}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
