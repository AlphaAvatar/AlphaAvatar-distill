#!/usr/bin/env python3
"""The shortened A-bsz3 adoption design, derived from what already exists.

    PYTHONPATH=src:scripts python scripts/autoinit/write_a_bsz3_adoption.py --write

**Why this is a writer and not a hand-typed document.** Three of the numbers
the design turns on are measurements that live somewhere else: the control
values are attempt75's, the identities are the C3 preregistration's, and the
seed-1 stop threshold is derived from the per-seed spread C3 itself observed.
Typing any of them here would create a second copy to keep in step with the
first, and this repository has already had a plan assert a bootstrap seed that
the computation did not use.

**What changed, and why.** The maintainer reviewed
`review/c3-operator-batching@ab4f32ed` on 2026-10-01 and accepted the A-bsz3
implementation while refusing the 16-probe non-inferiority design. The reason
given was SCOPE, not power: A-bsz3 is an execution-optimization validation, and
the programme does not need a >=90%-power population-level non-inferiority
claim to decide whether a batching knob is worth switching on. The withdrawn
design's power analysis was not wrong -- non-inferiority at a given margin does
need more data than superiority at the same margin -- it was answering a
question nobody asked.

So the claim shrinks with the design. Nothing produced by this plan may be
reported as a 95% non-inferiority conclusion.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3.a_bsz3 import (  # noqa: E402
    ATTENTION_IMPL_ID, execution_comparison, frozen_identities,
    item_token_counts,
)

PREREG = REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"
DECISION = REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i/c3_decision.json"
PROBES = REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i/c3_probe_results.json"
PRICING = REPO / "logs/stages/stage-1/phase_c3/plans/a_bsz3_pricing.json"
OUT = "logs/stages/stage-1/phase_c3/plans/a_bsz3_adoption.json"

#: The design this one replaces, and the hash it carried. Named so the
#: withdrawal is a fact in the record rather than an absence someone notices.
WITHDRAWN = {
    "plan": "logs/stages/stage-1/phase_c3/plans/a_bsz3_noninferiority.json",
    "preregistration_sha256":
        "5db8563c2b1590f1fd618f77696c629e93584cfd3153fb0a3a8de493627bb350",
    "design": "8 seeds x 2 arms = 16 probes, ADOPT iff one-sided LCB > -0.010",
    "withdrawn_utc": "2026-10-01",
    "withdrawn_by": "maintainer review of review/c3-operator-batching@ab4f32ed",
    "reason": (
        "SCOPE. A-bsz3 is an execution-optimization validation; the programme "
        "does not need a population-level non-inferiority claim about it. The "
        "power analysis that produced 16 probes is retained as a finding -- "
        "non-inferiority at a margin needs more data than superiority at the "
        "same margin -- and is simply not the question being asked."),
    "_what_survives": (
        "the implementation, the $0 padding analysis, the registered "
        "prediction that the digests will differ, and the rule that an "
        "execution knob which entered a hash would have answered the "
        "equivalence question by definition"),
}

#: The programme's smallest effect worth having. Reported as the SCALE of a
#: material loss; it is not used here as a confidence criterion.
SESOI = 0.010

#: TWO CONCEPTS THAT SHARE ONE ID TODAY, and the reason a differing digest
#: cannot be adopted on the strength of the behavioural study alone.
#:
#: Added on 2026-10-01 after the maintainer pointed out that the design, as
#: written, settled the SCIENTIFIC question and left the MATERIALIZATION
#: question open. Both halves of the old wording were true and the gap between
#: them was not stated.
IDENTITY_SEMANTICS = {
    "_the_distinction": (
        "scientific/semantic identity asks whether two executions pose the "
        "same operator-level question. Materialization/numerical identity "
        "asks whether two executions may be treated as the same resumable, "
        "deduplicable checkpoint. A-bsz1 and A-bsz3 share the first by "
        "construction. Whether they share the second is exactly what step 1 "
        "measures, and it is not settled in advance."),
    "what_compute_state_id_binds": [
        "root teacher", "target spec",
        "operator implementation id and signature hash",
        "calibration profile hash", "operator config hash", "seed",
    ],
    "what_it_does_NOT_bind": ["ExecutionConfig", "the resulting artifact digest"],
    "_so": (
        "`OperatorStep.identity()` is a SEMANTIC id. It is correct that it "
        "excludes the batching knobs -- otherwise two runs of the same "
        "science on different hardware would be different scientific states, "
        "a resume would not find its own journal, and the A-bsz1/A-bsz3 "
        "comparison would be answered by definition. But a semantic id is "
        "not sufficient to own, resume or deduplicate BYTES, and today it is "
        "the only id the beam has."),
    "where_the_hazard_lives": (
        "BeamSearch._restore looks a state up in StateStore.latest_by_state_id "
        "by state_id, then re-identifies the checkpoint on disk and refuses if "
        "the bytes disagree with the record. That guard catches a stale or "
        "tampered checkpoint. It cannot catch a record that is internally "
        "consistent and was produced by a DIFFERENT numerical protocol."),
    "adoption_logic": {
        "digests_identical": (
            "A-bsz3 is demonstrated to be a TRANSPARENT execution "
            "optimization for this case. It may retain the same scientific "
            "AND materialization identity, and adoption is decided from the "
            "runtime evidence alone. No additional mechanism is owed, and "
            "none may be built speculatively."),
        "digests_differ": (
            "A-bsz3 is a DISTINCT NUMERICAL MATERIALIZATION PROTOCOL even "
            "though the operator semantics, the hypothesis and the estimand "
            "are unchanged. The shortened behavioural sanity study may still "
            "be proposed. But A-bsz3 MAY NOT be adopted into D1/D2/D3 "
            "execution until the repository has an explicit way to bind the "
            "numerical execution fingerprint to materialization/resume "
            "identity. Two artifacts that differ in bytes must not share a "
            "resumable, deduplicable state identity in any future beam "
            "search."),
        "_the_behavioural_result_does_not_lift_this": (
            "a passing sanity check says the downstream behaviour is not "
            "materially worse. It says nothing about whether two different "
            "artifacts may share one resume key, which is an engineering "
            "correctness property and not a behavioural one."),
    },
    "_forbidden_resolution": (
        "do NOT invent `attention.activation_importance_bsz3`. The operator "
        "semantics are unchanged, a second impl id would fork the scientific "
        "identity to fix a materialization problem, and it would answer the "
        "equivalence question by definition -- the exact error this study was "
        "designed to avoid."),
    "_required_shape_if_it_is_ever_needed": (
        "the SMALLEST generic separation between semantic/scientific path "
        "identity and the numerical/materialization identity used for "
        "artifact ownership, resume and deduplication. Generic, not A-bsz3 "
        "specific: any execution knob that can move bytes has this property. "
        "NOT BUILT. If step 1 shows exact artifact identity it is never "
        "needed for this optimization, and building it first would be "
        "speculative infrastructure under AGENTS.md P8.2.1."),
    "regression_that_holds_today": (
        "tests/autoinit/test_corrections.py :: "
        "test_resume_refuses_a_record_from_a_different_numerical_protocol -- "
        "a matching semantic state id is not sufficient to reuse a "
        "checkpoint, driven through the real resume path."),
}


def _load(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing {what}: {path}")
    return json.loads(path.read_text())


def controls_from_attempt75(probe_results: dict[str, Any],
                            run_id: str) -> dict[str, Any]:
    """attempt75's three A_incumbent probes, as the reusable control arm.

    These are the exact probes the maintainer authorized for reuse: valid,
    protocol-uniform and fully scored, at the frozen C3 recovery seeds. Reuse
    is admissible because this study makes no operator-promotion and no
    population-level claim -- NOT because cross-session controls are generally
    sound, which they are not. The price is recorded in `claim_boundary`.
    """
    rows = []
    for p in probe_results["probes"]:
        if p["arm"] != "A_incumbent":
            continue
        rows.append({
            "probe_id": p["probe_id"],
            "seed": p["seed"],
            "initialization_artifact_digest":
                p["initialization_artifact_digest"],
            "correct": p["counts"]["correct"],
            "n_scorable": p["counts"]["n_scorable"],
            "correct_overall": p["rates"]["correct_overall"],
            "usable": p["counts"]["usable"],
            "usable_rollout_rate": p["rates"]["usable_rollout_rate"],
            "result_sha256": p["result_sha256"],
            "per_sample_sha256": p["per_sample_sha256"],
            "train_config_sha256": p["trained_run"]["config_sha256"],
        })
    rows.sort(key=lambda r: r["seed"])
    if len(rows) != 3:
        raise SystemExit(
            f"expected three A_incumbent controls in attempt75, found {len(rows)}")
    return {
        "arm": "A_incumbent",
        #: Named by the decision artifact, not defaulted here. A fallback
        #: literal would keep saying "attempt75" after the day it stopped
        #: being true.
        "source_run": run_id,
        "probes": rows,
        "evidence_root": "/home/ecs-user/aad-artifacts/phase_c3/attempt75",
        "_why_these_are_admissible": (
            "valid, protocol-uniform, fully scored, at the frozen C3 recovery "
            "seeds, and this study makes no operator-promotion claim"),
        "_what_they_are_not": (
            "a control arm measured in the same session as the treatment. "
            "See claim_boundary."),
    }


def seed1_stop_threshold(decision: dict[str, Any]) -> dict[str, Any]:
    """The fail-fast threshold, DERIVED from this instrument's own spread.

    A written constant would be a guess wearing a gate's clothing. What the
    gate has to clear is known and measured: across all three C3 contrasts --
    arms whose POOLED differences were within +/-0.002 of each other -- the
    per-seed deltas still ranged over +/-0.0224. A single-seed stop set below
    that would fire on protocols that are behaving identically.
    """
    per_seed = [abs(x) for c in decision["contrasts"].values()
                for x in c["per_seed_delta"]]
    pooled = [abs(c["delta"]) for c in decision["contrasts"].values()]
    observed_max = max(per_seed)
    #: 1.25x the largest observed single-seed excursion, rounded UP to the
    #: next 0.005. A stop threshold is a ceiling on tolerated noise, and a
    #: ceiling rounds up.
    threshold = math.ceil(observed_max * 1.25 / 0.005) * 0.005
    return {
        "threshold_delta": -round(threshold, 4),
        "metric": "correct_overall",
        "rule": ("STOP after seed 1 if (A_bsz3 - A_bsz1_control) at that seed "
                 f"is <= {-round(threshold, 4)}, or if any behavioural veto "
                 "fires"),
        "derivation": {
            "observed_max_abs_per_seed_delta": round(observed_max, 6),
            "observed_abs_pooled_deltas": [round(p, 6) for p in pooled],
            "margin_factor": 1.25,
            "rounded_up_to_multiple_of": 0.005,
            "_source": (
                "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i/"
                "c3_decision.json, all three contrasts"),
        },
        "_what_this_gate_catches": (
            f"a BROKEN protocol. {round(threshold, 4)} is {threshold / SESOI:.0f}x "
            "the programme's SESOI and larger than any single-seed excursion "
            "this instrument has produced between arms that pooled to "
            "indistinguishable."),
        "_what_it_deliberately_does_not_catch": (
            "a borderline result. A seed-1 delta between the threshold and 0 "
            "does NOT stop the study, because at one seed it is not "
            "distinguishable from the seed-level noise measured above. "
            "Stopping there would be reading noise as a verdict."),
        "_it_is_not_an_equivalence_margin": (
            "this is an ENGINEERING CATASTROPHIC-REGRESSION STOP and must "
            "never be reported as an equivalence margin. Clearing it means "
            "the protocol is not visibly broken at one seed; it does not "
            "bound the difference between the protocols, and no interval, "
            "power or equivalence statement follows from it."),
    }


def build(*, rate_note: str) -> dict[str, Any]:
    prereg = _load(PREREG, "the C3 preregistration")
    decision = _load(DECISION, "attempt75's stage-I decision")
    probe_results = _load(PROBES, "attempt75's probe results")
    pricing = _load(PRICING, "the A-bsz3 pricing")
    identities = frozen_identities()
    zero_cost = execution_comparison(item_token_counts())

    run_id = decision.get("run_id")
    if not run_id:
        raise SystemExit(f"{DECISION} names no run_id; the controls' source "
                         "run must come from the record, not a default")
    controls = controls_from_attempt75(probe_results, run_id)
    stop = seed1_stop_threshold(decision)
    seeds = identities["recovery_seeds"]

    doc: dict[str, Any] = {
        "schema": "aadistill.a_bsz3.adoption/v1",
        "stage": "A-bsz3",
        "designed_utc": "2026-10-01",
        "_contract": (
            "The ENGINEERING ADOPTION STUDY that decides whether the "
            "incumbent ATTENTION initialization should run at "
            "calibration_forward_batch_size=3 with length_sorted_v1 packing. "
            "Frozen before any A-bsz3 structural or behavioural result "
            "exists. AUTHORIZES NO SPEND."),
        "supersedes": WITHDRAWN,

        "_the_question": (
            "Does bsz3 + length_sorted_v1 materially reduce the execution "
            "cost of this operator's calibration without producing a material "
            "downstream regression? That is an execution-optimization "
            "validation. It is NOT a population-level non-inferiority claim "
            "and nothing here may be reported as one."),
        "_what_may_not_be_claimed_from_this": [
            "a 95% non-inferiority conclusion",
            "that A-bsz3 and A-bsz1 are statistically equivalent",
            "an operator promotion -- A-bsz3 is not a new operator",
            "anything about causal-KL, C4, or the B1-vs-B3 protocol choice",
            "that a differing artifact may be reused under the same state id",
        ],
        "_why_the_arms_are_the_same_scientific_state": (
            "both knobs are ExecutionConfig fields; neither enters a hash, a "
            "state id or a manifest identity. Giving the knob an "
            "implementation id would have answered the equivalence question "
            "by definition and measured nothing."),

        "identity_semantics": IDENTITY_SEMANTICS,

        "identities": identities,
        "operator": ATTENTION_IMPL_ID,

        "step_0": {
            "status": "DONE",
            "cost": "$0",
            "what": "forward count and padding from the frozen mixture's real lengths",
            "result": {
                "n_items": zero_cost["n_items"],
                "valid_tokens": zero_cost["valid_tokens"],
                "forwards": {
                    "A_bsz1": zero_cost["protocols"]["A_bsz1"]["physical_forwards"],
                    "A_bsz3": zero_cost["protocols"]["A_bsz3"]["physical_forwards"],
                    "ratio": zero_cost["deltas"]["physical_forwards_ratio"]},
                "padding_over_valid":
                    zero_cost["deltas"]["padding_over_valid"],
                "executed_positions_ratio":
                    zero_cost["deltas"]["executed_positions_ratio"],
            },
            "decides": ("nothing on its own. It establishes that the protocol "
                        "is worth measuring, and it is the PREDICTION the "
                        "observed counters in step 1 are checked against."),
        },

        "step_1": {
            "status": "IMPLEMENTED, NOT RUN, NOT FUNDED",
            "what": ("ONE same-device session: both protocols from the exact "
                     "same already-verified frozen parent, on the same "
                     "67-item calibration mixture."),
            "driver": "scripts/autoinit/compare_a_bsz3.py :: structural_half",
            "records": [
                "artifact digest, per protocol",
                "kept-head map, per protocol, and the diff",
                "per-head scores, Spearman rank correlation overall and per layer, score drift",
                "selection-boundary margins, and the margin at each flipped GQA group",
                "valid / padded / executed positions, OBSERVED by the operator as it ran",
                "runtime: scorer_seconds, CUDA-synchronized, over interleaved rounds",
                "end-to-end suffix wall clock",
                "peak VRAM",
            ],
            "measurement_protocol": {
                "rounds_per_protocol": 4,
                "interleaved": True,
                "warm_up_rounds_excluded": 1,
                "_why": (
                    "this operator was measured at 11.2732 s on this exact "
                    "parent geometry. A single sample of an 11-second "
                    "workload on a shared cloud GPU is not a measurement of a "
                    "ratio, and running the protocols back to back in a fixed "
                    "order makes the first one pay for allocator growth, "
                    "autotuning and kernel selection -- a difference between "
                    "the arms' ORDER, not between the protocols."),
                "same_session_required": (
                    "a cross-session timing reference already changed the SIGN "
                    "of a result once: a prior card's B1 was 2190.2 s against "
                    "2720.7 s on the measuring card, which would have reported "
                    "a 1.19x win as a 0.957x loss."),
                "device": "the approved L40S. When success IS digest equality, "
                          "a different card can manufacture a mismatch.",
            },
            "gates_that_void_the_comparison": [
                "A_bsz1 does not reproduce the frozen incumbent digest "
                f"{identities['incumbent_artifact_digest'][:12]}: then nothing "
                "measured describes the incumbent, and the control reuse in "
                "step 2 rests on exactly that identity",
                "the two protocols see different calibration_tokens: a masking "
                "defect, not a numerical result",
                "executed - padded != calibration_tokens for either protocol: "
                "the loop's counters and the collector's mask disagree",
                "either protocol fails to reproduce its OWN digest across "
                "rounds in one session",
            ],
            "outcomes": {
                "digests_identical": (
                    "NUMERICALLY EXACT FOR THIS CASE. No behavioural "
                    "experiment is owed at all; adoption turns on the measured "
                    "runtime alone, which is a maintainer judgement. The "
                    "scientific and the materialization identity may then both "
                    "be shared -- see `identity_semantics.adoption_logic`."),
                "digests_differ": (
                    "step 2 may be PROPOSED, as defined below -- AND A-bsz3 "
                    "becomes a distinct numerical materialization protocol "
                    "that may not enter D1/D2/D3 execution until the "
                    "repository binds the execution fingerprint to "
                    "materialization/resume identity. Those are two separate "
                    "consequences and passing step 2 clears only the first."),
                "comparison_void": (
                    "repair and re-run; a void comparison is not a result and "
                    "must not be reported as 'no difference found'"),
            },
            "runtime_threshold": {
                "none": True,
                "_withdrawn": (
                    "the 1.25x bar came from the causal-KL packing pilot -- a "
                    "different workload with a different forward count -- and "
                    "the maintainer withdrew it for this study on 2026-10-01."),
                "_how_it_is_judged_instead": (
                    "report the measured speedup and its spread. Its "
                    "operational value is judged against the volume of "
                    "attention-calibration work expected in D1/D2/D3, which "
                    "is not yet designed. Note the absolute scale: even an "
                    "unattainable 3x saves about 7.5 s per calibration."),
            },
        },

        "step_2": {
            "status": "CONDITIONAL on step 1, NOT FUNDED, NOT AUTHORIZED",
            "trigger": "step 1 found the artifact digests DIFFER",
            "what": ("an engineering behavioural sanity check: at most THREE "
                     "newly trained A-bsz3 treatment probes, compared against "
                     "attempt75's existing A controls."),
            "_not_sixteen_probes": (
                "8 seeds x 2 arms is withdrawn. Only treatment probes are "
                "trained; the control arm already exists."),
            "treatment": {
                "arm": "A_bsz3",
                "initialization": (
                    "the A-bsz3 ATTENTION step from the frozen shared parent "
                    f"{identities['shared_parent_artifact_digest'][:12]}, "
                    "digest-gated against the digest step 1 recorded"),
                "_why_that_gate": (
                    "step 1 characterized one A-bsz3 initialization. If a "
                    "later session rebuilds a different one, the behavioural "
                    "result does not describe what step 1 measured -- and a "
                    "protocol that cannot reproduce its own initialization "
                    "across sessions is not adoptable whatever its behaviour."),
                "_not_preserved_from_step_1": (
                    "P8.4: step 2 pays the parent replay regardless, so "
                    "rebuilding the initialization costs ~0.25 min while "
                    "preserving it would move 1.19 GB through a relay that is "
                    "already short of private-storage quota. Artifacts follow "
                    "consumers."),
                "seeds_in_fail_fast_order": seeds,
                "max_new_probes": 3,
            },
            "control": controls,
            "fail_fast": {
                "order": seeds,
                "after_seed_1": stop,
                "then": ("if no material or catastrophic regression is "
                         "observed, OPTIONALLY complete the other two seeds in "
                         "the SAME session"),
                "_one_session_not_two": (
                    "the stop decision happens on the pod. Splitting the "
                    "shapes into two sessions would pay a second setup and a "
                    "second 21.78-minute parent replay for nothing."),
                "_on_pod_inputs": (
                    "the seed-matched control's correct_overall and "
                    "usable_rollout_rate, plus the guardrail inputs. The "
                    "paired bootstrap runs OFF POD at $0 afterwards; it does "
                    "not need to be on the meter."),
            },
            "protocol_identity_requirements": {
                "_why_this_section_is_strict": (
                    "reusing controls measured in another session is the whole "
                    "saving, and it is sound only while the two fields are one "
                    "field. C2's confirmation set carried THREE generation "
                    "protocol fingerprints and produced no canonical verdict "
                    "for exactly this reason."),
                "must_equal_attempt75": {
                    "battery": probe_results["battery"]["artifact"],
                    "battery_manifest_sha256":
                        probe_results["battery"]["manifest_sha256"],
                    "battery_content_sha256":
                        probe_results["battery"]["content_sha256"],
                    "scoring_contract":
                        probe_results["scoring_contract"]["contract"],
                    "scoring_contract_digest":
                        probe_results["scoring_contract"]["digest"],
                    "generation_protocol_fingerprint":
                        probe_results["observed_generation_fingerprint"],
                    "evaluation_protocol_hash":
                        probe_results["observed_evaluation_protocol_hash"],
                    "recovery_recipe": prereg["recovery"]["recipe"],
                    "recovery_tokens": prereg["recovery"]["tokens"],
                    "n_prompts": probe_results["decision_inputs_audit"]["n_prompts"],
                    "n_scorable": probe_results["decision_inputs_audit"]["n_scorable"],
                    "scorable_prompt_set": (
                        "identical prompt ids and strata; "
                        "`decision_inputs` refuses a field that differs"),
                },
                "if_any_differs": (
                    "the comparison is VOID, not reinterpreted. A fingerprint "
                    "mismatch is the confound that killed C2's verdict, and "
                    "discovering it after training is still cheaper than "
                    "publishing it."),
                "_check_it_before_training": (
                    "the fingerprint is a property of the generation "
                    "configuration, which is known at session start. Verify it "
                    "against attempt75's value BEFORE the first probe trains, "
                    "not after it is scored."),
            },
            "reported_quantities": {
                "primary": ("Delta correct_overall = A_bsz3 - A_bsz1_control, "
                            "paired at prompt level, per seed and pooled"),
                "interval": (
                    "the same stratified prompt-cluster bootstrap C3 used, "
                    "reported as a DESCRIPTIVE interval. It is not a "
                    "non-inferiority test and must not be written as one."),
                "behaviour": ("usable_rollout delta, pooled and per seed, "
                              "WITH every component rate -- non_empty, "
                              "natural_termination, no_severe_repetition, "
                              "no_context_limit, protocol_valid"),
                "also": ["per-seed deltas", "McNemar counts per seed",
                         "per-set and per-domain diagnostics",
                         "the runtime result from step 1, restated beside it"],
                "sesoi_role": (
                    f"the programme's {SESOI} correctness SESOI is reported as "
                    "the SCALE of a material loss. It is not a confidence "
                    "criterion here and no extra seed may be invented to clear "
                    "a bound."),
            },
            "safety_checks_reused_from_c3": {
                "pooled_usable_delta_min": prereg["guardrails"]["pooled_usable_delta_min"],
                "per_seed_usable_delta_min": prereg["guardrails"]["per_seed_usable_delta_min"],
                "catastrophic_capability_veto": {
                    "rule": ("for each primary stratum: candidate usable rate "
                             "< 0.10 while control usable rate > 0.40"),
                    "candidate_max": 0.10, "control_min": 0.40,
                    "implementation":
                        "scripts/experiments/phase_c1/probe_results.py :: decision_inputs",
                    "asymmetry": "can veto the treatment, never the control",
                },
                "protocol_validity_veto": "as C3, unchanged",
                "_role": "veto only; never positive credit",
            },
            "implementation_trap_to_avoid": {
                "what": ("`C1ProbeRecord.allowed_arms` defaults to C1's two "
                         "ROLES, ['incumbent', 'treatment']. attempt75 passed "
                         "C3's three ARM IDS and stage I raised "
                         "`unknown arm 'A_incumbent'` AFTER all nine probes "
                         "were trained and scored."),
                "apply": ("pass this study's arm vocabulary explicitly at "
                          "every layer that has an arm-shaped default, and "
                          "enumerate those defaults in ONE pass before the "
                          "session rather than one per paid attempt."),
            },
            "terminal": (
                "an engineering adoption recommendation that pairs the "
                "measured runtime saving with the measured behavioural delta "
                "and the guardrail results. The adoption decision itself is "
                "the maintainer's."),
            "_what_passing_step_2_does_NOT_authorize": (
                "entry into D1/D2/D3 execution. If the digests differed, "
                "A-bsz3 is a distinct numerical materialization protocol, and "
                "the identity binding in `identity_semantics.adoption_logic` "
                "is a separate precondition that a behavioural result cannot "
                "satisfy."),
        },

        "claim_boundary": {
            "what_this_evidence_is": (
                "engineering evidence about an execution knob, conditional on "
                "the three frozen C3 recovery seeds and the frozen "
                "confirmation battery."),
            "the_price_of_the_shortened_design": (
                "treatment and control are measured in DIFFERENT SESSIONS on "
                "different physical hardware. C3 trained all nine of its "
                "probes in one session precisely to avoid that. Session is "
                "therefore an unquantified alternative explanation for any "
                "difference step 2 finds, and it is a stronger reason than "
                "power for refusing a formal equivalence claim."),
            "what_partly_bounds_it": (
                "step 1 re-derives the incumbent INITIALIZATION on the new "
                "hardware and gates it against "
                f"{identities['incumbent_artifact_digest'][:12]}. If that "
                "passes, the unverified session effect is confined to "
                "training and evaluation; the initialization half is "
                "demonstrably reproduced."),
            "alternative_the_maintainer_may_prefer": {
                "design": ("2 A-bsz3 treatment probes + 1 A-bsz1 control "
                           "BRIDGE probe retrained in the new session"),
                "buys": ("a measurement of the session effect at one seed, by "
                         "differencing the bridge against attempt75's control "
                         "at the same seed"),
                "costs": ("one seed of treatment evidence; same 3-probe cap, "
                          "same price"),
                "_recommendation": (
                    "3 treatments, as instructed. The bridge is recorded "
                    "because the choice is the maintainer's and the tradeoff "
                    "is real, not because the instruction is in doubt."),
            },
        },

        "pricing": {
            "_owner": "logs/stages/stage-1/phase_c3/plans/a_bsz3_pricing.json",
            "_derived_by": "scripts/experiments/phase_c3/a_bsz3_pricing.py",
            "gpu_rate_usd_per_hour": pricing["gpu_rate_usd_per_hour"],
            "_rate_note": rate_note,
            "shapes": {
                name: {
                    "expected_usd": row["price"]["expected"]["usd"],
                    "hard_usd": row["price"]["hard_ceiling"]["usd"],
                    "book": row["book"],
                    "FUNDABLE": row["FUNDABLE"],
                    "shortfalls_usd": row["shortfalls_usd"],
                } for name, row in pricing["shapes"].items()},
            "worst_case_both_steps_usd":
                pricing["worst_case_both_steps"]["usd"],
            "_versus_the_withdrawn_design": (
                "the 16-probe step 2 was estimated at ~$30. Three probes in "
                "one session are priced at a "
                f"${pricing['shapes']['step2_three_seeds']['price']['hard_ceiling']['usd']} "
                "hard ceiling from measured components."),
        },

        "funding_status": {
            "FUNDED": False,
            "step_1": (
                "fits the GPU engineering allowance, which is its correct "
                "book: it trains no probe and consumes no battery."),
            "step_2": (
                "does NOT fit. It trains recovery probes and scores them on "
                "the frozen confirmation battery, and the execution package "
                "states that the engineering allowance authorizes neither. "
                "Its book is the $55.00 formal allowance, which the exact "
                "derivation shows OVERSPENT by $0.9492."),
            "_therefore": (
                "step 1 needs a grant; step 2 needs a maintainer budget "
                "decision before it can be priced into a chain at all. "
                "Remaining balance anywhere is not permission."),
        },

        "authorizes": "nothing",
        "_authorizes_nothing": (
            "no provider resource, no probe, no battery consumption, no "
            "change to the shared parent's pinned micro_batch_size: 1, and no "
            "adoption on the structural result alone when the digests differ"),
    }
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    doc = build(rate_note=("the historical L40S securePrice. Re-quote live "
                           "immediately before any authorization."))
    #: Self-hash over the document without the field, the convention every
    #: other plan in this directory uses.
    doc["design_sha256"] = __import__("hashlib").sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    stop = doc["step_2"]["fail_fast"]["after_seed_1"]
    print(f"A-bsz3 adoption design")
    print(f"  controls reused      {len(doc['step_2']['control']['probes'])} "
          f"from {doc['step_2']['control']['source_run']}")
    print(f"  new probes, at most  {doc['step_2']['treatment']['max_new_probes']}")
    print(f"  seed-1 stop at       {stop['threshold_delta']}  "
          f"(from observed max |per-seed delta| "
          f"{stop['derivation']['observed_max_abs_per_seed_delta']})")
    for name, row in doc["pricing"]["shapes"].items():
        print(f"  {name:20s} hard ${row['hard_usd']:7.4f}  {row['book']:26s} "
              f"{'FUNDABLE' if row['FUNDABLE'] else 'NOT FUNDABLE'}")
    print(f"  design_sha256        {doc['design_sha256']}")

    if args.write:
        (REPO / args.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
