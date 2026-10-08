#!/usr/bin/env python3
"""The A3 design: ONE end-to-end experiment, derived from what already exists.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_a3/write_a3_design.py --write

**What A3 is.** The incumbent ATTENTION operator under B3's batching protocol,
run as a single chain:

    initialization -> recovery training -> evaluation -> aggregation -> closeout

**What changed on 2026-10-01.** The maintainer merged the step-1/step-2 split.
A3 answers the practical question once, without stopping for intermediate
approvals. The structural and runtime diagnostics are still collected — all of
them — but they are DIAGNOSTICS INSIDE the experiment, not gates in front of
it. In particular **a differing A-bsz3 artifact digest is a finding, not a
stop**: the chain proceeds into recovery unless an actual integrity failure
makes the experiment invalid.

**Why this is a writer and not a hand-typed document.** The numbers the design
turns on are measurements that live elsewhere: the controls are attempt75's,
the identities are the C3 preregistration's, the price is the live pricer's,
and the seed-level noise band is derived from the spread C3 itself observed.
Typing any of them here would create a second copy to keep in step with the
first, and this repository has already had a plan assert a bootstrap seed that
the computation did not use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from stages.phase_a3.a_bsz3 import (  # noqa: E402
    ATTENTION_IMPL_ID, execution_comparison, frozen_identities,
    item_token_counts,
)

PREREG = REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"
STAGE_I = REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i"
DECISION = STAGE_I / "c3_decision.json"
PROBES = STAGE_I / "c3_probe_results.json"
PRICING = REPO / "logs/stages/stage-1/phase_c3/plans/a3_pricing.json"
OUT = "logs/stages/stage-1/phase_c3/plans/a3_design.json"

#: The designs this one replaces. Both were superseded before producing a
#: single behavioural datum, and both are kept as records rather than deleted:
#: a withdrawal that leaves no trace reads later as a design that never
#: happened.
SUPERSEDED = [
    {"plan": "logs/stages/stage-1/phase_c3/plans/a_bsz3_noninferiority.json",
     "design": "8 seeds x 2 arms = 16 probes, ADOPT iff one-sided LCB > -0.010",
     "withdrawn_utc": "2026-10-01",
     "reason": ("SCOPE. A-bsz3 is an execution-optimization validation and the "
                "programme never needed a population-level non-inferiority "
                "claim about a batching knob.")},
    {"plan": "logs/stages/stage-1/phase_c3/plans/a_bsz3_adoption.json",
     "design": ("a cost-ordered split: one structural/runtime session, then a "
                "CONDITIONAL behavioural study of at most three probes with a "
                "-0.030 seed-1 fail-fast"),
     "withdrawn_utc": "2026-10-01",
     "reason": ("the split required an intermediate approval between the two "
                "halves and could terminate on a diagnostic. The maintainer "
                "wants the complete three-seed result from one chain, so the "
                "digest comparison became a finding and the fail-fast stopped "
                "being a normal experimental stop.")},
]

#: The programme's smallest correctness effect worth having. Reported as the
#: SCALE of a material loss. It is not a decision rule here.
SESOI = 0.010

#: --- identity: the distinction A3 records and does not resolve -----------
IDENTITY_SEMANTICS = {
    "_the_distinction": (
        "scientific/semantic identity asks whether two executions pose the "
        "same operator-level question. Materialization/numerical identity asks "
        "whether two executions may be treated as the same resumable, "
        "deduplicable checkpoint. A-bsz1 and A-bsz3 share the first by "
        "construction; whether they share the second is a thing A3 MEASURES."),
    "what_compute_state_id_binds": [
        "root teacher", "target spec",
        "operator implementation id and signature hash",
        "calibration profile hash", "operator config hash", "seed",
    ],
    "what_it_does_NOT_bind": ["ExecutionConfig", "the resulting artifact digest"],
    "_why_that_exclusion_is_correct": (
        "otherwise two runs of the same science at different batch sizes would "
        "be different SCIENTIFIC states, a resume would not find its own "
        "journal, and whether A-bsz1 and A-bsz3 agree would be settled by "
        "definition instead of measured."),
    "where_the_hazard_lives": (
        "BeamSearch._restore looks a state up in StateStore.latest_by_state_id "
        "by state_id, then re-identifies the checkpoint on disk and refuses if "
        "the bytes disagree with the record. That guard catches a stale or "
        "tampered checkpoint; it cannot catch a record that is internally "
        "consistent and was produced by a different numerical protocol."),
    "what_a_differing_digest_means_for_A3": (
        "NOTHING PROCEDURAL. It is recorded as a finding and the chain "
        "continues into recovery. A3 is not stopped to build a "
        "materialization-identity framework, and none may be built "
        "speculatively."),
    "what_it_means_afterwards": (
        "if A-bsz3 produces a different artifact and its downstream behaviour "
        "is acceptable, the closeout records it as a DISTINCT NUMERICAL "
        "MATERIALIZATION PROTOCOL. That label carries exactly one forward "
        "obligation: before A-bsz3 is used in D1/D2/D3 -- that is, before it "
        "enters a RESUMABLE BEAM SEARCH -- the repository must bind the "
        "numerical execution fingerprint to materialization/resume identity, "
        "so two artifacts differing in bytes cannot share one resume key. "
        "A3's behavioural result does not discharge that obligation, because "
        "it is an engineering correctness property rather than a behavioural "
        "one."),
    "_forbidden_resolution": (
        "do NOT invent `attention.activation_importance_bsz3`. The operator "
        "semantics are unchanged, a second impl id would fork the scientific "
        "identity to repair a materialization problem, and it would answer the "
        "equivalence question by definition."),
    "_required_shape_if_it_is_ever_needed": (
        "the SMALLEST generic separation between semantic/scientific path "
        "identity and the numerical/materialization identity used for artifact "
        "ownership, resume and deduplication. Generic, not A-bsz3 specific: "
        "any execution knob that can move bytes has this property. NOT BUILT, "
        "and if A3 finds the digests identical it is never needed."),
    "regression_that_holds_today": (
        "tests/initialization/test_corrections.py :: "
        "test_resume_refuses_a_record_from_a_different_numerical_protocol -- a "
        "matching semantic state id is not sufficient to reuse a checkpoint, "
        "driven through the real resume path and mutation-checked."),
}

#: --- the only reasons A3 stops ------------------------------------------
INTEGRITY_STOPS = [
    {"condition": "the frozen shared parent does not reproduce its digest",
     "why": ("every arm is defined relative to that parent; a different parent "
             "is a different experiment"),
     "action": "repair and re-run; never interpret as a result"},
    {"condition": ("A_bsz1 does not reproduce the frozen incumbent digest "
                   "53e30566c5f7"),
     "why": ("A-bsz1 IS canonical A. If it does not rebuild the incumbent then "
             "the controls A3 reuses do not describe what this session built, "
             "and the reuse rests on exactly that identity"),
     "action": "repair and re-run"},
    {"condition": ("an initialization is not reproducible WITHIN its own "
                   "declared numerical protocol -- the same protocol producing "
                   "two different digests across rounds in one session"),
     "why": ("a protocol that is not repeatable against itself cannot be "
             "compared with another one. This is NOT the same as A-bsz1 and "
             "A-bsz3 differing from each other, which is the finding"),
     "action": "repair and re-run"},
    {"condition": ("the two protocols accumulate different valid token counts, "
                   "or executed - padded != calibration_tokens"),
     "why": ("padded positions are masked out of the accumulator, so this is a "
             "masking defect and not a numerical effect"),
     "action": "repair and re-run"},
    {"condition": ("the recovery recipe, battery, scoring contract or "
                   "generation protocol does not match the attempt75 controls"),
     "why": ("the comparison pools two fields as one. C2's confirmation set "
             "carried THREE generation fingerprints and produced no canonical "
             "verdict for exactly this reason"),
     "action": ("repair before the first scientific probe trains -- the "
                "fingerprint is a property of the generation configuration and "
                "is knowable at session start")},
    {"condition": "corrupt or unreadable calibration, corpus or battery data",
     "why": "a measurement over corrupt input is not a measurement",
     "action": "repair and re-run"},
]

#: Everything that is NOT a stop, stated so a future session does not invent
#: one. Each of these ends the chain only if it also trips an integrity stop.
NOT_STOPS = [
    "the A-bsz3 artifact digest differing from A-bsz1's -- that is the finding",
    "a kept-head map that differs, however widely",
    "a low rank correlation or a large score drift",
    "a measured speedup smaller than hoped, or below 1.0",
    "a seed-1 correctness delta that looks bad -- see seed_level_noise",
    "any point estimate at any seed",
    "a usable-rollout component rate moving",
]


def _load(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing {what}: {path}")
    return json.loads(path.read_text())


def controls_from_attempt75(probe_results: dict[str, Any],
                            run_id: str) -> dict[str, Any]:
    """attempt75's three A_incumbent probes, as the reusable control arm.

    Reused rather than retrained, on the maintainer's instruction: they are
    valid, protocol-uniform and fully scored at the same frozen seeds. Their
    WEIGHTS were retired on 2026-10-01 and that is deliberate and sufficient --
    a control contributes its per-sample evidence to this comparison, never its
    bytes.
    """
    rows = []
    for p in probe_results["probes"]:
        if p["arm"] != "A_incumbent":
            continue
        rows.append({
            "probe_id": p["probe_id"], "seed": p["seed"],
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
        "arm": "A_incumbent", "source_run": run_id, "probes": rows,
        "evidence_root": "/home/ecs-user/aad-artifacts/phase_c3/attempt75",
        "retrained": False,
        "_weights_were_retired": (
            "logs/maintenance/inventories/archival_retirement_20261001.json -- "
            "nine objects, 19.984 GiB, measured reclaim 12-13 GiB -> 32-33 "
            "GiB. A3 consumes these controls' per-sample rows and scores, not "
            "their weights, so the retirement removes nothing this comparison "
            "reads."),
        "_what_they_are_not": (
            "a control arm measured in the same session as the treatment. See "
            "claim_boundary."),
    }


def seed_level_noise(decision: dict[str, Any]) -> dict[str, Any]:
    """How far a single seed moves on this instrument when nothing differs.

    DERIVED, and no longer a stop. The 2026-10-01 design used this to set a
    -0.030 seed-1 fail-fast; the 2026-10-01 decision withdrew that, because A3
    wants the complete three-seed result and a point estimate at one seed is
    not evidence that anything is wrong. The number survives as the band a
    reader should hold a per-seed delta against.
    """
    per_seed = [abs(x) for c in decision["contrasts"].values()
                for x in c["per_seed_delta"]]
    pooled = [abs(c["delta"]) for c in decision["contrasts"].values()]
    observed_max = max(per_seed)
    return {
        "observed_max_abs_per_seed_delta": round(observed_max, 6),
        "observed_abs_pooled_deltas": [round(p, 6) for p in pooled],
        "_what_this_says": (
            f"across all three C3 contrasts -- arms whose POOLED differences "
            f"were within {max(pooled):.6f} of zero -- per-seed deltas still "
            f"ranged over {observed_max:.6f}. A single seed moving by that "
            "much is the instrument, not the protocol."),
        "reporting_reference_only": True,
        "_not_a_stop": (
            "A3 trains all three seeds. No point estimate at any seed ends the "
            "chain; only the integrity stops do."),
        "_not_an_equivalence_margin": (
            "it bounds nothing and no interval, power or equivalence statement "
            "follows from it."),
        "_source": ("logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i/"
                    "c3_decision.json, all three contrasts"),
    }


def build(*, rate_note: str) -> dict[str, Any]:
    prereg = _load(PREREG, "the C3 preregistration")
    decision = _load(DECISION, "attempt75's stage-I decision")
    probe_results = _load(PROBES, "attempt75's probe results")
    pricing = _load(PRICING, "the A3 pricing")
    #: Both DERIVED from the pricing module, never restated. The hard ceiling
    #: comes out of the record the module wrote; the restart cost comes from
    #: the same component table, over the components preceding the first
    #: scientific probe.
    from stages.phase_a3.a3_pricing import pre_science_restart_usd

    hard_ceiling = float(pricing["price"]["hard_ceiling"]["usd"])
    restart = pre_science_restart_usd(
        float(pricing["price"]["gpu_rate_usd_per_hour"]))
    identities = frozen_identities()
    zero_cost = execution_comparison(item_token_counts())

    run_id = decision.get("run_id")
    if not run_id:
        raise SystemExit(f"{DECISION} names no run_id")
    controls = controls_from_attempt75(probe_results, run_id)
    seeds = identities["recovery_seeds"]
    battery = probe_results["battery"]
    contract = probe_results["scoring_contract"]
    audit = probe_results["decision_inputs_audit"]

    return {
        "schema": "aadistill.a3.design/v1",
        "stage": "A3",
        "designed_utc": "2026-10-01",
        "_contract": (
            "The COMPLETE A3 experiment: the incumbent ATTENTION operator "
            "under calibration_forward_batch_size=3 and length_sorted_v1 "
            "packing, run end to end as ONE chain. Frozen before any A3 "
            "structural or behavioural result exists. AUTHORIZES NO SPEND BY "
            "ITSELF -- the money is the 2026-10-01 amendment's."),
        "supersedes": SUPERSEDED,

        "_the_question": (
            "Does bsz3 + length_sorted_v1 materially reduce the execution cost "
            "of this operator's calibration without producing a material "
            "downstream regression? Answered ONCE, over three seeds, without "
            "intermediate approvals."),
        "_what_may_not_be_claimed_FROM_it": [
            "a formal population-level non-inferiority proof",
            "that A-bsz3 and A-bsz1 are statistically equivalent",
            "an operator promotion -- A-bsz3 is not a new operator",
            "anything about causal-KL, C4, or the B1-vs-B3 protocol choice",
            "that a differing artifact may be reused under the same state id",
        ],
        "_what_it_IS": (
            "the complete A3 engineering/scientific comparison, conditional on "
            "three frozen seeds and one frozen battery."),

        "chain": ["initialization", "recovery_training", "evaluation",
                  "aggregation", "closeout"],
        "operator": ATTENTION_IMPL_ID,
        "execution": {"calibration_forward_batch_size": 3,
                      "calibration_batch_packing": "length_sorted_v1"},
        "_not_a_new_operator": (
            "both knobs are ExecutionConfig fields; neither enters a hash, a "
            "state id or a manifest identity. No bsz3-specific operator id "
            "exists or may be created, and nothing is imported from "
            "causal_kl."),
        "identities": identities,
        "identity_semantics": IDENTITY_SEMANTICS,

        "initialization": {
            "parent": ("the exact frozen shared parent "
                       f"{identities['shared_parent_artifact_digest'][:12]}, "
                       "replayed under its digest gate"),
            "protocols": ["A_bsz1 (the incumbent reproduction)", "A_bsz3"],
            "rounds_per_protocol": 4,
            "interleaved": True,
            "warm_up_rounds_excluded": 1,
            "_why_interleaved_with_a_warm_up": (
                "the operator measures 11.2732 s on this parent geometry. A "
                "single sample of an 11-second workload run after the other "
                "protocol reports the arms' ORDER as much as the protocols, "
                "because the first pays for allocator growth, autotuning and "
                "kernel selection."),
            "diagnostics_collected": [
                "A-bsz1 incumbent reproduction and its digest gate",
                "A-bsz3 artifact digest",
                "kept-head map, per protocol, and the diff",
                "per-head scores",
                "Spearman rank correlation, overall and per layer",
                "score drift and relative score drift distributions",
                "selection-boundary margins, and the margin at each flipped group",
                "valid / padded / executed positions, observed as the operator ran",
                "scorer runtime and end-to-end suffix runtime, with spread",
                "peak VRAM",
            ],
            "_these_are_diagnostics_not_gates": (
                "they are recorded INSIDE the experiment. A differing A-bsz3 "
                "artifact digest is an experimental FINDING and the chain "
                "proceeds directly into recovery. Only an integrity failure "
                "stops it."),
            "driver": "scripts/stages/stage-1/phase_a3/compare_a_bsz3.py :: structural_half",
        },

        "recovery": {
            "treatment_arm": "A_bsz3",
            "probes": 3,
            "seeds": seeds,
            "_seeds_are": "the frozen C3 recovery seeds, unchanged",
            "initialization_source": (
                "round 0 of the A_bsz3 protocol IS the initialization the "
                "probes train from, so nothing is materialized twice"),
            "recipe": prereg["recovery"]["recipe"],
            "tokens": prereg["recovery"]["tokens"],
            "controls_are_not_retrained": True,
            "_why_not": (
                "attempt75's three A_incumbent probes are valid, "
                "protocol-uniform and fully scored at these same seeds. "
                "Retraining them would spend three probes to reproduce "
                "evidence that already exists."),
            "all_three_complete": (
                "verify the protocol identities BEFORE the first scientific "
                "probe, then train all three. No point estimate interrupts "
                "the sequence."),
        },
        "control": controls,

        "protocol_identity_requirements": {
            "_why_this_is_strict": (
                "reusing controls measured in another session is the whole "
                "saving, and it is sound only while the two fields are ONE "
                "field."),
            "verified_before_the_first_scientific_probe": True,
            "must_equal_attempt75": {
                "battery": battery["artifact"],
                "battery_manifest_sha256": battery["manifest_sha256"],
                "battery_content_sha256": battery["content_sha256"],
                "scoring_contract": contract["contract"],
                "scoring_contract_digest": contract["digest"],
                "generation_protocol_fingerprint":
                    probe_results["observed_generation_fingerprint"],
                "evaluation_protocol_hash":
                    probe_results["observed_evaluation_protocol_hash"],
                "recovery_recipe": prereg["recovery"]["recipe"],
                "recovery_tokens": prereg["recovery"]["tokens"],
                "n_prompts": audit["n_prompts"],
                "n_scorable": audit["n_scorable"],
                "scorable_prompt_set": (
                    "identical prompt ids and strata; `decision_inputs` "
                    "refuses a field that differs"),
            },
            "if_any_differs": "an integrity stop: repair, do not interpret",
        },

        "integrity_stops": INTEGRITY_STOPS,
        "not_stops": NOT_STOPS,
        "seed_level_noise": seed_level_noise(decision),

        "aggregation_and_reporting": {
            "_one_comparison": (
                "the three new A-bsz3 probes against the three matched "
                "attempt75 A controls, paired at prompt level by seed"),
            "report_at_minimum": [
                "pooled and per-seed correct_overall, both arms",
                "paired prompt-level deltas",
                "descriptive stratified prompt-cluster bootstrap interval",
                "McNemar counts, per seed",
                "usable_rollout_rate and ALL component rates -- non_empty, "
                "natural_termination, no_severe_repetition, no_context_limit, "
                "protocol_valid",
                "capability, domain and set breakdowns",
                "structural head-map differences",
                "initialization score drift and rank correlation",
                "initialization/runtime speedup and its spread",
                "peak VRAM",
                "artifact identities, every arm and probe",
                "exact consumed input hashes",
                "total GPU time and actual cost",
            ],
            "interval_is_descriptive": (
                "the same bootstrap machinery C3 used, reported as a "
                "DESCRIPTIVE interval. It is not a non-inferiority test and "
                "must not be written as one."),
            "sesoi_role": (
                f"the programme's {SESOI} correctness SESOI is reported as the "
                "SCALE of a material loss, not as a decision rule."),
            "safety_checks_reused_from_c3": {
                "pooled_usable_delta_min":
                    prereg["guardrails"]["pooled_usable_delta_min"],
                "per_seed_usable_delta_min":
                    prereg["guardrails"]["per_seed_usable_delta_min"],
                "catastrophic_capability_veto": {
                    "rule": ("per primary stratum: candidate usable rate < "
                             "0.10 while control usable rate > 0.40"),
                    "candidate_max": 0.10, "control_min": 0.40,
                    "implementation": ("scripts/stages/stage-1/phase_c1/"
                                       "probe_results.py :: decision_inputs"),
                },
                "_role": ("reported as safety observations. They do not gate "
                          "the chain's completion."),
            },
        },

        "closeout_states": {
            "digests_identical": (
                "A-bsz3 is a TRANSPARENT EXECUTION OPTIMIZATION for this case: "
                "one scientific and one materialization identity, and adoption "
                "turns on the measured runtime."),
            "digests_differ_behaviour_acceptable": (
                "record A-bsz3 as a DISTINCT NUMERICAL MATERIALIZATION "
                "PROTOCOL. Adoptable for standalone initialization on the "
                "runtime evidence; NOT adoptable into D1/D2/D3 resumable beam "
                "search until the identity binding exists."),
            "digests_differ_behaviour_materially_worse": (
                "A-bsz3 is not adopted. The runtime saving does not buy a "
                "correctness regression, and the measured delta is the "
                "record."),
            "_the_closeout_states_the_adoption_question": (
                "A3 ends with one complete closeout naming the incumbent and "
                "the adoption question as it then stands. The FFN experiment "
                "and the D-series wait for independent review."),
        },

        "claim_boundary": {
            "what_this_evidence_is": (
                "the complete A3 engineering/scientific comparison, "
                "conditional on the three frozen C3 recovery seeds and the "
                "frozen confirmation battery."),
            "the_cross_session_limitation": (
                "treatment and control are measured in DIFFERENT SESSIONS on "
                "different physical hardware. C3 trained all nine of its "
                "probes in one session precisely to avoid that. Session is an "
                "unquantified alternative explanation for any difference A3 "
                "finds, and it must remain explicit in every report."),
            "what_partly_bounds_it": (
                "A3 re-derives the incumbent INITIALIZATION on the new "
                "hardware and gates it against "
                f"{identities['incumbent_artifact_digest'][:12]}. If that "
                "passes, the unverified session effect is confined to training "
                "and evaluation."),
            "not_a_formal_proof": (
                "three seeds cannot support a population-level "
                "non-inferiority conclusion and A3 does not claim one."),
        },

        "pricing": {
            "_owner": "logs/stages/stage-1/phase_c3/plans/a3_pricing.json",
            "_derived_by": "scripts/stages/stage-1/phase_a3/a3_pricing.py",
            "gpu_rate_usd_per_hour": pricing["queried_rate_usd_per_hour"],
            "_rate_note": rate_note,
            "expected_usd": pricing["price"]["expected"]["usd"],
            "hard_ceiling_usd": pricing["price"]["hard_ceiling"]["usd"],
            "hard_ceiling_minutes": pricing["price"]["hard_ceiling"]["minutes"],
            "container_disk_gb": pricing["price"]["container_disk_gb"],
            "durable_requirement_gib":
                pricing["price"]["storage"]["durable_requirement"]["gib"],
            "book": pricing["book"],
            "FUNDABLE": pricing["FUNDABLE"],
            "limits_checked": pricing["_every_applicable_limit_is_checked"],
        },
        "funding": {
            "amended_utc": "2026-10-02",
            "amendment": ("A PHASE envelope, superseding the 2026-10-01 "
                          "minimal one: formal allowance 65.6523 -> 76.6523; "
                          "package total 75.6523 -> 86.6523; project "
                          "cumulative cap 400.0000 -> 410.0000; engineering "
                          "and the per-session envelope unchanged. Sized to "
                          "finish the STAGE rather than the next flawless "
                          "run."),
            #: DERIVED, both of them. `$1.4506` was a prose literal here and
            #: in the amendment, and the container-disk repair moved the
            #: billed rate while the literal did not.
            "funds": (f"the derived hard ceiling ${hard_ceiling:.4f} plus "
                      f"about six corrected pre-science restarts of "
                      f"${restart:.4f} = ${hard_ceiling + 6 * restart:.4f} on "
                      "the formal book"),
            "pre_science_restart_usd": restart,
            "_restart_is_derived": (
                "from the same component table the chain is, over the "
                "components a chain traverses before the first scientific "
                "probe: setup, the pre-provider gates, the teacher fetch, the "
                "operator registration, the parent replay and the "
                "initialization rounds. An attempt that aborts anywhere in "
                "there produced no measurement to retry."),
            "owner": "logs/budget/decisions.md, 2026-10-02",
            "_historical_overspend_preserved": (
                "attempt75's $0.9492 formal overspend stands as fact and is "
                "not rewritten into compliance."),
            "_a_mid_run_failure_that_cannot_be_funded_is_a_stop": (
                "if the remaining balance cannot fund a corrected attempt plus "
                "teardown, A3 stops and reports rather than amending again."),
        },

        "step_0_zero_cost": {
            "status": "DONE", "cost": "$0",
            "n_items": zero_cost["n_items"],
            "valid_tokens": zero_cost["valid_tokens"],
            "forwards": {
                "A_bsz1": zero_cost["protocols"]["A_bsz1"]["physical_forwards"],
                "A_bsz3": zero_cost["protocols"]["A_bsz3"]["physical_forwards"],
                "ratio": zero_cost["deltas"]["physical_forwards_ratio"]},
            "padding_over_valid": zero_cost["deltas"]["padding_over_valid"],
            "_role": ("the PREDICTION the observed position counters are "
                      "checked against; it decides nothing"),
        },

        "authorizes": "nothing by itself",
        "_what_is_forbidden": [
            "changing arms, seeds, data, the recovery recipe, the battery or "
            "the metric/decision semantics -- any of those is a stop for "
            "maintainer review",
            "retraining the attempt75 controls",
            "creating a bsz3-specific operator id",
            "building a materialization-identity framework during A3",
            "starting D1/D2/D3, the FFN experiment or Stage-0 v2",
            "increasing the approved budget",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    doc = build(rate_note=("the live L40S securePrice at pricing time. "
                           "Re-quote immediately before authorization."))
    doc["design_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    print("A3 design -- ONE end-to-end experiment")
    print(f"  chain              {' -> '.join(doc['chain'])}")
    print(f"  treatment probes   {doc['recovery']['probes']} at {doc['recovery']['seeds']}")
    print(f"  controls reused    {len(doc['control']['probes'])} from "
          f"{doc['control']['source_run']} (not retrained)")
    print(f"  integrity stops    {len(doc['integrity_stops'])}")
    print(f"  explicit non-stops {len(doc['not_stops'])}")
    print(f"  hard ceiling       ${doc['pricing']['hard_ceiling_usd']} "
          f"({doc['pricing']['hard_ceiling_minutes']} min), "
          f"FUNDABLE={doc['pricing']['FUNDABLE']}")
    print(f"  design_sha256      {doc['design_sha256']}")

    if args.write:
        (REPO / args.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
