"""Emit the machine-readable Phase A preregistration. Zero cost.

    PYTHONPATH=src .venv/bin/python scripts/autoinit/write_preregistration.py

Everything the run is committed to, assembled from the live objects rather than
transcribed, so a field cannot drift from what the code will actually do. The
companion prose document is `logs/autoinit_phase_a_preregistration.md`.

Two inputs are deliberately absent and marked `PENDING_MICRO_PREFLIGHT`: the
canonical control's usable-rollout rate and correctness on the recovery-search
battery. The *rules* that consume them are frozen here; the numbers are measured
before any searched candidate is probed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.calibration import DOMAIN_BALANCED_V1, NO_CALIBRATION  # noqa: E402
from aadistill.autoinit.cost import (  # noqa: E402
    L40S_MEASURED,
    activation_stats_bytes,
    branching_estimate,
    checkpoint_bytes,
    price_search,
)
from aadistill.autoinit.operators import V1_IMPLEMENTATIONS, registry_ledger  # noqa: E402
from aadistill.autoinit.ranking import (  # noqa: E402
    EPSILON_RESPONSE_V1,
    PARETO_V1,
    SCHEDULE_V1,
)
from aadistill.autoinit.recovery import (  # noqa: E402
    CAPABILITY_SCHEMA_V1,
    CATASTROPHIC_V1,
    E1_KD_HEAVY_0860K,
    PREFLIGHT_PLAN_V1,
    TRAINER_SOURCE_FILES_V1,
    FeasibilityRule,
    RecoveryProtocolFingerprint,
    recovery_scoring_contract,
    RuntimeEnvironmentFingerprint,
    trainer_source_digest,
    POOLED_COUNTS_V2,
    SEED_SA,
    SEED_SB,
    SEED_SC,
    EquivalenceRule,
    SuccessiveHalvingPlan,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

ADAPTER = get_adapter("qwen3")
TEACHER = ArchSpec.of("qwen3", dict(
    hidden_size=2560, num_hidden_layers=36, intermediate_size=9728,
    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))
TARGET = ArchSpec.of("qwen3", dict(
    hidden_size=1024, num_hidden_layers=28, intermediate_size=3072,
    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
    vocab_size=151936, tie_word_embeddings=True))

CANONICAL_CONTROL = {
    "path": "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
    "single_file_sha256": ("86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc"
                           "952cabd5df2633e54"),
    "num_parameters": 596_049_920,
    "injection": ("by frozen artifact hash via make_control_state; NOT a "
                  "re-executed composite, which would be built from this run's "
                  "calibration statistics rather than the original Stage-0 ones"),
}
PENDING = "PENDING_MICRO_PREFLIGHT"
#: The frozen recovery-search battery's content hash. Pinned here so the
#: supersession statement can assert, mechanically, that the *prompts* did not
#: change when the *scoring* did.
BATTERY_CONTENT_SHA256 = ("a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131e"
                          "acca59479f323")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_phase_a_preregistration.json")
    ap.add_argument("--state-eval", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--recovery-search", default="artifacts/stage3/recovery_search_v2")
    args = ap.parse_args()

    state_eval = json.loads((REPO_ROOT / args.state_eval / "manifest.json").read_text())
    battery = json.loads(
        (REPO_ROOT / args.recovery_search / "manifest.json").read_text())
    isolation = json.loads((REPO_ROOT / "logs/autoinit_role_isolation.json").read_text())
    thresholds = json.loads(
        (REPO_ROOT / "logs/autoinit_threshold_characterization.json").read_text())

    decomposed = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) == 1]
    composite = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) > 1]
    branching = branching_estimate(
        TEACHER, TARGET, ADAPTER, decomposed, n_profiles=1,
        beam_width=SCHEDULE_V1.width, warmup_levels=SCHEDULE_V1.warmup_levels,
        include_composite=composite)
    cost = price_search(
        TEACHER, TARGET, ADAPTER, decomposed,
        calibration_tokens=DOMAIN_BALANCED_V1.token_budget,
        suite_tokens=state_eval["counts"]["total_prediction_positions"],
        seq_len=892, n_profiles=1, beam_width=SCHEDULE_V1.width,
        warmup_levels=SCHEDULE_V1.warmup_levels, hardware=L40S_MEASURED,
        composite=composite)

    plan = SuccessiveHalvingPlan(
        plan_id="autoinit.v1.phase_a", recipe=E1_KD_HEAVY_0860K,
        searched_leaves=5, survivors=2,
        feasibility_min=-1.0,          # PENDING; see selection_rules below
        # Formula frozen, value pending the control characterization. Deliberately
        # NOT pre-filled from the historical prior: a fallback value would be the
        # second definition this rule exists to eliminate.
        equivalence=EquivalenceRule(
            n_pooled=battery["n_scorable_prompts"] * 2),
        feasibility=FeasibilityRule(n_pooled=battery["n_prompts"] * 2),
        catastrophic=CATASTROPHIC_V1,
        capability_schema=CAPABILITY_SCHEMA_V1,
        aggregation=POOLED_COUNTS_V2,
        survivor_rule=("rung 1: exclude searched leaves below the feasibility floor, "
                       "then take the top 2 by correct_overall; the canonical "
                       "control advances unconditionally and consumes no slot"),
        winner_rule=("final: pooled-count aggregate over sa and sb; among finalists "
                     "clearing the feasibility floor, top 1 by correct_overall. The "
                     "canonical control is eligible to win. Finalists inside the "
                     "equivalence interval go to seed sc."),
        battery_asset_id="recovery_search_v2")

    prereg = {
        "schema": "aadistill.autoinit.phase_a_preregistration/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREREGISTRATION DRAFT - NOT AUTHORIZED, NO COMPUTE LAUNCHED",

        "teacher": {
            "model_id": "Qwen/Qwen3-4B-Thinking-2507",
            "revision": "768f209d9ea81521153ed38c47d515654e938aea",
            "spec": TEACHER.as_dict(), "spec_hash": TEACHER.spec_hash,
            "num_parameters": ADAPTER.param_count(TEACHER),
        },
        "target_architecture": {
            "spec": TARGET.as_dict(), "spec_hash": TARGET.spec_hash,
            "num_parameters": ADAPTER.param_count(TARGET),
        },
        "adapter": ADAPTER.identity(),
        "operator_ledger": registry_ledger(),
        "operator_ledger_file_sha256": sha256_file(
            REPO_ROOT / "configs/autoinit/operator_ledger.json"),
        "active_calibration_profile": DOMAIN_BALANCED_V1.as_dict(),
        "no_calibration_sentinel": NO_CALIBRATION.qualified_id,

        "search_space": {
            "formula": "24 orderings x (1+P) DEPTH x P WIDTH x P FFN x 1 ATTENTION",
            "n_profiles": 1,
            "decomposed_paths": branching["complete_paths_unbeamed"],
            "composite_kept_separate": {
                "impl_id": "composite.stage1_sandwich_v0",
                "note": ("reaches the target in one step and does not compose with "
                         "the four structural kinds; it is a searchable leaf, and "
                         "is NOT the canonical control"),
            },
            "invocations_per_field": branching["invocations_per_field"],
            "states_materialized": [branching["states_materialized_min"],
                                    branching["states_materialized_max"]],
            "leaves": [branching["leaves_min"], branching["leaves_max"]],
            "per_level": branching["per_level"],
        },

        "beam": {
            "schedule": SCHEDULE_V1.as_dict(),
            "policy": PARETO_V1.as_dict(),
            "no_first_level_quality_pruning": SCHEDULE_V1.warmup_levels >= 1,
            "nll_status": "diagnostic only; not an objective, not a tie-break key",
            "epsilon_justification": thresholds["beam_epsilon"]["verdict"],
            "epsilon_response_rule": EPSILON_RESPONSE_V1.as_dict(),
        },

        "state_evaluation": {
            "asset": args.state_eval,
            "role": state_eval["role"],
            "suite_id": state_eval["suite_id"], "version": state_eval["version"],
            "content_sha256": state_eval["content_sha256"],
            "manifest_sha256": state_eval["manifest_sha256"],
            "items_sha256": state_eval["outputs"]["items"]["sha256"],
            "domains": state_eval["domains"],
            "critical_tags": state_eval["critical_tags"],
            "n_items": state_eval["counts"]["n_items"],
            "total_prediction_positions":
                state_eval["counts"]["total_prediction_positions"],
            "reference": "original teacher, recomputed per candidate",
            "reference_strategy": "RECOMPUTE",
            "tokenizer_sha256": state_eval["tokenizer"]["tokenizer_sha256"],
            "chat_template_sha256": state_eval["tokenizer"]["chat_template_sha256"],
        },

        "recovery": {
            "plan": plan.as_dict(),
            "seeds": {"sa": SEED_SA, "sb": SEED_SB, "sc_conditional": SEED_SC},
            "seed_aggregation": POOLED_COUNTS_V2.as_dict(),
            "canonical_control": CANONICAL_CONTROL,
            "battery": {
                "asset": args.recovery_search,
                "content_sha256": battery["content_sha256"],
                "manifest_sha256": battery["manifest_sha256"],
                "n_prompts": battery["n_prompts"],
                "n_scorable_prompts": battery["n_scorable_prompts"],
                "scorable_sets": battery["scorable_sets"],
                "behaviour_only_sets": battery["behaviour_only_sets"],
                "scorer_sha256": battery["scorers"]["sha256"],
                "limitation": battery["scorers"]["behaviour_only_note"],
            },
            "selection_rules": {
                "order": ["feasibility constraint", "capability objective",
                          "secondary diagnostic (reported, never reorders)"],
                "feasibility_metric": "usable_rollout_rate over ALL prompts",
                "feasibility_floor": PENDING,
                "feasibility_rule": plan.feasibility.as_dict(),
                "capability_schema": plan.capability_schema.as_dict(),
                "primary_metric": "correct_overall over SCORABLE prompts",
                "secondary_metric": "correct_given_usable over SCORABLE prompts",
                "equivalence_rule": plan.equivalence.as_dict(),
                "equivalence_interval": plan.equivalence.value,
                "catastrophic_capability_rule": plan.catastrophic.as_dict(),
                "correctness_semantics": (
                    "correct = correct IN A USABLE ROLLOUT. A rollout that answers "
                    "correctly and then loops or hits the context limit is not "
                    "counted correct; correct_but_unusable is reported separately "
                    "so the gap is visible."),
                "control_eligible_to_win": True,
                "control_exempt_from_feasibility_gate": True,
                "no_weighted_scalar": True,
            },
        },

        "data_hashes": {
            "operator_calibration": {
                "asset": "artifacts/stage1/e8_calibration_v1",
                "content_sha256": DOMAIN_BALANCED_V1.content_sha256,
                "items_file_sha256": DOMAIN_BALANCED_V1.items_file_sha256,
            },
            "initializer_state_eval": {
                "asset": args.state_eval,
                "content_sha256": state_eval["content_sha256"],
            },
            "recovery_search": {
                "asset": args.recovery_search,
                "content_sha256": battery["content_sha256"],
            },
            "final_promotion": {
                "asset": "artifacts/eval/battery_v2",
                "inclusion_mask_sha256": ("d6e24e0b09da1bcc692b1dc96d8236808d29551a"
                                          "9fc94a47d1d968fd3f73d6ba"),
                "status": "ISOLATED FROM THE ENTIRE SEARCH",
            },
            "recovery_training": {
                "pack": E1_KD_HEAVY_0860K.pack,
                "pack_sha256": E1_KD_HEAVY_0860K.pack_sha256,
                "rung": E1_KD_HEAVY_0860K.tokens,
            },
        },
        "role_isolation": {
            "report": "logs/autoinit_role_isolation.json",
            "report_sha256": isolation["report_sha256"],
            "passed": isolation["passed"], "complete": isolation["complete"],
            "exact_overlaps": len(isolation["exact_overlaps"]),
            "near_duplicate_counts": isolation["near_duplicate_counts"],
        },

        "artifact_identity_rules": {
            "checkpoint_identity": ("artifact digest over every sorted weight shard, "
                                    "the shard index, the config, the architecture "
                                    "signature and the tokenizer"),
            "metrics_bind_to": "artifact_digest",
            "frozen_single_file_hashes": ("still checkable via single_shard_sha256; a "
                                          "sharded rebuild is a different layout, "
                                          "not corruption"),
            "no_inherited_metrics": ("attach_evaluation refuses an evaluation whose "
                                     "artifact digest is not the state's own"),
        },
        "resume_rules": {
            "state_identity": "sha256(root teacher, target spec, ordered steps)",
            "restore_requires": ["artifact re-derived from disk matches the journal",
                                 "evaluation suite_hash matches this run's suite"],
            "note": ("a journal measured under a different suite is not adopted; "
                     "state identity is the path and does not include the suite"),
        },

        "budget": {
            "search_usd_low": round(cost.usd_low, 4),
            "search_usd_high": round(cost.usd_high, 4),
            "search_hours": [round(cost.seconds_low / 3600, 3),
                             round(cost.seconds_high / 3600, 3)],
            "hardware": L40S_MEASURED.as_dict(),
            "note": ("recovery probe cost is priced separately in the pilot "
                     "proposal; these are search-only figures"),
        },
        "storage": {
            "peak_working_gib": round(cost.peak_storage_bytes_working / 2**30, 1),
            "total_written_gib": round(cost.total_bytes_written / 2**30, 1),
            "retained_gib": round(cost.peak_storage_bytes_retained / 2**30, 1),
            "peak_gpu_resident_gib": round(cost.peak_resident_bytes / 2**30, 1),
            "teacher_ckpt_gib": round(checkpoint_bytes(TEACHER, ADAPTER) / 2**30, 2),
            "depth_only_intermediate_gib": round(
                checkpoint_bytes(TEACHER.replace(num_hidden_layers=28), ADAPTER)
                / 2**30, 2),
            "activation_stats_gib": round(activation_stats_bytes(TEACHER) / 2**30, 2),
            "provision_at_least_gib": 150,
        },
        "decision_statuses": {
            "resolved": "one finalist leads by more than the equivalence interval",
            "tie_pending": ("finalists equivalent after sa+sb; seed sc is owed and "
                            "winner is None"),
            "unresolved_equivalence": (
                "finalists still equivalent after sc; winner is None and the result "
                "is 'AutoInitializer v1 did not resolve a unique behavioural "
                "winner'. No fourth seed, and no state-id tie-break is used to "
                "manufacture a winner."),
        },
        "canonical_control_availability": json.loads(
            (REPO_ROOT / "logs/autoinit_control_availability.json").read_text())
            if (REPO_ROOT / "logs/autoinit_control_availability.json").is_file()
            else "NOT VERIFIED",
        "recovery_identity": {
            "model": ("RecoveryProtocolFingerprint (what must be identical) + "
                      "initialization artifact digest + seed = RecoveryProbeIdentity"),
            "protocol_excludes": ["student initialization artifact", "seed"],
            "why": ("initialization is the treatment and the seed is the replicate; "
                    "a protocol identity containing either would make every "
                    "comparable pair of arms differ by construction"),
            "matched_pair_predicate": (
                "both_materialized AND protocol_identical AND same_seed AND "
                "initializations_differ"),
            "trainer_source_digest": {
                **trainer_source_digest(REPO_ROOT),
                "policy": ("material identity is the declared source set, not "
                           "whole-repository git HEAD; repo commit and dirty flag "
                           "are recorded as provenance only, so a docs-only commit "
                           "leaves a control matched"),
                "declared_set": list(TRAINER_SOURCE_FILES_V1),
            },
            "runtime_environment": {
                "fields": ["image_digest", "python_version", "torch_version",
                           "transformers_version", "cuda_runtime",
                           "attention_backend"],
                "dev_box_observation_only": RuntimeEnvironmentFingerprint.observe(
                    image_digest=None).as_dict(),
                "binding_invariant": (
                    "the permanent controls and the later searched probes must "
                    "execute under the same frozen image digest; it is attested at "
                    "preflight Stage 0 and required before a control is trained"),
            },
            "materialization_requirement": {
                "required_fields": list(
                    RecoveryProtocolFingerprint.MATERIALIZATION_REQUIRED),
                "rule": ("a protocol whose required fields are not all non-null "
                         "cannot take part in a MATCHED comparison; "
                         "require_materialized() raises and matched_against() "
                         "returns is_single_variable_comparison: false"),
                "semantics": {
                    "unknown vs unknown": "unverifiable — NOT eligible for MATCHED",
                    "verified X vs verified X": "matched",
                },
                "why": ("None == None is True in Python and is not the claim "
                        "'verified identical'; without this gate a control trained "
                        "under an unrecorded runtime would compare as matched — "
                        "the exact defect that disqualified the historical runs"),
                "status_now": ("runtime_digest is null in this preregistration "
                               "because the image is chosen at pod creation. The "
                               "Phase-A protocol fingerprint is therefore NOT final "
                               "here; Stage 0 materializes it and this "
                               "preregistration must be re-emitted with the "
                               "attested digest before Phase A is authorized."),
                "attested_artifact": "logs/autoinit_phase_a_protocol_attested.json",
                "attested_by": "scripts/autoinit/attest_protocol.py",
                "stage_2_compares_against": ("the Stage-0 attested protocol hash, "
                                             "via RecoveryProbeIdentity."
                                             "require_attested()"),
            },
        },
        "recovery_scoring_contract": {
            **recovery_scoring_contract(REPO_ROOT),
            "validation": json.loads(
                (REPO_ROOT / "logs/autoinit_recovery_scoring_validation.json"
                 ).read_text())
            if (REPO_ROOT / "logs/autoinit_recovery_scoring_validation.json"
                ).is_file() else "NOT VALIDATED",
            "tool_usable_gate": {
                "definition": ("generic usable_rollout AND tool_call_emitted AND "
                               "tool_call_parsed AND tool_name_valid"),
                "meaning": ("structurally executable by an agent runtime, which "
                            "is what a Stage-5 trajectory requires; NOT argument "
                            "correctness"),
                "excluded": {
                    "tool_args_schema_ok": ("the xLAM `required` list is "
                                            "reconstructed from missing defaults, "
                                            "an interpretive step; stays "
                                            "diagnostic by decision"),
                    "tool_call_exact_match": ("that is correctness; folding it "
                                              "into usability would collapse the "
                                              "two axes this battery separates"),
                },
                "multi_call": ("the frozen scorer's own all-calls semantics "
                               "(`parsed` and `tool_name_valid` are all(...) over "
                               "the emitted calls); no second interpretation"),
                "worked_examples": {
                    "malformed JSON tool call": "unusable, incorrect",
                    "undeclared tool name": "unusable, incorrect",
                    "well-formed declared call, wrong arguments": "usable, incorrect",
                    "exact correct invocation": "usable, correct",
                    "tool call on a prompt offering no tools": "protocol invalid",
                },
            },
            "supersession": {
                "superseded": "recovery_search_scoring@v1",
                "when": "2026-08-13, BEFORE any paid measurement",
                "statement": (
                    "The earlier recovery-search scoring contract was superseded "
                    "before any paid measurement or candidate evaluation because "
                    "oracle validation demonstrated that tool-enabled prompts had "
                    "a structural zero in usable_rollout under the old generic "
                    "protocol-validity rule."),
                "classification": (
                    "pre-measurement metric defect correction, NOT an adaptive "
                    "response to experimental results: no control, no candidate "
                    "and no searched leaf had been measured when it was made"),
                "evidence": {
                    "policy": "perfect oracle on all 190 frozen prompts",
                    "tool_usable_rate": {"before": 0.0000, "after": 1.0000},
                    "overall_usable_rate": {"before": 0.8947, "after": 1.0000},
                    "prompts_affected": 20,
                    "capabilities_affected": ["tool"],
                    "of_capabilities_in_catastrophic_gate": 6,
                },
                "prompt_content_unchanged": True,
                "battery_content_sha256": BATTERY_CONTENT_SHA256,
                "history": ("both records are kept; logs/decisions.md carries the "
                            "v1 decision and the supersession as separate dated "
                            "entries"),
            },
        },
        "micro_preflight_plan": PREFLIGHT_PLAN_V1.as_dict(),
        "canonical_control_recipe_audit": (
            lambda r: {"historical_controls_are_recipe_matched":
                       r["historical_controls_are_recipe_matched"],
                       "consequence": r["consequence"],
                       "per_control": {k: {"recipe_matched_control":
                                            v["recipe_matched_control"],
                                            "passes_legacy_lineage_subset":
                                            v["passes_legacy_lineage_subset"],
                                            **v["comparison"]}
                                       for k, v in r["comparisons"].items()},
                       "intended_phase_a_comparison":
                           r.get("intended_phase_a_comparison"),
                       "report_sha256": r["report_sha256"]}
        )(json.loads((REPO_ROOT / "logs/autoinit_recovery_fingerprint_audit.json"
                      ).read_text()))
        if (REPO_ROOT / "logs/autoinit_recovery_fingerprint_audit.json").is_file()
        else "NOT AUDITED",
        "tool_scoring_audit": json.loads(
            (REPO_ROOT / "logs/autoinit_tool_scoring_audit.json").read_text())
            if (REPO_ROOT / "logs/autoinit_tool_scoring_audit.json").is_file()
            else None,
        "pending_before_launch": [
            "ATTEST the runtime at preflight Stage 0 and RE-EMIT this "
            "preregistration with the attested protocol fingerprint -- until then "
            "runtime_digest is null and no comparison here is eligible for MATCHED",
            "RERUN canonical sa/sb at 0.86M under the attested runtime and the "
            "frozen trainer source set -- the historical checkpoints are NOT "
            "recipe-matched (their trainer and runtime identity was never recorded, "
            "and the historical tree was dirty and unreconstructable)",
            "control pooled + per-seed usable_rollout_rate -> materializes the "
            "feasibility floor",
            "control pooled + per-seed correct_overall -> materializes the "
            "equivalence interval",
            "control per-capability usable rates -> the catastrophic rule's "
            "reference values",
            "GPU state-evaluator repeatability -> evaluated by the frozen "
            "conservative response rule (no automatic epsilon re-derivation)",
            "activation-statistics GPU/CPU split (collapses the cost range)",
        ],
    }
    prereg["preregistration_sha256"] = sha256_json(prereg)
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(prereg, indent=2, default=str) + "\n")
    print(json.dumps({
        "decomposed_paths": prereg["search_space"]["decomposed_paths"],
        "states": prereg["search_space"]["states_materialized"],
        "leaves": prereg["search_space"]["leaves"],
        "search_usd": [prereg["budget"]["search_usd_low"],
                       prereg["budget"]["search_usd_high"]],
        "peak_working_gib": prereg["storage"]["peak_working_gib"],
        "state_eval_positions": prereg["state_evaluation"][
            "total_prediction_positions"],
        "pending": len(prereg["pending_before_launch"]),
        "preregistration_sha256": prereg["preregistration_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
