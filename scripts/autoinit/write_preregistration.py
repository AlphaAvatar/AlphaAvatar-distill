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
from aadistill.autoinit.ranking import PARETO_V1, SCHEDULE_V1  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    E1_KD_HEAVY_0860K,
    POOLED_COUNTS_V1,
    SEED_SA,
    SEED_SB,
    SEED_SC,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_phase_a_preregistration.json")
    ap.add_argument("--state-eval", default="artifacts/stage1/state_eval_v1")
    ap.add_argument("--recovery-search", default="artifacts/stage3/recovery_search_v1")
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
        feasibility_min=-1.0,          # placeholder; see selection_rules below
        equivalence_interval=thresholds["recovery_thresholds"][
            "equivalence_interval"]["interval_at_prior"],
        aggregation=POOLED_COUNTS_V1,
        survivor_rule=("rung 1: exclude searched leaves below the feasibility floor, "
                       "then take the top 2 by correct_overall; the canonical "
                       "control advances unconditionally and consumes no slot"),
        winner_rule=("final: pooled-count aggregate over sa and sb; among finalists "
                     "clearing the feasibility floor, top 1 by correct_overall. The "
                     "canonical control is eligible to win. Finalists inside the "
                     "equivalence interval go to seed sc."),
        battery_asset_id="recovery_search_v1")

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
            "seed_aggregation": POOLED_COUNTS_V1.as_dict(),
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
                "feasibility_floor_rule": thresholds["recovery_thresholds"][
                    "feasibility_floor"],
                "primary_metric": "correct_overall over SCORABLE prompts",
                "secondary_metric": "correct_given_usable over SCORABLE prompts",
                "equivalence_interval": plan.equivalence_interval,
                "equivalence_interval_rule": thresholds["recovery_thresholds"][
                    "equivalence_interval"],
                "catastrophic_floor_rule": thresholds["recovery_thresholds"][
                    "catastrophic_per_capability_floor"],
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
        "pending_before_launch": [
            "canonical control usable_rollout_rate on recovery_search_v1",
            "canonical control correct_overall on recovery_search_v1",
            "canonical control per-set usable rates",
            "GPU state-evaluator repeatability (confirms or resets beam epsilon)",
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
