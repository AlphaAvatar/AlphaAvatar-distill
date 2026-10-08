"""The 2026-10-08 information-architecture migration, as data.

This module is the single declaration of what moves where. The mover below
executes it; `--pairs` emits the old->new table that extends
`logs/index.json :: historical_paths.map`; the migration record under
`logs/maintenance/source-relocations/info-architecture/v1/` is derived from
the same tables. Frozen records are never rewritten from here — path pairs
exist so a reader can follow a frozen record's old path forward.

Run from the repository root:

    python scripts/maintenance/migration/info_architecture.py --check
    python scripts/maintenance/migration/info_architecture.py --apply scripts
    python scripts/maintenance/migration/info_architecture.py --apply logs
    python scripts/maintenance/migration/info_architecture.py --apply artifacts
    python scripts/maintenance/migration/info_architecture.py --pairs
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------
# Scripts tree: whole-directory moves (git mv), applied in order.
# --------------------------------------------------------------------------
SCRIPT_DIR_MOVES: tuple[tuple[str, str], ...] = (
    # The experiment tree keeps its stage-first shape under a tree named for
    # what it holds; the family level is new.
    ("scripts/experiments/stage-1/phase_d_series",
     "scripts/stages/stage-1/families/d_series"),
    ("scripts/experiments/stage-1", "scripts/stages/stage-1"),
    ("scripts/experiments/stage-3", "scripts/stages/stage-3"),
    # The shared application-layer test suite.
    ("scripts/experiments/tests", "scripts/shared/tests"),
    # Repository tooling.
    ("scripts/architecture", "scripts/maintenance/architecture"),
    ("scripts/consolidate", "scripts/maintenance/consolidation"),
    ("scripts/rollout", "scripts/shared/rollout"),
)

#: The shared application layer: scripts/experiments/<mod>.py -> scripts/shared/<mod>.py
SHARED_APP_MODULES: tuple[str, ...] = (
    "calibration", "datasets", "deployment", "durable_stores",
    "epsilon_response", "historical_declarations", "micro_preflight",
    "operator_ledger", "preflight", "recipes", "recovery_policy",
    "run_layout", "search_cost_model", "source_sets",
)

#: Stage-1 experiment packages (import name is unchanged except the root).
STAGE1_PACKAGES: tuple[str, ...] = (
    "phase_a", "phase_a3", "phase_b", "phase_c1", "phase_c2", "phase_c3",
    "phase_d1", "measurement", "recovery_continuation",
)

# --------------------------------------------------------------------------
# Per-file moves: old path -> destination directory. Basenames are preserved.
# --------------------------------------------------------------------------
S1 = "scripts/stages/stage-1"
S3 = "scripts/stages/stage-3"
SH = "scripts/shared"
MA = "scripts/maintenance"

_AUTOINIT: dict[str, str] = {
    "aggregate_a3": f"{S1}/phase_a3",
    "aggregate_c3_stage_i": f"{S1}/phase_c3",
    "attest_protocol": f"{SH}/pod",
    "audit_recovery_search_v2": f"{SH}/data",
    "audit_skip_predicates": f"{S1}/phase_c1",
    "audit_tool_rendering": f"{SH}/pod",
    "audit_tool_scoring": f"{SH}/evaluation",
    "c2_replay_config_forensic": f"{S1}/phase_c2_replay",
    "characterize_thresholds": f"{S1}/phase_a",
    "compare_a_bsz3": f"{S1}/phase_a3",
    "compare_recovery_fingerprints": f"{S1}/recovery_continuation",
    "dry_run_search": f"{SH}/validation",
    "emit_protocol_compat_v2": f"{S1}/phase_a",
    "freeze_c2_comparison_inputs": f"{S1}/phase_c2",
    "historical_reuse_position": f"{S1}/phase_b",
    "issue_a3_authorization": f"{S1}/phase_a3",
    "issue_authorization": f"{SH}/pod",
    "issue_c1_authorization": f"{S1}/phase_c1",
    "issue_c2_authorization": f"{S1}/phase_c2",
    "issue_c2_baseline_completion_authorization": f"{S1}/phase_c2_baseline_completion",
    "issue_c2_behavioural_authorization": f"{S1}/phase_c2_behavioural",
    "issue_c2_full_search_authorization": f"{S1}/phase_c2_full_search",
    "issue_c2_replay_authorization": f"{S1}/phase_c2_replay",
    "issue_c3_authorization": f"{S1}/phase_c3",
    "issue_continuation_authorization": f"{S1}/recovery_continuation",
    "issue_continuation_b_authorization": f"{S1}/continuation_b",
    "issue_d1_authorization": f"{S1}/phase_d1",
    "issue_d1_replay_authorization": f"{S1}/phase_d1",
    "issue_measurement_authorization": f"{S1}/measurement",
    "issue_phase_a_authorization": f"{S1}/phase_a",
    "issue_phase_b_authorization": f"{S1}/phase_b",
    "issue_recovery_continuation_authorization": f"{S1}/recovery_continuation",
    "load_state_eval": f"{SH}/evaluation",
    "measure_causal_depth_runtime": f"{S1}/measurement",
    "measure_state_repeatability": f"{S1}/phase_a",
    "phase_a_frozen": f"{S1}/phase_a",
    "phase_a_search": f"{S1}/phase_a",
    "plan_search": f"{S1}/phase_a",
    "price_behavioural_continuation": f"{S1}/continuation_b",
    "price_c1": f"{S1}/phase_c1",
    "price_c2": f"{S1}/phase_c2",
    "price_c2_baseline_completion": f"{S1}/phase_c2_baseline_completion",
    "price_c2_behavioural_continuation": f"{S1}/phase_c2_behavioural",
    "price_phase_b": f"{S1}/phase_b",
    "probe_peak_memory": f"{S1}/phase_a",
    "profile_statistics_pass": f"{S1}/phase_a",
    "publish_selected_leaves": f"{SH}/rollout",
    "recompute_c2_behavioural_decision": f"{S1}/phase_c2_behavioural",
    "recompute_continuation_rung2": f"{S1}/continuation_b",
    "record_phase_b_historical_amendment": f"{S1}/phase_b",
    "record_phase_b_post_freeze": f"{S1}/phase_b",
    "record_pod_environment": f"{SH}/pod",
    "renderer_parity_gate": f"{SH}/validation",
    "reprice_d1_from_qualification": f"{S1}/phase_d1",
    "retire_relay_copies": f"{MA}/consolidation",
    "score_c1_confirmation": f"{S1}/phase_c1",
    "score_c2_screening": f"{S1}/phase_c2",
    "score_d1_screening": f"{S1}/phase_d1",
    "score_recovery_search": f"{SH}/evaluation",
    "stage_c1_bundle": f"{S1}/phase_c1",
    "stage_c2_behavioural_bundle": f"{S1}/phase_c2_behavioural",
    "stage_c2_bundle": f"{S1}/phase_c2",
    "stage_c2_full_search_bundle": f"{S1}/phase_c2_full_search",
    "stage_c2_probes_to_volume": f"{SH}/pod",
    "stage_c2_replay_bundle": f"{S1}/phase_c2_replay",
    "stage_d1_bundle": f"{S1}/phase_d1",
    "validate_recovery_scoring": f"{SH}/validation",
    "verify_attempt4_probe_reuse": f"{S1}/continuation_b",
    "verify_attempt5_probe_reuse": f"{S1}/continuation_b",
    "verify_c1_battery_isolation": f"{S1}/phase_c1",
    "verify_c1_scoring_equivalence": f"{S1}/phase_c1",
    "verify_control_checkpoints": f"{S1}/recovery_continuation",
    "verify_depth_backend_equivalence": f"{S1}/measurement",
    "verify_frozen_assets": f"{SH}/pod",
    "verify_historical_probe_reuse": f"{S1}/phase_b",
    "write_a3_attempt_closeouts": f"{S1}/phase_a3",
    "write_a3_design": f"{S1}/phase_a3",
    "write_c1_execution_preregistration": f"{S1}/phase_c1",
    "write_c2_behavioural_proposal": f"{S1}/phase_c2_behavioural",
    "write_c2_full_search_grant_proposal": f"{S1}/phase_c2_full_search",
    "write_c2_full_search_plan": f"{S1}/phase_c2_full_search",
    "write_c2_measured_optimization": f"{S1}/phase_c2",
    "write_continuation_b_preregistration": f"{S1}/continuation_b",
    "write_d1_design": f"{S1}/phase_d1",
    "write_d1_launch_readiness": f"{S1}/phase_d1",
    "write_phase_b_preregistration": f"{S1}/phase_b",
    "write_preregistration": f"{S1}/phase_a",
}

_POD: dict[str, str] = {
    "AGENTS.md": f"{SH}/pod",
    "a3_acquire.sh": f"{S1}/phase_a3",
    "autoinit_a3_driver.py": f"{S1}/phase_a3",
    "autoinit_a3_launch.py": f"{S1}/phase_a3",
    "autoinit_c1_driver.py": f"{S1}/phase_c1",
    "autoinit_c1_launch.py": f"{S1}/phase_c1",
    "autoinit_c2_behavioural_driver.py": f"{S1}/phase_c2_behavioural",
    "autoinit_c2_behavioural_launch.py": f"{S1}/phase_c2_behavioural",
    "autoinit_c2_replay_driver.py": f"{S1}/phase_c2_replay",
    "autoinit_c2_replay_launch.py": f"{S1}/phase_c2_replay",
    "autoinit_c3_driver.py": f"{S1}/phase_c3",
    "autoinit_c3_launch.py": f"{S1}/phase_c3",
    "autoinit_continuation_b_driver.py": f"{S1}/continuation_b",
    "autoinit_continuation_b_launch.py": f"{S1}/continuation_b",
    "autoinit_continuation_driver.py": f"{S1}/recovery_continuation",
    "autoinit_continuation_launch.py": f"{S1}/recovery_continuation",
    "autoinit_d1_behavioural_driver.py": f"{S1}/phase_d1",
    "autoinit_d1_driver.py": f"{S1}/phase_d1",
    "autoinit_d1_launch.py": f"{S1}/phase_d1",
    "autoinit_d1_replay_driver.py": f"{S1}/phase_d1",
    "autoinit_d1_replay_launch.py": f"{S1}/phase_d1",
    "autoinit_device_canary.py": f"{SH}/pod",
    "autoinit_device_canary_launch.py": f"{SH}/pod",
    "autoinit_engine_probe.py": f"{SH}/pod",
    "autoinit_measurement_launch.py": f"{S1}/measurement",
    "autoinit_phase_a_driver.py": f"{S1}/phase_a",
    "autoinit_phase_a_launch.py": f"{S1}/phase_a",
    "autoinit_phase_b_driver.py": f"{S1}/phase_b",
    "autoinit_phase_b_launch.py": f"{S1}/phase_b",
    "autoinit_phase_c2_baseline_driver.py": f"{S1}/phase_c2_baseline_completion",
    "autoinit_phase_c2_baseline_launch.py": f"{S1}/phase_c2_baseline_completion",
    "autoinit_phase_c2_driver.py": f"{S1}/phase_c2",
    "autoinit_phase_c2_full_search_driver.py": f"{S1}/phase_c2_full_search",
    "autoinit_phase_c2_full_search_launch.py": f"{S1}/phase_c2_full_search",
    "autoinit_phase_c2_launch.py": f"{S1}/phase_c2",
    "autoinit_preflight_driver.py": f"{SH}/pod",
    "autoinit_preflight_launch.py": f"{SH}/pod",
    "autoinit_preflight_setup.sh": f"{SH}/pod",
    "autoinit_recovery_continuation_driver.py": f"{S1}/recovery_continuation",
    "autoinit_recovery_continuation_launch.py": f"{S1}/recovery_continuation",
    "autoinit_science_inputs.py": f"{SH}/pod",
    "batch_invariance_diagnostic_launch.sh": f"{SH}/validation",
    "batch_invariance_diagnostic_remote.sh": f"{SH}/validation",
    "batching_refactor_cuda_launch.sh": f"{S1}/phase_c3",
    "benchmark_padding_truncation.py": f"{SH}/pod",
    "build_wheelhouse.py": f"{SH}/pod",
    "c3_acquire.sh": f"{S1}/phase_c3",
    "c3_batching_pilot_driver.py": f"{S1}/phase_c3",
    "c3_batching_pilot_launch.sh": f"{S1}/phase_c3",
    "c3_batching_pilot_remote.sh": f"{S1}/phase_c3",
    "c3_packing_pilot_driver.py": f"{S1}/phase_c3",
    "c3_packing_pilot_launch.sh": f"{S1}/phase_c3",
    "c3_packing_pilot_remote.sh": f"{S1}/phase_c3",
    "c3_packing_screen.py": f"{S1}/phase_c3",
    "c3_packing_screen_v2.py": f"{S1}/phase_c3",
    "c3_packing_v2_driver.py": f"{S1}/phase_c3",
    "c3_packing_v2_launch.sh": f"{S1}/phase_c3",
    "c3_packing_v2_remote.sh": f"{S1}/phase_c3",
    "canary.py": f"{SH}/pod",
    "checkpoint_inventory.py": f"{SH}/pod",
    "collect_artifacts.py": f"{SH}/pod",
    "cpu_test_env_args.py": f"{SH}/pod",
    "d0diag_driver.py": f"{S3}/d0",
    "d0diag_launch.sh": f"{S3}/d0",
    "d0diag_setup.sh": f"{S3}/d0",
    "d1_qualification_closeout.py": f"{S1}/phase_d1",
    "d1_qualification_driver.py": f"{S1}/phase_d1",
    "d1_qualification_launch.sh": f"{S1}/phase_d1",
    "d1_qualification_remote.sh": f"{S1}/phase_d1",
    "d1_qualification_report.py": f"{S1}/phase_d1",
    "d1_replay_acquire.sh": f"{S1}/phase_d1",
    "d1_replay_run.sh.template": f"{S1}/phase_d1",
    "e2diag_driver.py": f"{S3}/e2",
    "e2diag_launch.sh": f"{S3}/e2",
    "e2diag_setup.sh": f"{S3}/e2",
    "e2p1_driver.py": f"{S3}/e2",
    "e2p1_launch.sh": f"{S3}/e2",
    "e2p1_setup.sh": f"{S3}/e2",
    "e3_driver.py": f"{S3}/e3",
    "e3_launch.sh": f"{S3}/e3",
    "e3_setup.sh": f"{S3}/e3",
    "e4_driver.py": f"{S3}/e4",
    "e4_launch.sh": f"{S3}/e4",
    "e4_setup.sh": f"{S3}/e4",
    "e5_driver.py": f"{S3}/e5",
    "e5_launch.sh": f"{S3}/e5",
    "e5_pilot.py": f"{S3}/e5",
    "e5_setup.sh": f"{S3}/e5",
    "e6_driver.py": f"{S3}/e6",
    "e6_launch.sh": f"{S3}/e6",
    "e6_setup.sh": f"{S3}/e6",
    "e6_stage_checkpoints.py": f"{S3}/e6",
    "e6b_driver.py": f"{S3}/e6b",
    "e6b_launch.sh": f"{S3}/e6b",
    "e6b_setup.sh": f"{S3}/e6b",
    "e7_driver.py": f"{S3}/e7",
    "e7_launch.py": f"{S3}/e7",
    "e7_setup.sh": f"{S3}/e7",
    "e8a_driver.py": f"{S3}/e8",
    "e8a_launch.py": f"{S3}/e8",
    "e8a_setup.sh": f"{S3}/e8",
    "e8b_driver.py": f"{S3}/e8b",
    "e8b_launch.py": f"{S3}/e8b",
    "e8b_setup.sh": f"{S3}/e8b",
    "engineering_campaign_budget.py": f"{SH}/pod",
    "hashes_ckpt.txt": f"{SH}/pod",
    "hashes_ckpt_pca.txt": f"{SH}/pod",
    "hashes_ckpt_rand.txt": f"{SH}/pod",
    "hashes_ladder.txt": f"{SH}/pod",
    "hashes_transfer.txt": f"{SH}/pod",
    "orchestrate.sh": f"{SH}/pod",
    "p0asst_driver.py": f"{S3}/p0",
    "p0asst_launch.sh": f"{S3}/p0",
    "p0asst_setup.sh": f"{S3}/p0",
    "p2_driver.py": f"{S3}/p2",
    "p2_launch.sh": f"{S3}/p2",
    "p2_setup.sh": f"{S3}/p2",
    "post_run.sh": f"{SH}/pod",
    "reconstruct_training_events.py": f"{SH}/pod",
    "register_p0_real.py": f"{S3}/p0",
    "retain_checkpoints.py": f"{SH}/pod",
    "run_env.sh": f"{SH}/pod",
    "score_refs.sh": f"{SH}/evaluation",
    "setup.sh": f"{SH}/pod",
    "simulate_pod_env.sh": f"{SH}/pod",
    "start_job.py": f"{SH}/pod",
    "summarize_pytest_outcomes.py": f"{SH}/pod",
    "throughput_gate.py": f"{SH}/pod",
    "topk_adoption_driver.py": f"{S1}/phase_d1",
    "topk_adoption_launch.sh": f"{S1}/phase_d1",
    "topk_adoption_remote.sh": f"{S1}/phase_d1",
    "topk_adoption_report.py": f"{S1}/phase_d1",
    "train.sh": f"{SH}/pod",
    "verify_and_report.py": f"{SH}/pod",
    "watchdog.py": f"{SH}/pod",
}

_DATA: dict[str, str] = {
    "audit_d1_corpus": f"{S3}/e2",
    "audit_e1_mixture_rebuild": f"{S3}/e1",
    "audit_render_and_masks": f"{SH}/data",
    "audit_selection_rule": f"{S3}/e2",
    "battery_render": f"{SH}/data",
    "build_c1_confirmation_battery": f"{S1}/phase_c1",
    "build_c2_screening_battery": f"{S1}/phase_c2",
    "build_capability_battery": f"{S3}/e2",
    "build_cleaned_corpus": f"{SH}/data",
    "build_control_kd": f"{S3}/e7",
    "build_e5_arm_c": f"{S3}/e5",
    "build_e5_arm_r": f"{S3}/e5",
    "build_e8_calibration": f"{S3}/e8",
    "build_eval_behavior_v0": f"{SH}/data",
    "build_experiment1_configs": f"{S3}/e1",
    "build_experiment2_configs": f"{S3}/e2",
    "build_fineweb_kd": f"{S3}/e7",
    "build_holdout_v1": f"{SH}/data",
    "build_matched_rung": f"{SH}/data",
    "build_reasoning_heavy_calibration": f"{SH}/data",
    "build_recovery_search_battery": f"{SH}/data",
    "build_recovery_search_v2": f"{SH}/data",
    "build_stage2_v0": f"{SH}/data",
    "build_stage2_v1": f"{SH}/data",
    "build_stage3_pilot": f"{SH}/data",
    "build_state_eval_suite": f"{SH}/data",
    "build_token_ladder": f"{SH}/data",
    "build_warmup_v1": f"{SH}/data",
    "check_autoinit_role_isolation": f"{SH}/data",
    "check_e8_calibration_leakage": f"{S3}/e8",
    "check_stream_disjointness": f"{S3}/e7",
    "fetch_fineweb_docs": f"{SH}/data",
    "stage_e7_streams": f"{S3}/e7",
    "stage_e8_inputs": f"{S3}/e8",
    "validate_corpus_gate": f"{SH}/data",
    "verify_staged_r": f"{S3}/e1",
}

_EVALUATION: dict[str, str] = {
    "analyze_e3": f"{S3}/e3",
    "analyze_e4": f"{S3}/e4",
    "analyze_e6": f"{S3}/e6",
    "analyze_e6b": f"{S3}/e6b",
    "analyze_e7": f"{S3}/e7",
    "analyze_e8": f"{S3}/e8",
    "analyze_e8b_behaviour": f"{S3}/e8b",
    "audit_degeneration_replay": f"{SH}/evaluation",
    "audit_prompt_rendering": f"{SH}/evaluation",
    "audit_teacher_forced_by_role": f"{SH}/evaluation",
    "build_test_cases": f"{S3}/e1",
    "compare_geometry": f"{SH}/evaluation",
    "consolidate_e1": f"{S3}/e1",
    "diagnose_training_recall": f"{SH}/evaluation",
    "eval_behavior": f"{SH}/evaluation",
    "eval_general_text": f"{SH}/evaluation",
    "eval_ppl": f"{SH}/evaluation",
    "exposure_report": f"{S3}/e1",
    "measure_init_nll": f"{SH}/evaluation",
    "parameter_movement": f"{SH}/evaluation",
    "plot_e1_scaling": f"{S3}/e1",
    "plot_perf_trend": f"{SH}/evaluation",
    "probe_think_close": f"{SH}/evaluation",
    "reevaluate_stage23": f"{SH}/evaluation",
    "register_e6": f"{S3}/e6",
    "rescore_gsm8k": f"{SH}/evaluation",
    "rescore_with_template_state": f"{SH}/evaluation",
    "run_three_mode_diagnostic": f"{S3}/d0",
    "score_battery": f"{SH}/evaluation",
    "summarize_three_mode": f"{S3}/d0",
    "transition_table": f"{SH}/evaluation",
    "uncapped_eval": f"{SH}/evaluation",
    "unrestricted_pilot": f"{SH}/evaluation",
}

_TRAINING: dict[str, str] = {
    "analyze_e8b": f"{S3}/e8b",
    "audit_kd_decomposition": f"{SH}/training",
    "audit_stream_shapes": f"{SH}/training",
    "benchmark_e5_throughput": f"{S3}/e5",
    "build_and_stage_e8_init": f"{S3}/e8",
    "build_depth_only_init": f"{SH}/training",
    "build_e6b_configs": f"{S3}/e6b",
    "build_e7_configs": f"{S3}/e7",
    "build_e8_configs": f"{S3}/e8",
    "build_e8b_configs": f"{S3}/e8b",
    "collect_stage0": f"{SH}/training",
    "diagnose_e5_gradients": f"{S3}/e5",
    "diagnose_e5_normalization": f"{S3}/e5",
    "diagnose_loss_weights": f"{SH}/training",
    "e7_preflight": f"{S3}/e7",
    "init_stage1": f"{SH}/training",
    "plan_e7_budget": f"{S3}/e7",
    "plan_e8_budget": f"{S3}/e8",
    "plan_e8b_budget": f"{S3}/e8b",
    "preflight_e4": f"{S3}/e4",
    "profile_dp_memory": f"{S3}/e8b",
    "register_e3": f"{S3}/e3",
    "register_e4": f"{S3}/e4",
    "register_e6b": f"{S3}/e6b",
    "replay_lifecycle": f"{SH}/training",
    "reprice_e8b_after_gate": f"{S3}/e8b",
    "search_depth_map": f"{SH}/training",
    "size_e8b_memory": f"{S3}/e8b",
    "train_stage3": f"{SH}/training",
    "validate_e3_arms": f"{S3}/e3",
    "validate_e6b_arms": f"{S3}/e6b",
    "validate_e7_arms": f"{S3}/e7",
    "validate_e8_arms": f"{S3}/e8",
    "validate_e8b_arms": f"{S3}/e8b",
    "validate_padding_truncation": f"{SH}/training",
}

_VALIDATION: dict[str, str] = {
    "batch_invariance_diagnostic": f"{SH}/validation",
    "batch_invariance_finding": f"{SH}/validation",
    "batching_refactor_cuda_check": f"{S1}/phase_c3",
    "c2_full_search_cuda_check": f"{S1}/c2_full_search_cuda",
    "c2_full_search_performance_check": f"{S1}/c2_full_search_perf",
    "c2_state_eval_certification_check": f"{S1}/c2_state_eval_cert",
    "cuda_engineering_check": f"{SH}/validation",
    "cuda_engineering_launch": f"{SH}/validation",
    "device_observations": f"{SH}/validation",
    "parallel_item_forward_diagnostic": f"{SH}/validation",
}


def _expand(src_dir: str, table: dict[str, str], ext: str = ".py") -> dict[str, str]:
    out = {}
    for name, dest in table.items():
        base = name if "." in name else name + ext
        out[f"{src_dir}/{base}"] = f"{dest}/{base}"
    return out


def script_file_moves() -> dict[str, str]:
    moves: dict[str, str] = {}
    for mod in SHARED_APP_MODULES:
        moves[f"scripts/experiments/{mod}.py"] = f"scripts/shared/{mod}.py"
    moves["scripts/experiments/conftest.py"] = "scripts/conftest.py"
    moves["scripts/experiments/__init__.py"] = "scripts/stages/__init__.py"
    moves.update(_expand("scripts/autoinit", _AUTOINIT))
    moves.update(_expand("scripts/pod", _POD))
    moves.update(_expand("scripts/data", _DATA))
    moves.update(_expand("scripts/evaluation", _EVALUATION))
    moves.update(_expand("scripts/training", _TRAINING))
    moves.update(_expand("scripts/validation", _VALIDATION))
    return moves


# --------------------------------------------------------------------------
# Logs: individual record moves (git mv). Frozen payloads are untouched; these
# records are the D-series family material the stage index cleared to move,
# plus misfiled per-experiment and maintenance records.
# --------------------------------------------------------------------------
DS_LOGS = "logs/stages/stage-1/families/d_series"
LOG_MOVES: dict[str, str] = {
    "logs/shared/analyses/autoinit_d_series_battery_family.json":
        f"{DS_LOGS}/analyses/autoinit_d_series_battery_family.json",
    "logs/shared/analyses/autoinit_d_series_family_manifest.json":
        f"{DS_LOGS}/analyses/autoinit_d_series_family_manifest.json",
    "logs/shared/analyses/autoinit_d_series_source_evidence.json":
        f"{DS_LOGS}/analyses/autoinit_d_series_source_evidence.json",
    "logs/shared/analyses/autoinit_control_sb_packaging_repair.json":
        "logs/stages/stage-1/recovery_continuation/analyses/autoinit_control_sb_packaging_repair.json",
    "logs/shared/analyses/autoinit_causal_depth_pricing_bound.json":
        "logs/stages/stage-1/measurement/analyses/autoinit_causal_depth_pricing_bound.json",
    "logs/shared/analyses/autoinit_attempt5_retention_verification.json":
        "logs/maintenance/cleanup/autoinit_attempt5_retention_verification.json",
    "logs/shared/analyses/autoinit_relay_retention_20260822.json":
        "logs/maintenance/cleanup/autoinit_relay_retention_20260822.json",
    "logs/shared/analyses/autoinit_pilot_proposal.md":
        "logs/stages/stage-1/phase_a/analyses/autoinit_pilot_proposal.md",
}

# --------------------------------------------------------------------------
# Artifacts: local-only bytes (nothing under artifacts/ is git-tracked).
# Ordered: subtree extractions first, then the stage re-roots.
# --------------------------------------------------------------------------
A1 = "artifacts/stages/stage-1"
A3_ = "artifacts/stages/stage-3"
ARTIFACT_MOVES: tuple[tuple[str, str], ...] = (
    ("artifacts/stage3/d_series_behavioural_v1",
     f"{A1}/families/d_series/batteries/d_series_behavioural_v1"),
    ("artifacts/stage3/c1_confirmation_v1", f"{A1}/phase_c1/batteries/c1_confirmation_v1"),
    ("artifacts/stage3/c2_screening_v1", f"{A1}/phase_c2/batteries/c2_screening_v1"),
    ("artifacts/stage3/recovery_search_v1", f"{A1}/batteries/recovery_search_v1"),
    ("artifacts/stage3/recovery_search_v2", f"{A1}/batteries/recovery_search_v2"),
    ("artifacts/stage3/ladder_uniform_probe", "artifacts/shared/instruments/ladder_uniform_probe"),
    ("artifacts/qualification/d1_gpu", f"{A1}/phase_d1/qualification/d1_gpu"),
    ("artifacts/qualification/topk_adoption", f"{A1}/phase_d1/qualification/topk_adoption"),
    ("artifacts/pilots", f"{A1}/phase_c3/pilots"),
    ("artifacts/pilot", f"{A3_}/pilot"),
    ("artifacts/eval", f"{A3_}/eval"),
    ("artifacts/validation", "artifacts/shared/validation"),
    ("artifacts/autoinit/dryrun", "artifacts/shared/validation/dryrun"),
    ("artifacts/stage0", "artifacts/stages/stage-0"),
    ("artifacts/stage1", f"{A1}"),
    ("artifacts/stage2", "artifacts/stages/stage-2/v0"),
    ("artifacts/stage2_v2", "artifacts/stages/stage-2/v2"),
    ("artifacts/stage3", f"{A3_}"),
    # artifacts/audit stays: the runtime session-audit namespace (see README).
)

# Out-of-repo: the D1 finalists leave the scratch store for the durable one.
# Recorded here for the migration record; applied and hash-verified by hand.
OUT_OF_REPO_MOVES: tuple[tuple[str, str], ...] = (
    ("/home/ecs-user/aad-scratch/d1_search_20261006_210210",
     "/home/ecs-user/aad-artifacts/phase_d1/d1_search_20261006_210210"),
    ("/home/ecs-user/aad-scratch/d1_replay_002",
     "/home/ecs-user/aad-artifacts/phase_d1/d1_replay_002"),
)

# --------------------------------------------------------------------------
# Import renames (module namespace), applied to live Python only.
# --------------------------------------------------------------------------
IMPORT_RENAMES: tuple[tuple[str, str], ...] = (
    ("experiments.phase_d_series", "stages.d_series"),
) + tuple(
    (f"experiments.{p}", f"stages.{p}") for p in STAGE1_PACKAGES
) + tuple(
    (f"experiments.{m}", f"shared.{m}") for m in SHARED_APP_MODULES
)

#: Files whose frozen declaration tuples must stay byte-identical; the blanket
#: path rewriter skips them entirely and they are edited surgically.
FROZEN_DECLARATION_FILES: tuple[str, ...] = (
    "scripts/stages/stage-1/phase_a/plan.py",
    "scripts/stages/stage-1/phase_b/plan.py",
    "scripts/stages/stage-1/phase_b/continuation.py",
    "scripts/stages/stage-1/recovery_continuation/plan.py",
    "scripts/stages/stage-1/phase_c1/authorization.py",
    "scripts/stages/stage-1/phase_c2/session.py",
    "scripts/stages/stage-1/phase_a3/a3_authorization.py",
    "scripts/stages/stage-1/phase_d1/issue_d1_replay_authorization.py",
    "scripts/stages/stage-1/measurement/issue_measurement_authorization.py",
    "scripts/shared/source_sets.py",
    "configs/experiments/phase_a/source_sets.json",
)


def all_pairs() -> dict[str, str]:
    """Every old->new pair this migration creates, for historical_paths.map."""
    pairs: dict[str, str] = {}
    pairs.update(script_file_moves())
    for old, new in SCRIPT_DIR_MOVES:
        pairs[old] = new
    pairs.update(LOG_MOVES)
    for old, new in ARTIFACT_MOVES:
        pairs[old] = new
    return pairs


def _git_mv(old: Path, new: Path) -> None:
    new.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(old), str(new)], cwd=REPO, check=True)


def _apply_scripts() -> None:
    # Directory moves FIRST: a per-file move may create a destination directory
    # that a later whole-directory move would then nest itself inside.
    for rel_old, rel_new in SCRIPT_DIR_MOVES:
        old = REPO / rel_old
        if old.exists():
            _git_mv(old, REPO / rel_new)
    for rel_old, rel_new in script_file_moves().items():
        old = REPO / rel_old
        if old.exists():
            _git_mv(old, REPO / rel_new)


def _apply_logs() -> None:
    for rel_old, rel_new in LOG_MOVES.items():
        _git_mv(REPO / rel_old, REPO / rel_new)


def _apply_artifacts() -> None:
    for rel_old, rel_new in ARTIFACT_MOVES:
        old, new = REPO / rel_old, REPO / rel_new
        if not old.exists():
            print(f"skip (absent): {rel_old}")
            continue
        if new.exists():
            # A re-root whose destination was already created by an earlier
            # subtree extraction MERGES, entry by entry; a same-named entry on
            # both sides is a refusal, never an overwrite.
            if not (old.is_dir() and new.is_dir()):
                raise SystemExit(f"refusing: destination exists: {rel_new}")
            for entry in sorted(old.iterdir()):
                dest = new / entry.name
                if dest.exists():
                    raise SystemExit(f"refusing: collision: {dest}")
                shutil.move(str(entry), str(dest))
            old.rmdir()
            print(f"merged: {rel_old} -> {rel_new}")
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        print(f"moved: {rel_old} -> {rel_new}")


def _check() -> int:
    bad = 0
    for rel in list(script_file_moves()) + [o for o, _ in SCRIPT_DIR_MOVES] \
            + list(LOG_MOVES):
        if not (REPO / rel).exists():
            print(f"missing source: {rel}")
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", choices=["scripts", "logs", "artifacts"])
    ap.add_argument("--pairs", action="store_true")
    args = ap.parse_args()
    if args.check:
        bad = _check()
        print(f"{'FAIL' if bad else 'OK'}: {bad} missing sources")
        return 1 if bad else 0
    if args.pairs:
        for old, new in sorted(all_pairs().items()):
            print(f"{old} -> {new}")
        return 0
    if args.apply == "scripts":
        _apply_scripts()
    elif args.apply == "logs":
        _apply_logs()
    elif args.apply == "artifacts":
        _apply_artifacts()
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
