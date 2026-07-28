#!/usr/bin/env bash
# Session definition for a Stage 3 GPU pod session. Single source of truth:
# every other script in scripts/pod/ sources this file instead of hardcoding a
# run name, config, checkpoint or HF path.
#
# Current session: the CE/KD protocol-conflict intervention
# (logs/proposals/2026-07-28_kd_ce_conflict_intervention.md). A 2x2 on one pod:
# {kd_scope all, kd_scope all_no_think} x {seed 20260726, 20260728}, everything
# else identical. Two seeds per condition because a one-run behavior comparison
# is not readable at this project's measured noise floor (0.1290).

SESSION=kd_conflict
SESSION_DATE=20260728

# Arms, in run order. Format: RUN_NAME|CONFIG|STEP_TAG
# Interleaved control/treatment rather than grouped, so that if the session is
# cut short for any reason the arms completed so far still form a comparison
# instead of two runs of one condition.
ARMS=(
  "kdconf_ctrl_a|configs/stage3_kdconf_ctrl_a.json|step_001000"
  "kdconf_nothink_a|configs/stage3_kdconf_nothink_a.json|step_001000"
  "kdconf_ctrl_b|configs/stage3_kdconf_ctrl_b.json|step_001000"
  "kdconf_nothink_b|configs/stage3_kdconf_nothink_b.json|step_001000"
)

# Start checkpoints to stage, and reference checkpoints to score. Format:
#   HF_INCLUDE_GLOB|LOCAL_DEST|REVISION|SCORE_AS   (SCORE_AS empty = do not score)
# All four arms start from the Stage 1 init. Nothing is scored as a reference
# this session: the comparison is *within* the 2x2, all four arms on one pod,
# so there is no cross-device gap to close and no reason to spend GPU minutes
# re-scoring a checkpoint trained to a different step count.
REF_CKPTS=(
  "stage1/qwen3_0p6b_init_v0/checkpoint|artifacts/stage1/qwen3_0p6b_init_v0/checkpoint|b955bd2f79b03a5418e2b8ca518a35faf047f085|"
)

# Transfer artifacts on the private HF relay (staged from the dev box).
HF_REPO=AlphaAvatar/aadistill-artifacts
TRANSFER_BUNDLE=transfer/repo_20260728b.bundle
TRANSFER_DATA=transfer/stage2_data_20260726.tar.zst
HF_PREFIX_BASE=stage3

# Pre-registered abort rule (proposal 2026-07-28, rule A). Applies to all four
# arms: this changes the loss, so a broken arm should stop rather than burn its
# budget. If primary-val ce at ABORT_CHECK_STEP is above its step-0 value, or a
# non-finite loss appears, stop that arm and report it.
ABORT_ARMS="kdconf_ctrl_a kdconf_nothink_a kdconf_ctrl_b kdconf_nothink_b"
ABORT_CHECK_STEP=300

# Gate evals every arm runs (post_run.sh). Cap stays 512 for continuity with
# every existing student scorecard.
HOLDOUT=data/warmup/holdout_v1.jsonl
BEHAVIOR_PROMPTS=data/eval_behavior_v0/prompts.jsonl
BEHAVIOR_MAX_NEW_TOKENS=512

# Convenience accessors -------------------------------------------------------
arm_field() { # arm_field <arm-entry> <1|2|3>
  printf '%s' "$1" | cut -d'|' -f"$2"
}
arm_names() {
  local a
  for a in "${ARMS[@]}"; do arm_field "$a" 1; done
}
