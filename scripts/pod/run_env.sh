#!/usr/bin/env bash
# Session definition for a Stage 3 GPU pod session. Single source of truth:
# every other script in scripts/pod/ sources this file instead of hardcoding a
# run name, config, checkpoint or HF path.
#
# Current session: the packing / block_len control run + the project's first
# run-to-run variance measurement
# (logs/proposals/2026-07-28_stage3_packing_blocklen_control.md). Two arms on
# one pod: arm A is the control, arm B is A repeated at a different seed, so
# |A - B| is the noise floor the control's delta is read against.
#
# To retarget for a later session, edit ARMS / REF_CKPTS / TRANSFER_* below.
# Nothing downstream needs changing.

SESSION=packing_control
SESSION_DATE=20260728

# Arms, in run order. Format: RUN_NAME|CONFIG|STEP_TAG
# STEP_TAG is the final checkpoint directory name the run is expected to write.
ARMS=(
  "s2v1_bl2048|configs/stage3_s2v1_bl2048.json|step_002700"
  "s2v1_bl2048_seedB|configs/stage3_s2v1_bl2048_seedB.json|step_002700"
)

# Start checkpoints to stage before training, and reference checkpoints to score
# on `eval_behavior_v0` while the GPU is otherwise idle. Format:
#   HF_INCLUDE_GLOB|LOCAL_DEST|REVISION|SCORE_AS   (SCORE_AS empty = do not score)
# The Stage 1 init is the start point of both arms (not scored — it is not in
# any comparison). `s2v1_from_init@2700` is the baseline the control is measured
# against, so it MUST be re-scored here: behavior scorecards are only comparable
# within one device (decision record 2026-07-27), and the committed baseline
# scorecard was produced on a different pod.
REF_CKPTS=(
  "stage1/qwen3_0p6b_init_v0/checkpoint|artifacts/stage1/qwen3_0p6b_init_v0/checkpoint|b955bd2f79b03a5418e2b8ca518a35faf047f085|"
  "stage3/s2v1_from_init/step_002700/model|artifacts/stage3/s2v1_from_init/checkpoints/step_002700/model|3269440cb51efb5bf6d2d70370ee24ef11b31cf8|s2v1_from_init_step2700"
)

# Transfer artifacts on the private HF relay (staged from the dev box).
HF_REPO=AlphaAvatar/aadistill-artifacts
TRANSFER_BUNDLE=transfer/repo_20260728.bundle
TRANSFER_DATA=transfer/stage2_data_20260726.tar.zst
HF_PREFIX_BASE=stage3

# Pre-registered abort rule (proposal 2026-07-28, rule R4): the data path is the
# only change, so a healthy arm should track the baseline's early trajectory. If
# primary-val ce at step ABORT_CHECK_STEP is above its step-0 value, or a
# non-finite loss appears, stop that arm and report a negative result rather
# than retuning mid-session. Read by orchestrate.sh only.
ABORT_ARMS="s2v1_bl2048 s2v1_bl2048_seedB"
ABORT_CHECK_STEP=300

# Gate evals every arm runs (post_run.sh). Cap stays 512: every existing student
# scorecard was produced at 512, and the control's delta is against those.
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
