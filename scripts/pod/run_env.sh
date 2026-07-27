#!/usr/bin/env bash
# Session definition for a Stage 3 GPU pod session. Single source of truth:
# every other script in scripts/pod/ sources this file instead of hardcoding a
# run name, config, checkpoint or HF path.
#
# Current session: the start-point ablation
# (logs/proposals/2026-07-27_stage3_start_point_ablation.md) — two arms trained
# sequentially on one pod so the ~19%-of-spend setup cost is paid once.
#
# To retarget for a later session, edit ARMS / REF_CKPTS / TRANSFER_* below.
# Nothing downstream needs changing.

SESSION=start_point_ablation
SESSION_DATE=20260727

# Arms, in run order. Format: RUN_NAME|CONFIG|STEP_TAG
# STEP_TAG is the final checkpoint directory name the run is expected to write.
ARMS=(
  "s2v1_from_s1|configs/stage3_s2v1_from_s1.json|step_002700"
  "s2v1_from_init|configs/stage3_s2v1_from_init.json|step_002700"
)

# Start checkpoints to stage before training, and reference checkpoints to score
# on `eval_behavior_v0` while the GPU is otherwise idle. Format:
#   HF_INCLUDE_GLOB|LOCAL_DEST|REVISION|SCORE_AS   (SCORE_AS empty = do not score)
# The two start points double as the ablation's reference arms: s1@660 is the
# A1 branch point and the Stage 1 init is A2's. `s2_blocks_v1` is the completed
# A0 arm — not a start point, downloaded purely so all four checkpoints get a
# behavior scorecard from the same device, dtype and code.
REF_CKPTS=(
  "stage3/s1_ffn_norm_v0/step_000660/model|artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model|727c837e810ef58eefd5e5553155b459f21414e5|s1_ffn_norm_v0_step660"
  "stage1/qwen3_0p6b_init_v0/checkpoint|artifacts/stage1/qwen3_0p6b_init_v0/checkpoint|b955bd2f79b03a5418e2b8ca518a35faf047f085|"
  "stage3/s2_blocks_v1/step_002700/model|artifacts/stage3/s2_blocks_v1/checkpoints/step_002700/model|b1b5170cb45ce7b141c02c23ca4b1bb89918a85b|s2_blocks_v1_step2700"
)

# Transfer artifacts on the private HF relay (staged from the dev box).
HF_REPO=AlphaAvatar/aadistill-artifacts
TRANSFER_BUNDLE=transfer/repo_20260727.bundle
TRANSFER_DATA=transfer/stage2_data_20260726.tar.zst
HF_PREFIX_BASE=stage3

# Pre-registered abort rule (proposal 2026-07-27, "Abort rule for A2"): a
# from-init start at lr 2e-4 / warmup 60 is outside the regime those
# hyperparameters were tuned for (s1 used 3e-4 from init). If primary-val ce at
# step ABORT_CHECK_STEP is above its step-0 value, or a non-finite loss appears,
# stop that arm and report a negative result rather than retuning mid-session.
# Read by orchestrate.sh only (the pod-side scripts do not use these).
ABORT_ARMS="s2v1_from_init"
ABORT_CHECK_STEP=300

# Gate evals every arm runs (post_run.sh).
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
