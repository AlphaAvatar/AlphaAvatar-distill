#!/usr/bin/env bash
# Session definition for a GPU pod session. Single source of truth: every other
# script in scripts/pod/ sources this file instead of hardcoding a run name,
# config, checkpoint or HF path.
#
# Current session: the **Stage 3 teacher-target 2x2**
# (logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md). Four arms on
# one pod: public-target control vs teacher-native treatment, two seeds each.
# Both arms share the start checkpoint, the prompt set, the packing
# (best_fit @ block_len 8192), the schedule and the total training-token budget;
# they differ only in the assistant turn of each training sample.

SESSION=tt2x2
SESSION_DATE=20260730

# Arm entries are name|config|final-step-tag. The two seeds of an arm are
# separate entries because the trainer takes its seed from the config.
ARMS=(
  "tt2x2_ctrl_a|configs/stage3/tt2x2_ctrl_a.json|STEP_TAG"
  "tt2x2_ctrl_b|configs/stage3/tt2x2_ctrl_b.json|STEP_TAG"
  "tt2x2_treat_a|configs/stage3/tt2x2_treat_a.json|STEP_TAG"
  "tt2x2_treat_b|configs/stage3/tt2x2_treat_b.json|STEP_TAG"
)

# The 2x2's start checkpoint, scored on this GPU before training so all four
# arms have a same-device baseline. Greedy-decode behavior metrics differ
# between CPU and GPU on a damaged student, so the reference must be re-scored
# in-session rather than quoted from the run that produced it.
# Entry format: hf-glob|local-dest|revision|scorecard-name
REF_CKPTS=(
  "stage3/s2v1_from_init/step_002700/model|artifacts/stage3/s2v1_from_init/checkpoints/step_002700/model|main|s2v1_from_init_2700"
)

# Transfer artifacts on the private HF relay (staged from the dev box).
HF_REPO=AlphaAvatar/aadistill-artifacts
TRANSFER_BUNDLE=transfer/repo_20260730b.bundle
TRANSFER_DATA=transfer/stage2_data_20260726.tar.zst
# The 2x2 arms: built on the dev box by scripts/data/build_stage3_pilot.py from
# the hashed generation corpus, gitignored like every other built split.
TRANSFER_EXTRA=transfer/stage3_pilot_20260730.tar.gz
EXTRA_EXPECT="data/stage3_pilot/control/train data/stage3_pilot/treatment/train"
HF_PREFIX_BASE=tt2x2

# Pre-registered abort rule R4: stop an arm whose primary-val CE at the first
# eval exceeds its step-0 value, rather than retuning mid-session.
ABORT_ARMS="tt2x2_ctrl_a tt2x2_ctrl_b tt2x2_treat_a tt2x2_treat_b"
ABORT_CHECK_STEP=EVAL_EVERY

# Commit message the orchestrator uses for this session's logs (it no longer
# hardcodes one experiment's description).
SESSION_COMMIT_SUBJECT="stage3: teacher-target 2x2 on L40S — logs + write-up"
SESSION_COMMIT_BODY="Public-target control vs teacher-native treatment, two seeds
per arm, from stage3/s2v1_from_init/step_002700. Both arms share the start
checkpoint, the accepted prompt subset, best_fit packing at block_len 8192, the
schedule and the total training-token budget; they differ only in the assistant
turn of each sample and in the seed. Passes over the prompt set differ between
arms because teacher targets are several times longer — that is reported, not
corrected (maintainer decision 2026-07-30: hold total training tokens equal).

Primary readouts are protocol competence: format_ok, think_closed, terminated,
empty_answer, p(</think>) and p(<|im_end|>), with holdout NLL as a +/-1% guard
rail. Rules R1-R4 are pre-registered in
logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md. Behavior numbers
are read against the measured 0.1290 seed-only noise floor, which is why every
arm has two seeds."

# Paths setup.sh checks exist after the bundle + data land.
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
