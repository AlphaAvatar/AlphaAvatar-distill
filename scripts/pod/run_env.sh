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

SESSION=ttb
SESSION_DATE=20260730

# Arm entries are name|config|final-step-tag. The two seeds of an arm are
# separate entries because the trainer takes its seed from the config.
ARMS=(
  "ttb_ctrl_a|configs/stage3/ttb_ctrl_a.json|step_000137"
  "ttb_ctrl_b|configs/stage3/ttb_ctrl_b.json|step_000137"
  "ttb_treat_a|configs/stage3/ttb_treat_a.json|step_000137"
  "ttb_treat_b|configs/stage3/ttb_treat_b.json|step_000137"
)

# The 2x2's start checkpoint, scored on this GPU before training so all four
# arms have a same-device baseline. Greedy-decode behavior metrics differ
# between CPU and GPU on a damaged student, so the reference must be re-scored
# in-session rather than quoted from the run that produced it.
# Entry format: hf-glob|local-dest|revision|scorecard-name
# Two references, both scored on this GPU before training:
#   * the Stage 1 init IS the shared step-0 model of all four arms — it is the
#     origin every per-arm delta is measured from, so it must be scored here and
#     not quoted from a CPU run;
#   * step_002700 is an EXTERNAL REFERENCE ONLY. It is never an initialization
#     in this session (decision 2026-07-30); forking arms from it is what
#     invalidated the previous run.
REF_CKPTS=(
  "stage1/qwen3_0p6b_init_v0/checkpoint|artifacts/stage1/qwen3_0p6b_init_v0/checkpoint|main|stage1_init_v0_step0"
  "stage3/s2v1_from_init/step_002700/model|artifacts/stage3/s2v1_from_init/checkpoints/step_002700/model|main|s2v1_from_init_2700_reference"
)

# Transfer artifacts on the private HF relay (staged from the dev box).
HF_REPO=AlphaAvatar/aadistill-artifacts
TRANSFER_BUNDLE=transfer/repo_20260730c.bundle
TRANSFER_DATA=transfer/stage2_data_20260726.tar.zst
# The 2x2 arms: built on the dev box by scripts/data/build_stage3_pilot.py from
# the hashed generation corpus, gitignored like every other built split.
TRANSFER_EXTRA=transfer/stage3_pilot_20260730.tar.gz
EXTRA_EXPECT="data/stage3_pilot/control/train data/stage3_pilot/treatment/train"
HF_PREFIX_BASE=ttb

# Pre-registered abort rule R4: stop an arm whose primary-val CE at the first
# eval exceeds its step-0 value, rather than retuning mid-session.
ABORT_ARMS="ttb_ctrl_a ttb_ctrl_b ttb_treat_a ttb_treat_b"
ABORT_CHECK_STEP=22

# Commit message the orchestrator uses for this session's logs (it no longer
# hardcodes one experiment's description).
SESSION_COMMIT_SUBJECT="stage3: corrected teacher-target baseline from the Stage 1 init — logs + write-up"
SESSION_COMMIT_BODY="Every arm forks from the Stage 1 structural-initialization
checkpoint (artifacts/stage1/qwen3_0p6b_init_v0/checkpoint, model.safetensors
sha256 86fbba78...), before any Stage 3 training. This is the correction to the
2026-07-30 run, which forked both arms from s2v1_from_init/step_002700 — already
2,700 steps of PUBLIC-target training — and so compared the two target sets
conditioned on one of them. That run is relabelled a post-s2v1 continuation
diagnostic and its R2 outcome is void as evidence about teacher-native targets.

step_002700 appears here as an EXTERNAL REFERENCE ONLY, never an initialization.
The Stage 1 init is scored in-session as the shared step-0 model, so every
per-arm delta has an origin measured on the same GPU.

Public-target control vs teacher-native treatment, two seeds each, 137 steps x
2 blocks x 8192 tokens = 2,244,608 tokens per arm, identical across arms.
Optimizer and scheduler reset in every arm (no --resume). Corpus, train/val
split, best_fit packing, trainable set and ordering rules are unchanged and
shared; only the assistant target differs.

Read strictly as an early fixed-compute comparison of two complete target
recipes from the common Stage 1 initialization (maintainer, 2026-07-30). NOT a
convergence result: 137 steps gives the treatment ~3.0 corpus passes and the
control ~7.6, and no budget on a 487-prompt corpus converges this init. NOT a
per-supervised-token comparison: at equal tokens the treatment carries 18.9x the
supervised tokens, which is inseparable from the intervention.

Readout caveats carried over: p(</think>) is measured where the PUBLIC render
demands the token and is descriptive only here; protocol metrics are maximised
by a terse answer and must be read with the answer-length axis."

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
