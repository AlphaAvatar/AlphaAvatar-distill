#!/usr/bin/env bash
# Session definition for a GPU pod session. Single source of truth: every other
# script in scripts/pod/ sources this file instead of hardcoding a run name,
# config, checkpoint or HF path.
#
# **This is a template — no session is currently defined.** The previous
# session's arms (tt2x2_*, ttb_*) and their configs were removed in the
# 2026-07-31 cleanup; leaving them here would have pointed the orchestrator at
# files that no longer exist. Fill in ARMS before running a session, and see
# logs/PROPOSAL.md for the approved plan.

SESSION=unset
SESSION_DATE=00000000

# Arm entries are name|config|final-step-tag. A run of the corpus-scaling study
# differs from configs/stage3/recovery.json only in data_dir (the token-ladder
# subset) and schedule.total_steps, so each arm gets its own generated config.
ARMS=()

# Reference checkpoints scored on the same GPU before training, so every arm has
# a same-device baseline. Format: hf-glob|local-dest|revision|scorecard-name.
# The Stage 1 structural init is the pinned fork point for all recovery work.
REF_CKPTS=(
  "stage1/qwen3_0p6b_init_v0/checkpoint|artifacts/stage1/qwen3_0p6b_init_v0/checkpoint|main|stage1_init_v0_step0"
)

# Transfer artifacts on the private HF relay (staged from the dev box).
HF_REPO=AlphaAvatar/aadistill-artifacts
TRANSFER_BUNDLE=transfer/repo_UNSET.bundle
TRANSFER_DATA=transfer/stage2_data_20260726.tar.zst
TRANSFER_EXTRA=
EXTRA_EXPECT=
# Must be exported: verify_and_report.py reads it from the environment, and a
# non-exported value made a whole session fail upload verification against a
# path nothing had written to.
export HF_PREFIX_BASE=unset

# Pre-registered abort rule: stop an arm whose primary-val CE at the first eval
# exceeds its step-0 value, rather than retuning mid-session.
ABORT_ARMS=""
ABORT_CHECK_STEP=25

# Commit message the orchestrator uses for this session's logs.
SESSION_COMMIT_SUBJECT="stage3: ${SESSION} session — logs"
SESSION_COMMIT_BODY=""

# Optional: the reporter the orchestrator runs after fetching. Left EMPTY on
# purpose — verify_and_report.py is verify-only, and a session must supply its
# own write-up rather than inherit another experiment's decision rules.
REPORT_CMD=

# Paths setup.sh checks exist after the bundle + data land.
HOLDOUT=data/warmup/holdout_v1.jsonl
BEHAVIOR_PROMPTS=data/eval_behavior_v0/prompts.jsonl
# Formal behaviour measurement is UNRESTRICTED (AGENTS.md P18). This value is
# used only for cheap smoke checks and is recorded as a censored measurement.
BEHAVIOR_MAX_NEW_TOKENS=512

# Convenience accessors -------------------------------------------------------
arm_field() { printf '%s' "$1" | cut -d'|' -f"$2"; }
arm_names() { local a; for a in "${ARMS[@]}"; do arm_field "$a" 1; done; }
