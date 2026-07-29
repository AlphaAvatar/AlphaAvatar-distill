#!/usr/bin/env bash
# Session definition for a GPU pod session. Single source of truth: every other
# script in scripts/pod/ sources this file instead of hardcoding a run name,
# config, checkpoint or HF path.
#
# Current session: the **engine benchmark + teacher-corpus pilot**
# (logs/proposals/2026-07-29_engine_benchmark.md). This session does no
# training, so ARMS is empty and the driver is bench_and_generate.sh rather
# than orchestrate.sh. setup.sh is still the entry point and is unchanged: it
# only reads HF_REPO, the transfer artifacts, REF_CKPTS and the eval paths.

SESSION=engine_bench
SESSION_DATE=20260729

# No training arms this session.
ARMS=()

# No checkpoints to stage. The teacher is pulled from the Hub at its pinned
# revision by the benchmark itself, and nothing is being fine-tuned or scored
# against a reference student, so hashes_ckpt.txt is empty by design.
REF_CKPTS=()

# Transfer artifacts on the private HF relay (staged from the dev box).
HF_REPO=AlphaAvatar/aadistill-artifacts
TRANSFER_BUNDLE=transfer/repo_20260729.bundle
TRANSFER_DATA=transfer/stage2_data_20260726.tar.zst
HF_PREFIX_BASE=engine_bench

# Budget knobs read by bench_and_generate.sh (P6). The generation cap is
# enforced in-process at a batch boundary, so a stop still leaves complete,
# hashed artifacts and a manifest marked `complete: false`.
export GEN_MAX_HOURS=3.0
export HOURLY_USD=0.99   # measured at pod creation 2026-07-29 (L40S, US)
export N_PROMPTS=32
export LIMIT_PER_SLICE=200

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
