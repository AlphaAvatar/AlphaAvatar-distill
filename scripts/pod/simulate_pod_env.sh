#!/usr/bin/env bash
# Run the pod's test command with only the artifacts a pod session actually stages.
#
#   bash scripts/pod/simulate_pod_env.sh
#
# Why this exists. A pod checks out the session bundle, which carries only
# tracked files — everything under `artifacts/` is gitignored and does not
# travel. The dev box has all of it, so the test suite passes here and can still
# fail there, on a GPU that is already billing. That happened on 2026-08-08: the
# E6 pod reached its test gate and died on three tests whose inputs it had no way
# to possess, costing $0.10 and a full setup cycle to learn something a local run
# could have said for free.
#
# So: move aside everything a pod does not stage, run the pod's exact command,
# and restore unconditionally. A test that needs a gitignored artifact should
# declare that with `skipif`, and this is how you find the ones that do not.
#
# The hidden set is deliberately a superset of what any one experiment stages.
# If a future session stages more, the check is still sound — it just skips more
# than it needs to, which errs toward catching problems rather than missing them.
#
# TWO DEFECTS, both of which corrupted this repository on 2026-08-15 and cost a
# diagnosis that briefly read as data loss:
#
#   1. `restore` used `mv "$saved" "$dest"`. When the test run RECREATED the
#      destination -- `artifacts/audit` is recreated by any driver rehearsal --
#      `mv` moves the saved directory INSIDE the recreated one instead of
#      replacing it. It happened twice, burying the real `artifacts/audit` at
#      `artifacts/audit/artifacts@audit/artifacts@audit/` and silently turning
#      11 tests into skips. A skip is not a failure, so the suite still read
#      green.
#   2. Nothing prevented two sweeps overlapping. The second's `restore` walks
#      `$HIDE` and adopts the first's saved paths, so the two interleave.
#
# Fixed here: restore reproduces the EXACT pre-simulation state (a recreated
# destination is quarantined, never nested and never silently deleted), and a
# lock makes a concurrent sweep fail loudly instead of racing.
#
# CONSEQUENCE, and it is intended: `$HIDE.recreated` ACCUMULATES. Every sweep
# that recreates `artifacts/audit` leaves a quarantined copy there, and nothing
# prunes it, because "never delete a recreated destination" is the property
# that fixed the 2026-08-15 scare. Its contents are regenerated files -- the
# frozen-asset verifier rewrites `frozen_asset_verification.json` on every run
# -- so the directory is safe to remove by hand at any time. It is recurring
# scratch, not a retained artifact: see the WITHDRAWN `podsim_quarantine_residue`
# entry in `logs/checkpoint_tombstones.json` for why it must not be tombstoned.
set -u
PODSIM_ROOT=${PODSIM_ROOT:-"$(cd "$(dirname "$0")/../.." && pwd)"}
cd "$PODSIM_ROOT" || exit 1
HIDE=${HIDE_DIR:-/home/ecs-user/aad-scratch/podsim_hidden}
LOCK=${PODSIM_LOCK:-"${HIDE}.lock"}
QUAR="${HIDE}.recreated"

# Single-instance exclusion. `mkdir` is atomic, so exactly one sweep wins; the
# loser exits WITHOUT running and WITHOUT restoring, because the files under
# $HIDE belong to the holder.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "REFUSING: another pod simulation holds $LOCK." >&2
  echo "  Simulator sweeps must not overlap: the loser would restore the" >&2
  echo "  winner's saved paths. If no sweep is running, remove $LOCK." >&2
  exit 3
fi
HELD_LOCK=1

# A non-empty $HIDE at startup means a previous sweep died before restoring.
# Adopting those files would restore them under THIS run's assumptions; refuse
# and let a human look.
if [ -d "$HIDE" ] && [ -n "$(ls -A "$HIDE" 2>/dev/null)" ]; then
  echo "REFUSING: $HIDE is not empty, so a previous sweep did not restore:" >&2
  ls -A "$HIDE" | sed 's/^/    /' >&2
  echo "  Restore those by hand before simulating again." >&2
  rmdir "$LOCK" 2>/dev/null
  exit 4
fi
mkdir -p "$HIDE"

# --- the second dimension: HOME and Hugging Face -----------------------------
#
# Hiding gitignored ARTIFACTS was only half of what a pod does not have. The
# other half is $HOME. A fresh pod runs as root with an empty home, exports its
# own HF_HOME, and holds no Hugging Face dataset cache and no credential file --
# and this script said nothing about any of that.
#
# That gap cost C1 attempt 3R $0.3482 on 2026-09-04. Seven renderer-parity cases
# read `~/.cache/huggingface/hub` directly, so they passed here and could never
# pass on a pod; the simulator was run, was green, and described a machine that
# does not exist. Twelve of the fourteen failures were environment, not code.
#
# So the simulated process gets: a fresh empty HOME, an isolated HF_HOME with
# HF_HUB_CACHE beneath it, a non-empty synthetic HF_TOKEN exactly as pod setup
# exports one before its test gate -- and no path back to the dev box's real
# cache or real credential. The token is synthetic on purpose: these tests use
# monkeypatched network calls, so they test transport logic, never possession of
# a real credential, and a simulation that borrowed the operator's token would
# hide a test that had quietly started needing it.
#
# Every variable is saved and put back by `restore_env`, which the same EXIT trap
# runs before the artifacts are restored.
ENVROOT=${PODSIM_ENV_ROOT:-"${HIDE}.env"}
PODSIM_TOKEN=${PODSIM_HF_TOKEN:-"hf_podEquivalentSyntheticToken000000000"}

#: Anything that could point the process back at the dev box's real HF state.
ISOLATED_VARS="HOME HF_HOME HF_HUB_CACHE HF_TOKEN HUGGINGFACE_HUB_CACHE \
HF_DATASETS_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME"

SAVED_ENV=""
save_env() {
  for v in $ISOLATED_VARS; do
    if [ -n "${!v+set}" ]; then
      SAVED_ENV="${SAVED_ENV}${v}=${!v}"$'\n'
    else
      SAVED_ENV="${SAVED_ENV}${v}"$'\n'          # bare name == was unset
    fi
  done
}

restore_env() {
  [ -n "$SAVED_ENV" ] || return 0
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "$line" in
      *=*) export "${line%%=*}"="${line#*=}" ;;
      *)   unset "$line" ;;
    esac
  done <<< "$SAVED_ENV"
  rm -rf "$ENVROOT"
  SAVED_ENV=""
  echo "restored the environment"
}

# The trap is armed HERE, after the lock is held and after the leftover check --
# never before. That ordering is what stops a losing sweep from restoring the
# winner's saved paths: a loser exits at the lock check with no trap installed,
# so `restore` cannot run for it at all. An in-function "am I the holder?" guard
# was tried and removed: it was unreachable, and an unreachable safeguard invites
# exactly the false confidence this script has already cost once.
restore() {
  restore_env
  for p in "$HIDE"/*; do
    [ -e "$p" ] || continue
    n=$(basename "$p" | tr '@' '/')
    mkdir -p "$(dirname "$n")"
    if [ -e "$n" ]; then
      # The run recreated this path. It did not exist before the simulation, so
      # it must not survive it -- but quarantine rather than delete, because
      # deleting on a restore path is how a bug becomes data loss.
      mkdir -p "$QUAR"
      mv "$n" "$QUAR/$(basename "$p").recreated.$$" 2>/dev/null
      echo "  quarantined a recreated $n -> $QUAR"
    fi
    mv "$p" "$n"
  done
  rmdir "$HIDE" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "restored the hidden artifacts"
}
trap restore EXIT INT TERM

# Everything a pod does NOT get from the bundle. `artifacts/stage3/corpus_v2` and
# `artifacts/stage3/ladder_uniform_probe` are left in place because every pod
# session stages those from the relay.
#
# `recovery_search_v1` joined this list on 2026-08-14: the v2 migration stopped
# staging it, but `tests/autoinit/test_frozen_assets.py` still pointed at it, so
# seven tests read an artifact no pod possesses. The dev box had it, the suite
# passed here, and the pod's blocking test gate failed 7 minutes into a paid
# setup. **When an asset stops being staged, add it here in the same commit.**
#
# `artifacts/stage1/...` is NOT a safe blanket exception, and assuming it was cost
# a paid E8b-S2 pod on 2026-08-11. Sessions stage different initializations: an
# E8b s2/s3 pod builds DP and DC only (`NEED_COMPRESSED=0`) and never sees the
# compressed pair, so a test that assumed the compressed baseline was present ran
# there for the first time and failed. Simulate the session you are about to launch
# by hiding the initializations it does not stage, e.g. for s2/s3:
#
#   HIDDEN_PATHS="$(cat <<'EOS'
#   artifacts/audit
#   artifacts/stage3/ladder_uniform
#   artifacts/stage1/qwen3_0p6b_init_v0
#   artifacts/stage1/e8_contribution_init_v1
#   EOS
#   )" bash scripts/pod/simulate_pod_env.sh
#
# and pin the cpu set the pod will have (`taskset -c 0-12`), since the suite's
# behaviour depends on it.
HIDDEN_PATHS=${HIDDEN_PATHS:-"artifacts/audit
artifacts/stage3/ladder_uniform
artifacts/stage3/recovery_search_v1
artifacts/stage3/rescued
artifacts/stage3/e1_results.json
artifacts/stage3/e1_consolidated.json
artifacts/stage3/e4_p2_r1600k_sa
artifacts/stage3/e4_p2_r1600k_sb
data/warmup/holdout_v1.jsonl"}

n=0
while IFS= read -r p; do
  [ -n "$p" ] && [ -e "$p" ] || continue
  mv "$p" "$HIDE/$(echo "$p" | tr '/' '@')" && n=$((n + 1))
done <<< "$HIDDEN_PATHS"
echo "hid $n path(s) a pod session does not receive"

# Apply the HOME/HF isolation described above. Saved first, so the EXIT trap can
# put the environment back whatever happens next.
save_env
rm -rf "$ENVROOT"
mkdir -p "$ENVROOT/home" "$ENVROOT/hf/hub" || exit 5
export HOME="$ENVROOT/home"
export HF_HOME="$ENVROOT/hf"
export HF_HUB_CACHE="$ENVROOT/hf/hub"
export HF_TOKEN="$PODSIM_TOKEN"
# These would each defeat the isolation on their own by naming the real cache.
unset HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME

# Assert the isolation instead of assuming it: a simulation that silently kept
# the dev box's cache is the exact failure this dimension exists to prevent.
if [ -e "$HOME/.cache/huggingface" ]; then
  echo "REFUSING: the simulated HOME already has an HF cache at $HOME/.cache" >&2
  exit 6
fi
if compgen -G "$HF_HUB_CACHE/datasets--*" > /dev/null; then
  echo "REFUSING: the isolated hub cache is not empty; the dev box's datasets" >&2
  echo "  are visible and the seven renderer-parity cases would not skip." >&2
  exit 6
fi
# Never the token itself, only that one is present.
echo "isolated HOME=$HOME (empty), HF_HOME=$HF_HOME, HF_TOKEN set (${#HF_TOKEN} chars, synthetic)"

# Must stay byte-identical in its ignore list to the pod gate in
# `autoinit_preflight_setup.sh`, or this simulates a command the pod does
# not run. `tests/pod/test_phase_a_stages1_5_execute.py` is a PRE-flight
# rehearsal: it exists to execute the driver before a pod is created, it
# takes ~20 minutes, and on the pod it would spend a large share of the
# 2700 s gate re-proving what the dev box already proved -- against a
# timeout whose exit 90 kills the session.
#
# The interpreter is the repo venv directly, not `uv run`. Two reasons, and both
# are about fidelity: the pod's gate runs `/opt/train/bin/python -m pytest` against
# a project installed editable, with no PYTHONPATH, which `.venv/bin/python -m pytest`
# mirrors exactly -- and `uv run` would reach into `$HOME/.cache/uv`, which the
# isolation above deliberately empties, so it would re-resolve the environment
# inside a simulation rather than run the suite.
PODSIM_CMD=${PODSIM_CMD:-".venv/bin/python -m pytest tests/ -q \
  --ignore=tests/data/test_recovery_corpus_pipeline.py \
  --ignore=tests/pod/test_phase_a_stages1_5_execute.py"}

# The pod writes /workspace/pytest.log and the file dies with the pod; here the
# log survives, which is the whole point of simulating. `tail -12` alone threw
# away the FAILED list on every previous sweep.
#
# It lands OUTSIDE the repository on purpose. `logs/` is tracked and every entry
# in it must be classified in `CATALOG.md`, so writing there would both dirty the
# working tree the sweep is asserting is clean and fail a structural test.
#
# The default is derived from `$HIDE`, which is per-invocation, NOT a fixed global
# path. A fixed default is a shared mutable file: unsetting PODSIM_LOG for the
# suite (so nested runs cannot inherit it) made every nested simulation fall back
# to the SAME default and truncate the outer sweep's log while the outer shell
# still held an open fd at its own offset. The 2026-09-04 sweep A log came back
# with its `FAILED` lines punched out -- `grep` found nothing in a file whose tail
# plainly showed `3 failed`. Per-invocation by construction is the fix; inheriting
# is not, because that is the bug this default exists to avoid.
PODSIM_LOG=${PODSIM_LOG:-"${HIDE}.pytest.log"}
mkdir -p "$(dirname "$PODSIM_LOG")"

# `--junitxml` is a REPORTING flag: it changes nothing about which tests are
# selected or how they execute, so the command still runs the pod's own suite.
# It is the only way to name every skip and every pass exactly, which is what the
# readiness record has to assert -- and what attempt 3R's `tail -4` could not say.
if [ -n "${PODSIM_JUNIT:-}" ]; then
  mkdir -p "$(dirname "$PODSIM_JUNIT")"
  PODSIM_CMD="$PODSIM_CMD --junitxml=$PODSIM_JUNIT"
  echo "junit report -> $PODSIM_JUNIT"
fi

echo "running: $PODSIM_CMD"

# Capture, then UNSET, before running the suite. `tests/pod/test_simulator_restore.py`
# drives this very script as a subprocess, and every PODSIM_* control variable
# here is exported -- so a nested run inherited THIS invocation's settings. The
# 2026-09-04 sweep proved what that costs: the nested simulators inherited
# `PODSIM_JUNIT`, appended a second `--junitxml=` to their own one-line commands,
# overwrote the outer sweep's report, and failed five tests that were correct.
# They are inputs to one invocation and must not outlive it.
_podsim_log="$PODSIM_LOG"
_podsim_cmd="$PODSIM_CMD"
unset PODSIM_JUNIT PODSIM_LOG PODSIM_CMD PODSIM_ROOT PODSIM_ENV_ROOT \
      PODSIM_HF_TOKEN HIDE_DIR PODSIM_LOCK HIDDEN_PATHS

eval "$_podsim_cmd" > "$_podsim_log" 2>&1
PODSIM_RC=$?
grep -E '^(FAILED|ERROR) ' "$_podsim_log" || true
tail -12 "$_podsim_log"
echo "pytest rc=$PODSIM_RC; full log at $_podsim_log"
exit "$PODSIM_RC"
