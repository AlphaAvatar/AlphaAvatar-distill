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

# The trap is armed HERE, after the lock is held and after the leftover check --
# never before. That ordering is what stops a losing sweep from restoring the
# winner's saved paths: a loser exits at the lock check with no trap installed,
# so `restore` cannot run for it at all. An in-function "am I the holder?" guard
# was tried and removed: it was unreachable, and an unreachable safeguard invites
# exactly the false confidence this script has already cost once.
restore() {
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

PODSIM_CMD=${PODSIM_CMD:-"uv run pytest tests/ -q --ignore=tests/data/test_recovery_corpus_pipeline.py"}
eval "$PODSIM_CMD" 2>&1 | tail -12
