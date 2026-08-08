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
set -u
cd "$(dirname "$0")/../.." || exit 1
HIDE=${HIDE_DIR:-/home/ecs-user/aad-scratch/podsim_hidden}
mkdir -p "$HIDE"

restore() {
  for p in "$HIDE"/*; do
    [ -e "$p" ] || continue
    n=$(basename "$p" | tr '@' '/')
    mkdir -p "$(dirname "$n")"
    mv "$p" "$n"
  done
  rmdir "$HIDE" 2>/dev/null
  echo "restored the hidden artifacts"
}
trap restore EXIT INT TERM

# Everything a pod does NOT get from the bundle. `artifacts/stage3/corpus_v2`,
# `artifacts/stage3/ladder_uniform_probe` and `artifacts/stage1/...` are left in
# place because every pod session stages those from the relay.
HIDDEN_PATHS=${HIDDEN_PATHS:-"artifacts/audit
artifacts/stage3/ladder_uniform
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

uv run pytest tests/ -q --ignore=tests/data/test_recovery_corpus_pipeline.py 2>&1 | tail -12
