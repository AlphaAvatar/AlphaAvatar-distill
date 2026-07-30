#!/bin/bash
# Score the reference checkpoints on eval_behavior_v0 before training starts.
#   bash /workspace/score_refs.sh
# Marker: REFS_SCORED / REFS_FAILED:<name>.
#
# Runs first, while the GPU is idle, for two reasons: the ablation's result
# table needs every checkpoint scored on the *same* device, dtype and code (the
# dev-box baselines are CPU-scored and therefore not directly comparable), and
# a failure in the eval path surfaces in ~5 minutes instead of after 4.6 h of
# training.
set -x
exec > /workspace/score_refs.log 2>&1
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/aad || exit 1
source /workspace/run_env.sh || { echo "MARKER:REFS_FAILED:source_run_env" >> /workspace/run_markers.log; exit 1; }
fail() { echo "MARKER:REFS_FAILED:$1" >> /workspace/run_markers.log; exit 1; }

OUT=artifacts/stage3/reference_scorecards
mkdir -p "$OUT"

for entry in "${REF_CKPTS[@]}"; do
  dest=$(printf '%s' "$entry" | cut -d'|' -f2)
  name=$(printf '%s' "$entry" | cut -d'|' -f4)
  [ -n "$name" ] || continue

  # The shared step-0 model of a from-scratch fork needs the SAME readouts the
  # arms get, or the per-arm deltas have no origin to be measured from. Behavior
  # alone is not enough (proposal 11.4): holdout NLL is the guard rail, the two
  # probes are the mechanistic readouts, and INT8 is the deployment guard rail.
  uv run python scripts/evaluation/eval_ppl.py --data "$HOLDOUT" \
    --model "$dest" --out "$OUT/${name}_holdout_v1.json" || fail "holdout_$name"
  CUDA_VISIBLE_DEVICES= uv run python scripts/evaluation/eval_ppl.py \
    --data "$HOLDOUT" --model "$dest" --fake-quant int8 \
    --out "$OUT/${name}_holdout_v1_int8.json" || fail "holdout_int8_$name"
  uv run python scripts/evaluation/probe_think_close.py --model "$dest" \
    --per-group 4 --out "$OUT/${name}_probe_think_close.json" || fail "probe_$name"

  uv run python scripts/evaluation/eval_behavior.py --model "$dest" \
    --prompts "$BEHAVIOR_PROMPTS" --max-new-tokens "$BEHAVIOR_MAX_NEW_TOKENS" \
    --out "$OUT/${name}_behavior_v0.json" || fail "$name"
  uvx --from huggingface_hub hf upload "$HF_REPO" \
    "$OUT/${name}_behavior_v0.json" \
    "$HF_PREFIX_BASE/reference_scorecards/${name}_behavior_v0.json" \
    --repo-type model || fail "upload_$name"
  uvx --from huggingface_hub hf upload "$HF_REPO" \
    "$OUT/${name}_behavior_v0.generations.jsonl" \
    "$HF_PREFIX_BASE/reference_scorecards/${name}_behavior_v0.generations.jsonl" \
    --repo-type model || fail "upload_gen_$name"
  for extra in holdout_v1 holdout_v1_int8 probe_think_close; do
    uvx --from huggingface_hub hf upload "$HF_REPO" \
      "$OUT/${name}_${extra}.json" \
      "$HF_PREFIX_BASE/reference_scorecards/${name}_${extra}.json" \
      --repo-type model || fail "upload_${extra}_$name"
  done
done

echo "MARKER:REFS_SCORED" >> /workspace/run_markers.log
