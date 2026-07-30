#!/bin/bash
# Post-training gate evals + artifact upload for one arm.
#   bash /workspace/post_run.sh <RUN_NAME> <CONFIG> <STEP_TAG>
# Marker: POST_DONE:<RUN_NAME> / POST_FAILED:<RUN_NAME>:<step>.
set -x
RUN_NAME="$1"
CONFIG="$2"
STEP_TAG="$3"
[ -n "$RUN_NAME" ] && [ -n "$CONFIG" ] && [ -n "$STEP_TAG" ] || {
  echo "usage: post_run.sh <RUN_NAME> <CONFIG> <STEP_TAG>"; exit 2; }

exec > "/workspace/post_run_${RUN_NAME}.log" 2>&1
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/aad || exit 1
source /workspace/run_env.sh || { echo "MARKER:POST_FAILED:${RUN_NAME}:source_run_env" >> /workspace/run_markers.log; exit 1; }
fail() { echo "MARKER:POST_FAILED:${RUN_NAME}:$1" >> /workspace/run_markers.log; exit 1; }

RUN=artifacts/stage3/$RUN_NAME
CKPT=$RUN/checkpoints/$STEP_TAG/model
[ -d "$CKPT" ] || fail no_final_ckpt

# The trainer writes weights + config; tokenizer files come from the arm's own
# start checkpoint (read from the config so this stays arm-agnostic).
SRC=$(uv run python -c "import json,sys; print(json.load(open('$CONFIG'))['student_path'])") || fail read_student_path
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do
  cp -n "$SRC/$f" "$CKPT/$f" || fail "cp_$f"
done

# 1) bf16 holdout on GPU (comparable to every prior Stage 3 GPU number)
uv run python scripts/evaluation/eval_ppl.py --data "$HOLDOUT" \
  --model "$CKPT" --out "$RUN/eval_holdout_v1.json" || fail holdout_bf16
# 2) INT8 fake-quant (P9), CPU for comparability with the 2026-07-26 dev log
CUDA_VISIBLE_DEVICES= uv run python scripts/evaluation/eval_ppl.py \
  --data "$HOLDOUT" --model "$CKPT" \
  --fake-quant int8 --out "$RUN/eval_holdout_v1_int8.json" || fail holdout_int8
CUDA_VISIBLE_DEVICES= uv run python scripts/evaluation/eval_ppl.py \
  --data "$HOLDOUT" --model "$CKPT" \
  --fake-quant int8 --fake-quant-scope decoder \
  --out "$RUN/eval_holdout_v1_int8_decoder.json" || fail holdout_int8_dec

# 3) eval_behavior_v0 — the behavior scorecard (GPU, bf16, greedy)
uv run python scripts/evaluation/eval_behavior.py --model "$CKPT" \
  --prompts "$BEHAVIOR_PROMPTS" --max-new-tokens "$BEHAVIOR_MAX_NEW_TOKENS" \
  --out "$RUN/eval_behavior_v0.json" || fail behavior

# 3b) p(</think>) probe — the CE/KD conflict experiment's PRIMARY readout.
# Continuous and far less seed-noisy than the generation metrics, whose measured
# run-to-run floor is 0.1290 on the composite
# (logs/EXPERIMENTS.md, CE/KD conflict).
uv run python scripts/evaluation/probe_think_close.py --model "$CKPT" \
  --per-group 4 --out "$RUN/probe_think_close.json" || fail probe_think

# 4) generation smoke: greedy, 80 new tokens, same 3 prompts as every prior run
uv run python - "$CKPT" "$RUN/gen_smoke.json" <<'EOF' || fail gen_smoke
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ckpt, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(ckpt)
m = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to("cuda").eval()
prompts = ["What is 2+2?",
           "Write a Python function that returns the first n Fibonacci numbers.",
           "At what temperature does water boil at sea level?"]
out = {}
for p in prompts:
    text = tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        gen = m.generate(ids, max_new_tokens=80, do_sample=False)
    out[p] = tok.decode(gen[0, ids.shape[1]:])
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False)[:600])
EOF

cp "/workspace/console_${RUN_NAME}.log" "$RUN/console.log"

# 5) hashes of everything retained, then upload to the private HF repo
HASHFILE=artifacts/stage3/${RUN_NAME}_artifact_hashes_${SESSION_DATE}.txt
SMALL_FILES="train_log.jsonl run_manifest.json eval_holdout_v1.json \
eval_holdout_v1_int8.json eval_holdout_v1_int8_decoder.json eval_behavior_v0.json \
eval_behavior_v0.generations.jsonl probe_think_close.json gen_smoke.json console.log"
( cd /workspace/aad && for f in $SMALL_FILES; do echo "$RUN/$f"; done | xargs sha256sum && sha256sum "$CKPT"/* ) \
  > "$HASHFILE" || fail hash

uvx --from huggingface_hub hf upload "$HF_REPO" \
  "$CKPT" "$HF_PREFIX_BASE/$RUN_NAME/$STEP_TAG/model" --repo-type model || fail up_model
for f in $SMALL_FILES; do
  uvx --from huggingface_hub hf upload "$HF_REPO" \
    "$RUN/$f" "$HF_PREFIX_BASE/$RUN_NAME/$f" --repo-type model || fail "up_$f"
done
uvx --from huggingface_hub hf upload "$HF_REPO" \
  "$HASHFILE" "$HF_PREFIX_BASE/$RUN_NAME/$(basename "$HASHFILE")" --repo-type model || fail up_hashes

echo "MARKER:POST_DONE:${RUN_NAME}" >> /workspace/run_markers.log
