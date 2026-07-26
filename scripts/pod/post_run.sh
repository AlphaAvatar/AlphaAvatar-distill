#!/bin/bash
# Post-training gate evals + artifact upload for s2_blocks_v1.
set -x
exec > /workspace/post_run.log 2>&1
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/aad || exit 1
fail() { echo "MARKER:POST_FAILED:$1" >> /workspace/run_markers.log; exit 1; }

RUN=artifacts/stage3/s2_blocks_v1
CKPT=$RUN/checkpoints/step_002700/model
SRC=artifacts/stage3/s2_blocks_v0/checkpoints/step_000660/model
[ -d "$CKPT" ] || fail no_final_ckpt
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do
  cp -n "$SRC/$f" "$CKPT/$f" || fail "cp_$f"
done

# 1) bf16 holdout on GPU (comparable to s1/A-B GPU numbers)
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model "$CKPT" --out "$RUN/eval_holdout_v1.json" || fail holdout_bf16
# 2) INT8 fake-quant (P9), CPU for comparability with the 2026-07-26 dev log
CUDA_VISIBLE_DEVICES= uv run python scripts/eval_ppl.py \
  --data data/warmup/holdout_v1.jsonl --model "$CKPT" \
  --fake-quant int8 --out "$RUN/eval_holdout_v1_int8.json" || fail holdout_int8
CUDA_VISIBLE_DEVICES= uv run python scripts/eval_ppl.py \
  --data data/warmup/holdout_v1.jsonl --model "$CKPT" \
  --fake-quant int8 --fake-quant-scope decoder \
  --out "$RUN/eval_holdout_v1_int8_decoder.json" || fail holdout_int8_dec

# 3) generation smoke: greedy, 80 new tokens, same 3 prompts as s1/A-B
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

cp /workspace/console_s2v1.log "$RUN/console.log"

# 4) hashes of everything retained, then upload to the private HF repo
( cd /workspace/aad && sha256sum \
    "$RUN"/train_log.jsonl "$RUN"/run_manifest.json \
    "$RUN"/eval_holdout_v1.json "$RUN"/eval_holdout_v1_int8.json \
    "$RUN"/eval_holdout_v1_int8_decoder.json "$RUN"/gen_smoke.json \
    "$RUN"/console.log "$CKPT"/* \
  > artifacts/stage3/s2v1_artifact_hashes_2026-07-26.txt ) || fail hash
uvx --from huggingface_hub hf upload AlphaAvatar/aadistill-artifacts \
  "$CKPT" stage3/s2_blocks_v1/step_002700/model --repo-type model || fail up_model
for f in train_log.jsonl run_manifest.json eval_holdout_v1.json \
         eval_holdout_v1_int8.json eval_holdout_v1_int8_decoder.json \
         gen_smoke.json console.log; do
  uvx --from huggingface_hub hf upload AlphaAvatar/aadistill-artifacts \
    "$RUN/$f" "stage3/s2_blocks_v1/$f" --repo-type model || fail "up_$f"
done
uvx --from huggingface_hub hf upload AlphaAvatar/aadistill-artifacts \
  artifacts/stage3/s2v1_artifact_hashes_2026-07-26.txt \
  stage3/s2_blocks_v1/s2v1_artifact_hashes_2026-07-26.txt --repo-type model || fail up_hashes

echo "MARKER:POST_DONE" >> /workspace/run_markers.log
