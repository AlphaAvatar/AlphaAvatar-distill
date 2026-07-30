#!/usr/bin/env bash
# Pod-side driver: build the Stage 3 teacher-target corpus on a vLLM pod.
#
# Pre-registration: logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md
#
# This runs INSIDE vLLM's own official image (`vllm/vllm-openai:v0.26.0`), which
# is the whole point: decision 2026-07-30 says an engine is measured and used in
# its own current image on a host that image supports, never bent backwards to
# fit the training environment. That image already ships torch, transformers and
# vLLM, so there is no `uv sync` here and the project env is never built:
#   * the server is the image's own `vllm serve`;
#   * the client is the image's python with `src/` on PYTHONPATH.
# The server adapter uses only stdlib `urllib`, so the client needs nothing that
# the image does not already have.
#
# Markers (read by the dev-box driver): SERVER_UP, GEN_DONE / GEN_FAILED,
# HASHED, UPLOAD_DONE / UPLOAD_SKIPPED, SESSION_DONE.
set -uo pipefail

WORK=/workspace
REPO=$WORK/aad
export HF_HOME=$WORK/hf
MARKERS=$WORK/markers
mkdir -p "$MARKERS"

MODEL=Qwen/Qwen3-4B-Thinking-2507
REVISION=768f209d9ea81521153ed38c47d515654e938aea
PORT=${PORT:-8000}
OUT_GEN=${OUT_GEN:-artifacts/stage2_v2/teacher_corpus_750}

# Budget and scope (P6). Every one of these is pre-registered.
LIMIT_PER_SLICE=${LIMIT_PER_SLICE:-188}     # x4 slices = 752 prompts
N_CAND=${N_CAND:-2}                          # both sampled, no greedy
MAX_NEW=${MAX_NEW:-4096}                     # openmath cap stays 4096
GEN_MAX_HOURS=${GEN_MAX_HOURS:-2.5}
SLICES=${SLICES:-rag_evidence,multihop_qa,gsm8k,openmath}
SELECT=${SELECT:-stride}
HF_REPO=${HF_REPO:-AlphaAvatar/aadistill-artifacts}
HF_PREFIX=${HF_PREFIX:-stage3_teacher_corpus_20260730}

# `--max-model-len` must cover prompt + generation, not just generation. The
# 2026-07-30 session lost both cap-8192 cells by setting it to the cap alone,
# leaving no room for a 410-token prompt. Worst in-scope prompt is 2,765 tokens
# (preflight 4), so 4096 + 2765 + overhead fits inside 8192 with headroom.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

mark() { echo "$(date -u +%FT%TZ) $1" | tee -a "$MARKERS/session.log" > "$MARKERS/$1"; }
log()  { echo "$(date -u +%FT%TZ) $*" | tee -a "$MARKERS/session.log"; }

cd "$REPO" || { mark "SESSION_FAILED:no_repo"; exit 1; }

PY=$(command -v python3 || command -v python)
export PYTHONPATH=$REPO/src
log "image python: $PY ($($PY -c 'import sys;print(sys.version.split()[0])'))"
log "torch $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"
log "transformers $($PY -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo MISSING)"
log "vllm $($PY -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo MISSING)"
$PY -c 'import torch;assert torch.cuda.is_available()' 2>/dev/null \
  || { log "no CUDA visible to the image python"; mark "SESSION_FAILED:no_cuda"; exit 1; }
log "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"

# --- server ----------------------------------------------------------------
log "starting vllm serve on :$PORT (max_model_len=$MAX_MODEL_LEN)"
setsid nohup vllm serve "$MODEL" \
  --revision "$REVISION" --port "$PORT" --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
  < /dev/null > "$MARKERS/vllm_serve.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$MARKERS/server.pid"
log "server pid $SERVER_PID"

# Health-poll rather than sleep: the first run downloads ~8 GB of weights.
SERVER_UP=0
for i in $(seq 1 90); do
  sleep 20
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    SERVER_UP=1
    log "server healthy after $((i * 20))s"
    mark "SERVER_UP"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "vllm serve died during startup — see vllm_serve.log"
    tail -20 "$MARKERS/vllm_serve.log" | while read -r l; do log "  | $l"; done
    break
  fi
done
if [ "$SERVER_UP" != "1" ]; then
  mark "SESSION_FAILED:server_down"
  exit 1
fi

# --- generation ------------------------------------------------------------
# `--device cpu` keeps the client's reference copy of the teacher off the card:
# it is loaded only for its generation_config stop ids and the tokenizer, while
# the server owns 90% of GPU memory for its KV cache.
log "generation starting: $SLICES, $LIMIT_PER_SLICE/slice ($SELECT), n=$N_CAND, cap=$MAX_NEW, budget ${GEN_MAX_HOURS}h"
$PY scripts/rollout/generate_teacher_answers.py \
  --model "$MODEL@$REVISION" \
  --engine vllm_server --server-url "http://127.0.0.1:$PORT" \
  --slices "$SLICES" \
  --limit-per-slice "$LIMIT_PER_SLICE" --select "$SELECT" \
  --n "$N_CAND" \
  --max-new-tokens "$MAX_NEW" \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --snapshot \
  --device cpu \
  --max-hours "$GEN_MAX_HOURS" \
  --out "$OUT_GEN" 2>&1 | tee -a "$MARKERS/generate.log"

if [ -f "$REPO/$OUT_GEN/manifest.json" ]; then
  mark "GEN_DONE"
  log "accepted summary: $($PY -c "
import json;m=json.load(open('$REPO/$OUT_GEN/manifest.json'))
print('complete',m['complete'],'prompts',m['prompts_generated'],'/',m['prompts_requested'])
for k,v in m['slices'].items():
    print(' ',k,'accept@n',v.get('accept_at_n'),'accept@1',v.get('accept_at_1'))" 2>&1)"
else
  log "generation produced no manifest"
  mark "GEN_FAILED"
fi

# Stop the server by PID before anything else touches the GPU. Never `pkill -f`:
# it matches this script's own command line, which has cost this project time
# three times (STATE 10).
if kill -0 "$SERVER_PID" 2>/dev/null; then
  log "stopping vllm server (pid $SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null
  sleep 15
fi

# --- hashes + upload -------------------------------------------------------
# Hash before upload so the dev box can verify the relay end to end (P4).
find "$REPO/$OUT_GEN" -type f 2>/dev/null | sort | xargs -r sha256sum \
  > "$MARKERS/hashes_out.txt"
log "hashed $(wc -l < "$MARKERS/hashes_out.txt") output files"
mark "HASHED"

if [ -f "$WORK/hf/token" ]; then
  HF_TOKEN=$(cat "$WORK/hf/token") $PY - "$REPO/$OUT_GEN" "$HF_REPO" "$HF_PREFIX" \
    <<'PY' 2>&1 | tee -a "$MARKERS/upload.log"
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

src, repo_id, prefix = sys.argv[1:4]
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path=src, path_in_repo=prefix,
                  repo_id=repo_id, repo_type="model")
print("uploaded", src, "->", f"{repo_id}:{prefix}")
for f in api.list_repo_files(repo_id=repo_id, repo_type="model"):
    if f.startswith(prefix):
        print("  remote:", f)
PY
  if [ "${PIPESTATUS[0]}" = "0" ]; then mark "UPLOAD_DONE"; else mark "UPLOAD_FAILED"; fi
else
  log "no HF token at $WORK/hf/token — outputs stay pod-local"
  mark "UPLOAD_SKIPPED"
fi

mark "SESSION_DONE"
log "session complete"
