#!/usr/bin/env bash
# Pod-side driver: isolated-venv vLLM server benchmark + openmath cap arm.
#
# Plan and pre-registered rules:
#   logs/proposals/2026-07-30_isolated_engine_and_cap.md
#
# The whole point of this session is that vLLM gets its **own venv and its own
# torch build**. On 2026-07-29 vLLM could not be imported into the project env
# (its extension wants CUDA 13 against our cu128) and SGLang could only be
# installed by downgrading transformers a major version. So here it never
# touches the project environment: it lives in /opt/vllm-venv, runs as a
# separate process, and is reached over HTTP.
#
# No corpus is built. No refusal data is generated. Slices are the dense
# baseline's in-scope four (decision 2026-07-30).
set -uo pipefail

WORK=/workspace
REPO=$WORK/aad
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=$WORK/hf                 # shared, so the teacher downloads once
export PATH="$HOME/.local/bin:$PATH"
MARKERS=$WORK/markers
mkdir -p "$MARKERS"

VLLM_VENV=/opt/vllm-venv
PORT=${PORT:-8000}
MODEL=Qwen/Qwen3-4B-Thinking-2507
REVISION=768f209d9ea81521153ed38c47d515654e938aea
OUT_BENCH=artifacts/bench/engines_v1
OUT_CAP=artifacts/stage2_v2/openmath_cap16384

CAP_MAX_HOURS=${CAP_MAX_HOURS:-1.0}
HOURLY_USD=${HOURLY_USD:-0.99}
N_PROMPTS=${N_PROMPTS:-10}

mark() { echo "$(date -u +%FT%TZ) $1" | tee -a "$MARKERS/session.log" > "$MARKERS/$1"; }
log()  { echo "$(date -u +%FT%TZ) $*" | tee -a "$MARKERS/session.log"; }

cd "$REPO" || { mark "FAILED:no_repo"; exit 1; }

# --- A1: isolated venv + vLLM ----------------------------------------------
ENGINES=hf
SERVER_PID=""
log "creating isolated venv at $VLLM_VENV"
if uv venv "$VLLM_VENV" --python 3.12 > "$MARKERS/vllm_install.log" 2>&1 \
   && uv pip install --python "$VLLM_VENV/bin/python" vllm >> "$MARKERS/vllm_install.log" 2>&1; then
  VLLM_VER=$("$VLLM_VENV/bin/python" -c 'import vllm;print(vllm.__version__)' 2>>"$MARKERS/vllm_install.log")
  if [ -n "$VLLM_VER" ]; then
    log "vllm $VLLM_VER imports in its own venv (project env untouched)"
    mark "VLLM_VENV_READY"
  else
    log "vllm installed but does not import even in isolation — see vllm_install.log"
    tail -3 "$MARKERS/vllm_install.log" | while read -r l; do log "  | $l"; done
  fi
else
  log "isolated vllm install FAILED — see vllm_install.log (abort rule A1)"
  tail -5 "$MARKERS/vllm_install.log" | while read -r l; do log "  | $l"; done
  VLLM_VER=""
fi

# --- start the server -------------------------------------------------------
if [ -n "${VLLM_VER:-}" ]; then
  log "starting vllm serve on :$PORT"
  HF_HOME=$HF_HOME setsid nohup "$VLLM_VENV/bin/vllm" serve "$MODEL" \
    --revision "$REVISION" --port "$PORT" --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.55 \
    < /dev/null > "$MARKERS/vllm_serve.log" 2>&1 &
  SERVER_PID=$!
  # Health-poll rather than sleep: model download + load dominates and varies.
  for i in $(seq 1 60); do
    sleep 20
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
      log "server healthy after $((i * 20))s"
      mark "VLLM_SERVER_UP"
      ENGINES=hf,vllm_server
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "vllm serve died during startup — see vllm_serve.log"
      tail -5 "$MARKERS/vllm_serve.log" | while read -r l; do log "  | $l"; done
      break
    fi
  done
  if [ "$ENGINES" = "hf" ]; then
    log "server never became healthy (abort rule A1) — continuing with hf only"
  fi
fi

# --- engine benchmark -------------------------------------------------------
# gpu-memory-utilization 0.55 leaves room for the in-stack reference arm to hold
# the same teacher in the same GPU at the same time; both arms must run in one
# process to compare tokens.
log "benchmark starting, engines=$ENGINES"
uv run python scripts/bench_engines.py \
  --engines "$ENGINES" \
  --slices rag_evidence,multihop_qa,gsm8k,openmath \
  --n-prompts "$N_PROMPTS" \
  --max-new-tokens 4096 \
  --hf-batch-sizes 4 \
  --hourly-usd "$HOURLY_USD" \
  --vllm-server-url "http://127.0.0.1:$PORT" \
  --out "$OUT_BENCH" 2>&1 | tee -a "$MARKERS/bench.log"
[ -f "$REPO/$OUT_BENCH/report.json" ] && mark "BENCH_DONE" || mark "BENCH_FAILED"

# Free the server's GPU memory before the cap arm, which needs a large KV cache.
if [ -n "$SERVER_PID" ]; then
  log "stopping vllm server"
  kill "$SERVER_PID" 2>/dev/null
  sleep 20
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 10
fi

# --- openmath cap arm (in-stack, for comparability with the 4096 control) ----
# n=2 and batch 2: KV cache for 4 sequences x 16,384 tokens on this teacher is
# ~48 GB and does not fit the card. See the proposal's corrected scope.
log "openmath cap arm starting (cap 16384, n=2, batch 2, budget ${CAP_MAX_HOURS}h)"
uv run python scripts/generate_teacher_answers.py \
  --slices openmath --limit-per-slice 10 --n 2 --batch-size 2 \
  --max-new-tokens 16384 --max-hours "$CAP_MAX_HOURS" \
  --out "$OUT_CAP" 2>&1 | tee -a "$MARKERS/cap.log"
[ -f "$REPO/$OUT_CAP/manifest.json" ] && mark "CAP_DONE" || mark "CAP_FAILED"

# --- hashes + upload --------------------------------------------------------
find "$REPO/$OUT_BENCH" "$REPO/$OUT_CAP" -type f 2>/dev/null \
  | sort | xargs -r sha256sum > "$MARKERS/hashes_out.txt"
log "hashed $(wc -l < "$MARKERS/hashes_out.txt") output files"

mark "SESSION_DONE"
log "session complete"
