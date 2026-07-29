#!/usr/bin/env bash
# Pod-side driver: engine benchmark -> mechanical decision -> pilot corpus.
#
# Session plan and pre-registered rules:
#   logs/proposals/2026-07-29_engine_benchmark.md
#
# Runs unattended. Every stage writes an arm-scoped marker so the dev-box driver
# can tell progress from failure without parsing logs, following the protocol in
# scripts/pod/AGENTS.md.
#
# The engine installs are pod-side ONLY and deliberately not in the repo
# lockfile: the dev box is CPU-only and vLLM/SGLang are CUDA-only, so adding
# them to pyproject.toml would break `uv sync` on the machine where all the
# cheap work happens (P8.1). An install that fails is an integration-cost
# finding, recorded and skipped, not retried forever.
set -uo pipefail

WORK=/workspace
REPO=$WORK/aad                      # where setup.sh clones the bundle
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=$WORK/hf
export PATH="$HOME/.local/bin:$PATH"
OUT_BENCH=artifacts/bench/engines_v0
OUT_GEN=artifacts/stage2_v2/pilot
MARKERS=$WORK/markers
mkdir -p "$MARKERS"

# Budget (P6). Generation stops cleanly at a batch boundary when this is hit.
GEN_MAX_HOURS=${GEN_MAX_HOURS:-3.0}
HOURLY_USD=${HOURLY_USD:-0.86}
N_PROMPTS=${N_PROMPTS:-32}
LIMIT_PER_SLICE=${LIMIT_PER_SLICE:-200}

mark() { echo "$(date -u +%FT%TZ) $1" | tee -a "$MARKERS/session.log" > "$MARKERS/$1"; }
log()  { echo "$(date -u +%FT%TZ) $*" | tee -a "$MARKERS/session.log"; }

cd "$REPO" || { mark "FAILED:no_repo"; exit 1; }

# --- engine installs -------------------------------------------------------
# Each is independent: SGLang failing must not cost us the vLLM arm.
#
# `--python` is required and is not optional pedantry: `uv pip` does NOT read
# UV_PROJECT_ENVIRONMENT (that governs `uv sync`/`uv run`), so without it every
# install fails instantly with "No virtual environment found" and both serving
# arms get silently recorded as integration failures. Measured the hard way on
# this pod at 05:43 UTC — both installs "failed" in the same second, which is
# the tell that no download was ever attempted.
VENV_PY=/root/venv/bin/python
install_engine() { # install_engine <name> <pip-spec>
  local name=$1 spec=$2 t0 t1
  log "installing $name"
  t0=$(date +%s)
  if uv pip install --python "$VENV_PY" "$spec" > "$MARKERS/install_$name.log" 2>&1; then
    t1=$(date +%s)
    ENGINES=$ENGINES,$name
    log "$name ok in $((t1 - t0))s: $(uv pip show --python "$VENV_PY" "$name" 2>/dev/null | awk '/^Version/{print $2}')"
    mark "$(echo "$name" | tr '[:lower:]' '[:upper:]')_INSTALLED"
  else
    log "$name install FAILED after $(($(date +%s) - t0))s — arm skipped; see install_$name.log"
    tail -5 "$MARKERS/install_$name.log" | while read -r l; do log "  | $l"; done
  fi
}

ENGINES=hf
if [ -n "${ENGINES_OVERRIDE:-}" ]; then
  # Set when the installs have already been attempted and their outcome decided
  # out of band — see the 2026-07-29 session, where both serving engines turned
  # out to be incompatible with this project's pinned stack and re-running the
  # installs would have re-broken the restored environment.
  ENGINES=$ENGINES_OVERRIDE
  log "install step skipped; engines fixed to: $ENGINES"
else
  install_engine vllm vllm
  install_engine sglang "sglang[all]"
fi
log "engines to benchmark: $ENGINES"

# --- benchmark -------------------------------------------------------------
log "benchmark starting"
uv run python scripts/bench_engines.py \
  --engines "$ENGINES" \
  --n-prompts "$N_PROMPTS" \
  --max-new-tokens 4096 \
  --hourly-usd "$HOURLY_USD" \
  --out "$OUT_BENCH" 2>&1 | tee -a "$MARKERS/bench.log"

if [ ! -f "$REPO/$OUT_BENCH/decision.json" ]; then
  log "benchmark produced no decision.json"
  mark "BENCH_FAILED"
  exit 1
fi
mark "BENCH_DONE"
log "decision: $(cat "$REPO/$OUT_BENCH/decision.json" | tr -d '\n')"

# --- pilot corpus ----------------------------------------------------------
# Rule R4: no reference arm, no decision, no corpus. `--engine-from` exits
# non-zero on a null winner, so this is belt and braces rather than the only
# guard.
WINNER=$(uv run python -c "import json;print(json.load(open('$REPO/$OUT_BENCH/decision.json')).get('winner') or '')")
if [ -z "$WINNER" ]; then
  log "no winner selected (R4) — skipping generation, benchmark still uploads"
  mark "GEN_SKIPPED"
else
  # Take the batch size from the sweep rather than a guess. At cap 4096 the KV
  # cache for this teacher is tens of GB, and the 2026-07-29 session OOMed at
  # batch 8 — so the largest size the benchmark actually completed is the only
  # one known to fit. Falls back to 2 if the label cannot be parsed.
  GEN_BS=$(uv run python -c "
import json
runs = json.load(open('$REPO/$OUT_BENCH/report.json'))['arms'][0].get('runs', [])
ok = [r for r in runs if not r.get('oom') and r.get('tokens_per_s')]
print(max((int(r['label'].split('=')[1]) for r in ok if '=' in r['label']), default=2))
" 2>/dev/null || echo 2)
  log "generation starting on engine=$WINNER, batch_size=$GEN_BS, budget ${GEN_MAX_HOURS}h"
  uv run python scripts/generate_teacher_answers.py \
    --engine-from "$OUT_BENCH/decision.json" \
    --limit-per-slice "$LIMIT_PER_SLICE" \
    --n 4 \
    --batch-size "$GEN_BS" \
    --max-new-tokens 4096 \
    --max-hours "$GEN_MAX_HOURS" \
    --out "$OUT_GEN" 2>&1 | tee -a "$MARKERS/generate.log"
  if [ -f "$REPO/$OUT_GEN/manifest.json" ]; then
    mark "GEN_DONE"
  else
    log "generation produced no manifest"
    mark "GEN_FAILED"
  fi
fi

# --- hashes + upload -------------------------------------------------------
# Hash before upload so the dev box can verify the relay end to end (P4).
find "$REPO/$OUT_BENCH" "$REPO/$OUT_GEN" -type f 2>/dev/null \
  | sort | xargs -r sha256sum > "$MARKERS/hashes_out.txt"
log "hashed $(wc -l < "$MARKERS/hashes_out.txt") output files"

if [ -f "$WORK/hf/token" ]; then
  HF_TOKEN=$(cat "$WORK/hf/token")
  export HF_TOKEN
  uv run python - "$REPO" "$OUT_BENCH" "$OUT_GEN" <<'PY' 2>&1 | tee -a "$MARKERS/upload.log"
import sys, os
from pathlib import Path
from huggingface_hub import HfApi

repo, *outs = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
for out in outs:
    src = Path(repo) / out
    if not src.exists():
        print(f"skip {out}: not present")
        continue
    api.upload_folder(
        folder_path=str(src),
        path_in_repo=f"engine_bench_20260729/{out}",
        repo_id="AlphaAvatar/aadistill-artifacts",
        repo_type="model",
    )
    print(f"uploaded {out}")
PY
  mark "UPLOAD_DONE"
else
  log "no HF token at $WORK/hf/token — outputs stay pod-local"
  mark "UPLOAD_SKIPPED"
fi

mark "SESSION_DONE"
log "session complete"
