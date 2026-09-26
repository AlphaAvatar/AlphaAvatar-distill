#!/usr/bin/env bash
# The pod side of the C3 batching-adoption pilot. Shipped as a FILE, never as
# an inline heredoc: an unquoted heredoc is expanded by the LOCAL shell, which
# has already executed a `pip install` on the dev box and wasted a paid pod.
#
# Placeholders @BRANCH@ @COMMIT@ @RUN_ID@ are substituted by the launcher.
#
# Setup is the batch-invariance payload's, minus the parts this pilot does not
# need. It builds ONLY the formal science environment -- /opt/train, python
# 3.12, torch 2.11.0+cu128, transformers 5.13.1, installed OFFLINE from the
# relay wheelhouse. There is no second runtime here: the a4-vs-formal question
# was settled by the batch-invariance investigation, and the adoption gate is a
# ratio of two clocks that must be taken under ONE pinned runtime.
#
# Also absent: the 596M a4 checkpoint. This pilot scores the pre-ATTENTION
# parent, which the prefix builds from the root teacher; fetching an artifact
# no stage reads would be paying to move bytes nothing consumes.
set -uo pipefail

say() { echo "[pod $(date -u +%H:%M:%S)] $*"; }
OUTROOT=/workspace/out
mkdir -p "$OUTROOT"
export HF_HOME=/workspace/hf
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONHASHSEED=7          # a per-process random hash has failed a gate before
export TOKENIZERS_PARALLELISM=false

note() { echo "$*" >> "${OUTROOT}/session_notes.txt"; }

say "host: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo '(no nvidia-smi)')"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  > "${OUTROOT}/host_gpu.txt" 2>/dev/null || true

# --- the fetch bootstrap ----------------------------------------------------
# The image's python has torch and no `huggingface_hub`, and every input below
# arrives through it. A previous attempt died here in 4 seconds for $0.0485
# because this step did not exist.
#
# PINNED to the wheelhouse's own 1.23.0: a bootstrap dependency used to
# download bytes, not part of any measured environment.
say "bootstrap: huggingface_hub==1.23.0 into the image python"
python3 -m pip install -q --no-input --break-system-packages \
    "huggingface_hub[hf_transfer]==1.23.0" 2>&1 | tail -3
python3 -c "import huggingface_hub as h; print('  huggingface_hub', h.__version__)" \
  || { say "BOOTSTRAP FAILED: no huggingface_hub"; note "bootstrap_failed"; exit 19; }
if python3 -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  say "  hf_transfer enabled"
fi

# --- source -----------------------------------------------------------------
cd /workspace
git clone --quiet --branch @BRANCH@ --depth 60 \
  https://github.com/AlphaAvatar/AlphaAvatar-distill.git repo || {
  say "CLONE FAILED"; note "clone_failed"; exit 20; }
cd /workspace/repo
git checkout --quiet @COMMIT@ || { say "COMMIT @COMMIT@ NOT IN THE CLONE"
  note "commit_missing"; exit 21; }
SOURCE_SHA=$(git rev-parse HEAD)
say "source ${SOURCE_SHA}"
printf '%s\n' "$SOURCE_SHA" > "${OUTROOT}/source_sha"

# --- science inputs: the frozen calibration mixture -------------------------
# artifacts/ is gitignored, so the 67-item mixture is NOT in the clone. Without
# it nothing can run: the prefix resolves `calib.domain_balanced@v1` and
# `calib.reasoning_heavy@v2` from it, and so does the causal scorer.
say "fetching the frozen calibration mixture from the relay"
python3 - <<'PY'
import hashlib, os, pathlib, shutil, sys, time
from huggingface_hub import hf_hub_download
DEST = pathlib.Path("/workspace/repo/artifacts/stage1/e8_calibration_v1")
DEST.mkdir(parents=True, exist_ok=True)
PINNED = "c7202338109e459b17b70456461e8f304fadea7929ea547accee21adbbe7fd0b"
for attempt in range(4):
    try:
        p = hf_hub_download("AlphaAvatar/aadistill-artifacts", repo_type="model",
                            filename="e8_inputs_20260810/calibration_v1/items.jsonl",
                            token=os.environ["HF_TOKEN"])
        break
    except Exception as exc:
        print(f"  attempt {attempt+1}: {exc}", flush=True)
        time.sleep(8 * (attempt + 1))
else:
    sys.exit("CALIBRATION FETCH FAILED")
got = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if got != PINNED:
    sys.exit(f"CALIBRATION HASH MISMATCH: {got} != {PINNED}")
shutil.copy(p, DEST / "items.jsonl")
print(f"  items.jsonl staged, sha256 {got}", flush=True)
PY
[ $? -eq 0 ] || { say "CALIBRATION INPUT UNAVAILABLE"; note "calibration_missing"; exit 22; }

# --- the root teacher -------------------------------------------------------
# Public hub. Fetched HERE rather than lazily inside the driver so a network
# failure costs setup time and not a half-built prefix. The revision is the
# one the frozen path pins; `main` is a moving reference and would replay from
# whatever it pointed at today.
ROOT_REV=$(python3 -c "
import json;print(json.load(open('logs/stages/stage-1/phase_c1/runs/attempt9/c1_replay_record.json'))['path']['root_revision'])")
say "fetching the root teacher at ${ROOT_REV}"
python3 - "$ROOT_REV" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
rev = sys.argv[1]
for attempt in range(4):
    try:
        p = snapshot_download("Qwen/Qwen3-4B-Thinking-2507", revision=rev,
                              allow_patterns=["*.json", "*.safetensors",
                                              "*.txt", "*.jinja"])
        print("  root at", p, flush=True)
        break
    except Exception as exc:
        print(f"  attempt {attempt+1}: {exc}", flush=True)
        time.sleep(10 * (attempt + 1))
else:
    sys.exit("ROOT FETCH FAILED")
PY
[ $? -eq 0 ] || { say "ROOT TEACHER UNAVAILABLE"; note "root_missing"; exit 23; }

# --- the formal science environment -----------------------------------------
say "building /opt/train offline from the relay wheelhouse"
command -v uv >/dev/null || curl -LsSf "https://astral.sh/uv/0.11.11/install.sh" | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_DOWNLOADS=never

WHEELHOUSE=/workspace/wheelhouse
python3 - <<'PY'
import sys, time
import os
from huggingface_hub import snapshot_download
for attempt in range(4):
    try:
        snapshot_download("AlphaAvatar/aadistill-artifacts", repo_type="model",
                          allow_patterns=["transfer/wheelhouse_cu128_cp312/*"],
                          local_dir="/workspace/wh", token=os.environ["HF_TOKEN"])
        break
    except Exception as exc:
        print(f"  wheelhouse attempt {attempt+1}: {exc}", flush=True)
        time.sleep(10 * (attempt + 1))
else:
    sys.exit("WHEELHOUSE FETCH FAILED")
PY
mkdir -p "$WHEELHOUSE"
cp /workspace/wh/transfer/wheelhouse_cu128_cp312/*.whl "$WHEELHOUSE"/ 2>/dev/null || true
NWHL=$(ls "$WHEELHOUSE"/*.whl 2>/dev/null | wc -l)
say "wheelhouse: ${NWHL} wheels, $(du -sh "$WHEELHOUSE" 2>/dev/null | cut -f1)"
[ "$NWHL" -ge 91 ] || { say "WHEELHOUSE TOO SMALL (${NWHL})"; note "wheelhouse_incomplete:${NWHL}"; exit 24; }

cd /workspace/repo
# The pyproject pins a CPU torch index; the formal setup rewrites it to cu128
# before installing. Same three edits, same order.
sed -i 's|url = "https://download.pytorch.org/whl/cpu"|url = "https://download.pytorch.org/whl/cu128"|' pyproject.toml
sed -i 's|name = "pytorch-cpu"|name = "pytorch-cu128"|' pyproject.toml
sed -i 's|torch = { index = "pytorch-cpu" }|torch = { index = "pytorch-cu128" }|' pyproject.toml

uv venv /opt/train --python 3.12 \
  || { say "NO PYTHON 3.12"; note "py312_missing"; exit 25; }
uv pip install --python /opt/train/bin/python --offline --no-index \
  --find-links "$WHEELHOUSE" -r requirements-cu128.txt \
  || { say "OFFLINE INSTALL FAILED"; note "wheelhouse_unsatisfied"; exit 26; }
uv pip install --python /opt/train/bin/python --offline --no-index \
  --find-links "$WHEELHOUSE" --no-deps -e . \
  || { say "PROJECT INSTALL FAILED"; note "project_install_failed"; exit 27; }
# The pyproject edits above are LOCAL to the pod and must not reach the
# recorded source identity; restore them so `git status` describes the commit.
git checkout -- pyproject.toml 2>/dev/null || true

/opt/train/bin/python - <<'PY' | tee "${OUTROOT}/runtime.txt"
import json, torch, transformers
print(json.dumps({
    "torch": torch.__version__, "transformers": transformers.__version__,
    "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
}, indent=1))
PY
/opt/train/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { say "NO CUDA UNDER /opt/train"; note "no_cuda"; exit 28; }

# --- a $0 self-check before the science -------------------------------------
# The three process-global registries the driver needs are EMPTY in a fresh
# interpreter, and two of them were missing from the driver until a toy run
# found them -- each after the stage before it had succeeded. This costs
# seconds and would have caught both before the root teacher was resident.
say "preflight: the driver's own toy sequence, on this interpreter"
PYTHONPATH=src:scripts /opt/train/bin/python scripts/pod/c3_batching_pilot_driver.py \
    --toy --device cpu --out "${OUTROOT}/preflight" > "${OUTROOT}/preflight.log" 2>&1
PRE_RC=$?
if [ "$PRE_RC" -ne 0 ]; then
  say "PREFLIGHT FAILED (rc=${PRE_RC}); the paid sequence would fail the same way"
  tail -30 "${OUTROOT}/preflight.log"
  note "preflight_failed"
  exit 29
fi
say "preflight ok: $(python3 -c "import json;print(json.load(open('${OUTROOT}/preflight/pilot_result.json'))['verdict'])" 2>/dev/null || echo '(no verdict)')"

# --- the pilot ---------------------------------------------------------------
say "PILOT: one prefix replay at B=1, then both arms from that one parent"
PYTHONPATH=src:scripts /opt/train/bin/python scripts/pod/c3_batching_pilot_driver.py \
    --out "${OUTROOT}/pilot" 2>&1 | tee "${OUTROOT}/pilot.log"
RC=${PIPESTATUS[0]}
say "pilot rc=${RC}"

# Evidence is written stage by stage, so there is something to fetch on every
# path out of here. Nothing below decides anything; the launcher collects.
if [ -f "${OUTROOT}/pilot/pilot_summary.json" ]; then
  say "verdict: $(python3 -c "import json;print(json.load(open('${OUTROOT}/pilot/pilot_summary.json')).get('verdict'))" 2>/dev/null)"
fi
exit "$RC"
