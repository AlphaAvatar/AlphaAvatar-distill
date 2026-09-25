#!/usr/bin/env bash
# The pod side of the batch-invariance diagnostic. Shipped as a file, never as
# an inline heredoc: an unquoted heredoc is expanded by the LOCAL shell, which
# has already executed a `pip install` on the dev box and wasted a paid pod.
#
# Placeholders @BRANCH@ @COMMIT@ @RUN_ID@ are substituted by the launcher.
#
# It builds the environment the FORMAL SCIENCE runs in -- /opt/train, python
# 3.12, torch 2.11.0+cu128 and transformers 5.13.1 installed OFFLINE from the
# relay wheelhouse -- and then, on the same pod and the same weights, also
# measures under the image's own python (torch 2.9.1+cu130). Which runtime a
# number came from is the thing the rejected a4 finding could not say.
set -uo pipefail

say() { echo "[pod $(date -u +%H:%M:%S)] $*"; }
OUTROOT=/workspace/out
mkdir -p "$OUTROOT"
export HF_HOME=/workspace/hf
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONHASHSEED=7          # a per-process random hash has failed a gate before
export TOKENIZERS_PARALLELISM=false

# Record the failure reason even when a later step never runs.
note() { echo "$*" >> "${OUTROOT}/session_notes.txt"; }

say "host: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo '(no nvidia-smi)')"
say "image python: $(python3 -c 'import torch;print("torch",torch.__version__)' 2>&1 | tail -1)"

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

# --- science inputs: the calibration mixture --------------------------------
# artifacts/ is gitignored, so the frozen mixture is NOT in the clone. Without
# it the diagnostic silently falls back to synthetic token ids, and a claim
# about random ids is not a claim about the decision C3 would take.
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

# --- the parent -------------------------------------------------------------
# Public hub, so no relay and no dev-box uplink. Fetched ONCE, before either
# interpreter runs, so both read identical bytes from the same cache.
say "fetching the parent (public hub)"
python3 - <<'PY'
import sys, time
from huggingface_hub import snapshot_download
for attempt in range(4):
    try:
        p = snapshot_download("Qwen/Qwen3-4B-Thinking-2507",
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        print("  parent at", p, flush=True)
        break
    except Exception as exc:
        print(f"  attempt {attempt+1}: {exc}", flush=True)
        time.sleep(10 * (attempt + 1))
else:
    sys.exit("PARENT FETCH FAILED")
PY
[ $? -eq 0 ] || { say "PARENT UNAVAILABLE"; note "parent_missing"; exit 23; }

# --- environment C: the one the science runs in -----------------------------
say "building /opt/train offline from the relay wheelhouse"
command -v uv >/dev/null || curl -LsSf "https://astral.sh/uv/0.11.11/install.sh" | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_DOWNLOADS=never

WHEELHOUSE=/workspace/wheelhouse
python3 - <<'PY'
import os, sys, time
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

t0=$(date -u +%s)
uv venv /opt/train --python 3.12 \
  || { say "NO PYTHON 3.12"; note "py312_missing"; exit 25; }
# `uv pip install` against exported pins, NOT `uv sync`: a lock entry pins torch
# to a registry, and --find-links cannot override a registry-pinned source under
# --no-index. Offline and no-index together mean the paid path contains no PyPI.
uv pip install --python /opt/train/bin/python --offline --no-index \
  --find-links "$WHEELHOUSE" -r requirements-cu128.txt \
  || { say "OFFLINE INSTALL FAILED"; note "wheelhouse_unsatisfied"; exit 26; }
uv pip install --python /opt/train/bin/python --offline --no-index \
  --find-links "$WHEELHOUSE" --no-deps -e . \
  || { say "PROJECT INSTALL FAILED"; note "project_install_failed"; exit 27; }
say "offline install completed in $(( $(date -u +%s) - t0 ))s"

/opt/train/bin/python -c "
import torch, transformers
assert torch.cuda.is_available(), 'no CUDA in /opt/train'
print('  /opt/train: torch', torch.__version__, '| transformers', transformers.__version__,
      '|', torch.cuda.get_device_name(0))
" || { say "/opt/train UNUSABLE"; note "train_env_unusable"; exit 28; }

# --- the diagnostic ---------------------------------------------------------
# Three runs, each a complete report. Ordered so the most decisive comes first:
# if the session is cut short, the science environment's answer exists.
#
#   C-sdpa    the science runtime, the default attention backend
#   C-eager   the science runtime, eager attention -- the CPU rehearsal found
#             eager and sdpa-MATH diverging while FLASH was exact, so the
#             backend the decision stages run under is not a detail
#   B-image   the image's own torch 2.9.1+cu130, to test whether the runtime
#             itself explains the a4 result
run_one() {
  local label="$1" python="$2" attn="$3"
  local dir="${OUTROOT}/${label}"
  mkdir -p "$dir"
  say "diagnostic ${label} (python=${python} attn=${attn:-default})"
  local t=$(date -u +%s)
  # Full output to a file that travels back with the report; only the tail to
  # the launcher log. A diagnostic whose stderr was truncated to fit a console
  # is a diagnostic that cannot explain its own failure.
  AAD_CONTAINER_IMAGE="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404" \
  AAD_WHEELHOUSE_SOURCE="AlphaAvatar/aadistill-artifacts:transfer/wheelhouse_cu128_cp312" \
  AAD_REQUIREMENTS="requirements-cu128.txt" \
  PYTHONPATH=src:scripts "$python" scripts/validation/batch_invariance_diagnostic.py \
      --run-id "@RUN_ID@-${label}" --device cuda --dtype bfloat16 \
      ${attn:+--attn "$attn"} --out "$dir" > "${dir}/stdout.log" 2>&1
  local rc=$?
  tail -40 "${dir}/stdout.log"
  say "  ${label} rc=${rc} in $(( $(date -u +%s) - t ))s"
  echo "${label} rc=${rc}" >> "${OUTROOT}/run_status.txt"
}

cd /workspace/repo
run_one C_science_sdpa /opt/train/bin/python sdpa
run_one C_science_eager /opt/train/bin/python eager

# Environment B needs transformers; it comes from the SAME wheelhouse, pinned to
# the same 5.13.1, so the only variable between B and C is torch itself.
say "environment B: image python + wheelhouse transformers 5.13.1"
python3 -m pip install -q --break-system-packages --no-cache-dir --no-index \
  --find-links "$WHEELHOUSE" transformers==5.13.1 tokenizers safetensors huggingface_hub numpy 2>&1 | tail -3
if python3 -c "import torch, transformers, numpy" 2>/dev/null; then
  run_one B_image_sdpa python3 sdpa
else
  say "  environment B not constructible; recording that rather than guessing"
  note "env_b_unavailable"
fi

say "done; reports:"
find "$OUTROOT" -name report.json -printf '  %p\n' 2>/dev/null || true
exit 0
