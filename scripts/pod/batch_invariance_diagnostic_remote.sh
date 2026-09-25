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

# --- the a4 object ----------------------------------------------------------
# The rejected finding was measured on the 596M stage-1 student, NOT on the
# parent. Confirming, refining or superseding it means measuring the SAME
# checkpoint; the parent is measured too, because that is what C3's operators
# actually calibrate on. Reporting one as though it were the other is the kind
# of substitution this whole investigation exists to stop making.
say "fetching the 596M a4 checkpoint from the relay"
python3 - <<'PY'
import hashlib, os, pathlib, sys, time
from huggingface_hub import snapshot_download
PINNED = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"
for attempt in range(4):
    try:
        root = snapshot_download(
            "AlphaAvatar/aadistill-artifacts", repo_type="model",
            allow_patterns=["stage1/qwen3_0p6b_init_v0/checkpoint/*"],
            local_dir="/workspace/a4ckpt", token=os.environ["HF_TOKEN"])
        break
    except Exception as exc:
        print(f"  attempt {attempt+1}: {exc}", flush=True)
        time.sleep(10 * (attempt + 1))
else:
    sys.exit("A4 CHECKPOINT FETCH FAILED")
d = pathlib.Path(root) / "stage1/qwen3_0p6b_init_v0/checkpoint"
w = d / "model.safetensors"
got = hashlib.sha256(w.read_bytes()).hexdigest()
if got != PINNED:
    sys.exit(f"A4 CHECKPOINT HASH MISMATCH: {got} != {PINNED}")
print(f"  a4 checkpoint at {d}, weights sha256 {got}", flush=True)
PY
A4_OK=$?
A4_CKPT=/workspace/a4ckpt/stage1/qwen3_0p6b_init_v0/checkpoint
if [ "$A4_OK" -ne 0 ]; then
  # NOT fatal: the parent runs are the ones C3's decisions depend on, and a
  # session that reached them is worth more than one that refused at setup.
  say "a4 checkpoint unavailable; the parent runs proceed and this is recorded"
  note "a4_checkpoint_unavailable"
  A4_CKPT=""
fi

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
# Two OBJECTS and two RUNTIMES, ordered so the most decisive runs first: if the
# session is cut short, the answer C3 depends on already exists.
#
#   parent, C, sdpa    the 4B teacher under the SCIENCE runtime. This is the
#                      object C3's operators actually calibrate on, so this is
#                      the run a C3 reproducibility claim rests on.
#   parent, C, eager   same object and runtime, eager attention. The CPU
#                      rehearsal found eager and SDPA-MATH diverging while
#                      SDPA-FLASH was exact, so the backend the decision stages
#                      run under is not a detail.
#   a4_596m, C, sdpa   the checkpoint the REJECTED FINDING was measured on.
#                      Without it this session could only report something
#                      adjacent to the a4 claim, never CONFIRMED / REFINED /
#                      SUPERSEDED.
#   a4_596m, B, sdpa   the same object under the image's own torch 2.9.1+cu130,
#                      which is the closest reachable neighbour of a4's own
#                      unpinned runtime. This is the environment control: it
#                      asks whether the runtime alone moves the answer.
run_one() {
  local label="$1" python="$2" attn="$3" ckpt="$4"
  local dir="${OUTROOT}/${label}"
  mkdir -p "$dir"
  say "diagnostic ${label} (python=${python} attn=${attn:-default} ckpt=${ckpt:-hub-parent})"
  local t=$(date -u +%s)
  # Full output to a file that travels back with the report; only the tail to
  # the launcher log. A diagnostic whose stderr was truncated to fit a console
  # is a diagnostic that cannot explain its own failure.
  AAD_CONTAINER_IMAGE="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404" \
  AAD_WHEELHOUSE_SOURCE="AlphaAvatar/aadistill-artifacts:transfer/wheelhouse_cu128_cp312" \
  AAD_REQUIREMENTS="requirements-cu128.txt" \
  PYTHONPATH=src:scripts timeout "${RUN_MAX_S:-1500}" \
  "$python" scripts/validation/batch_invariance_diagnostic.py \
      --run-id "@RUN_ID@-${label}" --device cuda --dtype bfloat16 \
      ${attn:+--attn "$attn"} ${ckpt:+--checkpoint "$ckpt"} \
      --out "$dir" > "${dir}/stdout.log" 2>&1
  local rc=$?
  #: Per-run, not just per-session. The session bound would let one hung run
  #: consume the budget of the three that had not started, and those three are
  #: the comparison -- a single report cannot say whether the runtime or the
  #: object moved the answer. 1500 s against an expected 200-400 s.
  [ "$rc" = "124" ] && say "  ${label} EXCEEDED ${RUN_MAX_S:-1500}s and was stopped"
  tail -40 "${dir}/stdout.log"
  say "  ${label} rc=${rc} in $(( $(date -u +%s) - t ))s"
  echo "${label} rc=${rc}" >> "${OUTROOT}/run_status.txt"
}

cd /workspace/repo
run_one parent_C_science_sdpa  /opt/train/bin/python sdpa  ""
run_one parent_C_science_eager /opt/train/bin/python eager ""
if [ -n "$A4_CKPT" ]; then
  run_one a4_596m_C_science_sdpa /opt/train/bin/python sdpa "$A4_CKPT"
fi

# Environment B needs transformers; it comes from the SAME wheelhouse, pinned to
# the same 5.13.1, so the only variable between B and C is torch itself.
say "environment B: image python + wheelhouse transformers 5.13.1"
python3 -m pip install -q --break-system-packages --no-cache-dir --no-index \
  --find-links "$WHEELHOUSE" transformers==5.13.1 tokenizers safetensors huggingface_hub numpy 2>&1 | tail -3
if python3 -c "import torch, transformers, numpy" 2>/dev/null; then
  python3 -c "import torch;print('  env B torch', torch.__version__)"
  if [ -n "$A4_CKPT" ]; then
    run_one a4_596m_B_image_sdpa python3 sdpa "$A4_CKPT"
  else
    run_one parent_B_image_sdpa python3 sdpa ""
  fi
else
  say "  environment B not constructible; recording that rather than guessing"
  note "env_b_unavailable"
fi

say "done; reports:"
find "$OUTROOT" -name report.json -printf '  %p\n' 2>/dev/null || true
exit 0
