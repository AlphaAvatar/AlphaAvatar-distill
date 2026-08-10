#!/usr/bin/env bash
# Setup for E8 pod A: the contribution-guided depth search.
#
# Deliberately much smaller than a training pod. The search needs the teacher and
# the frozen calibration set and nothing else — no vLLM, no ladder pack, no
# corpus, no student checkpoint — so setup is short and the session is cheap.
#
# What is fatal here rather than discovered later:
#   * the calibration set not hashing to the frozen values in the committed relay
#     manifest — a different calibration set is a different selector;
#   * the teacher's RoPE base not resolving to 5,000,000 in /opt/train, which is
#     the transformers 4.x/5.x `rope_parameters` skew that silently moves every
#     number a session produces.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> TEACHER_READY
#          -> ROPE_OK -> TESTS_OK -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/e8a.status
mark() { echo "MARKER:$1"; echo "$(date -u +%FT%TZ) MARKER:$1" >>"$STATUS"; }
say()  { echo "[$(date -u +%T)] $*"; }

export HF_TOKEN="$(cat $WS/hf/token)"
export HF_HOME=/root/.cache/huggingface

say "apt: git, zstd"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git zstd >/dev/null
mark ENV_READY

say "installing huggingface_hub"
python3 -m pip install -q --no-input --break-system-packages \
    "huggingface_hub[hf_transfer]" 2>&1 | tail -3

cat > /workspace/fetch.py <<'FETCHEOF'
import os, shutil, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download
REPO = "AlphaAvatar/aadistill-artifacts"
TOKEN = os.environ["HF_TOKEN"]

def fetch(prefix, names, dest, tries=5):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        last = None
        for attempt in range(tries):
            try:
                p = hf_hub_download(REPO, f"{prefix}/{name}", repo_type="model",
                                    token=TOKEN)
                shutil.copy(p, dest / name)
                break
            except Exception as exc:                      # transient 5xx / CDN
                last = exc
                time.sleep(5 * (attempt + 1))
        else:
            sys.exit(f"FETCH FAILED {prefix}/{name}: {last}")
        print(f"  {prefix}/{name}", flush=True)
FETCHEOF

say "fetching the repo bundle"
python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
name = os.environ['BUNDLE_NAME']
p = hf_hub_download('AlphaAvatar/aadistill-artifacts', f'transfer/{name}',
                    repo_type='model', token=os.environ['HF_TOKEN'])
shutil.copy(p, f'/workspace/{name}')
print('bundle at', p)
"
rm -rf "$REPO"
git clone -q "$WS/$BUNDLE_NAME" "$REPO"
cd "$REPO"
git checkout -q "$SESSION_COMMIT"
git rev-parse HEAD
mark REPO_READY

# The calibration set IS the selector. Fetch it, then verify every hash the
# preregistration froze; a mismatch means the search would answer a different
# question than the one that was registered.
say "staging the frozen E8 calibration set"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
fetch('e8_inputs_20260810/calibration_v1',
      ['items.jsonl', 'docs.jsonl', 'general_docs.jsonl',
       'general_docs.manifest.json', 'manifest.json', 'leakage.json',
       'general_disjointness.json'],
      '/workspace/aad/artifacts/stage1/e8_calibration_v1')
"
cd "$REPO" && python3 - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
d = Path('artifacts/stage1/e8_calibration_v1')
man = json.loads((d / 'manifest.json').read_text())
items_sha = hashlib.sha256((d / 'items.jsonl').read_bytes()).hexdigest()
# Only two literals, both self-consistent inside the manifest. The file-level
# hash is taken FROM the manifest rather than transcribed here: the first version
# of this check pinned an items.jsonl hash copied from an intermediate build's
# console output, which would have aborted the pod at DATA_READY after a
# 45-minute setup. A hash that can be derived should never be re-typed.
FROZEN = {
    'content_sha256': 'd65c1f40e4837ea1bd5bcc33c68041a13b797c68f5be3c0686e0142ed761028f',
    'manifest_sha256': 'ecb72aa3b88818e93fb058d5d012e66274db9bc7b90234219501f0df86cef460',
}
for key in ('content_sha256', 'manifest_sha256'):
    if man.get(key) != FROZEN[key]:
        sys.exit(f'CALIBRATION {key} MISMATCH: {man.get(key)} != {FROZEN[key]}')
declared_items = man['outputs']['items']['sha256']
if items_sha != declared_items:
    sys.exit(f'CALIBRATION items.jsonl MISMATCH: {items_sha} != {declared_items}')
leak = json.loads((d / 'leakage.json').read_text())
if not leak.get('clean'):
    sys.exit(f'CALIBRATION LEAKAGE NOT CLEAN: {leak.get("findings")}')
dj = json.loads((d / 'general_disjointness.json').read_text())
if not dj.get('disjoint'):
    sys.exit('CALIBRATION GENERAL TEXT NOT DISJOINT')
print(f'calibration verified: {man["totals"]["items"]} items, '
      f'{man["totals"]["prediction_positions"]:,} positions, '
      f'{len(man["design"]["domains"])} domains, leakage clean, disjoint')
PYEOF
mark DATA_READY

say "training env via uv sync"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"
sed -i 's|url = "https://download.pytorch.org/whl/cpu"|url = "https://download.pytorch.org/whl/cu128"|' pyproject.toml
sed -i 's|name = "pytorch-cpu"|name = "pytorch-cu128"|' pyproject.toml
sed -i 's|torch = { index = "pytorch-cpu" }|torch = { index = "pytorch-cu128" }|' pyproject.toml
export UV_PROJECT_ENVIRONMENT=/opt/train
uv lock

# Cold-host tripwire, carried unchanged: setup time on an identical image, script
# and GPU has varied ~30x purely with how much the host had cached.
TRIP_S=${UV_TRIP_S:-360}
GRACE_S=${UV_GRACE_S:-180}
# Hard ceiling on renewed grace: a host still downloading after this
# is not worth waiting for even if it is technically progressing.
UV_MAX_S=${UV_MAX_S:-1500}
uv sync --group dev &
UV_PID=$!
t0=$(date -u +%s); graced=0
while kill -0 "$UV_PID" 2>/dev/null; do
  sleep 15
  el=$(( $(date -u +%s) - t0 ))
  [ "$el" -lt "$TRIP_S" ] && continue
  # Progress is measured across BOTH trees, and does not require the venv to
  # exist yet. The original clause looked only at /opt/train and only once
  # /opt/train/bin/python was present — but uv spends its first phase writing to
  # /root/.cache/uv, so a host downloading wheels at 5.5 MB/s was classified
  # cold. E8 pod A threw away two such hosts (1978 MB cached at 360s, which is
  # progress, not a hang) before this was noticed.
  #
  # Grace now renews while progress continues, bounded by UV_MAX_S so a genuinely
  # hung host still dies. The tripwire's purpose is to abandon *stalled* hosts,
  # not slow ones.
  if [ "$el" -lt "$UV_MAX_S" ]; then
    a=$(( $(du -sb /opt/train 2>/dev/null | cut -f1 || echo 0) \
        + $(du -sb /root/.cache/uv 2>/dev/null | cut -f1 || echo 0) )); sleep 20
    b=$(( $(du -sb /opt/train 2>/dev/null | cut -f1 || echo 0) \
        + $(du -sb /root/.cache/uv 2>/dev/null | cut -f1 || echo 0) ))
    if [ "$((b - a))" -gt 20000000 ]; then
      graced=$(( graced + 1 ))
      say "uv sync past ${TRIP_S}s but progressing ($(( (b-a)/1048576 )) MB/20s, grace ${graced}) — extending ${GRACE_S}s"
      TRIP_S=$(( TRIP_S + GRACE_S )); continue
    fi
  fi
  kill -9 "$UV_PID" 2>/dev/null || true
  cache=$(du -sm /root/.cache/uv 2>/dev/null | cut -f1 || echo 0)
  say "COLD HOST: uv sync unfinished after ${el}s (cache ${cache} MB). Abandoning."
  mark "HOST_COLD:${el}s:${cache}MB"
  exit 90                                        # the launcher redraws on 90
done
wait "$UV_PID" || { say "uv sync failed"; exit 1; }
say "uv sync completed in $(( $(date -u +%s) - t0 ))s"
/opt/train/bin/python -c "import torch, transformers; \
  assert torch.cuda.is_available(); \
  print('train torch', torch.__version__, torch.cuda.get_device_name(0), \
        '| transformers', transformers.__version__)"
mark TRAIN_ENV

say "downloading the teacher at the pinned revision"
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Thinking-2507', revision=os.environ['TEACHER_REVISION'],
                  token=os.environ['HF_TOKEN'],
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
print('teacher downloaded at', os.environ['TEACHER_REVISION'])
"
mark TEACHER_READY

# A wrong positional basis would invalidate every KL this pod measures, and it
# does not raise on its own. The search runs on the teacher, so the teacher is
# what gets checked.
say "checking the teacher's RoPE base resolves in /opt/train"
/opt/train/bin/python -c "
import os, sys, transformers
sys.path.insert(0, '/workspace/aad/src')
from transformers import AutoConfig
from aadistill.models.student import assert_rope_from_config, stored_rope_base
rev = os.environ['TEACHER_REVISION']
cfg = AutoConfig.from_pretrained('Qwen/Qwen3-4B-Thinking-2507', revision=rev)
stored = stored_rope_base(cfg)
base = assert_rope_from_config(cfg, 'teacher')
print(f'  transformers {transformers.__version__}: stored {stored:,.0f}, '
      f'runtime {base:,.0f} OK')
if abs(stored - 5_000_000) > 1:
    sys.exit(f'teacher records RoPE base {stored}, expected 5,000,000')
"
mark ROPE_OK

# Threads are capped, and the gate is time-boxed. Both because of one host:
# E8 pod B drew a 128-vCPU machine, torch created 208 threads for tiny-model
# tests, and the suite that runs in 70 s locally was still going after 66 minutes
# with 39,252 involuntary context switches. It was progressing, so no tripwire
# fired, and it silently consumed the session's budget for its second training
# arm. Oversubscription on a wide host is a host-shape hazard, not a code bug —
# so cap the threads, and if the suite still cannot finish, treat it as a bad draw
# and let the launcher redraw rather than eat the session.
say "CPU test suite (threads capped; wide hosts make torch oversubscribe)"
cd "$REPO"
set +e
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  timeout "${TESTS_MAX_S:-900}" /opt/train/bin/python -m pytest tests/ -q \
  --ignore=tests/data/test_recovery_corpus_pipeline.py > /workspace/pytest.log 2>&1
RC=$?
set -e
tail -4 /workspace/pytest.log
if [ "$RC" -eq 124 ]; then
  say "COLD HOST: the CPU test suite did not finish in ${TESTS_MAX_S:-900}s on $(nproc) vCPUs"
  mark "HOST_COLD:tests:${TESTS_MAX_S:-900}s:$(nproc)vcpu"
  exit 90
fi
[ "$RC" -eq 0 ] || { say "test suite failed rc=$RC"; exit 1; }
mark TESTS_OK

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mark SETUP_DONE
