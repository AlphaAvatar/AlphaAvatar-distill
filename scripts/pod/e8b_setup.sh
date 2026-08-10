#!/usr/bin/env bash
# Setup for E8 pod B: initialization-time diagnostics, then the two-seed 2.96M
# recovery, then the frozen autonomous evaluation.
#
# Two initializations are staged and BOTH are measured here, on this device, by
# the same evaluator: the pinned positional control `86fbba78…` and the
# contribution-guided treatment built on the dev box from pod A's frozen map. The
# control is remeasured rather than inherited — a historical NLL from a different
# reader path is not a comparison (decisions.md, 2026-08-10).
#
# The treatment arms are NOT trained by this script; the driver trains them, and
# only after `validate_e8_arms.py --require-init` passes. An initialization
# checkpoint is not complete until its own NLL artifact exists.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> VLLM_READY
#          -> INIT_READY -> TREATMENT_READY -> HOLDOUT_READY -> VALSTREAM_READY
#          -> TEACHER_READY -> ROPE_OK -> TESTS_OK -> MASK_OK -> ARMS_VALIDATED
#          -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/e8b.status
mark() { echo "MARKER:$1"; echo "$(date -u +%FT%TZ) MARKER:$1" >>"$STATUS"; }
say()  { echo "[$(date -u +%T)] $*"; }

export HF_TOKEN="$(cat $WS/hf/token)"
export HF_HOME=/root/.cache/huggingface

# ninja is NOT optional: vLLM's flashinfer sampling path JIT-compiles a kernel on
# first use and shells out to `ninja`. Without it the engine dies at the first
# LLM() call, after training has already been paid for.
say "apt: git, ninja, zstd"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git ninja-build zstd >/dev/null
command -v ninja >/dev/null || { echo "ninja missing after install"; exit 1; }
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

# The pack is needed twice over: the trainer reads `ladder_uniform`, the
# evaluation battery reads `ladder_uniform_probe`. Same bytes, two names, as in
# every prior training session. `audit.jsonl` is not optional — `ladder_blocks`
# reads it to select validation blocks and reconcile the block count.
say "fetching the ladder pack (train + probe) and the corpus"
python3 -c "
import shutil, sys
from pathlib import Path
sys.path.insert(0, '/workspace')
from fetch import fetch
fetch('stage3_recovery_corpus_v2/ladder_uniform',
      ['blocks.npz', 'ladder.json', 'audit.jsonl'],
      '/workspace/aad/artifacts/stage3/ladder_uniform_probe')
src = Path('/workspace/aad/artifacts/stage3/ladder_uniform_probe')
dst = Path('/workspace/aad/artifacts/stage3/ladder_uniform')
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    shutil.copy(f, dst / f.name)
print('pack staged as ladder_uniform and ladder_uniform_probe')
fetch('stage3_recovery_corpus_v2', ['sessions.jsonl'],
      '/workspace/aad/artifacts/stage3/corpus_v2')
"
python3 - <<'PYEOF'
import hashlib, sys
want = {
    '/workspace/aad/artifacts/stage3/ladder_uniform_probe/blocks.npz':
        '6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c',
    '/workspace/aad/artifacts/stage3/ladder_uniform/blocks.npz':
        '6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c',
    '/workspace/aad/artifacts/stage3/corpus_v2/sessions.jsonl':
        '2b4edc2e2cc16cd56dae3d340345e1a17e2c4a8baa9837650a7bf5e340fa6fcd',
}
for p, sha in want.items():
    got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print(f'  {p.rsplit("/", 2)[-2]}/{p.rsplit("/", 1)[-1]} {got[:16]}…')
    if got != sha:
        sys.exit(f'FROZEN ASSET MISMATCH {p}: {got}')
print('frozen assets verified')
PYEOF
test -f "$REPO/artifacts/stage3/ladder_uniform/blocks.npz"        # the trainer reads this
test -f "$REPO/artifacts/stage3/ladder_uniform_probe/blocks.npz"  # the battery reads this
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

# --- cold-host tripwire (carried unchanged from E5/E6) ---------------------
# Setup time varies ~30x on identical image, script and GPU purely with how much
# the host has cached: `uv sync` measured 44 s, ~50 s, then 62 MINUTES. A host
# that has not finished in TRIP_S is classified cold and abandoned; the launcher
# redraws. The grace clause spares a host that is genuinely linking.
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
/opt/train/bin/python -c "import torch, transformers, sympy; \
  assert torch.cuda.is_available(); \
  print('train torch', torch.__version__, torch.cuda.get_device_name(0), \
        '| transformers', transformers.__version__)"
mark TRAIN_ENV

say "vLLM venv on the container disk"
python3 -m venv /opt/vllm
/opt/vllm/bin/pip install -q --upgrade pip
/opt/vllm/bin/pip install -q vllm
/opt/vllm/bin/python -c "import vllm, torch, transformers; \
  print('vllm', vllm.__version__, '| torch', torch.__version__, \
        '| transformers', transformers.__version__, torch.cuda.is_available())"
mark VLLM_READY

say "staging the Stage 1 init — the fork point BOTH arms start from"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
fetch('stage1/qwen3_0p6b_init_v0/checkpoint',
      ['config.json', 'generation_config.json', 'model.safetensors',
       'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'],
      '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint')
# The control's own manifest, so the step-0 comparison can name the control's
# depth map from the record rather than inferring it from the geometry.
fetch('e8_inputs_20260810/stage1', ['qwen3_0p6b_init_v0_manifest.json'],
      '/workspace/aad/artifacts/stage1')
import shutil
shutil.move('/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0_manifest.json',
            '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/manifest.json')
"
python3 -c "
import hashlib, sys
p = '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors'
h = hashlib.sha256(open(p,'rb').read()).hexdigest()
want = '86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54'
print('stage1 init sha256', h)
sys.exit(0 if h == want else f'INIT HASH MISMATCH: {h}')
"
mark INIT_READY

# --- the treatment initialization: the entire causal variable -----------------
# Built on the dev box from pod A's frozen depth map, then staged here. Its hash
# is passed in by the launcher, which read it from the local build, so a wrong or
# stale checkpoint is fatal at setup rather than after 6.7 h of training.
say "staging the contribution-guided treatment initialization"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
fetch('e8_init_20260810/e8_contribution_init_v1/checkpoint',
      ['config.json', 'generation_config.json', 'model.safetensors',
       'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'],
      '/workspace/aad/artifacts/stage1/e8_contribution_init_v1/checkpoint')
fetch('e8_init_20260810/e8_contribution_init_v1', ['manifest.json'],
      '/workspace/aad/artifacts/stage1/e8_contribution_init_v1')
fetch('e8_init_20260810', ['depth_map.json', 'depth_search.json',
                           'e8_frozen_depth_map.json'],
      '/workspace/aad/artifacts/stage1/e8_depth_search')
"
python3 -c "
import hashlib, json, os, sys
p = '/workspace/aad/artifacts/stage1/e8_contribution_init_v1/checkpoint/model.safetensors'
h = hashlib.sha256(open(p,'rb').read()).hexdigest()
want = os.environ['TREATMENT_INIT_SHA256']
print('treatment init sha256', h)
if h != want:
    sys.exit(f'TREATMENT INIT HASH MISMATCH: {h} != {want}')
base = '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/config.json'
a = json.load(open(base)); b = json.load(open(
    '/workspace/aad/artifacts/stage1/e8_contribution_init_v1/checkpoint/config.json'))
if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
    sys.exit('TREATMENT CONFIG DIFFERS FROM THE CONTROL — only the depth map may change')
m = json.load(open('/workspace/aad/artifacts/stage1/e8_contribution_init_v1/manifest.json'))
d = m.get('init_diagnostics') or {}
# .get, not [], on every one: a manifest written before these keys existed would
# otherwise die with a bare KeyError on a billing pod instead of saying what is
# wrong. The pinned control's manifest is exactly such a manifest.
print('depth map source', d.get('depth_map_source'))
print('kept   ', d.get('kept_teacher_layers'))
print('removed', d.get('removed_teacher_layers'))
if d.get('depth_map_source') != 'explicit_kept_layers':
    sys.exit(f\"the treatment init records depth_map_source \"
             f\"{d.get('depth_map_source')!r}, not 'explicit_kept_layers' — it was \"
             f\"not built from a frozen depth map\")
if len(d.get('kept_teacher_layers') or []) != 28:
    sys.exit('treatment init does not keep 28 teacher layers')
if d.get('removed_teacher_layers') == [5, 7, 9, 11, 13, 15, 17, 19]:
    sys.exit('the treatment map IS the positional map; there is no treatment')
print('treatment initialization verified')
"
mark TREATMENT_READY

# The historical NLL series runs back to the Stage 1 gate through this file.
say "staging holdout_v1 and the E7 FineWeb validation stream"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
fetch('e8_inputs_20260810/warmup', ['holdout_v1.jsonl', 'holdout_v1.manifest.json'],
      '/workspace/aad/data/warmup')
fetch('e7_streams_20260809/e7_fineweb_val', ['blocks.npz', 'docs.jsonl', 'manifest.json'],
      '/workspace/aad/artifacts/stage3/e7_fineweb_val')
"
python3 -c "
import hashlib, sys
h = hashlib.sha256(open('/workspace/aad/data/warmup/holdout_v1.jsonl','rb').read()).hexdigest()
want = '2d49f637a711ae82510fd55a3af98e332314f972780841869508aebe7b3cd8e8'
print('holdout_v1 sha256', h)
sys.exit(0 if h == want else f'HOLDOUT MISMATCH: {h}')
"
mark HOLDOUT_READY
python3 -c "
import json, sys
m = json.load(open('/workspace/aad/artifacts/stage3/e7_fineweb_val/manifest.json'))
print('fineweb val', m['n_blocks'], 'x', m['block_len'])
if (m['n_blocks'], m['block_len']) != (512, 1024):
    sys.exit('validation stream is not the 512x1024 E7 stream')
"
mark VALSTREAM_READY

# KD needs the teacher, pinned to the revision every prior arm trained against.
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

# A silently wrong positional basis would invalidate every number this pod
# produces, and it does not raise on its own.
say "checking the RoPE base resolves correctly in both venvs"
for PY in /opt/train/bin/python /opt/vllm/bin/python; do
  $PY -c "
import sys, transformers
sys.path.insert(0, '/workspace/aad/src')
from transformers import AutoConfig
from aadistill.models.student import assert_rope_from_config, stored_rope_base
for p in ('/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint',
          '/workspace/aad/artifacts/stage1/e8_contribution_init_v1/checkpoint'):
    cfg = AutoConfig.from_pretrained(p)
    base = assert_rope_from_config(cfg, p)
    stored = stored_rope_base(cfg)
    print(f'  transformers {transformers.__version__}: {p.rsplit(chr(47),2)[-2]} '
          f'stored {stored:,.0f} runtime {base:,.0f} OK')
    if abs(stored - 5_000_000) > 1:
        raise SystemExit(f'{p} records RoPE base {stored}, expected 5,000,000')
"
done
mark ROPE_OK

say "CPU test suite"
cd "$REPO" && /opt/train/bin/python -m pytest tests/ -q \
    --ignore=tests/data/test_recovery_corpus_pipeline.py 2>&1 | tail -4
mark TESTS_OK

# The inclusion mask is rebuilt here, in this environment, from these staged
# files. If it does not reproduce the binding value the battery is not the frozen
# one and no training is worth paying for.
say "rebuilding the inclusion mask in the pod environment"
cd "$REPO" && PYTHONPATH=src /opt/train/bin/python - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, 'scripts/evaluation')
from diagnose_training_recall import rung_session_ids, stratified_sample
PACK = Path('artifacts/stage3/ladder_uniform_probe')
SESS = Path('artifacts/stage3/corpus_v2/sessions.jsonl')
want = set(rung_session_ids(PACK, 860000))
rung = [json.loads(l) for l in SESS.open() if l.strip() and json.loads(l)['id'] in want]
incl = [s for s in rung if s.get('correct') is True]
picked = stratified_sample(incl, 150, 20260804)
mask = hashlib.sha256(json.dumps(sorted(s['id'] for s in picked)).encode()).hexdigest()
print(f'rung {len(rung)} verified {len(incl)} sampled {len(picked)}')
print(f'inclusion mask {mask}')
if mask != 'd6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba':
    sys.exit(f'MASK MISMATCH {mask}')
print('mask matches the binding value')
PYEOF
mark MASK_OK

# The config-level half of the single-variable claim, re-proved here on the files
# that will actually train. The `--require-init` half runs in the driver, after
# both initializations have their own NLL records — which is the point of the gate.
say "validating the E8 arm configs against the frozen design"
cd "$REPO" && PYTHONPATH=src /opt/train/bin/python scripts/training/validate_e8_arms.py \
    --out artifacts/audit/e8_preflight_configs_pod.json
mark ARMS_VALIDATED

mark SETUP_DONE
say "setup complete — the driver measures both inits, gates, then trains two arms"
