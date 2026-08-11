#!/usr/bin/env bash
# Setup for ANY E8b session, selected by $E8B_SESSION. One script, not four copies:
# the E8a defects that cost the most were latent in a launcher that had been copied
# and diverged, so E8b's four sessions share one setup, one driver and one launcher.
#
#   S1  L40S   step-0 only: build DP/DC, stage FP/FC, measure all four, probe DP/DC
#   S2  A100   DP-sa + DC-sa
#   S3  A100   DP-sb + DC-sb
#   S4  L40S   FC-sa + FC-sb
#
# Every session stages exactly what its stages need. DP and DC are never
# transferred: they are deterministic functions of the pinned teacher, so each pod
# rebuilds them and re-asserts their hashes, and a corrupted or stale checkpoint
# cannot pass.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> VLLM_READY
#          -> [COMPRESSED_READY] -> [DEPTH_BUILT] -> HOLDOUT_READY
#          -> VALSTREAM_READY -> TEACHER_READY -> ROPE_OK -> TESTS_OK
#          -> MASK_OK -> ARMS_VALIDATED -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
SESSION=${E8B_SESSION:?E8B_SESSION must be s1|s2|s3|s4}
STATUS=$WS/e8b_$SESSION.status
mark() { echo "MARKER:$1"; echo "$(date -u +%FT%TZ) MARKER:$1" >>"$STATUS"; }
say()  { echo "[$(date -u +%T)] $*"; }

case "$SESSION" in
  s1)    NEED_DEPTH=1; NEED_COMPRESSED=1 ;;
  s2|s3) NEED_DEPTH=1; NEED_COMPRESSED=0 ;;
  s4)    NEED_DEPTH=0; NEED_COMPRESSED=1 ;;
  *) echo "unknown E8B_SESSION $SESSION"; exit 1 ;;
esac
say "session $SESSION: depth=$NEED_DEPTH compressed=$NEED_COMPRESSED"

export HF_TOKEN="$(cat $WS/hf/token)"
export HF_HOME=/root/.cache/huggingface

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
            except Exception as exc:
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
"
rm -rf "$REPO"; git clone -q "$WS/$BUNDLE_NAME" "$REPO"
cd "$REPO"; git checkout -q "$SESSION_COMMIT"; git rev-parse HEAD
mark REPO_READY

say "fetching the ladder pack and corpus"
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
for f in src.iterdir(): shutil.copy(f, dst / f.name)
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
    if got != sha: sys.exit(f'FROZEN ASSET MISMATCH {p}: {got}')
print('frozen assets verified')
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

# Cold-host tripwire. Progress is summed across /opt/train AND /root/.cache/uv,
# because uv writes to the cache first; measuring only the venv classified a host
# downloading at 5.5 MB/s as stalled and cost E8a two draws. Grace renews while
# progress continues, bounded by UV_MAX_S so a genuinely hung host still dies.
TRIP_S=${UV_TRIP_S:-360}
GRACE_S=${UV_GRACE_S:-180}
UV_MAX_S=${UV_MAX_S:-1500}
uv sync --group dev &
UV_PID=$!
t0=$(date -u +%s); graced=0
while kill -0 "$UV_PID" 2>/dev/null; do
  sleep 15
  el=$(( $(date -u +%s) - t0 ))
  [ "$el" -lt "$TRIP_S" ] && continue
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
  exit 90
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

# --- the fully compressed pair: FP's pinned init and FC's contribution init ------
if [ "$NEED_COMPRESSED" = "1" ]; then
  say "staging the fully compressed initializations (FP control, FC treatment)"
  python3 -c "
import shutil, sys
from pathlib import Path
sys.path.insert(0, '/workspace')
from fetch import fetch
CK = ['config.json','generation_config.json','model.safetensors',
      'tokenizer.json','tokenizer_config.json','chat_template.jinja']
fetch('stage1/qwen3_0p6b_init_v0/checkpoint', CK,
      '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint')
fetch('e8_inputs_20260810/stage1', ['qwen3_0p6b_init_v0_manifest.json'],
      '/workspace/aad/artifacts/stage1')
shutil.move('/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0_manifest.json',
            '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/manifest.json')
fetch('e8_init_20260810/e8_contribution_init_v1/checkpoint', CK,
      '/workspace/aad/artifacts/stage1/e8_contribution_init_v1/checkpoint')
fetch('e8_init_20260810/e8_contribution_init_v1', ['manifest.json'],
      '/workspace/aad/artifacts/stage1/e8_contribution_init_v1')
fetch('e8_init_20260810', ['depth_map.json','e8_frozen_depth_map.json'],
      '/workspace/aad/artifacts/stage1/e8_depth_search')
fetch('e8_inputs_20260810/calibration_v1', ['manifest.json','leakage.json'],
      '/workspace/aad/artifacts/stage1/e8_calibration_v1')
"
  python3 - <<'PYEOF'
import hashlib, json, sys
want = {
 '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors':
   '86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54',
 '/workspace/aad/artifacts/stage1/e8_contribution_init_v1/checkpoint/model.safetensors':
   '7a0694a5d5c59f8e0b0ebc9ac8648b1ec026bf93cab026d33c61ca8fc85d1edb',
}
for p, sha in want.items():
    got = hashlib.sha256(open(p,'rb').read()).hexdigest()
    print(f'  {p.rsplit("/",3)[1]} {got[:16]}…')
    if got != sha: sys.exit(f'COMPRESSED INIT MISMATCH {p}: {got}')
a = json.load(open('/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/config.json'))
b = json.load(open('/workspace/aad/artifacts/stage1/e8_contribution_init_v1/checkpoint/config.json'))
if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
    sys.exit('FP and FC configs differ — only the depth map may change')
print('FP and FC verified; identical student config')
PYEOF
  mark COMPRESSED_READY
fi

# --- the depth-only pair: rebuilt here from the teacher, hashes asserted ---------
if [ "$NEED_DEPTH" = "1" ]; then
  say "building DP and DC from the pinned teacher (no transfer)"
  cd "$REPO"
  for M in positional contribution; do
    case "$M" in positional) O=e8b_dp_init ;; contribution) O=e8b_dc_init ;; esac
    PYTHONPATH=src /opt/train/bin/python scripts/training/build_depth_only_init.py \
        --map "$M" --out "artifacts/stage1/$O" 2>&1 | tail -3
  done
  python3 - <<'PYEOF'
import hashlib, json, sys
want = {
 'e8b_dp_init': 'd4db65eb8f7ae6d8a847c2db9a9e5e307e449f50f3bd129e07a1b20f6ec5f3cd',
 'e8b_dc_init': 'eb9e95481988b296a77c30d7b4754069f1874330fca9ad198f4457029e11e182',
}
cfg = None
for name, sha in want.items():
    base = f'/workspace/aad/artifacts/stage1/{name}'
    got = hashlib.sha256(open(f'{base}/checkpoint/model.safetensors','rb').read()).hexdigest()
    m = json.load(open(f'{base}/manifest.json'))
    v = m['verification']
    print(f'  {name} {got[:16]}… params {m["student"]["num_parameters"]:,} '
          f'bitwise-vs-ablated-teacher {v["bitwise_identical_to_ablated_teacher"]}')
    if got != sha: sys.exit(f'DEPTH-ONLY INIT MISMATCH {name}: {got}')
    if not v['bitwise_identical_to_ablated_teacher']:
        sys.exit(f'{name} is not the ablated teacher (max diff {v["max_abs_logit_diff"]})')
    if m['student']['num_parameters'] != 3_215_021_568:
        sys.exit(f'{name} parameter count {m["student"]["num_parameters"]}')
    if cfg is None: cfg = m['config_sha256']
    elif cfg != m['config_sha256']: sys.exit('DP and DC configs differ')
print(f'DP and DC verified; shared config {cfg[:16]}…')
PYEOF
  mark DEPTH_BUILT
fi

say "staging holdout_v1 and the FineWeb validation stream"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
fetch('e8_inputs_20260810/warmup', ['holdout_v1.jsonl','holdout_v1.manifest.json'],
      '/workspace/aad/data/warmup')
fetch('e7_streams_20260809/e7_fineweb_val', ['blocks.npz','docs.jsonl','manifest.json'],
      '/workspace/aad/artifacts/stage3/e7_fineweb_val')
"
python3 -c "
import hashlib, json, sys
h = hashlib.sha256(open('/workspace/aad/data/warmup/holdout_v1.jsonl','rb').read()).hexdigest()
if h != '2d49f637a711ae82510fd55a3af98e332314f972780841869508aebe7b3cd8e8':
    sys.exit(f'HOLDOUT MISMATCH: {h}')
m = json.load(open('/workspace/aad/artifacts/stage3/e7_fineweb_val/manifest.json'))
if (m['n_blocks'], m['block_len']) != (512, 1024):
    sys.exit('validation stream is not the 512x1024 stream')
print('holdout_v1 and the 512x1024 validation stream verified')
"
mark HOLDOUT_READY
mark VALSTREAM_READY

say "downloading the teacher at the pinned revision"
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Thinking-2507', revision=os.environ['TEACHER_REVISION'],
                  token=os.environ['HF_TOKEN'],
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
"
mark TEACHER_READY

# A wrong positional basis would invalidate every number this pod produces, and it
# does not raise on its own. Checked from the CONFIG, which needs no materialized
# model — a meta-device model has no buffer values and killed an E8a session.
say "checking the RoPE base resolves in every venv, for every staged checkpoint"
for PY in /opt/train/bin/python /opt/vllm/bin/python; do
  $PY -c "
import glob, sys, transformers
sys.path.insert(0, '/workspace/aad/src')
from transformers import AutoConfig
from aadistill.models.student import assert_rope_from_config, stored_rope_base
paths = sorted(glob.glob('/workspace/aad/artifacts/stage1/*/checkpoint/config.json'))
if not paths: sys.exit('no staged checkpoint to check')
for p in paths:
    # rsplit(1), not rsplit(2): AutoConfig needs the directory holding config.json,
    # i.e. .../<init>/checkpoint. Stripping two components hands it .../<init>,
    # which has no config.json, and transformers reports the confusing
    # 'Unrecognized model ... should have a model_type key'. Cost one draw.
    d = p.rsplit('/', 1)[0]
    name = d.rsplit('/', 2)[-2]
    cfg = AutoConfig.from_pretrained(d)
    base = assert_rope_from_config(cfg, d)
    stored = stored_rope_base(cfg)
    print(f'  transformers {transformers.__version__}: {name} '
          f'stored {stored:,.0f} runtime {base:,.0f} OK')
    if abs(stored - 5_000_000) > 1: sys.exit(f'{d} records RoPE base {stored}')
"
done
mark ROPE_OK

# Threads capped and the gate time-boxed: a 128-vCPU host gave torch 208 threads and
# a 70-second suite ran past 66 minutes, silently consuming an E8a session's budget.
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

say "rebuilding the inclusion mask in this environment"
cd "$REPO" && PYTHONPATH=src /opt/train/bin/python - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, 'scripts/evaluation')
from diagnose_training_recall import rung_session_ids, stratified_sample
want = set(rung_session_ids(Path('artifacts/stage3/ladder_uniform_probe'), 860000))
sess = Path('artifacts/stage3/corpus_v2/sessions.jsonl')
rung = [json.loads(l) for l in sess.open() if l.strip() and json.loads(l)['id'] in want]
incl = [s for s in rung if s.get('correct') is True]
picked = stratified_sample(incl, 150, 20260804)
mask = hashlib.sha256(json.dumps(sorted(s['id'] for s in picked)).encode()).hexdigest()
print(f'rung {len(rung)} verified {len(incl)} sampled {len(picked)}')
print(f'mask {mask}')
if mask != 'd6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba':
    sys.exit(f'MASK MISMATCH {mask}')
PYEOF
mark MASK_OK

say "validating the E8b arms for this session"
cd "$REPO" && PYTHONPATH=src /opt/train/bin/python scripts/training/validate_e8b_arms.py \
    --session "$SESSION" --out "artifacts/audit/e8b_${SESSION}_preflight_setup.json"
mark ARMS_VALIDATED

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mark SETUP_DONE
say "setup complete for session $SESSION"
