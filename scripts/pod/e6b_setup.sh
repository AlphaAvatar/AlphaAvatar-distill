#!/usr/bin/env bash
# Setup for the Experiment 6b pod: train P2 CE-heavy at the 2.96M rung, then
# evaluate both arms on the frozen 150-prompt battery.
#
# E6b TRAINS, so unlike E6 this stages the training ladder pack and the teacher.
# It stages **no trained checkpoints at all**: both arms start from the Stage 1
# PCA init, and the six control arms are re-scored on the dev box from
# generations that already exist. Nothing large crosses the dev-box uplink,
# which is what made E6 transfer-bound.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> VLLM_READY
#          -> INIT_READY -> TEACHER_READY -> ROPE_OK -> TESTS_OK -> MASK_OK
#          -> ARMS_VALIDATED -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/e6b.status
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
uv sync --group dev &
UV_PID=$!
t0=$(date -u +%s); graced=0
while kill -0 "$UV_PID" 2>/dev/null; do
  sleep 15
  el=$(( $(date -u +%s) - t0 ))
  [ "$el" -lt "$TRIP_S" ] && continue
  if [ "$graced" -eq 0 ] && [ -x /opt/train/bin/python ]; then
    a=$(du -sb /opt/train 2>/dev/null | cut -f1 || echo 0); sleep 20
    b=$(du -sb /opt/train 2>/dev/null | cut -f1 || echo 0)
    if [ "$((b - a))" -gt 20000000 ]; then      # >20 MB in 20 s: still linking
      say "uv sync past ${TRIP_S}s but linking ($(( (b-a)/1048576 )) MB/20s) — one ${GRACE_S}s grace"
      graced=1; TRIP_S=$(( TRIP_S + GRACE_S )); continue
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
from transformers import AutoConfig, AutoModelForCausalLM
from aadistill.models.student import assert_rope_matches_config
p = '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint'
cfg = AutoConfig.from_pretrained(p)
m = AutoModelForCausalLM.from_config(cfg)
base = assert_rope_matches_config(m, cfg, p)
print(f'  transformers {transformers.__version__}: rope base {base:,.0f} OK')
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

# The single-variable claim, re-proved here on the configs that will actually
# train, in this environment. Cheap, and the alternative is discovering a
# confounded comparison after paying for it.
say "validating the E6b arms against their registration"
cd "$REPO" && PYTHONPATH=src /opt/train/bin/python scripts/training/validate_e6b_arms.py \
    --registration logs/e6b_registration.json \
    --out artifacts/audit/e6b_preflight_pod.json
mark ARMS_VALIDATED

mark SETUP_DONE
say "setup complete — two arms will train from the Stage 1 init"
