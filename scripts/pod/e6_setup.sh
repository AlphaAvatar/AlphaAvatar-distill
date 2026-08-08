#!/usr/bin/env bash
# Setup for the Experiment 6 pod: normalize the high-rung E1 PCA checkpoints
# onto the frozen 150-prompt unrestricted protocol.
#
# **Nothing trains here.** E6 evaluates six existing checkpoints and writes no
# weights, so this setup stages no training pack, no teacher and no holdout set.
# The training venv is still built because the harness's `forced` mode runs a
# float32 HF forward, and because the CPU test suite runs in it.
#
# Six checkpoints, from two stores:
#   relay  — e1_r1600k_{sa,sb}_pca, e1_r2960k_sa_pca, e1_r5500k_sa_pca
#   devbox — e1_r2960k_sb_pca, e1_r5500k_sb_pca   (scp'd by the launcher; the
#            relay never received them, and its LFS quota cannot take them)
# Every one is hash-verified against the registration before it is evaluated.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> VLLM_READY
#          -> INIT_READY -> CKPT_READY -> ROPE_OK -> TESTS_OK -> MASK_OK
#          -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/e6.status
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
python3 -c "import huggingface_hub as h; print('huggingface_hub', h.__version__)"

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

# The evaluation battery is drawn from the packed ladder at the 0.86M rung, so
# the probe pack and the corpus are the only data E6 needs. The TRAINING pack
# (`ladder_uniform`) is deliberately not staged: nothing here trains, and an
# absent training pack is a stronger guarantee than a policy that says so.
say "fetching the probe pack and the corpus"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
# audit.jsonl is not optional: ladder_blocks() reads it to pick validation
# blocks and to reconcile the block count, so a pack without it loads on the dev
# box (where the file happens to exist) and fails on the pod.
fetch('stage3_recovery_corpus_v2/ladder_uniform',
      ['blocks.npz', 'ladder.json', 'audit.jsonl'],
      '/workspace/aad/artifacts/stage3/ladder_uniform_probe')
fetch('stage3_recovery_corpus_v2', ['sessions.jsonl'],
      '/workspace/aad/artifacts/stage3/corpus_v2')
"
python3 - <<'PYEOF'
import hashlib, sys
want = {
    '/workspace/aad/artifacts/stage3/ladder_uniform_probe/blocks.npz':
        '6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c',
    '/workspace/aad/artifacts/stage3/corpus_v2/sessions.jsonl':
        '2b4edc2e2cc16cd56dae3d340345e1a17e2c4a8baa9837650a7bf5e340fa6fcd',
}
for p, sha in want.items():
    got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print(f'  {p.rsplit("/", 1)[-1]} {got[:16]}…')
    if got != sha:
        sys.exit(f'FROZEN ASSET MISMATCH {p}: {got}')
print('frozen evaluation assets verified')
PYEOF
test ! -d "$REPO/artifacts/stage3/ladder_uniform"   # no training pack, by design
mark DATA_READY

say "training env via uv sync (needed for forced mode and the test suite)"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"
sed -i 's|url = "https://download.pytorch.org/whl/cpu"|url = "https://download.pytorch.org/whl/cu128"|' pyproject.toml
sed -i 's|name = "pytorch-cpu"|name = "pytorch-cu128"|' pyproject.toml
sed -i 's|torch = { index = "pytorch-cpu" }|torch = { index = "pytorch-cu128" }|' pyproject.toml
export UV_PROJECT_ENVIRONMENT=/opt/train
uv lock

# --- cold-host tripwire (carried unchanged from E5) ------------------------
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

# The init supplies the tokenizer/template files every checkpoint is dressed
# with, and the config the RoPE check reads. Its weights are never loaded here.
say "staging the Stage 1 init (tokenizer, template, config)"
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

# Only the FOUR relay arms are staged here. The other two exist only on the dev
# box and are scp'd concurrently with this setup so 4.8 GB of transfer overlaps
# `uv sync` instead of following it — which means they are not necessarily on
# disk yet. Staging them here would race the upload and fail on a partial file,
# so the launcher joins the upload and runs the same script again over all six.
# `--stores relay` is what makes that split explicit rather than accidental.
say "staging and hash-verifying the four relay checkpoints"
/opt/train/bin/python "$REPO/scripts/pod/e6_stage_checkpoints.py" \
    --registration "$REPO/logs/e6_registration.json" \
    --relay-dest /workspace/ckpt \
    --devbox-src /workspace/ckpt_local \
    --stores relay \
    --init /workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
    --out /workspace/aad/artifacts/audit/e6_checkpoint_manifest.json
mark CKPT_READY

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
# one and no generation is worth paying for.
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

mark SETUP_DONE
say "setup complete — nothing has trained and nothing will"
