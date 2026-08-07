#!/usr/bin/env bash
# Setup for the Experiment 5 pod: teacher-prefix (C) vs student-prefix (R).
#
# E5 continues from the P2-0.86M checkpoints, so unlike E4 they MUST be staged
# here: they are the rollout student for arm R and the start point for all four
# training arms. They live only on the dev box, so they were uploaded to the
# relay under e5_start/ and are hash-verified on arrival.
#
# Two stacks, as in every prior session: the training venv (uv sync) for
# training, teacher-forced scoring and NLL, and a vLLM venv on the container
# disk for free/oracle rollouts.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> VLLM_READY
#          -> CKPT_READY -> START_CKPT_READY -> ROPE_OK -> TESTS_OK -> ARMS_VALIDATED
#          -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/e5.status
mark() { echo "MARKER:$1"; echo "$(date -u +%FT%TZ) MARKER:$1" >>"$STATUS"; }
say()  { echo "[$(date -u +%T)] $*"; }

export HF_TOKEN="$(cat $WS/hf/token)"
export HF_HOME=/root/.cache/huggingface

say "apt: git, ninja, zstd"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git ninja-build zstd >/dev/null
command -v ninja >/dev/null || { echo "ninja missing"; exit 1; }
mark ENV_READY

say "installing huggingface_hub"
python3 -m pip install -q --no-input --break-system-packages \
    "huggingface_hub[hf_transfer]" 2>&1 | tail -3
python3 -c "import huggingface_hub as h; print('huggingface_hub', h.__version__)"

# Explicit per-file fetch with retry. `snapshot_download(allow_patterns=...)`
# enumerates the ENTIRE repo tree before filtering, and on this relay (700+
# files) that call 504s at the HF gateway -- it killed the 2026-08-06 pilot in
# setup. Naming the files avoids the tree walk completely, fails loudly if one
# is missing, and makes the staged set an explicit contract rather than a glob.
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

say "fetching the uniform ladder pack and the corpus"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
LAD = 'stage3_recovery_corpus_v2/ladder_uniform'
for nm in ('ladder_uniform', 'ladder_uniform_probe'):
    fetch(LAD, ['blocks.npz', 'ladder.json', 'audit.jsonl'],
          f'/workspace/aad/artifacts/stage3/{nm}')
fetch('stage3_recovery_corpus_v2', ['sessions.jsonl'],
      '/workspace/aad/artifacts/stage3/corpus_v2')
"
test -f "$REPO/artifacts/stage3/ladder_uniform/blocks.npz"
test -f "$REPO/artifacts/stage3/ladder_uniform_probe/blocks.npz"
test -f "$REPO/artifacts/stage3/corpus_v2/sessions.jsonl"
# holdout_v1.jsonl is gitignored so it does not ship in the bundle; the launcher
# transfers it and the hash is asserted here, before anything trains.
mkdir -p "$REPO/data/warmup"
cp "${HOLDOUT_SRC:-/workspace/aad_holdout/holdout_v1.jsonl}" "$REPO/data/warmup/holdout_v1.jsonl"
python3 -c "
import hashlib,sys
h=hashlib.sha256(open('$REPO/data/warmup/holdout_v1.jsonl','rb').read()).hexdigest()
want='2d49f637a711ae82510fd55a3af98e332314f972780841869508aebe7b3cd8e8'
print('holdout_v1 sha256', h)
sys.exit(0 if h==want else f'HOLDOUT MISMATCH {h}')
"
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
uv sync --group dev
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

say "staging the Stage 1 init and the teacher"
python3 -c "
import sys; sys.path.insert(0, '/workspace')
from fetch import fetch
fetch('stage1/qwen3_0p6b_init_v0/checkpoint',
      ['config.json', 'generation_config.json', 'model.safetensors',
       'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'],
      '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint')
"
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Thinking-2507', revision=os.environ['TEACHER_REVISION'],
                  token=os.environ['HF_TOKEN'],
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
print('teacher downloaded')
"
# The fork point every arm starts from, verified before anything trains.
python3 -c "
import hashlib, sys
p = '/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors'
h = hashlib.sha256(open(p,'rb').read()).hexdigest()
want = '86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54'
print('stage1 init sha256', h)
sys.exit(0 if h == want else f'INIT HASH MISMATCH: {h}')
"
mark CKPT_READY

# The P2-0.86M checkpoints: rollout student for arm R and start point for every
# training arm. Hash-verified against the values recorded when they were
# retained on 2026-08-05.
say "staging the P2-0.86M start checkpoints from the relay"
python3 -c "
import hashlib, os, sys
from pathlib import Path
sys.path.insert(0, '/workspace')
from fetch import fetch
want = {'sa': '4aface45a12cd02e', 'sb': '9828b1780a5eb4e2'}
for seed in os.environ.get('E5_SEEDS', 'sa').split(','):
    dest = f'/workspace/ckpt/p2_ceheavy_{seed}'
    fetch(f'e5_start/p2_ceheavy_{seed}',
          ['config.json', 'generation_config.json', 'model.safetensors',
           'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'], dest)
    got = hashlib.sha256(Path(dest, 'model.safetensors').read_bytes()).hexdigest()
    if not got.startswith(want[seed]):
        sys.exit(f'p2_ceheavy_{seed} HASH MISMATCH: {got[:16]}')
    print(f'p2_ceheavy_{seed} staged and hash-verified ({got[:16]})')
"
mark START_CKPT_READY


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

say "CPU test suite (includes the LoRA and single-variable guarantees)"
cd "$REPO" && /opt/train/bin/python -m pytest tests/ -q \
    --ignore=tests/data/test_recovery_corpus_pipeline.py 2>&1 | tail -4

# Arm C is REUSED from attempt 1, not rebuilt. It is a mask move over teacher
# trajectories -- no generation -- and the attempt-1 corpora were verified
# against the current schema and packing checks on the dev box before upload:
# 2,294 examples per seed, 0 missing fields, 0 unrenderable, 1,034 blocks,
# 905,488 candidate CE tokens, 362 system blocks. Regenerating would only risk
# a difference; hashes are asserted here instead.
say "staging the arm C corpora from attempt 1"
python3 -c "
import os, shutil, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download('AlphaAvatar/aadistill-artifacts', 'e5_start/e5_arm_c.tar.gz',
                    repo_type='model', token=os.environ['HF_TOKEN'])
shutil.copy(p, '/workspace/e5_arm_c.tar.gz')
"
python3 -c "
import hashlib, sys
h = hashlib.sha256(open('/workspace/e5_arm_c.tar.gz','rb').read()).hexdigest()
want = 'fdf44b34a89164a88c59256129e692fd89021bcf53209831ca6d8d9eb6e49bee'
print('arm C bundle sha256', h)
sys.exit(0 if h == want else f'ARM C BUNDLE MISMATCH: {h}')
"
tar xzf /workspace/e5_arm_c.tar.gz --no-same-owner -C "$REPO/artifacts/stage3"
python3 -c "
import hashlib, sys
from pathlib import Path
want = {
 'e5_arm_c_sa/examples.jsonl': '4bc23c3fb54c8feff621b75445977fc47d24a343854f5131f6a4a770cdd46179',
 'e5_arm_c_sb/examples.jsonl': '03c9ba9af78908a29a1ad9b3fb7abd5ede53737ce66fe566c8867cdacb104e96',
 'e5_arm_c_sa/system_ids.json': '18ace28a0ee785852af508c89bdffd7578b5527874589ed4572404f55df39535',
 'e5_arm_c_sb/system_ids.json': '18ace28a0ee785852af508c89bdffd7578b5527874589ed4572404f55df39535',
}
for rel, exp in want.items():
    got = hashlib.sha256(Path('$REPO/artifacts/stage3', rel).read_bytes()).hexdigest()
    if got != exp:
        sys.exit(f'ARM C MISMATCH {rel}: {got}')
    print(f'  {rel} verified')
"
# The reused corpora must satisfy the CURRENT packing contract in THIS
# environment, not merely match a hash from the environment that made them.
cd "$REPO" && PYTHONPATH=src /opt/train/bin/python -c "
import json, sys
from pathlib import Path
from aadistill.data.e5_pack import REQUIRED_FIELDS, example_to_rendered, pack_e5
for seed in ('sa', 'sb'):
    d = Path('artifacts/stage3', f'e5_arm_c_{seed}')
    rows = [json.loads(l) for l in (d/'examples.jsonl').open() if l.strip()]
    sysids = json.loads((d/'system_ids.json').read_text())
    bad = 0
    for e in rows:
        if [f for f in REQUIRED_FIELDS if f not in e]:
            bad += 1; continue
        try:
            example_to_rendered(e)
            assert e['ids'][:e['n_system_tokens']] == sysids[e['system_key']]
        except Exception:
            bad += 1
    if bad or not rows:
        sys.exit(f'arm C {seed} fails the current contract: {bad}/{len(rows)}')
    print(f'  arm C {seed}: {len(rows)} examples, {len(pack_e5(rows, sysids, block_len=8192))} blocks, contract OK')
"
mark ARMS_VALIDATED
mark TESTS_OK

mark SETUP_DONE
say "setup complete"
