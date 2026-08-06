#!/usr/bin/env bash
# Setup for the Experiment 4 pod: P2-CE-heavy scaled 0.86M -> 1.60M.
#
# Two new training arms (e4_p2_r1600k_sa, _sb) from the Stage 1 PCA init, plus a
# re-evaluation of the existing P1-1.60M checkpoints through the CURRENT harness
# -- their recorded numbers came from the older behaviour-wave harness with the
# degeneration stop active and are not comparable. P2-0.86M is not retrained and
# not staged: its results are the reference and its NLL runs on the dev-box CPU.
#
# Two stacks, as in every prior session: the training venv (uv sync) for
# training, teacher-forced scoring and NLL, and a vLLM venv on the container
# disk for free/oracle rollouts.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> VLLM_READY
#          -> CKPT_READY -> P1REF_READY -> ROPE_OK -> TESTS_OK -> ARMS_VALIDATED
#          -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/e4.status
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
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
tok = os.environ['HF_TOKEN']; repo = 'AlphaAvatar/aadistill-artifacts'
root = Path('/workspace/aad/artifacts/stage3')
d = snapshot_download(repo, repo_type='model', token=tok,
                      allow_patterns=['stage3_recovery_corpus_v2/ladder_uniform/*'])
src = Path(d) / 'stage3_recovery_corpus_v2/ladder_uniform'
for nm in ('ladder_uniform', 'ladder_uniform_probe'):
    dest = root / nm; dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir(): shutil.copy(f, dest / f.name)
print('pack staged as ladder_uniform and ladder_uniform_probe')
p = hf_hub_download(repo, 'stage3_recovery_corpus_v2/sessions.jsonl',
                    repo_type='model', token=tok)
(root / 'corpus_v2').mkdir(parents=True, exist_ok=True)
shutil.copy(p, root / 'corpus_v2/sessions.jsonl')
print('corpus staged')
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
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok = os.environ['HF_TOKEN']; repo = 'AlphaAvatar/aadistill-artifacts'
d = snapshot_download(repo, repo_type='model', token=tok,
                      allow_patterns=['stage1/qwen3_0p6b_init_v0/checkpoint/*'])
src = Path(d) / 'stage1/qwen3_0p6b_init_v0/checkpoint'
dest = Path('/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint')
dest.mkdir(parents=True, exist_ok=True)
for f in src.iterdir(): shutil.copy(f, dest / f.name)
print('stage1 init staged')
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

# P1-1.60M is staged and hash-verified against the recorded relay digests. It is
# NOT retrained -- only re-evaluated, because its existing numbers came from the
# older 76-prompt behaviour wave with the degeneration stop active, which is a
# different measurement from the 150-example unrestricted harness.
say "staging the P1-1.60M reference checkpoints from the relay"
python3 -c "
import hashlib, os, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
tok = os.environ['HF_TOKEN']; repo = 'AlphaAvatar/aadistill-artifacts'
want = {'e1_r1600k_sa_pca': '6f77676ab8fde397ef7af75fda3e62171b5c84f315c439a1abb49917e46f6697',
        'e1_r1600k_sb_pca': 'e432d57e598d57e1633392e92955c8185faab57909f75f44bc1c349db6ccf39e'}
init = Path('/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint')
for arm, sha in want.items():
    dest = Path(f'/workspace/ckpt/{arm}/model'); dest.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo, f'e1_scaling_20260801/{arm}/step_001761/model/model.safetensors',
                        repo_type='model', token=tok)
    shutil.copy(p, dest / 'model.safetensors')
    got = hashlib.sha256((dest / 'model.safetensors').read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f'{arm} HASH MISMATCH: {got}')
    for f in ('config.json', 'generation_config.json', 'tokenizer.json',
              'tokenizer_config.json', 'chat_template.jinja'):
        shutil.copy(init / f, dest / f)
    print(f'{arm} staged and hash-verified')
"
mark P1REF_READY


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

# The same pre-launch gate that ran on the dev box, re-run here on the real
# weights in this environment. It asserts the freeze policy of all six arms,
# that A2's adapter is a no-op at initialization, and that merging leaves a
# plain checkpoint behind.
say "Experiment 4 preflight on the real student"
cd "$REPO" && PYTHONPATH=src /opt/train/bin/python scripts/training/preflight_e4.py \
    --out artifacts/audit/e4_preflight_pod.json
mark ARMS_VALIDATED
mark TESTS_OK

mark SETUP_DONE
say "setup complete"
