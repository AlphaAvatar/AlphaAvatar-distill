#!/usr/bin/env bash
# Setup for the D0 diagnostic pod (2026-08-04): D0.3 three-mode evaluation and
# D0.4 KD decomposition on P0-real-sa / P0-real-sb.
#
# NO TRAINING. Nothing here or downstream calls optimizer.step(); the only
# backward passes are the D0.4 gradient probe, whose gradients are read and
# discarded.
#
# Two stacks as usual: the training venv (uv sync -> transformers 5.13.1) for
# forwards, teacher-forced scoring and the KD decomposition, and a vLLM venv on
# the container disk for the free and oracle rollouts.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> TRAIN_ENV -> VLLM_READY
#          -> CKPT_READY -> ROPE_OK -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/d0diag.status
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
dest = root / 'ladder_uniform_probe'; dest.mkdir(parents=True, exist_ok=True)
for f in src.iterdir(): shutil.copy(f, dest / f.name)
print('pack:', sorted(x.name for x in dest.iterdir()))
p = hf_hub_download(repo, 'stage3_recovery_corpus_v2/sessions.jsonl',
                    repo_type='model', token=tok)
(root / 'corpus_v2').mkdir(parents=True, exist_ok=True)
shutil.copy(p, root / 'corpus_v2/sessions.jsonl')
print('corpus staged')
"
test -f "$REPO/artifacts/stage3/ladder_uniform_probe/blocks.npz"
test -f "$REPO/artifacts/stage3/corpus_v2/sessions.jsonl"
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

say "staging P0-real-sa / P0-real-sb and the teacher"
python3 -c "
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
tok = os.environ['HF_TOKEN']; repo = 'AlphaAvatar/aadistill-artifacts'
for arm in ('e1_r0860k_sa_pca', 'e1_r0860k_sb_pca'):
    d = snapshot_download(repo, repo_type='model', token=tok,
          allow_patterns=[f'e1_scaling_20260801/{arm}/step_001023/model/*',
                          f'e1_scaling_20260801/{arm}/run_manifest.json'])
    src = Path(d) / f'e1_scaling_20260801/{arm}'
    dest = Path(f'/workspace/ckpt/{arm}'); dest.mkdir(parents=True, exist_ok=True)
    for f in src.rglob('*'):
        if f.is_file():
            o = dest / f.relative_to(src); o.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, o)
    print('staged', arm)
"
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Thinking-2507', revision=os.environ['TEACHER_REVISION'],
                  token=os.environ['HF_TOKEN'],
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
print('teacher downloaded')
"
mark CKPT_READY

# A silently wrong positional basis would invalidate every number this pod
# produces, and it does not raise on its own.
say "checking the RoPE base resolves correctly in both venvs"
for PY in /opt/train/bin/python /opt/vllm/bin/python; do
  $PY -c "
import sys, transformers
sys.path.insert(0, '/workspace/aad/src')
from transformers import AutoConfig, AutoModelForCausalLM
from aadistill.models.student import assert_rope_matches_config
p = '/workspace/ckpt/e1_r0860k_sa_pca/step_001023/model'
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
mark SETUP_DONE
say "setup complete"
