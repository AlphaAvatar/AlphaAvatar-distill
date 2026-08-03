#!/usr/bin/env bash
# Experiment 2 Phase 1 pod setup. One pod trains AND runs the vLLM battery, so
# it needs both stacks:
#
#   * the training venv (torch cu128) — byte-identical path to the one that
#     produced D0, which is what makes the comparison valid;
#   * a separate vLLM venv on the CONTAINER disk (a torch install onto the
#     network mount took >9 min in a previous session).
#
# The pod must have been created with `--min-cuda-version 13.0`: vLLM 0.26's
# wheel links libcudart.so.13, and `--torch-backend=cu128` does not help because
# it changes torch, not the vLLM extension. Experiment 1 needed a second pod for
# exactly this reason.
#
# Markers: ENV_READY -> REPO_READY -> DATA_READY -> VLLM_READY -> CKPT_READY
#          -> TESTS_PASSED -> SETUP_DONE
set -euo pipefail

WS=/workspace
REPO=$WS/aad
STATUS=$WS/e2p1.status
mark() { echo "MARKER:$1"; echo "$(date -u +%FT%TZ) MARKER:$1" >>"$STATUS"; }
say()  { echo "[$(date -u +%T)] $*"; }

export HF_TOKEN="$(cat $WS/hf/token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/root/.cache/huggingface
REPO_ID=AlphaAvatar/aadistill-artifacts

say "apt: git, ninja (FlashInfer JIT-builds the top-k kernel during warmup)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git ninja-build zstd >/dev/null
command -v ninja >/dev/null || { echo "ninja missing"; exit 1; }
mark ENV_READY

# --- repo ------------------------------------------------------------------
# The base image ships no huggingface_hub, and swallowing this install with
# `|| true` cost a pod: the failure only surfaced as an ImportError one line
# later. Install it loudly and prove the import before using it.
say "installing huggingface_hub"
python3 -m pip install -q --no-input --break-system-packages \
    "huggingface_hub[hf_transfer]" 2>&1 | tail -3
python3 -c "import huggingface_hub as h; print('huggingface_hub', h.__version__)"

say "fetching the repo bundle"
python3 - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
name = os.environ["BUNDLE_NAME"]
p = hf_hub_download("AlphaAvatar/aadistill-artifacts", f"transfer/{name}",
                    repo_type="model", token=os.environ["HF_TOKEN"])
shutil.copy(p, f"/workspace/{name}")
print("bundle at", p)
PY
rm -rf "$REPO"
git clone -q "$WS/$BUNDLE_NAME" "$REPO"
cd "$REPO"
git checkout -q "$SESSION_COMMIT"
# P4: the corpus-v2 manifest recorded `code_state_error` because the bundle was
# unpacked outside a git checkout. This IS a checkout, and the commit is pinned.
git rev-parse HEAD
mark REPO_READY

# --- data + battery + the D1 pack -----------------------------------------
say "fetching data, capability battery and the D1 pack"
python3 - <<'PY'
import os, shutil, subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
tok = os.environ["HF_TOKEN"]; repo = "AlphaAvatar/aadistill-artifacts"
root = Path("/workspace/aad")

p = hf_hub_download(repo, "transfer/stage2_data_20260726.tar.zst",
                    repo_type="model", token=tok)
# --no-same-owner: this mount rejects chown and tar exits non-zero without it.
subprocess.run(["tar", "--no-same-owner", "-I", "zstd", "-xf", p, "-C", str(root)],
               check=True)

for prefix, dest in (
    (os.environ["BATTERY_PREFIX"], root / "artifacts/eval/battery_v2"),
    (os.environ["PACK_PREFIX"], root / "artifacts/stage3/rung_0860k_clean_median"),
):
    d = snapshot_download(repo, repo_type="model", token=tok,
                          allow_patterns=[f"{prefix}/*"])
    src = Path(d) / prefix
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy(f, dest / f.name)
    print("staged", dest, sorted(x.name for x in dest.iterdir()))
PY
test -f "$REPO/data/warmup/holdout_v1.jsonl"
test -f "$REPO/data/eval_behavior_v0/prompts.jsonl"
test -f "$REPO/artifacts/eval/battery_v2/manifest.json"
test -f "$REPO/artifacts/stage3/rung_0860k_clean_median/blocks.npz"
mark DATA_READY

# --- training venv ---------------------------------------------------------
say "training venv (torch cu128) on the container disk"
python -m venv /opt/train
/opt/train/bin/pip install -q --upgrade pip
/opt/train/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
/opt/train/bin/pip install -q transformers accelerate safetensors numpy huggingface_hub
/opt/train/bin/python -c "import torch;print('train torch',torch.__version__,torch.cuda.is_available())"

# --- vLLM venv -------------------------------------------------------------
say "vLLM venv on the container disk"
python -m venv /opt/vllm
/opt/vllm/bin/pip install -q --upgrade pip
/opt/vllm/bin/pip install -q vllm
/opt/vllm/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,torch.cuda.is_available())"
mark VLLM_READY

# --- checkpoints: the two D0 endpoints and the Stage 1 PCA init ------------
say "staging checkpoints"
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
tok = os.environ["HF_TOKEN"]; repo = "AlphaAvatar/aadistill-artifacts"
want = {
  "e1_scaling_20260801/e1_r0860k_sa_pca/step_001023/model":
      "/workspace/d0/e1_r0860k_sa_pca/step_001023/model",
  "e1_scaling_20260801/e1_r0860k_sb_pca/step_001023/model":
      "/workspace/d0/e1_r0860k_sb_pca/step_001023/model",
  "stage1/qwen3_0p6b_init_v0/checkpoint":
      "/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
}
for prefix, dest in want.items():
    d = snapshot_download(repo, repo_type="model", token=tok,
                          allow_patterns=[f"{prefix}/*"])
    src = Path(d) / prefix
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy(f, dest / f.name)
    print("staged", dest)
PY
# The fork point every D1 arm starts from, verified before anything trains.
python3 - <<'PY'
import hashlib, sys
p = "/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors"
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
want = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"
print("stage1 init sha256", h)
sys.exit(0 if h == want else f"INIT HASH MISMATCH: {h} != {want}")
PY
mark CKPT_READY

# --- tests -----------------------------------------------------------------
say "CPU test suite (fails loudly before any paid generation)"
/opt/train/bin/pip install -q pytest
cd "$REPO" && /opt/train/bin/python -m pytest tests/ -q --ignore=tests/data/test_recovery_corpus_pipeline.py 2>&1 | tail -5
mark TESTS_PASSED
mark SETUP_DONE
say "setup complete"
