#!/bin/bash
# Pod setup for the s2_blocks_v1 session (2026-07-26). Emits marker lines.
set -x
exec > /workspace/setup.log 2>&1

export DEBIAN_FRONTEND=noninteractive
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"
# token is written to $HF_HOME/token by the caller before this script runs

fail() { echo "MARKER:SETUP_FAILED:$1"; exit 1; }

apt-get update -qq && apt-get install -y -qq zstd > /dev/null || fail apt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || fail uv_install
export PATH="$HOME/.local/bin:$PATH"

cd /workspace || fail cd
# --- transfer bundle + data from the private HF repo ---
export HF_HUB_ENABLE_HF_TRANSFER=0
uvx --from huggingface_hub hf download AlphaAvatar/aadistill-artifacts \
  transfer/repo_20260726.bundle transfer/stage2_data_20260726.tar.zst \
  --repo-type model --local-dir /workspace/xfer || fail hf_transfer_download

sha256sum -c /workspace/hashes_transfer.txt || fail transfer_hash
git clone /workspace/xfer/transfer/repo_20260726.bundle aad || fail clone
cd /workspace/aad || fail cd_repo
git checkout main || fail checkout

# --- data ---
tar --use-compress-program=unzstd -xf /workspace/xfer/transfer/stage2_data_20260726.tar.zst || fail untar
ls data/stage2_v1/train data/stage2/val data/warmup/holdout_v1.jsonl || fail data_layout

# --- cu128 torch + env ---
sed -i 's|url = "https://download.pytorch.org/whl/cpu"|url = "https://download.pytorch.org/whl/cu128"|' pyproject.toml || fail sed
sed -i 's|name = "pytorch-cpu"|name = "pytorch-cu128"|' pyproject.toml
sed -i 's|torch = { index = "pytorch-cpu" }|torch = { index = "pytorch-cu128" }|' pyproject.toml
uv lock || fail uv_lock
uv sync || fail uv_sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || fail cuda_check
echo "MARKER:ENV_READY"

# --- start checkpoint from HF (A/B revision-pinned) ---
uvx --from huggingface_hub hf download AlphaAvatar/aadistill-artifacts \
  --repo-type model --revision 526caa780132dfcc522fcd1f8093fa7351e0db0c \
  --include "stage3/s2_blocks_v0/step_000660/model/*" \
  --local-dir /workspace/hfckpt || fail ckpt_download
mkdir -p artifacts/stage3/s2_blocks_v0/checkpoints/step_000660
cp -r /workspace/hfckpt/stage3/s2_blocks_v0/step_000660/model \
  artifacts/stage3/s2_blocks_v0/checkpoints/step_000660/model || fail ckpt_place
sha256sum -c /workspace/hashes_ckpt.txt || fail ckpt_hash
echo "MARKER:CKPT_READY"

# --- tests ---
uv run pytest tests/ -q 2>&1 | tail -3
uv run pytest tests/ -q > /workspace/pytest.log 2>&1 || fail tests
echo "MARKER:TESTS_PASSED"
echo "MARKER:SETUP_DONE"
