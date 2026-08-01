#!/bin/bash
# Pod setup, parameterized by run_env.sh. Emits MARKER: lines to /workspace/setup.log.
#
# Markers, in order: ENV_READY -> CKPT_READY -> TESTS_PASSED -> SETUP_DONE.
# Any failure emits SETUP_FAILED:<step> and exits non-zero.
set -x
exec > /workspace/setup.log 2>&1

export DEBIAN_FRONTEND=noninteractive
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"
# token is written to $HF_HOME/token by the caller before this script runs

fail() { echo "MARKER:SETUP_FAILED:$1"; exit 1; }

source /workspace/run_env.sh || fail source_run_env

apt-get update -qq && apt-get install -y -qq zstd > /dev/null || fail apt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || fail uv_install
export PATH="$HOME/.local/bin:$PATH"

cd /workspace || fail cd
# --- transfer bundle + data from the private HF repo ---
export HF_HUB_ENABLE_HF_TRANSFER=0
# Everything the session needs is downloaded before anything is verified, so the
# single `sha256sum -c` below covers the whole manifest. Downloading one artifact
# after the check would make that check fail on the entry it has not fetched yet.
uvx --from huggingface_hub hf download "$HF_REPO" \
  "$TRANSFER_BUNDLE" "$TRANSFER_DATA" ${TRANSFER_EXTRA:+"$TRANSFER_EXTRA"} \
  --repo-type model --local-dir /workspace/xfer || fail hf_transfer_download

sha256sum -c /workspace/hashes_transfer.txt || fail transfer_hash
# Re-runnable: a setup that failed after cloning must not be blocked by its own
# leftovers on the retry. The clone is disposable — the repo lives in the bundle.
rm -rf /workspace/aad
git clone "/workspace/xfer/$TRANSFER_BUNDLE" aad || fail clone
cd /workspace/aad || fail cd_repo
git checkout main || fail checkout

# --- data ---
tar --use-compress-program=unzstd -xf "/workspace/xfer/$TRANSFER_DATA" || fail untar
ls data/stage2_v1/train data/stage2/val "$HOLDOUT" || fail data_layout

# Optional third artifact: data built on the dev box that is neither tracked in
# the repo nor part of the standing mixture — the Stage 3 2x2 arms, which are
# derived from a hashed generation corpus and are gitignored like every other
# built split. gzip rather than zstd so the same tarball can be read by an
# engine image that ships no zstd.
if [ -n "${TRANSFER_EXTRA:-}" ]; then
  # Already downloaded and hash-verified above, with the rest of the manifest.
  tar xzf "/workspace/xfer/$TRANSFER_EXTRA" || fail untar_extra
  for d in ${EXTRA_EXPECT:-}; do
    ls "$d" > /dev/null || fail "extra_missing_$d"
  done
  echo "extra data staged: ${EXTRA_EXPECT:-<unchecked>}"
fi
# The behavior prompt set is tracked in the repo, so it arrives with the bundle.
ls "$BEHAVIOR_PROMPTS" || fail behavior_prompts_missing

# --- cu128 torch + env ---
sed -i 's|url = "https://download.pytorch.org/whl/cpu"|url = "https://download.pytorch.org/whl/cu128"|' pyproject.toml || fail sed
sed -i 's|name = "pytorch-cpu"|name = "pytorch-cu128"|' pyproject.toml
sed -i 's|torch = { index = "pytorch-cpu" }|torch = { index = "pytorch-cu128" }|' pyproject.toml
uv lock || fail uv_lock
uv sync || fail uv_sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || fail cuda_check
echo "MARKER:ENV_READY"

# --- start + reference checkpoints from HF (each revision-pinned) ---
for entry in "${REF_CKPTS[@]}"; do
  glob=$(printf '%s' "$entry" | cut -d'|' -f1)
  dest=$(printf '%s' "$entry" | cut -d'|' -f2)
  rev=$(printf '%s' "$entry" | cut -d'|' -f3)
  uvx --from huggingface_hub hf download "$HF_REPO" \
    --repo-type model --revision "$rev" \
    --include "$glob/*" --local-dir /workspace/hfckpt || fail "ckpt_download_$glob"
  mkdir -p "$(dirname "$dest")"
  cp -r "/workspace/hfckpt/$glob" "$dest" || fail "ckpt_place_$glob"
done
# A session with no checkpoints to stage (e.g. the 2026-07-29 engine benchmark,
# whose teacher comes straight from the Hub) leaves this manifest empty, and
# `sha256sum -c` treats an empty file as an error rather than a no-op.
if [ -s "${CKPT_HASHES:-/workspace/hashes_ckpt.txt}" ]; then
  sha256sum -c "${CKPT_HASHES:-/workspace/hashes_ckpt.txt}" || fail ckpt_hash
else
  echo "no checkpoints staged this session; skipping ckpt hash check"
fi
# --- the packed token ladder Experiment 1 trains on ---
# Pulled straight from the relay and hash-checked: it is the experiment's data
# identity, and a silently different pack would make every rung unreproducible.
if [ -n "${LADDER_PREFIX:-}" ]; then
  uvx --from huggingface_hub hf download "$HF_REPO" --repo-type model \
    --include "$LADDER_PREFIX/*" --local-dir /workspace/hfladder || fail ladder_download
  mkdir -p "$LADDER_DEST"
  cp "/workspace/hfladder/$LADDER_PREFIX"/* "$LADDER_DEST/" || fail ladder_place
  sha256sum -c /workspace/hashes_ladder.txt || fail ladder_hash
  ls "$LADDER_DEST/blocks.npz" "$LADDER_DEST/audit.jsonl" "$LADDER_DEST/ladder.json" || fail ladder_layout
  echo "MARKER:LADDER_READY"
fi

echo "MARKER:CKPT_READY"

# --- tests ---
uv run pytest tests/ -q 2>&1 | tail -3
uv run pytest tests/ -q > /workspace/pytest.log 2>&1 || fail tests
echo "MARKER:TESTS_PASSED"
echo "MARKER:SETUP_DONE"
