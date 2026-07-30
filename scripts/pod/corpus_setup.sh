#!/usr/bin/env bash
# Pod-side setup for the teacher-corpus session, inside vLLM's official image.
#
# Deliberately much leaner than setup.sh: that one builds the training
# environment (cu128 torch, uv sync, full test suite) because it has to train.
# This pod only generates, and the engine image already ships torch,
# transformers, huggingface_hub and vLLM. Building a second environment on top
# is what broke the 2026-07-29 session, so nothing here installs a dependency.
#
# Neither `git` nor `zstd` is assumed present in a third-party image: the repo
# and the prompt slices arrive as one gzip tarball, which stock `tar` handles.
#
# Markers: ENV_READY -> XFER_OK -> SETUP_DONE (or SETUP_FAILED:<reason>)
set -uo pipefail

WORK=/workspace
XFER=${XFER:-stage3_corpus_transfer_20260730.tar.gz}
HF_REPO=${HF_REPO:-AlphaAvatar/aadistill-artifacts}
LOG=$WORK/setup.log

exec > >(tee -a "$LOG") 2>&1
mark() { echo "$(date -u +%FT%TZ) MARKER:$1"; }
fail() { mark "SETUP_FAILED:$1"; exit 1; }

echo "=== corpus_setup starting $(date -u +%FT%TZ) ==="
mkdir -p "$WORK/hf" "$WORK/markers" || fail mkdir
cd "$WORK" || fail cd

PY=$(command -v python3 || command -v python) || fail no_python
echo "python: $PY $($PY -c 'import sys;print(sys.version.split()[0])')"
for mod in torch transformers vllm huggingface_hub; do
  v=$($PY -c "import $mod;print(getattr($mod,'__version__','?'))" 2>/dev/null) \
    || fail "missing_module_$mod"
  echo "  $mod $v"
done
$PY -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)' || fail no_cuda
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || fail no_nvidia_smi
mark ENV_READY

[ -f "$WORK/hf/token" ] || fail no_hf_token
HF_TOKEN=$(cat "$WORK/hf/token") $PY - "$HF_REPO" "$XFER" "$WORK" <<'PY' || fail hf_download
import os, sys
from huggingface_hub import hf_hub_download
repo_id, name, work = sys.argv[1:4]
p = hf_hub_download(repo_id=repo_id, filename=f"transfer/{name}",
                    repo_type="model", local_dir=f"{work}/xfer",
                    token=os.environ["HF_TOKEN"])
print("downloaded", p)
PY

[ -f "$WORK/hashes_transfer.txt" ] || fail no_hash_manifest
sha256sum -c "$WORK/hashes_transfer.txt" || fail transfer_hash_mismatch
mark XFER_OK

tar xzf "$WORK/xfer/transfer/$XFER" -C "$WORK" || fail untar
[ -d "$WORK/aad/src/aadistill" ] || fail no_src
for f in rag_evidence multihop_qa code_math; do
  [ -s "$WORK/aad/data/stage2_v1/train/$f.jsonl" ] || fail "missing_slice_$f"
done
echo "prompt slices:"
wc -l "$WORK/aad"/data/stage2_v1/train/{rag_evidence,multihop_qa,code_math}.jsonl

# The client imports the repo from src/ rather than installing it: no build step,
# and the code that runs is exactly the tracked tree at the recorded commit.
cd "$WORK/aad" || fail cd_repo
PYTHONPATH=$WORK/aad/src $PY -c "
from aadistill.rollout.engines import VLLMServerEngine
from aadistill.rollout.snapshots import write_snapshot
from aadistill.data.verify import verify, select
from aadistill.data.dataset import load_jsonl
print('client imports OK')
" || fail client_import

mark SETUP_DONE
echo "=== corpus_setup done $(date -u +%FT%TZ) ==="
