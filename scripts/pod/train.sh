#!/bin/bash
# Launch s2_blocks_v1 training detached; markers to /workspace/run_markers.log
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/aad || exit 1
mkdir -p artifacts/stage3
nohup bash -c '
  uv run python scripts/train_stage3.py --config configs/stage3_s2_blocks_v1.json \
    > /workspace/console_s2v1.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "MARKER:TRAIN_DONE"; else echo "MARKER:TRAIN_FAILED rc=$rc"; fi >> /workspace/run_markers.log
' > /dev/null 2>&1 &
echo "MARKER:TRAIN_LAUNCHED pid=$!" >> /workspace/run_markers.log
echo launched
