#!/bin/bash
# Launch one arm's training detached.
#   bash /workspace/train.sh <RUN_NAME> <CONFIG> [--resume]
# Markers go to /workspace/run_markers.log, arm-scoped so a multi-arm session
# can tell which arm finished: TRAIN_DONE:<RUN_NAME> / TRAIN_FAILED:<RUN_NAME>.
# Console goes to /workspace/console_<RUN_NAME>.log.
RUN_NAME="$1"
CONFIG="$2"
RESUME="${3:-}"
[ -n "$RUN_NAME" ] && [ -n "$CONFIG" ] || { echo "usage: train.sh <RUN_NAME> <CONFIG> [--resume]"; exit 2; }

export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/aad || exit 1
mkdir -p artifacts/stage3

nohup bash -c "
  uv run python scripts/train_stage3.py --config '$CONFIG' $RESUME \
    >> /workspace/console_${RUN_NAME}.log 2>&1
  rc=\$?
  if [ \$rc -eq 0 ]; then echo 'MARKER:TRAIN_DONE:${RUN_NAME}'; else echo \"MARKER:TRAIN_FAILED:${RUN_NAME} rc=\$rc\"; fi >> /workspace/run_markers.log
" > /dev/null 2>&1 &
echo "MARKER:TRAIN_LAUNCHED:${RUN_NAME} pid=$!" >> /workspace/run_markers.log
echo launched
