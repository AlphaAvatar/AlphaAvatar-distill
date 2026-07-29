#!/bin/bash
# Teacher session: score the teacher on eval_behavior_v0, then run the top-n
# verified-generation pilot. Emits MARKER: lines to /workspace/session.log.
#
# Markers, in order:
#   ENV_READY -> TESTS_PASSED -> TEACHER_SCORED -> PILOT_DONE
# Any failure emits SESSION_FAILED:<step> and exits non-zero.
#
# Differs from setup.sh/train.sh: no student checkpoints, no training, no HF
# relay. The repo bundle and the pilot prompt pack are scp'd in (they are ~10 MB
# together), and the only large download is the teacher itself.
#
# The two jobs are ordered deliberately. The teacher scorecard is the cheap,
# high-value one — it is the ceiling the README figure is missing — so it lands
# first and is safe even if the pilot is cut short. The pilot streams
# candidates.jsonl and targets.jsonl as it goes, so a killed pilot still leaves
# usable partial data; only manifest.json needs the run to finish.
set -x
exec > /workspace/session.log 2>&1

export DEBIAN_FRONTEND=noninteractive
export UV_PROJECT_ENVIRONMENT=/root/venv
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME"

TEACHER=${TEACHER:-Qwen/Qwen3-4B-Thinking-2507@768f209d}
PILOT_LIMIT=${PILOT_LIMIT:-100}
PILOT_N=${PILOT_N:-4}
PILOT_BATCH=${PILOT_BATCH:-8}
BEHAVIOR_MAX_NEW_TOKENS=${BEHAVIOR_MAX_NEW_TOKENS:-4096}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-4096}

fail() { echo "MARKER:SESSION_FAILED:$1"; exit 1; }

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || fail uv_install
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/aad || fail cd_repo

# cu128 re-lock, as in every prior session (documented deviation from the
# dev-box CPU lockfile).
sed -i 's|url = "https://download.pytorch.org/whl/cpu"|url = "https://download.pytorch.org/whl/cu128"|' pyproject.toml || fail sed
sed -i 's|name = "pytorch-cpu"|name = "pytorch-cu128"|' pyproject.toml
sed -i 's|torch = { index = "pytorch-cpu" }|torch = { index = "pytorch-cu128" }|' pyproject.toml
uv lock || fail uv_lock
uv sync || fail uv_sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))" || fail cuda_check
echo "MARKER:ENV_READY"

# The pod runs a different Python and torch build than the dev box; the suite is
# the bridge that makes results from the two comparable.
uv run pytest tests/ -q > /workspace/pytest.log 2>&1 || fail tests
tail -3 /workspace/pytest.log
echo "MARKER:TESTS_PASSED"

# --- 1. teacher scorecard, native thinking mode, no prefill ---------------
# The cap must fit a full reasoning trace; truncation is reported, not hidden.
uv run python scripts/evaluation/eval_behavior.py \
  --model "$TEACHER" \
  --max-new-tokens "$BEHAVIOR_MAX_NEW_TOKENS" \
  --out artifacts/teacher/eval_behavior_v0.json || fail teacher_scorecard
echo "MARKER:TEACHER_SCORED"

# --- 2. top-n verified-generation pilot -----------------------------------
uv run python scripts/rollout/generate_teacher_answers.py \
  --model "$TEACHER" \
  --data-dir data/pilot_pack \
  --limit-per-slice "$PILOT_LIMIT" \
  --n "$PILOT_N" \
  --batch-size "$PILOT_BATCH" \
  --max-new-tokens "$GEN_MAX_NEW_TOKENS" \
  --out artifacts/stage2_v2/pilot || fail pilot
echo "MARKER:PILOT_DONE"
