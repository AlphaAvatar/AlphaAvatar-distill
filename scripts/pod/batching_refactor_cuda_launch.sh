#!/usr/bin/env bash
# Focused real-CUDA validation of the operator-topology + batching refactor.
#
# One resource, one job, teardown in a trap. Runs under nohup on the dev box so
# the paid pod never depends on a conversation staying open, and tears the pod
# down on EVERY exit path -- success, failure, or the script itself being
# killed. A pod that outlives its job is the failure this project has already
# paid for more than once.
#
#   nohup bash scripts/pod/batching_refactor_cuda_launch.sh <run-id> > LOG 2>&1 &
#
set -uo pipefail

RUN_ID="${1:?usage: $0 <run-id>}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${REPO_DIR}/artifacts/validation/${RUN_ID}"
BRANCH="review/c3-operator-batching"
COMMIT="${VALIDATION_COMMIT:?VALIDATION_COMMIT must name the exact source to run}"
IMAGE="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
GPU="NVIDIA L40S"
DISK_GB=60
CEILING_USD=0.25
RATE_USD_H=1.09
# Useful runtime after the $0.05 teardown reserve, in seconds.
MAX_SECONDS=$(python3 -c "print(int((${CEILING_USD}-0.05)/${RATE_USD_H}*3600))")

mkdir -p "$OUT"
LOG="${OUT}/launch.log"
POD_ID=""
STARTED_EPOCH=$(date -u +%s)

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

KEY=$(grep -i apikey ~/.runpod/config.toml | sed "s/.*= *//;s/[\"']//g")
gql() {
  curl -s --max-time 30 -H "Authorization: Bearer ${KEY}" \
       -H "Content-Type: application/json" -d "$1" https://api.runpod.io/graphql
}

# --- teardown, on every exit path ------------------------------------------
teardown() {
  local rc=$?
  if [ -n "$POD_ID" ]; then
    say "teardown: removing pod ${POD_ID}"
    for attempt in 1 2 3 4 5; do
      # Both spellings: `remove pod` is deprecated in favour of `pod delete`,
      # and a teardown that depends on which CLI version is installed is a
      # teardown that can silently not happen.
      # Both spellings, but only try the second if the first did not take:
      # calling delete on an already-removed pod returns a 404 that reads like
      # a teardown failure in the log when it is the opposite.
      runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1 \
        || runpodctl pod delete "$POD_ID" >>"$LOG" 2>&1 || true
      sleep 6
      local still
      still=$(gql "{\"query\":\"query { myself { pods { id } } }\"}" \
              | python3 -c "import json,sys;print(sum(1 for p in json.load(sys.stdin)['data']['myself']['pods'] if p['id']=='${POD_ID}'))" 2>/dev/null || echo 1)
      if [ "$still" = "0" ]; then
        say "teardown: provider confirms ${POD_ID} is gone (attempt ${attempt})"
        printf '%s' "confirmed" > "${OUT}/teardown_confirmed"
        break
      fi
      say "teardown: still listed after attempt ${attempt}; retrying"
    done
  fi
  local elapsed=$(( $(date -u +%s) - STARTED_EPOCH ))
  python3 - "$OUT" "$elapsed" "$RATE_USD_H" "$POD_ID" "$rc" <<'PY'
import json, sys
out, elapsed, rate, pod, rc = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
minutes = elapsed / 60.0
# Billed from pod creation, not from when the job started.
gpu = round(minutes / 60.0 * rate, 4)
disk = round(60 * 0.10 / (30 * 24 * 60) * minutes, 4)   # container disk, billed separately
json.dump({"elapsed_minutes": round(minutes, 3), "rate_usd_per_hour": rate,
           "gpu_usd": gpu, "container_disk_usd": disk,
           "all_in_usd": round(gpu + disk, 4), "pod_id": pod or None,
           "script_exit": rc,
           "_basis": "wall clock from pod creation to confirmed teardown x quoted secure rate, plus container disk billed separately"},
          open(f"{out}/cost.json", "w"), indent=1)
PY
  say "cost: $(python3 -c "import json;print(json.load(open('${OUT}/cost.json'))['all_in_usd'])") USD all-in"
  say "exit ${rc}"
}
trap teardown EXIT

# --- acquire ----------------------------------------------------------------
say "creating ${GPU} (securePrice \$${RATE_USD_H}/h, ceiling \$${CEILING_USD}, useful runtime ${MAX_SECONDS}s)"
# Flag spellings verified against `runpodctl create pod --help` at $0 BEFORE
# spending: this CLI takes camelCase (--gpuType, --containerDiskSize,
# --volumeSize, --imageName, --secureCloud), not the kebab-case names an older
# note used. `--cost` is a hard $/hr ceiling the provider itself enforces, so a
# price move cannot silently buy a more expensive pod than the authorization
# priced. `--startSSH` is required or no sshd runs; `--ports 22/tcp` is required
# or no public mapping ever appears.
CREATE=$(runpodctl create pod \
  --name "batching-cuda-${RUN_ID}" \
  --imageName "$IMAGE" \
  --gpuType "$GPU" \
  --gpuCount 1 \
  --containerDiskSize "$DISK_GB" \
  --volumeSize 0 \
  --ports "22/tcp" \
  --cost "$RATE_USD_H" \
  --startSSH \
  --secureCloud 2>&1)
echo "$CREATE" >>"$LOG"
POD_ID=$(echo "$CREATE" | sed -n 's/.*pod "\([a-z0-9]\{10,\}\)".*/\1/p' | head -1)
[ -z "$POD_ID" ] && POD_ID=$(echo "$CREATE" | grep -oE '\b[a-z0-9]{13,16}\b' | head -1)
if [ -z "$POD_ID" ]; then say "FAILED to create a pod; nothing billed"; exit 3; fi
say "pod ${POD_ID}"
printf '%s' "$POD_ID" > "${OUT}/pod_id"

# --- readiness: GraphQL runtime, never the CLI uptime field -----------------
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 60); do
  sleep 10
  INFO=$(gql "{\"query\":\"query { pod(input:{podId:\\\"${POD_ID}\\\"}) { runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic } } } }\"}")
  read -r SSH_HOST SSH_PORT < <(echo "$INFO" | python3 -c "
import json,sys
try:
    rt=json.load(sys.stdin)['data']['pod']['runtime'] or {}
except Exception: rt={}
for p in (rt.get('ports') or []):
    if p.get('privatePort')==22 and p.get('isIpPublic'):
        print(p['ip'], p['publicPort']); break
else: print('', '')
" 2>/dev/null || echo " ")
  [ -n "$SSH_HOST" ] && break
  say "waiting for a public port 22 mapping..."
done
if [ -z "$SSH_HOST" ]; then say "pod never exposed port 22"; exit 3; fi
say "ssh ${SSH_HOST}:${SSH_PORT}"

SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p ${SSH_PORT} root@${SSH_HOST}"
for _ in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 5; done

# --- run ---
# NOTE: this heredoc is UNQUOTED so ${BRANCH}/${COMMIT}/${RUN_ID} expand here.
# That means the LOCAL shell also performs command substitution on it: a
# backtick in a COMMENT inside this block is executed on the dev box, which is
# how attempt a2 ran 'pip install transformers' locally and sent nonsense to
# the pod for $0.0214. No backticks and no unescaped $( below this line.-----------------------------------------------------------------
say "setup + validation (bounded at ${MAX_SECONDS}s)"
timeout "${MAX_SECONDS}" $SSH bash -s <<REMOTE >>"$LOG" 2>&1
set -euo pipefail
export HF_TOKEN='${HF_TOKEN:-}'
export HF_HOME=/workspace/hf
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd /workspace
git clone --quiet --branch ${BRANCH} --depth 40 \
  https://github.com/AlphaAvatar/AlphaAvatar-distill.git repo
cd repo
git checkout --quiet ${COMMIT}
echo "SOURCE_SHA=\$(git rev-parse HEAD)"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
# The RunPod pytorch image marks its python EXTERNALLY-MANAGED (PEP 668), so a
# plain 'pip install' refuses with a hint rather than installing -- attempt a1
# died there in 9 seconds. In a disposable container the override is the right
# answer; there is no system package manager to conflict with. 'numpy' is also
# absent from the image and torch warns about it on import.
pip install -q --break-system-packages --no-cache-dir \
    numpy transformers huggingface_hub safetensors 2>&1 | tail -3
python -c "import numpy, transformers, safetensors, huggingface_hub as h; \
print('deps ok: numpy', numpy.__version__, '| transformers', transformers.__version__)"
PYTHONPATH=src:scripts python scripts/validation/batching_refactor_cuda_check.py \
  --run-id ${RUN_ID} --device cuda --out /workspace/out
echo "CHECK_RC=\$?"
REMOTE
RC=$?
say "remote finished rc=${RC}"

# --- collect BEFORE teardown ------------------------------------------------
say "fetching evidence"
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${SSH_PORT}" \
    "root@${SSH_HOST}:/workspace/out/batching_refactor_cuda_report.json" \
    "${OUT}/" >>"$LOG" 2>&1 || say "WARNING: could not fetch the report"
[ -f "${OUT}/batching_refactor_cuda_report.json" ] \
  && say "verdict: $(python3 -c "import json;print(json.load(open('${OUT}/batching_refactor_cuda_report.json'))['verdict'])")" \
  || say "no report retrieved"
exit "$RC"
