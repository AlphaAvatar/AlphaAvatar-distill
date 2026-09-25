#!/usr/bin/env bash
# Batch-invariance root-cause diagnostic, in the environment the science runs in.
#
# WHY THIS EXISTS RATHER THAN REUSING batching_refactor_cuda_launch.sh
# -------------------------------------------------------------------
# That launcher ran the a4 validation, and the a4 environment was the reason its
# finding was rejected: image `1.0.3-cu1281` (torch 2.9.1+cu128) plus an
# unconstrained `pip install transformers`. The repository turns out to contain
# THREE distinct runtimes, and the difference between them is exactly what is
# under investigation:
#
#   A  a4, rejected          image 1.0.3-cu1281 | torch 2.9.1+cu128 | tf: latest
#   B  engineering CUDA      image 1.1.0-cu1300 | torch 2.9.1+cu130 | image python
#   C  FORMAL SCIENCE        image 1.1.0-cu1300 | torch 2.11.0+cu128 | tf 5.13.1
#                            under /opt/train, offline from the relay wheelhouse
#
# C is the one the operator search actually executes in -- `POD_IMAGE
# ['remote_python']` is `/opt/train/bin/python`, and C1 attempts 17 and 18
# recorded torch 2.11.0+cu128 / transformers 5.13.1 in their evidence. So C is
# where a claim about C3's decisions has to be made. B is measured too, on the
# same pod and the same weights, because "does the runtime change the answer" is
# a question this session is supposed to settle rather than assume.
#
# ONE resource, ONE job, teardown in a trap on every exit path.
#
#   nohup bash scripts/pod/batch_invariance_diagnostic_launch.sh <run-id> > LOG 2>&1 &
#
set -uo pipefail

RUN_ID="${1:?usage: $0 <run-id>}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${REPO_DIR}/artifacts/validation/batch_invariance/${RUN_ID}"
BRANCH="review/c3-operator-batching"
COMMIT="${DIAGNOSTIC_COMMIT:?DIAGNOSTIC_COMMIT must name the exact source to run}"

#: The formal image, not a4's. See the table above.
IMAGE="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"
GPU="NVIDIA L40S"
#: 3.82 GiB wheelhouse + 7.6 GiB parent + a 16 GiB fp32 materialization headroom.
DISK_GB=120
RATE_USD_H=1.09

# --- the budget, DERIVED by a tested script --------------------------------
# Not inline. The arithmetic used to live here in bash, where the only way to
# find out what it computed was to create a pod; `batch_invariance_budget.py`
# has unit tests and this script just obeys it. The ceiling is owned by the
# AUTHORIZATION record and the costs by the CAMPAIGN record -- one owner each,
# because while two copies of a number agree, no gate can tell you which it read.
CAMPAIGN="logs/stages/stage-1/phase_c3/investigations/batch-invariance-root-cause/v1/campaign.json"
SESSION_CAP="${SESSION_CAP:-2.50}"
BUDGET=$("${REPO_DIR}/.venv/bin/python" \
         "${REPO_DIR}/scripts/pod/batch_invariance_budget.py" \
         "${REPO_DIR}/${CAMPAIGN}" --session-cap "$SESSION_CAP" --rate "$RATE_USD_H")
BUDGET_RC=$?
if [ "$BUDGET_RC" -ne 0 ]; then
  echo "budget could not be established; not creating a pod"
  exit 4
fi
read -r CAMPAIGN_CEILING SPENT_USD TEARDOWN_RESERVE REMAINING_USD \
        CEILING_USD MAX_SECONDS BUDGET_OK <<<"$BUDGET"
if [ "$BUDGET_OK" != "yes" ]; then
  echo "campaign \$${CAMPAIGN_CEILING}, spent \$${SPENT_USD}, remaining \$${REMAINING_USD};"
  echo "after the \$${TEARDOWN_RESERVE} teardown reserve that funds only \$${CEILING_USD}."
  echo "Below what a useful attempt needs. Not creating a pod."
  exit 4
fi

mkdir -p "$OUT"
LOG="${OUT}/launch.log"
POD_ID=""
STARTED_EPOCH=$(date -u +%s)

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- the relay credential ---------------------------------------------------
# The wheelhouse and the calibration mixture both live on the private relay, and
# a missing credential surfaces on the pod as RepositoryNotFoundError -- which
# reads like a missing repo, not a missing token. So it is resolved HERE, at $0,
# and a pod is not created without one.
#
# Not from $HF_TOKEN alone: this dev box authenticates from the stored login and
# has no such variable, so the env-var-only check refused a fully authenticated
# machine. `get_token()` is the same resolution `huggingface_hub` itself uses.
HF_TOKEN="${HF_TOKEN:-$("${REPO_DIR}/.venv/bin/python" -c \
  'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null)}"
if [ -z "$HF_TOKEN" ]; then
  say "no Hugging Face credential: neither \$HF_TOKEN nor a stored login."
  say "The wheelhouse and the calibration mixture are both on the private"
  say "relay. Refusing to create a pod that cannot obtain its inputs."
  exit 4
fi
export HF_TOKEN

KEY=$(grep -i apikey ~/.runpod/config.toml | sed "s/.*= *//;s/[\"']//g")
gql() {
  curl -s --max-time 30 -H "Authorization: Bearer ${KEY}" \
       -H "Content-Type: application/json" -d "$1" https://api.runpod.io/graphql
}

# --- teardown, on every exit path ------------------------------------------
WATCHDOG_PID=""
teardown() {
  local rc=$?
  # The watchdog's job ends when the pod does. Verified by pid AND by the
  # process actually being this watchdog: a reused pid answers kill(pid, 0).
  if [ -n "$WATCHDOG_PID" ] && [ -r "/proc/${WATCHDOG_PID}/cmdline" ] \
     && tr '\0' ' ' < "/proc/${WATCHDOG_PID}/cmdline" | grep -q "watchdog.py.*${POD_ID}"; then
    say "stopping watchdog ${WATCHDOG_PID}"
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [ -n "$POD_ID" ]; then
    say "teardown: removing pod ${POD_ID}"
    for attempt in 1 2 3 4 5; do
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
  python3 - "$OUT" "$elapsed" "$RATE_USD_H" "$POD_ID" "$rc" "$DISK_GB" <<'PY'
import json, sys
out, elapsed, rate, pod, rc, disk_gb = (
    sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
    int(sys.argv[5]), int(sys.argv[6]))
minutes = elapsed / 60.0
gpu = round(minutes / 60.0 * rate, 4)
# Container disk is billed SEPARATELY at $0.10/GB/month. A GPU-only ceiling
# once missed $1.69 of it and nothing in that record disagreed.
disk = round(disk_gb * 0.10 / (30 * 24 * 60) * minutes, 4)
json.dump({"elapsed_minutes": round(minutes, 3), "rate_usd_per_hour": rate,
           "container_disk_gb": disk_gb,
           "gpu_usd": gpu, "container_disk_usd": disk,
           "all_in_usd": round(gpu + disk, 4), "pod_id": pod or None,
           "script_exit": rc,
           "_basis": "wall clock from pod creation to confirmed teardown x quoted "
                     "secure rate, plus container disk billed separately"},
          open(f"{out}/cost.json", "w"), indent=1)
PY
  say "cost: $(python3 -c "import json;print(json.load(open('${OUT}/cost.json'))['all_in_usd'])") USD all-in"
  say "exit ${rc}"
}
trap teardown EXIT

# --- acquire ----------------------------------------------------------------
say "budget: campaign \$${CAMPAIGN_CEILING} - spent \$${SPENT_USD} = \$${REMAINING_USD}"
say "        less \$${TEARDOWN_RESERVE} teardown reserve, capped at \$${SESSION_CAP} -> session \$${CEILING_USD}"
say "creating ${GPU} (securePrice \$${RATE_USD_H}/h, ceiling \$${CEILING_USD}, useful runtime ${MAX_SECONDS}s)"
CREATE=$(runpodctl create pod \
  --name "batch-inv-${RUN_ID}" \
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
printf '%s' "$STARTED_EPOCH" > "${OUT}/pod_start_epoch"

# --- the independent watchdog ----------------------------------------------
# The EXIT trap above is the FIRST layer and it does not run if this script is
# SIGKILLed, if the box reboots, or if the conversation that started it goes
# away. A launcher whose only teardown is its own trap is an ownerless billing
# resource waiting to happen -- this project has already lost ~116 minutes of
# L40S to a backstop that "remained" without checking anything, and RunPod's own
# --terminate-after has never been observed to fire here. So a separate process
# owns the deadline, reads only the provider control plane, and outlives this.
WATCHDOG_MINUTES=$(python3 -c "print(int(${MAX_SECONDS}/60) + 12)")
setsid nohup "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/scripts/pod/watchdog.py" \
  --pod-id "$POD_ID" --session-start-epoch "$STARTED_EPOCH" \
  --price-per-hour "$RATE_USD_H" --hard-minutes "$WATCHDOG_MINUTES" \
  --authorized-usd "$CEILING_USD" \
  --journal "${OUT}/watchdog.jsonl" > "${OUT}/watchdog.out" 2>&1 < /dev/null &
WATCHDOG_PID=$!
say "watchdog pid ${WATCHDOG_PID}, hard deadline ${WATCHDOG_MINUTES} min"
printf '%s' "$WATCHDOG_PID" > "${OUT}/watchdog_pid"

# --- readiness: GraphQL runtime, never the CLI uptime field -----------------
# 15 minutes with no runtime means it is not starting; that is a delete-and-
# recreate decision, and this script exits rather than idling against a pod
# that will not boot.
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 90); do
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

SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -p ${SSH_PORT} root@${SSH_HOST}"
for _ in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 5; done

# --- run --------------------------------------------------------------------
# The remote payload is a SEPARATE file, rendered and shipped rather than
# embedded in an unquoted heredoc. Attempt a2 cost $0.0214 because backticks
# inside an unquoted heredoc were executed by the LOCAL shell.
REMOTE_SH="${OUT}/remote.sh"
sed -e "s|@BRANCH@|${BRANCH}|g" \
    -e "s|@COMMIT@|${COMMIT}|g" \
    -e "s|@RUN_ID@|${RUN_ID}|g" \
    "${REPO_DIR}/scripts/pod/batch_invariance_diagnostic_remote.sh" > "$REMOTE_SH"
say "shipping remote payload ($(wc -l < "$REMOTE_SH") lines)"

say "setup + diagnostic (bounded at ${MAX_SECONDS}s)"
timeout "${MAX_SECONDS}" $SSH "HF_TOKEN=${HF_TOKEN} bash -s" < "$REMOTE_SH" >>"$LOG" 2>&1
RC=$?
say "remote finished rc=${RC}"

# --- collect BEFORE teardown ------------------------------------------------
# Gated on the pod having RUN, never on it having SUCCEEDED: a diagnostic that
# failed in its last stage still holds every number the earlier ones produced,
# and $2.82 of verified checkpoints were once deleted by a collector that ran
# only on a clean terminal state.
say "fetching evidence"
scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${SSH_PORT}" \
    "root@${SSH_HOST}:/workspace/out/." "${OUT}/" >>"$LOG" 2>&1 \
  || say "WARNING: could not fetch /workspace/out"
FOUND=$(find "$OUT" -name report.json | wc -l)
say "retrieved ${FOUND} report(s)"
for r in $(find "$OUT" -name report.json | sort); do
  say "  $(basename "$(dirname "$r")"): locus=$(python3 -c "import json;print(json.load(open('$r'))['conclusion']['divergence_locus'])" 2>/dev/null || echo unreadable)"
done
exit "$RC"
