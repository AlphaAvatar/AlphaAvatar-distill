#!/usr/bin/env bash
# Dev-box driver for the 2026-08-04 diagnostic pod. Runs under nohup so the
# session survives the orchestrating process.
#
# Budget discipline, in three independent layers:
#   1. GPU price is CHECKED before creating anything. An RTX A6000 above the
#      authorized $0.33/h, or absent, stops the run — there is no fallback to a
#      pricier card, because the authorization named both the card and the price.
#   2. `--terminate-after` is an absolute UTC deadline set from BACKSTOP_HOURS.
#      It is RunPod-side, so it fires even if this launcher, the pod-side driver
#      and the poller are all dead. At 5 h x $0.33 that caps exposure at $1.65.
#   3. The launcher deletes the pod itself the moment ALL_DONE lands, so the
#      backstop is a floor on failure, never the normal path (pods idle-bill).
#
#   SCR=… SESSION_COMMIT=… BUNDLE_NAME=… bash scripts/pod/e2diag_launch.sh
set -uo pipefail

SCR=${SCR:?}
SESSION_COMMIT=${SESSION_COMMIT:?}
BUNDLE_NAME=${BUNDLE_NAME:?}
GPU_NAME=${GPU_NAME:-NVIDIA RTX A6000}
MAX_PRICE=${MAX_PRICE:-0.33}
BACKSTOP_HOURS=${BACKSTOP_HOURS:-5}
STARTUP_LIMIT_MIN=${STARTUP_LIMIT_MIN:-15}
MAX_POD_ATTEMPTS=${MAX_POD_ATTEMPTS:-2}
REF_REVISION=${REF_REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}
TEACHER_REVISION=${TEACHER_REVISION:-768f209d9ea81521153ed38c47d515654e938aea}
CONTROL_SRC=${CONTROL_SRC:-artifacts/stage3/rescued/e1_ctl_r0250k_sa_pca_stepmatched}
LOG=$SCR/e2diag_launch.log
KEY=$(cat "$SCR/rp_key")
STATE=$SCR/e2diag.state

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
cost() {
  local mins=$(( ($(date -u +%s) - $(cat "$SCR/pod_start_epoch")) / 60 ))
  printf '%d min elapsed, $%.2f billed\n' "$mins" "$(echo "$mins/60*$MAX_PRICE" | bc -l)"
}
gql() {
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
    -H 'Content-Type: application/json' -d "$1"
}

# --- 1. price guard --------------------------------------------------------
say "checking $GPU_NAME availability at <= \$$MAX_PRICE/h"
PRICE=$(gql '{"query":"query { gpuTypes { id displayName memoryInGb lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }"}' |
  GPU_NAME="$GPU_NAME" python3 -c "
import sys, json, os
want = os.environ['GPU_NAME']
d = json.load(sys.stdin)['data']['gpuTypes']
for g in d:
    if g['id'] == want or g['displayName'] == want.replace('NVIDIA ', ''):
        p = (g.get('lowestPrice') or {}).get('uninterruptablePrice')
        print(p if p is not None else '')
        break
else:
    print('')
")
if [ -z "$PRICE" ]; then
  say "ABORT: $GPU_NAME not offered right now. Not substituting another GPU."
  echo "LAUNCH_ABORT:gpu_unavailable" > "$STATE"; exit 1
fi
if [ "$(echo "$PRICE > $MAX_PRICE" | bc -l)" = "1" ]; then
  say "ABORT: $GPU_NAME is \$$PRICE/h, above the authorized \$$MAX_PRICE/h."
  say "Not falling back to a pricier GPU. Stopping for a human decision."
  echo "LAUNCH_ABORT:price_$PRICE" > "$STATE"; exit 1
fi
say "$GPU_NAME available at \$$PRICE/h — within the \$$MAX_PRICE cap"

create_pod() {
  local deadline
  deadline=$(date -u -d "+${BACKSTOP_HOURS} hours" +%Y-%m-%dT%H:%M:%SZ)
  say "  backstop --terminate-after $deadline (${BACKSTOP_HOURS}h => max \$$(echo "$BACKSTOP_HOURS*$MAX_PRICE" | bc -l))"
  runpodctl pod create \
    --image "${POD_IMAGE:-runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404}" \
    --gpu-id "$GPU_NAME" --gpu-count 1 \
    --container-disk-in-gb 100 --volume-in-gb 0 \
    --min-cuda-version 13.0 \
    --ports "22/tcp" \
    --name "aadistill-e2diag-a$1" \
    --terminate-after "$deadline" 2>&1 |
  python3 -c "import sys,re; m=re.search(r'\"id\":\s*\"([^\"]+)\"', sys.stdin.read()); print(m.group(1) if m else '')"
}

endpoint() {
  gql "{\"query\":\"query { pod(input:{podId:\\\"$POD_ID\\\"}) { runtime { ports { ip publicPort privatePort type } } } }\"}" |
  python3 -c "import sys,json
d=json.load(sys.stdin); rt=(d.get('data') or {}).get('pod',{}).get('runtime')
print('' if not rt else next((f\"{p['ip']} {p['publicPort']}\" for p in (rt.get('ports') or []) if p.get('privatePort')==22 and p.get('type')=='tcp'),''))" 2>/dev/null
}

teardown() {
  say "deleting pod $POD_ID"
  runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
  say "pod deleted — $(cost)"
}

# --- 2. create, bounded ----------------------------------------------------
EP=""; POD_ID=""
for attempt in $(seq 1 "$MAX_POD_ATTEMPTS"); do
  POD_ID=$(create_pod "$attempt")
  [ -z "$POD_ID" ] && { say "attempt $attempt: create failed"; continue; }
  date -u +%s > "$SCR/pod_start_epoch"; echo "$POD_ID" > "$SCR/pod_id"
  say "attempt $attempt: created $POD_ID"
  DEADLINE_TS=$(( $(date -u +%s) + STARTUP_LIMIT_MIN * 60 )); i=0
  while [ "$(date -u +%s)" -lt "$DEADLINE_TS" ]; do
    EP=$(endpoint); [ -n "$EP" ] && break
    i=$((i+1)); [ $((i % 6)) -eq 0 ] && say "  starting ($((i*20))s) — $(cost)"
    sleep 20
  done
  [ -n "$EP" ] && break
  say "attempt $attempt: no TCP 22 in ${STARTUP_LIMIT_MIN} min — deleting"
  runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1; POD_ID=""
done
if [ -z "$EP" ]; then
  say "ABORT: ${MAX_POD_ATTEMPTS} pods failed to start. Nothing is running."
  echo "LAUNCH_FAILED:no_endpoint" > "$STATE"; exit 1
fi
HOST=${EP%% *}; PORT=${EP##* }
say "pod ready at $HOST:$PORT — $(cost)"
echo "$HOST $PORT" > "$SCR/ssh_endpoint"
echo "RUNNING" > "$STATE"

SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=30"
SCP="scp -P $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
for i in $(seq 1 30); do $SSH "root@$HOST" true 2>/dev/null && break; sleep 10; done

say "transferring session"
$SSH "root@$HOST" 'mkdir -p /workspace/hf /workspace/ckpt && chmod 700 /workspace/hf'
hf auth token > "$SCR/hf_token" && chmod 600 "$SCR/hf_token"
$SCP "$SCR/hf_token" "root@$HOST:/workspace/hf/token" >>"$LOG" 2>&1
$SSH "root@$HOST" 'chmod 600 /workspace/hf/token'; rm -f "$SCR/hf_token"
$SCP scripts/pod/e2diag_setup.sh "root@$HOST:/workspace/" >>"$LOG" 2>&1

# The control checkpoint exists ONLY on this dev box: the relay holds its
# evaluation JSONs but not its weights.
say "pushing the control checkpoint (2.3 GB) — the relay does not have it"
$SCP -r "$CONTROL_SRC" "root@$HOST:/workspace/ckpt/" >>"$LOG" 2>&1 \
  || { say "FATAL: control checkpoint push failed"; teardown; echo "LAUNCH_FAILED:ckpt" > "$STATE"; exit 1; }
say "control checkpoint pushed — $(cost)"

say "running setup"
$SSH "root@$HOST" "cd /workspace && SESSION_COMMIT=$SESSION_COMMIT \
  BUNDLE_NAME=$BUNDLE_NAME REF_REVISION=$REF_REVISION \
  TEACHER_REVISION=$TEACHER_REVISION bash /workspace/e2diag_setup.sh" \
  >>"$SCR/e2diag_setup.log" 2>&1
if [ $? -ne 0 ]; then
  say "FATAL: setup failed. $(cost)"; tail -25 "$SCR/e2diag_setup.log" | tee -a "$LOG"
  teardown; echo "LAUNCH_FAILED:setup" > "$STATE"; exit 1
fi
say "setup done — $(cost)"

say "starting the three diagnostic stages"
$SSH "root@$HOST" "cd /workspace/aad && nohup /opt/train/bin/python \
  scripts/pod/e2diag_driver.py --stage all --ref-revision $REF_REVISION \
  --teacher-revision $TEACHER_REVISION > /workspace/e2diag_run.log 2>&1 &" >>"$LOG" 2>&1
say "driver running; poller takes over — $(cost)"
