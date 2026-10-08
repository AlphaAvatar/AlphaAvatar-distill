#!/usr/bin/env bash
# Dev-box driver for the 2026-08-04 diagnostic pod. Runs under nohup so the
# session survives the orchestrating process.
#
# Budget discipline, in three independent layers:
#   1. GPU price is CHECKED before creating anything. An RTX A6000 above the
#      authorized $0.33/h, or absent, stops the run — there is no fallback to a
#      pricier card, because the authorization named both the card and the price.
#   2. `--terminate-after` is an absolute UTC deadline set from BACKSTOP_MINUTES.
#      It is RunPod-side, so it fires even if this launcher, the pod-side driver
#      and the poller are all dead. At 5 h x $0.33 that caps exposure at $1.65.
#   3. The launcher deletes the pod itself the moment ALL_DONE lands, so the
#      backstop is a floor on failure, never the normal path (pods idle-bill).
#
#   SCR=… SESSION_COMMIT=… BUNDLE_NAME=… bash scripts/pod/p0asst_launch.sh
set -uo pipefail

SCR=${SCR:?}
SESSION_COMMIT=${SESSION_COMMIT:?}
BUNDLE_NAME=${BUNDLE_NAME:?}
GPU_NAME=${GPU_NAME:-NVIDIA RTX A6000}
MAX_PRICE=${MAX_PRICE:-0.33}
# Minutes, not hours: `date -d "+4.5 hours"` is rejected by GNU date and
# silently produced an EMPTY --terminate-after on 2026-08-04, creating a pod
# with no backstop at all -- the one safety property that must survive the
# orchestrator dying. Integer minutes always parse.
BACKSTOP_MINUTES=${BACKSTOP_MINUTES:-270}
STARTUP_LIMIT_MIN=${STARTUP_LIMIT_MIN:-15}
MAX_POD_ATTEMPTS=${MAX_POD_ATTEMPTS:-2}
REF_REVISION=${REF_REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}
TEACHER_REVISION=${TEACHER_REVISION:-768f209d9ea81521153ed38c47d515654e938aea}
LOG=$SCR/p0asst_launch.log
KEY=$(cat "$SCR/rp_key")
STATE=$SCR/p0asst.state

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
# `runpodctl pod create` provisions SECURE cloud, so the guard must read
# securePrice. lowestPrice/uninterruptablePrice is the COMMUNITY floor: on
# 2026-08-04 it read $0.33 for an A6000 that then billed $0.53, and two earlier
# pods were costed against that wrong number.
PRICE=$(gql "{\"query\":\"query { gpuTypes(input:{id:\\\"$GPU_NAME\\\"}) { id securePrice communityPrice lowestPrice(input:{gpuCount:1}) { stockStatus } } }\"}" |
  python3 -c "
import sys, json
d = json.load(sys.stdin)['data']['gpuTypes']
if not d:
    print('')
else:
    g = d[0]
    sys.stderr.write(f\"  securePrice={g.get('securePrice')} communityPrice={g.get('communityPrice')} stock={(g.get('lowestPrice') or {}).get('stockStatus')}\\n\")
    print(g.get('securePrice') if g.get('securePrice') is not None else '')
" 2>>"$LOG")
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
  deadline=$(date -u -d "+${BACKSTOP_MINUTES} minutes" +%Y-%m-%dT%H:%M:%SZ)
  if [ -z "$deadline" ]; then
    echo "[$(date -u +%FT%TZ)] FATAL: empty backstop deadline; refusing" >>"$LOG"
    return 1
  fi
  echo "[$(date -u +%FT%TZ)]   backstop $deadline (${BACKSTOP_MINUTES} min)" >>"$LOG"
  runpodctl pod create \
    --image "${POD_IMAGE:-runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404}" \
    --gpu-id "$GPU_NAME" --gpu-count 1 \
    --container-disk-in-gb 100 --volume-in-gb 0 \
    --min-cuda-version 13.0 \
    --ports "22/tcp" \
    --name "aadistill-p0asst-a$1" \
    --terminate-after "$deadline" >"$SCR/create_raw_$1.txt" 2>&1
  python3 - "$SCR/create_raw_$1.txt" <<'PYEOF'
import json, re, sys
raw = open(sys.argv[1]).read()
pid = ""
try:                                   # runpodctl emits JSON on success
    pid = json.loads(raw).get("id", "")
except Exception:
    m = re.search(r'"id"\s*:\s*"([^"]+)"', raw)
    pid = m.group(1) if m else ""
print(pid)
PYEOF
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
EP=""; POD_ID=${POD_ID:-}
for attempt in $(seq 1 "$MAX_POD_ATTEMPTS"); do
  if [ -z "$POD_ID" ]; then
    POD_ID=$(create_pod "$attempt")
    if [ -z "$POD_ID" ]; then
      say "attempt $attempt: create failed — raw output follows"
      head -20 "$SCR/create_raw_$attempt.txt" | tee -a "$LOG"
      continue
    fi
    # Refuse a pod that provisioned above the authorized rate.
    ACTUAL=$(python3 -c "
import json,sys
try: print(json.load(open('$SCR/create_raw_$attempt.txt')).get('costPerHr',''))
except Exception: print('')")
    if [ -n "$ACTUAL" ] && [ "$(echo "$ACTUAL > $MAX_PRICE" | bc -l)" = "1" ]; then
      say "ABORT: pod provisioned at \$$ACTUAL/h, above the authorized \$$MAX_PRICE/h. Deleting."
      runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
      echo "LAUNCH_ABORT:actual_price_$ACTUAL" > "$STATE"; exit 1
    fi
    say "attempt $attempt: pod costPerHr \$$ACTUAL (authorized \$$MAX_PRICE)"
    date -u +%s > "$SCR/pod_start_epoch"; echo "$POD_ID" > "$SCR/pod_id"
    say "attempt $attempt: created $POD_ID"
  else
    say "attempt $attempt: reusing existing pod $POD_ID"
  fi
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
# Read the cached token file directly. `hf auth token` is not a subcommand in
# every CLI version: where it is missing it prints usage to STDERR and nothing to
# stdout, so the redirect produces a 0-byte file, scp copies it happily, and the
# pod fails 8 minutes later with "Illegal header value b'Bearer '". Verify the
# size here rather than discovering it on the pod.
TOKEN_SRC=${TOKEN_SRC:-$HOME/.cache/huggingface/token}
if [ ! -s "$TOKEN_SRC" ]; then
  say "FATAL: no HF token at $TOKEN_SRC"; teardown
  echo "LAUNCH_FAILED:no_token" > "$STATE"; exit 1
fi
$SCP "$TOKEN_SRC" "root@$HOST:/workspace/hf/token" >>"$LOG" 2>&1
$SSH "root@$HOST" 'test -s /workspace/hf/token' \
  || { say "FATAL: token arrived empty on the pod"; teardown
       echo "LAUNCH_FAILED:empty_token" > "$STATE"; exit 1; }
$SCP scripts/pod/p0asst_setup.sh "root@$HOST:/workspace/" >>"$LOG" 2>&1

say "running setup"
$SSH "root@$HOST" "cd /workspace && SESSION_COMMIT=$SESSION_COMMIT \
  BUNDLE_NAME=$BUNDLE_NAME REF_REVISION=$REF_REVISION \
  TEACHER_REVISION=$TEACHER_REVISION bash /workspace/p0asst_setup.sh" \
  >>"$SCR/p0asst_setup.log" 2>&1
if [ $? -ne 0 ]; then
  say "FATAL: setup failed. $(cost)"; tail -25 "$SCR/p0asst_setup.log" | tee -a "$LOG"
  teardown; echo "LAUNCH_FAILED:setup" > "$STATE"; exit 1
fi
say "setup done — $(cost)"

say "starting P0-assistant training then the D0.3 harness"
$SSH "root@$HOST" "cd /workspace/aad && nohup /opt/train/bin/python \
  scripts/pod/p0asst_driver.py --stage all > /workspace/p0asst_run.log 2>&1 &" >>"$LOG" 2>&1
say "driver running; poller takes over — $(cost)"
