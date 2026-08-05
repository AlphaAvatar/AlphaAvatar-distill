#!/usr/bin/env bash
# Dev-box orchestrator for the Experiment 3 pod (attention-update restriction).
# Runs under nohup so a paid pod never depends on a conversation staying open.
#
#   SCR=… SESSION_COMMIT=… BUNDLE_NAME=… bash scripts/pod/e3_launch.sh
#
# Budget discipline, in four independent layers:
#   1. GPU securePrice is CHECKED before creating anything, and the pod's actual
#      costPerHr is re-checked after creation. Above the authorized rate, stop —
#      no fallback to a pricier card.
#   2. `--terminate-after` is an absolute UTC deadline from BACKSTOP_MINUTES.
#      It is RunPod-side, so it fires even if this launcher, the pod-side driver
#      and the poller are all dead.
#   3. This script polls for ALL_DONE and deletes the pod itself, so the backstop
#      is a floor on failure, never the normal path (pods idle-bill).
#   4. Checkpoint transfer is time-boxed. Results are fetched BEFORE weights, so
#      a slow link costs weights, never the experiment.
set -uo pipefail

SCR=${SCR:?}
SESSION_COMMIT=${SESSION_COMMIT:?}
BUNDLE_NAME=${BUNDLE_NAME:?}
GPU_NAME=${GPU_NAME:-NVIDIA L40S}
MAX_PRICE=${MAX_PRICE:-0.99}
# Integer minutes: `date -d "+7.5 hours"` is rejected by GNU date and silently
# produces an EMPTY --terminate-after, i.e. a pod with no backstop at all.
# 450 min x $0.99 = $7.43 worst case, against ~$6.2 expected.
BACKSTOP_MINUTES=${BACKSTOP_MINUTES:-450}
STARTUP_LIMIT_MIN=${STARTUP_LIMIT_MIN:-15}
MAX_POD_ATTEMPTS=${MAX_POD_ATTEMPTS:-2}
POLL_LIMIT_MIN=${POLL_LIMIT_MIN:-430}
CKPT_TRANSFER_LIMIT_MIN=${CKPT_TRANSFER_LIMIT_MIN:-45}
TEACHER_REVISION=${TEACHER_REVISION:-768f209d9ea81521153ed38c47d515654e938aea}
STORE=${STORE:-/home/ecs-user/aad-artifacts/e3}
LOG=$SCR/e3_launch.log
KEY=$(cat "$SCR/rp_key")
STATE=$SCR/e3.state

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
# securePrice. lowestPrice/uninterruptablePrice is the COMMUNITY floor and has
# already caused two runs to be under-reported.
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
    --container-disk-in-gb 150 --volume-in-gb 0 \
    --min-cuda-version 13.0 \
    --ports "22/tcp" \
    --name "aadistill-e3-a$1" \
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
    ACTUAL=$(python3 -c "
import json,sys
try: print(json.load(open('$SCR/create_raw_$attempt.txt')).get('costPerHr',''))
except Exception: print('')")
    if [ -n "$ACTUAL" ] && [ "$(echo "$ACTUAL > $MAX_PRICE" | bc -l)" = "1" ]; then
      say "ABORT: pod provisioned at \$$ACTUAL/h, above \$$MAX_PRICE/h. Deleting."
      runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
      echo "LAUNCH_ABORT:actual_price_$ACTUAL" > "$STATE"; exit 1
    fi
    say "attempt $attempt: pod costPerHr \$$ACTUAL (authorized \$$MAX_PRICE)"
    date -u +%s > "$SCR/pod_start_epoch"; echo "$POD_ID" > "$SCR/pod_id"
    say "attempt $attempt: created $POD_ID"
  else
    say "attempt $attempt: reusing existing pod $POD_ID"
  fi
  # runpodctl 2.7.1 always reports uptimeSeconds 0; readiness is a real TCP 22
  # mapping from GraphQL plus an actual SSH attempt, never the CLI field.
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
# Read the cached token file directly: `hf auth token` is missing in some CLI
# versions, where the redirect produces a 0-byte file and the pod fails eight
# minutes later with "Illegal header value b'Bearer '".
TOKEN_SRC=${TOKEN_SRC:-$HOME/.cache/huggingface/token}
if [ ! -s "$TOKEN_SRC" ]; then
  say "FATAL: no HF token at $TOKEN_SRC"; teardown
  echo "LAUNCH_FAILED:no_token" > "$STATE"; exit 1
fi
$SCP "$TOKEN_SRC" "root@$HOST:/workspace/hf/token" >>"$LOG" 2>&1
$SSH "root@$HOST" 'test -s /workspace/hf/token' \
  || { say "FATAL: token arrived empty on the pod"; teardown
       echo "LAUNCH_FAILED:empty_token" > "$STATE"; exit 1; }
$SCP scripts/pod/e3_setup.sh "root@$HOST:/workspace/" >>"$LOG" 2>&1
$SSH "root@$HOST" 'mkdir -p /workspace/aad_holdout'
$SCP data/warmup/holdout_v1.jsonl "root@$HOST:/workspace/aad_holdout/" >>"$LOG" 2>&1

say "running setup"
$SSH "root@$HOST" "cd /workspace && SESSION_COMMIT=$SESSION_COMMIT \
  BUNDLE_NAME=$BUNDLE_NAME TEACHER_REVISION=$TEACHER_REVISION \
  HOLDOUT_SRC=/workspace/aad_holdout/holdout_v1.jsonl \
  bash /workspace/e3_setup.sh" \
  >>"$SCR/e3_setup.log" 2>&1
if [ $? -ne 0 ]; then
  say "FATAL: setup failed. $(cost)"; tail -25 "$SCR/e3_setup.log" | tee -a "$LOG"
  teardown; echo "LAUNCH_FAILED:setup" > "$STATE"; exit 1
fi
say "setup done — $(cost)"

say "starting the E3 driver: A1 (2 seeds) -> freeze gate -> A2 (2 seeds) -> evaluation"
# `nohup ... &` alone does NOT detach here: the remote wrapper shell stays alive
# holding the SSH channel open, so this call blocks until the driver exits and
# the launcher never reaches its polling loop. Measured on 2026-08-05 -- the run
# completed, but with no progress logging for five hours. `setsid` plus a
# closed stdin puts the driver in its own session so the channel closes at once.
$SSH "root@$HOST" "cd /workspace/aad && setsid nohup /opt/train/bin/python \
  scripts/pod/e3_driver.py --stage all > /workspace/e3_run.log 2>&1 < /dev/null & \
  disown" >>"$LOG" 2>&1
say "driver running — $(cost)"

# --- 3. poll to completion -------------------------------------------------
DEADLINE_TS=$(( $(date -u +%s) + POLL_LIMIT_MIN * 60 )); LAST=""
while [ "$(date -u +%s)" -lt "$DEADLINE_TS" ]; do
  sleep 120
  STATUS_TXT=$($SSH "root@$HOST" 'cat /workspace/e3.status 2>/dev/null | tail -1' 2>/dev/null)
  if [ -n "$STATUS_TXT" ] && [ "$STATUS_TXT" != "$LAST" ]; then
    LAST="$STATUS_TXT"; say "  $STATUS_TXT — $(cost)"
  fi
  case "$STATUS_TXT" in
    *ALL_DONE*) say "driver reported ALL_DONE — $(cost)"; break ;;
    *ABORTED_BEFORE_A2*) say "driver aborted before A2 — $(cost)"; break ;;
  esac
done

# --- 4. fetch results, then weights, then delete ---------------------------
mkdir -p "$STORE"
say "bundling small artifacts on the pod"
$SSH "root@$HOST" 'cd /workspace/aad && tar czf /workspace/e3_side.tar.gz \
  artifacts/audit configs/stage3/e3 \
  $(ls -d artifacts/stage3/e3_*/train_log.jsonl artifacts/stage3/e3_*/run_manifest.json 2>/dev/null) \
  2>/dev/null; cp /workspace/e3_run.log /workspace/e3.status /workspace/ 2>/dev/null; \
  sha256sum /workspace/e3_side.tar.gz' >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e3_side.tar.gz" "$STORE/" >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e3_run.log" "$STORE/" >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e3.status" "$STORE/" >>"$LOG" 2>&1
if [ -s "$STORE/e3_side.tar.gz" ]; then
  say "results bundle retrieved ($(du -h "$STORE/e3_side.tar.gz" | cut -f1)) — $(cost)"
else
  say "WARNING: results bundle is empty or missing"
fi

say "hashing checkpoints on the pod"
$SSH "root@$HOST" 'cd /workspace/aad/artifacts/stage3 && \
  find e3_*/checkpoints/step_001023 -type f \( -name "*.safetensors" -o -name "*.json" \
    -o -name "*.jinja" \) | sort | xargs sha256sum' > "$SCR/e3_pod_hashes.txt" 2>>"$LOG"
cp "$SCR/e3_pod_hashes.txt" "$STORE/" 2>/dev/null

say "fetching checkpoints (time-boxed to ${CKPT_TRANSFER_LIMIT_MIN} min)"
timeout "${CKPT_TRANSFER_LIMIT_MIN}m" $SCP -r \
  "root@$HOST:/workspace/aad/artifacts/stage3/e3_a1_frozen_attn_sa/checkpoints/step_001023" \
  "$STORE/e3_a1_frozen_attn_sa" >>"$LOG" 2>&1
for arm in e3_a1_frozen_attn_sb e3_a2_lora_attn_sa e3_a2_lora_attn_sb; do
  ELAPSED=$(( ($(date -u +%s) - $(cat "$SCR/pod_start_epoch")) / 60 ))
  if [ "$ELAPSED" -gt "$POLL_LIMIT_MIN" ]; then
    say "WARNING: past the poll limit; skipping $arm weights to stop billing"
    break
  fi
  timeout "${CKPT_TRANSFER_LIMIT_MIN}m" $SCP -r \
    "root@$HOST:/workspace/aad/artifacts/stage3/$arm/checkpoints/step_001023" \
    "$STORE/$arm" >>"$LOG" 2>&1 \
    || say "WARNING: $arm weights not retrieved (results bundle is unaffected)"
done

teardown
echo "DONE" > "$STATE"
say "session complete. artifacts under $STORE"
