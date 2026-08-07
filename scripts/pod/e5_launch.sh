#!/usr/bin/env bash
# Dev-box orchestrator for the FORMAL E5 pod. The separate pilot pod was
# removed after two attempts showed setup dominates a short session (53 of 57
# minutes, paid again every time); the validation is folded in as stage 1 here
# so setup is paid once. A failed gate stops the run before paid generation.
# Runs under nohup so a paid pod never depends on a conversation staying open.
#
#   SCR=… SESSION_COMMIT=… BUNDLE_NAME=… bash scripts/pod/e4_launch.sh
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
#
# Attempt 3 runs under $8.24: what survived attempts 1-2 ($6.74) plus $1.50.
# 499 min x $0.99 = $8.23, so the RunPod-side deadline cannot on its own exceed
# the authorization even if every other layer fails. Expected work is ~372 min
# on measured phases, leaving ~127 min of headroom for a cold setup, an
# abandoned pod, or an R corpus that packs worse than assumed. The gates
# re-price from ACTUAL elapsed session time regardless.
BACKSTOP_MINUTES=${BACKSTOP_MINUTES:-499}
STARTUP_LIMIT_MIN=${STARTUP_LIMIT_MIN:-15}
MAX_POD_ATTEMPTS=${MAX_POD_ATTEMPTS:-2}
# Seconds between create attempts when the GPU is out of capacity.
CREATE_RETRY_DELAY_S=${CREATE_RETRY_DELAY_S:-300}
POLL_LIMIT_MIN=${POLL_LIMIT_MIN:-487}   # inside the 499-min backstop
CKPT_TRANSFER_LIMIT_MIN=${CKPT_TRANSFER_LIMIT_MIN:-40}
TEACHER_REVISION=${TEACHER_REVISION:-768f209d9ea81521153ed38c47d515654e938aea}
STORE=${STORE:-/home/ecs-user/aad-artifacts/e5}
LOG=$SCR/e5_launch.log
KEY=$(cat "$SCR/rp_key")
STATE=$SCR/e5.state

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
    --name "aadistill-e5-a$1" \
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
      head -3 "$SCR/create_raw_$attempt.txt" | tee -a "$LOG"
      # `stockStatus: Low` does not mean available: the 2026-08-06 pilot launch
      # passed the price guard and then got "There are no longer any instances
      # available" on create. Capacity is a race, so back off and re-try the
      # REGISTERED card rather than substituting a different one -- a different
      # GPU would also change the throughput this pilot exists to measure.
      # Waiting costs nothing: no pod exists yet.
      if [ "$attempt" -lt "$MAX_POD_ATTEMPTS" ]; then
        say "  no capacity; retrying in ${CREATE_RETRY_DELAY_S}s ($((MAX_POD_ATTEMPTS-attempt)) left)"
        sleep "$CREATE_RETRY_DELAY_S"
      fi
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
    # Written ONCE per session, at the FIRST create. On 2026-08-07 a pod that
    # never exposed TCP 22 was deleted and replaced, and resetting this epoch
    # hid its $0.25 from every gate while handing the replacement a fresh full
    # deadline -- together putting the worst case above the authorization.
    # Billing is per session, not per pod, so the origin is too.
    [ -f "$SCR/pod_start_epoch" ] || date -u +%s > "$SCR/pod_start_epoch"
    echo "$POD_ID" > "$SCR/pod_id"
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
$SCP scripts/pod/e5_setup.sh "root@$HOST:/workspace/" >>"$LOG" 2>&1
$SSH "root@$HOST" 'mkdir -p /workspace/aad_holdout'
$SCP data/warmup/holdout_v1.jsonl "root@$HOST:/workspace/aad_holdout/" >>"$LOG" 2>&1

say "running setup"
$SSH "root@$HOST" "cd /workspace && SESSION_COMMIT=$SESSION_COMMIT \
  BUNDLE_NAME=$BUNDLE_NAME TEACHER_REVISION=$TEACHER_REVISION \
  HOLDOUT_SRC=/workspace/aad_holdout/holdout_v1.jsonl \
  E5_SEEDS=sa,sb bash /workspace/e5_setup.sh" \
  >>"$SCR/e5_setup.log" 2>&1
if [ $? -ne 0 ]; then
  say "FATAL: setup failed. $(cost)"; tail -25 "$SCR/e5_setup.log" | tee -a "$LOG"
  teardown; echo "LAUNCH_FAILED:setup" > "$STATE"; exit 1
fi
say "setup done — $(cost)"

say "starting FORMAL E5: validation gate -> R generation -> feasibility -> 4 arms -> eval"
# `nohup ... &` alone does NOT detach here: the remote wrapper shell stays alive
# holding the SSH channel open, so this call blocks until the driver exits and
# the launcher never reaches its polling loop. Measured on 2026-08-05 -- the run
# completed, but with no progress logging for five hours. `setsid` plus a
# closed stdin puts the driver in its own session so the channel closes at once.
# The gates measure their own elapsed time, but the pod has been billing since
# creation -- startup plus the ~53-min setup. Handing the driver a zero starting
# balance would understate spend by ~$1 at both gates.
SPENT_AT_DRIVER_START=$(echo "($(date -u +%s) - $(cat "$SCR/pod_start_epoch"))/3600*$MAX_PRICE" | bc -l)
# The gates are budgeted against the RunPod-side deadline, not the $9.12
# authorization. The deadline is the binding constraint: a plan that fits $9.12
# but not the backstop gets killed by RunPod mid-training, which is the exact
# outcome the gates exist to prevent. $9.12 remains the ceiling neither can cross.
GATE_CEILING=$(echo "$BACKSTOP_MINUTES/60*$MAX_PRICE" | bc -l)
say "driver budget: \$$(printf '%.2f' "$SPENT_AT_DRIVER_START") already billed, \
gate ceiling \$$(printf '%.2f' "$GATE_CEILING") (backstop-bound, under the \$9.12 authorization)"
$SSH "root@$HOST" "cd /workspace/aad && setsid nohup /opt/train/bin/python \
  scripts/pod/e5_driver.py --stage all \
  --spent-usd $(printf '%.3f' "$SPENT_AT_DRIVER_START") \
  --authorized-usd $(printf '%.2f' "$GATE_CEILING") \
  > /workspace/e5_run.log 2>&1 < /dev/null & \
  disown" >>"$LOG" 2>&1
say "driver running — $(cost)"

# --- 3. poll to completion -------------------------------------------------
DEADLINE_TS=$(( $(date -u +%s) + POLL_LIMIT_MIN * 60 )); LAST=""
while [ "$(date -u +%s)" -lt "$DEADLINE_TS" ]; do
  sleep 120
  STATUS_TXT=$($SSH "root@$HOST" 'cat /workspace/e5.status 2>/dev/null | tail -1' 2>/dev/null)
  if [ -n "$STATUS_TXT" ] && [ "$STATUS_TXT" != "$LAST" ]; then
    LAST="$STATUS_TXT"; say "  $STATUS_TXT — $(cost)"
  fi
  case "$STATUS_TXT" in
    *ALL_DONE*) say "driver reported ALL_DONE — $(cost)"; break ;;
    *ABORTED_AT_GATE*) say "STOPPED AT A GATE — tearing down — $(cost)"; break ;;
  esac
  # Liveness, not just markers. The driver now catches every exception, but it
  # can still be OOM-killed or die with the pod's python. Polling for a marker
  # that will never arrive kept a finished pod billing on 2026-08-07 -- so a dead
  # driver with no terminal marker ends the run here instead of at POLL_LIMIT_MIN.
  if ! $SSH "root@$HOST" 'pgrep -f "[e]5_driver.py" >/dev/null' 2>/dev/null; then
    say "DRIVER PROCESS GONE with no terminal marker — tearing down — $(cost)"
    say "  last status line: ${STATUS_TXT:-<none>}"
    break
  fi
done

# --- 4. fetch results, then weights, then delete ---------------------------
mkdir -p "$STORE"
say "bundling small artifacts on the pod"
$SSH "root@$HOST" 'cd /workspace/aad && tar czf /workspace/e5_side.tar.gz \
  artifacts/audit artifacts/stage3/e5_pilot_sa configs/stage3/e5 \
  $(ls -d artifacts/stage3/e5_*/manifest.json 2>/dev/null) \
  2>/dev/null; cp /workspace/e5_run.log /workspace/e5.status /workspace/ 2>/dev/null; \
  sha256sum /workspace/e5_side.tar.gz' >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e5_side.tar.gz" "$STORE/" >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e5_run.log" "$STORE/" >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e5.status" "$STORE/" >>"$LOG" 2>&1
if [ -s "$STORE/e4_side.tar.gz" ]; then
  say "results bundle retrieved ($(du -h "$STORE/e4_side.tar.gz" | cut -f1)) — $(cost)"
else
  say "WARNING: results bundle is empty or missing"
fi

say "hashing checkpoints on the pod"
$SSH "root@$HOST" 'cd /workspace/aad/artifacts/stage3 && \
  find e5_[cr]_s*/checkpoints/step_000738 -type f \( -name "*.safetensors" -o -name "*.json" \
    -o -name "*.jinja" \) | sort | xargs sha256sum' > "$SCR/e5_pod_hashes.txt" 2>>"$LOG"
cp "$SCR/e5_pod_hashes.txt" "$STORE/" 2>/dev/null
say "fetching checkpoints (time-boxed to ${CKPT_TRANSFER_LIMIT_MIN} min)"
for arm in e5_c_sa e5_c_sb e5_r_sa e5_r_sb; do
  timeout "${CKPT_TRANSFER_LIMIT_MIN}m" $SCP -r \
    "root@$HOST:/workspace/aad/artifacts/stage3/$arm/checkpoints/step_000738" \
    "$STORE/$arm" >>"$LOG" 2>&1 || say "WARNING: $arm weights not retrieved"
done
teardown
echo "DONE" > "$STATE"
say "session complete. artifacts under $STORE"
