#!/usr/bin/env bash
# Dev-box driver for Experiment 2 Phase 1. Runs under nohup/tmux so the session
# survives the orchestrating process, per the standing rule that a long paid run
# is never held open by an interactive poller.
#
# It waits for the pod, transfers the session, runs setup, then runs the pod-side
# driver — which stops itself at the throughput gate if the gate fails. Teardown
# is tied to the completion marker, never to --terminate-after.
#
#   POD_ID=… SESSION_COMMIT=… BUNDLE_NAME=… bash scripts/pod/e2p1_launch.sh
set -uo pipefail

SCR=${SCR:?}
POD_ID=${POD_ID:-}
SESSION_COMMIT=${SESSION_COMMIT:?}
BUNDLE_NAME=${BUNDLE_NAME:?}
BATTERY_PREFIX=${BATTERY_PREFIX:-e2p1_20260803/battery_v2}
PACK_PREFIX=${PACK_PREFIX:-e2p1_20260803/rung_0860k_clean_median}
LOG=$SCR/e2p1_launch.log
KEY=$(cat "$SCR/rp_key")

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
cost() {
  local mins=$(( ($(date -u +%s) - $(cat "$SCR/pod_start_epoch")) / 60 ))
  printf '%d min elapsed, $%.2f billed\n' "$mins" "$(echo "$mins/60*0.99" | bc -l)"
}

endpoint() {
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"query { pod(input:{podId:\\\"$POD_ID\\\"}) { runtime { ports { ip publicPort privatePort type } } } }\"}" |
  python3 -c "import sys,json
d=json.load(sys.stdin); rt=(d.get('data') or {}).get('pod',{}).get('runtime')
print('' if not rt else next((f\"{p['ip']} {p['publicPort']}\" for p in (rt.get('ports') or []) if p.get('privatePort')==22 and p.get('type')=='tcp'),''))" 2>/dev/null
}

# A pod that has not exposed TCP 22 within STARTUP_LIMIT_MIN is not starting
# slowly, it is not starting. Two pods burned 22 and 18 minutes at
# `runtime: null` before this bound existed; at $0.99/h that is $0.66 of pure
# scheduling failure with nothing to show. The bound is deliberately tight —
# recreating is free and lands on a different machine, whereas waiting is not —
# and the launcher now recreates rather than asking a human to.
STARTUP_LIMIT_MIN=${STARTUP_LIMIT_MIN:-15}
MAX_POD_ATTEMPTS=${MAX_POD_ATTEMPTS:-4}

create_pod() {
  local deadline
  deadline=$(date -u -d '+8 hours' +%Y-%m-%dT%H:%M:%SZ)
  # A cuda-13-NATIVE image, paired with --min-cuda-version 13.0. The previous
  # spec asked cuda-13 hosts to run a cu128 image, which none of them had
  # cached: five pods in a row sat at `runtime: null` through what was almost
  # certainly a ~20 GB pull, costing $1.30 and starting nothing. The disk is
  # 120 GB rather than 150 to widen placement — the session needs ~100 GB
  # (two arms x 9 checkpoints at 4.3 GB, plus the 4B teacher and two venvs).
  runpodctl pod create \
    --image "${POD_IMAGE:-runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404}" \
    --gpu-id "NVIDIA L40S" --gpu-count 1 \
    --container-disk-in-gb 120 --volume-in-gb 0 \
    --min-cuda-version 13.0 \
    --ports "22/tcp,8888/http" \
    --name "aadistill-e2p1-a$1" \
    --terminate-after "$deadline" 2>&1 |
  python3 -c "import sys,re; m=re.search(r'\"id\":\s*\"([^\"]+)\"', sys.stdin.read()); print(m.group(1) if m else '')"
}

EP=""
for attempt in $(seq 1 "$MAX_POD_ATTEMPTS"); do
  if [ -z "${POD_ID:-}" ] || [ "$attempt" -gt 1 ]; then
    POD_ID=$(create_pod "$attempt")
    [ -z "$POD_ID" ] && { say "attempt $attempt: pod create failed"; continue; }
    date -u +%s > "$SCR/pod_start_epoch"
    echo "$POD_ID" > "$SCR/pod_id"
    say "attempt $attempt: created pod $POD_ID"
  fi
  say "waiting for $POD_ID (hard limit ${STARTUP_LIMIT_MIN} min)"
  DEADLINE_TS=$(( $(date -u +%s) + STARTUP_LIMIT_MIN * 60 ))
  i=0
  while [ "$(date -u +%s)" -lt "$DEADLINE_TS" ]; do
    EP=$(endpoint)
    [ -n "$EP" ] && break
    i=$((i + 1))
    [ $((i % 6)) -eq 0 ] && say "  still starting ($((i*20))s) — $(cost)"
    sleep 20
  done
  [ -n "$EP" ] && break
  say "attempt $attempt: no TCP 22 within ${STARTUP_LIMIT_MIN} min — $(cost). Deleting."
  runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
  echo "$STARTUP_LIMIT_MIN" >> "$SCR/wasted_startup_min"
  POD_ID=""
done
if [ -z "$EP" ]; then
  say "ABORT: ${MAX_POD_ATTEMPTS} pods failed to start. No pod is running."
  echo "LAUNCH_FAILED:no_endpoint" > "$SCR/e2p1.state"
  exit 1
fi
HOST=$(echo "$EP" | cut -d' ' -f1); PORT=$(echo "$EP" | cut -d' ' -f2)
say "pod ready at $HOST:$PORT — $(cost)"
echo "$HOST $PORT" > "$SCR/ssh_endpoint"

SSH="ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30"
for i in $(seq 1 30); do
  $SSH "root@$HOST" true 2>/dev/null && break
  say "  ssh not accepting yet ($((i*10))s)"; sleep 10
done

say "transferring session"
$SSH "root@$HOST" 'mkdir -p /workspace/hf && chmod 700 /workspace/hf'
hf auth token > "$SCR/hf_token" && chmod 600 "$SCR/hf_token"
scp -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$SCR/hf_token" "root@$HOST:/workspace/hf/token" >>"$LOG" 2>&1
$SSH "root@$HOST" 'chmod 600 /workspace/hf/token'
scp -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    scripts/pod/e2p1_setup.sh "root@$HOST:/workspace/" >>"$LOG" 2>&1
rm -f "$SCR/hf_token"

say "running setup (bundle @ $SESSION_COMMIT)"
$SSH "root@$HOST" "cd /workspace && SESSION_COMMIT=$SESSION_COMMIT \
  BUNDLE_NAME=$BUNDLE_NAME BATTERY_PREFIX=$BATTERY_PREFIX PACK_PREFIX=$PACK_PREFIX \
  bash /workspace/e2p1_setup.sh" >>"$SCR/e2p1_setup.log" 2>&1
SETUP_RC=$?
if [ $SETUP_RC -ne 0 ]; then
  say "FATAL: setup failed rc=$SETUP_RC. $(cost). See $SCR/e2p1_setup.log"
  tail -30 "$SCR/e2p1_setup.log" | tee -a "$LOG"
  say "deleting the pod to stop the meter"
  runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
  echo "LAUNCH_FAILED:setup" > "$SCR/e2p1.state"
  exit 1
fi
say "setup done — $(cost)"

say "running the FIRST D0 endpoint and the throughput gate"
$SSH "root@$HOST" "cd /workspace/aad && /opt/train/bin/python scripts/pod/e2p1_driver.py --stage d0_sa" \
  >>"$SCR/e2p1_run.log" 2>&1
D0_RC=$?
$SSH "root@$HOST" "cd /workspace/aad && /opt/train/bin/python scripts/pod/e2p1_driver.py --stage gate" \
  >>"$SCR/e2p1_run.log" 2>&1
GATE_RC=$?
say "D0 rc=$D0_RC gate rc=$GATE_RC — $(cost)"

# Always pull telemetry, pass or fail: a failed gate must still leave evidence.
say "fetching partial output and telemetry"
mkdir -p artifacts/eval/e2p1
scp -r -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "root@$HOST:/workspace/aad/artifacts/eval/e2p1" artifacts/eval/ >>"$LOG" 2>&1
scp -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "root@$HOST:/workspace/e2p1.status" "$SCR/e2p1.status" >>"$LOG" 2>&1

if [ $GATE_RC -ne 0 ]; then
  say "GATE FAILED. Not running the second D0 endpoint or any D1 training."
  say "$(cost)"
  runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
  say "pod deleted"
  echo "GATE_FAIL" > "$SCR/e2p1.state"
  exit 2
fi

say "gate passed — continuing Phase 1"
$SSH "root@$HOST" "cd /workspace/aad && nohup /opt/train/bin/python scripts/pod/e2p1_driver.py --stage rest \
  > /workspace/e2p1_rest.log 2>&1 &" >>"$LOG" 2>&1
echo "RUNNING" > "$SCR/e2p1.state"
say "phase 1 continuing on the pod; poller takes over"
