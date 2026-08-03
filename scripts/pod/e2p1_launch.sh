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
POD_ID=${POD_ID:?}
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

say "waiting for pod $POD_ID"
EP=""
for i in $(seq 1 180); do            # up to 60 min; a cold image pull is slow
  EP=$(endpoint)
  [ -n "$EP" ] && break
  [ $((i % 6)) -eq 0 ] && say "  still starting ($((i*20))s) — $(cost)"
  sleep 20
done
if [ -z "$EP" ]; then
  say "FATAL: pod never exposed TCP 22. $(cost). Deleting to stop the meter."
  runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
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
