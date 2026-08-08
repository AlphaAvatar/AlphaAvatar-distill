#!/usr/bin/env bash
# Dev-box orchestrator for the Experiment 6 pod: evaluation only, six arms.
#
#   SCR=… SESSION_COMMIT=… BUNDLE_NAME=… nohup bash scripts/pod/e6_launch.sh &
#
# Nothing trains, so the session is short and the budget layers are tighter than
# any previous run:
#   1. GPU securePrice is CHECKED before creating anything, and the pod's actual
#      costPerHr is re-checked after creation. Above the authorized rate, stop.
#   2. `--terminate-after` is an absolute UTC deadline from BACKSTOP_MINUTES,
#      enforced RunPod-side, so it fires even if every process here is dead.
#   3. This script polls for ALL_DONE and deletes the pod itself, because pods
#      idle-bill after the work finishes.
#   4. The driver re-prices before every arm from ACTUAL elapsed time and stops
#      rather than starting an arm it cannot pay for.
#
# 150 min x $0.99 = $2.48. Two abandoned cold-host draws add ~$0.40, so the
# worst path is ~$2.88 against the $3.47 authorized for E6.
set -uo pipefail

SCR=${SCR:?}
SESSION_COMMIT=${SESSION_COMMIT:?}
BUNDLE_NAME=${BUNDLE_NAME:?}
GPU_NAME=${GPU_NAME:-NVIDIA L40S}
MAX_PRICE=${MAX_PRICE:-0.99}
BACKSTOP_MINUTES=${BACKSTOP_MINUTES:-150}
STARTUP_LIMIT_MIN=${STARTUP_LIMIT_MIN:-15}
MAX_POD_ATTEMPTS=${MAX_POD_ATTEMPTS:-12}
MAX_HOST_DRAWS=${MAX_HOST_DRAWS:-3}
CREATE_RETRY_DELAY_S=${CREATE_RETRY_DELAY_S:-420}
POLL_LIMIT_MIN=${POLL_LIMIT_MIN:-145}
# The two sb high-rung checkpoints exist only here. 4.8 GB over scp; time-boxed
# so a slow link fails the run loudly instead of silently eating the budget.
CKPT_UPLOAD_LIMIT_MIN=${CKPT_UPLOAD_LIMIT_MIN:-25}
AUTHORIZED_USD=${AUTHORIZED_USD:-3.47}
STORE=${STORE:-/home/ecs-user/aad-artifacts/e6}
DEVBOX_CKPT=${DEVBOX_CKPT:-/home/ecs-user/AlphaAvatar-distill/artifacts/stage3/rescued}
LOG=$SCR/e6_launch.log
KEY=$(cat "$SCR/rp_key")
STATE=$SCR/e6.state

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
# `runpodctl pod create` provisions SECURE cloud, so the guard reads securePrice.
# communityPrice / lowestPrice.uninterruptablePrice is the COMMUNITY floor and
# has already caused two runs to be under-reported.
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
    --name "aadistill-e6-a$1" \
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

# --- 2. create, bounded, with cold-host redraw -----------------------------
REDRAWS=0
for draw in $(seq 1 "$MAX_HOST_DRAWS"); do
EP=""; POD_ID=${POD_ID:-}
for attempt in $(seq 1 "$MAX_POD_ATTEMPTS"); do
  if [ -z "$POD_ID" ]; then
    POD_ID=$(create_pod "$attempt")
    if [ -z "$POD_ID" ]; then
      say "attempt $attempt: create failed — raw output follows"
      head -3 "$SCR/create_raw_$attempt.txt" | tee -a "$LOG"
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
    # Written ONCE per SESSION, at the FIRST create: billing is per session, not
    # per pod, so a redraw must not hand the replacement a fresh meter.
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
$SSH "root@$HOST" 'mkdir -p /workspace/hf /workspace/ckpt /workspace/ckpt_local && chmod 700 /workspace/hf'
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
$SCP scripts/pod/e6_setup.sh "root@$HOST:/workspace/" >>"$LOG" 2>&1

# The two sb high-rung checkpoints are dev-box-only. Start them now, in the
# background, so 4.8 GB of transfer overlaps `uv sync` instead of following it.
# A cold-host redraw kills this and the next draw starts it again.
say "uploading the two dev-box-only checkpoints in the background"
( for arm in e1_r2960k_sb_pca e1_r5500k_sb_pca; do
    $SSH "root@$HOST" "mkdir -p /workspace/ckpt_local/$arm"
    timeout "${CKPT_UPLOAD_LIMIT_MIN}m" $SCP \
      "$DEVBOX_CKPT/$arm/model.safetensors" \
      "root@$HOST:/workspace/ckpt_local/$arm/" || exit 1
    echo "uploaded $arm"
  done; touch "$SCR/ckpt_upload_ok" ) >>"$SCR/e6_upload.log" 2>&1 &
UPLOAD_PID=$!

say "running setup"
$SSH "root@$HOST" "cd /workspace && SESSION_COMMIT=$SESSION_COMMIT \
  BUNDLE_NAME=$BUNDLE_NAME bash /workspace/e6_setup.sh" \
  >>"$SCR/e6_setup.log" 2>&1
SETUP_RC=$?
if [ "$SETUP_RC" -eq 90 ]; then
  REDRAWS=$((REDRAWS+1))
  kill "$UPLOAD_PID" 2>/dev/null; rm -f "$SCR/ckpt_upload_ok"
  COLD_LINE=$(grep -a "COLD HOST" "$SCR/e6_setup.log" | tail -1)
  say "COLD HOST on draw $draw — abandoning $POD_ID and redrawing. $(cost)"
  say "  $COLD_LINE"
  printf '%s draw=%d pod=%s %s billed_to_date=%s\n' \
    "$(date -u +%FT%TZ)" "$draw" "$POD_ID" "$COLD_LINE" "$(cost)" \
    >> "$SCR/redraws.log"
  runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1
  POD_ID=""; EP=""
  if [ "$draw" -lt "$MAX_HOST_DRAWS" ]; then continue; fi
  say "ABORT: ${MAX_HOST_DRAWS} consecutive cold hosts. Nothing is running."
  echo "LAUNCH_FAILED:all_hosts_cold" > "$STATE"; exit 1
fi
if [ "$SETUP_RC" -ne 0 ]; then
  say "FATAL: setup failed (rc $SETUP_RC). $(cost)"
  tail -25 "$SCR/e6_setup.log" | tee -a "$LOG"
  teardown; echo "LAUNCH_FAILED:setup" > "$STATE"; exit 1
fi
say "setup done on draw $draw after $REDRAWS redraw(s) — $(cost)"
break
done
[ -n "${EP:-}" ] || { say "ABORT: no usable host"; exit 1; }

# Setup stages the four relay arms itself but cannot verify the two uploaded
# ones until they land, so the upload is joined here and re-verified on the pod.
say "waiting for the dev-box checkpoint upload"
wait "$UPLOAD_PID"
if [ ! -f "$SCR/ckpt_upload_ok" ]; then
  say "FATAL: dev-box checkpoint upload failed or timed out. $(cost)"
  tail -10 "$SCR/e6_upload.log" | tee -a "$LOG"
  teardown; echo "LAUNCH_FAILED:ckpt_upload" > "$STATE"; exit 1
fi
say "dev-box checkpoints uploaded — $(cost)"
$SSH "root@$HOST" "cd /workspace/aad && /opt/train/bin/python \
  scripts/pod/e6_stage_checkpoints.py \
    --registration logs/e6_registration.json --relay-dest /workspace/ckpt \
    --devbox-src /workspace/ckpt_local \
    --init artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
    --out artifacts/audit/e6_checkpoint_manifest.json" >>"$LOG" 2>&1 \
  || { say "FATAL: checkpoint staging or hash verification failed. $(cost)"
       teardown; echo "LAUNCH_FAILED:ckpt_verify" > "$STATE"; exit 1; }
say "all six checkpoints staged and hash-verified — $(cost)"

say "starting E6 evaluation: 6 arms x (free, oracle, forced) on the frozen battery"
# `nohup … &` alone does NOT detach over ssh: the remote wrapper shell holds the
# channel open, so the call blocks for the whole run and the launcher never
# reaches its polling loop. `setsid` with closed stdin puts the driver in its own
# session so the channel closes at once.
SPENT_AT_DRIVER_START=$(echo "($(date -u +%s) - $(cat "$SCR/pod_start_epoch"))/3600*$MAX_PRICE" | bc -l)
GATE_CEILING=$(echo "$BACKSTOP_MINUTES/60*$MAX_PRICE" | bc -l)
say "driver budget: \$$(printf '%.2f' "$SPENT_AT_DRIVER_START") already billed, \
gate ceiling \$$(printf '%.2f' "$GATE_CEILING") (backstop-bound, under the \$$AUTHORIZED_USD authorization)"
$SSH "root@$HOST" "cd /workspace/aad && setsid nohup /opt/train/bin/python \
  scripts/pod/e6_driver.py --stage all \
  --spent-usd $(printf '%.3f' "$SPENT_AT_DRIVER_START") \
  --authorized-usd $(printf '%.2f' "$GATE_CEILING") \
  > /workspace/e6_run.log 2>&1 < /dev/null & \
  disown" >>"$LOG" 2>&1
say "driver running — $(cost)"

# --- 3. poll to completion -------------------------------------------------
DEADLINE_TS=$(( $(date -u +%s) + POLL_LIMIT_MIN * 60 )); LAST=""
while [ "$(date -u +%s)" -lt "$DEADLINE_TS" ]; do
  sleep 120
  STATUS_TXT=$($SSH "root@$HOST" 'cat /workspace/e6.status 2>/dev/null | tail -1' 2>/dev/null)
  if [ -n "$STATUS_TXT" ] && [ "$STATUS_TXT" != "$LAST" ]; then
    LAST="$STATUS_TXT"; say "  $STATUS_TXT — $(cost)"
  fi
  case "$STATUS_TXT" in
    *ALL_DONE*) say "driver reported ALL_DONE — $(cost)"; break ;;
    *ABORTED_AT_GATE*) say "STOPPED AT A GATE — tearing down — $(cost)"; break ;;
  esac
  # Liveness, not just markers: a driver that is OOM-killed writes no terminal
  # marker, and polling for one that will never arrive keeps a finished pod
  # billing.
  if ! $SSH "root@$HOST" 'pgrep -f "[e]6_driver.py" >/dev/null' 2>/dev/null; then
    say "DRIVER PROCESS GONE with no terminal marker — tearing down — $(cost)"
    say "  last status line: ${STATUS_TXT:-<none>}"
    break
  fi
done

# --- 4. fetch every artifact, verify, then delete --------------------------
# E6 writes no weights, so the results ARE the artifacts and they are small.
mkdir -p "$STORE"
say "bundling artifacts on the pod"
$SSH "root@$HOST" 'cd /workspace/aad && tar czf /workspace/e6_artifacts.tar.gz \
  artifacts/audit/three_mode artifacts/audit/e6_checkpoint_manifest.json \
  artifacts/audit/e6_notrain_proof.json 2>/dev/null; \
  cp /workspace/e6_run.log /workspace/e6.status /workspace/ 2>/dev/null; \
  sha256sum /workspace/e6_artifacts.tar.gz' | tee -a "$LOG" > "$SCR/e6_pod_hashes.txt"
$SCP "root@$HOST:/workspace/e6_artifacts.tar.gz" "$STORE/" >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e6_run.log" "$STORE/" >>"$LOG" 2>&1
$SCP "root@$HOST:/workspace/e6.status" "$STORE/" >>"$LOG" 2>&1
cp "$SCR/e6_pod_hashes.txt" "$STORE/" 2>/dev/null

# Verify the retrieved bundle against the pod-side digest BEFORE deleting the
# pod, so a corrupted transfer is recoverable rather than discovered afterwards.
POD_SHA=$(awk '{print $1}' "$SCR/e6_pod_hashes.txt" | tail -1)
LOCAL_SHA=$(sha256sum "$STORE/e6_artifacts.tar.gz" 2>/dev/null | awk '{print $1}')
if [ -n "$POD_SHA" ] && [ "$POD_SHA" = "$LOCAL_SHA" ]; then
  say "artifacts verified: $LOCAL_SHA ($(du -h "$STORE/e6_artifacts.tar.gz" | cut -f1))"
else
  say "WARNING: artifact digest mismatch (pod ${POD_SHA:-none} local ${LOCAL_SHA:-none}) — retrying once"
  $SCP "root@$HOST:/workspace/e6_artifacts.tar.gz" "$STORE/" >>"$LOG" 2>&1
  LOCAL_SHA=$(sha256sum "$STORE/e6_artifacts.tar.gz" 2>/dev/null | awk '{print $1}')
  [ "$POD_SHA" = "$LOCAL_SHA" ] && say "artifacts verified on retry" \
    || say "ERROR: artifacts NOT verified; keeping the pod is not authorized, reporting instead"
fi

teardown
echo "DONE" > "$STATE"
say "session complete. artifacts under $STORE"
