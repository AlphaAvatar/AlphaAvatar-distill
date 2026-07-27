#!/usr/bin/env bash
# Dev-box driver for a teacher session: wait on markers, fetch each artifact as
# it lands, verify it, then delete the pod.
#
#   POD_ID=<id> HOST=<ip> PORT=<port> nohup bash scripts/pod/teacher_session_fetch.sh &
#
# Runs under nohup on the dev box, not inside an agent session: a paid pod must
# not depend on a conversation staying open. Everything it does is logged to
# $LOG and is safe to re-run — fetches are idempotent and the pod is deleted
# only after both artifacts are on disk and parse.
set -u

LOG=${LOG:-artifacts/teacher_session_fetch.log}
POLL_SECONDS=${POLL_SECONDS:-300}
MAX_HOURS=${MAX_HOURS:-7}
# Which marker ends the session. `both` waits for the pilot too; `teacher` stops
# as soon as the scorecard is on disk — used when the remaining work is cheaper
# on a different engine, so the meter should not keep running.
STOP_AFTER=${STOP_AFTER:-both}
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes"

mkdir -p "$(dirname "$LOG")" artifacts/teacher artifacts/stage2_v2
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

markers() {
  timeout 60 ssh $SSHOPT -p "$PORT" "root@$HOST" \
    "grep -ho 'MARKER:[A-Z_]*' /workspace/session.log 2>/dev/null | sort -u" 2>/dev/null
}

fetch() { # fetch <remote-glob> <local-dir>
  timeout 600 scp $SSHOPT -P "$PORT" -q -r "root@$HOST:$1" "$2" 2>>"$LOG"
}

say "watching pod $POD_ID at $HOST:$PORT (poll ${POLL_SECONDS}s, cap ${MAX_HOURS}h)"
teacher_done=0
pilot_done=0
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))

while [ "$(date +%s)" -lt "$deadline" ]; do
  seen=$(markers)
  say "markers: $(echo "$seen" | tr '\n' ' ')"

  if echo "$seen" | grep -q SESSION_FAILED; then
    say "SESSION_FAILED — leaving the pod up for inspection, not deleting"
    exit 1
  fi

  if [ "$teacher_done" = 0 ] && echo "$seen" | grep -q TEACHER_SCORED; then
    fetch "/workspace/aad/artifacts/teacher/*" artifacts/teacher/
    if python3 -c "import json;json.load(open('artifacts/teacher/eval_behavior_v0.json'))" 2>>"$LOG"; then
      teacher_done=1
      say "teacher scorecard fetched and parses"
    else
      say "teacher scorecard did not parse — will retry next poll"
    fi
  fi

  if [ "$pilot_done" = 0 ] && echo "$seen" | grep -q PILOT_DONE; then
    fetch "/workspace/aad/artifacts/stage2_v2/pilot" artifacts/stage2_v2/
    if python3 -c "import json;json.load(open('artifacts/stage2_v2/pilot/manifest.json'))" 2>>"$LOG"; then
      pilot_done=1
      say "pilot artifacts fetched and manifest parses"
    else
      say "pilot manifest did not parse — will retry next poll"
    fi
  fi

  if [ "$STOP_AFTER" = teacher ] && [ "$teacher_done" = 1 ]; then
    say "teacher scorecard done and STOP_AFTER=teacher; stopping the session"
    timeout 60 ssh $SSHOPT -p "$PORT" "root@$HOST" \
      "pkill -f teacher_session.sh; pkill -f generate_teacher_answers" 2>>"$LOG"
    fetch "/workspace/session.log" artifacts/
    say "deleting pod $POD_ID"
    runpodctl remove pod "$POD_ID" >> "$LOG" 2>&1 || runpodctl pod remove "$POD_ID" >> "$LOG" 2>&1
    say "DONE (teacher only)"
    exit 0
  fi

  if [ "$teacher_done" = 1 ] && [ "$pilot_done" = 1 ]; then
    say "both artifacts on disk; deleting pod $POD_ID"
    runpodctl remove pod "$POD_ID" >> "$LOG" 2>&1 || runpodctl pod remove "$POD_ID" >> "$LOG" 2>&1
    say "DONE"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done

# Out of time: keep whatever landed, and still stop the meter. The pod's own
# --terminate-after is the backstop behind this one.
say "deadline reached (teacher=$teacher_done pilot=$pilot_done); fetching partials"
fetch "/workspace/aad/artifacts/teacher/*" artifacts/teacher/
fetch "/workspace/aad/artifacts/stage2_v2/pilot" artifacts/stage2_v2/
fetch "/workspace/session.log" artifacts/
say "deleting pod $POD_ID"
runpodctl remove pod "$POD_ID" >> "$LOG" 2>&1 || runpodctl pod remove "$POD_ID" >> "$LOG" 2>&1
say "DONE (partial)"
