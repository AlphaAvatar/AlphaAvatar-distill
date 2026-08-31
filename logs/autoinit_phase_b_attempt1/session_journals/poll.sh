#!/bin/bash
# Phase-B session poll. Appends one line per check; stops when the launcher exits.
OUT=/home/ecs-user/phase_b_scr/poll.jsonl
while true; do
  ALIVE=$(ps -eo args --no-headers | grep -c "[a]utoinit_phase_b_launch")
  POD=$(runpodctl pod list 2>/dev/null | grep -c "4dbqycjrivhq17")
  LAST=$(tail -1 /home/ecs-user/phase_b_scr/launcher.out 2>/dev/null | tr -d '"' | cut -c1-160)
  printf '{"utc":"%s","launcher_alive":%s,"pod_listed":%s,"last":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$ALIVE" "$POD" "$LAST" >> "$OUT"
  [ "$ALIVE" -eq 0 ] && { printf '{"utc":"%s","event":"launcher_exited"}\n' "$(date -u +%FT%TZ)" >> "$OUT"; break; }
  sleep 300
done
