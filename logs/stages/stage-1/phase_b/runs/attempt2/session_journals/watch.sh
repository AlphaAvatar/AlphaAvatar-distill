#!/bin/bash
# Waits for the Phase-B launcher to exit, then reports. Never touches the run.
while kill -0 2859459 2>/dev/null; do sleep 60; done
echo "=== LAUNCHER EXITED $(date -u +%FT%TZ) ==="
tail -40 /home/ecs-user/phase_b_retry_scr/launcher.out
echo "=== pod_id ==="; cat /home/ecs-user/phase_b_retry_scr/pod_id 2>/dev/null
