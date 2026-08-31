#!/bin/bash
while kill -0 3020006 2>/dev/null; do sleep 60; done
echo "=== LAUNCHER EXITED $(date -u +%FT%TZ) ==="
tail -60 /home/ecs-user/phase_b_a4_scr/launcher.out
