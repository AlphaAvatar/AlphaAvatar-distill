#!/bin/bash
while kill -0 3086692 2>/dev/null; do sleep 60; done
echo "=== LAUNCHER EXITED $(date -u +%FT%TZ) ==="
tail -60 /home/ecs-user/phase_b_a5_scr/launcher.out
