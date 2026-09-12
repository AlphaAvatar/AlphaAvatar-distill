#!/bin/bash
while kill -0 2897860 2>/dev/null; do sleep 60; done
echo "=== LAUNCHER EXITED $(date -u +%FT%TZ) ==="
tail -50 /home/ecs-user/phase_b_a3_scr/launcher.out
