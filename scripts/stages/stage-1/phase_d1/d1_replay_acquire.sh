#!/bin/bash
# Bounded reacquisition for the D1 replay. ONE create call per invocation, by
# the package's own rule -- a corrected attempt is an explicit new subrun, never
# an invisible provider retry -- so the loop re-INVOKES the launcher rather than
# raising --create-attempts.
#
# THE GPU IS NOT NEGOTIABLE. Success for this session IS digest equality, so the
# device is part of the experiment: A40 at $0.49 and several other 48 GB cards
# are in stock and cheaper, and taking one would risk different kernel selection
# on a different architecture, surface as a FixedPathDigestMismatch -- a
# scientific stop condition -- and manufacture a finding out of an
# infrastructure substitution.
#
# IT CLASSIFIES THE CURRENT ATTEMPT, NOT THE FILE. The first version grepped
# `tail -40` of the launcher's APPEND-MODE log, so when attempt 2 created a pod
# and died in setup, the grep matched attempt 1's "could not create a pod" line
# and reported capacity. It then slept and would have retried an identical
# deterministic setup failure up to 24 times at $0.15 each -- about $3.60,
# through the $3.50 cap, on one unchanged defect. Exactly the rule this script
# exists to enforce, broken by its own evidence-reading.
#
# Each attempt therefore writes to its OWN log, and the decision is made from
# that file alone. Retries only a provider-capacity abort; anything else stops
# the loop, because a deterministic failure repeated unchanged is not a repair.
set -u
R=d1_replay_001
SCR=/home/ecs-user/aad-scratch
LOG=$SCR/${R}.launcher.log
ACQ=$SCR/${R}.acquire.log
MAX_ATTEMPTS=24
SLEEP_SECONDS=150

say() { echo "$(date -u +%FT%TZ) $*" >> "$ACQ"; }

say "start: up to ${MAX_ATTEMPTS} attempts, ${SLEEP_SECONDS}s apart, L40S only"
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  ATTEMPT_LOG=$SCR/${R}.attempt${i}.log

  LIVE=$(cd /home/ecs-user/AlphaAvatar-distill && PYTHONPATH=src .venv/bin/python - <<'PY' 2>/dev/null
import os, sys
sys.path.insert(0, "src")
from aadistill.infrastructure.session_runner import RunPodProvider, read_api_key
try:
    p = RunPodProvider(read_api_key(os.path.expanduser("~/.runpod/config.toml")))
    o = p.observe("query { myself { pods { id } } }")
    pods = ((o.data or {}).get("myself") or {}).get("pods") or []
    print("UNKNOWN" if not o.ok else len(pods))
except Exception:
    print("UNKNOWN")
PY
)
  if [ "$LIVE" != "0" ]; then
    say "attempt ${i}: provider reports pods=${LIVE}; refusing to create another. STOP"
    break
  fi

  say "attempt ${i}/${MAX_ATTEMPTS}: invoking the launcher -> ${ATTEMPT_LOG}"
  : > "$ATTEMPT_LOG"
  AAD_ATTEMPT_LOG="$ATTEMPT_LOG" "$SCR/${R}_run.sh"
  # The per-attempt log is the ONLY evidence this decision reads.
  cat "$ATTEMPT_LOG" >> "$LOG"

  if grep -q "LAUNCHER_EXIT=0" "$ATTEMPT_LOG"; then
    say "attempt ${i}: launcher exited 0. DONE"
    break
  fi
  if grep -q "ABORT: could not create a pod" "$ATTEMPT_LOG"; then
    say "attempt ${i}: provider capacity, nothing created, \$0. Backing off ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
    continue
  fi
  say "attempt ${i}: FAILED for a reason that is NOT provider capacity. STOP for diagnosis"
  grep -E "ABORT|SETUP_RC|MARKER|pod deleted|LAUNCHER_EXIT" "$ATTEMPT_LOG" | tail -8 >> "$ACQ"
  break
done
say "acquisition loop finished"
echo "ACQUIRE_DONE" >> "$ACQ"
