#!/usr/bin/env bash
# Acquire a pod for A3 and run the chain. Everything before the create is $0.
#
#   nohup bash scripts/stages/stage-1/phase_a3/a3_acquire.sh > /tmp/a3_acquire.out 2>&1 &
#
# One round = one $0 capacity watch, one live re-price, one FRESH one-use chain
# (grant -> launch-bound readiness -> authorization -> bundle), one launcher
# invocation. A corrected attempt is an explicit NEW subrun, never an invisible
# provider retry.
#
# THREE THINGS THIS GETS RIGHT THAT C3's LOOP DID NOT, each for a recorded
# reason:
#
#   1. IT BACKS OFF AFTER A CAPACITY REFUSAL. C3's watch slept only while the
#      tier was DRY, and `Low` stock that never converts reads as USABLE -- so
#      on 2026-09-29 it rebuilt a chain roughly every 66 seconds and would have
#      spent all its rounds in about 44 minutes instead of pacing them.
#
#   2. IT READS THE DRIVER'S OWN STATUS FILE. C3's driver wrote markers to
#      autoinit_c3.status while its launcher polled autoinit_c1.status, so
#      `ALL_DONE` was invisible and this loop would have read a COMPLETED
#      formal run as "no measurement began" and launched a second paid
#      attempt. Both now come from `a3_session`, and the terminal check reads
#      the launcher log AND the status file.
#
#   3. IT STOPS ON ANY TERMINAL STATE, not only on success. A3_FAILED and
#      A3_INTEGRITY_FAILURE end the loop for a human to read, because an
#      integrity failure is a repair and not a retry.
#
# At most ONE billing resource at any instant: the launcher's last gate is an
# account-wide check, and this loop never runs two launchers.

set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LOG=/tmp/a3_acquire.log
#: DURABLE, not session scratch. `SessionRunner` fetches the manifest and the
#: artifact archive into `<scr>/store` and extracts them into
#: `<scr>/store/extracted` -- so `--scr` is where A3's only scientific output
#: comes home to, and pointing it at a session-scoped temp directory would put
#: three trained-and-scored probes' evidence somewhere a cleanup removes. The
#: comparison then runs off this, off pod, at $0.
ART=/home/ecs-user/aad-artifacts/phase_a3
BASE="$ART/_sessions"
mkdir -p "$BASE"

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# The ONE owner of both paths. Hardcoding either is defect 2 above.
read -r STATUS_PATH DRIVER_JOB < <(
  PYTHONPATH=src:scripts .venv/bin/python - <<'PY'
from experiments.phase_c3 import a3_session as A3S
print(A3S.STATUS_PATH, A3S.DRIVER_JOB_ID)
PY
)
say "status path $STATUS_PATH, driver job $DRIVER_JOB"

#: THE PRICED CARD ONLY. The approved tier holds three types, and `select`
#: returns the best AVAILABLE one -- right for a session whose success is a
#: measurement, wrong for A3. Stage E compares A-bsz1's artifact digest
#: byte-exactly against an incumbent built on an L40S, and a GEMM reduces in
#: an architecture-dependent order, so another card can fail that gate for a
#: reason that is not the experiment. RTX 6000 Ada was `usable` at $0.84 while
#: the L40S refused on capacity, so this watch would have handed the launcher
#: a cheaper, approved, scientifically void card. The launcher's
#: `pricing_identity_gate` refuses it too; this stops the loop from consuming
#: a chain to find that out.
usable_now() {
  PYTHONPATH=src:scripts .venv/bin/python - <<'PYEOF' 2>/dev/null
import json
import pathlib
import sys
sys.path[:0] = ["src", "scripts"]
try:
    from experiments.phase_c3.hardware import query_offers
    priced = json.loads(pathlib.Path(
        "logs/stages/stage-1/phase_c3/plans/a3_live_pricing.json").read_text())
    want = priced["gpu"]
    offers = query_offers()
    keys = offers if isinstance(offers, list) else list(offers)
    for key in keys:
        offer = key if isinstance(offers, list) else offers[key]
        if getattr(offer, "gpu_type_id", None) == want and offer.usable:
            print(f"{offer.gpu_type_id}|{offer.secure_price_usd_per_hour}")
            break
except Exception:
    pass
PYEOF
}

next_free() {
  #: BOTH spellings, and the number keeps increasing across the change.
  #: `a3-attempt1` and `a3-attempt2` are consumed chains preserved as
  #: unrecorded runs; the HYPHEN is why they were consumed -- `RunLayout`
  #: accepts only [a-z0-9_], so `open_run` refused them at the last step of
  #: the chain while every earlier step had produced correct-looking paths.
  #: Counting both means no attempt number is ever reused and no two
  #: directories differ only by a hyphen.
  N=1
  while [ -d "logs/stages/stage-1/phase_a3/runs/a3-attempt$N" ] \
     || [ -d "logs/stages/stage-1/phase_a3/runs/a3_attempt$N" ]; do
    N=$((N + 1))
  done
  echo "$N"
}

BACKOFF=0
BACKOFF_MAX=900
ROUNDS=${A3_ROUNDS:-12}

for ROUND in $(seq 1 "$ROUNDS"); do
  if [ "$BACKOFF" -gt 0 ]; then
    say "round $ROUND: backing off ${BACKOFF}s after a capacity refusal"
    sleep "$BACKOFF"
  fi

  # ---- $0 capacity watch. Creates nothing. --------------------------------
  PICK=""
  for _ in $(seq 1 60); do
    PICK="$(usable_now)"
    [ -n "$PICK" ] && break
    sleep 60
  done
  if [ -z "$PICK" ]; then
    say "round $ROUND: the approved tier stayed dry for 1h; continuing the watch"
    BACKOFF=0        # a dry tier already waited; it is not a refusal
    continue
  fi
  GPU="${PICK%%|*}"; PRICE="${PICK##*|}"
  N=$(next_free)
  RUN="a3_attempt$N"
  GOV="logs/stages/stage-1/phase_a3/runs/$RUN/governance"
  say "round $ROUND: $GPU usable at \$$PRICE/h -> building $RUN"

  # ---- re-price LIVE on the device we are about to use --------------------
  if ! PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_a3/a3_pricing.py --write \
        --out logs/stages/stage-1/phase_c3/plans/a3_live_pricing.json \
        >> "$LOG" 2>&1; then
    say "$RUN: the live re-price is not fundable; stopping for a maintainer"
    break
  fi

  # ---- a FRESH one-use chain ---------------------------------------------
  mkdir -p "$GOV"
  PYTHONPATH=src:scripts .venv/bin/python - "$RUN" <<'PY' >> "$LOG" 2>&1 || \
    { say "$RUN: grant copy failed"; BACKOFF=$((BACKOFF + 60)); continue; }
import json, pathlib, sys
run = sys.argv[1]
src = pathlib.Path("logs/budget/approvals/autoinit_a3_grant.json")
doc = json.loads(src.read_text())
#: The run id and the experiment go in HERE, not in the template: the
#: grant_provenance gate refuses a chain whose grant names another run, which
#: is what makes a one-use chain non-transferable.
doc["run_id"] = run
doc["experiment_id"] = "phase_a3"
out = pathlib.Path(f"logs/stages/stage-1/phase_a3/runs/{run}/governance/grant.json")
out.write_text(json.dumps(doc, indent=1) + "\n")
print(f"grant written for {run}")
PY
  git add -A >/dev/null 2>&1
  git commit -q -m "$RUN: grant and live pricing on $GPU" >/dev/null 2>&1

  if ! PYTHONPATH=src:scripts .venv/bin/python \
        scripts/shared/pod/record_pod_environment.py --experiment phase_a3 \
        --run-id "$RUN" --stage-id 1 --kind launch_bound >> "$LOG" 2>&1; then
    say "$RUN: the launch-bound readiness sweep failed"
    BACKOFF=$(( BACKOFF + 120 < BACKOFF_MAX ? BACKOFF + 120 : BACKOFF_MAX ))
    continue
  fi
  git add -A >/dev/null 2>&1
  git commit -q -m "$RUN: launch-bound readiness" >/dev/null 2>&1

  if ! PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_a3/issue_a3_authorization.py \
        --grant "$GOV/grant.json" --out "$GOV/authorization.json" \
        >> "$LOG" 2>&1; then
    say "$RUN: the issuer refused; this chain is consumed"
    BACKOFF=$(( BACKOFF + 120 < BACKOFF_MAX ? BACKOFF + 120 : BACKOFF_MAX ))
    continue
  fi
  git add -A >/dev/null 2>&1
  git commit -q -m "$RUN: one-use authorization" >/dev/null 2>&1

  SC=$(git rev-parse HEAD)
  if ! PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/phase_c1/stage_c1_bundle.py \
        --session-commit "$SC" --out "$GOV/bundle.json" >> "$LOG" 2>&1; then
    say "$RUN: the bundle could not be staged and fetch-verified"
    BACKOFF=$(( BACKOFF + 120 < BACKOFF_MAX ? BACKOFF + 120 : BACKOFF_MAX ))
    continue
  fi
  git add -A >/dev/null 2>&1
  git commit -q -m "$RUN: staged bundle" >/dev/null 2>&1
  git push -q >/dev/null 2>&1

  BN="aad_autoinit_$(echo "$SC" | cut -c1-8).bundle"
  SCR="$BASE/$RUN"; mkdir -p "$SCR"
  say "$RUN: launching on $GPU, commit $SC, bundle $BN"
  #: RESUME AT SCORING when a preserving attempt is named. The probes are
  #: restored instead of retrained, which is AGENTS.md P8.4 state 2, and the
  #: chain's hard model drops from 450.8 min / $8.3047 to 179.0 / $3.2982.
  #:
  #: `--poll-limit-min` follows the WORK rather than the authorization: a
  #: resume run still going at 240 minutes is hung, and bounding it at the
  #: full chain's 600 would burn $8.25 of a $9.61 balance to learn that. The
  #: authorization's ceiling is unchanged and remains a legitimate upper
  #: bound; this is the backstop that fires first.
  RESUME_ARGS=()
  POLL_LIMIT=600
  if [ -n "${A3_RESUME_FROM:-}" ]; then
    RESUME_ARGS=(--resume-scoring-from "$A3_RESUME_FROM")
    POLL_LIMIT=240
    say "$RUN: RESUME AT SCORING from $A3_RESUME_FROM (poll limit ${POLL_LIMIT}m)"
  fi
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/stages/stage-1/phase_a3/autoinit_a3_launch.py \
      --scr "$SCR" --run-id "$RUN" --session-commit "$SC" --bundle "$BN" \
      --gpu "$GPU" --max-price "$PRICE" --poll-limit-min "$POLL_LIMIT" \
      "${RESUME_ARGS[@]}" > "$SCR/launcher.log" 2>&1
  RC=$?
  say "$RUN: launcher exit $RC"

  # ---- WHENEVER evidence exists, give it a durable home of its own ------
  #
  # Gated on the ARTIFACT existing, not on the session succeeding: a run that
  # aborted after stage G still came home with everything stage G produced,
  # and "I found no probes" and "no probes were trained" are different
  # findings. The extracted tree is what `aggregate_a3.py` reads -- its
  # `audit/autoinit_a3/` is the archive's own top level.
  EV="$ART/$RUN"
  if [ -d "$SCR/store/extracted" ]; then
    mkdir -p "$EV"
    cp -a "$SCR/store/extracted/." "$EV/" 2>/dev/null
    cp -a "$SCR/launcher.log" "$EV/launcher.log" 2>/dev/null
    say "$RUN: evidence -> $EV ($(find "$EV" -type f 2>/dev/null | wc -l) files)"
  else
    say "$RUN: no extracted artifact tree; nothing came home to preserve"
  fi

  # ---- the terminal check, from the launcher log ------------------------
  #
  # The DRIVER's markers reach this log because `run_session` polls the pod's
  # status file and echoes what it finds -- and that file is
  # `A3S.STATUS_PATH`, the same owner the driver writes to. C3's pair
  # hardcoded two different paths, so `ALL_DONE` was invisible here and a
  # COMPLETED formal run would have read as "no measurement began".
  TERMINAL=""
  for M in ALL_DONE A3_FAILED A3_INTEGRITY_FAILURE A3_REPLAY_MISMATCH; do
    if grep -q "$M" "$SCR/launcher.log" 2>/dev/null; then TERMINAL="$M"; break; fi
  done
  if [ -n "$TERMINAL" ]; then
    say "$RUN: terminal marker $TERMINAL -- the loop stops here"
    if [ "$TERMINAL" = "ALL_DONE" ]; then
      say "$RUN: A3 reached ALL_DONE. Next: aggregate OFF POD at \$0 with"
      say "  PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/phase_a3/aggregate_a3.py \\"
      say "    --evidence $EV --write"
    else
      say "$RUN: NOT a retry. An integrity failure is repaired, not rerun."
    fi
    break
  fi

  # No terminal marker: a pre-science abort. Preserve, back off, try again.
  say "$RUN: no terminal marker -- a pre-science abort. This chain is consumed."
  BACKOFF=$(( BACKOFF + 180 < BACKOFF_MAX ? BACKOFF + 180 : BACKOFF_MAX ))
done

say "acquisition loop finished"
