#!/usr/bin/env bash
# Acquire an approved GPU for formal C3, then launch. Shipped as a FILE, never
# an inline heredoc: an unquoted heredoc is expanded by the LOCAL shell.
#
# **Polling creates nothing.** The $0 capacity watch reads provider inventory
# and writes no run directory, no authorization and no bundle. A one-use chain
# is built only when a usable approved GPU has just been seen and an actual
# acquisition is about to happen, which is what keeps stock polling from
# turning the C3 run tree into a log-count explosion.
#
# Selection is the fixed order in `stages.phase_c3.hardware`:
#
#     NVIDIA L40S -> NVIDIA RTX 6000 Ada Generation -> NVIDIA L40
#
# by AVAILABILITY only. Capacity on this account flickers between polls, so the
# window between "usable" and "create" is deliberately as short as the chain
# allows, and a capacity refusal is an ordinary transient: back off, re-poll,
# build a FRESH chain (P12.1 -- an invocation that ends before measurement
# consumed its chain even at $0).
#
# Each attempt re-prices LIVE on the device it is about to use. $1.09/h is not
# carried from L40S onto another part.
set -uo pipefail
cd /home/ecs-user/AlphaAvatar-distill
export PYTHONPATH=src:scripts
export PYTHONHASHSEED=7

# Session-scoped, and REQUIRED rather than defaulted. This line held one
# session's literal scratchpad uuid, so a later session running the same script
# wrote its launcher logs and its per-attempt scratch into a directory belonging
# to a session that had ended. A hardcoded session name has already cost this
# project two paid runs; the fix is to make the caller state it, and to refuse
# rather than guess.
# No apostrophe in the message: inside ${VAR:?word} a single quote OPENS a
# quoted string, and `session's` left the whole file unparseable.
BASE="${C3_SCRATCH_BASE:?set C3_SCRATCH_BASE to the CURRENT session scratchpad directory}"
mkdir -p "$BASE" || { echo "cannot create $BASE" >&2; exit 2; }
LOG="$BASE/c3_acquire.log"
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

# A chain that launched and never created a resource is CONSUMED (P12.1) and
# owes a closeout. Attempts 67-70 were left with directory scaffolding and no
# outcome record, and had to be closed retrospectively as bookkeeping; writing
# it here means the record exists at the moment the chain is spent.
retire_chain() {
  local n="$1" gpu="$2" price="$3"
  local d="logs/stages/stage-1/phase_c3/runs/attempt${n}/closeout"
  mkdir -p "$d"
  cat > "$d/outcome.json" <<EOF
{
 "schema": "aadistill.autoinit.c3_outcome/v1",
 "run_id": "attempt${n}",
 "experiment_id": "phase_c3",
 "status": "CAPACITY_REFUSED_NO_RESOURCE_CREATED",
 "_what_this_is": "A one-use chain built, launched, and refused at pod creation. Every pre-provider gate passed; the provider had no usable approved capacity at the moment of the create call, which is the ordinary behaviour of 'Low' stock on this account.",
 "gpu_type_id": "${gpu}",
 "_quoted_price_per_hour": ${price},
 "launcher_exit": 11,
 "stages_run": [],
 "measurement_began": false,
 "provider_resource_created": false,
 "pod_ids": [],
 "cost": {"actual_usd": 0.0, "_why_zero": "no pod was ever created, so nothing billed"},
 "final_provider_state": "nothing to tear down; no resource was created",
 "_chain_disposition": "CONSUMED even at \$0 (P12.1): the authorization is one-use and must not be reused. The incrementing attempt number is not a scope expansion.",
 "_written_by": "scripts/stages/stage-1/phase_c3/c3_acquire.sh at the moment the chain was spent"
}
EOF
  say "attempt${n}: chain retired at \$0 (${gpu})"
}

next_free() {
  local n=1
  while [ -d "logs/stages/stage-1/phase_c3/runs/attempt$n" ]; do n=$((n+1)); done
  echo "$n"
}

# The whole tier's live state, one line: "<gpu id>|<price>" or empty.
usable_now() {
  .venv/bin/python - <<'PY' 2>/dev/null
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from stages.phase_c3.hardware import query_offers, select
try:
    chosen = select(query_offers())
except Exception:
    chosen = None
if chosen:
    print(f"{chosen.gpu_type_id}|{chosen.secure_price_usd_per_hour}")
PY
}

# Backoff after a capacity refusal, in seconds, capped. The $0 watch sleeps
# only while the tier is DRY, and `Low` stock that never converts reads as
# USABLE -- so on 2026-09-29 the loop rebuilt a chain roughly every 66 seconds
# and would have spent all forty rounds in about 44 minutes instead of pacing
# them over hours. attempt75 acquired on round 5, so it cost nothing that time.
BACKOFF=0
BACKOFF_MAX=900

for ROUND in $(seq 1 40); do
  if [ "$BACKOFF" -gt 0 ]; then
    say "round $ROUND: backing off ${BACKOFF}s after a capacity refusal"
    sleep "$BACKOFF"
  fi
  # ---- $0 capacity watch. Creates nothing. --------------------------------
  PICK=""
  for _ in $(seq 1 120); do
    PICK="$(usable_now)"
    [ -n "$PICK" ] && break
    sleep 60
  done
  if [ -z "$PICK" ]; then
    say "round $ROUND: the approved tier stayed dry for 2h; continuing the watch"
    BACKOFF=0        # a dry tier already waited; it is not a refusal
    continue
  fi
  GPU="${PICK%%|*}"; PRICE="${PICK##*|}"
  N=$(next_free)
  say "round $ROUND: $GPU usable at \$$PRICE/h -> building attempt$N"

  # ---- re-price LIVE on the device we are about to use --------------------
  .venv/bin/python - "$GPU" >> "$LOG" 2>&1 <<'PY' || { say "attempt$N: repricing failed"; continue; }
import sys, json, pathlib, subprocess, datetime
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from stages.phase_c3 import formal_pricing as P
from stages.phase_c3.hardware import GPU_DISPLAY_NAMES, require_approved

gpu = require_approved(sys.argv[1])
rate = P.query_live_secure_price(gpu)
f = P.assess(rate)
rec = {
    "schema": "aadistill.phase_c3.live_pricing/v1",
    "_contract": ("The LIVE re-price required immediately before the formal "
                  "C3 authorization, on the device this session will actually "
                  "use. The authorization carries hard_ceiling.usd, NEVER the "
                  "$30.00 envelope."),
    "queried_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gpu_type_id": gpu,
    "gpu_display_name": GPU_DISPLAY_NAMES.get(gpu, ""),
    "_field_read": "securePrice, never communityPrice",
    "_selected_by": ("availability over the approved fixed order; not price, "
                     "not benchmark, not communityPrice"),
    "session_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
    **f.as_dict(),
}
pathlib.Path("logs/stages/stage-1/phase_c3/plans/c3_live_pricing.json").write_text(
    json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
if not f.fundable:
    print("NOT FUNDABLE:", f.shortfalls()); raise SystemExit(1)
print(f"repriced on {gpu}: ${rate}/h -> hard ${f.price.hard_usd}")
PY

  # ---- fresh one-use chain ------------------------------------------------
  mkdir -p "logs/stages/stage-1/phase_c3/runs/attempt$N/governance"
  cp logs/budget/approvals/autoinit_c3_grant.json \
     "logs/stages/stage-1/phase_c3/runs/attempt$N/governance/grant.json"
  git add -A >/dev/null 2>&1 && git commit -q -m "attempt$N: grant and live pricing on $GPU" >/dev/null 2>&1

  .venv/bin/python scripts/shared/pod/record_pod_environment.py --experiment phase_c3 \
      --run-id "attempt$N" --stage-id 1 --kind launch_bound >> "$LOG" 2>&1 \
      || { say "attempt$N: readiness sweep failed"; continue; }
  git add -A >/dev/null 2>&1 && git commit -q -m "attempt$N: launch-bound readiness" >/dev/null 2>&1

  .venv/bin/python scripts/stages/stage-1/phase_c3/issue_c3_authorization.py \
      --grant "logs/stages/stage-1/phase_c3/runs/attempt$N/governance/grant.json" \
      --out "logs/stages/stage-1/phase_c3/runs/attempt$N/governance/authorization.json" \
      >> "$LOG" 2>&1 || { say "attempt$N: issue refused"; continue; }
  git add -A >/dev/null 2>&1 && git commit -q -m "attempt$N: one-use authorization" >/dev/null 2>&1

  SC=$(git rev-parse HEAD)
  .venv/bin/python scripts/stages/stage-1/phase_c1/stage_c1_bundle.py --session-commit "$SC" \
      --out "logs/stages/stage-1/phase_c3/runs/attempt$N/governance/bundle.json" \
      >> "$LOG" 2>&1 || { say "attempt$N: bundle failed"; continue; }
  git add -A >/dev/null 2>&1 && git commit -q -m "attempt$N: staged bundle" >/dev/null 2>&1
  git push -q >/dev/null 2>&1

  BN="aad_autoinit_$(echo "$SC" | cut -c1-8).bundle"
  SCR="$BASE/c3_attempt$N"; mkdir -p "$SCR"
  say "attempt$N: launching on $GPU, commit $SC, bundle $BN"
  .venv/bin/python -u scripts/stages/stage-1/phase_c3/autoinit_c3_launch.py --scr "$SCR" \
      --run-id "attempt$N" --session-commit "$SC" --bundle "$BN" \
      --gpu "$GPU" --max-price "$PRICE" --host-draws 3 \
      > "$SCR/launcher.log" 2>&1
  say "attempt$N: launcher exit $?"

  if grep -q "ALL_DONE" "$SCR/launcher.log" 2>/dev/null; then
    say "attempt$N: ALL_DONE — formal C3 reached a terminal state"; break
  fi
  if grep -qE "STAGE_START:G|recovery_probes" "$SCR/launcher.log" 2>/dev/null; then
    say "attempt$N: recovery training began — stopping the loop, the session is frozen"
    break
  fi
  # A create refusal is the ordinary `Low`-stock transient. Close the chain --
  # it is CONSUMED even at $0 (P12.1) -- and back off before asking again, so
  # a dry spell does not burn the round budget in under an hour.
  retire_chain "$N" "$GPU" "$PRICE"
  BACKOFF=$(( BACKOFF == 0 ? 60 : BACKOFF * 2 ))
  [ "$BACKOFF" -gt "$BACKOFF_MAX" ] && BACKOFF=$BACKOFF_MAX
  say "attempt$N: no measurement began; retired at \$0, re-entering the watch"
done
say "acquisition loop finished"
