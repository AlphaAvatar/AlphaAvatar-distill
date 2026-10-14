#!/usr/bin/env bash
# GPU-validation capacity acquisition (Stage V, and stage W by parameter).
# Runs on the DEV BOX, never on the pod.
#
#   tmux new-session -d -s d2val "bash scripts/stages/stage-1/phase_d2/d2val_acquire.sh"
#   tmux new-session -d -s d2v3val "ACQ_VALIDATION_DIR=logs/stages/stage-1/families/d_series/validations/d2v3-gpu-validation/v2 ACQ_STAGES=W ACQ_RUN_PREFIX=d2v3val bash scripts/stages/stage-1/phase_d2/d2val_acquire.sh"
#
# The validation directory owns the grant (authorization.json: commit, session
# cap, status) and the campaign; the defaults are Stage V's, unchanged.
#
# ONE launcher invocation per attempt, each with a FRESH run id, at the grant's
# executable commit and session cap. It retries ONLY a provider capacity
# refusal (nothing created, $0) after backing off, and books every attempt in
# the campaign as it ends -- refusals at $0 with no resource. The moment a pod
# has existed it books that run and STOPS, success or failure: a retry after a
# paid attempt needs a recorded engineering cause, never a loop. It refuses to
# start an attempt while the provider lists ANY pod: at most one billing
# resource at any instant.
set -u
cd "$(dirname "$0")/../../../.." || exit 2
VDIR=${ACQ_VALIDATION_DIR:-logs/stages/stage-1/families/d_series/validations/d2-gpu-validation/v1}
STAGES=${ACQ_STAGES:-V}
PREFIX=${ACQ_RUN_PREFIX:-d2val}
case "$STAGES" in V|W) ;; *) echo "ACQ_STAGES must be V or W, got $STAGES" >&2; exit 4 ;; esac
RUNS=artifacts/stages/stage-1/phase_d1/qualification/topk_adoption
ACQ=${ACQ_LOG:-${RUNS}/${PREFIX}_acquire.log}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}
SLEEP_SECONDS=${SLEEP_SECONDS:-180}
PY=.venv/bin/python
mkdir -p "$RUNS"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ACQ"; }

read -r COMMIT CAP STATUS < <("$PY" - "$VDIR/authorization.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
print(a["executable_commit"], f"{a['session_cap_usd']:.2f}", a["status"])
PY
)
if [ "$STATUS" != "GRANTED" ]; then say "authorization is $STATUS; STOP"; exit 4; fi
KEY=$(grep -i apikey ~/.runpod/config.toml | sed "s/.*= *//;s/[\"']//g")
pods() {
  curl -s --max-time 30 -H "Authorization: Bearer ${KEY}" \
       -H "Content-Type: application/json" \
       -d '{"query":"query { myself { pods { id } } }"}' https://api.runpod.io/graphql \
    | "$PY" -c "import json,sys
try: print(len(json.load(sys.stdin)['data']['myself']['pods']))
except Exception: print('UNKNOWN')"
}
book() {  # run_id outcome [evidence]
  local ev=()
  [ -n "${3:-}" ] && ev=(--evidence-session "$3")
  PYTHONPATH=src:scripts "$PY" scripts/shared/pod/book_engineering_subrun.py \
    --run-dir "$RUNS/$1" --validation-dir "$VDIR" --subrun-id "$1" \
    --outcome "$2" "${ev[@]}" >> "$ACQ" 2>&1 || say "BOOKING FAILED for $1"
}

say "start: commit ${COMMIT}, session cap \$${CAP}, up to ${MAX_ATTEMPTS} attempts ${SLEEP_SECONDS}s apart"
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  LIVE=$(pods)
  if [ "$LIVE" != "0" ]; then
    say "attempt ${i}: provider lists pods=${LIVE}; refusing to create another. STOP"
    break
  fi
  RUN=${PREFIX}_$(date -u +%Y%m%d_%H%M%S)
  say "attempt ${i}/${MAX_ATTEMPTS}: ${RUN}"
  ADOPT_CAMPAIGN="$VDIR/campaign.json" ADOPT_COMMIT="$COMMIT" \
  ADOPT_BRANCH=exec/d1-behavioural-readiness ADOPT_STAGES="$STAGES" \
  ADOPT_PREDICTION_ROWS=all_v1 SESSION_CAP="$CAP" \
    bash scripts/stages/stage-1/phase_d1/topk_adoption_launch.sh "$RUN" \
    > "$RUNS/${RUN}.stdout" 2>&1
  RC=$?
  LOG="$RUNS/$RUN/launch.log"
  if [ ! -f "$RUNS/$RUN/pod_id" ] && grep -q "FAILED to create a pod; nothing billed" "$LOG" 2>/dev/null; then
    book "$RUN" "provider capacity refusal: no pod created, \$0"
    say "attempt ${i}: capacity refusal, nothing created. Backing off ${SLEEP_SECONDS}s"
    sleep "$SLEEP_SECONDS"
    continue
  fi
  EV=""
  [ -f "$RUNS/$RUN/out/adoption_all_v1/adoption.json" ] && EV=out/adoption_all_v1/adoption.json
  book "$RUN" "stage ${STAGES} session: launcher rc=${RC}" "$EV"
  say "attempt ${i}: a pod existed (launcher rc=${RC}); booked; STOP for review"
  break
done
say "ACQUIRE_DONE"
