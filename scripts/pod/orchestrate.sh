#!/usr/bin/env bash
# Durable, OS-level orchestrator for a multi-arm Stage 3 GPU session.
#
# Runs on the dev box (inside tmux/nohup) and drives the RunPod pod over SSH end
# to end: wait for setup -> score reference checkpoints -> for each arm {train ->
# gate evals -> upload} -> fetch + independently verify -> write up -> commit and
# push -> delete the pod.
#
# The session (arms, configs, checkpoints, HF paths) is defined in run_env.sh;
# this script contains no run-specific identifiers.
#
# Safety rules encoded here:
#   * the pod is deleted ONLY after remote artifact verification passes;
#   * every phase logs to $LOG; transient SSH errors are retried, not fatal;
#   * on a fatal error the pod is LEFT RUNNING for inspection and the RunPod
#     --terminate-after deadline is the cost backstop;
#   * an arm that fails after producing checkpoints is resumed, not restarted;
#   * if one arm fails terminally, later arms still run and everything already
#     produced is still fetched, verified and written up.
#
# Usage:  POD_ID=<id> HOST=<ip> PORT=<port> bash scripts/pod/orchestrate.sh
# Inspect: tail -f artifacts/stage3/<session>_orchestrator.log
# Status:  cat artifacts/stage3/<session>_orchestrator.status

set -uo pipefail

# Do not inherit a surprising PATH/venv from whatever launched the tmux session.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
unset VIRTUAL_ENV

REPO=/home/ecs-user/AlphaAvatar-distill
source "$REPO/scripts/pod/run_env.sh"

POD_ID="${POD_ID:?set POD_ID}"
HOST="${HOST:?set HOST}"
PORT="${PORT:?set PORT}"
SSHK=$HOME/.runpod/ssh/runpodctl-ssh-key
SSH_OPTS="-i $SSHK -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
 -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20
 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o BatchMode=yes"

OUTDIR=$REPO/artifacts/stage3
LOG=$OUTDIR/${SESSION}_orchestrator.log
STATUS=$OUTDIR/${SESSION}_orchestrator.status
mkdir -p "$OUTDIR"

log() { printf '%s | %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
setst() { printf '%s\n' "$1" >"$STATUS"; log "STATUS=$1"; }

# Remote exec with retries; prints stdout on success.
rsh() {
  local cmd="$1" tries="${2:-6}" i out rc
  for ((i = 1; i <= tries; i++)); do
    out=$(timeout 300 ssh $SSH_OPTS -p "$PORT" root@"$HOST" "$cmd" 2>>"$LOG")
    rc=$?
    if [[ $rc -eq 0 ]]; then printf '%s' "$out"; return 0; fi
    log "ssh attempt $i/$tries failed (rc=$rc), retrying in 20s"
    sleep 20
  done
  return 1
}

fatal() {
  setst "FAILED:$1"
  log "FATAL: $1"
  log "pod $POD_ID LEFT RUNNING for inspection (--terminate-after is the backstop)"
  log "inspect: ssh -i $SSHK -p $PORT root@$HOST"
  exit 1
}

log "===== orchestrator start (pid $$) session=$SESSION pod=$POD_ID host=$HOST:$PORT ====="
log "arms: $(arm_names | tr '\n' ' ')"

# Pre-registered abort check (run_env.sh ABORT_ARMS/ABORT_CHECK_STEP). Reads the
# arm's own train_log.jsonl on the pod and prints one of PENDING / OK / ABORT:...
# Sent base64-encoded so no quoting survives the trip through ssh.
ABORT_PY=$(base64 -w0 <<'PY'
import json, sys
run, step = sys.argv[1], int(sys.argv[2])
path = f"artifacts/stage3/{run}/train_log.jsonl"
ce = {}
try:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "eval_result" and e.get("val_set", "val") == "val":
            ce[e.get("step")] = e.get("val_ce")
        if e.get("event") == "train_step":
            loss = e.get("loss")
            if loss is not None and loss != loss:
                print(f"ABORT:non_finite_loss_at_step_{e.get('step')}")
                raise SystemExit
except FileNotFoundError:
    print("PENDING")
    raise SystemExit
if 0 not in ce or step not in ce:
    print("PENDING")
elif ce[step] > ce[0]:
    print(f"ABORT:val_ce_rose_{ce[0]:.6f}_to_{ce[step]:.6f}_at_step_{step}")
else:
    print(f"OK:val_ce_{ce[0]:.6f}_to_{ce[step]:.6f}_at_step_{step}")
PY
)

# ---------------------------------------------------------------- phase 1: setup
setst "RUNNING:wait_setup"
markers=""
for ((i = 1; i <= 140; i++)); do   # up to ~70 min
  # `|| true` inside the remote command: grep exits 1 when no marker has been
  # written yet, which is normal progress, not an SSH failure.
  markers=$(rsh "grep -a 'MARKER:' /workspace/setup.log 2>/dev/null || true" 3 || true)
  case "$markers" in
    *SETUP_FAILED*) log "setup markers: $markers"
                    log "$(rsh 'tail -40 /workspace/setup.log' 3 || true)"
                    fatal "setup_failed" ;;
    *SETUP_DONE*)   break ;;
  esac
  ((i % 10 == 0)) && log "waiting for setup ($((i * 30))s elapsed)"
  sleep 30
done
[[ "$markers" == *SETUP_DONE* ]] || fatal "setup_timeout"

for m in ENV_READY CKPT_READY TESTS_PASSED SETUP_DONE; do
  grep -q "MARKER:$m" <<<"$markers" || fatal "missing_marker_$m"
done
log "all four setup markers present"

# Independent re-verification of both hash manifests.
tv=$(rsh "cd /workspace && sha256sum -c hashes_transfer.txt 2>&1" 3 || true)
grep -q "FAILED" <<<"$tv" && { log "$tv"; fatal "transfer_hash_mismatch"; }
log "transfer hashes re-verified: $(tr '\n' ' ' <<<"$tv")"

cv=$(rsh "cd /workspace/aad && sha256sum -c /workspace/hashes_ckpt.txt 2>&1" 3 || true)
grep -q "FAILED" <<<"$cv" && { log "$cv"; fatal "ckpt_hash_mismatch"; }
log "start-checkpoint hashes re-verified: $(grep -c ': OK' <<<"$cv") files OK"

pt=$(rsh "tail -3 /workspace/pytest.log 2>/dev/null" 3 || true)
log "pod pytest: $(tr '\n' ' ' <<<"$pt")"

# ------------------------------------------------- phase 2: reference scorecards
setst "RUNNING:score_refs"
rsh "setsid nohup bash /workspace/score_refs.sh >/dev/null 2>&1 </dev/null & echo started" 3 >/dev/null \
  || fatal "score_refs_launch_failed"
refs_ok=0
for ((i = 1; i <= 60; i++)); do   # up to 30 min
  rm=$(rsh "cat /workspace/run_markers.log 2>/dev/null || true" 3 || true)
  if [[ "$rm" == *REFS_SCORED* ]]; then refs_ok=1; break; fi
  if [[ "$rm" == *REFS_FAILED* ]]; then
    log "$(rsh 'tail -30 /workspace/score_refs.log' 3 || true)"
    fatal "score_refs_failed"
  fi
  sleep 30
done
((refs_ok == 1)) || fatal "score_refs_timeout"
log "reference scorecards done"

# ------------------------------------------------------ phase 3: per-arm training
ARMS_DONE=()
ARMS_FAILED=()
for arm in "${ARMS[@]}"; do
  RUN_NAME=$(arm_field "$arm" 1)
  CONFIG=$(arm_field "$arm" 2)
  STEP_TAG=$(arm_field "$arm" 3)
  REMOTE_RUN=/workspace/aad/artifacts/stage3/$RUN_NAME

  setst "RUNNING:train:$RUN_NAME"
  log "--- arm $RUN_NAME (config $CONFIG, final $STEP_TAG) ---"

  attempt=1
  rsh "bash /workspace/train.sh '$RUN_NAME' '$CONFIG'" 3 >/dev/null || fatal "train_launch_failed_$RUN_NAME"
  log "training launched ($RUN_NAME, attempt 1)"

  max_attempts=3
  train_ok=0
  aborted=0
  abort_checked=0
  [[ " $ABORT_ARMS " == *" $RUN_NAME "* ]] || abort_checked=1   # no rule for this arm
  while :; do
    rm=$(rsh "cat /workspace/run_markers.log 2>/dev/null || true" 3 || true)
    if [[ "$rm" == *"TRAIN_DONE:$RUN_NAME"* ]]; then
      train_ok=1; log "TRAIN_DONE observed for $RUN_NAME"; break
    fi

    # Pre-registered abort rule: stop rather than retune mid-session.
    if ((abort_checked == 0)); then
      verdict=$(rsh "cd /workspace/aad && echo '$ABORT_PY' | base64 -d | python3 - '$RUN_NAME' '$ABORT_CHECK_STEP'" 3 || true)
      case "$verdict" in
        ABORT*)
          log "ABORT RULE FIRED for $RUN_NAME: $verdict"
          rsh "pkill -f 'train_stage3.py --config $CONFIG' || true" 3 >/dev/null || true
          aborted=1
          break ;;
        OK*)
          log "abort rule cleared for $RUN_NAME: $verdict"
          abort_checked=1 ;;
      esac
    fi
    if [[ "$rm" == *"TRAIN_FAILED:$RUN_NAME"* ]]; then
      log "TRAIN_FAILED observed for $RUN_NAME; console tail:"
      log "$(rsh "tail -30 /workspace/console_${RUN_NAME}.log 2>/dev/null || true" 3 || true)"
      if ((attempt >= max_attempts)); then break; fi
      have=$(rsh "ls -1 $REMOTE_RUN/checkpoints 2>/dev/null | grep -c '^step_' || true" 3 || echo 0)
      log "checkpoints present on pod for $RUN_NAME: ${have:-0}"
      if [[ "${have:-0}" -gt 0 ]]; then
        rsh "sed -i '/TRAIN_FAILED:$RUN_NAME/d' /workspace/run_markers.log" 3 >/dev/null || true
        rsh "bash /workspace/train.sh '$RUN_NAME' '$CONFIG' --resume" 3 >/dev/null \
          || fatal "resume_launch_failed_$RUN_NAME"
        ((attempt++))
        log "relaunched $RUN_NAME with --resume (attempt $attempt)"
      else
        break
      fi
    fi
    if ((SECONDS % 600 < 60)); then
      last=$(rsh "grep -aE '^step |^eval step' /workspace/console_${RUN_NAME}.log 2>/dev/null | tail -2 || true" 3 || true)
      log "progress[$RUN_NAME]: $(tr '\n' ' | ' <<<"$last")"
    fi
    sleep 60
  done

  if ((aborted == 1)); then
    # A fired abort rule is a *result*, not an infrastructure failure: it is
    # reported as a negative result and the session moves on (proposal 2026-07-27).
    log "ARM ABORTED BY PRE-REGISTERED RULE: $RUN_NAME"
    ARMS_FAILED+=("$RUN_NAME(aborted-by-rule)")
    continue
  fi

  if ((train_ok != 1)); then
    # Do not kill the session: the arms are independent, and whatever this arm
    # produced is still worth fetching and reporting.
    log "ARM FAILED: $RUN_NAME — continuing with remaining arms"
    ARMS_FAILED+=("$RUN_NAME")
    continue
  fi

  setst "RUNNING:post_run:$RUN_NAME"
  post_attempt=0
  post_ok=0
  while :; do
    ((post_attempt++))
    rsh "setsid nohup bash /workspace/post_run.sh '$RUN_NAME' '$CONFIG' '$STEP_TAG' >/dev/null 2>&1 </dev/null & echo started" 3 >/dev/null \
      || fatal "post_run_launch_failed_$RUN_NAME"
    log "post_run.sh launched for $RUN_NAME (attempt $post_attempt)"
    for ((i = 1; i <= 180; i++)); do   # up to 90 min
      rm=$(rsh "cat /workspace/run_markers.log 2>/dev/null || true" 3 || true)
      if [[ "$rm" == *"POST_DONE:$RUN_NAME"* ]]; then post_ok=1; break; fi
      if [[ "$rm" == *"POST_FAILED:$RUN_NAME"* ]]; then
        log "POST_FAILED: $(grep -a "POST_FAILED:$RUN_NAME" <<<"$rm")"
        log "$(rsh "tail -30 /workspace/post_run_${RUN_NAME}.log" 3 || true)"
        break
      fi
      ((i % 10 == 0)) && log "post_run[$RUN_NAME] running ($((i * 30))s)"
      sleep 30
    done
    ((post_ok == 1)) && { log "POST_DONE observed for $RUN_NAME"; break; }
    ((post_attempt >= 2)) && break
    rsh "sed -i '/POST_FAILED:$RUN_NAME/d' /workspace/run_markers.log" 3 >/dev/null || true
    log "retrying post_run once for $RUN_NAME"
  done

  if ((post_ok == 1)); then
    ARMS_DONE+=("$RUN_NAME")
  else
    log "ARM POST FAILED: $RUN_NAME — continuing with remaining arms"
    ARMS_FAILED+=("$RUN_NAME")
  fi
done

log "arms completed: ${ARMS_DONE[*]:-none}; failed: ${ARMS_FAILED[*]:-none}"
((${#ARMS_DONE[@]} > 0)) || fatal "no_arm_completed"

# ------------------------------------------------------- phase 4: fetch + verify
setst "RUNNING:fetch_and_verify"
scp_try() {
  local src="$1" dst="$2" i
  for ((i = 1; i <= 5; i++)); do
    timeout 600 scp $SSH_OPTS -P "$PORT" "root@$HOST:$src" "$dst" 2>>"$LOG" && return 0
    log "scp attempt $i failed for $src"
    sleep 15
  done
  return 1
}

REF_LOCAL=$REPO/artifacts/stage3/reference_scorecards
mkdir -p "$REF_LOCAL"
for entry in "${REF_CKPTS[@]}"; do
  name=$(printf '%s' "$entry" | cut -d'|' -f4)
  [ -n "$name" ] || continue
  for f in "${name}_behavior_v0.json" "${name}_behavior_v0.generations.jsonl"; do
    scp_try "/workspace/aad/artifacts/stage3/reference_scorecards/$f" "$REF_LOCAL/$f" \
      || log "WARNING: could not fetch reference scorecard $f"
  done
done

for RUN_NAME in "${ARMS_DONE[@]}"; do
  LOCAL_RUN=$REPO/artifacts/stage3/$RUN_NAME
  REMOTE_RUN=/workspace/aad/artifacts/stage3/$RUN_NAME
  mkdir -p "$LOCAL_RUN"
  for f in train_log.jsonl run_manifest.json eval_holdout_v1.json \
           eval_holdout_v1_int8.json eval_holdout_v1_int8_decoder.json \
           eval_behavior_v0.json eval_behavior_v0.generations.jsonl \
           gen_smoke.json console.log; do
    scp_try "$REMOTE_RUN/$f" "$LOCAL_RUN/$f" || fatal "fetch_failed_${RUN_NAME}_$f"
  done
  HASHFILE_REL=artifacts/stage3/${RUN_NAME}_artifact_hashes_${SESSION_DATE}.txt
  scp_try "/workspace/aad/$HASHFILE_REL" "$REPO/$HASHFILE_REL" \
    || fatal "fetch_failed_hashfile_$RUN_NAME"
  log "fetched run artifacts + pod hash list for $RUN_NAME"
done

cd "$REPO" || fatal "cd_repo"
VERIFIED=1
for RUN_NAME in "${ARMS_DONE[@]}"; do
  if uv run python scripts/pod/verify_and_report.py verify --run "$RUN_NAME" >>"$LOG" 2>&1; then
    log "UPLOAD VERIFICATION PASSED for $RUN_NAME"
  else
    VERIFIED=0
    log "UPLOAD VERIFICATION FAILED for $RUN_NAME"
  fi
done
((VERIFIED == 1)) || fatal "upload_verification_failed"

# ---------------------------------------------------------------- phase 5: report
setst "RUNNING:report"
uv run python scripts/pod/verify_and_report.py report --run "$(IFS=,; echo "${ARMS_DONE[*]}")" >>"$LOG" 2>&1 \
  || fatal "report_generation_failed"
log "experiment write-up generated"

# ---------------------------------------------------------------- phase 6: git
setst "RUNNING:git"
git add -A logs/ scripts/pod/ >>"$LOG" 2>&1
if git diff --cached --quiet; then
  log "nothing to commit"
else
  git commit -q -F - >>"$LOG" 2>&1 <<MSG
stage3: start-point ablation on mixture v1 (L40S) — logs + write-up

Autonomous session, arms: ${ARMS_DONE[*]}. Each arm ran the same 2700-step
leg from a different start point at seed 20260726, then gate evals (bf16
holdout, INT8 fake-quant at both scopes, eval_behavior_v0, generation smoke),
artifact upload to the private HF repo, and independent upload verification.
Reference checkpoints were scored on eval_behavior_v0 on the same GPU so all
arms are directly comparable.

Write-up is auto-generated from the runs' own logs and reports measured
numbers plus mechanical gate checks; the stage verdict is left open for review.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
  log "committed: $(git log --oneline -1)"
fi

if git -c credential.helper='!gh auth git-credential' push origin main >>"$LOG" 2>&1; then
  log "pushed to origin/main"
else
  log "WARNING: git push failed — commit is local only, see log above"
fi

# ---------------------------------------------------------------- phase 7: teardown
if ((VERIFIED == 1)); then
  setst "RUNNING:teardown"
  if runpodctl pod delete "$POD_ID" >>"$LOG" 2>&1; then
    log "pod $POD_ID deleted"
  else
    log "WARNING: pod delete failed — DELETE MANUALLY: runpodctl pod delete $POD_ID"
  fi
  remaining=$(runpodctl pod list 2>/dev/null | grep -c '"id"' || true)
  log "pods remaining: ${remaining:-unknown}"
  log "balance: $(runpodctl user 2>/dev/null | grep -o '"clientBalance":[^,]*' || true)"
else
  log "verification not passed — pod intentionally NOT deleted"
fi

setst "DONE arms=${ARMS_DONE[*]} failed=${ARMS_FAILED[*]:-none}"
log "===== orchestrator finished ====="
