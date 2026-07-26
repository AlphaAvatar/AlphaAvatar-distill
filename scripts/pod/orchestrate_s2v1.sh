#!/usr/bin/env bash
# Durable, OS-level orchestrator for the Stage 3 `s2_blocks_v1` GPU session.
#
# Runs on the dev box (inside tmux) and drives the RunPod pod over SSH end to
# end: wait for setup -> train -> gate evals -> upload -> independently verify
# the upload -> generate + commit + push the write-up -> delete the pod.
#
# Safety rules encoded here:
#   * the pod is deleted ONLY after remote artifact verification passes;
#   * every phase logs to $LOG; transient SSH errors are retried, not fatal;
#   * on a fatal error the pod is LEFT RUNNING for inspection and the RunPod
#     --terminate-after deadline is the cost backstop.
#
# Inspect:  tail -f artifacts/stage3/s2v1_orchestrator.log
# Status:   cat artifacts/stage3/s2v1_orchestrator.status

set -uo pipefail

# Do not inherit a surprising PATH/venv from whatever launched the tmux session.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
unset VIRTUAL_ENV

REPO=/home/ecs-user/AlphaAvatar-distill
POD_ID=ippwmpc8wzed24
HOST=82.221.170.242
PORT=37435
SSHK=$HOME/.runpod/ssh/runpodctl-ssh-key
SSH_OPTS="-i $SSHK -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
 -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20
 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o BatchMode=yes"

RUN_REL=artifacts/stage3/s2_blocks_v1
LOCAL_RUN=$REPO/$RUN_REL
OUTDIR=$REPO/artifacts/stage3
LOG=$OUTDIR/s2v1_orchestrator.log
STATUS=$OUTDIR/s2v1_orchestrator.status
HASHFILE_REL=artifacts/stage3/s2v1_artifact_hashes_2026-07-26.txt
HF_REPO=AlphaAvatar/aadistill-artifacts
HF_PREFIX=stage3/s2_blocks_v1
REMOTE_RUN=/workspace/aad/$RUN_REL

mkdir -p "$OUTDIR" "$LOCAL_RUN"

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

log "===== orchestrator start (pid $$) pod=$POD_ID host=$HOST:$PORT ====="

# ---------------------------------------------------------------- phase 1
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

# ---------------------------------------------------------------- phase 2
setst "RUNNING:train"
launch_fresh() { rsh "bash /workspace/train.sh" 3; }
launch_resume() {
  rsh "cd /workspace/aad && UV_PROJECT_ENVIRONMENT=/root/venv HF_HOME=/workspace/hf \
PATH=\$HOME/.local/bin:\$PATH setsid nohup bash -c \
'uv run python scripts/train_stage3.py --config configs/stage3_s2_blocks_v1.json --resume \
>> /workspace/console_s2v1.log 2>&1; rc=\$?; \
if [ \$rc -eq 0 ]; then echo MARKER:TRAIN_DONE; else echo MARKER:TRAIN_FAILED rc=\$rc; fi \
>> /workspace/run_markers.log' >/dev/null 2>&1 </dev/null & echo relaunched" 3
}

attempt=0
max_attempts=3
train_ok=0
launch_fresh >/dev/null || fatal "train_launch_failed"
log "training launched (attempt 1)"
((attempt++))

while :; do
  rm=$(rsh "cat /workspace/run_markers.log 2>/dev/null || true" 3 || true)
  if [[ "$rm" == *TRAIN_DONE* ]]; then
    train_ok=1
    log "TRAIN_DONE observed"
    break
  fi
  if [[ "$rm" == *TRAIN_FAILED* ]]; then
    log "TRAIN_FAILED observed; console tail:"
    log "$(rsh 'tail -30 /workspace/console_s2v1.log 2>/dev/null || true' 3 || true)"
    if ((attempt >= max_attempts)); then fatal "train_failed_after_${attempt}_attempts"; fi
    have=$(rsh "ls -1 $REMOTE_RUN/checkpoints 2>/dev/null | grep -c '^step_' || true" 3 || echo 0)
    log "checkpoints present on pod: ${have:-0}"
    if [[ "${have:-0}" -gt 0 ]]; then
      rsh "sed -i '/TRAIN_FAILED/d' /workspace/run_markers.log" 3 >/dev/null || true
      launch_resume >/dev/null || fatal "resume_launch_failed"
      ((attempt++))
      log "relaunched with --resume (attempt $attempt)"
    else
      fatal "train_failed_no_checkpoint_to_resume"
    fi
  fi
  # progress heartbeat every ~10 min
  if ((SECONDS % 600 < 60)); then
    last=$(rsh "grep -aE '^step |^eval step' /workspace/console_s2v1.log 2>/dev/null | tail -2 || true" 3 || true)
    log "progress: $(tr '\n' ' | ' <<<"$last")"
  fi
  sleep 60
done
((train_ok == 1)) || fatal "train_did_not_complete"

# ---------------------------------------------------------------- phase 3
setst "RUNNING:post_run"
post_attempt=0
while :; do
  ((post_attempt++))
  rsh "setsid nohup bash /workspace/post_run.sh >/dev/null 2>&1 </dev/null & echo started" 3 >/dev/null \
    || fatal "post_run_launch_failed"
  log "post_run.sh launched (attempt $post_attempt)"
  done_post=0
  for ((i = 1; i <= 180; i++)); do   # up to 90 min
    rm=$(rsh "cat /workspace/run_markers.log 2>/dev/null || true" 3 || true)
    if [[ "$rm" == *POST_DONE* ]]; then done_post=1; break; fi
    if [[ "$rm" == *POST_FAILED* ]]; then
      log "POST_FAILED: $(grep -a POST_FAILED <<<"$rm")"
      log "$(rsh 'tail -30 /workspace/post_run.log' 3 || true)"
      break
    fi
    ((i % 10 == 0)) && log "post_run running ($((i * 30))s)"
    sleep 30
  done
  ((done_post == 1)) && { log "POST_DONE observed"; break; }
  ((post_attempt >= 2)) && fatal "post_run_failed"
  rsh "sed -i '/POST_FAILED/d' /workspace/run_markers.log" 3 >/dev/null || true
  log "retrying post_run once"
done

# ---------------------------------------------------------------- phase 4
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
for f in train_log.jsonl run_manifest.json eval_holdout_v1.json \
         eval_holdout_v1_int8.json eval_holdout_v1_int8_decoder.json \
         gen_smoke.json console.log; do
  scp_try "$REMOTE_RUN/$f" "$LOCAL_RUN/$f" || fatal "fetch_failed_$f"
done
scp_try "/workspace/aad/$HASHFILE_REL" "$REPO/$HASHFILE_REL" || fatal "fetch_failed_hashfile"
log "fetched run artifacts + pod hash list to $LOCAL_RUN"

cd "$REPO" || fatal "cd_repo"
if uv run python scripts/pod/verify_and_report_s2v1.py verify >>"$LOG" 2>&1; then
  log "UPLOAD VERIFICATION PASSED"
  VERIFIED=1
else
  VERIFIED=0
  fatal "upload_verification_failed"
fi

# ---------------------------------------------------------------- phase 5
setst "RUNNING:report"
uv run python scripts/pod/verify_and_report_s2v1.py report >>"$LOG" 2>&1 \
  || fatal "report_generation_failed"
log "experiment write-up generated"

# ---------------------------------------------------------------- phase 6
setst "RUNNING:git"
git add -A logs/ scripts/pod/ >>"$LOG" 2>&1
if git diff --cached --quiet; then
  log "nothing to commit"
else
  git commit -q -F - >>"$LOG" 2>&1 <<'MSG'
stage3: s2_blocks_v1 recovery run on mixture v1 (L40S) — logs + write-up

Autonomous session. Training, gate evals (bf16 holdout, INT8 fake-quant at
both scopes, generation smoke), artifact upload to the private HF repo, and
independent upload verification (LFS sha256 for the weights, download+hash
for small files) all completed by scripts/pod/orchestrate_s2v1.sh.

Experiment write-up is auto-generated from the run's own logs and reports
measured numbers plus mechanical gate checks; the stage verdict is left open
for review.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
  log "committed: $(git log --oneline -1)"
fi

if git -c credential.helper='!gh auth git-credential' push origin main >>"$LOG" 2>&1; then
  log "pushed to origin/main"
else
  log "WARNING: git push failed — commit is local only, see log above"
fi

# ---------------------------------------------------------------- phase 7
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

setst "DONE"
log "===== orchestrator finished ====="
