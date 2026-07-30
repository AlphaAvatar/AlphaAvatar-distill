#!/usr/bin/env bash
# Durable dev-box orchestrator for the Stage 3 teacher-corpus build.
#
# Runs under nohup/tmux on the dev box and drives one vLLM pod end to end:
# wait for setup -> generate -> fetch -> verify against the pod's own hashes ->
# build both 2x2 arms locally -> delete the pod.
#
# It must survive the session that launched it: a paid pod cannot depend on a
# conversation staying open (memory: durable-orchestration-when-backgrounding).
#
# Safety rules encoded here rather than left to judgement at the time:
#   * the pod is deleted ONLY after the fetched artifacts match the pod's own
#     sha256 manifest;
#   * on a fatal error the pod is LEFT RUNNING and --terminate-after is the cost
#     backstop;
#   * transient SSH failures are retried, not treated as fatal;
#   * the corpus is hashed before anything is built from it (P4).
#
# Usage:  POD_ID=<id> HOST=<ip> PORT=<port> nohup bash scripts/pod/orchestrate_corpus.sh &
# Inspect: tail -f artifacts/stage3/corpus_orchestrator.log
# Status:  cat artifacts/stage3/corpus_orchestrator.status
set -uo pipefail

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
unset VIRTUAL_ENV

REPO=/home/ecs-user/AlphaAvatar-distill
SESSION=corpus
OUT_GEN=${OUT_GEN:-artifacts/stage2_v2/teacher_corpus_750}
HF_REPO=${HF_REPO:-AlphaAvatar/aadistill-artifacts}
HF_PREFIX=${HF_PREFIX:-stage3_teacher_corpus_20260730}

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

rsh() {
  local cmd="$1" tries="${2:-6}" i out rc
  for ((i = 1; i <= tries; i++)); do
    out=$(timeout 600 ssh $SSH_OPTS -p "$PORT" root@"$HOST" "$cmd" 2>>"$LOG")
    rc=$?
    if [[ $rc -eq 0 ]]; then printf '%s' "$out"; return 0; fi
    log "ssh attempt $i/$tries failed (rc=$rc), retrying in 20s"
    sleep 20
  done
  return 1
}

scp_try() {
  local src="$1" dst="$2" i
  for ((i = 1; i <= 5; i++)); do
    timeout 900 scp $SSH_OPTS -P "$PORT" "root@$HOST:$src" "$dst" 2>>"$LOG" && return 0
    log "scp attempt $i failed for $src"
    sleep 15
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

log "===== corpus orchestrator start (pid $$) pod=$POD_ID host=$HOST:$PORT ====="

# ------------------------------------------------------------- phase 1: setup
setst "RUNNING:wait_setup"
markers=""
for ((i = 1; i <= 80; i++)); do   # up to ~40 min
  markers=$(rsh "grep -a 'MARKER:' /workspace/setup.log 2>/dev/null || true" 3 || true)
  case "$markers" in
    *SETUP_FAILED*) log "setup markers: $markers"
                    log "$(rsh 'tail -40 /workspace/setup.log' 3 || true)"
                    fatal "setup_failed" ;;
    *SETUP_DONE*)   break ;;
  esac
  ((i % 6 == 0)) && log "waiting for setup ($((i * 30))s elapsed)"
  sleep 30
done
[[ "$markers" == *SETUP_DONE* ]] || fatal "setup_timeout"
for m in ENV_READY XFER_OK SETUP_DONE; do
  grep -q "MARKER:$m" <<<"$markers" || fatal "missing_marker_$m"
done
log "setup complete: $(tr '\n' ' ' <<<"$markers")"

# -------------------------------------------------------- phase 2: generation
setst "RUNNING:generate"
rsh "setsid nohup bash /workspace/teacher_corpus_session.sh >/dev/null 2>&1 </dev/null & echo started" 3 >/dev/null \
  || fatal "generation_launch_failed"
log "generation session launched"

gen_ok=0
for ((i = 1; i <= 400; i++)); do   # up to ~3.3 h, outside the 2.5 h gen budget
  rm=$(rsh "cat /workspace/markers/session.log 2>/dev/null || true" 3 || true)
  case "$rm" in
    *SESSION_DONE*)     gen_ok=1; break ;;
    *SESSION_FAILED*)   log "$(rsh 'tail -40 /workspace/markers/session.log' 3 || true)"
                        log "$(rsh 'tail -30 /workspace/markers/vllm_serve.log' 3 || true)"
                        fatal "session_failed" ;;
  esac
  if ((i % 10 == 0)); then
    last=$(rsh "tail -2 /workspace/markers/generate.log 2>/dev/null || true" 3 || true)
    log "progress ($((i * 30))s): $(tr '\n' ' | ' <<<"$last")"
  fi
  sleep 30
done
((gen_ok == 1)) || fatal "generation_timeout"

if ! grep -q "GEN_DONE" <<<"$rm"; then
  log "$(rsh 'tail -40 /workspace/markers/generate.log' 3 || true)"
  fatal "generation_did_not_complete"
fi
log "generation done: $(grep -aE 'accept|complete' <<<"$rm" | tail -6 | tr '\n' ' ')"

# ---------------------------------------------------- phase 3: fetch + verify
setst "RUNNING:fetch_and_verify"
LOCAL_GEN=$REPO/$OUT_GEN
mkdir -p "$LOCAL_GEN/rollout_snapshot"
scp_try "/workspace/markers/hashes_out.txt" "$LOCAL_GEN/pod_hashes_out.txt" \
  || fatal "fetch_failed_hashes"
for f in manifest.json targets.jsonl candidates.jsonl; do
  scp_try "/workspace/aad/$OUT_GEN/$f" "$LOCAL_GEN/$f" || fatal "fetch_failed_$f"
done
for f in manifest.json rollouts.jsonl; do
  scp_try "/workspace/aad/$OUT_GEN/rollout_snapshot/$f" \
    "$LOCAL_GEN/rollout_snapshot/$f" || log "WARNING: no snapshot $f"
done
scp_try "/workspace/markers/session.log" "$LOCAL_GEN/pod_session.log" || true
scp_try "/workspace/markers/generate.log" "$LOCAL_GEN/pod_generate.log" || true
log "fetched corpus artifacts"

# Verify what landed against the hashes the pod computed before uploading. This
# is the condition for deleting the pod, so it is checked here and not inferred.
VERIFIED=1
while read -r want path; do
  [ -n "${path:-}" ] || continue
  rel=${path#*/aad/}
  local_path=$REPO/$rel
  [ -f "$local_path" ] || { log "not fetched (ok if optional): $rel"; continue; }
  got=$(sha256sum "$local_path" | awk '{print $1}')
  if [ "$got" != "$want" ]; then
    log "HASH MISMATCH $rel: pod $want local $got"
    VERIFIED=0
  fi
done < "$LOCAL_GEN/pod_hashes_out.txt"
((VERIFIED == 1)) || fatal "artifact_hash_mismatch"
log "all fetched artifacts match the pod's own sha256 manifest"

# ------------------------------------------------------ phase 4: build the arms
# Done on the dev box, on CPU, from the hashed corpus: the arms are cheap to
# build and the pod is billing, so it should not be held open for this.
setst "RUNNING:build_arms"
cd "$REPO" || fatal "cd_repo"
if uv run python scripts/data/build_stage3_pilot.py \
     --targets "$OUT_GEN/targets.jsonl" \
     --out data/stage3_pilot \
     --block-len 8192 >>"$LOG" 2>&1; then
  log "both 2x2 arms built from the hashed corpus"
  setst "RUNNING:teardown"
else
  # Not fatal for the pod: the corpus is fetched and hash-verified, so the arms
  # can be rebuilt offline. Holding a paid pod open for a CPU build would cost
  # real money for nothing.
  log "WARNING: arm build failed — corpus is safe; rebuild with"
  log "  uv run python scripts/data/build_stage3_pilot.py --targets $OUT_GEN/targets.jsonl --out data/stage3_pilot --block-len 8192"
fi

# ---------------------------------------------------------- phase 5: teardown
if runpodctl pod delete "$POD_ID" >>"$LOG" 2>&1; then
  log "pod $POD_ID deleted"
else
  log "WARNING: pod delete failed — DELETE MANUALLY: runpodctl pod delete $POD_ID"
fi
log "pods remaining: $(runpodctl pod list 2>/dev/null | grep -c '"id"' || echo unknown)"
log "balance: $(runpodctl user 2>/dev/null | grep -o '"clientBalance":[^,]*' || true)"

setst "DONE"
log "===== corpus orchestrator finished ====="
