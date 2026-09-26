#!/usr/bin/env bash
# The C3 batching-adoption pilot: one owned session, one prefix, two scorers.
#
# Modelled on batch_invariance_diagnostic_launch.sh, which is the launcher that
# has actually worked on this provider and this image. What differs, and why:
#
#   * THE PRICE IS RE-QUERIED LIVE, immediately before creation, and every
#     derived bound -- session ceiling, deadline, watchdog, ledger -- is built
#     from the live number. `L40S_MEASURED.price_per_hour_usd = $0.99` is
#     historical throughput-profile provenance and has already been reported to
#     a maintainer as though it were a quote. It is not one.
#   * The payload is a DRIVER, not a diagnostic: it replays the frozen prefix
#     once, verifies eea90c91..., and scores both arms from that one parent.
#   * The gate is a ratio of two measured wall clocks, so the two arms must run
#     under the same card and the same pinned runtime, in the order the pilot
#     record froze before either result existed.
#
# ONE resource, ONE job, teardown in a trap on every exit path.
#
#   nohup bash scripts/pod/c3_batching_pilot_launch.sh <run-id> > LOG 2>&1 &
#
set -uo pipefail

RUN_ID="${1:?usage: $0 <run-id>}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${REPO_DIR}/artifacts/pilots/c3_batching_adoption/${RUN_ID}"
BRANCH="review/c3-operator-batching"
COMMIT="${PILOT_COMMIT:?PILOT_COMMIT must name the exact source to run}"

#: The FORMAL image. The science runs under /opt/train (torch 2.11.0+cu128,
#: transformers 5.13.1) built offline from the relay wheelhouse, which is what
#: C1 attempts 17 and 18 recorded and therefore what a claim about C3's
#: scorer has to be made in.
IMAGE="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"
GPU="NVIDIA L40S"
#: NO LARGER CLASS IS AUTHORIZED. The derived conservative VRAM bound is
#: 13.24 GiB for the B4 arm against this card's 48 GiB; a bigger card is not a
#: fallback for a bound that does not fit.
DISK_GB=120

CAMPAIGN="logs/stages/stage-1/phase_c3/pilots/batching-adoption/v1/campaign.json"
SESSION_CAP="${SESSION_CAP:-2.50}"

mkdir -p "$OUT"
LOG="${OUT}/launch.log"
POD_ID=""
STARTED_EPOCH=""

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

KEY=$(grep -i apikey ~/.runpod/config.toml | sed "s/.*= *//;s/[\"']//g")
gql() {
  curl -s --max-time 30 -H "Authorization: Bearer ${KEY}" \
       -H "Content-Type: application/json" -d "$1" https://api.runpod.io/graphql
}

# --- THE LIVE QUOTE, before anything is derived from a price ----------------
# Not a constant, not the measured profile's field, and not the last session's
# number. If the live rate makes the ceiling infeasible this script STOPS; it
# does not quietly shrink the scientific work to fit.
say "re-querying ${GPU} securePrice"
PRICE_JSON=$(gql "{\"query\":\"query { gpuTypes(input:{id:\\\"${GPU}\\\"}) { id securePrice lowestPrice(input:{gpuCount:1}) { stockStatus } } }\"}")
echo "$PRICE_JSON" >> "$LOG"
read -r RATE_USD_H STOCK < <(echo "$PRICE_JSON" | python3 -c "
import json,sys
try:
    rows = json.load(sys.stdin)['data']['gpuTypes'] or []
except Exception:
    rows = []
if not rows:
    print('', ''); raise SystemExit
r = rows[0]
print(r.get('securePrice') or '', (r.get('lowestPrice') or {}).get('stockStatus') or '')
" 2>/dev/null || echo " ")
if [ -z "$RATE_USD_H" ]; then
  say "no live securePrice for ${GPU}; refusing to price a session from a"
  say "historical constant. Nothing created, nothing billed."
  exit 4
fi
say "live securePrice \$${RATE_USD_H}/h (stock ${STOCK:-unknown})"
printf '%s' "$RATE_USD_H" > "${OUT}/live_price_usd_per_hour"

# --- the budget, DERIVED by a tested script, from the LIVE rate -------------
# The ceiling is owned by the AUTHORIZATION record and the costs by the
# CAMPAIGN record -- one owner each. This script computes none of it.
BUDGET=$("${REPO_DIR}/.venv/bin/python" \
         "${REPO_DIR}/scripts/pod/engineering_campaign_budget.py" \
         "${REPO_DIR}/${CAMPAIGN}" --session-cap "$SESSION_CAP" --rate "$RATE_USD_H")
if [ $? -ne 0 ]; then
  say "budget could not be established; not creating a pod"
  exit 4
fi
read -r CAMPAIGN_CEILING SPENT_USD TEARDOWN_RESERVE REMAINING_USD \
        CEILING_USD MAX_SECONDS BUDGET_OK <<<"$BUDGET"
if [ "$BUDGET_OK" != "yes" ]; then
  say "campaign \$${CAMPAIGN_CEILING}, spent \$${SPENT_USD}, remaining \$${REMAINING_USD};"
  say "after the \$${TEARDOWN_RESERVE} teardown reserve that funds only \$${CEILING_USD}"
  say "at the LIVE rate of \$${RATE_USD_H}/h. Below what a useful attempt needs."
  say "STOPPING rather than shrinking the scientific work."
  exit 4
fi

# --- the relay credential, at $0 --------------------------------------------
HF_TOKEN="${HF_TOKEN:-$("${REPO_DIR}/.venv/bin/python" -c \
  'from huggingface_hub import get_token; print(get_token() or "")' 2>/dev/null)}"
if [ -z "$HF_TOKEN" ]; then
  say "no Hugging Face credential: neither \$HF_TOKEN nor a stored login."
  say "Refusing to create a pod that cannot obtain its inputs."
  exit 4
fi
export HF_TOKEN

# --- the calibration mixtures, VERIFIED HERE and pushed later ---------------
# Attempt a1 died 24 minutes in because the pod had one mixture and the prefix
# needs two: DEPTH and FFN on `domain_balanced@v1`, WIDTH on
# `reasoning_heavy@v2`. And `reasoning_heavy_v2` is a DEV-BOX-ONLY asset --
# built here at $0 and never uploaded -- so no relay fetch could ever have
# produced it.
#
# So the list is DERIVED from the pilot's own steps, each entry carrying the
# profile's own pinned sha256, and both files travel from this machine. 1.4 MB
# over a measured 0.72 MB/s uplink is about two seconds, and it removes the
# question of which mixture lives where.
INPUTS="${OUT}/required_inputs.jsonl"
"${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/scripts/pod/c3_batching_pilot_driver.py" \
    --required-inputs --repo "$REPO_DIR" > "$INPUTS" 2>>"$LOG"
if [ ! -s "$INPUTS" ]; then
  say "could not derive the required calibration mixtures; not creating a pod"
  exit 4
fi
MIX_OK=$("${REPO_DIR}/.venv/bin/python" - "$REPO_DIR" "$INPUTS" <<'PY'
import hashlib, json, pathlib, sys
repo, listing = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
bad = []
for line in listing.read_text().splitlines():
    if not line.strip():
        continue
    e = json.loads(line)
    f = repo / e["items_path"]
    if not f.is_file():
        bad.append(f"{e['profile_id']}: {e['items_path']} is absent on this machine")
        continue
    got = hashlib.sha256(f.read_bytes()).hexdigest()
    if e["items_file_sha256"] and got != e["items_file_sha256"]:
        bad.append(f"{e['profile_id']}: {e['items_path']} hashes {got[:16]}…, "
                   f"the profile pins {e['items_file_sha256'][:16]}…")
    else:
        print(f"  {e['profile_id']} -> {e['items_path']} ({f.stat().st_size} B)",
              file=sys.stderr)
for b in bad:
    print(f"  {b}", file=sys.stderr)
print("no" if bad else "yes")
PY
)
say "calibration mixtures required by the pilot's own steps ($(wc -l < "$INPUTS")):"
while read -r line; do
  [ -z "$line" ] && continue
  say "  $(printf '%s' "$line" | python3 -c "import json,sys;e=json.load(sys.stdin);print(e['profile_id'],'->',e['items_path'])")"
done < "$INPUTS"
if [ "$MIX_OK" != "yes" ]; then
  say "a required mixture is missing or has drifted on THIS machine."
  say "Refusing to create a pod that cannot resolve its own prefix."
  exit 4
fi

# --- the payload, rendered from the COMMIT ----------------------------------
# From `git show ${COMMIT}:`, never the working tree: the pod checks out
# ${COMMIT}, so shipping the working copy would leave a record naming a commit
# that never produced it.
REMOTE_SH="${OUT}/remote.sh"
REMOTE_SRC="scripts/pod/c3_batching_pilot_remote.sh"
if ! git -C "$REPO_DIR" show "${COMMIT}:${REMOTE_SRC}" > "${REMOTE_SH}.in" 2>/dev/null; then
  say "commit ${COMMIT} does not contain ${REMOTE_SRC}; nothing to ship"
  exit 4
fi
sed -e "s|@BRANCH@|${BRANCH}|g" -e "s|@COMMIT@|${COMMIT}|g" \
    -e "s|@RUN_ID@|${RUN_ID}|g" "${REMOTE_SH}.in" > "$REMOTE_SH"
rm -f "${REMOTE_SH}.in"
if grep -q '@[A-Z_]\{2,\}@' "$REMOTE_SH"; then
  say "unsubstituted placeholder in the rendered payload:"
  grep -n '@[A-Z_]\{2,\}@' "$REMOTE_SH" | tee -a "$LOG"
  exit 4
fi

# --- teardown, on every exit path -------------------------------------------
WATCHDOG_PID=""
teardown() {
  local rc=$?
  if [ -n "$WATCHDOG_PID" ] && [ -r "/proc/${WATCHDOG_PID}/cmdline" ] \
     && tr '\0' ' ' < "/proc/${WATCHDOG_PID}/cmdline" | grep -q "watchdog.py.*${POD_ID}"; then
    say "stopping watchdog ${WATCHDOG_PID}"
    kill "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [ -n "$POD_ID" ]; then
    say "teardown: removing pod ${POD_ID}"
    for attempt in 1 2 3 4 5; do
      runpodctl remove pod "$POD_ID" >>"$LOG" 2>&1 \
        || runpodctl pod delete "$POD_ID" >>"$LOG" 2>&1 || true
      sleep 6
      local still
      still=$(gql "{\"query\":\"query { myself { pods { id } } }\"}" \
              | python3 -c "import json,sys;print(sum(1 for p in json.load(sys.stdin)['data']['myself']['pods'] if p['id']=='${POD_ID}'))" 2>/dev/null || echo 1)
      if [ "$still" = "0" ]; then
        say "teardown: provider confirms ${POD_ID} is gone (attempt ${attempt})"
        printf '%s' "confirmed" > "${OUT}/teardown_confirmed"
        break
      fi
      say "teardown: still listed after attempt ${attempt}; retrying"
    done
  fi
  if [ -n "$STARTED_EPOCH" ]; then
    local elapsed=$(( $(date -u +%s) - STARTED_EPOCH ))
    python3 - "$OUT" "$elapsed" "$RATE_USD_H" "$POD_ID" "$rc" "$DISK_GB" <<'PY'
import json, sys
out, elapsed, rate, pod, rc, disk_gb = (
    sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4],
    int(sys.argv[5]), int(sys.argv[6]))
minutes = elapsed / 60.0
gpu = round(minutes / 60.0 * rate, 4)
# Container disk is billed SEPARATELY at $0.10/GB/month. A GPU-only ceiling
# once missed $1.69 of it and nothing in that record disagreed.
disk = round(disk_gb * 0.10 / (30 * 24 * 60) * minutes, 4)
json.dump({"elapsed_minutes": round(minutes, 3), "rate_usd_per_hour": rate,
           "_rate_is": "the LIVE securePrice quoted immediately before creation",
           "container_disk_gb": disk_gb, "gpu_usd": gpu,
           "container_disk_usd": disk, "all_in_usd": round(gpu + disk, 4),
           "pod_id": pod or None, "script_exit": rc,
           "_basis": "wall clock from pod creation to confirmed teardown x the "
                     "live secure rate, plus container disk billed separately"},
          open(f"{out}/cost.json", "w"), indent=1)
PY
    say "cost: $(python3 -c "import json;print(json.load(open('${OUT}/cost.json'))['all_in_usd'])") USD all-in"
  fi
  say "exit ${rc}"
}
trap teardown EXIT

# --- acquire ----------------------------------------------------------------
say "budget: campaign \$${CAMPAIGN_CEILING} - spent \$${SPENT_USD} = \$${REMAINING_USD}"
say "        less \$${TEARDOWN_RESERVE} reserve, capped at \$${SESSION_CAP} -> session \$${CEILING_USD}"
say "creating ${GPU} at the live \$${RATE_USD_H}/h, useful runtime ${MAX_SECONDS}s"
STARTED_EPOCH=$(date -u +%s)
CREATE=$(runpodctl create pod \
  --name "c3-pilot-${RUN_ID}" \
  --imageName "$IMAGE" \
  --gpuType "$GPU" \
  --gpuCount 1 \
  --containerDiskSize "$DISK_GB" \
  --volumeSize 0 \
  --ports "22/tcp" \
  --cost "$RATE_USD_H" \
  --startSSH \
  --secureCloud 2>&1)
echo "$CREATE" >>"$LOG"
POD_ID=$(echo "$CREATE" | sed -n 's/.*pod "\([a-z0-9]\{10,\}\)".*/\1/p' | head -1)
#: The fallback regex used to be `\b[a-z0-9]{13,16}\b`, which matched the word
#: `specifications` out of "There are no longer any instances available with
#: the requested specifications" -- so a REFUSED create produced a pod id, a
#: watchdog was started against it, and the script polled a nonexistent pod
#: for fifteen minutes. An error is checked for first, and the id is then
#: confirmed against the provider, because the only authority on whether a pod
#: exists is the provider.
if echo "$CREATE" | grep -qiE '^error:|no longer any instances|not available|insufficient'; then
  say "provider refused the create:"
  echo "$CREATE" | grep -iE '^error:|no longer any instances|not available|insufficient' | while read -r l; do say "  $l"; done
  POD_ID=""
fi
[ -z "$POD_ID" ] && POD_ID=$(echo "$CREATE" | grep -oE '\b[a-z0-9]*[0-9][a-z0-9]*\b' | grep -E '^.{13,16}$' | head -1)
if [ -z "$POD_ID" ]; then say "FAILED to create a pod; nothing billed"; exit 3; fi
#: CONFIRMED, not parsed. A pod id this script believes in but the provider
#: has never heard of is worse than no pod: the watchdog guards nothing and
#: the teardown has nothing to remove, while a real resource could be billing
#: under an id nobody recorded.
EXISTS=$(gql "{\"query\":\"query { myself { pods { id } } }\"}" \
         | python3 -c "import json,sys;print(sum(1 for p in json.load(sys.stdin)['data']['myself']['pods'] if p['id']=='${POD_ID}'))" 2>/dev/null || echo 0)
if [ "$EXISTS" != "1" ]; then
  say "the provider does not list a pod ${POD_ID}; the create did not succeed."
  say "Full output:"; echo "$CREATE" | while read -r l; do say "  $l"; done
  POD_ID=""
  exit 3
fi
say "pod ${POD_ID} (confirmed by the provider)"
printf '%s' "$POD_ID" > "${OUT}/pod_id"
printf '%s' "$STARTED_EPOCH" > "${OUT}/pod_start_epoch"

# --- the independent watchdog ------------------------------------------------
# The EXIT trap is the FIRST layer and does not run if this script is SIGKILLed
# or the box reboots. RunPod's own --terminate-after has never been observed to
# fire here, so a separate process owns the deadline and outlives this one.
WATCHDOG_MINUTES=$(python3 -c "print(int(${MAX_SECONDS}/60) + 12)")
setsid nohup "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/scripts/pod/watchdog.py" \
  --pod-id "$POD_ID" --session-start-epoch "$STARTED_EPOCH" \
  --price-per-hour "$RATE_USD_H" --hard-minutes "$WATCHDOG_MINUTES" \
  --authorized-usd "$CEILING_USD" \
  --journal "${OUT}/watchdog.jsonl" > "${OUT}/watchdog.out" 2>&1 < /dev/null &
WATCHDOG_PID=$!
say "watchdog pid ${WATCHDOG_PID}, hard deadline ${WATCHDOG_MINUTES} min"
printf '%s' "$WATCHDOG_PID" > "${OUT}/watchdog_pid"

# --- readiness: GraphQL runtime, never the CLI uptime field -----------------
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 90); do
  sleep 10
  INFO=$(gql "{\"query\":\"query { pod(input:{podId:\\\"${POD_ID}\\\"}) { runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic } } } }\"}")
  read -r SSH_HOST SSH_PORT < <(echo "$INFO" | python3 -c "
import json,sys
try:
    rt=json.load(sys.stdin)['data']['pod']['runtime'] or {}
except Exception: rt={}
for p in (rt.get('ports') or []):
    if p.get('privatePort')==22 and p.get('isIpPublic'):
        print(p['ip'], p['publicPort']); break
else: print('', '')
" 2>/dev/null || echo " ")
  [ -n "$SSH_HOST" ] && break
  say "waiting for a public port 22 mapping..."
done
if [ -z "$SSH_HOST" ]; then say "pod never exposed port 22 in 15 min"; exit 3; fi
say "ssh ${SSH_HOST}:${SSH_PORT}"

SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -p ${SSH_PORT} root@${SSH_HOST}"
for _ in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 5; done

# --- push the mixtures -------------------------------------------------------
# BEFORE the payload runs, so a transport failure costs seconds rather than a
# wheelhouse build and a root-teacher download.
# READ ON FD 3, and `ssh -n`. Attempt a3 pushed the FIRST mixture and stopped:
# `ssh` reads standard input by default, and the loop's standard input WAS the
# listing, so the first `ssh mkdir` swallowed the remaining lines. The pilot
# then refused two minutes later for exactly the input the loop had eaten.
# Either fix alone is sufficient; both are here because the failure is silent
# and costs a paid acquisition to observe.
say "pushing the calibration mixtures"
$SSH -n "mkdir -p /workspace/mixtures" >>"$LOG" 2>&1
PUSHED=0
while read -r line <&3; do
  [ -z "$line" ] && continue
  REL=$(printf '%s' "$line" | python3 -c "import json,sys;print(json.load(sys.stdin)['items_path'])")
  DIR=$(dirname "$REL")
  $SSH -n "mkdir -p /workspace/mixtures/${DIR}" >>"$LOG" 2>&1
  timeout 300 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -P "${SSH_PORT}" "${REPO_DIR}/${REL}" \
      "root@${SSH_HOST}:/workspace/mixtures/${REL}" >>"$LOG" 2>&1 \
    || { say "FAILED to push ${REL}"; exit 3; }
  PUSHED=$((PUSHED + 1))
  say "  pushed ${REL}"
done 3< "$INPUTS"
#: COUNTED, because "the loop ran" and "the loop ran to the end" are different
#: claims and a3 could not tell them apart.
WANT=$(grep -c . "$INPUTS")
if [ "$PUSHED" -ne "$WANT" ]; then
  say "pushed ${PUSHED} of ${WANT} required mixtures; refusing to continue"
  exit 3
fi
say "pushed ${PUSHED}/${WANT} required mixtures"

# --- run --------------------------------------------------------------------
say "setup + pilot (bounded at ${MAX_SECONDS}s)"
timeout "${MAX_SECONDS}" $SSH "HF_TOKEN=${HF_TOKEN} bash -s" < "$REMOTE_SH" >>"$LOG" 2>&1
RC=$?
say "remote finished rc=${RC}"

# --- collect BEFORE teardown ------------------------------------------------
# Gated on the pod having RUN, never on it having SUCCEEDED: a pilot that died
# in its second arm still measured the first, and $2.82 of verified checkpoints
# were once deleted by a collector that ran only on a clean terminal state.
# EVIDENCE ONLY. Attempt a1 pulled `/workspace/out` wholesale: 9.1 GiB of
# prefix checkpoints over a 0.72 MB/s uplink, with the pod still billing, and
# it had to be killed after ~15 minutes. Nothing on this machine consumes
# those weights -- the decision reads scores, digests and per-item values, all
# of which are JSON. The checkpoints stay on the pod and die with it.
say "fetching evidence (JSON and logs only; checkpoints stay on the pod)"
$SSH -n "cd /workspace/out && find . \( -name '*.json' -o -name '*.log' -o -name '*.txt' -o -name '*.jsonl' \) -not -path './*/work/*' -print0 | tar --null -czf /workspace/evidence.tgz --files-from=-" >>"$LOG" 2>&1 \
  || say "WARNING: could not pack the evidence"
timeout 600 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${SSH_PORT}" \
    "root@${SSH_HOST}:/workspace/evidence.tgz" "${OUT}/evidence.tgz" >>"$LOG" 2>&1 \
  || say "WARNING: could not fetch the evidence archive"
if [ -f "${OUT}/evidence.tgz" ]; then
  mkdir -p "${OUT}/out"
  tar --no-same-owner -xzf "${OUT}/evidence.tgz" -C "${OUT}/out" >>"$LOG" 2>&1 \
    || say "WARNING: could not unpack the evidence archive"
  say "evidence: $(du -sh "${OUT}/out" 2>/dev/null | cut -f1), $(find "${OUT}/out" -type f | wc -l) files"
fi
SUMMARY=$(find "$OUT" -name pilot_summary.json | head -1)
if [ -n "$SUMMARY" ]; then
  say "verdict: $(python3 -c "import json;print(json.load(open('${SUMMARY}')).get('verdict','(none)'))" 2>/dev/null || echo unreadable)"
else
  say "no pilot_summary.json retrieved"
  FAIL=$(find "$OUT" -name pilot_failure.json | head -1)
  [ -n "$FAIL" ] && say "failure: $(python3 -c "import json;print(json.load(open('${FAIL}'))['error'])" 2>/dev/null)"
fi
exit "$RC"
