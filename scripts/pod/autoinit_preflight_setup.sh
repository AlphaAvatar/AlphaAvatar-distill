#!/usr/bin/env bash
# Setup for the AutoInitializer micro-preflight. One session, four stages.
#
# Stages what Stage 0-3 need and nothing else: the canonical init, the recovery
# pack, the frozen state-eval and recovery-search assets, the teacher, both
# venvs, and the CPU suite. No E8 initializations, no depth builds, no corpus.
#
# The uv cold-host tripwire is GONE as of 2026-08-14, replaced by an offline
# install from a relay wheelhouse: it existed to tell "slow PyPI" from "hung
# host", could only do so by spending 8-28 min of paid setup, and still lost
# four of five host draws. The cgroup CPU-budget function below is copied
# verbatim from e8b_setup.sh and stays: it is what kept a 66-minute test suite
# off a 128-vCPU host that reported `nproc` 128 while the cgroup granted a
# fraction.
#
# Markers: ENV_READY -> REPO_READY -> ASSETS_STAGED -> TRAIN_ENV -> ASSETS_READY
#          -> VLLM_READY -> TEACHER_READY -> ROPE_OK -> TESTS_OK
#          -> AUTHORIZATION_OK -> SETUP_DONE

set -euo pipefail

WS=/workspace
REPO=$WS/aad
# The launcher probes this file to decide whether setup succeeded, so it must
# be the file the launcher names. Hardcoding the preflight's filename cost
# $0.1324: the continuation's setup ran to SETUP_DONE with SETUP_RC=0, wrote its
# markers here, and the launcher grepped autoinit_continuation.status, found no
# SETUP_DONE, and reported setup_failed on a session that had succeeded.
# Double-quoted, and no apostrophe in the message: an unquoted `${v:?...}`
# word is expanded, so a bare ' opens a quote that swallows whatever
# follows until the next one. `bash -n` still passed on it.
STATUS="${SESSION_STATUS:?the launcher must name the session status file}"
mark() { echo "MARKER:$1"; echo "$(date -u +%FT%TZ) MARKER:$1" >>"$STATUS"; }
say()  { echo "[$(date -u +%T)] $*"; }

export HF_TOKEN="$(cat $WS/hf/token)"
export HF_HOME=/root/.cache/huggingface

say "apt: git, ninja, zstd"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git ninja-build zstd >/dev/null
command -v ninja >/dev/null || { echo "ninja missing after install"; exit 1; }
mark ENV_READY

python3 -m pip install -q --no-input --break-system-packages \
    "huggingface_hub[hf_transfer]" 2>&1 | tail -3

say "fetching the repo bundle"
python3 -c "
import os, shutil
from huggingface_hub import hf_hub_download
name = os.environ['BUNDLE_NAME']
p = hf_hub_download('AlphaAvatar/aadistill-artifacts', f'transfer/{name}',
                    repo_type='model', token=os.environ['HF_TOKEN'])
shutil.copy(p, f'/workspace/{name}')
"
rm -rf "$REPO"; git clone -q "$WS/$BUNDLE_NAME" "$REPO"
cd "$REPO"; git checkout -q "$SESSION_COMMIT"; git rev-parse HEAD
mark REPO_READY

# THE SESSION SAYS WHICH SCIENCE INPUTS, and this script no longer knows their
# names, their destinations or their hashes. Until 2026-08-18 the block below
# was three literal `fetch(prefix, [names], dest)` calls, a directory walk that
# mirrored the recovery pack, and a `want = {...}` dict of four sha256 pins —
# executed unconditionally for every session. The sessions' own `relay_inputs`
# named at most three of the ten files staged: the micro-preflight and the
# continuation consumed the calibration without declaring it, and the device
# canary was given the whole recovery pack it had not asked for. That is the
# relay-side twin of the local-asset defect that cost the canary retry $0.0637,
# and it survived the fix that closed the other one.
#
# `SESSION_RELAY_INPUTS` is the session's own manifest as JSON: for each input
# the REPOSITORY it comes from, a path within it, the repository directory it is
# staged into on the pod, an optional second directory, and an optional sha256.
# A session that declares nothing stages nothing.
#
# `repo` is declared per item rather than fixed here, because the five
# Attempt-12 leaves live in a private TRANSPORT repo: the main relay had
# 1.60 GiB of headroom against 5.55 GiB of leaves, and pushing them by scp
# needed 1.99 MB/s against a dev box observed at 0.44-0.72 MB/s. This shell
# still names no repository, path, filename or digest of its own.
: "${SESSION_RELAY_INPUTS?the launcher must declare the relay science inputs for this session, even when there are none}"
say "staging the session's declared science inputs from the relay"
cd "$REPO"
# `REPO=` inline rather than the literal `/workspace/aad` the old block carried,
# so this exact code can be executed for real against a temporary tree. Four paid
# pods have now died inside lines no $0 path could reach; a staging block that
# can only run on a pod is one of them waiting to happen.
REPO="$REPO" python3 - <<'FETCHEOF'
import hashlib, json, os, shutil, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download
TOKEN = os.environ["HF_TOKEN"]
REPO = Path(os.environ["REPO"])

inputs = json.loads(os.environ["SESSION_RELAY_INPUTS"])
if not inputs:
    print("  (this session declares no relay science input)", flush=True)

# Retry policy unchanged from the hardcoded version: five attempts, linear
# backoff. It is the one part of this block that was ever load-bearing on a real
# host, so it is transformed rather than rewritten.
def fetch_one(repo, path, tries=5):
    last = None
    for attempt in range(tries):
        try:
            return hf_hub_download(repo, path, repo_type="model", token=TOKEN)
        except Exception as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    sys.exit(f"FETCH FAILED {repo}:{path}: {last}")

# Staged first, verified second, so a digest mismatch names the file rather than
# stopping the run at whichever fetch happened to come next.
staged = []
for item in inputs:
    src, dest = item["path"], item.get("dest")
    repo = item.get("repo")
    if not repo:
        sys.exit(f"SESSION_RELAY_INPUTS carries {src} with no repo; this script "
                 "names no repository of its own")
    if not dest:
        sys.exit(f"SESSION_RELAY_INPUTS carries {src} with no dest; setup only "
                 "receives inputs it is meant to stage")
    name = src.rsplit("/", 1)[-1]
    cached = fetch_one(repo, src)
    landed = []
    for into in (dest, item.get("also_stage_to")):
        if not into:
            continue
        d = REPO / into
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, d / name)
        landed.append(f"{into}/{name}")
    print(f"  {repo}:{src} -> {', '.join(landed)}", flush=True)
    staged.append((src, item.get("sha256"), landed))

# Every declared digest, at every destination the file landed in. The old block
# pinned the mirrored `ladder_uniform/blocks.npz` separately from the probe copy;
# checking each landing site keeps that, without a second list to maintain.
checked = 0
for src, want, landed in staged:
    if not want:
        continue
    for rel in landed:
        got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        if got != want:
            sys.exit(f"FROZEN ASSET MISMATCH {rel}: {got}")
        print(f"  {rel} {got[:16]}...", flush=True)
        checked += 1
print(f"  staged {len(staged)} inputs, verified {checked} digests", flush=True)
FETCHEOF

# Dev-box-only artifacts (untracked, ~1.6 MB total) that the launcher scp'd to
# $WS/assets before this ran. Verified by content hash below, not merely by
# presence: the whole preflight is a measurement of these exact prompts.
#
# THE SESSION SAYS WHICH ONES, and this script no longer knows their names.
# Until 2026-08-18 the two lines here were `cp -r "$WS/assets/state_eval_v1"` and
# `cp -r "$WS/assets/recovery_search_v2"`, unconditionally, under `set -e`. The
# device-canary retry declared `LOCAL_ASSETS = ()` because it honestly needed
# neither, the launcher therefore scp'd neither, and this script copied them
# anyway — into an empty directory. It died here, at $0.0637, and the session's
# declaration had been correct the whole time.
#
# `SESSION_ASSETS` is a comma-separated list of `name:install_dir` pairs, built
# from the session's own manifest. An empty value installs nothing, which is what
# a session that declared nothing must get.
# No apostrophe in this message. Line 29 already records why: inside a
# ${v?word} expansion a bare ' opens a quote that swallows the rest of the file,
# and writing one here cost a `bash -n` failure within a minute of typing it.
: "${SESSION_ASSETS?the launcher must declare the local assets for this session, even when there are none}"
say "installing the session's declared local assets: ${SESSION_ASSETS:-(none)}"
if [ -n "$SESSION_ASSETS" ]; then
  IFS=',' read -r -a _assets <<< "$SESSION_ASSETS"
  for entry in "${_assets[@]}"; do
    name="${entry%%:*}"; into="${entry#*:}"
    [ -n "$name" ] && [ -n "$into" ] && [ "$name" != "$into" ] || {
      say "MALFORMED SESSION_ASSETS ENTRY: ${entry}"; mark "SESSION_ASSETS_MALFORMED"; exit 99; }
    [ -e "$WS/assets/$name" ] || {
      say "DECLARED ASSET NOT STAGED: $name — the launcher did not scp it"
      mark "DECLARED_ASSET_MISSING:${name}"; exit 99; }
    mkdir -p "$REPO/$into"
    cp -r "$WS/assets/$name" "$REPO/$into/"
    say "  $name -> $into/"
  done
fi
# The four sha256 pins that used to live here are now fields on the session's
# own declarations, verified above at every destination each file lands in.
# `calib.domain_balanced@v1`'s items-file hash is one of them; the derived
# token-content hash d65c1f40... is a different quantity and is still checked by
# `resolve()` itself at stage 1, which is where it can be computed.
mark ASSETS_STAGED

say "training env: offline install from the relay wheelhouse"
# uv is PINNED. The cache and resolver behaviour must be the one the wheelhouse
# was built against, and `install.sh` without a version installs whatever is
# latest that day.
command -v uv >/dev/null || {
  curl -LsSf "https://astral.sh/uv/${UV_VERSION:-0.11.11}/install.sh" | sh; }
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"
sed -i 's|url = "https://download.pytorch.org/whl/cpu"|url = "https://download.pytorch.org/whl/cu128"|' pyproject.toml
sed -i 's|name = "pytorch-cpu"|name = "pytorch-cu128"|' pyproject.toml
sed -i 's|torch = { index = "pytorch-cpu" }|torch = { index = "pytorch-cu128" }|' pyproject.toml

# --- offline dependency materialization -------------------------------------
# Four of five host draws on 2026-08-14 died here, every one in the uv-sync
# window, resolving and pulling ~3.8 GiB from PyPI over the drawn host's
# network. The one healthy host did it in 45 s. The wheels now come from the
# relay, which pods read fast, and the install runs `--offline --no-index`, so
# the paid critical path contains no PyPI at all.
#
# `uv lock` is gone from the pod: `uv-cu128.lock` is that resolution, committed
# and reviewed, and `--frozen` forbids re-resolving it. A resolve on the pod was
# both a network round trip and a resolution nobody had seen.
WHEELHOUSE=${WHEELHOUSE:-/workspace/wheelhouse}
say "fetching the wheelhouse from the relay"
python3 - <<'WHEELEOF'
import os, sys, time
from huggingface_hub import snapshot_download
for attempt in range(4):
    try:
        p = snapshot_download("AlphaAvatar/aadistill-artifacts", repo_type="model",
                              allow_patterns=["transfer/wheelhouse_cu128_cp312/*"],
                              local_dir="/workspace/wh",
                              token=os.environ["HF_TOKEN"])
        break
    except Exception as exc:
        print(f"  wheelhouse attempt {attempt + 1} failed: {exc}", flush=True)
        time.sleep(10 * (attempt + 1))
else:
    sys.exit("WHEELHOUSE FETCH FAILED")
WHEELEOF
mkdir -p "$WHEELHOUSE"
cp /workspace/wh/transfer/wheelhouse_cu128_cp312/*.whl "$WHEELHOUSE"/ 2>/dev/null || true
NWHL=$(ls "$WHEELHOUSE"/*.whl 2>/dev/null | wc -l)
say "wheelhouse: ${NWHL} wheels, $(du -sh "$WHEELHOUSE" | cut -f1)"
[ "$NWHL" -ge 91 ] || { say "WHEELHOUSE TOO SMALL (${NWHL} wheels)"; \
  mark "WHEELHOUSE_INCOMPLETE:${NWHL}"; exit 92; }

# No tripwire here any more, because there is nothing left to stall on: the
# wheels are local, `--offline --no-index` forbids a network round trip, and
# `--frozen` forbids a resolve. The cold-host detector existed to tell "slow
# PyPI" from "hung host" and could only do it by burning 8-28 min of paid setup
# first. An install that reads local files either works or fails immediately.
#
# It is deliberately NOT kept as a fallback: a fallback to the network would
# reinstate the exact failure mode this removes, and would do it silently.
# `uv pip install`, NOT `uv sync`. Attempt 3 died here in 4 minutes: `uv sync
# --frozen` installs each package from the source recorded in the LOCK, and
# torch's entry is `registry = "https://download.pytorch.org/whl/cu128"`.
# `--find-links` adds a source; it does not override a registry-pinned one, so
# with `--no-index` uv had no way to obtain torch and said so. `uv pip install`
# against exported pins treats every package as a plain requirement, which
# `--find-links` can satisfy.
#
# The interpreter is PINNED to 3.12 and uv may not download one. The wheelhouse
# is cp312 — built for the 3.12.3 this pod image actually ran, read off a real
# run's recorded runtime fingerprint — and uv left to choose picks the newest it
# can find: on the dev box that was 3.14, against which every cp312 wheel is
# unusable. `UV_PYTHON_DOWNLOADS=never` also keeps a python build off the wire.
t0=$(date -u +%s)
export UV_PYTHON_DOWNLOADS=never
uv venv /opt/train --python 3.12 \
  || { say "COULD NOT CREATE /opt/train ON PYTHON 3.12"; mark "PY312_MISSING"; exit 94; }
uv pip install --python /opt/train/bin/python --offline --no-index \
  --find-links "$WHEELHOUSE" -r "$REPO/requirements-cu128.txt" \
  || { say "OFFLINE INSTALL FAILED — the wheelhouse does not satisfy the pins"
       mark "WHEELHOUSE_UNSATISFIED"; exit 93; }
# The project itself is local source, so it needs no index and no dependency
# resolution: everything it depends on was just installed from the wheelhouse.
# It IS built, though — hatchling and its chain (editables, pathspec, pluggy,
# trove-classifiers, packaging, tomlkit) are in the wheelhouse for exactly that,
# so build isolation resolves offline too. Installing the project rather than
# leaning on PYTHONPATH keeps `import aadistill` meaning what it meant before.
uv pip install --python /opt/train/bin/python --offline --no-index \
  --find-links "$WHEELHOUSE" --no-deps -e "$REPO" \
  || { say "PROJECT INSTALL FAILED"; mark "PROJECT_INSTALL_FAILED"; exit 95; }
say "offline install completed in $(( $(date -u +%s) - t0 ))s"
/opt/train/bin/python -c "import torch, transformers, sympy; \
  assert torch.cuda.is_available(); \
  print('train torch', torch.__version__, torch.cuda.get_device_name(0), \
        '| transformers', transformers.__version__)"
mark TRAIN_ENV

# The frozen search assets are checked against PREREGISTERED constants, not
# against hashes read out of their own manifests -- the latter proves only that a
# file is self-consistent. All three state_eval identities (content, canonical
# manifest, raw items) and recovery_search's content + manifest + scoring
# contract are verified. A mismatch blocks before any scientific measurement.
#
# It runs HERE, after the train venv exists, and not beside the asset staging above, because
# it needs `/opt/train`: the scoring-contract digest comes from
# `aadistill.autoinit.recovery`, and importing that package imports torch. Placed
# earlier it invoked an interpreter that does not exist yet, and the `||` branch
# reported the missing interpreter as an identity mismatch -- which cost a $0.03
# pod on 2026-08-13 and, worse, would have read as a corrupted asset. The gate is
# still well before Stage 0: the driver has not started.
say "verifying the frozen assets against the preregistered constants"
FROZEN_RC=0
FROZEN_OUT=$(cd "$REPO" && PYTHONPATH=src /opt/train/bin/python \
    scripts/autoinit/verify_frozen_assets.py 2>&1) || FROZEN_RC=$?
if [ "$FROZEN_RC" -ne 0 ]; then
  say "FROZEN ASSET GATE FAILED -- output follows verbatim, because 'the "
  say "verifier could not run' and 'these are not the preregistered assets' are "
  say "different findings and must not be reported as the same one:"
  echo "$FROZEN_OUT" | tail -25
  mark "FROZEN_ASSETS_FAILED"
  exit 91
fi
echo "$FROZEN_OUT" | tail -5
mark ASSETS_READY

# --- the vLLM environment, offline and pinned ------------------------------
# This step hung for 76 minutes on 2026-08-14 and cost $1.37: `pip install vllm`
# was UNPINNED and went to PyPI, on a host whose network had already failed
# three cold draws. The train venv beside it installed offline in 11 seconds.
#
# So the same treatment, and the whole environment rather than one wheel:
# `requirements-vllm.txt` is 196 exact pins from `uv pip compile vllm==0.27.1`,
# every wheel is staged on the relay, and the install is `--offline --no-index`.
# `pip install --upgrade pip` is gone too — it was a second network call, and an
# unpinned one.
#
# The two environments stay SEPARATE: /opt/train is torch 2.11.0+cu128 with
# transformers 5.13.1, /opt/vllm is torch 2.13.0 with transformers 5.15.0. That
# split is real and is what the RoPE check below verifies in both venvs.
#
# Pinning does NOT replace observation: the engine probe still reports the vLLM
# version, torch version, dtype, context and stop tokens that actually loaded,
# and the driver still attests the observed generation protocol against them.
say "vLLM venv, offline from the relay wheelhouse"
WH_VLLM=${WH_VLLM:-/workspace/wheelhouse_vllm}
python3 - <<'VLLMWHEELEOF'
import os, sys, time
from huggingface_hub import snapshot_download
for attempt in range(4):
    try:
        snapshot_download("AlphaAvatar/aadistill-artifacts", repo_type="model",
                          allow_patterns=["transfer/wheelhouse_vllm_cp312/*"],
                          local_dir="/workspace/whv",
                          token=os.environ["HF_TOKEN"])
        break
    except Exception as exc:
        print(f"  vllm wheelhouse attempt {attempt + 1} failed: {exc}", flush=True)
        time.sleep(10 * (attempt + 1))
else:
    sys.exit("VLLM WHEELHOUSE FETCH FAILED")
VLLMWHEELEOF
mkdir -p "$WH_VLLM"
cp /workspace/whv/transfer/wheelhouse_vllm_cp312/*.whl "$WH_VLLM"/ 2>/dev/null || true
NWHLV=$(ls "$WH_VLLM"/*.whl 2>/dev/null | wc -l)
say "vllm wheelhouse: ${NWHLV} wheels, $(du -sh "$WH_VLLM" | cut -f1)"
# Count, then BYTES. A version pin says which release; the manifest says which
# file. Verified before the install, so a truncated, re-uploaded or partial
# wheelhouse cannot reach a paid run — the 175/196 partial that the relay quota
# produced is exactly the state this refuses.
# `|| { ... }` on the command line, not `[ $? -eq 0 ]` after the heredoc: this
# script runs under `set -e`, so a failing python3 would kill it at this line
# and the marker below would never be written. The launcher classifies by
# marker, so that would have reported an unclassified death instead of a
# wheelhouse mismatch.
# The guard is one logical line: the heredoc body starts at the first
# unescaped newline, so a `{ ... }` group broken across two lines would have
# its closing brace read as python source, and the file would not parse at all.
python3 - "$WH_VLLM" "$REPO/wheelhouse_vllm_sha256.json" <<'VERIFYWHEELEOF' \
  || { say "VLLM WHEELHOUSE DOES NOT MATCH ITS FROZEN HASHES"; mark "VLLM_WHEELHOUSE_HASH_MISMATCH"; exit 96; }
import hashlib, json, pathlib, sys
wh, man = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
problems, checked = [], 0
for row in man["wheels"]:
    p = wh / row["file"]
    if not p.is_file():
        problems.append(f"MISSING {row['file']}"); continue
    if p.stat().st_size != row["bytes"]:
        problems.append(f"SIZE {row['file']}"); continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != row["sha256"]:
        problems.append(f"SHA256 {row['file']}")
    checked += 1
extra = sorted({p.name for p in wh.glob("*.whl")} - {r["file"] for r in man["wheels"]})
print(f"wheelhouse verified {checked}/{man['n_wheels']} against "
      f"{sys.argv[2].rsplit('/', 1)[-1]}", flush=True)
if extra:
    problems.append(f"UNDECLARED {extra[:5]}")
if problems:
    print("WHEELHOUSE VERIFICATION FAILED:", *problems[:10], sep="\n  ", flush=True)
    sys.exit(1)
VERIFYWHEELEOF
tv0=$(date -u +%s)
uv venv /opt/vllm --python 3.12 \
  || { say "COULD NOT CREATE /opt/vllm ON PYTHON 3.12"; mark "PY312_MISSING"; exit 94; }
uv pip install --python /opt/vllm/bin/python --offline --no-index \
  --find-links "$WH_VLLM" -r "$REPO/requirements-vllm.txt" \
  || { say "OFFLINE VLLM INSTALL FAILED — the wheelhouse does not satisfy the pins"
       mark "VLLM_WHEELHOUSE_UNSATISFIED"; exit 97; }
say "vllm offline install completed in $(( $(date -u +%s) - tv0 ))s"
/opt/vllm/bin/python -c "import vllm, torch; \
  print('vllm', vllm.__version__, '| torch', torch.__version__, torch.cuda.is_available())"
mark VLLM_READY

say "downloading the teacher at the pinned revision"
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Thinking-2507', revision=os.environ['TEACHER_REVISION'],
                  token=os.environ['HF_TOKEN'],
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt'])
"
mark TEACHER_READY

say "checking the RoPE base resolves in every venv"
for PY in /opt/train/bin/python /opt/vllm/bin/python; do
  $PY -c "
import glob, sys, transformers
sys.path.insert(0, '/workspace/aad/src')
from transformers import AutoConfig
from aadistill.models.student import assert_rope_from_config, stored_rope_base
paths = sorted(glob.glob('/workspace/aad/artifacts/stage1/*/checkpoint/config.json'))
if not paths: sys.exit('no staged checkpoint to check')
for p in paths:
    d = p.rsplit('/', 1)[0]
    cfg = AutoConfig.from_pretrained(d)
    base = assert_rope_from_config(cfg, d)
    stored = stored_rope_base(cfg)
    print(f'  transformers {transformers.__version__}: {d.rsplit(\"/\", 2)[-2]} '
          f'stored {stored:,.0f} runtime {base:,.0f} OK')
    if abs(stored - 5_000_000) > 1: sys.exit(f'{d} records RoPE base {stored}')
"
done
mark ROPE_OK

cpu_budget() {
  local q p n=""
  if [ -r /sys/fs/cgroup/cpu.max ]; then                    # cgroup v2
    read -r q p < /sys/fs/cgroup/cpu.max || true
    if [ "${q:-max}" != "max" ] && [ "${p:-0}" -gt 0 ] 2>/dev/null; then
      n=$(( q / p ))
    fi
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then     # cgroup v1
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 0)
    if [ "$q" -gt 0 ] 2>/dev/null && [ "$p" -gt 0 ] 2>/dev/null; then
      n=$(( q / p ))
    fi
  fi
  # No quota (a bare host, like the dev box) means the affinity mask is the truth.
  # NOT `nproc`: coreutils documents that it honours OMP_NUM_THREADS, so once this
  # script starts setting that variable `nproc` stops reporting the machine and
  # starts reporting our own cap — it returned 8 on a 13-cpu set that way.
  if [ -z "$n" ] || [ "$n" -lt 1 ]; then
    n=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null \
        || nproc --all)
  fi
  if [ "$n" -gt 16 ]; then n=16; fi                         # the suite needs no more
  echo "$n"
}
NCPU=$(cpu_budget)
CPUS=${TESTS_CPUS:-0-$(( NCPU - 1 ))}
NTHREADS=$(( NCPU < 8 ? NCPU : 8 ))
say "CPU test suite ($(nproc) vCPUs visible, cgroup budget ${NCPU}; cpu set ${CPUS})"
cd "$REPO"
set +e
tt0=$(date -u +%s)
# The ignore list comes from the session's manifest, and a test pins it equal to
# the pod simulator's — a simulation that runs a different command from the pod
# is not a simulation. Unquoted on purpose: `SESSION_TEST_IGNORES` is a
# space-separated flag list, and quoting it would pass one long argument.
OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS OPENBLAS_NUM_THREADS=$NTHREADS \
  taskset -c "$CPUS" \
  timeout "${TESTS_MAX_S:-2700}" /opt/train/bin/python -m pytest tests/ -q \
  ${SESSION_TEST_IGNORES:-} > /workspace/pytest.log 2>&1
RC=$?
tt=$(( $(date -u +%s) - tt0 ))
set -e
# On failure, name EVERY failure before the tail. C1 attempt 3R died here with
# `14 failed, 2650 passed` and brought home exactly three names: the tail is four
# lines and /workspace/pytest.log dies with the pod, so the other eleven had to be
# reconstructed afterwards by guessing at the pod's environment. One grep is free.
if [ "$RC" -ne 0 ]; then
  echo "--- every failing nodeid ---"
  grep -E '^(FAILED|ERROR) ' /workspace/pytest.log || true
  echo "--- log tail ---"
fi
tail -4 /workspace/pytest.log
if [ "$RC" -eq 124 ]; then
  say "COLD HOST: the CPU test suite did not finish in ${TESTS_MAX_S:-2700}s"
  mark "HOST_COLD:tests:${TESTS_MAX_S:-2700}s:$(nproc)vcpu:cpuset${CPUS}"
  exit 90
fi
[ "$RC" -eq 0 ] || { say "test suite failed rc=$RC"; exit 1; }
say "test suite passed in ${tt}s on cpu set ${CPUS}"
mark "TESTS_OK:${tt}s"

# The authorization must be loadable and bound to the live plan BEFORE the
# driver starts, so a tampered or stale artifact fails at $0.30 rather than
# after a stage has run.
#
# It is THIS SESSION's authorization, passed in by the launcher. Hardcoding the
# micro-preflight artifact and the preflight plan made every session depend on
# an unrelated one: on 2026-08-14 the continuation died here ($0.1369) because
# the preflight plan hash had moved under `pooled_counts@v2` and a historical
# artifact no longer matched it. Re-issuing that artifact would only postpone
# the same failure to the next time either plan moves.
: "${SESSION_AUTH_PATH:?the launcher must name the session authorization}"
: "${SESSION_PLAN_HASH:?the launcher must name the session plan hash}"
# Which artifact TYPE this session's authorization is. The default is the narrow
# `SpendAuthorization`, whose `allows_phase_a` is a hard False — so the assertion
# guarding the preflight and the continuation is unchanged, and a Phase-A
# artifact still cannot be loaded by them (its `phase_a_authorized` trips
# `SpendAuthorization.load`). Only a session that explicitly declares
# SESSION_KIND=phase_a gets the type that can say yes, and that type refuses
# anything not issued under the Phase-A schema.
SESSION_KIND="${SESSION_KIND:-spend}"
say "verifying $SESSION_AUTH_PATH binds to this session's plan (kind=$SESSION_KIND)"
if [ "$SESSION_KIND" = "phase_a" ]; then
  # Deliberately NOT checked here: the science plan. Setup has no executing plan
  # to compare against, so a check here could only compare two strings the
  # launcher supplied. The driver's Stage 0 rebuilds the plan and calls
  # `require_science_plan` on the rebuilt object, which is strictly stronger, and
  # it also runs `assert_preregistered` against the frozen artifact.
  cd "$REPO" && PYTHONPATH=src SESSION_AUTH_PATH="$SESSION_AUTH_PATH" \
    SESSION_PLAN_HASH="$SESSION_PLAN_HASH" /opt/train/bin/python -c "
import os
from aadistill.autoinit.phase_a import PhaseAAuthorization
a = PhaseAAuthorization.load(os.environ['SESSION_AUTH_PATH'])
a.require_plan(os.environ['SESSION_PLAN_HASH'])
assert a.allows_phase_a is True, 'a Phase-A session needs a Phase-A authorization'
assert a.automatic_followon_start is False, 'nothing chains off Phase A'
print(f'  {a.authorization_id}: stages {list(a.authorized_stages)}, '
      f'hard \${a.hard_cap_usd:.2f}, phase A {a.allows_phase_a}, '
      f'followon {a.automatic_followon_start}')
" || { say "THE SESSION AUTHORIZATION DOES NOT BIND TO THIS SESSION'S PLAN"; mark "AUTHORIZATION_MISMATCH"; exit 98; }
elif [ "$SESSION_KIND" = "phase_b" ]; then
  # A FOURTH type, and it needs its own branch for the same reason the third did.
  # Phase B's artifact is a `PhaseBAuthorization`: its plan hash lives in
  # `phase_b_session_plan_hash` and its floor in `planning_floor_usd`, because the
  # pricing review removed `expected_usd` — no expected-value assumption over
  # survivor identity or tie-break probability is defined anywhere. The spend
  # branch below therefore cannot read it, and attempt 2 proved that at $0.2300:
  # `SpendAuthorization.load` raised `KeyError: 'preflight_plan_hash'` here, one
  # step after the test gate passed.
  #
  # Routing Phase B through the spend branch would be worse than the crash even if
  # the artifact carried those keys: `SpendAuthorization.load` falls back to
  # `HARNESS_SOURCE_FILES_V1` when `harness_source_files` is absent, so the check
  # would pass while binding Phase B to PHASE A's file list.
  cd "$REPO" && PYTHONPATH=src SESSION_AUTH_PATH="$SESSION_AUTH_PATH" \
    SESSION_PLAN_HASH="$SESSION_PLAN_HASH" /opt/train/bin/python -c "
import os
from aadistill.autoinit.phase_b import PhaseBAuthorization
a = PhaseBAuthorization.load(os.environ['SESSION_AUTH_PATH'])
a.require_plan(os.environ['SESSION_PLAN_HASH'])
assert a.allows_phase_b is True, 'a Phase-B session needs a Phase-B authorization'
assert a.allows_phase_a is False, 'this artifact claims Phase A authorization'
assert a.automatic_followon_start is False, 'nothing chains off Phase B'
print(f'  {a.authorization_id}: stages {list(a.authorized_stages)}, '
      f'hard \${a.hard_cap_usd:.4f}, phase B {a.allows_phase_b}, '
      f'phase A {a.allows_phase_a}, followon {a.automatic_followon_start}')
" || { say "THE SESSION AUTHORIZATION DOES NOT BIND TO THIS SESSION'S PLAN"; mark "AUTHORIZATION_MISMATCH"; exit 98; }
elif [ "$SESSION_KIND" = "continuation_b" ]; then
  # A FIFTH type. Phase B's behavioural continuation is NOT a Phase-B session
  # with a flag turned off: its grant prices one missing `sb` and at most two
  # conditional `sc`, and it must never be substitutable for the `$35.6660`
  # artifact that books a 16.5 h P=2 search already bought and retained.
  #
  # `ContinuationAuthorization.runs_search` is False BY TYPE — there is no field
  # to set — so this branch asserts it as a contract the artifact cannot express
  # otherwise. `PhaseBAuthorization.load` would reject a continuation artifact on
  # schema, and the spend default would raise the attempt-2 KeyError; neither is
  # a safe place for this session to land.
  cd "$REPO" && PYTHONPATH=src SESSION_AUTH_PATH="$SESSION_AUTH_PATH" \
    SESSION_PLAN_HASH="$SESSION_PLAN_HASH" /opt/train/bin/python -c "
import os
from aadistill.autoinit.phase_b_continuation import ContinuationAuthorization
a = ContinuationAuthorization.load(os.environ['SESSION_AUTH_PATH'])
a.require_plan(os.environ['SESSION_PLAN_HASH'])
assert a.runs_search is False, 'the continuation cannot purchase Stage 1 again'
assert a.allows_phase_b is True, 'the continuation is a Phase-B session'
assert a.allows_phase_a is False, 'this artifact claims Phase A authorization'
assert a.automatic_followon_start is False, 'nothing chains off the continuation'
print(f'  {a.authorization_id}: stages {list(a.authorized_stages)}, '
      f'hard \${a.hard_cap_usd:.4f}, search {a.runs_search}, '
      f'phase B {a.allows_phase_b}, followon {a.automatic_followon_start}')
" || { say "THE SESSION AUTHORIZATION DOES NOT BIND TO THIS SESSION'S PLAN"; mark "AUTHORIZATION_MISMATCH"; exit 98; }
elif [ "$SESSION_KIND" = "c1" ]; then
  # A SIXTH type. Phase-C1's grant measures a harness containing the C1 launcher,
  # driver, fixed-path replayer and the new ATTENTION operator, none of which
  # appear in any earlier file set, and it carries a ceiling derived from
  # logs/phase_c1_pricing.json for six 0.86M probes rather than for a search or a
  # continuation. Every other branch would either refuse the artifact on schema or
  # — worse — accept it while binding C1 to another phase's file list and price.
  #
  # This branch exists because a missing one is not a type error: SESSION_KIND
  # falls through to `spend`, and attempt 2 of Phase B proved what that costs
  # ($0.2300, a KeyError one step after the test gate passed).
  cd "$REPO" && PYTHONPATH=src SESSION_AUTH_PATH="$SESSION_AUTH_PATH" \
    SESSION_PLAN_HASH="$SESSION_PLAN_HASH" /opt/train/bin/python -c "
import os
from aadistill.autoinit.c1_authorization import C1Authorization
a = C1Authorization.load(os.environ['SESSION_AUTH_PATH'])
a.require_plan(os.environ['SESSION_PLAN_HASH'])
assert a.authorizes_c1_isolation is True, 'a C1 session needs a C1 authorization'
assert a.allows_phase_a is False, 'this artifact claims Phase A authorization'
assert a.allows_beam_search is False, 'C1 replays one fixed path and runs no search'
assert a.automatic_followon_start is False, 'nothing chains off C1'
print(f'  {a.authorization_id}: stages {list(a.authorized_stages)}, '
      f'hard \${a.hard_cap_usd:.4f}, C1 {a.authorizes_c1_isolation}, '
      f'search {a.allows_beam_search}, phase A {a.allows_phase_a}')
" || { say "THE SESSION AUTHORIZATION DOES NOT BIND TO THIS SESSION'S PLAN"; mark "AUTHORIZATION_MISMATCH"; exit 98; }
elif [ "$SESSION_KIND" = "recovery_continuation" ]; then
  # A THIRD type, not a relaxation of the second. The continuation's artifact
  # carries `phase_a_authorized: true` (it runs Phase-A stages), so the spend
  # branch below would refuse it — and the phase_a branch above would accept a
  # full-search authorization in its place, at the search's ceiling.
  cd "$REPO" && PYTHONPATH=src SESSION_AUTH_PATH="$SESSION_AUTH_PATH" \
    SESSION_PLAN_HASH="$SESSION_PLAN_HASH" /opt/train/bin/python -c "
import os
from aadistill.autoinit.recovery_continuation import RecoveryContinuationAuthorization
a = RecoveryContinuationAuthorization.load(os.environ['SESSION_AUTH_PATH'])
a.require_plan(os.environ['SESSION_PLAN_HASH'])
assert a.authorizes_recovery_continuation is True
assert a.allows_beam_search is False, 'the continuation cannot reach a search'
assert a.automatic_followon_start is False, 'nothing chains off the continuation'
print(f'  {a.authorization_id}: stages {list(a.authorized_stages)}, '
      f'hard \${a.hard_cap_usd:.4f}, search {a.allows_beam_search}, '
      f'followon {a.automatic_followon_start}')
" || { say "THE SESSION AUTHORIZATION DOES NOT BIND TO THIS SESSION'S PLAN"; mark "AUTHORIZATION_MISMATCH"; exit 98; }
else
  cd "$REPO" && PYTHONPATH=src SESSION_AUTH_PATH="$SESSION_AUTH_PATH" \
    SESSION_PLAN_HASH="$SESSION_PLAN_HASH" /opt/train/bin/python -c "
import os
from aadistill.autoinit.authorization import SpendAuthorization
a = SpendAuthorization.load(os.environ['SESSION_AUTH_PATH'])
a.require_plan(os.environ['SESSION_PLAN_HASH'])
assert a.allows_phase_a is False, 'this artifact claims Phase A authorization'
print(f'  {a.authorization_id}: stages {list(a.authorized_stages)}, '
      f'hard \${a.hard_cap_usd:.2f}, phase A {a.allows_phase_a}')
" || { say "THE SESSION AUTHORIZATION DOES NOT BIND TO THIS SESSION'S PLAN"; mark "AUTHORIZATION_MISMATCH"; exit 98; }
fi
mark AUTHORIZATION_OK

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mark SETUP_DONE
say "setup complete"
