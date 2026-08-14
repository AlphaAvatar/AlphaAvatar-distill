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

say "staging the canonical init and the recovery pack from the relay"
python3 - <<'FETCHEOF'
import os, shutil, sys, time
from pathlib import Path
from huggingface_hub import hf_hub_download
RELAY, TOKEN = "AlphaAvatar/aadistill-artifacts", os.environ["HF_TOKEN"]

def fetch(prefix, names, dest, tries=5):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        last = None
        for attempt in range(tries):
            try:
                p = hf_hub_download(RELAY, f"{prefix}/{name}", repo_type="model",
                                    token=TOKEN)
                shutil.copy(p, dest / name); break
            except Exception as exc:
                last = exc; time.sleep(5 * (attempt + 1))
        else:
            sys.exit(f"FETCH FAILED {prefix}/{name}: {last}")
        print(f"  {prefix}/{name}", flush=True)

CK = ["config.json", "generation_config.json", "model.safetensors",
      "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
fetch("stage1/qwen3_0p6b_init_v0/checkpoint", CK,
      "/workspace/aad/artifacts/stage1/qwen3_0p6b_init_v0/checkpoint")
fetch("stage3_recovery_corpus_v2/ladder_uniform",
      ["blocks.npz", "ladder.json", "audit.jsonl"],
      "/workspace/aad/artifacts/stage3/ladder_uniform_probe")
src = Path("/workspace/aad/artifacts/stage3/ladder_uniform_probe")
dst = Path("/workspace/aad/artifacts/stage3/ladder_uniform"); dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir(): shutil.copy(f, dst / f.name)
FETCHEOF

# The two frozen search assets are dev-box artifacts (untracked, ~1.6 MB total),
# so the launcher scp'd them to $WS/assets before this ran. Verified by content
# hash here, not merely by presence: the whole preflight is a measurement of
# these exact prompts.
say "installing and verifying the frozen search assets"
mkdir -p "$REPO/artifacts/stage1" "$REPO/artifacts/stage3"
cp -r "$WS/assets/state_eval_v1" "$REPO/artifacts/stage1/"
cp -r "$WS/assets/recovery_search_v2" "$REPO/artifacts/stage3/"
cd "$REPO" && PYTHONPATH=src python3 - <<'VERIFYEOF'
import hashlib, json, sys
want = {
 "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors":
   "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54",
 "artifacts/stage3/ladder_uniform_probe/blocks.npz":
   "6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c",
 "artifacts/stage3/ladder_uniform/blocks.npz":
   "6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c",
}
for path, sha in want.items():
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != sha: sys.exit(f"FROZEN ASSET MISMATCH {path}: {got}")
    print(f"  {path.rsplit('/', 2)[-2]}/{path.rsplit('/', 1)[-1]} {got[:16]}...")
VERIFYEOF

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
OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS OPENBLAS_NUM_THREADS=$NTHREADS \
  taskset -c "$CPUS" \
  timeout "${TESTS_MAX_S:-2700}" /opt/train/bin/python -m pytest tests/ -q \
  --ignore=tests/data/test_recovery_corpus_pipeline.py > /workspace/pytest.log 2>&1
RC=$?
tt=$(( $(date -u +%s) - tt0 ))
set -e
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
say "verifying $SESSION_AUTH_PATH binds to this session's plan"
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
mark AUTHORIZATION_OK

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mark SETUP_DONE
say "setup complete"
