"""Run the PRODUCTION setup script end to end, then read it back the way the
launcher does.

Three consecutive paid failures came from lines and interfaces in
`autoinit_preflight_setup.sh` that no rehearsal had ever executed:

* `$0.07`   — `uv sync --frozen` cannot install a registry-pinned wheel offline;
* `$1.3672` — an unpinned `pip install vllm` on the paid critical path;
* `$0.1369` — setup asserted an unrelated session's authorization;
* `$0.1324` — setup wrote its markers to a filename the launcher does not probe.

Each was found by paying for the next one. Rehearsing only the block just edited
stopped being the economical strategy, so this executes **every control-flow line
of the real script**, in a real `/workspace`, under its own
`set -euo pipefail`, with its heredocs, env expansion and marker writes intact —
and then feeds the result to the launcher's own `PROBE_COMMAND` and
`parse_setup_probe`.

**No setup logic is reimplemented here.** Only expensive external operations are
stubbed: apt, pip, the Hub fetches (served from local fixtures), the uv install
and venv creation, the pod-side pytest run, git, and nvidia-smi.

What still runs for real, because it is cheap and it is what the script is for:

* the sha256 verification of the 1.19 GiB canonical checkpoint and the recovery
  pack against the constants embedded in the script;
* `verify_frozen_assets.py` against the preregistered identities;
* the **196/196 vLLM wheel byte gate**, against the real wheels;
* the RoPE base check, through real transformers and real `aadistill`;
* the session authorization gate, against the real artifact and plan hash;
* the cgroup CPU-budget arithmetic.

The sandbox is bubblewrap: the script hardcodes `WS=/workspace`, which no test
may create on the host.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"
LAUNCH = REPO / "scripts/pod/autoinit_preflight_launch.py"
#: The byte gate needs the real wheels. They are ~3.6 GiB and are not in the
#: repo, so the location is overridable and the test skips without them rather
#: than pretending to have checked.
#: A stable path, deliberately not a session scratchpad: the default used to
#: point at one, which would have made this test silently skip in the next
#: session -- the exact rot the rehearsal exists to prevent.
WHEELHOUSE_VLLM = Path(os.environ.get(
    "AAD_VLLM_WHEELHOUSE",
    os.path.expanduser("~/aad-artifacts/wheelhouse_vllm_cp312")))

STUB_HF = '''\
"""Serves every Hub fetch from a local fixture tree, so the script's real
download code runs without a network."""
import os, shutil
from pathlib import Path

FIXTURES = Path(os.environ["REHEARSAL_FIXTURES"])


def hf_hub_download(repo_id, filename, repo_type=None, token=None, **kw):
    src = FIXTURES / filename
    if not src.is_file():
        raise FileNotFoundError(f"rehearsal fixture missing: {filename}")
    return str(src)


def snapshot_download(repo_id, repo_type=None, allow_patterns=None,
                      local_dir=None, token=None, revision=None, **kw):
    if local_dir is None:
        return str(FIXTURES)
    dest = Path(local_dir)
    for pattern in (allow_patterns or []):
        prefix = pattern.rsplit("/", 1)[0]
        src, out = FIXTURES / prefix, dest / prefix
        if out.is_dir() and any(out.iterdir()):
            continue          # bind-mounted already; nothing to fetch
        if not src.is_dir():
            raise FileNotFoundError(f"rehearsal fixture missing: {prefix}")
        out.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy(f, out / f.name)
    return str(dest)
'''

STUB_SITECUSTOMIZE = '''\
"""The rehearsal box has no GPU. Patch only what the script asserts."""
import builtins
_real = builtins.__import__


def _hook(name, *a, **k):
    mod = _real(name, *a, **k)
    if name.split(".")[0] == "torch":
        try:
            import torch
            torch.cuda.is_available = lambda: True
            torch.cuda.get_device_name = lambda i=0: "NVIDIA L40S (rehearsal)"
        except Exception:
            pass
    return mod


builtins.__import__ = _hook
'''


def _write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def _skip_reason() -> str | None:
    if not shutil.which("bwrap"):
        return "bubblewrap is needed to provide /workspace without touching the host"
    if not (REPO / ".venv/bin/python").exists():
        return "the repo venv provides the real torch/transformers/aadistill"
    if not (REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
            / "model.safetensors").is_file():
        return "the canonical checkpoint is not staged locally"
    if len(list(WHEELHOUSE_VLLM.glob("*.whl"))) != 196:
        return (f"the 196-wheel vLLM wheelhouse is not at {WHEELHOUSE_VLLM}; "
                "build it with scripts/pod/build_wheelhouse.py --from-pins "
                "or set AAD_VLLM_WHEELHOUSE")
    return None


@pytest.fixture(scope="module")
def sandbox():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    tmp = Path(tempfile.mkdtemp(prefix="aad-setup-rehearsal-"))
    try:
        yield _build(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _build(tmp: Path) -> dict:
    ws, home, bin_ = tmp / "workspace", tmp / "home", tmp / "bin"
    fetch_stubs, venv_stubs = tmp / "stubs_fetch", tmp / "stubs_venv"
    fixtures = tmp / "fixtures"
    for d in (ws, home, bin_, fetch_stubs, venv_stubs, fixtures):
        d.mkdir(parents=True, exist_ok=True)

    # Two disjoint stub sets. Only the FETCHING interpreter gets the
    # huggingface_hub stand-in; the venv interpreters need the real package,
    # because transformers imports `huggingface_hub.utils`.
    _write(fetch_stubs / "huggingface_hub.py", STUB_HF)
    _write(venv_stubs / "vllm.py", '__version__ = "0.27.1"\n')
    _write(venv_stubs / "sitecustomize.py", STUB_SITECUSTOMIZE)

    # What the git stub will "clone": the working tree minus the heavy trees.
    src = tmp / "repo_src"
    shutil.copytree(REPO, src, symlinks=True,
                    ignore=shutil.ignore_patterns(".git", ".venv", "artifacts",
                                                  "__pycache__"))
    (src / "artifacts").mkdir(exist_ok=True)

    # Fixtures are SYMLINKS: the script's own `shutil.copy` still copies them
    # into the workspace, which is the code under test; copying them twice is
    # only disk.
    ck = fixtures / "stage1/qwen3_0p6b_init_v0/checkpoint"
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.symlink_to(REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint")
    lad = fixtures / "stage3_recovery_corpus_v2/ladder_uniform"
    lad.parent.mkdir(parents=True, exist_ok=True)
    lad.symlink_to(REPO / "artifacts/stage3/ladder_uniform_probe")
    # The cu128 wheelhouse is COUNT-checked only (>= 91), so empty files are a
    # faithful stand-in. The vLLM one is BYTE-checked, so it is bind-mounted
    # real below.
    cu = fixtures / "transfer/wheelhouse_cu128_cp312"
    cu.mkdir(parents=True, exist_ok=True)
    for i in range(91):
        (cu / f"stub_{i:03d}-1.0-py3-none-any.whl").write_bytes(b"")
    (fixtures / "transfer/aad_rehearsal.bundle").write_bytes(b"bundle stand-in")

    for name, rel in (("state_eval_v1", "artifacts/stage1/state_eval_v1"),
                      ("recovery_search_v2", "artifacts/stage3/recovery_search_v2")):
        shutil.copytree(REPO / rel, ws / "assets" / name)
    _write(ws / "hf/token", "rehearsal-token\n")

    _write(bin_ / "apt-get", "#!/bin/sh\nexit 0\n", 0o755)
    _write(bin_ / "ninja", "#!/bin/sh\nexit 0\n", 0o755)
    _write(bin_ / "nvidia-smi",
           '#!/bin/sh\necho "NVIDIA L40S, 46068 MiB, 580.159.03"\n', 0o755)
    # `curl … | sh` is the real line, and a fresh pod has no uv, so the stub
    # emits the shell text that installs one and the pipeline itself executes.
    _write(bin_ / "curl", f"""#!/bin/sh
cat <<'INSTALLER'
mkdir -p "$HOME/.local/bin"
cp {bin_}/uv_impl "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
INSTALLER
""", 0o755)
    _write(bin_ / "uv_impl", f"""#!/bin/sh
case "$1" in
  venv) d="$2"; mkdir -p "$d/bin"
        cp {bin_}/venv_python "$d/bin/python"; chmod +x "$d/bin/python"; exit 0 ;;
  *) exit 0 ;;
esac
""", 0o755)
    # The venv interpreter is the dev box's real one. Only the pod-side pytest
    # run is stood in for -- it is the expensive operation, and it has passed on
    # real hardware on every attempt.
    _write(bin_ / "venv_python", f"""#!/bin/sh
for a in "$@"; do
  if [ "$a" = "pytest" ]; then
    echo "..... [100%]"; echo "rehearsal stub: pod-side suite not re-run here"
    exit 0
  fi
done
PYTHONPATH="{venv_stubs}:${{PYTHONPATH:-}}" exec {REPO}/.venv/bin/python "$@"
""", 0o755)
    _write(bin_ / "git", f"""#!/bin/sh
case "$1" in
  clone) shift; [ "$1" = "-q" ] && shift; mkdir -p "$2"; cp -a "{src}/." "$2/"
         exit 0 ;;
  rev-parse) echo "$REHEARSAL_COMMIT"; exit 0 ;;
  *) exit 0 ;;
esac
""", 0o755)
    _write(bin_ / "python3", f"""#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then exit 0; fi
PYTHONPATH="{fetch_stubs}:${{PYTHONPATH:-}}" exec {sys.executable} "$@"
""", 0o755)
    return {"tmp": tmp, "ws": ws, "home": home, "bin": bin_,
            "fixtures": fixtures}


def _run_setup(box: dict, **env_extra) -> subprocess.CompletedProcess:
    from aadistill.autoinit.continuation import CONTINUATION_PLAN_V1

    ws, home, bin_ = box["ws"], box["home"], box["bin"]
    wh_vllm = "/workspace/whv/transfer/wheelhouse_vllm_cp312"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    env = {
        # Deliberately WITHOUT ~/.local/bin: a fresh pod has no uv, so the
        # install branch must actually run.
        "PATH": f"{bin_}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "SESSION_STATUS": "/workspace/autoinit_continuation.status",
        "SESSION_AUTH_PATH": "logs/autoinit_continuation_authorization.json",
        "SESSION_PLAN_HASH": CONTINUATION_PLAN_V1.plan_hash,
        "BUNDLE_NAME": "aad_rehearsal.bundle",
        "SESSION_COMMIT": commit, "REHEARSAL_COMMIT": commit,
        "REHEARSAL_FIXTURES": "/workspace/fixtures",
        "TEACHER_REVISION": "768f209d9ea81521153ed38c47d515654e938aea",
        "WHEELHOUSE": "/workspace/wheelhouse", "WH_VLLM": wh_vllm,
        "TESTS_MAX_S": "600", "LANG": "C.UTF-8",
    }
    env.update(env_extra)
    cmd = [
        "bwrap",
        # /tmp is a tmpfs FIRST, so the binds below are not masked by it.
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
        "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
        "--ro-bind-try", "/bin", "/bin", "--ro-bind-try", "/sbin", "/sbin",
        "--ro-bind-try", "/sys", "/sys",
        # uv venvs symlink to an interpreter under ~/.local/share/uv, so the
        # real home must be visible even though HOME points elsewhere.
        "--ro-bind-try", os.path.expanduser("~/.local"),
        os.path.expanduser("~/.local"),
        "--ro-bind", str(REPO), str(REPO),
        "--bind", str(box["tmp"]), str(box["tmp"]),
        "--bind", str(ws), "/workspace",
        "--bind", str(home), str(home),
        "--ro-bind", str(box["fixtures"]), "/workspace/fixtures",
        "--ro-bind", str(WHEELHOUSE_VLLM), wh_vllm,
        "--setenv", "PATH", env["PATH"],
        "--", "bash", str(SETUP),
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=1800)


def _launcher_readback(box: dict, rc: int, stdout: str) -> dict:
    """Exactly what the launcher does: redirect setup to setup.log, append
    SETUP_RC, then run PROBE_COMMAND and parse_setup_probe."""
    ws = box["ws"]
    (ws / "setup.log").write_text(stdout + f"SETUP_RC={rc}\n")
    spec = importlib.util.spec_from_file_location("preflight_launch", LAUNCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_launch"] = mod
    spec.loader.exec_module(mod)
    cmd = mod.PROBE_COMMAND.format(
        status=str(ws / "autoinit_continuation.status"),
        log=str(ws / "setup.log"))
    out = subprocess.run(["bash", "-c", cmd], capture_output=True,
                         text=True).stdout
    return mod.parse_setup_probe(out)


def test_the_real_setup_runs_to_SETUP_DONE_and_the_launcher_reads_it(sandbox):
    """rc 0, AUTHORIZATION_OK, SETUP_DONE through SESSION_STATUS, and the
    launcher's own probe classifies it as a success. Attempt 6 satisfied the
    first three and failed the fourth."""
    r = _run_setup(sandbox)
    assert r.returncode == 0, (
        f"setup exited {r.returncode}\n--- stdout ---\n{r.stdout[-4000:]}\n"
        f"--- stderr ---\n{r.stderr[-2000:]}")

    status = sandbox["ws"] / "autoinit_continuation.status"
    assert status.is_file(), "no markers were written to the named status file"
    markers = status.read_text()
    # The full ladder, in order, from the script's own header.
    for name in ("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV",
                 "ASSETS_READY", "VLLM_READY", "TEACHER_READY", "ROPE_OK",
                 "TESTS_OK", "AUTHORIZATION_OK", "SETUP_DONE"):
        assert f"MARKER:{name}" in markers, f"{name} missing from {markers}"
    assert markers.index("AUTHORIZATION_OK") < markers.index("SETUP_DONE")

    # The gates that matter ran for real, not as stubs.
    assert "wheelhouse verified 196/196" in r.stdout
    assert "86fbba78e8a2a324" in r.stdout          # canonical checkpoint sha256
    assert "stored 5,000,000 runtime 5,000,000 OK" in r.stdout
    assert "autoinit.control_characterization" in r.stdout

    # And the launcher agrees. This is the composition attempt 6 got wrong.
    probe = _launcher_readback(sandbox, r.returncode, r.stdout)
    assert probe["setup_done"] not in ("", "0"), probe
    assert probe["setup_rc"] == "0", probe
    assert probe["host_cold"] in ("", "0"), probe


def test_an_unnamed_session_status_cannot_be_silently_accepted(sandbox):
    """The one fail-closed case: the launcher must name the status file.

    Left unset, setup must die before writing anything, and the launcher must
    read that as a failure -- not as a session that quietly wrote its markers
    somewhere nobody looks, which is what cost $0.1324.

    A wrong or stale *authorization* is covered where it is cheap to exercise,
    in test_continuation_rehearsal.py, which runs that block directly.
    """
    # A fresh pod has no status file; the happy-path test above left one here.
    (sandbox["ws"] / "autoinit_continuation.status").unlink(missing_ok=True)
    r = _run_setup(sandbox, SESSION_STATUS="")
    assert r.returncode != 0
    assert "must name the session status file" in r.stderr
    probe = _launcher_readback(sandbox, r.returncode, r.stdout)
    assert probe["setup_done"] in ("", "0"), probe
