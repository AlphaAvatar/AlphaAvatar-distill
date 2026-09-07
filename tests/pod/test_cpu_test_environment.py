"""One CPU-test environment, so an exact skip comparison compares like with like.

C1 attempt 5's `--strict` skip-set comparison is correct machinery that was
pointed at two different machines. The launch-bound diagnostic runs on a CPU dev
box with an isolated empty HF cache; the paid pod is an L40S with the pinned
teacher already downloaded. Compared exactly, a HEALTHY pod would have been
refused for being healthy:

* `torch.cuda.is_available()` is False here and True there — and the one GPU
  predicate in the selected suite RUNS without CUDA and SKIPS with it, so it
  would have decided the opposite way on the pod;
* the tokenizer cases skip here on an empty cache and would RUN there.

These pin the repair: the pytest command — and only the pytest command — runs
under one declared, hardware- and cache-neutral environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit import cpu_test_env as CTE  # noqa: E402

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"
SIM = REPO / "scripts/pod/simulate_pod_env.sh"
EMITTER = REPO / "scripts/pod/cpu_test_env_args.py"


def _child(env: dict[str, str], code: str) -> str:
    """Run a probe in a child process under exactly `env`."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env).stdout.strip()


def _scoped(scope_root: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """The parent environment with the contract applied, as `env(1)` would.

    `scope_root` is the root the contract derives HOME and the HF root from, and
    both are created so the child meets the same layout a pod does.
    """
    out = dict(os.environ if base is None else base)
    for key in CTE.UNSET:
        out.pop(key, None)
    over = CTE.overlay(scope_root)
    out.update(over)
    Path(over["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(over["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    return out


# --- the contract itself -----------------------------------------------------

def test_the_contract_sets_and_clears_exactly_what_was_declared(tmp_path):
    o = CTE.overlay(tmp_path)
    assert o["CUDA_VISIBLE_DEVICES"] == ""
    # The argument is the SCOPE root: HOME is a subdirectory and the HF root a
    # SIBLING, so HOME stays genuinely empty.
    assert o["HOME"] == f"{tmp_path}/home"
    assert o["HF_HOME"] == f"{tmp_path}/hf"
    assert o["HF_HUB_CACHE"] == f"{tmp_path}/hf/hub"
    assert not o["HF_HOME"].startswith(o["HOME"] + "/")
    assert set(CTE.UNSET) == {"HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE",
                              "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"}


def test_the_token_is_preserved_and_never_a_skip_predicate(tmp_path):
    """A real pod exports a real token and the simulation a synthetic one. Both
    are non-empty; the CACHE is what is neutralized. Whether the token is real
    must never decide a skip — that guard is what cost attempt 5 a grant."""
    assert "HF_TOKEN" in CTE.PRESERVED
    assert "HF_TOKEN" not in CTE.overlay(tmp_path) and "HF_TOKEN" not in CTE.UNSET
    assert "AAD_SYNTHETIC_HF_TOKEN" not in CTE.overlay(tmp_path), (
        "the pod must never be told it holds a synthetic credential")
    kept = _scoped(tmp_path, {"HF_TOKEN": "real-token", "XDG_CACHE_HOME": "/x"})
    assert kept["HF_TOKEN"] == "real-token"
    assert "XDG_CACHE_HOME" not in kept


def test_env_args_are_command_scoped(tmp_path):
    """`env -u … K=V cmd`, not `export`. Exporting over setup would neutralize
    the teacher download and the CUDA proof that must run for real."""
    args = CTE.env_args(tmp_path)
    for key in CTE.UNSET:
        i = args.index(key)
        assert args[i - 1] == "-u", f"{key} is not cleared, it is being SET"
    assert f"HOME={tmp_path}/home" in args
    assert "CUDA_VISIBLE_DEVICES=" in args
    # `env` consumes these; nothing here exports.
    assert not any(a.startswith("export ") for a in args)


def test_both_runners_consume_the_one_declaration():
    """Not two prose lists. Two lists drifting apart IS the defect."""
    setup, sim = SETUP.read_text(), SIM.read_text()
    assert "cpu_test_env_args.py" in setup and "cpu_test_env_args.py" in sim
    assert EMITTER.read_text().count("from aadistill.autoinit.cpu_test_env import") == 1
    # The pod scopes it to one command; the simulator isolates its own subshell.
    assert "env $CPU_TEST_ENV" in setup
    assert "--format sh" in sim


def test_the_setup_gate_removes_the_temporary_home_and_reasserts_the_runtime():
    text = SETUP.read_text()
    assert "CPU_TEST_HOME=$(mktemp -d" in text
    assert 'rm -rf "$CPU_TEST_HOME"' in text
    assert "import torch; assert torch.cuda.is_available()" in text
    assert "REAL_HF_HUB=" in text and "is gone" in text
    # After the gate, before TESTS_OK.
    assert text.index("import torch; assert torch.cuda.is_available()") < \
        text.index('mark "TESTS_OK')
    # And the real teacher is never re-downloaded to prove the point.
    after = text[text.index("SUMMARY_RC=$?"):text.index('mark "TESTS_OK')]
    assert "snapshot_download" not in after and "huggingface-cli download" not in after


# --- 4. the GPU predicate ----------------------------------------------------

GPU_NODEID = ("tests/autoinit/test_causal_depth_measurement_job.py::"
              "test_the_entrypoint_refuses_a_host_run_when_no_hardware_is_injected")


def test_the_gpu_predicate_direction_is_recorded_correctly():
    """The audit prose once had this backwards. The code is unambiguous."""
    src = (REPO / "tests/autoinit/test_causal_depth_measurement_job.py").read_text()
    body = src[src.index("def test_the_entrypoint_refuses_a_host_run"):]
    assert "if torch.cuda.is_available():" in body.split("args =")[0]
    assert "pytest.skip" in body.split("args =")[0], (
        "the guard SKIPS when CUDA is present and RUNS when it is absent")


def test_cuda_is_unavailable_under_the_contract_environment(tmp_path):
    """The decision the predicate reads, taken in a child under the exact env.

    On this CPU dev box CUDA is absent anyway, so what this proves is that the
    contract does not somehow re-enable it. The claim it stands in for — that
    `CUDA_VISIBLE_DEVICES=""` hides an L40S — cannot be executed at `$0` and is
    recorded as such in the parity audit.
    """
    seen = _child(_scoped(tmp_path),
                  "import torch;print(torch.cuda.is_available())")
    assert seen == "False", seen
    assert _scoped(tmp_path)["CUDA_VISIBLE_DEVICES"] == ""


def test_the_gpu_predicate_makes_the_same_decision_under_the_contract(tmp_path):
    """The real nodeid, run twice under the exact command-scoped environment."""
    env = _scoped(tmp_path)
    env["PYTHONPATH"] = str(REPO / "src")
    outcomes = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-m", "pytest", GPU_NODEID, "-q",
                            "--no-header", "-rs"],
                           cwd=REPO, capture_output=True, text=True, env=env)
        outcomes.append("skipped" if " skipped" in r.stdout else
                        ("passed" if " passed" in r.stdout else r.stdout[-300:]))
    assert outcomes[0] == outcomes[1] == "passed", outcomes
    # And it is NOT excused by a waiver.
    reg = json.loads((REPO / "configs/autoinit/c1_skip_predicate_classification.json"
                      ).read_text())["predicates"]
    assert not any("test_causal_depth_measurement_job" in k for k in reg), (
        "the GPU predicate must be NORMALIZED by the contract, not waived")


# --- 5. the teacher cache ----------------------------------------------------

def test_the_child_sees_an_empty_cache_while_the_real_one_is_populated(tmp_path):
    """The three facts the pod needs: the real cache may be full outside the
    scope, the child sees an empty one, and the real cache survives."""
    real = tmp_path / "real_home/.cache/huggingface/hub"
    (real / "models--Qwen--Qwen3-4B-Thinking-2507/snapshots/abc").mkdir(parents=True)
    (real / "models--Qwen--Qwen3-4B-Thinking-2507/snapshots/abc/config.json"
     ).write_text("{}")
    populated = len(list(real.glob("models--*")))
    assert populated == 1

    scope = tmp_path / "cpu_test_scope"
    env = _scoped(scope, {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "real_home"),
                          "HF_TOKEN": "real-token"})
    probe = ("import os,glob;"
             "print(os.environ['HF_HUB_CACHE'], len(glob.glob(os.environ['HF_HUB_CACHE']+'/models--*')))")
    out = _child(env, probe)
    seen_root, seen_n = out.rsplit(" ", 1)
    assert seen_root == f"{scope}/hf/hub"
    assert int(seen_n) == 0, "the child can see the real cache"

    # The real cache is untouched and still visible to the normal environment.
    assert len(list(real.glob("models--*"))) == populated
    assert (real / "models--Qwen--Qwen3-4B-Thinking-2507/snapshots/abc/config.json").is_file()


def test_the_shared_hub_resolver_follows_the_contract(tmp_path):
    """`battery_render.hub_cache()` resolves at call time, so it must land inside
    the isolated root — that resolver is what the seven parity cases read."""
    scope = tmp_path / "scope"
    env = _scoped(scope, {"PATH": os.environ["PATH"], "HOME": str(tmp_path)})
    env["PYTHONPATH"] = f"{REPO / 'src'}{os.pathsep}{REPO / 'scripts/data'}"
    out = _child(env, "import battery_render as b;print(b.hub_cache())")
    assert out == f"{scope}/hf/hub", out


def test_the_host_local_store_is_located_through_home(tmp_path, monkeypatch):
    """The reason the fresh HOME can neutralize anything at all.

    A hardcoded `/home/ecs-user/aad-artifacts` is immune to it, which is what let
    host-local cases run in the diagnostic and skip on the pod.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert CTE.host_local_store() == tmp_path / "aad-artifacts"
    for module, name in (
            ("tests/pod/test_recovery_continuation_session.py", "HOST_LOCAL_PHASE_A_STORE"),
            ("tests/pod/test_reconstruct_training_events.py", "E6B"),
            ("tests/pod/test_continuation_rehearsal.py", "STAGED"),
            ("scripts/pod/autoinit_recovery_continuation_launch.py", "CKPT_STORE")):
        text = (REPO / module).read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith(f"{name} ="))
        assert "Path.home()" in line, f"{module}::{name} still hardcodes a host path"

    #: `test_c1_session_contract.py::CANDIDATE` was in that list until
    #: 2026-09-08. It is not there now because the constant is GONE: the C1
    #: candidate authorization is built deterministically into `tmp_path` from
    #: the same payload builder the CLI issuer uses, so there is no host-local
    #: store to locate through anything. That is the stronger outcome — a fresh
    #: HOME neutralizes a `Path.home()` path, but it cannot make a stale fixture
    #: describe the current tree, which is how a correct gate came to report a
    #: false alarm.
    contract = (REPO / "tests/pod/test_c1_session_contract.py").read_text()
    assert not any(ln.startswith("CANDIDATE =") for ln in contract.splitlines()), (
        "a host-local CANDIDATE constant is back in test_c1_session_contract.py; "
        "the candidate is built into tmp_path and must stay in-repo")
