"""The C1 grant permits ONE provider resource. The launcher must be unable to
create a second.

The grant says it plainly: one issuance, one launch attempt, one provider
resource, and "after consumption there is NO retry and NO replacement pod". The
launcher did not enforce any of it. It defaulted to `--create-attempts 8` and
`--host-draws 3`, so:

* a cold or endpoint-less first pod was deleted and a SECOND one drawn — a
  replacement pod, which is exactly the thing the grant forbids;
* a create failure slept `--create-retry-seconds` (300 s) and tried again, up to
  seven more times, waiting on the stock the grant says not to chase.

Neither would have been visible in a launch transcript as a violation; both look
like ordinary resilience. That is why this is enforced by TYPE and by spec
construction rather than by a default a future launch command has to remember.

`SessionRunner` is untouched. Phase A, Phase B, both continuations and the
preflight keep multi-draw acquisition, which is correct for them: they have no
one-resource grant.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
import types

from aadistill.infrastructure.session import ExecutionCommands  # noqa: E402
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))

from aadistill.infrastructure import session_runner as SR  # noqa: E402
from aadistill.infrastructure.provider import PodState  # noqa: E402

LAUNCHER = REPO / "scripts/pod/autoinit_c1_launch.py"


def _launcher():
    spec = importlib.util.spec_from_file_location("c1_launch_one", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["c1_launch_one"] = mod
    spec.loader.exec_module(mod)
    return mod


C1 = _launcher()
#: `--run-id` is required by the launcher: a run is named before it is launched,
#: and no test builds an argv the operator could not type. Nothing here opens a
#: run — these tests read `create_attempts`/`host_draws` off the parsed namespace
#: and drive `SessionRunner` directly — so the id names no directory.
BASE_ARGV = ["--scr", "/tmp/c1-one-resource", "--session-commit", "0" * 40,
             "--bundle", "aad_test.bundle", "--run-id", "one_resource"]


# --- A. the defaults realize one resource -----------------------------------

def test_A_the_c1_defaults_are_one_create_attempt_and_one_host_draw():
    args = C1.build_parser().parse_args(BASE_ARGV)
    assert args.create_attempts == 1
    assert args.host_draws == 1
    assert C1.C1_PROVIDER_RESOURCES == 1


def test_A2_the_spec_realizes_them_too():
    """The value the RUNNER will read, not just the one the parser produced."""
    args = C1.build_parser().parse_args(BASE_ARGV)
    C1.spec(args)                       # must not raise
    assert args.create_attempts == 1 and args.host_draws == 1


# --- B. anything else is refused --------------------------------------------

@pytest.mark.parametrize("flag,value", [
    ("--create-attempts", "2"), ("--create-attempts", "8"),
    ("--host-draws", "2"), ("--host-draws", "3"),
])
def test_B_the_parser_refuses_more_than_one(flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        C1.build_parser().parse_args([*BASE_ARGV, flag, value])
    assert exc.value.code == 2
    assert "more than one provider resource" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["create_attempts", "host_draws"])
def test_B2_spec_construction_refuses_a_hand_built_namespace(field):
    """A namespace is not a parser. Device-canary attempt 1 died at $0.0603 on
    exactly that difference, so the rule is enforced in both places."""
    args = C1.build_parser().parse_args(BASE_ARGV)
    setattr(args, field, 3)
    with pytest.raises(SystemExit, match="exactly one provider resource"):
        C1.spec(args)


def test_B3_zero_is_refused_too():
    """One, not 'at most one' — a session that creates nothing is a bug."""
    with pytest.raises(SystemExit):
        C1.build_parser().parse_args([*BASE_ARGV, "--host-draws", "0"])


# --- C/D/E. what the runner actually does with those values -----------------

class _CountingProvider:
    def __init__(self):
        self.terminated: list[str] = []

    def get(self, pod_id):
        return PodState(pod_id=pod_id, exists=True, desired_status="RUNNING")

    def terminate(self, pod_id):
        self.terminated.append(pod_id)
        return []


def _runner(monkeypatch, *, outcome, create_ok=True):
    """The REAL `SessionRunner.run` acquisition loop, with only the provider
    boundary faked. Everything between `create()` and the draw decision is
    production code."""
    r = object.__new__(SR.SessionRunner)
    r.provider = _CountingProvider()
    r.cli = "runpodctl"
    r.pod_id = ""
    r.price = 1.09
    r.start_epoch = 0.0
    r.ev = {}
    r.scr = Path("/tmp/c1-one-resource-scr")
    r.scr.mkdir(parents=True, exist_ok=True)
    r.say = lambda m: r.ev.setdefault("said", []).append(m)
    r.save = lambda: None
    r.make_plan = lambda: True
    r.run_prechecks = lambda: True
    r.launch_watchdog = lambda: Path("/dev/null")
    r.teardown_now = lambda why: r.ev.setdefault("teardown", []).append(why)
    r.plan = types.SimpleNamespace(hard_terminate_minutes=834.0)
    r.spec = types.SimpleNamespace(        #: The runner reads its executables from the spec now, so a stub spec
        #: must declare them; there is no default for it to fall back on.
        #: The REAL type, not a SimpleNamespace. A fake that names the fields
        #: it happens to need goes stale silently the moment the type gains
        #: one -- which is exactly what happened when `workspace_root`,
        #: `checkout_root` and `min_cuda_version` were added: twenty tests
        #: failed with AttributeError on a double, not on the code.
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            remote_python="/opt/train/bin/python",
            workspace_root="/workspace", checkout_root="/workspace/aad",
            min_cuda_version="13.0"),
session_id="autoinit-c1")

    args = C1.build_parser().parse_args(BASE_ARGV)
    r.a = types.SimpleNamespace(
        create_attempts=args.create_attempts, host_draws=args.host_draws,
        create_retry_seconds=args.create_retry_seconds,
        gpu="NVIDIA L40S", max_price=1.09, image="img", disk_gb=200)

    created: list[int] = []

    def fake_run(cmd, **kw):
        if "create" in cmd:
            created.append(1)
            out = ('{"id":"pod%d","costPerHr":1.09}' % len(created)
                   if create_ok else "error: no capacity")
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if "remove" in cmd:
            r.provider.terminated.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(SR.subprocess, "run", fake_run)
    monkeypatch.setattr(SR.time, "sleep",
                        lambda s: r.ev.setdefault("slept", []).append(s))
    monkeypatch.setattr(SR.SessionRunner, "setup_on_draw",
                        lambda self, draw: outcome)
    return r, created


def test_C_a_cold_first_resource_creates_one_and_tears_it_down(monkeypatch):
    r, created = _runner(monkeypatch, outcome="cold")

    assert r.run() is False
    assert len(created) == 1, f"created {len(created)} provider resources"
    assert r.ev.get("teardown") == ["setup cold"]
    assert "redrawing" not in " ".join(r.ev.get("said", []))
    assert not r.ev.get("slept"), "C1 must not sleep against stock"


def test_D_a_no_endpoint_first_resource_does_the_same(monkeypatch):
    r, created = _runner(monkeypatch, outcome="no_endpoint")

    assert r.run() is False
    assert len(created) == 1
    assert r.ev.get("teardown") == ["setup no_endpoint"]
    assert "redrawing" not in " ".join(r.ev.get("said", []))


def test_E_a_create_failure_calls_create_once_and_returns_for_review(monkeypatch):
    """No id was returned, so no resource exists and nothing is billing."""
    r, created = _runner(monkeypatch, outcome="ok", create_ok=False)

    assert r.run() is False
    assert len(created) == 1, "C1 must not retry provider creation"
    assert not r.ev.get("slept"), "C1 must not wait on changing stock"
    assert r.pod_id == ""
    assert not r.provider.terminated, "nothing was created, so nothing to remove"
    assert not r.ev.get("teardown")


def test_C2_any_other_setup_failure_also_ends_the_session(monkeypatch):
    r, created = _runner(monkeypatch, outcome="tests_failed")
    assert r.run() is False
    assert len(created) == 1
    assert r.ev.get("teardown") == ["setup tests_failed"]


# --- F. the generic runner keeps multi-draw for everyone else ---------------

def test_F_the_generic_runner_still_supports_multiple_draws():
    """This repair is C1-specific. Weakening `SessionRunner` would remove
    acquisition resilience from five other sessions that have no one-resource
    grant."""
    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert "for draw in range(1, self.a.host_draws + 1)" in src
    assert "for attempt in range(1, self.a.create_attempts + 1)" in src


@pytest.mark.parametrize("launcher", [
    "autoinit_phase_a_launch", "autoinit_phase_b_launch",
    "autoinit_continuation_launch", "autoinit_preflight_launch",
])
def test_F2_other_launchers_are_untouched(launcher):
    from session_specs import load_session_launcher

    mod = load_session_launcher(launcher)
    argv = ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"]
    if launcher == "autoinit_continuation_launch":
        argv += ["--transport", "relay"]
    args = mod.build_parser().parse_args(argv)
    assert args.host_draws > 1, f"{launcher} lost multi-draw acquisition"
    assert args.create_attempts > 1


# --- the redraw branch is unreachable, not merely unused ---------------------

def test_the_redraw_branch_cannot_be_reached_with_one_draw():
    """`draw < host_draws` is `1 < 1`. Stated as arithmetic, because that is the
    whole mechanism — there is no separate C1 copy of the acquisition loop."""
    host_draws = C1.build_parser().parse_args(BASE_ARGV).host_draws
    assert host_draws == 1
    assert not (1 < host_draws), "the cold/no_endpoint redraw branch is reachable"


def test_the_grant_and_the_launcher_agree_on_one_resource():
    import json

    grant = json.loads(
        (REPO / "logs/autoinit_c1_attempt9_grant.json").read_text())
    assert grant["one_use"]["provider_resources_permitted"] == \
        C1.C1_PROVIDER_RESOURCES == 1
    assert grant["one_use"]["launch_attempts_permitted"] == 1
