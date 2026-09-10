"""A returned provider id is the boundary. The price check is not.

`SessionRunner.create()` used to evaluate the returned `costPerHr` BEFORE
registering the resource:

    if self.price > self.a.max_price:
        self.say("ABORT: provisioned at ... — deleting")
        subprocess.run([self.cli, "remove", "pod", pid], ...)   # one shot
        return False                                            # unconfirmed
    ...
    self.pod_id = pid          # never reached on that path

Everything downstream keys on `self.pod_id`. So on that path there was no
watchdog, no `teardown_now`, no `provider_confirms_gone`, no `final_pod_state`,
no cost, and nothing for `run_session`'s exception handler to tear down — while
a pod existed and was billing. The session recorded itself exactly like a `$0`
pre-provider refusal, and the one-use grant would have been reported unconsumed
after being consumed.

A `$0` refusal happens in `check_gpu_offered`, before `create` is called at all.
Once an id comes back, the money has started and the only question is who owns
the resource.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

from aadistill.infrastructure.session import ExecutionCommands  # noqa: E402
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure import session_runner as SR  # noqa: E402
from aadistill.infrastructure.provider import PodState  # noqa: E402

RUNNER_SRC = REPO / "src/aadistill/infrastructure/session_runner.py"


class _Provider:
    """Answers `get()` from a script; the last state repeats forever."""

    def __init__(self, states):
        self.states = list(states)
        self.gets = 0
        self.terminated: list[str] = []

    def get(self, pod_id):
        st = self.states[min(self.gets, len(self.states) - 1)]
        self.gets += 1
        #: `billing` is DERIVED on the real PodState, not a field — unknown
        #: counts as billing, which is the direction that does not lose money.
        #: The fake builds real states and lets the production property decide.
        return PodState(pod_id=pod_id, exists=st[0], desired_status=st[1])

    def terminate(self, pod_id):
        self.terminated.append(pod_id)
        return []


GONE = (False, "TERMINATED")       # exists False -> billing False
BILLING = (True, "RUNNING")        # RUNNING is not a GONE status


def _runner(tmp_path, monkeypatch, *, returned_price, max_price=1.09,
            states=(GONE,), create_ok=True, confirm=True):
    """The REAL `create()` and `teardown_now()`, with the provider faked.

    Built without `__init__`, which verifies a harness digest against a live
    authorization — not what is under test here.
    """
    r = object.__new__(SR.SessionRunner)
    r.provider = _Provider(states)
    r.cli = "runpodctl"
    r.repo_root = REPO
    r.pod_id = ""
    r.price = 0.0
    r.start_epoch = 0.0
    r.ev = {}
    r.scr = tmp_path / "scr"
    r.scr.mkdir(parents=True, exist_ok=True)
    r.say = lambda m: r.ev.setdefault("said", []).append(m)
    r.save = lambda: r.ev.setdefault("saves", []).append(dict(r.ev))
    r._watchdog_for = ""
    r.plan = types.SimpleNamespace(hard_terminate_minutes=834.0,
                                   hard_terminate_usd=15.1475)
    r.auth = types.SimpleNamespace(hard_cap_usd=15.1475)
    r.spec = types.SimpleNamespace(
        #: The runner reads its executables from the spec now, so a stub spec
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
        session_id="autoinit-c1",
        teardown=types.SimpleNamespace(require_provider_confirmation=confirm))
    r.a = types.SimpleNamespace(
        create_attempts=1, host_draws=1, create_retry_seconds=300.0,
        gpu="NVIDIA L40S", max_price=max_price, image="img", disk_gb=200,
        out="logs/unused.json")

    calls = {"create": 0, "remove": 0, "slept": [], "watchdogs": []}

    class _Popen:
        def __init__(self, cmd, **kw):
            calls["watchdogs"].append(cmd)

    def fake_run(cmd, **kw):
        if "create" in cmd:
            calls["create"] += 1
            body = (json.dumps({"id": "pod1", "costPerHr": returned_price})
                    if create_ok else "error: no capacity available")
            return subprocess.CompletedProcess(cmd, 0, body, "")
        if "remove" in cmd:
            calls["remove"] += 1
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(SR.subprocess, "run", fake_run)
    monkeypatch.setattr(SR.subprocess, "Popen", _Popen)
    monkeypatch.setattr(SR.time, "sleep", lambda s: calls["slept"].append(s))
    return r, calls


# --- A. pre-query accepted, RETURNED price rejected -------------------------

def test_A_an_over_price_resource_is_owned_before_it_is_rejected(tmp_path,
                                                                 monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.10, max_price=1.09)

    assert r.create() is False

    # exactly one create, no redraw, no create-retry backoff. The 10 s sleeps
    # that DO appear are the teardown confirmation loop, which is the point.
    assert calls["create"] == 1
    assert r.a.create_retry_seconds not in calls["slept"]
    assert set(calls["slept"]) <= {10}

    # the resource was REGISTERED before rejection
    assert r.pod_id == "pod1"
    assert r.ev["pod_id"] == "pod1"
    assert r.start_epoch > 0
    assert (r.scr / "pod_id").read_text() == "pod1"
    assert (r.scr / "pod_start_epoch").exists()
    assert r.ev["actual_price_per_hour"] == 1.10

    # ...and the grant is treated as consumed
    assert r.ev["provider_resource_created"] is True
    assert r.ev["one_use_grant_consumed"] is True

    # classified as a POST-provider abort, explicitly not a $0 refusal
    abort = r.ev["post_provider_abort"]
    assert abort["is_zero_dollar_pre_provider_refusal"] is False
    assert abort["setup_started"] is False and abort["driver_started"] is False
    assert abort["retried"] is False and abort["replaced"] is False
    assert abort["returned_price_per_hour"] == 1.10

    # canonical teardown ran and confirmed
    assert calls["remove"] == 1
    assert r.ev["provider_confirms_gone"] is True
    assert r.ev["final_pod_state"]["billing"] is False
    assert "cost" in r.ev
    # the id survives termination — an abort that erases its own evidence
    # cannot be ledgered
    assert r.ev["pod_id"] == "pod1"


def test_A2_registration_is_persisted_before_the_price_is_judged(tmp_path,
                                                                 monkeypatch):
    """The save must precede the rejection, or a crash in teardown loses it."""
    r, _ = _runner(tmp_path, monkeypatch, returned_price=1.10)
    r.create()
    first = r.ev["saves"][0]
    assert first["pod_id"] == "pod1"
    assert first["provider_resource_created"] is True
    assert "post_provider_abort" not in first, (
        "the resource must be recorded as owned BEFORE it is rejected")


def test_A3_neither_setup_nor_the_driver_is_reached(tmp_path, monkeypatch):
    """The watchdog IS expected here — it is the backstop for a pod that exists.
    What must not run is anything that spends more: setup and the driver."""
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.10)
    monkeypatch.setattr(SR.SessionRunner, "setup_on_draw",
                        lambda self, draw: pytest.fail("setup ran"))
    assert r.create() is False
    assert len(calls["watchdogs"]) == 1
    assert r.ev["post_provider_abort"]["setup_started"] is False
    assert r.ev["post_provider_abort"]["driver_started"] is False


def test_A4_run_stops_without_a_second_resource(tmp_path, monkeypatch):
    """End to end through the REAL acquisition loop."""
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.10)
    r.make_plan = lambda: True
    r.run_prechecks = lambda: True
    monkeypatch.setattr(SR.SessionRunner, "setup_on_draw",
                        lambda self, draw: pytest.fail("setup ran"))

    assert r.run() is False
    assert calls["create"] == 1
    assert "redrawing" not in " ".join(r.ev.get("said", []))


# --- B. teardown NOT confirmed ----------------------------------------------

def test_B_an_unconfirmed_teardown_keeps_the_evidence(tmp_path, monkeypatch):
    """Still billing after the confirmation loop. The record must say so."""
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.10,
                       states=(BILLING,))

    assert r.create() is False

    assert r.ev["pod_id"] == "pod1", "the pod must stay visible in evidence"
    assert r.ev["provider_confirms_gone"] is False
    assert r.ev["final_pod_state"]["billing"] is True
    assert r.ev["final_pod_state"]["desired_status"] == "RUNNING"
    assert r.ev["cost"]["price_per_hour"] == 1.10
    assert r.ev["provider_resource_created"] is True
    assert calls["create"] == 1, "no second resource after a failed confirmation"
    assert r.ev["post_provider_abort"]["is_zero_dollar_pre_provider_refusal"] is False


def test_B2_the_confirmation_loop_polls_the_provider(tmp_path, monkeypatch):
    """Fake clock, not real sleeping: the loop's shape is what is asserted."""
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.10,
                       states=(BILLING,))
    r.create()
    assert r.provider.gets == 18, "the full confirmation budget must be spent"
    assert calls["slept"] and all(s == 10 for s in calls["slept"])


def test_B3_confirmation_stops_as_soon_as_billing_ends(tmp_path, monkeypatch):
    r, _ = _runner(tmp_path, monkeypatch, returned_price=1.10,
                   states=(BILLING, GONE))
    r.create()
    assert r.provider.gets == 2
    assert r.ev["provider_confirms_gone"] is True


# --- C. the accepted price path is unchanged --------------------------------

def test_C_an_accepted_price_registers_one_resource_and_proceeds(tmp_path,
                                                                 monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.09, max_price=1.09)

    assert r.create() is True
    assert calls["create"] == 1
    assert calls["remove"] == 0
    assert r.pod_id == "pod1"
    assert r.ev["actual_price_per_hour"] == 1.09
    assert r.ev["provider_resource_created"] is True
    assert "post_provider_abort" not in r.ev


def test_C2_the_happy_path_launches_the_watchdog_and_enters_setup(tmp_path,
                                                                  monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.09)
    r.make_plan = lambda: True
    r.run_prechecks = lambda: True
    seen: list[str] = []
    monkeypatch.setattr(SR.SessionRunner, "launch_watchdog",
                        lambda self: (seen.append("watchdog"), Path("/dev/null"))[1])
    monkeypatch.setattr(SR.SessionRunner, "setup_on_draw",
                        lambda self, draw: (seen.append("setup"), "cold")[1])

    r.run()
    assert seen == ["watchdog", "setup"]
    assert calls["create"] == 1, "no second resource on the happy path either"


# --- D. the no-ID create failure is unchanged -------------------------------

def test_D_a_create_failure_is_still_a_zero_dollar_pre_provider_refusal(
        tmp_path, monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.09, create_ok=False)

    assert r.create() is False
    assert calls["create"] == 1
    assert calls["remove"] == 0
    assert calls["slept"] == []
    assert r.pod_id == ""
    assert "pod_id" not in r.ev
    assert "provider_resource_created" not in r.ev
    assert "one_use_grant_consumed" not in r.ev
    assert "post_provider_abort" not in r.ev
    assert not r.provider.terminated
    assert "provider_confirms_gone" not in r.ev


# --- the shape that made the defect possible --------------------------------

def test_there_is_no_raw_unconfirmed_remove_inside_create():
    """One deletion owner. A second, C1-only implementation is what the review
    forbade, and a raw `remove` returning before registration is what existed."""
    src = RUNNER_SRC.read_text()
    body = src.split("def create(self)")[1].split("def launch_watchdog")[0]
    assert '"remove", "pod", pid' not in body
    assert "self.teardown_now(" in body


def test_registration_precedes_the_price_check_in_source_order():
    src = RUNNER_SRC.read_text()
    body = src.split("def create(self)")[1].split("def launch_watchdog")[0]
    assert body.index("self.pod_id = pid") < body.index("self.save()") \
        < body.index("if self.price > self.a.max_price:")


# --- the watchdog is established at registration, exactly once --------------
#
# The backstop used to start in `run()`, AFTER `create()` returned True. So on
# the over-price path — where `create()` returns False — no watchdog ever
# existed, and if this process had died during teardown the pod would have
# billed to the provider's own `--terminate-after` and nothing else.

def test_the_watchdog_starts_at_registration_before_the_price_is_judged(
        tmp_path, monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.10)

    assert r.create() is False

    assert len(calls["watchdogs"]) == 1, "exactly one backstop for one resource"
    cmd = calls["watchdogs"][0]
    assert "--pod-id" in cmd and cmd[cmd.index("--pod-id") + 1] == "pod1"
    assert r.ev["watchdog_owns_pod"] == "pod1"
    assert r.ev["watchdog_journals"] == [str(r.scr / "watchdog.jsonl")]
    # started BEFORE the rejection was recorded
    first = r.ev["saves"][0]
    assert first["watchdog_owns_pod"] == "pod1"
    assert "post_provider_abort" not in first


def test_the_watchdog_survives_an_unconfirmed_teardown(tmp_path, monkeypatch):
    """It is the hard-cap backstop precisely when confirmation fails."""
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.10,
                       states=(BILLING,))
    r.create()
    assert len(calls["watchdogs"]) == 1
    assert r.ev["provider_confirms_gone"] is False
    assert r.ev["watchdog_owns_pod"] == "pod1"
    assert r.ev["pod_id"] == "pod1"


def test_the_accepted_path_starts_exactly_one_watchdog(tmp_path, monkeypatch):
    """`run()` no longer starts a second one after `create()` succeeds."""
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.09)
    r.make_plan = lambda: True
    r.run_prechecks = lambda: True
    monkeypatch.setattr(SR.SessionRunner, "setup_on_draw",
                        lambda self, draw: "cold")

    r.run()
    assert len(calls["watchdogs"]) == 1, (
        f"{len(calls['watchdogs'])} watchdogs for one provider resource")
    assert calls["create"] == 1


def test_launch_watchdog_is_idempotent_for_one_resource(tmp_path, monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.09)
    assert r.create() is True
    assert len(calls["watchdogs"]) == 1
    r.launch_watchdog()
    r.launch_watchdog()
    assert len(calls["watchdogs"]) == 1
    assert r.ev["watchdog_journals"] == [str(r.scr / "watchdog.jsonl")]


def test_a_new_provider_resource_gets_its_own_watchdog(tmp_path, monkeypatch):
    """Idempotence is per RESOURCE, not per session: a legitimate redraw in
    another phase must still get a backstop."""
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.09)
    r.create()
    assert len(calls["watchdogs"]) == 1
    r.pod_id = "pod2"                     # as a redraw would leave it
    r.launch_watchdog()
    assert len(calls["watchdogs"]) == 2
    assert r.ev["watchdog_owns_pod"] == "pod2"


def test_a_create_failure_starts_no_watchdog(tmp_path, monkeypatch):
    r, calls = _runner(tmp_path, monkeypatch, returned_price=1.09, create_ok=False)
    assert r.create() is False
    assert calls["watchdogs"] == []
    assert "watchdog_owns_pod" not in r.ev
    assert "watchdog_journals" not in r.ev


def test_run_no_longer_starts_the_watchdog_itself():
    """One call site. Two would race to terminate the same pod."""
    src = RUNNER_SRC.read_text()
    run_body = src.split("    def run(self) -> bool:")[1]
    assert "self.launch_watchdog()" not in run_body.split("def ")[0]
    create_body = src.split("def create(self)")[1].split("def launch_watchdog")[0]
    assert "self.launch_watchdog()" in create_body
