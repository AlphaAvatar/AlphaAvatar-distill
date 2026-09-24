"""No created billing resource may outlive its owner.

The failure this pins is attempt14's, exactly. The artifact collector exited 5
because a required pattern matched nothing — that session was the first in its
campaign to train no probe, so the per-probe records directory it declared as
required did not exist. The teardown gate then refused to delete a pod whose
required artifact was only on it, which is correct, and the launcher returned,
which is also correct on its own. Together they left an L40S billing for about
116 minutes with nobody to stop it.

The old code said "the watchdog remains the backstop" and checked nothing. In
attempt14 the watchdog happened to be alive, so the sentence happened to be
true; had it died, that sentence was the only thing between a blocked gate and
an unbounded bill.

The invariant now enforced: every created billing resource has an owner until
the provider confirms it gone. Preserving a pod for its evidence stays the
preferred outcome — but only while an owner is VERIFIED, and pid liveness alone
is not ownership, because pids are reused.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.session_runner import SessionRunner  # noqa: E402


class _Runner:
    """Only the state `verify_watchdog_owns_pod` reads, and nothing else.

    Constructed without `SessionRunner.__init__` on purpose: the method under
    test must not depend on a live provider, a spec or a pod.
    """

    def __init__(self, *, pod_id, pid, launched_for=None, hard=237.29):
        self.pod_id = pod_id
        self._watchdog_pid = pid
        self._watchdog_for = pod_id if launched_for is None else launched_for
        self.plan = type("P", (), {"hard_terminate_minutes": hard})()

    verify_watchdog_owns_pod = SessionRunner.verify_watchdog_owns_pod


def test_a_live_watchdog_naming_this_pod_IS_ownership():
    """The positive case, with a real process. attempt14's situation.

    Without this the suite would be satisfied by a verifier that returned
    `owned=False` unconditionally -- and that would not be a safety
    improvement, it would tear down every blocked session's pod and destroy
    exactly the evidence the gate exists to preserve.

    A real child is spawned whose cmdline carries both tokens the verifier
    requires, because a fake that returned a string would be testing the
    fixture rather than `/proc`.
    """
    import subprocess
    import time

    pod = "6rwnh4rd9woxt7"
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import time; time.sleep(30)  # watchdog --pod-id {pod}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):          # let /proc settle
            if Path(f"/proc/{proc.pid}/cmdline").exists():
                break
            time.sleep(0.02)
        out = _Runner(pod_id=pod, pid=proc.pid).verify_watchdog_owns_pod()
        assert out["owned"] is True, out
        assert out["pid"] == proc.pid
        assert pod in out["why"]
        assert out["hard_minutes"] == pytest.approx(237.29)
    finally:
        proc.kill()
        proc.wait()

    #: And once it dies, ownership is gone -- the same runner, the same pid.
    after = _Runner(pod_id=pod, pid=proc.pid).verify_watchdog_owns_pod()
    assert after["owned"] is False, after


def test_a_dead_watchdog_is_not_ownership():
    """A pid that is not running cannot terminate anything."""
    #: A pid that is certainly free: allocate and reap a child.
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)

    out = _Runner(pod_id="somepod", pid=pid).verify_watchdog_owns_pod()
    assert out["owned"] is False
    assert "not running" in out["why"] or "unreadable" in out["why"], out


def test_no_watchdog_at_all_is_not_ownership():
    out = _Runner(pod_id="somepod", pid=0).verify_watchdog_owns_pod()
    assert out["owned"] is False
    assert "no watchdog was launched" in out["why"]


def test_a_watchdog_launched_for_a_DIFFERENT_pod_is_not_ownership():
    """A redraw leaves the previous resource's watchdog behind.

    If a second pod is created and its watchdog fails to launch, the stale
    handle from the first would otherwise read as ownership of the second.
    """
    out = _Runner(pod_id="pod_B", pid=os.getpid(),
                  launched_for="pod_A").verify_watchdog_owns_pod()
    assert out["owned"] is False
    assert "pod_A" in out["why"] and "pod_B" in out["why"]


def test_pid_reuse_does_not_count_as_ownership():
    """Liveness alone is not ownership.

    `os.kill(pid, 0)` succeeds for ANY live process, so a recycled pid would
    report an owner that does not exist. The cmdline must still name this pod
    and be a watchdog.
    """
    out = _Runner(pod_id="6rwnh4rd9woxt7", pid=os.getpid()).verify_watchdog_owns_pod()
    assert out["owned"] is False
    assert "reused" in out["why"] or "not this pod's watchdog" in out["why"], out


def test_no_pod_means_nothing_to_own():
    """A $0 pre-provider refusal created no resource, so there is no owner to
    demand. Reporting that as unowned would make every free refusal look like
    an orphan."""
    out = _Runner(pod_id="", pid=0).verify_watchdog_owns_pod()
    assert out["owned"] is True
    assert "nothing was created" in out["why"]


def test_the_blocked_path_tears_down_when_it_cannot_verify_an_owner():
    """The repair, read off the source of the path attempt14 took.

    Asserted against the code rather than by driving a provider: the branch
    exists, it calls the verifier, and the no-owner leg calls `teardown_now`
    instead of returning with a live resource.
    """
    import inspect

    src = inspect.getsource(SessionRunner.collect_and_teardown)
    assert "verify_watchdog_owns_pod()" in src, (
        "the blocked-teardown path does not verify ownership")
    blocked = src[src.index("if not decision.allowed:"):]
    assert "teardown_now(" in blocked, (
        "the blocked path can still return with an unowned billing resource")
    assert "watchdog remains the backstop" not in src, (
        "the unverified assertion is still there")
    #: And the verified leg must still PREFER preserving the pod, or the repair
    #: would have traded artifact loss for bill safety instead of fixing the
    #: combination.
    assert "retained for evidence" in blocked
