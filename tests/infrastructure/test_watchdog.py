"""The cost backstop must act on provider state and verify that it worked.

Two E6b failures are pinned here. RunPod's `--terminate-after` was the
documented last-resort layer since E4 and has never been observed to fire: on
2026-08-08 the deadline was 00:28:47 and the pod was `RUNNING` at 00:34. And the
watcher tailed the orchestrator log, so a launcher that had blocked and stopped
writing looked identical to a session that had finished.

So: terminate against the provider, confirm the pod is gone, retry when it is
not, journal every attempt — and never let log silence produce an idle verdict.
"""

import json

import pytest

from aadistill.infrastructure.provider import PodState, SimulatedProvider
from aadistill.infrastructure.watchdog import (
    GONE, LIVE, LIVE_STALLED, LIVE_UNOBSERVED, PROVIDER_UNKNOWN,
    Journal, MarkerSnapshot, SessionWatcher, Watchdog, WatchdogPolicy,
)


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make(tmp_path, provider, *, hard_minutes=60.0, start_offset_min=0.0,
         **policy_kwargs):
    clock = FakeClock()
    policy = WatchdogPolicy(
        pod_id="pod-1",
        session_start_epoch=clock.now - start_offset_min * 60,
        price_per_hour=0.99,
        hard_terminate_minutes=hard_minutes,
        authorized_usd=9.0,
        poll_seconds=60.0,
        verify_delay_seconds=0.0,
        **policy_kwargs,
    )
    journal = Journal(tmp_path / "watchdog.jsonl")
    dog = Watchdog(provider, policy, journal, clock=clock, sleep=lambda s: None)
    return dog, clock, journal


def test_under_the_limit_it_only_watches(tmp_path):
    provider = SimulatedProvider("pod-1")
    dog, _, journal = make(tmp_path, provider, start_offset_min=10)
    obs = dog.tick()
    assert obs.action == "none"
    assert provider.calls == ["get"], "no termination below the hard limit"
    assert [r["event"] for r in journal.records()] == ["poll"]


def test_past_the_limit_it_terminates_and_verifies(tmp_path):
    provider = SimulatedProvider("pod-1")
    dog, _, journal = make(tmp_path, provider, hard_minutes=60,
                           start_offset_min=61)
    obs = dog.tick()

    assert obs.action == "verified_gone"
    assert provider.exists is False
    events = [r["event"] for r in journal.records()]
    assert "hard_limit_reached" in events
    assert "terminate_attempt" in events
    assert "terminate_verify" in events, (
        "a termination that is not verified is exactly --terminate-after")
    assert "terminated" in events


def test_a_termination_that_is_accepted_and_ignored_is_retried(tmp_path):
    """`--terminate-after` fired in the sense that a deadline existed."""
    provider = SimulatedProvider("pod-1", ignored_terminations=2)
    dog, _, journal = make(tmp_path, provider, hard_minutes=60,
                           start_offset_min=61, verify_polls=2)
    obs = dog.tick()

    assert obs.action == "verified_gone"
    rounds = [r for r in journal.records() if r["event"] == "terminate_attempt"]
    assert len(rounds) == 3, (
        "the watchdog must keep going while the pod keeps running, however "
        "cheerfully the API answered")
    assert all(r["any_ok"] for r in rounds[:2]), (
        "the ignored calls reported success; that is the point")


def test_a_failing_termination_call_is_retried_and_recorded(tmp_path):
    provider = SimulatedProvider("pod-1", terminate_failures=2)
    dog, _, journal = make(tmp_path, provider, hard_minutes=60,
                           start_offset_min=61, verify_polls=1)
    assert dog.tick().action == "verified_gone"
    attempts = [r for r in journal.records() if r["event"] == "terminate_attempt"]
    assert len(attempts) == 3
    assert attempts[0]["attempts"][0]["error"].startswith("SimulatedError")


def test_a_pod_that_will_not_die_escalates_rather_than_going_quiet(tmp_path):
    provider = SimulatedProvider("pod-1", ignored_terminations=999)
    dog, _, journal = make(tmp_path, provider, hard_minutes=60,
                           start_offset_min=61, terminate_rounds=2,
                           verify_polls=1)
    obs = dog.tick()
    assert obs.action == "termination_failed"
    events = [r["event"] for r in journal.records()]
    assert "TERMINATION_FAILED" in events, (
        "a backstop that gives up silently is not a backstop")


def test_a_provider_error_does_not_read_as_a_gone_pod(tmp_path):
    """An unanswered poll fails toward 'still billing', which is the safe side."""
    provider = SimulatedProvider("pod-1", poll_errors=1)
    dog, _, _ = make(tmp_path, provider, hard_minutes=60, start_offset_min=61)
    obs = dog.tick()
    assert obs.state.error is not None
    assert obs.state.billing is True
    assert obs.action == "verified_gone", (
        "the error poll must still lead to a termination round")
    assert provider.exists is False


def test_the_meter_starts_at_the_session_not_the_current_pod(tmp_path):
    """A cold-host redraw replaces the pod; it must not reset the budget."""
    provider = SimulatedProvider("pod-1")
    dog, _, _ = make(tmp_path, provider, hard_minutes=60, start_offset_min=45)
    assert dog.poll().over_hard_limit is False
    dog.policy = WatchdogPolicy(
        **{**dog.policy.__dict__,
           "session_start_epoch": dog.policy.session_start_epoch - 20 * 60})
    assert dog.poll().over_hard_limit is True


def test_accrued_cost_tracks_real_age(tmp_path):
    provider = SimulatedProvider("pod-1")
    dog, _, _ = make(tmp_path, provider, hard_minutes=600, start_offset_min=120)
    obs = dog.poll()
    assert obs.elapsed_minutes == pytest.approx(120.0)
    assert obs.accrued_usd == pytest.approx(2 * 0.99, abs=1e-6)


def test_run_exits_when_the_pod_is_gone(tmp_path):
    provider = SimulatedProvider("pod-1", exists=False)
    dog, _, journal = make(tmp_path, provider)
    assert dog.run(max_ticks=5) == "pod_gone"
    assert journal.records()[-1]["reason"] == "pod_gone"


def test_the_journal_is_append_only_jsonl(tmp_path):
    provider = SimulatedProvider("pod-1")
    dog, _, journal = make(tmp_path, provider, hard_minutes=60,
                           start_offset_min=61)
    dog.tick()
    dog.tick()
    lines = journal.path.read_text().splitlines()
    assert len(lines) > 2
    for line in lines:
        json.loads(line)          # every line stands alone


# --------------------------------------------------------------------------
# Watcher correctness — silence is not evidence
# --------------------------------------------------------------------------

def test_a_silent_orchestrator_over_a_live_pod_is_billing_not_idle():
    """The E6b monitoring gap, stated as an assertion."""
    watcher = SessionWatcher()
    verdict = watcher.assess(
        PodState("pod-1", exists=True, desired_status="RUNNING"),
        MarkerSnapshot(last_marker=None, age_seconds=None,
                       orchestrator_log_age_seconds=7 * 3600))
    assert verdict.state == LIVE_UNOBSERVED
    assert verdict.billing is True
    assert "blocked" in verdict.reason


def test_stalled_markers_over_a_live_pod_are_billing():
    watcher = SessionWatcher(stall_seconds=1800)
    verdict = watcher.assess(
        PodState("pod-1", exists=True, desired_status="RUNNING"),
        MarkerSnapshot(last_marker="TRAIN_DONE:arm-a", age_seconds=4000))
    assert verdict.state == LIVE_STALLED
    assert verdict.billing is True


def test_advancing_markers_over_a_live_pod_are_healthy():
    watcher = SessionWatcher()
    verdict = watcher.assess(
        PodState("pod-1", exists=True, desired_status="RUNNING"),
        MarkerSnapshot(last_marker="TRAIN_DONE:arm-a", age_seconds=30))
    assert verdict.state == LIVE
    assert verdict.billing is True


def test_only_the_provider_can_say_the_session_stopped():
    watcher = SessionWatcher()
    gone = watcher.assess(PodState("pod-1", exists=False,
                                   desired_status="TERMINATED"),
                          MarkerSnapshot(last_marker="ALL_DONE", age_seconds=5))
    assert gone.state == GONE and gone.billing is False

    unknown = watcher.assess(
        PodState("pod-1", exists=True, error="URLError: timed out"),
        MarkerSnapshot(last_marker="ALL_DONE", age_seconds=5))
    assert unknown.state == PROVIDER_UNKNOWN and unknown.billing is True


def test_there_is_no_verdict_reachable_without_provider_state():
    """Structural: `assess` cannot be called with markers alone."""
    watcher = SessionWatcher()
    with pytest.raises(TypeError):
        watcher.assess(MarkerSnapshot(last_marker=None))  # type: ignore[arg-type]
