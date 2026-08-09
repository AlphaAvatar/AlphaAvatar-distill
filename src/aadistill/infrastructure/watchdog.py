"""An independent budget watchdog, and a watcher that cannot mistake silence.

Two E6b failures live here.

**The backstop that was never observed to fire.** RunPod's `--terminate-after`
has been the documented last-resort cost layer since E4 and has never once been
seen to act. On 2026-08-08 the deadline was 00:28:47 and the pod was still
`RUNNING` at 00:34. It is now demoted to a redundant third layer; the layer that
is *supposed* to work is `Watchdog`, which polls the provider itself, decides
against its own clock, terminates, and — the part `--terminate-after` skipped —
**verifies the pod actually disappeared**, retrying until it has.

**The silence that was read as idleness.** The session watcher tailed the
orchestrator log. The launcher had blocked on its driver-start ssh, so it wrote
no lines, and seven hours of a billing pod looked exactly like seven hours of
nothing happening. `SessionWatcher` therefore takes a provider observation as a
required argument: there is no code path in this module that produces a verdict
from a log alone, and `LOG_SILENT` is never itself a terminal state.

The watchdog depends on the provider control plane and on its own durable state
file. It never opens an SSH connection, reads a remote path, or waits on the
driver, so blocked SSH, a hung driver, a crashed trainer and a failed artifact
collection are all — from here — indistinguishable and irrelevant.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .provider import PodProvider, PodState, TerminationAttempt


@dataclass(frozen=True)
class WatchdogPolicy:
    """When to kill, and how hard to try.

    `session_start_epoch` is the epoch of the session's **first** pod create,
    not the current pod's. Billing is per session: a cold-host redraw replaces
    the pod and must not hand the replacement a fresh meter (E6b launcher,
    `pod_start_epoch` written once).
    """

    pod_id: str
    session_start_epoch: float
    price_per_hour: float
    hard_terminate_minutes: float
    authorized_usd: float
    poll_seconds: float = 60.0
    terminate_rounds: int = 6
    verify_polls: int = 6
    verify_delay_seconds: float = 15.0
    # A pod that outlives this many minutes past its hard limit is escalated in
    # the journal as `TERMINATION_FAILED`, which is the state a human must see.
    escalate_after_minutes: float = 20.0

    def elapsed_minutes(self, now: float) -> float:
        return max(0.0, (now - self.session_start_epoch) / 60.0)

    def accrued_usd(self, now: float) -> float:
        return self.elapsed_minutes(now) / 60.0 * self.price_per_hour


@dataclass(frozen=True)
class Observation:
    """One watchdog poll: provider truth plus the watchdog's own arithmetic."""

    at_epoch: float
    elapsed_minutes: float
    accrued_usd: float
    state: PodState
    over_hard_limit: bool
    action: str  # "none" | "terminate" | "verified_gone" | "termination_failed"

    def as_dict(self) -> dict:
        return {
            "at_epoch": round(self.at_epoch, 3),
            "elapsed_minutes": round(self.elapsed_minutes, 2),
            "accrued_usd": round(self.accrued_usd, 4),
            "pod_exists": self.state.exists,
            "pod_billing": self.state.billing,
            "desired_status": self.state.desired_status,
            "poll_error": self.state.error,
            "over_hard_limit": self.over_hard_limit,
            "action": self.action,
        }


class Journal:
    """Append-only JSONL, flushed and fsynced per record.

    The watchdog's entire value in a post-mortem is that its record survives the
    thing that went wrong. Buffered writes do not survive an OOM kill.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields) -> dict:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in
                self.path.read_text().splitlines() if line.strip()]


class Watchdog:
    """Poll the provider, enforce the hard limit, verify the pod is gone."""

    def __init__(self, provider: PodProvider, policy: WatchdogPolicy,
                 journal: Journal, *,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.provider = provider
        self.policy = policy
        self.journal = journal
        self.clock = clock
        self.sleep = sleep
        self.first_over_limit_epoch: float | None = None

    # -- one cycle ---------------------------------------------------------
    def poll(self) -> Observation:
        now = self.clock()
        state = self.provider.get(self.policy.pod_id)
        elapsed = self.policy.elapsed_minutes(now)
        over = elapsed >= self.policy.hard_terminate_minutes
        return Observation(
            at_epoch=now, elapsed_minutes=elapsed,
            accrued_usd=self.policy.accrued_usd(now), state=state,
            over_hard_limit=over, action="none")

    def tick(self) -> Observation:
        """Poll once, and terminate if the hard limit has passed.

        A poll error does not reset the clock and does not stop the escalation:
        `PodState.billing` treats unknown as billing, so an unreachable control
        plane still leads to a termination attempt rather than to a shrug.
        """
        obs = self.poll()
        self.journal.write("poll", **obs.as_dict())

        if not obs.state.billing:
            return Observation(**{**obs.__dict__, "action": "verified_gone"})
        if not obs.over_hard_limit:
            return obs

        if self.first_over_limit_epoch is None:
            self.first_over_limit_epoch = obs.at_epoch
            self.journal.write(
                "hard_limit_reached",
                elapsed_minutes=round(obs.elapsed_minutes, 2),
                accrued_usd=round(obs.accrued_usd, 4),
                hard_terminate_minutes=self.policy.hard_terminate_minutes,
                authorized_usd=self.policy.authorized_usd)

        gone = self.terminate_and_verify()
        return Observation(**{**obs.__dict__,
                              "action": "verified_gone" if gone
                              else "termination_failed"})

    # -- termination -------------------------------------------------------
    def terminate_and_verify(self) -> bool:
        """Terminate, then confirm by polling. Retry the whole round.

        Termination is not the request; it is the pod being gone. E6b's backstop
        "fired" in the sense that a deadline existed, and the pod ran for six
        more minutes. Every attempt and every response is journalled, including
        the ones that claimed success.
        """
        for round_no in range(1, self.policy.terminate_rounds + 1):
            attempts = self._attempt(round_no)
            self.journal.write(
                "terminate_attempt", round=round_no,
                attempts=[a.as_dict() for a in attempts],
                any_ok=any(a.ok for a in attempts))
            for poll_no in range(1, self.policy.verify_polls + 1):
                state = self.provider.get(self.policy.pod_id)
                self.journal.write(
                    "terminate_verify", round=round_no, poll=poll_no,
                    pod_exists=state.exists, desired_status=state.desired_status,
                    poll_error=state.error, billing=state.billing)
                if not state.billing:
                    self.journal.write("terminated", round=round_no,
                                       polls=poll_no,
                                       desired_status=state.desired_status)
                    return True
                if poll_no < self.policy.verify_polls:
                    self.sleep(self.policy.verify_delay_seconds)
        self.journal.write(
            "TERMINATION_FAILED", pod_id=self.policy.pod_id,
            rounds=self.policy.terminate_rounds,
            note="pod still billing after every termination round; human "
                 "intervention required")
        return False

    def _attempt(self, round_no: int) -> list[TerminationAttempt]:
        try:
            return list(self.provider.terminate(self.policy.pod_id))
        except Exception as exc:  # noqa: BLE001 - a watchdog may not die here
            # Broad by design. This runs on the layer whose entire job is to
            # survive whatever killed everything else; an unexpected exception
            # from a provider client must become a journalled failed attempt,
            # not the end of the watchdog.
            return [TerminationAttempt(
                method="provider.terminate", verified_transport=False,
                ok=False, error=f"{type(exc).__name__}: {exc}")]

    # -- loop --------------------------------------------------------------
    def run(self, *, max_ticks: int | None = None) -> str:
        """Poll until the pod is gone, termination is impossible, or ticks run out.

        Returns a terminal reason. `max_ticks` bounds the loop for tests and for
        `--once` operation; in a session it is left unset and the watchdog exits
        when the pod does.
        """
        self.journal.write(
            "watchdog_start", pod_id=self.policy.pod_id,
            hard_terminate_minutes=self.policy.hard_terminate_minutes,
            authorized_usd=self.policy.authorized_usd,
            price_per_hour=self.policy.price_per_hour,
            session_start_epoch=self.policy.session_start_epoch)
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            ticks += 1
            obs = self.tick()
            if obs.action == "verified_gone":
                self.journal.write("watchdog_end", reason="pod_gone", ticks=ticks)
                return "pod_gone"
            if obs.action == "termination_failed":
                over_for = (obs.at_epoch - (self.first_over_limit_epoch
                                            or obs.at_epoch)) / 60.0
                if over_for >= self.policy.escalate_after_minutes:
                    self.journal.write(
                        "watchdog_end", reason="termination_failed",
                        ticks=ticks, over_limit_minutes=round(over_for, 2))
                    return "termination_failed"
            self.sleep(self.policy.poll_seconds)
        self.journal.write("watchdog_end", reason="max_ticks", ticks=ticks)
        return "max_ticks"


# --------------------------------------------------------------------------
# Watcher correctness
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkerSnapshot:
    """Durable status markers, and how stale they are.

    `age_seconds` is None when the marker has never been seen at all, which is
    a different thing from a marker that stopped advancing.
    """

    last_marker: str | None = None
    age_seconds: float | None = None
    orchestrator_log_age_seconds: float | None = None


# Verdicts. Note that none of them is `IDLE`: this module has no way to conclude
# that nothing is running, because in the one case where that inference was
# drawn it was wrong by seven hours and $4.
LIVE = "LIVE_AND_BILLING"
LIVE_STALLED = "LIVE_AND_BILLING_MARKERS_STALLED"
LIVE_UNOBSERVED = "LIVE_AND_BILLING_NO_MARKERS"
PROVIDER_UNKNOWN = "PROVIDER_UNKNOWN_ASSUME_BILLING"
GONE = "POD_GONE"


@dataclass(frozen=True)
class Verdict:
    state: str
    billing: bool
    reason: str
    recommended_action: str


class SessionWatcher:
    """Answer "is anything running" from provider state, never from log silence.

    `assess` takes the provider observation as a required positional argument.
    That is the design: a caller cannot reach a verdict with markers alone, so
    the E6b inference — quiet log, therefore quiet session — is not expressible.
    """

    def __init__(self, *, stall_seconds: float = 1800.0) -> None:
        self.stall_seconds = stall_seconds

    def assess(self, state: PodState, markers: MarkerSnapshot) -> Verdict:
        if state.error is not None:
            return Verdict(
                PROVIDER_UNKNOWN, True,
                f"the control plane did not answer ({state.error}); an "
                "unanswered poll is not an absent pod",
                "retry the poll; treat the session as billing until it answers")
        if not state.billing:
            return Verdict(GONE, False,
                           f"provider reports {state.desired_status or 'no pod'}",
                           "none; the meter has stopped")

        if markers.last_marker is None:
            return Verdict(
                LIVE_UNOBSERVED, True,
                "the pod exists and is billing, and no status marker has ever "
                "been observed — this is what a launcher blocked on its "
                "driver-start ssh looks like from outside",
                "check the driver out of band (job descriptor / pidfile), not "
                "the orchestrator log")
        if (markers.age_seconds is not None
                and markers.age_seconds >= self.stall_seconds):
            return Verdict(
                LIVE_STALLED, True,
                f"the pod is billing and the last marker "
                f"({markers.last_marker}) is {markers.age_seconds:.0f}s old",
                "the driver may be hung; the hard limit still applies")
        return Verdict(
            LIVE, True,
            f"the pod is billing and markers are advancing "
            f"(last: {markers.last_marker})",
            "continue polling")
