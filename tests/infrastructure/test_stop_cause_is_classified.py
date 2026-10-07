"""A pod that stops billing has several possible causes, and the launcher used
to assert one of them.

On 2026-10-06 D1's formal search stopped at 449.8 minutes against its own
1125.5-minute hard threshold. The launcher logged

    the pod is gone — the watchdog acted

and `finish_emergency` recorded

    the watchdog terminated the pod at the hard threshold (1126 min / $20.45)

while the watchdog's own journal recorded `action: none` on all 446 ticks with
`over_hard_limit: false` throughout. The real cause was an exhausted RunPod
account balance. Two records asserted a cause neither had checked, and the
closeout then threw instead of recording the loss.

The generic classifier is regressed here. The 2026-10-06 journal itself is
D1's evidence and its assertion lives with D1 (AGENTS.md 2.8a).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aadistill.infrastructure.artifact_gate import ArtifactError  # noqa: E402
from aadistill.infrastructure.provider import PodState  # noqa: E402
from aadistill.infrastructure.session_runner import (  # noqa: E402
    STOP_BY_PROVIDER, STOP_BY_WATCHDOG, STOP_STOPPED_NOT_GONE, STOP_UNKNOWN,
    SessionRunner, observed_stop_cause,
)

#: The incident that produced this classifier is regressed against the REAL
#: journal it left behind, and that assertion used to live here behind a
#: `skipif(not REAL_JOURNAL.is_file())`. The skip-predicate audit flagged it as
#: this suite's one UNACCOUNTED predicate -- the signal is classified and
#: nothing said the two machines must decide it the same way -- and the guard
#: turned out to be dead: the journal is tracked, so it never fired. A dead
#: guard on a premise nobody asserts is one `git rm` away from disabling the
#: only evidence that the watchdog did not act, silently.
#:
#: Under AGENTS.md 2.8a a closed run's own journal is that experiment's state,
#: so the historical assertion moved to
#: `scripts/experiments/stage-1/phase_d1/tests/test_d1_stop_cause_incident.py`,
#: where it asserts the fixture's PRESENCE rather than conditioning on it.
#: What stays here is the generic behaviour: four causes, decided from what the
#: journal says, over journals these tests write themselves.

GONE = PodState(pod_id="p", exists=False, desired_status="TERMINATED")
def _journal(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "watchdog_p.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


class TestTheThreeCauses:

    def test_a_watchdog_action_in_the_journal_is_a_watchdog_termination(
            self, tmp_path):
        j = _journal(tmp_path, [
            {"event": "poll", "action": "none"},
            {"event": "poll", "action": "terminate"},
        ])
        got = observed_stop_cause(GONE, watchdog_journal=j,
                                  elapsed_minutes=500.0,
                                  hard_terminate_minutes=1125.5)
        assert got["cause"] == STOP_BY_WATCHDOG
        assert got["watchdog_took_an_action"] is True

    def test_past_the_hard_threshold_is_a_watchdog_termination(self, tmp_path):
        """Even with a journal showing nothing: at or over its own threshold the
        budget boundary is the expected cause."""
        j = _journal(tmp_path, [{"event": "poll", "action": "none"}])
        got = observed_stop_cause(GONE, watchdog_journal=j,
                                  elapsed_minutes=1130.0,
                                  hard_terminate_minutes=1125.5)
        assert got["cause"] == STOP_BY_WATCHDOG
        assert got["over_hard_limit"] is True

    def test_short_of_the_threshold_with_no_action_is_the_provider(self, tmp_path):
        j = _journal(tmp_path, [{"event": "poll", "action": "none"}])
        got = observed_stop_cause(GONE, watchdog_journal=j,
                                  elapsed_minutes=100.0,
                                  hard_terminate_minutes=1125.5)
        assert got["cause"] == STOP_BY_PROVIDER

    def test_a_pod_that_still_exists_is_distinguished_from_one_that_is_gone(
            self, tmp_path):
        """`not state.billing` conflates them, and the difference matters: a
        stopped pod that still exists may hold a recoverable workdir."""
        j = _journal(tmp_path, [{"event": "poll", "action": "none"}])
        stopped = PodState(pod_id="p", exists=True, desired_status="EXITED")
        got = observed_stop_cause(stopped, watchdog_journal=j,
                                  elapsed_minutes=100.0,
                                  hard_terminate_minutes=1125.5)
        assert got["cause"] == STOP_STOPPED_NOT_GONE
        assert got["pod_exists"] is True

    def test_no_journal_at_all_is_unknown_rather_than_either(self, tmp_path):
        got = observed_stop_cause(GONE, watchdog_journal=tmp_path / "absent",
                                  elapsed_minutes=100.0,
                                  hard_terminate_minutes=1125.5)
        assert got["cause"] == STOP_UNKNOWN

    def test_an_unparseable_line_does_not_break_the_classification(self, tmp_path):
        p = tmp_path / "watchdog_p.jsonl"
        p.write_text('{"event": "poll", "action": "none"}\nnot json\n')
        got = observed_stop_cause(GONE, watchdog_journal=p,
                                  elapsed_minutes=100.0,
                                  hard_terminate_minutes=1125.5)
        assert got["cause"] == STOP_BY_PROVIDER
        assert got["watchdog_ticks"] == 1


def _emergency_runner(tmp_path, streams: tuple[str, ...], cause: dict | None):
    runner = SessionRunner.__new__(SessionRunner)
    runner.ev = {} if cause is None else {"stop_cause": cause}
    runner.said = []
    runner.say = runner.said.append
    runner.save = lambda: None
    runner.scr = tmp_path
    runner.start_epoch = None          # `elapsed()` reads 0.0
    runner.plan = type("P", (), {"hard_terminate_minutes": 1125.5,
                                 "hard_terminate_usd": 20.45})()
    runner.spec = type("S", (), {
        "artifacts": type("A", (), {
            "event_streams": staticmethod(lambda ctx: streams)})()})()
    runner.context = lambda **kw: None
    return runner


class TestTheEmergencyCloseoutNoLongerThrows:

    def test_a_session_declaring_no_event_streams_records_its_loss(self, tmp_path):
        """THE DEFECT. `finish_emergency` never passed `streams_at_risk`, so the
        gate took its strict "no evidence" rule, `truncating` was true, and a
        session with no declared streams had an empty `incomplete` and could
        never satisfy "name the streams you are truncating". The helper settles
        exactly this case from the SPEC and was wired into
        `collect_and_teardown` and not into here."""
        r = _emergency_runner(tmp_path, (), {"cause": STOP_BY_PROVIDER})
        SessionRunner.finish_emergency(r)          # must not raise
        assert r.ev["teardown_gate"]["allowed"] is True

    def test_the_recorded_reason_does_not_claim_the_watchdog_acted(self, tmp_path):
        r = _emergency_runner(tmp_path, (), {"cause": STOP_BY_PROVIDER})
        SessionRunner.finish_emergency(r)
        reason = r.ev["teardown_gate"]["reason"]
        assert STOP_BY_PROVIDER in reason
        assert "did not cause" in reason
        assert "the watchdog terminated the pod" not in reason

    def test_a_real_watchdog_termination_is_still_reported_as_one(self, tmp_path):
        """Narrowing the claim must not stop the true case from being stated."""
        r = _emergency_runner(tmp_path, (), {"cause": STOP_BY_WATCHDOG})
        SessionRunner.finish_emergency(r)
        reason = r.ev["teardown_gate"]["reason"]
        assert STOP_BY_WATCHDOG in reason
        assert "acted as designed" in reason

    def test_a_session_that_does_declare_streams_still_names_them(self, tmp_path):
        """The strict rule is right for an informed caller and is kept."""
        r = _emergency_runner(tmp_path, ("/w/a/train_log.jsonl",),
                              {"cause": STOP_BY_PROVIDER})
        SessionRunner.finish_emergency(r)
        gate = r.ev["teardown_gate"]
        assert gate["allowed"] is True
        assert "train_log.jsonl" in json.dumps(gate)

    def test_the_old_call_shape_is_what_used_to_raise(self, tmp_path):
        """Non-vacuity: the gate itself still refuses an uninformed caller, so
        the fix is the argument this call site now passes and not a weakened
        gate."""
        from aadistill.infrastructure.artifact_gate import evaluate_teardown

        with pytest.raises(ArtifactError, match="must name the streams"):
            evaluate_teardown(
                {"training_complete": False, "evaluation_complete": False,
                 "artifact_manifest_created": False,
                 "required_files_present": False,
                 "final_streams_quiescent": False},
                emergency_budget=True, emergency_reason="x",
                incomplete_event_streams=())
