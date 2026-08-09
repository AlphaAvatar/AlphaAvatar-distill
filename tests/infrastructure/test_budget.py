"""Four thresholds, priced from what was measured rather than what was hoped.

E6b was priced at 3.625 s/step from E4's comparable arms and sustained 4.15 —
a 14% miss worth ~$0.81, on a session whose authorization and kill point were
the same number, so the miss arrived with no reserve behind it.
"""

import pytest

from aadistill.infrastructure.budget import (
    BudgetError, MEASURED_STEP_SECONDS, Phase, StepTime,
    SUPERSEDED_STEP_SECONDS, plan_session,
)

MEASURED = StepTime(MEASURED_STEP_SECONDS, "E6b arms, L40S, 2026-08-08")


def a_plan(**over):
    kwargs = dict(
        price_per_hour=0.99, authorized_usd=9.50, arms=2, steps_per_arm=2916,
        step_time=MEASURED, setup_minutes=25.0, eval_minutes_per_arm=10.0,
        transfer_minutes=20.0,
    )
    kwargs.update(over)
    return plan_session(**kwargs)


def test_the_four_thresholds_are_strictly_ordered():
    p = a_plan()
    assert (p.expected_minutes < p.soft_stop_minutes
            < p.hard_terminate_minutes)
    assert p.hard_terminate_minutes == pytest.approx(
        p.soft_stop_minutes + p.artifact_recovery_reserve_minutes)
    assert p.hard_terminate_usd <= p.authorized_usd


def test_the_reserve_is_time_held_back_after_the_soft_stop():
    """The property E6b lacked: room to collect artifacts inside the budget."""
    p = a_plan(artifact_recovery_reserve_minutes=45.0)
    assert p.hard_terminate_minutes - p.soft_stop_minutes == pytest.approx(45.0)
    # A phase that would run past the soft stop may not start, even though it
    # would finish before the hard limit.
    assert p.may_start(p.soft_stop_minutes - 10, 5) is True
    assert p.may_start(p.soft_stop_minutes - 10, 30) is False


def test_phase_at_names_each_regime():
    p = a_plan()
    assert p.phase_at(0) == "nominal"
    assert p.phase_at(p.expected_minutes + 1) == "over_expected"
    assert p.phase_at(p.soft_stop_minutes + 1) == "artifact_recovery"
    assert p.phase_at(p.hard_terminate_minutes) == "terminate"


def test_a_plan_over_its_authorization_reports_the_shortfall_and_refuses():
    """It must not quietly shrink the run to fit; that is the maintainer's call."""
    with pytest.raises(BudgetError) as exc:
        a_plan(authorized_usd=7.12)
    message = str(exc.value)
    assert "shortfall" in message
    assert "$7.12" in message
    assert "do not shrink it silently" in message


def test_e6b_at_the_measured_step_time_does_not_fit_its_authorization():
    """The arithmetic that would have caught the overrun before launch."""
    with pytest.raises(BudgetError):
        plan_session(
            price_per_hour=0.99, authorized_usd=7.12, arms=2,
            steps_per_arm=2916, step_time=MEASURED, setup_minutes=25.0,
            eval_minutes_per_arm=10.0, transfer_minutes=20.0)


def test_a_step_time_below_the_measured_floor_needs_a_recorded_reason():
    superseded = StepTime(SUPERSEDED_STEP_SECONDS, "E4 comparable arms")
    with pytest.raises(BudgetError) as exc:
        a_plan(step_time=superseded)
    assert "below the measured floor" in str(exc.value)
    assert "E4" in str(exc.value)

    ok = a_plan(step_time=superseded,
                below_floor_reason="shorter blocks, no teacher in memory",
                authorized_usd=9.5)
    assert any("below-floor step time accepted" in n for n in ok.notes)


def test_the_floor_is_a_parameter_not_a_global_truth():
    """A different workload may be genuinely faster; it declares its own floor."""
    fast = StepTime(1.2, "vLLM generation, measured 2026-08-09")
    p = a_plan(step_time=fast, step_time_floor=1.0)
    assert p.notes == ()
    assert p.expected_minutes < 200


def test_an_unsourced_step_time_is_refused():
    with pytest.raises(BudgetError, match="must record its source"):
        StepTime(4.15, "   ")


def test_a_session_with_no_recovery_reserve_is_refused():
    with pytest.raises(BudgetError, match="reserve must be positive"):
        a_plan(artifact_recovery_reserve_minutes=0)


def test_the_breakdown_accounts_for_the_whole_expected_time():
    p = a_plan(other_phases=(Phase("checkpoint_fetch", 45.0),),
               authorized_usd=10.50)
    assert sum(x.minutes for x in p.breakdown) == pytest.approx(
        p.expected_minutes)
    assert {x.name for x in p.breakdown} >= {
        "setup", "train", "evaluate", "transfer", "checkpoint_fetch"}


def test_the_plan_serializes_for_the_session_record():
    d = a_plan().as_dict()
    for key in ("expected_usd", "soft_stop_usd", "artifact_recovery_reserve_usd",
                "hard_terminate_usd", "step_time", "breakdown"):
        assert key in d
    assert d["step_time"]["source"].startswith("E6b")
