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


def test_a_soft_stop_reserve_lands_before_the_soft_stop_not_after_it():
    """Placement is the whole point, so it is asserted rather than assumed.

    A reserve added as a phase would be multiplied by the contingency and would
    inflate the expected figure an authorization request quotes. A reserve added
    after the soft stop moves only the watchdog's kill time, which protects the
    pod and not the experiment: a driver's `afford()` refuses to START anything
    that would cross the SOFT stop, so a risk that materializes early is paid
    for out of the later stages' budget.
    """
    from aadistill.infrastructure.budget import Phase, StepTime, plan_session

    common = dict(price_per_hour=1.0, authorized_usd=1e6, arms=0, steps_per_arm=0,
                  step_time=StepTime(4.15, "measured"), setup_minutes=10.0,
                  other_phases=(Phase("work", 90.0),),
                  contingency_fraction=0.10,
                  artifact_recovery_reserve_minutes=20.0)
    bare = plan_session(**common)
    with_reserve = plan_session(**common,
                                soft_stop_reserves=(Phase("risk", 30.0),))

    assert with_reserve.expected_minutes == bare.expected_minutes, (
        "the reserve inflated the expected figure; it would then also be "
        "multiplied by the contingency")
    assert with_reserve.soft_stop_minutes == bare.soft_stop_minutes + 30.0
    assert with_reserve.hard_terminate_minutes == bare.hard_terminate_minutes + 30.0
    # The reserve is *between* expected and soft, not between soft and hard.
    assert (with_reserve.hard_terminate_minutes
            - with_reserve.soft_stop_minutes) == pytest.approx(20.0), (
        "the artifact-recovery reserve changed size; the new reserve was added "
        "after the soft stop instead of before it")


def test_soft_stop_reserves_are_named_in_the_record():
    from aadistill.infrastructure.budget import Phase, StepTime, plan_session

    plan = plan_session(
        price_per_hour=1.0, authorized_usd=1e6, arms=0, steps_per_arm=0,
        step_time=StepTime(4.15, "measured"), setup_minutes=10.0,
        soft_stop_reserves=(Phase("fallback", 12.0), Phase("correction", 8.0)),
        artifact_recovery_reserve_minutes=20.0)
    d = plan.as_dict()
    assert [r["name"] for r in d["soft_stop_reserves"]] == ["fallback", "correction"]
    assert d["soft_stop_reserve_minutes"] == 20.0
    assert plan.soft_stop_minutes == pytest.approx(10.0 * 1.10 + 20.0)


def test_a_negative_soft_stop_reserve_is_refused():
    """A negative reserve would buy budget by asserting a risk is impossible."""
    from aadistill.infrastructure.budget import (
        BudgetError, Phase, StepTime, plan_session,
    )

    with pytest.raises(BudgetError, match="cannot be negative"):
        plan_session(price_per_hour=1.0, authorized_usd=1e6, arms=0,
                     steps_per_arm=0, step_time=StepTime(4.15, "measured"),
                     setup_minutes=10.0,
                     soft_stop_reserves=(Phase("wishful", -30.0),),
                     artifact_recovery_reserve_minutes=20.0)
