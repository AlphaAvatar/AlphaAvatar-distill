"""The launcher must not stop polling while the pod is still alive.

Phase B inherited Phase A's `--poll-limit-min 1320` — 22 h. The corrected Phase-B
envelope terminates at **1925.87 min (32.10 h)**, so the launcher would have
stopped polling **606 min, just over ten hours**, before the session reached its
own operational hard bound. A launcher that stops polling exits; the pod does
not. This project's most expensive recorded waste is idle pod time, and
`--terminate-after` has never once been observed to stop a pod.

So the contract asserted here is relational, not a number:

    poll lifetime > plan hard-terminate lifetime + enough slack to observe the
    end, fetch the products home, and confirm the pod is gone.

Every term of that slack is an existing bounded knob — the poll interval, the
checkpoint-fetch limit, and a teardown time measured from attempt 3 — so the
default moves automatically the next time the envelope does. Which is exactly how
1320 became wrong: it was right when it was written and nothing re-derived it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

import autoinit_phase_a_launch as pal  # noqa: E402
import autoinit_phase_b_launch as pbl  # noqa: E402

#: What Phase A ran under, and what Phase B silently inherited.
PHASE_A_HISTORICAL_DEFAULT = 1320


def args(**over):
    parsed = pbl.build_parser().parse_args(
        ["--scr", "/tmp/does-not-matter", "--session-commit", "d" * 40,
         "--bundle", "b.bundle"])
    for k, v in over.items():
        setattr(parsed, k, v)
    return parsed


def plan_for(a):
    return pbl.phase_b_budget(a).plan(price_per_hour=a.max_price,
                                      authorized_usd=float("inf"))


def ctx_for(a, plan=None):
    return types.SimpleNamespace(args=a, plan=plan if plan is not None else plan_for(a),
                                 auth=None, evidence={})


# --- the contract -----------------------------------------------------------


def test_the_derived_lifetime_exceeds_the_hard_terminate_lifetime():
    a = args()
    hard = plan_for(a).hard_terminate_minutes
    derived = pbl.phase_b_poll_limit_minutes(a)
    assert derived > hard, (derived, hard)
    assert derived - hard == pytest.approx(
        a.poll_seconds / 60.0 + a.ckpt_fetch_limit_min + pbl.TEARDOWN_OBSERVED_MINUTES)


def test_the_inherited_phase_a_default_would_NOT_have_survived_this_gate():
    """The defect, stated as a measurement rather than a description."""
    a = args(poll_limit_min=PHASE_A_HISTORICAL_DEFAULT)
    hard = plan_for(a).hard_terminate_minutes
    assert PHASE_A_HISTORICAL_DEFAULT < hard, (
        "the inherited default already exceeds the hard bound, so this test no "
        "longer describes the situation it was written for")
    ok, why = pbl.poll_lifetime_gate(ctx_for(a))
    assert not ok and "still alive and billing" in why
    assert hard - PHASE_A_HISTORICAL_DEFAULT > 600, (
        f"short by only {hard - PHASE_A_HISTORICAL_DEFAULT:.0f} min")


def test_phase_As_own_default_is_untouched():
    """A historical contract several completed sessions ran under."""
    a = pal.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "d" * 40, "--bundle", "b"])
    assert a.poll_limit_min == PHASE_A_HISTORICAL_DEFAULT


def test_the_phase_b_default_is_derived_rather_than_declared():
    """`None` until resolved, so no constant can go stale in the parser."""
    assert args().poll_limit_min is None
    import inspect

    source = inspect.getsource(pbl.main)
    assert "phase_b_poll_limit_minutes(args)" in source
    assert "args.poll_limit_min is None" in source


def test_the_derived_value_FOLLOWS_the_plan_rather_than_sitting_beside_it():
    """Mutate the priced envelope; the lifetime must move with it.

    A number that happens to be larger today is not a contract. Lengthening the
    search must lengthen the polling, or the two drift apart again.
    """
    a = args()
    base = pbl.phase_b_poll_limit_minutes(a)
    # Drive the plan directly, so this proves the derivation reads the PLAN and
    # not a module constant that merely happens to be large enough today.
    plan = plan_for(a)
    stretched = type(plan)(
        **{**plan.__dict__,
           "hard_terminate_minutes": plan.hard_terminate_minutes + 600})
    assert pbl.phase_b_poll_limit_minutes(a, plan=stretched) == pytest.approx(base + 600)


# --- the gate ---------------------------------------------------------------


@pytest.mark.parametrize("offset,expected", [
    (-1.0, False),        # below the hard bound
    (0.0, False),         # exactly the hard bound: the pod outlives the poll
    (1.0, False),         # above it, but with no room to collect or tear down
    (41.0, False),        # one minute short of the derived slack
    (42.0, True),         # exactly the derived slack
    (600.0, True),        # generous
])
def test_the_gate_requires_real_teardown_slack(offset, expected):
    a = args()
    hard = plan_for(a).hard_terminate_minutes
    a.poll_limit_min = hard + offset
    ok, why = pbl.poll_lifetime_gate(ctx_for(a))
    assert ok is expected, (offset, why)


def test_the_gate_accepts_the_derived_default():
    a = args()
    a.poll_limit_min = pbl.phase_b_poll_limit_minutes(a)
    ok, why = pbl.poll_lifetime_gate(ctx_for(a))
    assert ok, why
    assert "teardown slack" in why


def test_the_gate_refuses_an_unresolved_default():
    ok, why = pbl.poll_lifetime_gate(ctx_for(args()))
    assert not ok and "never resolved" in why


def test_the_gate_runs_before_a_pod_exists():
    session = pbl.spec(args())
    names = [getattr(g, "__name__", type(g).__name__) for g in session.precheck]
    assert "poll_lifetime_gate" in names


def test_the_slack_is_built_from_bounded_knobs_not_a_lump():
    """Each term is something the launcher already bounds elsewhere."""
    a = args(poll_seconds=300.0, ckpt_fetch_limit_min=45)
    hard = plan_for(a).hard_terminate_minutes
    derived = pbl.phase_b_poll_limit_minutes(a)
    assert derived - hard == pytest.approx(5.0 + 45 + pbl.TEARDOWN_OBSERVED_MINUTES)


def test_the_measured_teardown_anchor_is_stated_and_conservative():
    """2.8 min observed at attempt 3's teardown; the anchor carries margin."""
    assert pbl.TEARDOWN_OBSERVED_MINUTES >= 3 * 2.8
