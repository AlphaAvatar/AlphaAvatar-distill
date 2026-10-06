"""The provider ACCOUNT must be able to fund a session before a pod exists.

THE GATE THAT WAS MISSING. On 2026-10-06 D1's formal search passed every
internal gate -- identity, session contract, staged science inputs, launch
readiness, session commit and lineage, the canonical bundle -- was authorized to
`$21.4897` over `1125.55` minutes, and RunPod stopped it at `449.8` minutes with
39 of 92 expansions complete, because the ACCOUNT had run out of money.
`$8.1716` bought no endpoint.

Every gate in this project asked whether the experiment was PERMITTED to spend.
None asked whether the provider would still be paid. Those are different
questions and both have to be answered before a pod is created.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aadistill.infrastructure.provider import (  # noqa: E402
    AccountBalance, SimulatedProvider,
)
from aadistill.infrastructure.session import BudgetSpec  # noqa: E402
from aadistill.infrastructure.session_runner import SessionRunner  # noqa: E402


class TestWhatTheBalanceAnswers:

    def test_a_balance_that_covers_the_requirement_passes(self):
        ok, why = AccountBalance(
            known=True, client_balance_usd=199.95).covers(30.0)
        assert ok and "199.9500" in why and "30.0000" in why

    def test_a_short_balance_refuses(self):
        ok, why = AccountBalance(
            known=True, client_balance_usd=12.0).covers(30.0)
        assert not ok
        assert "stopped part-way" in why

    def test_an_unreadable_balance_refuses_rather_than_assuming(self):
        """Unknown is not a negative answer anywhere else here -- but a balance
        that cannot be read also cannot be shown to cover the session, and the
        refusal is free."""
        ok, why = AccountBalance(known=False, error="502").covers(30.0)
        assert not ok
        assert "could not be read" in why

    def test_under_balance_refuses_even_when_the_number_looks_fine(self):
        """The PROVIDER decides when it stops a pod, so its own judgement wins
        over comparing two numbers here."""
        ok, why = AccountBalance(known=True, client_balance_usd=99.0,
                                 under_balance=True).covers(30.0)
        assert not ok
        assert "UNDER its minimum" in why

    def test_exactly_the_requirement_is_enough(self):
        assert AccountBalance(known=True, client_balance_usd=30.0).covers(30.0)[0]


def _runner(required, *, balance=1000.0, under=False):
    """A runner at the gate, with a tripwire where `create` would be."""
    runner = SessionRunner.__new__(SessionRunner)
    runner.ev = {}
    runner.said = []
    runner.say = runner.said.append
    runner.save = lambda: None
    runner.plan = type("P", (), {"hard_terminate_minutes": 1125.5})()
    runner.spec = type("S", (), {
        "budget": BudgetSpec(
            arms=0, steps_per_arm=0, step_seconds=1.0, step_source="t",
            setup_minutes=1.0, transfer_minutes=1.0,
            account_balance_required_usd=required)})()
    runner.provider = SimulatedProvider(
        "pod", client_balance_usd=balance, under_balance=under)
    return runner


class TestTheGate:

    def test_a_session_that_declares_no_requirement_is_not_gated(self):
        """Every launcher written before this keeps its behaviour, and the
        provider is not even asked."""
        r = _runner(None)
        assert SessionRunner.check_account_funds(r) is True
        assert "account_balance" not in r.ev
        assert "account_balance" not in r.provider.calls

    def test_a_covered_requirement_passes_and_records_the_observed_balance(self):
        r = _runner(30.0, balance=199.95)
        assert SessionRunner.check_account_funds(r) is True
        rec = r.ev["account_balance"]
        assert rec["ok"] is True
        assert rec["client_balance_usd"] == pytest.approx(199.95)
        assert rec["required_usd"] == pytest.approx(30.0)
        assert any("account funds OK" in line for line in r.said)

    def test_a_short_balance_refuses_at_zero_dollars(self):
        r = _runner(30.0, balance=8.0)
        assert SessionRunner.check_account_funds(r) is False
        assert r.ev["account_balance"]["ok"] is False
        assert any("ABORT" in line for line in r.said)

    def test_an_under_balance_account_refuses(self):
        r = _runner(30.0, balance=99.0, under=True)
        assert SessionRunner.check_account_funds(r) is False

    def test_the_requirement_may_be_derived_from_the_priced_plan(self):
        """A callable, so a session can scale the requirement with its own
        ceiling instead of carrying a constant."""
        r = _runner(lambda plan: plan.hard_terminate_minutes / 100.0,
                    balance=20.0)
        assert SessionRunner.check_account_funds(r) is True
        assert r.ev["account_balance"]["required_usd"] == pytest.approx(11.255)


class TestItRunsBeforeAnythingIsCreated:

    def test_a_refused_balance_stops_run_before_create(self):
        """The whole value of the gate is that it is free. A refusal that still
        reached `create()` would cost exactly what it exists to prevent."""

        class Tripped(Exception):
            pass

        r = _runner(30.0, balance=1.0)
        r.a = type("A", (), {"dry_run": False, "host_draws": 1})()
        r.make_plan = lambda: True
        r.run_prechecks = lambda: True

        def create():
            raise Tripped("create() was reached on a refused balance")

        r.create = create
        assert SessionRunner.run(r) is False
        assert r.ev["account_balance"]["ok"] is False

    def test_the_prechecks_are_not_even_reached_on_a_refusal(self):
        """Ordered before `run_prechecks` deliberately: the account question is
        cheaper than the session's own gates and decides the same launch."""
        r = _runner(30.0, balance=1.0)
        r.a = type("A", (), {"dry_run": False, "host_draws": 1})()
        r.make_plan = lambda: True
        reached = []
        r.run_prechecks = lambda: reached.append(1) or True
        r.create = lambda: pytest.fail("create() reached")
        assert SessionRunner.run(r) is False
        assert reached == []

    def test_a_dry_run_reports_the_same_answer_a_launch_would_get(self):
        """The gate sits before the dry-run stop, so `--dry-run` cannot report a
        launchable chain against an account that could not fund it."""
        r = _runner(30.0, balance=1.0)
        r.a = type("A", (), {"dry_run": True, "host_draws": 1})()
        r.make_plan = lambda: True
        r.run_prechecks = lambda: True
        r.create = lambda: pytest.fail("create() reached")
        assert SessionRunner.run(r) is False
        assert r.ev["account_balance"]["ok"] is False
        assert r.ev.get("terminal") != "DRY_RUN_GATES_PASSED"


class TestTheRequirementBelongsToTheCampaign:

    def test_the_requirement_has_no_numeric_default_in_core(self):
        """`$30` is this campaign's per-session envelope, not a property of the
        session machinery, and a constant here would be the same defect as a
        hardcoded experiment id in reusable core (AGENTS.md P3).

        Checked on the FIELD rather than by grepping the file for `30.0`: a
        substring search over `session.py` also matches
        `artifact_recovery_reserve_minutes: float = 30.0`, which is minutes and
        has nothing to do with money. The first version of this test did exactly
        that and failed on it.
        """
        import dataclasses

        field, = [f for f in dataclasses.fields(BudgetSpec)
                  if f.name == "account_balance_required_usd"]
        assert field.default is None, (
            "a dollar amount defaulted in reusable core would silently become "
            "every future session's requirement")

    def test_the_threshold_is_an_argument_everywhere_in_core(self):
        """`covers` takes the amount; it never consults a module constant."""
        import inspect

        from aadistill.infrastructure import provider as P

        src = inspect.getsource(P.AccountBalance.covers)
        assert "required_usd" in inspect.signature(
            P.AccountBalance.covers).parameters
        #: No numeric literal is compared against the balance.
        for line in src.splitlines():
            if "client_balance" in line and "<" in line:
                assert "required_usd" in line, line

    def test_the_default_is_ungated(self):
        assert BudgetSpec.account_balance_required_usd is None

    def test_a_spec_without_a_budget_at_all_is_ungated(self):
        """A hand-built spec double has no `budget`, and the gate must not turn
        every test that drives `run()` into an AttributeError. Six existing core
        tests did exactly that when this gate was first wired in.

        Safe because a real `SessionSpec.budget` is a required field of a frozen
        dataclass and `account_balance_required_usd` has a default, so the
        tolerance can only ever apply to a double.
        """
        import types

        r = SessionRunner.__new__(SessionRunner)
        r.ev, r.said = {}, []
        r.say = r.said.append
        r.spec = types.SimpleNamespace()
        assert SessionRunner.check_account_funds(r) is True
        assert "account_balance" not in r.ev

    def test_a_real_session_spec_always_carries_the_field(self):
        """So the tolerance above cannot hide a real session skipping the gate."""
        import dataclasses

        from aadistill.infrastructure.session import SessionSpec

        budget, = [f for f in dataclasses.fields(SessionSpec)
                   if f.name == "budget"]
        assert budget.default is dataclasses.MISSING, (
            "if `budget` ever gains a default, a real session could reach the "
            "gate without one and be silently ungated")
        assert any(f.name == "account_balance_required_usd"
                   for f in dataclasses.fields(BudgetSpec))
