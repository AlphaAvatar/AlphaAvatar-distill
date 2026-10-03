"""C3 is priced from a LIVE rate, and the envelope is never the grant.

The 2026-09-28 amendment raised the per-session ceiling to `$30.00` and made
the provider price a live input. Two ways that goes wrong, and both are
cheap to prevent:

* **issuing at the envelope.** `$30` is what the package allows, not what C3
  costs. An authorization carrying `$30` instead of the derived `$22.1451`
  would be a 36% over-authorization that every downstream gate would accept.
* **treating `$1.09/h` as a bound.** It is a historical observation. A live
  price of `$1.45/h` must be *re-priced*, not refused -- and one of `$1.60/h`
  must be refused on the arithmetic, not on the fact that it differs.

Everything here is pure: the rate and the envelopes are parameters, so the
whole decision surface is testable without spending anything or touching the
provider.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3 import formal_pricing as P  # noqa: E402

#: The amended envelopes, as parameters rather than as live reads, so these
#: tests describe the decision rule and not today's ledger position.
ENV = {"per_session_envelope_usd": 30.0, "project_cap_usd": 400.0,
       "cumulative_spend_usd": 350.0307, "formal_remaining_usd": 32.3824,
       "engineering_remaining_usd": 1.9395,
       #: THE LIMIT THAT WENT UNCHECKED. `formal + engineering = package`
       #: exactly, so it looked implied -- and raising the formal allowance
       #: alone leaves a session that fits its own book and breaks the
       #: package. Supplied here because `evaluate_limits` RAISES on a
       #: missing envelope rather than treating an absent limit as a passing
       #: one.
       "package_remaining_usd": 32.3824}


def test_the_rate_is_an_input_and_nothing_hardcodes_1_09():
    src = (REPO / "scripts/experiments/stage-1/phase_c3/formal_pricing.py").read_text()
    body = src.split('"""', 2)[2]
    assert "1.09" not in body, (
        "the historical $1.09/h observation is a literal in the pricing "
        "body; it is an input, not a bound")


def test_container_disk_is_billed_on_top_of_the_gpu_rate():
    """A GPU-hour rate is not the bill. Omitting disk understated $0.33."""
    p = P.price_c3(1.09)
    assert p.billed_rate_usd_per_hour > p.gpu_rate_usd_per_hour
    expected_disk = 120 * 0.10 / (30 * 24)
    assert p.billed_rate_usd_per_hour == pytest.approx(
        1.09 + expected_disk, abs=1e-9)


def test_the_measured_components_reproduce_the_committed_figures():
    """At the historical rate the derivation must match the priced table."""
    doc = json.loads(
        (REPO / "logs/stages/stage-1/phase_c3/plans/c3_pricing_9probe.json").read_text())
    p = P.price_c3(1.09)
    assert p.expected_minutes == pytest.approx(doc["expected"]["minutes"], abs=0.05)
    assert p.expected_usd == pytest.approx(doc["expected"]["usd"], abs=5e-4)
    assert p.hard_usd == pytest.approx(doc["hard_ceiling"]["usd"], abs=5e-4)


def test_the_hard_ceiling_applies_the_allowances_it_declares():
    """Worst-case setup substituted; training and evaluation scaled."""
    a = P.allowances()
    comps = P.component_minutes()
    hand = sum(
        a["setup_worst_case_minutes"] if k == "setup"
        else v * a["train_and_eval_overrun_factor"]
        if k.startswith(("recovery_", "evaluation_")) else v
        for k, v in comps.items())
    assert P.price_c3(1.09).hard_minutes == pytest.approx(hand, abs=1e-6)
    #: And the hard bound is strictly above the expected one.
    p = P.price_c3(1.09)
    assert p.hard_usd > p.expected_usd


@pytest.mark.parametrize("rate,fundable", [
    (1.09, True),      # the historical observation
    (1.45, True),      # higher, still inside the envelope -> RE-PRICED
    (1.60, False),     # above what the envelope funds -> refused on arithmetic
])
def test_a_different_live_price_is_repriced_not_refused(rate, fundable):
    f = P.assess(rate, ENV)
    assert f.fundable is fundable
    #: Whatever the verdict, the price itself was computed rather than rejected.
    assert f.price.hard_usd > 0


def test_an_unfundable_price_names_which_condition_and_by_how_much():
    f = P.assess(1.60, ENV)
    assert f.fundable is False
    short = f.shortfalls()
    assert "per_session_envelope" in short
    assert short["per_session_envelope"] == pytest.approx(
        f.price.hard_usd - 30.0, abs=1e-4)
    #: The others still hold at this rate, so only one is named.
    assert "project_cap" not in short and "formal_allowance" not in short
    assert "package_total" not in short


def test_each_of_the_four_conditions_can_fail_independently():
    """All FOUR bind. A test that only ever exercises one proves one.

    This said "three" and checked three while a fourth limit existed and was
    never evaluated -- the package total. The count in the name is part of the
    claim, so it moves with the conditions.
    """
    #: cap binds: plenty of envelope and allowance, no headroom
    f = P.assess(1.09, {**ENV, "cumulative_spend_usd": 395.0,
                        "formal_remaining_usd": 50.0,
                        "per_session_envelope_usd": 100.0})
    assert f.fundable is False and "project_cap" in f.shortfalls()

    #: formal allowance binds
    f = P.assess(1.09, {**ENV, "formal_remaining_usd": 5.0,
                        "per_session_envelope_usd": 100.0,
                        "cumulative_spend_usd": 0.0})
    assert f.fundable is False and "formal_allowance" in f.shortfalls()

    #: envelope binds
    f = P.assess(1.09, {**ENV, "per_session_envelope_usd": 10.0,
                        "formal_remaining_usd": 999.0,
                        "cumulative_spend_usd": 0.0})
    assert f.fundable is False and "per_session_envelope" in f.shortfalls()

    #: package total binds, with every OTHER limit deliberately wide open --
    #: which is exactly the state the old three-condition check called
    #: fundable.
    f = P.assess(1.09, {**ENV, "package_remaining_usd": 0.9903,
                        "formal_remaining_usd": 999.0,
                        "per_session_envelope_usd": 100.0,
                        "cumulative_spend_usd": 0.0})
    assert f.fundable is False and "package_total" in f.shortfalls()


def test_a_missing_envelope_raises_rather_than_passing_silently():
    """An absent limit is not a passing limit."""
    for drop in P.REQUIRED_ENVELOPE_KEYS:
        partial = {k: v for k, v in ENV.items() if k != drop}
        with pytest.raises(P.C3PricingError, match="missing"):
            P.assess(1.09, partial)


def test_the_book_decides_which_allowance_is_charged():
    """A session that trains no probe is charged to engineering, not formal."""
    env = {**ENV, "formal_remaining_usd": 999.0,
           "engineering_remaining_usd": 0.10,
           "per_session_envelope_usd": 100.0, "cumulative_spend_usd": 0.0}
    assert P.assess(1.09, env, book="formal_allowance").fundable
    engineering = P.assess(1.09, env, book="gpu_engineering_allowance")
    assert engineering.fundable is False
    assert "gpu_engineering_allowance" in engineering.shortfalls()
    with pytest.raises(P.C3PricingError, match="unknown book"):
        P.assess(1.09, env, book="petty_cash")


def test_the_derived_ceiling_is_what_an_authorization_would_carry():
    """The envelope is not the grant -- the whole point of section 2."""
    f = P.assess(1.09, ENV)
    assert f.fundable
    assert f.price.hard_usd < f.envelopes["per_session_envelope_usd"], (
        "this assertion is only meaningful while the derived ceiling is "
        "strictly below the envelope, which is the situation that makes "
        "issuing at the envelope a real mistake rather than a hypothetical")
    d = f.as_dict()
    assert d["price"]["hard_ceiling"]["usd"] == f.price.hard_usd
    assert "the authorization receives hard_ceiling.usd" in \
        d["_the_envelope_is_not_the_grant"]


@pytest.mark.parametrize("bad", [0.0, -1.0, 1000.0, True, "1.09", None])
def test_an_implausible_or_non_numeric_rate_is_refused(bad):
    """A string rate silently becoming a number is how a launcher mis-prices."""
    with pytest.raises(P.C3PricingError):
        P.price_c3(bad)


def test_the_live_query_reads_secure_price_and_never_community_price():
    """I once reported communityPrice while the launcher priced on secure."""
    src = (REPO / "scripts/experiments/stage-1/phase_c3/formal_pricing.py").read_text()
    assert 'rows[0].get("securePrice")' in src
    #: communityPrice may be *fetched* for the record, but never substituted.
    assert 'price = rows[0].get("communityPrice")' not in src
    assert "must not fall back" in src


def test_the_live_query_sets_the_user_agent_the_edge_requires():
    """RunPod's edge answers Python-urllib with 403 on every query."""
    src = (REPO / "scripts/experiments/stage-1/phase_c3/formal_pricing.py").read_text()
    assert "USER_AGENT" in src and "User-Agent" in src


def test_the_budget_plan_reproduces_the_derived_ceiling_exactly():
    """Two models of one quantity is how a reserve gets double-counted.

    `formal_pricing` derives the hard bound from the component table;
    `c3_budget_spec` builds a `BudgetSpec` whose `plan()` re-derives it. They
    must agree, and getting there took two real corrections:

    * the spec applied `plan_session`'s own contingency ON TOP of the
      component table's overrun factor -- `$0.37` of double-counted risk,
      which the planner refused rather than silently absorbed;
    * the closeout component was passed as BOTH `transfer_minutes` and the
      artifact-recovery reserve, another `$0.28`. It is one 15-minute
      allowance; held back as the reserve it also buys a real teardown margin
      below the ceiling.

    And then the ceiling itself had to be CEILED rather than rounded: `round()`
    took it down `$0.00003`, leaving a plan that terminated fractionally above
    what it authorized.
    """
    from experiments.phase_c3 import authorization as A

    spec = A.c3_budget_spec(REPO)
    rate = A.c3_billed_rate_usd_per_hour(REPO)
    hard = A.c3_hard_ceiling_usd(REPO)
    plan = spec.plan(price_per_hour=rate, authorized_usd=hard)

    #: The planner's own terminate point is the priced hard bound. Minutes
    #: are rate-independent, so any valid rate derives the same figure.
    assert plan.hard_terminate_minutes == pytest.approx(
        P.price_c3(1.0).hard_minutes, abs=0.01)
    assert plan.hard_terminate_minutes / 60.0 * rate == pytest.approx(hard, abs=5e-4)
    #: The reserve is held back, not spent: a real teardown margin.
    assert plan.soft_stop_minutes < plan.hard_terminate_minutes
    assert plan.hard_terminate_minutes - plan.soft_stop_minutes == pytest.approx(
        plan.artifact_recovery_reserve_minutes, abs=1e-6)
    assert spec.contingency_fraction == 0.0, (
        "a contingency here would price the component table's overrun factor "
        "a second time")
    assert spec.transfer_minutes == 0.0, (
        "the closeout allowance is the recovery reserve; counting it twice "
        "put the plan above its own ceiling")


def test_the_hard_ceiling_is_ceiled_not_rounded():
    """A limit rounds down; a ceiling rounds up."""
    import math

    p = P.price_c3(1.09)
    exact = p.hard_minutes / 60.0 * p.billed_rate_usd_per_hour
    assert p.hard_usd >= exact, "the ceiling is below the work it bounds"
    assert p.hard_usd == math.ceil(exact * 10_000) / 10_000
