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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3 import formal_pricing as P  # noqa: E402

#: The amended envelopes, as parameters rather than as live reads, so these
#: tests describe the decision rule and not today's ledger position.
ENV = {"per_session_envelope_usd": 30.0, "project_cap_usd": 400.0,
       "cumulative_spend_usd": 350.0307, "formal_remaining_usd": 32.3824}


def test_the_rate_is_an_input_and_nothing_hardcodes_1_09():
    src = (REPO / "scripts/experiments/phase_c3/formal_pricing.py").read_text()
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
    #: The other two still hold at this rate, so only one is named.
    assert "project_cap" not in short and "formal_allowance" not in short


def test_each_of_the_three_conditions_can_fail_independently():
    """All three bind. A test that only ever exercises one proves one."""
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


def test_the_derived_ceiling_is_what_an_authorization_would_carry():
    """The envelope is not the grant -- the whole point of section 2."""
    f = P.assess(1.09, ENV)
    assert f.fundable
    assert f.price.hard_usd < f.per_session_envelope_usd, (
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
    src = (REPO / "scripts/experiments/phase_c3/formal_pricing.py").read_text()
    assert 'rows[0].get("securePrice")' in src
    #: communityPrice may be *fetched* for the record, but never substituted.
    assert 'price = rows[0].get("communityPrice")' not in src
    assert "must not fall back" in src


def test_the_live_query_sets_the_user_agent_the_edge_requires():
    """RunPod's edge answers Python-urllib with 403 on every query."""
    src = (REPO / "scripts/experiments/phase_c3/formal_pricing.py").read_text()
    assert "USER_AGENT" in src and "User-Agent" in src
