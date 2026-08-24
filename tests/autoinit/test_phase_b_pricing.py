"""Phase B's repricing: what paid work is actually still owed.

The point of the cross-phase procedure is that Phase A's probes are *evidence*,
not something to buy again. A repricing that quietly re-billed them, or that
built its best and worst cases out of two incompatible worlds, would produce a
budget ask nobody could act on.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from price_phase_b import (  # noqa: E402
    CONTROL, PHASE_A_FINALISTS, PHASE_B_SEARCHED_LEAVES, SURVIVORS_AT_SB,
    observed_probes, price,
)


def test_the_historical_inventory_is_read_off_disk_not_assumed():
    """Attempt 7 ran eleven probes; that is what the pricing may reuse."""
    seen = observed_probes()
    assert sum(len(v) for v in seen.values()) == 11, seen
    for finalist in PHASE_A_FINALISTS:
        assert seen[finalist] == {"sa", "sb", "sc"}, finalist
    # The fact that drives the whole sc worst case, and the easiest to get wrong.
    assert seen[CONTROL] == {"sa", "sb"}, "the control has no sc on record"


def test_the_three_unadmitted_phase_a_leaves_are_visible_but_unused():
    """They hold sa probes and are retained off-pod. The procedure admits only
    the two finalists, so their evidence is on record and deliberately unspent —
    which is a cheap thing for a reviewer to revisit, and invisible if unstated."""
    seen = observed_probes()
    others = set(seen) - set(PHASE_A_FINALISTS) - {CONTROL}
    assert others == {"158b96cf651f", "281a02c3ac18", "4e429f7ed722"}
    assert all(seen[o] == {"sa"} for o in others)
    assert "not_admitted" in price()["reuse"]


def test_reuse_removes_the_priors_from_the_sa_bill():
    p = price()["probes"]
    # 8 candidates at sa; the three priors are already observed there.
    assert p["sa_missing"] == PHASE_B_SEARCHED_LEAVES
    assert price()["procedure"]["candidates_at_sa"]["total"] == \
        PHASE_B_SEARCHED_LEAVES + len(PHASE_A_FINALISTS) + 1


def test_the_best_and_worst_cases_describe_ONE_world_each():
    """The trap: pricing sb as if the survivors were new while pricing sc as if
    they were the priors. Both ends must be internally consistent."""
    p = price()["probes"]
    # Worst: both survivors are new, so both owe sb AND sc, and the control owes
    # sc too because it has none on record.
    assert p["sb_missing_high"] == SURVIVORS_AT_SB
    assert p["sc_missing_high"] == SURVIVORS_AT_SB + 1
    assert p["sc_missing_high"] > p["sb_missing_high"], (
        "the control has no sc but does have sb, so sc must cost one more "
        "than sb in the same world")
    # Best: the priors survive, their sb is on record, and no tie-break fires.
    assert p["sb_missing_low"] == 0 and p["sc_missing_low"] == 0
    assert p["total_low"] == 5 and p["total_high"] == 10


def test_the_totals_are_bounded_and_ordered():
    r = price()
    lo, hi = r["total"]["expected_usd"], r["total"]["hard_usd"]
    assert 0 < lo < hi
    # Composition: the reported totals really are search + probes + reserve.
    p, s = r["probes"], r["search"]
    assert abs(lo - (s["usd_low"] + p["total_low"] * p["per_probe_usd"]
                     + r["setup_reserve_usd"])) < 0.01
    assert abs(hi - (s["usd_high"] + p["total_high"] * p["per_probe_usd"]
                     + r["setup_reserve_usd"])) < 0.01


def test_it_prices_a_two_profile_search_not_a_one_profile_one():
    s = price()["search"]
    # P=1 is 7-13 leaves and ~106 GiB; P=2 is 8-20 and ~245 GiB. Pricing Phase B
    # at P=1 would understate both the bill and the disk it must be given.
    assert s["leaves_max"] >= 20
    assert s["peak_storage_gib_working"] > 200
    assert s["provision_container_disk_gib"] >= 300


def test_the_record_does_not_claim_to_be_an_authorization():
    r = price()
    assert "NOT an authorization" in r["_contract"]
    assert "grant" in r["_contract"]
