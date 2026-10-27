"""Nested ladders from one master order: nesting, composition, coverage, reserve.

Units are synthetic and opaque (strata and families are arbitrary strings), so
nothing here can pass by knowing a real corpus. The properties a scaling claim
leans on each get a test that a plausible wrong implementation fails:
re-sampling per rung breaks nesting; ignoring shares breaks composition; plain
hash order (no round-robin) leaves a small rung on few families; a reserve that
takes a family's only unit removes it from every rung.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.nested_ladder import (  # noqa: E402
    LadderError, LadderUnit, build_ladder, composition_capacity, cut_rungs,
    family_round_robin, nested_order, require_nested, rung_accounting,
)


def corpus(n_per_family=(30, 20, 10), families=(8, 5, 3), base=100):
    units = []
    for s_i, (n, f) in enumerate(zip(n_per_family, families)):
        stratum = f"s{s_i}"
        for fam in range(f):
            for k in range(n):
                size = base + 17 * ((k * 7 + fam * 3 + s_i) % 11)
                units.append(LadderUnit(
                    unit_id=f"{stratum}-f{fam}-{k}", stratum=stratum, size=size,
                    family=f"{stratum}.fam{fam}", counts={"reasoning": size - 10, "answer": 10},
                    attributes={"generator": f"g{fam % 2}"}))
    return units


SHARES = {"s0": 0.5, "s1": 0.3, "s2": 0.2}


def test_rungs_are_prefixes_and_nested():
    lad = build_ladder(corpus(), SHARES, [5_000, 10_000, 20_000], salt="t")
    ids = [lad["order"][:r["n_units"]] for r in lad["rungs"]]
    require_nested(ids)
    assert all(r["accounting"]["size"] >= r["target"] for r in lad["rungs"])


def test_order_is_deterministic_and_input_order_free():
    units = corpus()
    a = nested_order(units, SHARES, salt="t")
    b = nested_order(list(reversed(units)), SHARES, salt="t")
    assert [u.unit_id for u in a] == [u.unit_id for u in b]
    c = nested_order(units, SHARES, salt="other")
    assert [u.unit_id for u in a] != [u.unit_id for u in c]


@pytest.mark.parametrize("target", [3_000, 8_000, 15_000])
def test_every_rung_holds_the_declared_composition(target):
    order = nested_order(corpus(), SHARES, salt="t")
    (row,) = cut_rungs(order, [target], shares=SHARES)
    acc = rung_accounting(order[:row["n_units"]], SHARES)
    # Largest remainder keeps each stratum within one unit's size of its share.
    assert acc["max_share_deviation"] < 300 / acc["size"] + 1e-9


def test_round_robin_spreads_a_small_prefix_over_families():
    units = [u for u in corpus() if u.stratum == "s0"]
    rr = family_round_robin(units, salt="t")
    assert len({u.family for u in rr[:8]}) == 8
    # A plain salted sort, the wrong implementation, concentrates the prefix.
    import hashlib
    plain = sorted(units, key=lambda u: hashlib.sha256(f"t|{u.unit_id}".encode()).hexdigest())
    assert len({u.family for u in plain[:8]}) < 8


def test_counts_and_attributes_are_reported_not_used_to_cut():
    order = nested_order(corpus(), SHARES, salt="t")
    acc = rung_accounting(order[:50], SHARES)
    assert acc["counts"]["reasoning"] + acc["counts"]["answer"] == acc["size"]
    assert acc["distinct_attribute_values"] == {"generator": 2}


def test_rung_beyond_composition_capacity_is_refused():
    units = corpus(n_per_family=(30, 20, 1), families=(8, 5, 1))  # s2 tiny
    order = nested_order(units, SHARES, salt="t")
    cap = composition_capacity(order, SHARES)
    with pytest.raises(LadderError, match="composition capacity"):
        cut_rungs(order, [cap + 1_000], shares=SHARES)
    rows = cut_rungs(order, [cap + 1_000], shares=SHARES, allow_composition_drift=True)
    assert rows[0]["composition_drift"] is True


def test_unreachable_rung_is_reported():
    order = nested_order(corpus(), SHARES, salt="t")
    total = sum(u.size for u in order)
    rows = cut_rungs(order, [10, total + 1])
    assert rows[0]["reachable"] and not rows[1]["reachable"]


def test_reserve_is_disjoint_and_keeps_every_family():
    units = corpus()
    lad = build_ladder(units, SHARES, [5_000, 20_000], salt="t", reserve=4_000)
    reserved = set(lad["reserved"])
    assert reserved and not reserved & set(lad["order"])
    fams = {u.family for u in units if u.unit_id in set(lad["order"])}
    assert fams == {u.family for u in units}
    assert lad["reserved_accounting"]["size"] >= 4_000


def test_targets_must_increase_and_ids_must_be_unique():
    order = nested_order(corpus(), SHARES, salt="t")
    with pytest.raises(LadderError):
        cut_rungs(order, [100, 100])
    dup = corpus()[:2] + corpus()[:1]
    with pytest.raises(LadderError, match="duplicate"):
        nested_order(dup, SHARES, salt="t")


def test_a_declared_stratum_without_units_is_refused():
    with pytest.raises(LadderError, match="no units"):
        nested_order(corpus(), {**SHARES, "missing": 0.1}, salt="t")
