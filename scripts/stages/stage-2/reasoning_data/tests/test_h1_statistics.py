"""H1's decision rule on synthetic outcomes: a floor is never read as equivalence.

570 items in 60 rule families over the four static axes, two seeds -- the
shape of the formal fluid static set (interactive episodes are exploratory in
H1 and never enter the rule). Each item has a latent difficulty shared by both arms
and both seeds (that is what makes the comparison paired); an observation is
correct when the difficulty plus a little noise falls below the arm's rate.
Outcomes come from a fixed RNG per scenario.
"""
from __future__ import annotations

import random

import pytest

from stages.reasoning_data import h1_statistics as H

AXES = ("deduction", "induction", "composition", "analogy")
SEEDS = (20260726, 20260727)


def items(n=570, families=60):
    return [(f"i{k}", f"fam{k % families}", AXES[(k % families) % len(AXES)]) for k in range(n)]


def rows(rate_a, rate_b, *, seed=0, per_seed_b=None, only=None, a_only=None, noise=0.08):
    rng = random.Random(seed)
    out = []
    for item, fam, ax in items():
        u = rng.random()
        for j, s in enumerate(SEEDS):
            rb = per_seed_b[j] if per_seed_b else rate_b
            a_hit = (rate_a > 0 and u + rng.gauss(0, noise) < rate_a) if a_only is None else item in a_only
            out.append({"arm": "A", "seed": s, "item_id": item, "cluster": fam, "axis": ax,
                        "correct": a_hit})
            hit = (rb > 0 and u + rng.gauss(0, noise) < rb) if only is None else item in only
            out.append({"arm": "B", "seed": s, "item_id": item, "cluster": fam, "axis": ax,
                        "correct": hit})
    return out


def test_all_zero_is_floor_not_resolved_never_no_go():
    r = H.decide(rows(0.0, 0.0))["primary"]
    assert r["ci"] == [0.0, 0.0] and r["bootstrap"]["degenerate"]
    assert r["label"] == "NOT_RESOLVED" and r["flags"] == ["FLOOR"]
    assert r["reading"].startswith("no equivalence")


def test_sparse_successes_in_one_family_are_not_resolved():
    r = H.decide(rows(0.0, 0.0, only={"i0", "i60", "i120"}))["primary"]
    assert r["resolution"]["informative_clusters"] == 1
    assert r["label"] == "NOT_RESOLVED"


def test_both_arms_below_the_floor_stay_not_resolved_even_with_discordance():
    b = {f"i{k}" for k in range(1, 9)}                     # 8 items in 8 families
    a = {f"i{k}" for k in range(20, 25)}                   # 5 others
    r = H.decide(rows(0, 0, only=b, a_only=a))["primary"]
    assert r["resolution"]["resolved"]                    # discordance alone is not enough
    assert r["rates"]["A"] < H.FLOOR_RATE and r["rates"]["B"] < H.FLOOR_RATE
    assert r["label"] == "NOT_RESOLVED" and r["flags"] == ["FLOOR"]


def test_equality_with_one_arm_at_the_floor_is_never_no_go():
    a = {f"i{k}" for k in range(0, 570, 57)}               # A: 10 items, about 1.8%
    b = {f"i{k}" for k in range(3, 570, 29)}               # B: 20 items, about 3.5%
    r = H.decide(rows(0, 0, only=b, a_only=a))["primary"]
    assert r["resolution"]["resolved"] and -H.SESOI < r["ci"][0] and r["ci"][1] < H.SESOI
    assert r["label"] == "INCONCLUSIVE" and r["flags"] == ["FLOOR_A"]


def test_a_clear_gain_in_both_seeds_is_go():
    r = H.decide(rows(0.10, 0.35, seed=1))["primary"]
    assert r["label"] == "GO" and r["ci"][0] > H.SESOI and r["resolution"]["resolved"]


def test_a_gain_off_the_floor_is_go_with_the_floor_flag():
    r = H.decide(rows(0.0, 0.30, seed=2))["primary"]
    assert r["label"] == "GO" and r["flags"] == ["FLOOR_A"]


def test_resolved_equality_is_no_go():
    r = H.decide(rows(0.40, 0.40, seed=4))["primary"]
    assert r["resolution"]["resolved"] and r["label"] == "NO_GO"


def test_seeds_that_disagree_are_inconclusive_even_when_the_pool_clears():
    r = H.decide(rows(0.10, None, seed=5, per_seed_b=(0.42, 0.13)))["primary"]
    assert r["ci"][0] > H.SESOI
    assert min(r["per_seed_delta"].values()) < H.SESOI and r["label"] == "INCONCLUSIVE"


def test_a_clear_loss_is_reversed():
    assert H.decide(rows(0.40, 0.15, seed=6))["primary"]["label"] == "REVERSED"


def test_the_rule_is_deterministic_and_reports_its_constants():
    a, b = H.decide(rows(0.2, 0.3, seed=7)), H.decide(rows(0.2, 0.3, seed=7))
    assert a == b and a["constants"]["level"] == 0.95 and a["constants"]["iterations"] == 10_000


def test_axes_get_the_same_rule_and_holm():
    r = H.decide(rows(0.10, 0.35, seed=8))
    assert set(r["by_axis"]) == set(AXES)
    assert all("holm_gain_survives" in v for v in r["by_axis"].values())


def test_unpaired_or_duplicated_rows_are_refused():
    rs = rows(0.2, 0.2)
    with pytest.raises(ValueError, match="unpaired"):
        H.decide(rs[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        H.decide(rs + rs[:1])


def test_interactive_rows_are_reported_beside_the_decision_never_inside_it():
    rs = rows(0.2, 0.4, seed=3)
    rs[0] = {**rs[0], "axis": "interactive_adaptation"}
    with pytest.raises(ValueError, match="exploratory"):
        H.decide(rs)
