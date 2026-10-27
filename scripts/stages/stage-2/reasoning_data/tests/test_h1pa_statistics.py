"""H1-PA's decision rule on synthetic outcomes: three arms, two decided comparisons, one descriptive.

The shape is the formal fluid static set's (570 items, 60 rule families, four
axes, two seeds). Each item has one latent difficulty shared by every arm and
seed (that is what pairs the comparisons); an observation is correct when the
difficulty plus a little noise falls below the arm's rate.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from stages.reasoning_data import h1pa_statistics as PS

REPO = Path(__file__).resolve().parents[5]
PREREG = REPO / "logs/stages/stage-2/reasoning_data/plans/h1pa_preregistration.json"
AXES = ("deduction", "induction", "composition", "analogy")
SEEDS = (20260726, 20260801)


def prereg():
    return json.loads(PREREG.read_text())


def rows(rates: dict, *, seed=0, noise=0.08, decoding=None):
    dec = decoding or prereg()["evaluation"]["decoding"]["sha256"]
    rng = random.Random(seed)
    out = []
    for k in range(570):
        fam, ax, u = f"fam{k % 60}", AXES[(k % 60) % 4], rng.random()
        for s in SEEDS:
            for arm, rate in rates.items():
                out.append({"arm": arm, "seed": s, "item_id": f"i{k}", "cluster": fam, "axis": ax,
                            "correct": rate > 0 and u + rng.gauss(0, noise) < rate, "decoding_sha256": dec})
    return out


def test_the_preregistration_binds_the_rule_on_disk():
    PS.check_bound(prereg())
    assert prereg()["primary_comparison"] == "B" and prereg()["decision_rule"]["rule_id"] == PS.RULE_ID


def test_each_comparison_gets_its_own_label_and_c_none():
    r = PS.decide(rows({"PCA-Original": 0.10, "D4-Original": 0.10, "D4-Augmented": 0.30}), prereg=prereg())
    c = r["comparisons"]
    assert c["A"]["contrast"] == "D4-Original - PCA-Original" and c["A"]["primary"]["label"] == "NO_GO"
    assert c["B"]["contrast"] == "D4-Augmented - D4-Original" and c["B"]["primary"]["label"] == "GO"
    assert r["primary_label"] == "GO"
    assert c["C"]["role"] == "descriptive" and c["C"]["label"] is None and "primary" not in c["C"]
    assert c["C"]["delta"] > 0.05


def test_reversed_and_inconclusive_are_reachable():
    c = PS.decide(rows({"PCA-Original": 0.30, "D4-Original": 0.10, "D4-Augmented": 0.15}),
                  prereg=prereg())["comparisons"]
    assert c["A"]["primary"]["label"] == "REVERSED"
    assert c["B"]["primary"]["label"] == "INCONCLUSIVE"


def test_both_arms_at_the_floor_are_not_resolved():
    c = PS.decide(rows({"PCA-Original": 0.0, "D4-Original": 0.0, "D4-Augmented": 0.0}),
                  prereg=prereg())["comparisons"]
    assert c["B"]["primary"]["label"] == "NOT_RESOLVED" and c["B"]["primary"]["flags"] == ["FLOOR"]


def test_a_greedy_row_refuses():
    r = rows({"PCA-Original": 0.1, "D4-Original": 0.1, "D4-Augmented": 0.1})
    r[0] = {**r[0], "decoding_sha256": None}
    with pytest.raises(ValueError, match="never mixed"):
        PS.decide(r, prereg=prereg())


def test_a_changed_rule_refuses(tmp_path):
    p = prereg()
    p["decision_rule"]["bound_sha256"][PS.BOUND[1]] = "0" * 64
    with pytest.raises(ValueError, match="differs from the preregistered"):
        PS.decide(rows({"PCA-Original": 0.1, "D4-Original": 0.1, "D4-Augmented": 0.1}), prereg=p)


def test_interactive_rows_are_refused_by_the_rule_they_apply():
    r = rows({"PCA-Original": 0.1, "D4-Original": 0.1, "D4-Augmented": 0.1})
    r[0] = {**r[0], "axis": "interactive_adaptation"}
    with pytest.raises(ValueError, match="exploratory"):
        PS.decide(r, prereg=prereg())
