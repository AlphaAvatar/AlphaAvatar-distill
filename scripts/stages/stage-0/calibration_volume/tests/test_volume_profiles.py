"""The volume sampler's design properties, on a synthetic pool, at $0.

Each test is one property the volume design claims and an analysis relies on:
nesting within a seed (the higher dose contains the lower), independence
across seeds (a different seed is a different draw), the episode as the unit
(its turns never split), and composition held fixed (every cell present at
every dose, quotas in proportion).
"""
from __future__ import annotations

import json
import random

import pytest

from stages.calibration_volume import build_volume_profiles as B


def _pool():
    units = {}
    rng = random.Random(0)
    for dom, subs in {"axisA": ["g1", "g2"], "supporting": ["s1"]}.items():
        for sub in subs:
            for i in range(40):
                uid = f"{sub}-{i}"
                n_turns = 3 if sub == "g2" else 1
                items = [{"item_id": f"{uid}#t{k}", "unit_id": uid, "domain": dom,
                          "subtype": sub, "family_id": f"{sub}.f{i % 4}",
                          "scored_positions": rng.randint(500, 1500), "turn": k,
                          "tags": {"assistant": []}, "ids": [1, 2], "n_tokens": 2}
                         for k in range(n_turns)]
                units[uid] = {"unit_id": uid, "items": items, "domain": dom, "subtype": sub,
                              "family": items[0]["family_id"],
                              "scored": sum(i["scored_positions"] for i in items)}
    return units


DESIGN = {"reference_uesp": 6000,
          "cells": {"axisA": {"g1": ["*"], "g2": ["*"]}, "supporting": {"s1": ["*"]}}}
WEIGHTS = {"axisA": 0.75, "supporting": 0.25}


def _sample(units, cellmap, dose, seed):
    items, census = B.sample(DESIGN, units, cellmap, WEIGHTS, dose, seed)
    return {i["unit_id"] for i in items}, items, census


def test_nested_within_seed_and_cells_always_present():
    units = _pool()
    cellmap = B.cells(DESIGN, units)
    prev = set()
    for dose in (1, 2, 4):
        uids, items, census = _sample(units, cellmap, dose, seed=11)
        assert prev <= uids, "a higher dose must contain the lower one"
        assert {i["subtype"] for i in items} == {"g1", "g2", "s1"}
        assert not census.get("_short_cells")
        prev = uids


def test_seeds_are_different_draws():
    units = _pool()
    cellmap = B.cells(DESIGN, units)
    a, _, _ = _sample(units, cellmap, 1, seed=1)
    b, _, _ = _sample(units, cellmap, 1, seed=2)
    assert a != b
    assert len(a & b) < len(a)


def test_episode_turns_travel_together():
    units = _pool()
    cellmap = B.cells(DESIGN, units)
    _, items, _ = _sample(units, cellmap, 2, seed=5)
    by_unit = {}
    for it in items:
        by_unit.setdefault(it["unit_id"], []).append(it["turn"])
    for uid, turns in by_unit.items():
        assert sorted(turns) == list(range(len(units[uid]["items"])))


def test_quotas_follow_the_weights():
    units = _pool()
    cellmap = B.cells(DESIGN, units)
    _, items, census = _sample(units, cellmap, 4, seed=3)
    prim = sum(i["scored_positions"] for i in items if i["domain"] == "axisA")
    supp = sum(i["scored_positions"] for i in items if i["domain"] == "supporting")
    assert prim >= 0.75 * 4 * 6000 and supp >= 0.25 * 4 * 6000
    # overshoot is at most one unit per cell
    assert prim < 0.75 * 4 * 6000 + 2 * max(u["scored"] for u in units.values())


def test_a_missing_cell_refuses():
    units = _pool()
    with pytest.raises(SystemExit):
        B.cells({"cells": {"axisA": {"g9": ["*"]}}}, units)
