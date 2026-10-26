"""The mixed calibration design's properties, at $0.

Each test is one property the design or the preregistration claims: the 50/50
split by realized positions, 2x nested in 4x, episodes whole, the half/axis
labels DEPTH aggregates over, rule families interleaved, the committed
profiles reproducible from their items, and the decision rule bound by hash.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from stages.calibration_volume import build_mixed_profiles as M
from stages.calibration_volume import mixed_statistics as MS

REPO = Path(__file__).resolve().parents[5]
DESIGN = {"halves": {"fluid": {"axes": {"axisA": ["gA1", "gA2"], "interactive_adaptation": ["gI1", "gI2"]}},
                     "supporting": {"subtypes": ["s1", "s2"]}}}


def _pool():
    rng = random.Random(0)
    units = {}

    def add(domain, subtype, n, turns=1, fams=3, lo=500, hi=1500):
        for i in range(n):
            uid = f"{subtype}-{i}"
            items = [{"item_id": f"{uid}#t{k}", "unit_id": uid, "session_id": f"{uid}#t{k}", "domain": domain,
                      "subtype": subtype, "generator_id": subtype if domain != "supporting" else None,
                      "family_id": f"{subtype}.f{i % fams}" if domain != "supporting" else None,
                      "scored_positions": rng.randint(lo, hi), "n_tokens": 2, "n_prediction_positions": 1,
                      "turn": k, "tags": {"assistant": []}, "ids": [1, 2]} for k in range(turns)]
            units[uid] = {"unit_id": uid, "items": items, "domain": domain, "subtype": subtype,
                          "family": items[0]["family_id"] or subtype,
                          "scored": sum(i["scored_positions"] for i in items)}
    add("axisA", "gA1", 30)
    add("axisA", "gA2", 30)
    add("interactive_adaptation", "gI1", 4, turns=4, fams=1, lo=2000, hi=4000)
    add("interactive_adaptation", "gI2", 4, turns=4, fams=2, lo=2000, hi=4000)
    add("supporting", "s1", 200, lo=50, hi=400)
    add("supporting", "s2", 200, lo=50, hi=400)
    return units


def _draw(fluid_nominal, seed=7):
    units = _pool()
    cells = M.cellmap(DESIGN, units)
    items, census = M.sample(DESIGN, units, cells, seed, fluid_nominal)
    return units, items, census


def test_halves_are_50_50_by_realized_positions():
    for nominal in (6000, 12000):
        units, items, census = _draw(nominal)
        fluid = sum(i["scored_positions"] for i in items if i["domain"] == "fluid")
        sup = sum(i["scored_positions"] for i in items if i["domain"] == "supporting")
        # within half the largest supporting unit per supporting cell
        assert abs(fluid - sup) <= 2 * 400 / 2 + 1
        assert sup > 0 and fluid >= nominal


def test_2x_is_nested_in_4x_and_episodes_are_whole():
    units, small, _ = _draw(6000)
    _, large, _ = _draw(12000)
    a, b = {i["unit_id"] for i in small}, {i["unit_id"] for i in large}
    assert a < b
    for items in (small, large):
        by_unit = {}
        for it in items:
            by_unit.setdefault(it["unit_id"], set()).add(it["turn"])
        for uid, turns in by_unit.items():
            assert turns == {it["turn"] for it in units[uid]["items"]}, f"{uid} split"


def test_labels_are_half_and_axis():
    _, items, census = _draw(6000)
    assert {i["domain"] for i in items} == {"fluid", "supporting"}
    assert {i["subtype"] for i in items if i["domain"] == "fluid"} == {"axisA", "interactive_adaptation"}
    assert {i["subtype"] for i in items if i["domain"] == "supporting"} == {"s1", "s2"}
    # the interactive axis is ONE cell over both generators
    assert [c for c in census if c.startswith("fluid/interactive_adaptation")] == ["fluid/interactive_adaptation/*"]


def test_family_order_spans_families_first():
    units = _pool()
    uids = [u for u in units if units[u]["subtype"] == "gA1"]
    order = M.family_order(("fluid", "axisA", "gA1"), uids, units, seed=3)
    assert sorted(order) == sorted(uids)
    assert len({units[u]["family"] for u in order[:3]}) == 3
    assert order == M.family_order(("fluid", "axisA", "gA1"), uids, units, seed=3)


def test_nearest_stop_is_monotone_in_the_quota():
    units = _pool()
    uids = [u for u in units if units[u]["subtype"] == "s1"]
    prev = 0
    for q in range(0, 20000, 137):
        taken, got = M._take(("supporting", "s1", "*"), uids, units, 5, q, nearest=True)
        assert len(taken) >= max(prev, 1)
        prev = len(taken)


def test_single_profile_tree_is_34_and_the_historical_tree_92():
    assert sum(M.TREE.values()) == 34 and sum(M.TREE_2P.values()) == 92
    assert all(M.DEPTH_MAX[lv] <= M.TREE[lv] for lv in M.TREE)


def test_padded_slots_follow_length_sorted_micro_batches():
    assert M.padded_slots([1, 5, 2, 9, 3], 3) == 3 * 3 + 2 * 9


def test_committed_profiles_reproduce_from_their_items():
    from aadistill.initialization.calibration.profiles import load_profiles, mixture_content_sha256
    doc = json.loads((REPO / "configs/stages/stage-0/calibration_volume/profiles_mixed.json").read_text())
    load_profiles(doc, register=False)
    design = REPO / doc["design"]
    assert hashlib.sha256(design.read_bytes()).hexdigest() == doc["design_sha256"]
    for p in doc["profiles"]:
        path = REPO / p["items_path"]
        if not path.exists():
            pytest.skip("profile items are local artifacts (rebuild with build_mixed_profiles.py)")
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == p["items_file_sha256"]
        items = [json.loads(line) for line in body.decode().splitlines()]
        assert mixture_content_sha256(items) == p["content_sha256"]
        fluid = sum(len(i["tags"]["assistant"]) for i in items if i["domain"] == "fluid")
        assert abs(fluid / p["metadata"]["uesp"] - 0.5) < 0.01


def test_preregistration_binds_the_rule():
    prereg = json.loads((REPO / "logs/stages/stage-0/calibration_volume/plans/mixed_preregistration.json").read_text())
    MS.check_bound(prereg)
    h1pa = json.loads((REPO / "logs/stages/stage-2/reasoning_data/plans/h1pa_preregistration.json").read_text())
    assert prereg["evaluation"]["decoding"] == h1pa["evaluation"]["decoding"]
    assert prereg["evaluation"]["set"] == h1pa["evaluation"]["set"]
    assert prereg["decision_rule"]["constants"] == h1pa["decision_rule"]["constants"]
    want = prereg["evaluation"]["decoding"]["sha256"]
    with pytest.raises(ValueError, match="unknown arms"):
        MS.decide([{"arm": "D4-Original", "decoding_sha256": want}], prereg=prereg)
    with pytest.raises(ValueError, match="never mixed"):
        MS.decide([{"arm": "Mix-Augmented", "decoding_sha256": "greedy"}], prereg=prereg)
