"""The mixed calibration design's properties, at $0.

Each test is one property the design or the preregistration claims: the halves
EXACTLY equal by realized positions (no tolerance), the closure rule's declared
order and its refusal when no exact set exists, episodes whole, the half/axis
labels DEPTH aggregates over, rule families interleaved, the committed profile
reproducible from its items and exact through the core's own counting, the
superseded v1 kept byte-unchanged, 4x out of scope, and the decision rule bound
by hash.
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
                     "supporting": {"subtypes": ["s1", "s2"], "balance": {"max_closing_units": 3}}}}
V2 = "configs/stages/stage-0/calibration_volume/mixed_design_v2.json"


def _pool(sup_lo=1):
    rng = random.Random(0)
    units = {}

    def add(domain, subtype, n, turns=1, fams=3, lo=500, hi=1500):
        for i in range(n):
            uid = f"{subtype}-{i}"
            items = []
            for k in range(turns):
                n = rng.randint(lo, hi)
                items.append({"item_id": f"{uid}#t{k}", "unit_id": uid, "session_id": f"{uid}#t{k}", "domain": domain,
                              "subtype": subtype, "generator_id": subtype if domain != "supporting" else None,
                              "family_id": f"{subtype}.f{i % fams}" if domain != "supporting" else None,
                              "scored_positions": n, "n_tokens": n + 8, "n_prediction_positions": n + 7,
                              "turn": k, "tags": {"assistant": list(range(7, n + 7))}})
            units[uid] = {"unit_id": uid, "items": items, "domain": domain, "subtype": subtype,
                          "family": items[0]["family_id"] or subtype,
                          "scored": sum(i["scored_positions"] for i in items)}
    add("axisA", "gA1", 30)
    add("axisA", "gA2", 30)
    add("interactive_adaptation", "gI1", 4, turns=4, fams=1, lo=2000, hi=4000)
    add("interactive_adaptation", "gI2", 4, turns=4, fams=2, lo=2000, hi=4000)
    add("supporting", "s1", 200, lo=sup_lo, hi=400)
    add("supporting", "s2", 200, lo=sup_lo, hi=400)
    return units


def _draw(fluid_nominal, seed=7, sup_lo=1):
    units = _pool(sup_lo)
    cells = M.cellmap(DESIGN, units)
    items, census = M.sample(DESIGN, units, cells, seed, fluid_nominal)
    return units, items, census


def _half(items, half):
    return sum(len(i["tags"]["assistant"]) for i in items if i["domain"] == half)


def test_halves_are_exactly_equal_by_realized_positions():
    for nominal in (6000, 12000):
        for seed in (1, 7, 11):
            _, items, census = _draw(nominal, seed)
            assert _half(items, "fluid") == _half(items, "supporting") == census["_balance"]["target"]
            assert _half(items, "fluid") >= nominal


def test_sample_stops_with_evidence_when_no_exact_set_exists():
    # every supporting unit has >= 50 positions and the fitted prefixes leave 31: nothing closes it
    with pytest.raises(M.Infeasible) as e:
        _draw(12000, seed=11, sup_lo=50)
    assert e.value.evidence["residual"] == 31 and e.value.evidence["closest_gap"]["one_unit"] == 19


def test_fit_prefix_never_exceeds_its_quota_and_skips_nothing():
    units = _pool()
    uids = [u for u in units if units[u]["subtype"] == "s1"]
    order = M.family_order(("supporting", "s1", "*"), uids, units, 5)
    for q in range(500, 20000, 731):
        taken, got, rest = M._take(("supporting", "s1", "*"), uids, units, 5, q, fit=True)
        assert got <= q and taken + rest == order
        assert not rest or got + units[rest[0]]["scored"] > q


def test_close_exact_takes_the_fewest_units_earliest_in_the_sequence():
    sizes = {"a": 5, "b": 3, "c": 7, "d": 2, "e": 8, "f": 1}
    seq = list(sizes)
    assert M.close_exact(0, seq, sizes, 3) == []
    assert M.close_exact(7, seq, sizes, 3) == ["c"]           # one unit beats any pair (a+d)
    assert M.close_exact(10, seq, sizes, 3) == ["b", "c"]     # no single unit; b+c is the pair whose last member comes first
    assert M.close_exact(12, seq, sizes, 3) == ["a", "c"]     # a+c and d+? ...: the last member earliest wins
    assert M.close_exact(20, seq, sizes, 3) == ["a", "c", "e"]
    with pytest.raises(M.Infeasible) as e:
        M.close_exact(40, seq, sizes, 3)
    assert e.value.evidence["closest_gap"] == {"one_unit": 32, "two_units": 25}
    assert M.interleave([["a1", "a2", "a3"], ["b1"], ["c1", "c2"]]) == ["a1", "b1", "c1", "a2", "c2", "a3"]


def test_require_exact_refuses_any_difference():
    ok = {"token_axis": {"fluid": 10, "supporting": 10}, "prediction_axis": {"fluid": 10, "supporting": 10},
          "depth_half_weight": {"fluid": 0.5, "supporting": 0.5}}
    M.require_exact(ok)
    for bad in ({**ok, "token_axis": {"fluid": 10, "supporting": 9}},
                {**ok, "prediction_axis": {"fluid": 11, "supporting": 10}},
                {**ok, "depth_half_weight": {"fluid": 0.6, "supporting": 0.4}}):
        with pytest.raises(SystemExit):
            M.require_exact(bad)


def test_episodes_are_whole():
    units, items, _ = _draw(12000)
    by_unit = {}
    for it in items:
        by_unit.setdefault(it["unit_id"], set()).add(it["turn"])
    for uid, turns in by_unit.items():
        assert turns == {it["turn"] for it in units[uid]["items"]}, f"{uid} split"
    assert len({it["item_id"] for it in items}) == len(items)


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


def test_single_profile_tree_is_34_and_the_historical_tree_92():
    assert sum(M.TREE.values()) == 34 and sum(M.TREE_2P.values()) == 92
    assert all(M.DEPTH_MAX[lv] <= M.TREE[lv] for lv in M.TREE)


def test_padded_slots_follow_length_sorted_micro_batches():
    assert M.padded_slots([1, 5, 2, 9, 3], 3) == 3 * 3 + 2 * 9


def test_committed_profile_is_exact_through_the_core():
    from aadistill.initialization.calibration.profiles import load_profiles, mixture_content_sha256
    design = json.loads((REPO / V2).read_text())
    doc = json.loads((REPO / design["output"]["profiles_doc"]).read_text())
    load_profiles(doc, register=False)
    assert hashlib.sha256((REPO / doc["design"]).read_bytes()).hexdigest() == doc["design_sha256"]
    (p,) = doc["profiles"]
    assert (p["profile_id"], p["version"]) == ("calib.s0m_mix_2x", 2)
    half = p["metadata"]["uesp_by_half"]
    assert half["fluid"] == half["supporting"] and p["metadata"]["uesp"] == 2 * half["fluid"]
    path = REPO / p["items_path"]
    if not path.exists():
        pytest.skip("profile items are local artifacts (rebuild with build_mixed_profiles.py)")
    body = path.read_bytes()
    assert hashlib.sha256(body).hexdigest() == p["items_file_sha256"]
    items = [json.loads(line) for line in body.decode().splitlines()]
    assert mixture_content_sha256(items) == p["content_sha256"]
    view = M.operator_view(items)
    M.require_exact(view)
    assert view["token_axis"] == view["prediction_axis"] == half
    assert sorted(view["depth_domain_map"]["supporting"]) == sorted(design["halves"]["supporting"]["subtypes"])
    assert sorted(view["depth_domain_map"]["fluid"]) == sorted(design["halves"]["fluid"]["axes"])


def test_superseded_v1_is_kept_unchanged_and_4x_is_out_of_scope():
    design = json.loads((REPO / V2).read_text())
    sup = design["supersedes"]
    for k in ("design", "profiles", "feasibility"):
        assert hashlib.sha256((REPO / sup[k]["path"]).read_bytes()).hexdigest() == sup[k]["sha256"], k
    old = {f"{p['profile_id']}@v{p['version']}": p["content_sha256"]
           for p in json.loads((REPO / sup["profiles"]["path"]).read_text())["profiles"]}
    assert old == sup["profiles"]["content_sha256"]
    assert list(design["doses"]) == ["2x"] and design["search"]["profiles"] == ["calib.s0m_mix_2x@v2"]
    app = json.loads((REPO / "logs/stages/stage-0/calibration_volume/plans/mixed_application.json").read_text())
    assert "entry_4x" not in app and set(app["phases"]) == {"MC1_search", "MC2_recovery_and_evaluation"}
    keys = []

    def walk(o):
        if isinstance(o, dict):
            keys.extend(o)
            for v in o.values():
                walk(v)
    walk(app["money"])
    walk(app["phases"])
    assert not [k for k in keys if "4x" in k]
    env = app["money"]["envelope"]
    assert env["total_usd"] == 33.05 == round(sum(v for k, v in env.items() if not k.startswith("_") and k != "total_usd"), 2)
    for rec in app["records"].values():
        assert hashlib.sha256((REPO / rec["path"]).read_bytes()).hexdigest() == rec["sha256"], rec["path"]


def test_preregistration_binds_the_rule_and_the_profile():
    prereg = json.loads((REPO / "logs/stages/stage-0/calibration_volume/plans/mixed_preregistration.json").read_text())
    MS.check_bound(prereg)
    doc = json.loads((REPO / prereg["calibration_profile"]["profiles_doc"]).read_text())
    (p,) = doc["profiles"]
    assert prereg["calibration_profile"]["id"] == f"{p['profile_id']}@v{p['version']}"
    assert prereg["calibration_profile"]["content_sha256"] == p["content_sha256"]
    h1pa = json.loads((REPO / "logs/stages/stage-2/reasoning_data/plans/h1pa_preregistration.json").read_text())
    assert prereg["evaluation"]["decoding"] == h1pa["evaluation"]["decoding"]
    assert prereg["evaluation"]["set"] == h1pa["evaluation"]["set"]
    assert prereg["decision_rule"]["constants"] == h1pa["decision_rule"]["constants"]
    want = prereg["evaluation"]["decoding"]["sha256"]
    with pytest.raises(ValueError, match="unknown arms"):
        MS.decide([{"arm": "D4-Original", "decoding_sha256": want}], prereg=prereg)
    with pytest.raises(ValueError, match="never mixed"):
        MS.decide([{"arm": "Mix-Augmented", "decoding_sha256": "greedy"}], prereg=prereg)
