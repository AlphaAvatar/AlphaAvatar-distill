"""The packed search's v2 inputs from synthetic tool items over the real v1 calibration and suite (temp dirs only)."""
from __future__ import annotations

import json

import pytest

from stages.teacher_modes import packed_inputs as PI

pytestmark = pytest.mark.skipif(not (PI.SUITE_V1 / "items.jsonl").is_file(),
                                reason="the v1 calibration and suite artifacts are not on this machine")


def tool_items(mode: str, split: str, tasks: int, turns: int = 2, per_turn: int = 300) -> list[dict]:
    """Synthetic admitted tool items in `tool_episodes.item_of_turn`'s form."""
    out = []
    for k in range(tasks):
        for t in range(turns):
            prompt, n = 50, per_turn
            ids = [7] * prompt + [9] * n
            base = prompt - 1
            tags = {"assistant": [base + j for j in range(n)], "eos": [base + n - 1],
                    "final_answer": [base + j for j in range(2, n - 1)],
                    "tool_call_open": [base + 3], "tool_call_close": [base + 8]}
            if mode == "H-T":
                tags["think_close"] = [base + 1]
            out.append({"item_id": f"tc1-{split}{k}|{mode}|t{t}", "task_id": f"tc1-{split}{k}", "turn": t, "ids": ids,
                        "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1, "tags": tags,
                        "domain": "supporting", "subtype": f"tool_calling@v1|{mode}", "type": "tool_calling@v1",
                        "mode": mode, "critical_group": mode, "split": split, "source": {"dataset": "toy"},
                        "family_id": "fam", "category": "multi_turn"})
    return out


def plan_for(items: dict, cal_target: float = 4200.0, se_target: float = PI.SE_POSITIONS) -> tuple[dict, dict]:
    """A one-category generation plan (these items are all multi_turn) and the frozen orders of their tasks."""
    cells = [{"split": "calibration", "category": "multi_turn", "unit": "scored_positions", "target": cal_target,
              "min_tasks": 0},
             {"split": "state_eval", "category": "multi_turn", "unit": "scored_positions", "target": se_target,
              "min_tasks": PI.SE_MIN_TASKS}]
    orders = {}
    for rows in items.values():
        for x in rows:
            orders.setdefault(x["split"], [])
            if x["task_id"] not in orders[x["split"]]:
                orders[x["split"]].append(x["task_id"])
    return {"lineages": {"hybrid": {"modes": {m: {"cells": cells} for m in PI.MODES}}}}, orders


@pytest.fixture(scope="module", autouse=True)
def the_plan():
    """profiles_v2 / suite_v2 select tool episodes through the frozen generation plan (`tool_episodes_by_plan`)."""
    from stages.teacher_modes import tool_generation_plan as TGP
    mp = pytest.MonkeyPatch()
    items = {m: tool_items(m, "calibration", 40) + tool_items(m, "state_eval", 40) for m in PI.MODES}
    mp.setattr(TGP, "for_consumers", lambda *a, **k: plan_for(items))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("v2")
    items = {m: tool_items(m, "calibration", 12) + tool_items(m, "state_eval", 25) for m in PI.MODES}
    res = PI.build(items, profiles_out=d / "profiles.json", suite_out=d / "suite", items_out=d / "items",
                   types_out=d / "types.json", freeze_out=d / "freeze.json")
    return d, res


def test_profiles_keep_v1s_cells_and_add_the_tool_cell(built):
    d, res = built
    assert res["profiles_status"].startswith("COMPLETE")
    doc = json.loads((d / "profiles.json").read_text())
    v1 = json.loads(PI.PROFILES_V1.read_text())
    for p1, p2 in zip(v1["profiles"], doc["profiles"]):
        assert p2["version"] == 2 and p2["profile_id"] == p1["profile_id"]
        old = PI.jsonl(PI.REPO / p1["items_path"])
        new = PI.jsonl(d / "items" / f"{p1['profile_id']}.v2.items.jsonl")
        census = doc["census"][p1["profile_id"]]
        for key, c in census.items():
            dom, sm = key.split("/", 1)
            o = [x["item_id"] for x in old if (x["domain"], x["subtype"]) == (dom, sm)]
            n = [x["item_id"] for x in new if (x["domain"], x["subtype"]) == (dom, sm)]
            if dom != "supporting":
                assert n == o, key                                    # v1's cell, item for item
            elif not sm.startswith("tool_calling@v1"):
                assert n == o[:len(n)] and c["scored"] >= c["quota"] - PI.QUOTA_EPS, key   # a prefix of v1's cell
            else:
                assert n and c["scored"] >= c["quota"] - PI.QUOTA_EPS and not c["short"], key
        sup = [c["quota"] for k, c in census.items() if k.startswith("supporting/")]
        assert len(sup) == 12 and len(set(round(q, 6) for q in sup)) == 1   # six sub-types x two modes, equal


def test_the_suite_adds_whole_tool_episodes_and_v2s_groups(built):
    d, res = built
    man = json.loads((d / "suite" / "manifest.json").read_text())
    assert man["version"] == 2 and man["critical_groups"] == PI.CRITICAL_GROUPS_V2
    assert all(man["tool_calling"]["reached"].values())
    assert "tool_calling@v1|H-T" in man["domains"]["supporting"]
    v1 = PI.jsonl(PI.SUITE_V1 / "items.jsonl")
    v2 = PI.jsonl(d / "suite" / "items.jsonl")
    assert [x["item_id"] for x in v2[:len(v1)]] == [x["item_id"] for x in v1]
    tool = v2[len(v1):]
    key = lambda x: (x["task_id"], x["mode"])  # noqa: E731 -- a task id is the same for every teacher and mode
    assert {key(x): sum(1 for y in tool if key(y) == key(x)) for x in tool} == {key(x): 2 for x in tool}


def test_the_freeze_binds_the_packed_protocol(built):
    d, res = built
    rec = json.loads((d / "freeze.json").read_text())
    pid = rec["suite_identity"]["measurement_protocol_id"]
    assert pid == res["measurement_protocol_id"] and pid != "1c8305246073213e58d0c1098ba4e4e0"
    assert rec["suite_identity"]["forward_context"]["packing"] == "sequential_truncate_v1"


def test_the_binding_refuses_v1s_groups_for_a_v2_suite(built):
    from stages.teacher_modes import hybrid_state_eval as HSE
    d, _ = built
    ident = json.loads(PI.TEACHER.read_text())
    with pytest.raises(HSE.BindingError, match="critical groups"):
        HSE.load(d / "suite", ident, d / "types.json")


def test_a_short_tool_cell_leaves_nothing_frozen(tmp_path):
    items = {m: tool_items(m, "calibration", 1, turns=1, per_turn=50) for m in PI.MODES}
    res = PI.build(items, profiles_out=tmp_path / "p.json", suite_out=tmp_path / "s", items_out=tmp_path / "i",
                   types_out=tmp_path / "t.json", freeze_out=tmp_path / "f.json")
    assert res["profiles_status"].startswith("SHORT") and res["frozen"] is False
    assert not (tmp_path / "f.json").exists()
