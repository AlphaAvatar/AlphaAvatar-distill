"""The two gates the third review asked for, each tested from its refusing side:

* calibration v2 and state_eval_hybrid@v2 reach their tool cell only when EVERY category meets its own share -- a
  surplus of another category's whole episodes never covers a short or thin one -- and nothing is frozen unless the
  maintainer approved that category's SHORTFALL;
* every consumer reads the pool and the plan through the same identity checks as the session, so an edited frozen
  order or plan binding is refused even when the old content sha256 is kept."""
from __future__ import annotations

import json

import pytest

from aadistill.data.tool_calls import canonical_call
from stages.teacher_modes import packed_inputs as PI
from stages.teacher_modes import tool_generation_plan as TGP
from stages.teacher_modes import tool_pool as TP

needs_v1 = pytest.mark.skipif(not (PI.SUITE_V1 / "items.jsonl").is_file(),
                              reason="the v1 calibration and suite artifacts are not on this machine")


def items(mode, split, category, tasks, per_turn=300, start=0):
    out = []
    for k in range(start, start + tasks):
        for t in range(2):
            ids = [7] * 50 + [9] * per_turn
            tags = {"assistant": [49 + j for j in range(per_turn)], "eos": [49 + per_turn - 1],
                    "final_answer": [49 + j for j in range(2, per_turn - 1)], "tool_call_open": [52],
                    "tool_call_close": [57]}
            if mode == "H-T":
                tags["think_close"] = [50]
            out.append({"item_id": f"{split}-{category}{k}|{mode}|t{t}", "task_id": f"{split}-{category}{k}",
                        "turn": t, "ids": ids, "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1,
                        "tags": tags, "domain": "supporting", "subtype": f"tool_calling@v1|{mode}",
                        "type": "tool_calling@v1", "mode": mode, "critical_group": mode, "split": split,
                        "source": {"dataset": "toy"}, "family_id": "f", "category": category})
    return out


def plan_and_orders(tool_items, approvals=()):
    """A two-category plan (multi_turn 0.8, tool_selection 0.2) over the items' tasks, in their order."""
    cells = []
    for split, total, mn in (("calibration", 4200.0, 0), ("state_eval", float(PI.SE_POSITIONS), PI.SE_MIN_TASKS)):
        for cat, share in (("multi_turn", 0.8), ("tool_selection", 0.2)):
            cells.append({"split": split, "category": cat, "unit": "scored_positions", "target": total * share,
                          "min_tasks": mn})
    orders = {}
    for rows in tool_items.values():
        for x in rows:
            orders.setdefault(x["split"], [])
            if x["task_id"] not in orders[x["split"]]:
                orders[x["split"]].append(x["task_id"])
    plan = {"lineages": {"hybrid": {"modes": {m: {"cells": cells} for m in PI.MODES}}},
            "approved_shortfalls": list(approvals)}
    return plan, orders


def build(tmp_path, monkeypatch, tool_items, approvals=()):
    monkeypatch.setattr(TGP, "for_consumers", lambda *a, **k: plan_and_orders(tool_items, approvals))
    return PI.build(tool_items, profiles_out=tmp_path / "p.json", suite_out=tmp_path / "suite",
                    items_out=tmp_path / "items", types_out=tmp_path / "types.json", freeze_out=tmp_path / "f.json")


def surplus_but_short():
    """Plenty of multi_turn positions (far past every total quota), tool_selection far below its 20% share."""
    return {m: items(m, "calibration", "multi_turn", 30) + items(m, "calibration", "tool_selection", 1, per_turn=100)
            + items(m, "state_eval", "multi_turn", 40) + items(m, "state_eval", "tool_selection", 6, per_turn=100)
            for m in PI.MODES}


class TestCategoryGates:
    @needs_v1
    def test_a_short_category_is_not_covered_by_another_categorys_surplus(self, tmp_path, monkeypatch):
        tool = surplus_but_short()
        res = build(tmp_path, monkeypatch, tool)
        assert res["profiles_status"].startswith("SHORT") and res["frozen"] is False
        assert not (tmp_path / "f.json").exists()                             #: no freeze record
        supply = {s: sum(PI.scored(x) for x in tool["H-T"] if x["split"] == s) for s in ("calibration", "state_eval")}
        doc = json.loads((tmp_path / "p.json").read_text())
        cell = doc["census"]["calib.hybrid_d4_base"]["supporting/tool_calling@v1|H-T"]
        assert supply["calibration"] >= 2 * cell["quota"]            #: the admitted positions would cover the total
        assert cell["short"] and cell["unmet_categories"] == ["tool_selection"]
        man = json.loads((tmp_path / "suite/manifest.json").read_text())
        tc = man["tool_calling"]
        assert supply["state_eval"] >= 2 * PI.SE_POSITIONS
        assert tc["reached"] == {"H-T": False, "H-N": False} and tc["short_categories"]["H-T"] == ["tool_selection"]

    @needs_v1
    def test_a_thin_category_blocks_the_freeze_even_with_its_positions(self, tmp_path, monkeypatch):
        tool = {m: items(m, "calibration", "multi_turn", 30) + items(m, "calibration", "tool_selection", 10)
                + items(m, "state_eval", "multi_turn", 40) + items(m, "state_eval", "tool_selection", 3, per_turn=900)
                for m in PI.MODES}
        res = build(tmp_path, monkeypatch, tool)
        tc = res["suite_tool_calling"]
        assert tc["thin_categories"]["H-N"] == ["tool_selection"] and not any(tc["reached"].values())
        assert res["frozen"] is False and not (tmp_path / "f.json").exists()

    @needs_v1
    def test_a_category_with_no_admitted_episode_is_short_too(self, tmp_path, monkeypatch):
        tool = {m: items(m, "calibration", "multi_turn", 30) + items(m, "state_eval", "multi_turn", 60)
                for m in PI.MODES}
        res = build(tmp_path, monkeypatch, tool)
        assert res["suite_tool_calling"]["short_categories"]["H-T"] == ["tool_selection"]
        assert res["frozen"] is False

    @needs_v1
    def test_only_the_maintainers_approval_lets_a_short_category_freeze(self, tmp_path, monkeypatch):
        tool = surplus_but_short()
        appr = [{"split": s, "mode": m, "category": "tool_selection", "approved_by": "maintainer (test)"}
                for s in ("calibration", "state_eval") for m in PI.MODES]
        res = build(tmp_path, monkeypatch, tool, approvals=appr)
        assert res["profiles_status"].startswith("COMPLETE with the maintainer's approved shortfalls")
        assert all(res["suite_tool_calling"]["reached"].values()) and res.get("measurement_protocol_id")
        rec = json.loads((tmp_path / "f.json").read_text())
        cats = rec["tool_calling"]["reached"]
        assert cats == {"H-T": True, "H-N": True}
        man = json.loads((tmp_path / "suite/manifest.json").read_text())
        assert man["counts"]["tool_calling@v1|H-T"]["by_category"]["tool_selection"]["approved_shortfall"]


def pool_tasks():
    out = []
    for split in ("training", "calibration", "state_eval", "validation"):
        k = 0
        while TP.split_of(f"server:{split}{k}") != split:
            k += 1
        srv = f"{split}{k}"
        for cat in ("tool_selection", "no_tool"):
            for j in range(4):
                name = f"{srv}-{split}_{cat}"
                call = {"name": name, "arguments": {"q": str(j)}}
                out.append({"task_id": f"tc1-{split}-{cat}-{j}", "category": cat,
                            "source": {"dataset": "toy/t", "revision": "r", "recorded": {"steps": 1}},
                            "family": None, "split": None, "servers": [srv], "tools": [{"type": "function", "function": {
                                "name": name, "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
                            "messages": [{"role": "user", "content": f"do {split} {cat} {j}"}],
                            "backend": {"kind": "replay", "table": {canonical_call(call): "ok"}},
                            "final_check": {"kind": "no_call"} if cat == "no_tool" else
                            {"kind": "calls_tools", "values": [cat]}, "max_turns": 3})
    return out


@pytest.fixture()
def bound(tmp_path, monkeypatch):
    """A real pool, protocol and plan in a temp dir, read through the consumers' own entry points."""
    pool_dir = tmp_path / "pool"
    TP.write_pool(TP.assemble(pool_tasks()), pool_dir, sources={"toy": {"converted_sha256": "0" * 64, "tasks": 32}},
                  extra={})
    proto = json.loads(TGP.PROTOCOL.read_text())
    proto["pool"] = {**proto["pool"], "dir": str(pool_dir), "plan": str(tmp_path / "plan.json")}
    (tmp_path / "protocol.json").write_text(json.dumps(proto))
    monkeypatch.setattr(TGP, "PROTOCOL", tmp_path / "protocol.json")
    plan = TGP.build(pool_dir)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    assert TGP.for_consumers()[0]["pool"]["content_sha256"] == plan["pool"]["content_sha256"]   #: reads as written
    return {"tmp": tmp_path, "pool": pool_dir, "plan": plan, "proto": proto}


class TestConsumerIdentity:
    def test_an_edited_frozen_order_is_refused_even_with_the_old_content_sha(self, bound):
        man_p = bound["pool"] / "pool.json"
        man = json.loads(man_p.read_text())
        man["orders"]["training"] = list(reversed(man["orders"]["training"]))
        man_p.write_text(json.dumps(man))                                     #: content_sha256 kept as it was
        with pytest.raises(TP.PoolIdentityError, match="order"):
            TGP.for_consumers()
        import hashlib
        man["files"]["training"]["order_sha256"] = hashlib.sha256(
            TP.canonical(man["orders"]["training"]).encode()).hexdigest()   #: the order's own sha made consistent
        man_p.write_text(json.dumps(man))
        with pytest.raises(TP.PoolIdentityError, match="content_sha256"):
            TGP.for_consumers()

    def test_a_rewritten_pool_json_is_refused_by_the_plan(self, bound):
        man_p = bound["pool"] / "pool.json"
        man = json.loads(man_p.read_text())
        man["written_utc"] = "2099-01-01T00:00:00Z"                           #: every identity field unchanged
        man_p.write_text(json.dumps(man, indent=1) + "\n")
        with pytest.raises(TGP.PlanIdentityError, match="another pool.json"):
            TGP.for_consumers()

    @pytest.mark.parametrize("edit", ["pool_json", "target", "bound", "approval", "shares"])
    def test_an_edited_plan_binding_is_refused(self, bound, edit):
        plan = json.loads((bound["tmp"] / "plan.json").read_text())
        cell = plan["lineages"]["hybrid"]["modes"]["H-T"]["cells"][0]
        if edit == "pool_json":
            plan["pool"]["pool_json_sha256"] = "0" * 64
        elif edit == "target":
            cell["target"] *= 0.5
        elif edit == "bound":
            cell["bound"] += 1000
        elif edit == "approval":
            plan["approved_shortfalls"] = [{"split": "state_eval", "mode": "H-N", "category": "schema_arguments"}]
        else:
            plan["shares"]["no_tool"] = 0.5
        (bound["tmp"] / "plan.json").write_text(json.dumps(plan))
        with pytest.raises(TGP.PlanIdentityError):
            TGP.for_consumers()

    def test_the_consumers_refuse_through_the_same_check(self, bound, tmp_path):
        from stages.teacher_modes import packed_packs as PP
        plan = json.loads((bound["tmp"] / "plan.json").read_text())
        plan["lineages"]["thinking_2507"]["modes"]["T2507"]["cells"][0]["target"] = 1.0
        (bound["tmp"] / "plan.json").write_text(json.dumps(plan))
        with pytest.raises(TGP.PlanIdentityError):
            PI.tool_cell([], "state_eval", "H-T", PI.SE_POSITIONS)
        docs = tmp_path / "d.jsonl"
        docs.write_text("")
        with pytest.raises(TGP.PlanIdentityError):
            PP.build("C1-6cat", tool_docs=docs, out=tmp_path / "out", pool_dir=bound["pool"])
