"""The bounded generation: the plan's cells and bounds, and the session's rounds on a scripted engine -- each cell
walks its frozen order, stops at its target, never passes its bound, retries only refused tasks, records a
SHORTFALL, and a replacement session recomputes the same rounds from what was sealed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aadistill.data.tool_calls import canonical_call
from stages.teacher_modes import tool_generation_plan as TGP
from stages.teacher_modes import tool_pool as TP
from stages.teacher_modes import tool_session as TS

THINK, OPEN, CLOSE, EOS = 151668, 151657, 151658, 151645
TURN = 300


def tasks(n_per_cat=40, cats=("tool_selection", "no_tool")) -> list[dict]:
    out, k = [], 0
    while TP.split_of(f"server:s{k}") != "training":
        k += 1
    srv = f"s{k}"
    for cat in cats:
        for j in range(n_per_cat):
            name = f"{srv}-{cat}"
            call = {"name": name, "arguments": {"q": f"{cat}{j}"}}
            out.append({"task_id": f"tc1-{cat}-{j:03d}", "category": cat, "source": {
                "dataset": "toy/tools", "revision": "r1", "recorded": {"steps": 0 if cat == "no_tool" else 1}},
                "family": None, "split": None, "servers": [srv], "tools": [{"type": "function", "function": {
                    "name": name, "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                                                 "required": ["q"]}}}],
                "messages": [{"role": "user", "content": f"handle {cat} {j}"}], "later_user_turns": [],
                "backend": {"kind": "replay", "table": {canonical_call(call): "done"}},
                "final_check": {"kind": "no_call"} if cat == "no_tool" else
                {"kind": "calls_tools", "values": [cat]}, "max_turns": 4})
    return out


def fails_first(tid: str) -> bool:
    return int(hashlib.sha256(tid.encode()).hexdigest(), 16) % 3 == 0


def episode(task: dict, k: int) -> dict:
    think = [7] * 40 + [THINK]
    turns = []
    if task["category"] != "no_tool":
        call = json.loads(next(iter(task["backend"]["table"])))
        bad = k == 0 and fails_first(task["task_id"])            #: a first candidate that breaks the protocol
        raw = ("plan</think>\n" if not bad else "no close ") + f"<tool_call>{json.dumps(call)}</tool_call>"
        turns.append({"raw": raw, "stop_reason": "eos", "responses": ["done"], "prompt_token_ids": [1] * 50,
                      "output_token_ids": think + [OPEN] + [8] * (TURN - len(think) - 3) + [CLOSE, EOS]})
    turns.append({"raw": "plan</think>\nAll done.", "stop_reason": "eos", "responses": [],
                  "prompt_token_ids": [1] * 80, "output_token_ids": think + [9] * (TURN - len(think) - 1) + [EOS]})
    return {"turns": turns}


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    pool = TP.assemble(tasks())
    TP.write_pool(pool, tmp_path / "pool", sources={"toy": {"converted_sha256": "0" * 64, "tasks": 80}}, extra={})
    proto = json.loads(TS.PROTOCOL.read_text())
    proto["bounded_plan"]["targets"]["training"]["per_mode"] = {"H-T": 40_000, "H-N": 40_000, "T2507": 40_000}
    proto["lineages"]["hybrid"]["splits"] = ["training"]
    proto["bounded_plan"]["rounds"]["first_round_bound_lineages"] = []      #: the generic round rule under test
    plan = TGP.build(tmp_path / "pool", protocol=proto)
    man, splits = TP.load_pool(tmp_path / "pool")
    by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
    calls = []

    def run_group(c, out, inputs, f, k, ids, run, mark, *, label=None, extra_args=()):
        rows = [json.loads(x) for x in (inputs / f["file"]).read_text().splitlines()]
        want = [r["task_id"] for r in rows if ids is None or r["task_id"] in ids]
        calls.append((label, len(want)))
        return {"label": label, "complete": True, "records": {t: episode(by_id[t], k) for t in want}}
    from stages.reasoning_data import generation_driver as GD
    monkeypatch.setattr(GD, "run_group", run_group)
    return {"tmp": tmp_path, "proto": proto, "plan": plan, "man": man, "by_id": by_id, "calls": calls}


def test_cells_stop_at_their_target_walk_their_frozen_order_and_respect_the_bound(setup):
    s = setup
    res = TS.generate_mode(s["proto"], s["plan"], "hybrid", "H-T", s["by_id"], s["man"], s["tmp"] / "out", "id",
                           None, lambda m: None)
    assert res["complete"]
    cells = {c["category"]: c for c in res["cells"]}
    plan_cells = {c["category"]: c for c in s["plan"]["lineages"]["hybrid"]["modes"]["H-T"]["cells"]}
    for cat in ("tool_selection", "no_tool"):
        c, p = cells[cat], plan_cells[cat]
        assert c["status"] == "REACHED" and c["units"] >= c["target"] and c["attempted"] <= p["bound"]
        assert c["units"] - c["target"] < 600 * 2           #: stops within the round's last top-up, not the pool
    short = [c for c in res["cells"] if c["status"] == "SHORTFALL"]
    assert {c["category"] for c in short} == set(TGP.json.loads(TGP.SIX.read_text())["tool_calling"]["categories"]) \
        - {"tool_selection", "no_tool"}                         #: no supply: recorded, never filled from another
    assert all(c["attempted"] == 0 for c in short)
    first = [n for label, n in s["calls"] if label.endswith("pass0")]
    retry = [n for label, n in s["calls"] if label.endswith("pass1")]
    assert retry and all(r < f for r, f in zip(retry, first))   #: second candidates only for refused tasks


def test_a_replacement_session_recomputes_the_same_rounds(setup):
    s = setup
    a = TS.generate_mode(s["proto"], s["plan"], "hybrid", "H-T", s["by_id"], s["man"], s["tmp"] / "out", "id",
                         None, lambda m: None)
    first_calls = list(s["calls"])
    s["calls"].clear()
    b = TS.generate_mode(s["proto"], s["plan"], "hybrid", "H-T", s["by_id"], s["man"], s["tmp"] / "out", "id",
                         None, lambda m: None)
    assert a["cells"] == b["cells"] and s["calls"] == first_calls          #: identical inputs, rewritten identically


def test_round_tasks_never_pass_the_bound_and_stop_when_done():
    bp = json.loads(TS.PROTOCOL.read_text())["bounded_plan"]
    order = [{"task_id": f"t{i}", "source": {"recorded": {"steps": 1}}} for i in range(100)]
    cell = {"target": 3000.0, "unit": "supervised_tokens", "bound": 20, "tasks": {"expected": 8}, "min_tasks": 0}
    assert len(TGP.round_tasks(cell, order, {}, 1, bp=bp, mode="H-T", candidates=2)) == 8
    nxt = TGP.round_tasks(cell, order, {"attempted": 8, "units": 1200.0, "admitted": 4}, 2, bp=bp, mode="H-T",
                          candidates=2)
    assert nxt[0] == "t8" and len(nxt) == 12                               #: capped by the bound (20 - 8)
    assert TGP.round_tasks(cell, order, {"attempted": 20, "units": 1200.0}, 3, bp=bp, mode="H-T", candidates=2) == []
    assert TGP.round_tasks(cell, order, {"attempted": 9, "units": 3000.0}, 2, bp=bp, mode="H-T", candidates=2) == []
    assert TGP.round_tasks(cell, order, {}, 5, bp=bp, mode="H-T", candidates=2) == []       #: past max_rounds


def test_the_plan_prices_the_bound_and_binds_the_pool(setup):
    p = setup["plan"]
    assert p["pool"]["content_sha256"] == setup["man"]["content_sha256"]
    tot = p["price"]["total_usd"]
    assert 0 < tot["lower"] <= tot["expected"] <= tot["upper"] < p["price"]["proposed_hard_cap_usd"]
    up = p["price"]["sessions"]["hybrid"]["upper"]
    assert up["engine_invocations"] == 2 * 2 * json.loads(TS.PROTOCOL.read_text())["bounded_plan"]["rounds"]["max_rounds"]
    assert Path(TS.PROTOCOL).is_file()
