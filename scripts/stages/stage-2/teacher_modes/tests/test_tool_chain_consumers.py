"""The tool-calling chain's consumers on synthetic episodes with the real task structure, $0: pool -> bounded plan ->
admission -> packed inputs (profiles v2, state_eval_hybrid@v2, its freeze) -> the C1-6cat pack -> the training plan's
binding. Every step is the real code; only the episodes (what a GPU would generate) and C1's existing documents are
synthetic, so a wiring defect a formal generation would expose surfaces here first."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aadistill.data.tool_calls import canonical_call
from stages.teacher_modes import packed_inputs as PI
from stages.teacher_modes import packed_packs as PP
from stages.teacher_modes import tool_admission as TA
from stages.teacher_modes import tool_generation_plan as TGP
from stages.teacher_modes import tool_pool as TP

REPO = Path(__file__).resolve().parents[5]
CATS = ("tool_selection", "schema_arguments", "no_tool", "multiple_parallel", "multi_turn", "error_recovery",
        "mcp_agent")
SPLITS = ("training", "validation", "calibration", "state_eval")
THINK, OPEN, CLOSE, EOS = 151668, 151657, 151658, 151645
TURN = 300                                         #: generated tokens per synthetic turn


def server_in(split: str, taken: set) -> str:
    k = 0
    while True:
        s = f"srv{k}"
        if s not in taken and TP.split_of(f"server:{s}") == split:
            taken.add(s)
            return s
        k += 1


def tasks() -> list[dict]:
    out, taken = [], set()
    for split in SPLITS:
        srv = server_in(split, taken)
        for cat in CATS:
            for j in range(6):
                name = f"{srv}-{split}_{cat}"
                call = {"name": name, "arguments": {"q": f"{cat}{j}"}}
                out.append({
                    "task_id": f"tc1-{split}-{cat}-{j}", "category": cat, "category_detail": None,
                    "source": {"dataset": "toy/tools", "revision": "r1", "recorded": {"steps": 0 if cat == "no_tool"
                                                                                      else 1}},
                    "family": None, "split": None, "servers": [srv], "tools": [{"type": "function", "function": {
                        "name": name, "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                                                     "required": ["q"]}}}],
                    "messages": [{"role": "user", "content": f"please handle {split} {cat} number {j}"}],
                    "later_user_turns": [], "backend": {"kind": "replay", "table": {canonical_call(call): "done"}},
                    "final_check": ({"kind": "no_call"} if cat == "no_tool" else
                                    {"kind": "calls_tools", "values": [f"{split}_{cat}"]}), "max_turns": 4})
    return out


def episode(task: dict, thinking: bool) -> dict:
    """What the teacher would generate: a call turn then an answer (no_tool: the answer only), with ids."""
    think_ids = [7] * 40 + [THINK] if thinking else []
    think_txt = "plan</think>\n" if thinking else ""
    turns = []
    if task["category"] != "no_tool":
        call = json.dumps({"name": task["tools"][0]["function"]["name"],
                           "arguments": json.loads(next(iter(task["backend"]["table"])))["arguments"]})
        body = [8] * (TURN - len(think_ids) - 3)
        turns.append({"raw": f"{think_txt}<tool_call>{call}</tool_call>", "stop_reason": "eos", "responses": ["done"],
                      "prompt_token_ids": [1] * 50, "output_token_ids": think_ids + [OPEN] + body + [CLOSE, EOS]})
    turns.append({"raw": f"{think_txt}All done.", "stop_reason": "eos", "responses": [],
                  "prompt_token_ids": [1] * 80, "output_token_ids": think_ids + [9] * (TURN - len(think_ids) - 1) + [EOS]})
    return {"task_id": task["task_id"], "turns": turns, "candidate": 0}


@pytest.fixture()
def chain(tmp_path, monkeypatch):
    pool = TP.assemble(tasks())
    assert not pool["excluded"], pool["excluded"][:3]
    pool_dir = tmp_path / "pool"
    TP.write_pool(pool, pool_dir, sources={"toy": {"converted_sha256": "0" * 64, "tasks": 168}}, extra={})
    proto = json.loads(TGP.PROTOCOL.read_text())
    bp = proto["bounded_plan"]["targets"]
    bp["training"]["per_mode"] = {"H-T": 14_000, "H-N": 14_000, "T2507": 14_000}
    bp["validation"] = {"H-T": {"unit": "tasks", "value": 2}, "H-N": {"unit": "tasks", "value": 2},
                        "T2507": {"unit": "supervised_tokens", "value": 600}}
    proto["pool"] = {**proto["pool"], "dir": str(pool_dir), "plan": str(tmp_path / "plan.json")}
    (tmp_path / "protocol.json").write_text(json.dumps(proto))
    monkeypatch.setattr(TGP, "PROTOCOL", tmp_path / "protocol.json")
    plan = TGP.build(pool_dir, protocol=proto)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    man, splits = TP.load_pool(pool_dir)
    ordered = [t for s in sorted(splits) for t in [{x["task_id"]: x for x in splits[s]}[i] for i in man["orders"][s]]]
    out = {}
    for mode, thinking in (("H-T", True), ("H-N", False), ("T2507", True)):
        out[mode] = TA.admit(ordered, [episode(t, thinking) for t in ordered], mode=mode, stop_ids=[EOS])
    return {"tmp": tmp_path, "pool": pool_dir, "plan": plan, "admitted": out, "man": man}


def test_the_plan_bounds_every_cell_and_admission_counts_by_level(chain):
    for li in chain["plan"]["lineages"].values():
        for m in li["modes"].values():
            for c in m["cells"]:
                assert 0 < c["bound"] <= c["supply"] and c["tasks"]["expected"] <= c["bound"]
    census = chain["admitted"]["H-T"]["census"]
    assert census["training|tool_selection"]["admitted"] == 6
    assert census["training|tool_selection"]["admitted_by_verification"] == {
        "protocol+schema_validity+recorded_execution+target_tools_called": 6}   #: calls_tools checks recall only
    assert {d["doc_id"].split("|")[1] for d in chain["admitted"]["H-N"]["documents"]} == {"H-N"}


def test_packed_inputs_select_by_the_plan_and_freeze(chain):
    items = {m: chain["admitted"][m]["items"] for m in ("H-T", "H-N")}
    sel, got, census = PI.tool_cell(items["H-T"], "state_eval", "H-T", PI.SE_POSITIONS)
    assert got >= PI.SE_POSITIONS and not PI.unmet(census)
    assert {c for c, v in census.items() if v["tasks"]} == set(CATS) and all(v["tasks"] >= 5 for v in census.values())
    order = chain["man"]["orders"]["state_eval"]
    seen = list(dict.fromkeys(x["task_id"] for x in sel))
    assert seen == sorted(seen, key=order.index)                   #: returned in the frozen order
    t = chain["tmp"]
    res = PI.build(items, profiles_out=t / "profiles.json", suite_out=t / "suite", items_out=t / "items",
                   types_out=t / "types.json", freeze_out=t / "freeze.json")
    assert res["suite_tool_calling"]["reached"] == {"H-T": True, "H-N": True}
    assert res["suite_tool_calling"]["thin_categories"] == {"H-T": [], "H-N": []}
    assert res["profiles_status"].startswith("COMPLETE"), res["profiles_status"]
    assert res.get("measurement_protocol_id")
    man = json.loads((t / "suite/manifest.json").read_text())
    assert man["counts"]["tool_calling@v1|H-N"]["scored_positions"] >= PI.SE_POSITIONS


def test_the_c1_pack_takes_its_plan_quota_and_binds(chain, monkeypatch):
    from aadistill.data.token_documents import TokenDocument
    from stages.teacher_modes import packed_training_plan as PTP
    t = chain["tmp"]
    docs_path = t / "t2507_docs.jsonl"
    docs_path.write_text("".join(json.dumps(d) + "\n" for d in chain["admitted"]["T2507"]["documents"]))
    tool_total = sum(c["target"] for c in TGP.cells_for(chain["plan"], "thinking_2507", "T2507", "training"))
    plan, orders = TGP.for_consumers(pool_dir=chain["pool"])
    cells = {s: TGP.cells_for(plan, "thinking_2507", "T2507", s) for s in ("training", "validation")}
    picked, _, _ = PP.tool_documents(docs_path, "T2507", cells=cells, orders=orders)
    per = sum(len(d.output_ids) for d in picked)          #: the five other types sized to the tool type's selection
    train, dtype = [], {}
    for typ, n in (("openmath", 1), ("gsm8k", 1), ("code", 1), ("rag_evidence", 1), ("multihop_qa", 1),
                   ("primary.x", 6)):
        for k in range(n):                           #: longer than the 8,192-token sequence: packed truncated
            d = TokenDocument(f"{typ}:{k}", (5, 6), tuple([7] * (per - 1) + [EOS]))
            train.append(d)
            dtype[d.doc_id] = typ
    val = [TokenDocument("v:0", (5,), (7, EOS))]
    dtype["v:0"] = "openmath"
    monkeypatch.setattr(PP, "c1_documents", lambda: (train, val, dtype))
    tm, six = json.loads(PP.TOKEN_MATCHED.read_text()), json.loads(PP.SIX.read_text())   #: C1's design, scaled
    tm["scale"].update(per_mode_total=12 * per, categories={"base_counterpart": 5 * per + tool_total, "fluid": 6 * per})
    six["base_category"]["C1_6cat"].update(tool_calling_target=tool_total, kept_byte_for_byte={
        "competition_math (openmath)": per, "gsm8k": per, "multihop_qa": per, "rag_evidence": per, "code": per})
    for name, d in (("tm.json", tm), ("six.json", six)):
        (t / name).write_text(json.dumps(d))
    monkeypatch.setattr(PP, "TOKEN_MATCHED", t / "tm.json")
    monkeypatch.setattr(PP, "SIX", t / "six.json")
    rec = PP.build("C1-6cat", tool_docs=docs_path, out=t / "packs", pool_dir=chain["pool"])
    quota = rec["manifest"]["tool_calling"]["quota"]["cells"]
    assert all(v["status"] == "REACHED" for v in quota.values()), quota
    assert rec["manifest"]["proportions"]["counts"]["tool_calling"] >= tool_total
    plan = PTP.bind(PTP.packed_units(), t / "packs")
    c1 = [u for u in plan["units"] if u["arm"] == "C1-6cat"]
    assert c1 and all(isinstance(u["config"]["schedule"]["total_steps"], int) for u in c1)
