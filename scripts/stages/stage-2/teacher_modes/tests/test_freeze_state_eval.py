"""freeze_state_eval: whole naturally terminated items in queue order until 12,000-style positions AND the minimum task
count; each mode's critical tags at the right positions; the manifest the hybrid binding and the shared loader accept;
an evaluator built on it (one critical group per mode) and its measurement protocol id."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import freeze_state_eval as FZ  # noqa: E402

STOP = [151645, 151643]


def _rec(n_think: int, n_answer: int, mode: str, stop=True, length=False) -> dict:
    p = [11] * 30
    o = ([151667] + [5] * n_think + [FZ.THINK_CLOSE] if mode == "H-T" else []) + [7] * n_answer + ([151645] if stop else [])
    return {"turns": [{"prompt_token_ids": p, "output_token_ids": o,
                       "stop_reason": "length_limited" if length else "eos"}]}


def test_each_modes_tags_sit_on_the_positions_that_predict_them():
    t = {"task_id": "t", "subtype": "gsm8k", "domain": "supporting", "queue_position": 0}
    it = FZ.item_of(t, _rec(4, 3, "H-T"), "H-T", STOP)
    base = 29                                       # position i predicts token i + 1; the prompt has 30 tokens
    assert it["tags"]["assistant"] == list(range(base, base + 10))
    assert it["tags"]["think_close"] == [base + 5] and it["ids"][base + 6] == FZ.THINK_CLOSE
    assert it["tags"]["final_answer"] == [base + 6, base + 7, base + 8] and it["tags"]["eos"] == [base + 9]
    assert it["critical_group"] == "H-T" and it["subtype"] == "gsm8k|H-T"
    n = FZ.item_of(t, _rec(0, 3, "H-N"), "H-N", STOP)
    assert "think_close" not in n["tags"] and n["tags"]["final_answer"] == [base, base + 1, base + 2]
    assert n["tags"]["eos"] == [base + 3]
    assert FZ.item_of(t, _rec(4, 3, "H-T", stop=False, length=True), "H-T", STOP) is None
    bad = _rec(0, 3, "H-N")
    bad["turns"][0]["output_token_ids"].insert(0, FZ.THINK_CLOSE)       # a non-thinking turn that closes a think block
    assert FZ.item_of(t, bad, "H-N", STOP) is None
    assert FZ.item_of(t, {"turns": [{"prompt_token_ids": [1], "output_token_ids": [7, 151645], "stop_reason": "eos"}]},
                      "H-T", STOP) is None            # thinking mode without its </think>


def test_selection_stops_when_both_positions_and_tasks_are_met_and_reports_a_shortfall():
    tasks = [{"task_id": f"q{k}", "subtype": "gsm8k", "domain": "supporting", "queue_position": k} for k in range(10)]
    recs = {(f"q{k}", m): _rec(40, 20, m) for k in range(10) for m in FZ.MODES}
    recs[("q1", "H-T")] = _rec(40, 20, "H-T", stop=False, length=True)
    items, census = FZ.select(tasks, recs, {"positions": 100, "min_tasks": 3}, STOP)
    assert census["gsm8k|H-T"]["tasks"] == 3 and census["gsm8k|H-T"]["skipped"] == {"not naturally terminated or protocol": 1}
    assert [i["task_id"] for i in items if i["mode"] == "H-T"] == ["q0", "q2", "q3"]     # positions met at 2, tasks at 3
    assert census["gsm8k|H-N"]["tasks"] == 5 and census["gsm8k|H-N"]["reached"]           # 21 positions each
    _, short = FZ.select(tasks, recs, {"positions": 10 ** 6, "min_tasks": 3}, STOP)
    assert not short["gsm8k|H-T"]["reached"]


def test_a_frozen_suite_loads_through_the_hybrid_binding_and_has_a_protocol_id(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    ip = json.loads(FZ.INPUTS_PROTOCOL.read_text())
    types_doc = json.loads((REPO / ip["inputs"]["types"]["path"]).read_text())
    tasks, recs = [], {}
    for t in types_doc["types"]:
        for k in range(2):
            tid = f"{t['subtype']}-{k}"
            tasks.append({"task_id": tid, "subtype": t["subtype"], "domain": t["domain"], "queue_position": k})
            for m in FZ.MODES:
                recs[(tid, m)] = _rec(6, 4, m)
    items, census = FZ.select(tasks, recs, {"positions": 1, "min_tasks": 2}, STOP)
    ident = json.loads(FZ.TEACHER.read_text())
    man = FZ.manifest(items, census, ident, types_doc, ip, "e" * 64)
    body = "".join(json.dumps(i, sort_keys=True) + "\n" for i in items)
    (tmp_path / "items.jsonl").write_text(body)
    import hashlib
    man.update(types_sha256=hashlib.sha256((REPO / ip["inputs"]["types"]["path"]).read_bytes()).hexdigest(),
               items_sha256=hashlib.sha256(body.encode()).hexdigest())
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    from stages.teacher_modes import hybrid_state_eval as HSE
    suite, loaded, _ = HSE.load(tmp_path, ident, REPO / ip["inputs"]["types"]["path"])
    assert suite.critical_groups == {k: tuple(v) for k, v in HSE.CRITICAL_GROUPS.items()}
    assert {i.critical_group for i in loaded} == {"H-T", "H-N"} and len(loaded) == 52
    pid = FZ.protocol_id(tmp_path)
    assert len(pid["measurement_protocol_id"]) == 32 and "critical_aggregation" in pid["reduction"]
