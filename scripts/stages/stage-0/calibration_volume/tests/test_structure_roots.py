"""A resumed search journals each restored state again without its decision.

d2's structure.json carried a third, empty root record per (operator, profile)
after the attempt-3 resume, and the formal analysis read that one and failed on
`kept_heads`.
"""
from __future__ import annotations

import json

from stages.calibration_volume.analyze_structure import load_roots


def test_an_empty_resumed_record_does_not_replace_the_decision(tmp_path):
    full = {"depth": 1, "applied_kinds": ["ATTENTION"], "profiles": ["p_base"],
            "decision": {"kept_heads": [[0, 1]]}}
    resumed = {"depth": 1, "applied_kinds": ["ATTENTION"], "profiles": ["p_base"],
               "decision": {}}
    p = tmp_path / "structure.json"
    for rows in ([full, resumed], [resumed, full]):
        p.write_text(json.dumps(rows))
        assert load_roots(p)["ATTENTION"]["p_base"]["decision"] == {"kept_heads": [[0, 1]]}


def test_a_resumed_leaf_takes_its_step_traces_from_its_ancestors(tmp_path):
    from stages.calibration_volume.analyze_formal import top1
    depth = {"kind": "DEPTH", "profile_id": "p", "trace": {"removed_layers": [2, 3]}}
    rows = [
        {"state_id": "a", "parent_id": "root", "steps": [depth]},
        {"state_id": "a", "parent_id": "root", "steps": [dict(depth, trace={})]},   # resumed
        {"state_id": "b", "parent_id": "a",
         "steps": [dict(depth, trace={}), {"kind": "FFN", "profile_id": "p", "trace": {}}]},
    ]
    (tmp_path / "search" / "search").mkdir(parents=True)
    (tmp_path / "search" / "search" / "states.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "search" / "stage1_selection.json").write_text(
        json.dumps({"selected": [{"state_id": "b", "path": "DEPTH->FFN"}]}))
    assert top1(tmp_path)["per_step"][0]["trace"] == {"removed_layers": [2, 3]}
