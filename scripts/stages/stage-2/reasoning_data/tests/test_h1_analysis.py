"""H1's analysis at $0 on synthetic evidence shaped as the formal evaluation writes it."""
from __future__ import annotations

import json
import random

import pytest

from stages.reasoning_data import h1_analysis as H
from stages.reasoning_data import unit_driver as UD


def _evidence(tmp_path, p_correct):
    plan = UD.load_plan()
    items, _ = H.fluid_meta()
    rng = random.Random(7)
    for u in plan["units"]:
        d = tmp_path / u["unit"]
        (d / "report").mkdir(parents=True)
        (d / "evaluation/rows").mkdir(parents=True)
        sched = UD.schedule_for(plan, u["arm"])
        probe = UD.derive_recipe(plan["recipe"], run_name="-", purpose="-", out_dir="-", data_dir="-",
                                 student_path="-", seed=u["seed"], sched=sched,
                                 overrides=plan["config"]["recipe"]["overrides"])
        inputs = UD.unit_inputs(plan, u, sched, UD.recipe_identity(probe), validation=False)
        (d / "report/report.json").write_text(json.dumps({
            "formal": True, "inputs": inputs, "attempts": [{"attempt": 1}],
            "checkpoint": {"artifact_digest": "a", "weights_digest": "w", "destination": "x"},
            "evaluation_descriptor_sha256": "e"}))
        rows = [{"id": i, "correct": rng.random() < p_correct[u["arm"]], "formal": True} for i in items]
        (d / "evaluation/rows/fluid_reasoning_720__dev_v1.rows.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        (d / "evaluation/formal_report.json").write_text(json.dumps({"capabilities": {
            "supporting.math_mixed": {"status": "measured", "units": {"stage3_capability_846/gsm8k": {
                "n": 100, "headline": "correct", "headline_value": 0.1}}},
            "rollout.protocol_integrity": {"status": "measured", "units": {"stage3_capability_846/eval_behavior_v0": {
                "metrics": {"overall": {"terminated": 0.5, "truncated_at_cap": 0.5}}}}}}}))
    return tmp_path


def test_a_large_gain_is_a_go_and_every_axis_is_reported_apart(tmp_path):
    doc = H.analyse(_evidence(tmp_path, {"A": 0.05, "B": 0.40}))
    assert doc["primary_label"] == "GO"
    assert set(doc["by_axis_and_tier"]) >= set(H.STATIC_AXES) | {"interactive_adaptation"}
    assert "interactive_adaptation" in doc["interactive_exploratory"]
    assert doc["decision"]["n_items"] < 720            # interactive episodes are outside the decision
    cell = doc["by_axis_and_tier"]["inductive_rule_discovery"]["B-s20260726"]
    assert set(cell) >= {"all"} and cell["all"]["n"] > 0
    caps = doc["capabilities_by_unit"]["A-s20260726"]
    assert caps["supporting.math_mixed"]["units"]["stage3_capability_846/gsm8k"]["correct"] == 0.1
    assert caps["rollout.protocol_integrity"]["units"]["stage3_capability_846/eval_behavior_v0"]["terminated"] == 0.5


def test_both_arms_at_the_floor_is_not_resolved_never_equivalence(tmp_path):
    doc = H.analyse(_evidence(tmp_path, {"A": 0.0, "B": 0.0}))
    assert doc["primary_label"] == "NOT_RESOLVED"


def test_a_report_that_is_not_the_frozen_unit_refuses(tmp_path):
    ev = _evidence(tmp_path, {"A": 0.1, "B": 0.1})
    p = ev / "B-s20260727/report/report.json"
    doc = json.loads(p.read_text())
    doc["inputs"]["seed"] = 1
    p.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="not this frozen unit"):
        H.analyse(ev)
