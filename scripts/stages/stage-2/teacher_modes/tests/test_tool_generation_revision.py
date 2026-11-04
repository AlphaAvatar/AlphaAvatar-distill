"""The bounded plan on measured yields and an import (maintainer 2026-11-04): per cell, the sealed units apart from the
whole supply's projection and what the bound buys; the rounds walk NEW tasks only; the session starts each cell from
its sealed state; the formal kinds' launch path. Nothing here generates or spends."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from stages.teacher_modes import tool_episodes as TE
from stages.teacher_modes import tool_generation_plan as TGP
from stages.teacher_modes import tool_session as TS
from stages.teacher_modes import token_matched_phase as TM

PROTO = json.loads(TGP.PROTOCOL.read_text())


def _task(i, shape=TE.FULL_EPISODE, split="training", cat="schema_arguments"):
    t = {"task_id": f"t{i}", "split": split, "category": cat, "gold_steps": [[[{"name": "f", "arguments": {}}]]],
         "source": {"recorded": {"steps": 1}}, "later_user_turns": []}
    if shape == TE.SINGLE_STEP:
        t["episode_shape"] = TE.SINGLE_STEP
    return t


def _cells(priors, imported, *, supply=6, single=4, target=100.0):
    tasks = [_task(i) for i in range(supply)] + [_task(supply + i, TE.SINGLE_STEP) for i in range(single)]
    by_id = {t["task_id"]: t for t in tasks}
    man = {"orders": {"training": [t["task_id"] for t in tasks]}}
    proto = json.loads(json.dumps(PROTO))
    proto["lineages"] = {"hybrid": {"modes": {"H-N": proto["lineages"]["hybrid"]["modes"]["H-N"]}, "splits": ["training"]}}
    proto["bounded_plan"]["targets"]["training"]["per_mode"] = {"H-N": target}
    return TGP.lineage_cells(proto, {"schema_arguments": 1.0}, "hybrid", man, by_id, priors, imported), by_id


def _priors(full=(5.0, 10.0, 20.0), single=(1.0, 2.0, 3.0)):
    k = "hybrid|H-N|training|schema_arguments|"
    return {"strata": {k + TE.FULL_EPISODE: {"units_per_attempted_task": list(full), "generated_tokens_per_attempted_task": 100.0},
                       k + TE.SINGLE_STEP: {"units_per_attempted_task": list(single), "generated_tokens_per_attempted_task": 50.0}}}


def _imported(done):
    return {"modes": {"H-N": {"tasks": {tid: {"split": "training", "category": "schema_arguments", "shape": TE.FULL_EPISODE,
                                              "final": {"outcome": "admitted" if u else "rejected", "units": u}}
                                        for tid, u in done.items()}}}}


def test_a_cell_keeps_sealed_projected_bounded_and_unmeasured_apart():
    cells, _ = _cells(_priors(), _imported({"t0": 30.0, "t1": 0.0}))
    c = cells["H-N"]["cells"][0]
    assert c["sealed"] == {"tasks": 2, "admitted": 1, "units": 30.0}
    assert c["supply_by_shape"] == {TE.FULL_EPISODE: 6, TE.SINGLE_STEP: 4}
    #: whole supply: 30 sealed + 4 new full x [5, 10, 20] + 4 single x [1, 2, 3]
    assert c["projection"]["whole_supply_units"] == [30 + 20 + 4, 30 + 40 + 8, 30 + 80 + 12]
    assert c["status"] == "SHORTFALL_PROJECTED" and c["task_ids"][:2] == ["t2", "t3"]      #: imported ones skipped
    assert c["basis_of_new_tasks"] == {f"{TE.FULL_EPISODE}:measured": 4, f"{TE.SINGLE_STEP}:measured": 4}
    assert c["projection"]["unmeasured_units_within_bound"] == 0


def test_a_cell_reached_by_its_sealed_units_attempts_nothing_and_an_exhausted_one_is_exact():
    cells, _ = _cells(_priors(), _imported({f"t{i}": 30.0 for i in range(4)}), supply=4, single=0)
    c = cells["H-N"]["cells"][0]
    assert c["status"] == "REACHED" and c["bound"] == 0 and c["task_ids"] == []
    cells, _ = _cells(_priors(), _imported({f"t{i}": 10.0 for i in range(4)}), supply=4, single=0)
    assert cells["H-N"]["cells"][0]["status"] == "SHORTFALL_EXHAUSTED"


def test_unmeasured_strata_are_priced_by_proxy_or_prior_and_counted_apart():
    cells, _ = _cells({"strata": {}}, None)
    c = cells["H-N"]["cells"][0]
    assert set(k.split(":", 1)[1] for k in c["basis_of_new_tasks"]) == {"prior"}
    assert c["projection"]["unmeasured_units_within_bound"] > 0


def test_the_rounds_walk_new_tasks_and_count_the_imported_yield():
    bp = PROTO["bounded_plan"]
    cell = {"target": 100.0, "bound": 6, "tasks": {"expected": 2}, "unit": "supervised_tokens"}
    order = [_task(i) for i in range(6)]
    assert TGP.round_tasks(cell, order, {"attempted": 0, "units": 40.0, "imported": 4}, 1, bp=bp, mode="H-N",
                           candidates=2) == ["t0", "t1"]
    #: round 2: observed 60 units over 4 imported + 2 new = 10 per task -> ceil(40 / 10 x 1.1) = 5, room 4
    got = TGP.round_tasks(cell, order, {"attempted": 2, "units": 60.0, "imported": 4}, 2, bp=bp, mode="H-N", candidates=2)
    assert got == ["t2", "t3", "t4", "t5"]
    assert TGP.round_tasks(cell, order, {"attempted": 2, "units": 120.0, "imported": 4}, 2, bp=bp, mode="H-N",
                           candidates=2) == []


def test_the_session_starts_each_cell_from_its_sealed_state_and_attempts_only_its_task_ids(tmp_path, monkeypatch):
    from stages.reasoning_data import generation_driver as GD
    cells, by_id = _cells(_priors(), _imported({"t0": 30.0, "t1": 0.0}))
    plan = {"lineages": {"hybrid": {"modes": cells}}}
    seen = []

    def run_group(c, out, inputs, f, k, ids, run, mark, *, label=None, extra_args=()):
        rows = [json.loads(x)["task_id"] for x in (inputs / f["file"]).read_text().splitlines()]
        seen.append((label, rows))
        return {"label": label, "complete": True, "segments": [], "records": {}}
    monkeypatch.setattr(GD, "run_group", run_group)
    res = TS.generate_mode(PROTO, plan, "hybrid", "H-N", by_id, None, tmp_path, "id", None, lambda m: None)
    first = seen[0][1]
    assert "t0" not in first and "t1" not in first and set(first) <= set(cells["H-N"]["cells"][0]["task_ids"])
    assert res["cells"][0]["imported"] == 2 and res["cells"][0]["units"] == 30.0


def test_the_formal_kinds_take_the_container_placement_and_the_relayed_task_file():
    for kind in ("tool_generation", "tool_generation_t2507"):
        spec = TM.launch_spec(kind, 60)
        assert TM.the_pilot_placement_is_the_planned_one in spec.extra_prechecks
        assert TM.the_bound_task_file_is_relayed in spec.extra_prechecks
        assert (spec.on_poll, spec.products_secured) == (TM.secure_pilot, TM.pilot_secured)
        assert not any("tool_pool" in a.repo_path for a in spec.local_assets)       #: never the whole pool (P8.4)
        assert TM.SHARDS[kind][1].match("tool_H-N_r1_pass0.part0000.done")
    a = SimpleNamespace(volume_mount_path="", volume_gb=None, run_id="r", resume_from="/x/secured")
    cmd = TM.tool_generation_command(SimpleNamespace(args=a), None)
    assert f"--resume-from {TM.SE.POD_REPO}/{TM.PILOT_RESUME_DIR}/secured" in cmd and "/workspace/teacher_modes" in cmd


def test_the_generation_scoped_gate_admits_the_hybrids_engine_and_refuses_thinking_2507():
    from stages.teacher_modes import tool_pilot as TPL
    assert TPL.l40s_generation_problems(kind="tool_generation", lineage="hybrid") == []
    bad = TPL.l40s_generation_problems(kind="tool_generation_t2507", lineage="thinking_2507")
    assert any("Qwen3-4B-Thinking-2507" in b for b in bad)
