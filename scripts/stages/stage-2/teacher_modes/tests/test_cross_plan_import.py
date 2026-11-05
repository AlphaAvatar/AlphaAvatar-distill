"""Cross-plan reuse of a completed formal tool session (maintainer 2026-11-05, review of f06ccd4, section 1), on a toy
pool with the REAL sealing (`generation_driver.run_group`) and a scripted engine: a COMPLETE session that left a
SHORTFALL is imported under its own identity; a rebuilt plan starts from its units and attempts only new tasks; that
plan's session, interrupted, resumes in a replacement session; no task is ever generated twice; the dev box's
admission reads old and new shards side by side and the units reconcile. Plus the round rules this round added (the
deadline stop at a round boundary, the last round's whole bound, the whole-supply cell) and the merge refusals."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from stages.teacher_modes import tool_admission as TA
from stages.teacher_modes import tool_generation_import as TGI
from stages.teacher_modes import tool_generation_plan as TGP
from stages.teacher_modes import tool_pool as TP
from stages.teacher_modes import tool_session as TS
from stages.teacher_modes.tests.test_tool_generation_rounds import episode, tasks

REPO = TGP.REPO


@pytest.fixture()
def toy(tmp_path):
    pool = TP.assemble(tasks(n_per_cat=60))
    TP.write_pool(pool, tmp_path / "pool", sources={"toy": {"converted_sha256": "0" * 64, "tasks": 120}}, extra={})
    proto = json.loads(TS.PROTOCOL.read_text())
    proto["lineages"]["hybrid"]["splits"] = ["training"]
    proto["lineages"]["hybrid"]["modes"] = {"H-T": proto["lineages"]["hybrid"]["modes"]["H-T"]}
    proto["bounded_plan"]["rounds"]["first_round_bound_lineages"] = []      #: the generic round rule
    man, splits = TP.load_pool(tmp_path / "pool")
    by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
    return {"tmp": tmp_path, "proto": proto, "man": man, "by_id": by_id, "submitted": []}


def runner(toy, *, die_at_pass1: bool = False):
    """A scripted engine behind the real run_group: every task of the segment's input, streamed; ``die_at_pass1``
    kills the generator before it writes anything for a second-candidate invocation."""
    def run(cmd, log, tick):
        rows = [json.loads(x) for x in Path(cmd[cmd.index("--tasks") + 1]).read_text().splitlines()]
        k = int(cmd[cmd.index("--seed-base") + 1]) - TS.protocol()["execution"]["seed_base"]
        toy["submitted"] += [(r["task_id"], k) for r in rows]
        if die_at_pass1 and k == 1:
            return 1
        with Path(cmd[cmd.index("--stream-completed") + 1]).open("a") as fh:
            for r in rows:
                fh.write(json.dumps({"task_id": r["task_id"], **episode(toy["by_id"][r["task_id"]], k)}) + "\n")
        Path(log).write_text("[x] engine ready in 20.0s\n")
        return 0
    return run


def cells_plan(toy, per_mode: float, imported=(), bound: int | None = None) -> dict:
    proto = json.loads(json.dumps(toy["proto"]))
    proto["bounded_plan"]["targets"]["training"]["per_mode"]["H-T"] = per_mode
    modes = TGP.lineage_cells(proto, TGP.design_shares(), "hybrid", toy["man"], toy["by_id"], None, list(imported))
    for c in modes["H-T"]["cells"]:
        if bound is not None and c["task_ids"]:
            c["bound"] = min(bound, c["bound"])
            c["task_ids"] = c["task_ids"][:c["bound"]]
    return {"lineages": {"hybrid": {"modes": modes}}, "proto": proto}


def session(toy, plan: dict, out: Path, identity: str, run, **kw) -> dict:
    res = TS.generate_mode(plan["proto"], plan, "hybrid", "H-T", toy["by_id"], None, out, identity, run,
                           lambda m: None, **kw)
    (out / "tool_session.json").write_text(json.dumps({
        "lineage": "hybrid", "kind": "tool_generation", "identity": identity,
        "terminal": "ALL_DONE" if res["complete"] else "RUN_FAILED", "modes": {"H-T": res}}))
    return res


def import_of(toy, plan: dict, store: Path) -> dict:
    files = {TGI.PLAN_REL: json.dumps({**plan, "pool": {"content_sha256": toy["man"]["content_sha256"]}}).encode()}
    return TGI.build_session("old_run", store, toy["by_id"], at=lambda rel: files.get(rel) or (REPO / rel).read_bytes())


def test_a_complete_session_with_a_shortfall_is_imported_reused_resumed_and_never_regenerated(toy):
    # 1. the earlier plan's session: bounded at 8 tasks a cell, so it completes SHORT of a 60,000-token target
    old_plan = cells_plan(toy, 60_000, bound=8)
    old = toy["tmp"] / "old"
    res = session(toy, old_plan, old, "old-identity", runner(toy))
    assert res["complete"]
    short = {c["category"]: c for c in res["cells"] if c["attempted"]}
    assert short and all(c["status"] == "SHORTFALL" and c["shortfall"] == "bound" for c in short.values())
    old_submitted = list(toy["submitted"])
    shards = {p.name: p.read_bytes() for p in old.glob("tool_H-T_r*_pass*.part*.jsonl")}

    # 2. its import: every attempted task DONE, the refused ones with both candidates, reconciled exactly
    rec = import_of(toy, old_plan, old)
    tasks_ = rec["modes"]["H-T"]["tasks"]
    assert set(tasks_) == {t for t, k in old_submitted if k == 0}
    assert all(len(t["candidates"]) == 2 for t in tasks_.values() if t["final"]["outcome"] == "rejected")
    assert rec["modes"]["H-T"]["reconciliation"]["differ"] == [] and rec["identity"] == "old-identity"

    # 3. the rebuilt plan starts from the imported units and holds only new tasks
    new_plan = cells_plan(toy, 60_000, imported=[rec])
    for c in new_plan["lineages"]["hybrid"]["modes"]["H-T"]["cells"]:
        if c["category"] in short:
            assert c["sealed"]["units"] == short[c["category"]]["units"]
            assert not set(c["task_ids"]) & set(tasks_)

    # 4. its session dies at the first second-candidate invocation; a replacement (same identity, the dev box's
    #    copy of the sealed shards) resumes and completes
    toy["submitted"].clear()
    first = toy["tmp"] / "new1"
    assert not session(toy, new_plan, first, "new-identity", runner(toy, die_at_pass1=True))["complete"]
    second = toy["tmp"] / "new2"
    second.mkdir()
    for f in first.glob("tool_*"):
        if f.is_file():
            shutil.copy2(f, second / f.name)
    shutil.copytree(first / "inputs", second / "inputs")
    done = session(toy, new_plan, second, "new-identity", runner(toy))
    assert done["complete"]
    firsts = [t for t, k in toy["submitted"] if k == 0]
    assert len(firsts) == len(set(firsts))                       #: no first candidate generated twice across sessions
    assert not set(firsts) & set(tasks_)                         #: nothing imported is generated again
    assert {p.name: p.read_bytes() for p in old.glob("tool_H-T_r*_pass*.part*.jsonl")} == shards   #: old seals intact

    # 5. the dev box reads both stores, each under its own identity, and the units reconcile
    out = toy["tmp"] / "admitted"
    imp_path = toy["tmp"] / "import.json"
    imp_path.write_text(json.dumps(rec))
    TA.main(["--pool", str(toy["tmp"] / "pool"), "--store", str(second), "--identity", "new-identity",
             "--import-record", str(imp_path), "--import-store", str(old), "--mode", "H-T", "--out", str(out)])
    docs = [json.loads(x) for x in (out / "tool_documents.jsonl").read_text().splitlines()]
    got = {}
    for d in docs:
        cat = toy["by_id"][d["task_id"]]["category"] if "task_id" in d else d.get("category")
        got[cat] = got.get(cat, 0) + len(d["output_ids"])
    for c in done["cells"]:
        if c["category"] in short:
            assert got[c["category"]] == c["units"]               #: old + new supervised tokens, counted once each


def test_the_admission_refuses_a_task_generated_again_although_imported(toy):
    plan = cells_plan(toy, 60_000, bound=4)
    old = toy["tmp"] / "old"
    session(toy, plan, old, "old-identity", runner(toy))
    rec = import_of(toy, plan, old)
    again = toy["tmp"] / "again"
    session(toy, plan, again, "again-identity", runner(toy))     #: the SAME tasks under a new identity
    imp_path = toy["tmp"] / "import.json"
    imp_path.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="generated again although imported"):
        TA.main(["--pool", str(toy["tmp"] / "pool"), "--store", str(again), "--identity", "again-identity",
                 "--import-record", str(imp_path), "--import-store", str(old), "--mode", "H-T",
                 "--out", str(toy["tmp"] / "x")])


def test_an_incomplete_session_or_a_missing_second_candidate_is_not_imported(toy):
    plan = cells_plan(toy, 60_000, bound=6)
    old = toy["tmp"] / "old"
    session(toy, plan, old, "old-identity", runner(toy, die_at_pass1=True))
    with pytest.raises(TGI.ImportRefused, match="only a completed session"):
        import_of(toy, plan, old)
    ok = toy["tmp"] / "ok"
    session(toy, plan, ok, "ok-identity", runner(toy))
    for f in ok.glob("tool_H-T_r1_pass1.*"):
        f.unlink()
    with pytest.raises(TGI.ImportRefused, match="without its second candidate"):
        import_of(toy, plan, ok)


def test_two_import_records_never_list_the_same_task():
    rec = {"lineage": "hybrid", "modes": {"H-T": {"tasks": {"t1": {
        "split": "training", "category": "c", "shape": "s", "final": {"outcome": "admitted", "units": 5.0}}}}}}
    assert set(TGI.merged([rec, {**rec, "lineage": "thinking_2507"}], "hybrid", "H-T")) == {"t1"}
    with pytest.raises(TGI.ImportRefused, match="two import records"):
        TGI.merged([rec, rec], "hybrid", "H-T")


def test_a_deadline_stops_at_a_round_boundary_with_every_attempted_task_done(toy):
    plan = cells_plan(toy, 60_000)
    for c in plan["lineages"]["hybrid"]["modes"]["H-T"]["cells"]:
        c["tasks"]["expected"] = min(5, c["bound"])              #: round 1 falls short: a round 2 is due
    out = toy["tmp"] / "dl"
    res = session(toy, plan, out, "dl-identity", runner(toy), deadline=0.0)    #: already past after round 1
    assert res["complete"] and res["deadline_stop"]["round"] == 2
    assert {lab for lab in (p.name.split(".")[0] for p in out.glob("*.complete"))} == {"tool_H-T_r1_pass0",
                                                                                       "tool_H-T_r1_pass1"}
    short = [c for c in res["cells"] if c["status"] == "SHORTFALL" and c["attempted"]]
    assert short and all(c["shortfall"] == "deadline" for c in short)
    import_of(toy, plan, out)                                    #: importable whole: no half-done task


def test_the_last_round_takes_the_whole_bound_and_a_whole_supply_cell_is_bound_by_its_supply(toy):
    bp = json.loads(TS.PROTOCOL.read_text())["bounded_plan"]
    last = int(bp["rounds"]["max_rounds"])
    order = [{"task_id": f"t{i}", "source": {"recorded": {"steps": 1}}} for i in range(100)]
    cell = {"target": 3000.0, "unit": "supervised_tokens", "bound": 40, "tasks": {"expected": 8}, "min_tasks": 0}
    state = {"attempted": 12, "units": 1200.0, "admitted": 4}
    assert TGP.round_tasks(cell, order, state, last, bp=bp, mode="H-T", candidates=2) == [f"t{i}" for i in range(12, 40)]
    assert len(TGP.round_tasks(cell, order, state, last - 1, bp=bp, mode="H-T", candidates=2)) < 28
    assert len(TGP.round_tasks(cell, order, {}, 1, bp=bp, mode="H-T", candidates=2)) == 8
    assert len(TGP.round_tasks({**cell, "first_round": "bound"}, order, {}, 1, bp=bp, mode="H-T", candidates=2)) == 40
    wide = [{"lineage": "hybrid", "mode": "H-T", "split": "training", "category": "tool_selection"}]
    modes = TGP.lineage_cells(toy["proto"], TGP.design_shares(), "hybrid", toy["man"], toy["by_id"], whole_supply=wide)
    c = next(c for c in modes["H-T"]["cells"] if c["category"] == "tool_selection")
    assert c["whole_supply"] and c["bound"] == c["new_supply"] == 60


def test_the_committed_plan_binds_both_imports_and_the_common_support_composition():
    plan = json.loads(TGP.PLAN.read_text())
    if not plan.get("imports"):
        pytest.skip("no rebuilt plan committed")
    assert [r["lineage"] for r in plan["imports"]] == ["hybrid", "thinking_2507", "thinking_2507", "thinking_2507",
                                                      "hybrid", "hybrid"]
    assert plan["common_support"]["training_shares"] == TGP.training_shares()
    t = {c["category"]: c for c in plan["lineages"]["thinking_2507"]["modes"]["T2507"]["cells"] if c["split"] == "training"}
    done = TGI.merged([json.loads((REPO / r["path"]).read_text()) for r in plan["imports"]], "thinking_2507", "T2507")
    for cat, c in t.items():
        mine = [x for x in done.values() if x["split"] == "training" and x["category"] == cat]
        assert c["sealed"]["tasks"] == len(mine) and not set(c["task_ids"]) & set(done)
    hn = next(c for c in plan["lineages"]["hybrid"]["modes"]["H-N"]["cells"]
              if (c["split"], c["category"]) == ("training", "schema_arguments"))
    assert hn["whole_supply"] and hn["bound"] == hn["new_supply"] and hn["target"] == pytest.approx(11800, abs=1)


def test_the_reach_bound_gives_a_small_gap_a_95_percent_round():
    """A budget-limited lineage's one round (maintainer 2026-11-05): 1,133 tokens missing, admitted episodes of about
    4,264 tokens, a lower admission yield of 0.159 -- the mean-yield prefix is 2 tasks; P(>= 1 admission) >= 0.95 needs
    ceil(ln 0.05 / ln 0.841) = 18."""
    from stages.teacher_modes import tool_episodes as TE
    key = f"thinking_2507|T2507|training|error_recovery|{TE.FULL_EPISODE}"
    priors = {"strata": {key: {"admission_yield_wilson90": [0.1589, 0.3473], "units_per_admitted": 4264.08}}}
    tasks_ = [{"task_id": f"e{i}", "split": "training", "category": "error_recovery"} for i in range(40)]
    n = TGP.reach_bound(tasks_, 1132.7, lineage="thinking_2507", mode="T2507", priors=priors)
    assert n == 18
    assert TGP.reach_bound(tasks_, 9 * 4264.08, lineage="thinking_2507", mode="T2507", priors=priors) == 40  #: supply
    assert TGP.reach_bound(tasks_, 0.0, lineage="thinking_2507", mode="T2507", priors=priors) == 0
    assert TGP.reach_bound(tasks_, 100.0, lineage="thinking_2507", mode="T2507", priors={"strata": {}}) is None


def test_a_pack_arm_refuses_a_plan_of_the_other_composition(monkeypatch):
    from stages.teacher_modes import packed_packs as PP
    plan = {"common_support": {"training_shares": {}}}
    monkeypatch.setattr(TGP, "for_consumers", lambda pool_dir=None: (plan, {}))
    with pytest.raises(PP.PackBuildError, match="registered tool composition"):
        PP.select("A1-6cat", tool_docs=Path("x"))
    plan.pop("common_support")
    with pytest.raises(PP.PackBuildError, match="common_support tool composition"):
        PP.select("A1-6cat-cs", tool_docs=Path("x"))
