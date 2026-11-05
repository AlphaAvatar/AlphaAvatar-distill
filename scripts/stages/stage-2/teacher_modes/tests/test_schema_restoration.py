"""The original-composition restoration's $0 machinery (maintainer 2026-11-05, '原始数据配比恢复'): a pool that only
ADDS tasks keeps its base byte for byte, a record sealed on the base is reused by task identity and never
re-attempted, a category can be split into fixed per-source sub-quotas, a new source is never priced at another
source's measured yield, and `schema_synth`'s tasks are deterministic and admit exactly their gold call under the
pool's own verifier."""
from __future__ import annotations

import json

import pytest

from stages.teacher_modes import packed_packs as PP
from stages.teacher_modes import schema_synth as SS
from stages.teacher_modes import schema_synth_audit as SSA
from stages.teacher_modes import tool_generation_plan as TGP
from stages.teacher_modes import tool_generation_priors as TGPR
from stages.teacher_modes import tool_pilot as TPL
from stages.teacher_modes import tool_pool as TP
from stages.teacher_modes.tests.test_tool_generation_rounds import tasks

PIVOT = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"


def one_per_domain(split: str = "training") -> list[dict]:
    return [SS.task(d, 0, split) for d in SS.DOMAINS]


@pytest.fixture()
def pools(tmp_path):
    base = TP.assemble(tasks(n_per_cat=12))
    TP.write_pool(base, tmp_path / "base", sources={"toy": {"converted_sha256": "0" * 64, "tasks": 24}}, extra={})
    addition = [SS.task(d, i, "training") for d in SS.DOMAINS[:2] for i in range(3)]
    TP.extend(tmp_path / "base", addition, tmp_path / "ext", pool_id="toy+synth",
              sources={"synth": {"converted_sha256": "1" * 64, "tasks": len(addition)}})
    bman, bsplits = TP.load_pool(tmp_path / "base")
    man, splits = TP.load_pool(tmp_path / "ext")
    return {"tmp": tmp_path, "bman": bman, "bsplits": bsplits, "man": man, "splits": splits, "addition": addition}


def test_an_extension_keeps_its_base_byte_for_byte_and_refuses_a_changed_part(pools):
    man, bman = pools["man"], pools["bman"]
    assert man["base"]["content_sha256"] == bman["content_sha256"] != man["content_sha256"]
    for split, order in bman["orders"].items():
        assert man["orders"][split][:len(order)] == order                     #: every base task keeps its place
        assert pools["splits"][split][:len(order)] == pools["bsplits"][split]  #: and its bytes
    assert len(pools["splits"]["training"]) == len(pools["bsplits"]["training"]) + len(pools["addition"])
    assert TGP.pool_contents(man) == (man["content_sha256"], bman["content_sha256"])
    f = pools["tmp"] / "ext" / "training.tasks.jsonl"
    f.write_text(f.read_text().replace("Thanks!", "Thanks."))
    with pytest.raises(TP.PoolIdentityError, match="sha256"):
        TP.load_pool(pools["tmp"] / "ext")


def test_a_record_sealed_on_the_base_is_imported_by_task_identity_and_never_reattempted(pools, tmp_path):
    by_id = {t["task_id"]: t for rows in pools["splits"].values() for t in rows}
    done = [t for t in pools["bsplits"]["training"] if t["category"] == "tool_selection"][:5]
    rec = {"lineage": "hybrid", "run_id": "old", "pool_content_sha256": pools["bman"]["content_sha256"],
           "modes": {"H-T": {"tasks": {t["task_id"]: {
               "split": "training", "category": t["category"], "shape": "full_episode@v1",
               "task_sha256": TPL.task_sha256(t), "candidates": [],
               "final": {"outcome": "admitted", "units": 100.0}} for t in done}}}}
    path = tmp_path / "rec.json"
    path.write_text(json.dumps(rec))
    got = TGP.import_records(TGP.pool_contents(pools["man"]), paths=[path], by_id=by_id)
    assert [p for p, _ in got] == [path]
    proto = json.loads(TGP.PROTOCOL.read_text())
    proto["lineages"]["hybrid"]["splits"] = ["training"]
    proto["lineages"]["hybrid"]["modes"] = {"H-T": proto["lineages"]["hybrid"]["modes"]["H-T"]}
    proto["bounded_plan"]["rounds"]["first_round_bound_lineages"] = []
    cells = TGP.lineage_cells(proto, TGP.design_shares(), "hybrid", pools["man"], by_id, None, [rec])["H-T"]["cells"]
    sel = next(c for c in cells if c["category"] == "tool_selection")
    assert sel["sealed"] == {"tasks": 5, "admitted": 5, "units": 500.0}
    assert not set(sel["task_ids"]) & {t["task_id"] for t in done}
    #: a changed task under the base's content refuses; an unrelated pool's record is skipped
    rec["modes"]["H-T"]["tasks"][done[0]["task_id"]]["task_sha256"] = "f" * 64
    path.write_text(json.dumps(rec))
    with pytest.raises(TGP.PlanIdentityError, match="absent or changed"):
        TGP.import_records(TGP.pool_contents(pools["man"]), paths=[path], by_id=by_id)
    rec["pool_content_sha256"] = "e" * 64
    path.write_text(json.dumps(rec))
    assert TGP.import_records(TGP.pool_contents(pools["man"]), paths=[path], by_id=by_id) == []


def test_source_groups_split_a_category_into_fixed_and_rest_sub_quotas():
    bp = {"targets": {"training": {"unit": "supervised_tokens", "per_mode": {"H-N": 1000.0}}}}
    groups = {"training": {"schema_arguments": [
        {"group": "pivot", "datasets": [PIVOT], "per_mode": {"H-N": 30.0}},
        {"group": "synth", "datasets": [SS.SOURCE], "rest": True}]}}
    cells = TGP.cell_targets(bp, {"schema_arguments": 0.15, "no_tool": 0.85}, "hybrid", "H-N", ["training"],
                             source_groups=groups)
    sch = {c.get("group"): c for c in cells if c["category"] == "schema_arguments"}
    assert set(sch) == {"pivot", "synth"} and sch["pivot"]["target"] == 30.0
    assert sch["synth"]["target"] == pytest.approx(150.0 - 30.0)
    assert [c for c in cells if c["category"] == "no_tool"][0].get("group") is None   #: other categories unchanged
    bad = {"training": {"schema_arguments": [dict(g, rest=True) for g in groups["training"]["schema_arguments"]]}}
    with pytest.raises(ValueError, match="exactly one 'rest'"):
        TGP.cell_targets(bp, {"schema_arguments": 0.15}, "hybrid", "H-N", ["training"], source_groups=bad)
    by_id = {"a": {"category": "schema_arguments", "source": {"dataset": PIVOT}},
             "b": {"category": "schema_arguments", "source": {"dataset": SS.SOURCE}}}
    assert [t["source"]["dataset"] for t in TGP.cell_order(["a", "b"], by_id, "schema_arguments", [SS.SOURCE])] == \
        [SS.SOURCE]


def test_pack_selection_takes_each_source_group_apart_and_refuses_without_sources(tmp_path):
    docs = tmp_path / "docs.jsonl"
    rows = [{"doc_id": f"{t}|H-N|t0", "task_id": t, "split": "training", "category": "schema_arguments",
             "family": "f", "prompt_ids": [1, 2], "output_ids": [3] * 10} for t in ("p1", "p2", "s1", "s2")]
    docs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cells = {"training": [{"split": "training", "category": "schema_arguments", "group": "pivot", "datasets": [PIVOT],
                           "unit": "supervised_tokens", "target": 10.0},
                          {"split": "training", "category": "schema_arguments", "group": "synth",
                           "datasets": [SS.SOURCE], "unit": "supervised_tokens", "target": 20.0}], "validation": []}
    orders = {"training": ["s1", "p1", "s2", "p2"], "validation": []}
    with pytest.raises(PP.PackBuildError, match="source-group cell"):
        PP.tool_documents(docs, "H-N", cells=cells, orders=orders)
    ds = {"p1": PIVOT, "p2": PIVOT, "s1": SS.SOURCE, "s2": SS.SOURCE}
    train, _, quota = PP.tool_documents(docs, "H-N", cells=cells, orders=orders, dataset_of=ds)
    assert sorted(d.meta["task_id"] for d in train) == ["p1", "s1", "s2"]
    assert quota["cells"]["training/schema_arguments/pivot"]["episodes"] == 1
    assert quota["cells"]["training/schema_arguments/synth"]["selected"] == 20


def test_a_new_source_has_its_own_strata_and_is_priced_only_by_a_labelled_proxy():
    assert TGPR.stratum_key("hybrid", "H-N", "training", "schema_arguments", "full_episode@v1", PIVOT) == \
        "hybrid|H-N|training|schema_arguments|full_episode@v1"
    assert TGPR.stratum_key("hybrid", "H-N", "training", "schema_arguments", "full_episode@v1", SS.SOURCE).endswith(
        "|" + SS.SOURCE)
    task = SS.task(SS.DOMAINS[0], 0, "training")
    bases = [b for _, b in TGP.stratum_candidates("hybrid", "H-N", task)]
    assert bases == ["measured", "proxy:training", "proxy:other_source", "proxy:other_source_training"]
    v1 = {"split": "training", "category": "schema_arguments", "source": {"dataset": PIVOT}}
    assert [b for _, b in TGP.stratum_candidates("hybrid", "H-N", v1)] == ["measured", "proxy:training"]


def test_schema_synth_is_deterministic_and_admits_exactly_its_gold_call():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SSA.TOKENIZER))
    a, b = one_per_domain(), one_per_domain()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    for t in a:
        c = SSA.check(t, tok)
        assert c["problems"] == [] and c["unexplained"] == [], (t["source"]["config"], c["problems"])
        assert c["mutations_refused"] and all(c["mutations_refused"].values())
    splits = SS.assign_splits()
    assert {s for s in splits.values()} == {"training", "state_eval", "validation", "calibration", "reserve"}
    assert len({t["family"] for t in a}) == len(SS.DOMAINS)                 #: one environment per domain


# --- the R1 approval's generation kinds (maintainer 2026-11-05, sections I-IV) --------------------------------------

def test_the_restored_designs_kinds_have_their_own_records_and_keep_their_lineage():
    from types import SimpleNamespace
    from stages.teacher_modes import token_matched_phase as TM
    assert {k: TM.GENERATION_LINEAGE[k] for k in ("tool_generation_sr", "tool_generation_sr_t2507")} == {
        "tool_generation_sr": "hybrid", "tool_generation_sr_t2507": "thinking_2507"}
    for new, old in (("tool_generation_sr", "tool_generation"), ("tool_generation_sr_t2507", "tool_generation_t2507")):
        assert new in TM.STUDY.kinds and new in TM.CONTAINER_KINDS
        assert TM.KIND_RECORDS[new][0] != TM.KIND_RECORDS[old][0]        #: its own protocol, plan and admission
        assert TM.KIND_RECORDS[new][1] != TM.KIND_RECORDS[old][1]
        assert TM.KIND_RECORDS[new][4].endswith("tool_source_admission_schema_synth.json")
        assert TM.SHARDS[new][0] == TM.SHARDS[old][0]
        spec = TM.launch_spec(new, 60)
        assert TM.the_pilot_placement_is_the_planned_one in spec.extra_prechecks
        assert TM.the_bound_task_file_is_relayed in spec.extra_prechecks
    assert "tool_generation_sr_t2507" in TM.IN_SESSION_DEVICE_ADMISSION and "tool_generation_sr_t2507" in \
        TM.THINKING_2507_KINDS
    a = SimpleNamespace(volume_mount_path="", volume_gb=None, run_id="r", resume_from=None)
    cmd = TM.DRIVERS["tool_generation_sr_t2507"][1](SimpleNamespace(args=a), None)
    assert "--lineage thinking_2507 --kind tool_generation_sr_t2507" in cmd
    #: the two designs' protocols differ only off the engine: same teachers, template switches, sampling and execution
    old, new = (json.loads((TM.REPO_ROOT / TM.KIND_RECORDS[k][0]).read_text())
                for k in ("tool_generation", "tool_generation_sr"))
    assert old["execution"] == new["execution"]
    for li in old["lineages"]:
        for m, v in old["lineages"][li]["modes"].items():
            w = new["lineages"][li]["modes"][m]
            assert (v["template_kwargs"], v["sampling"]) == (w["template_kwargs"], w["sampling"])


def test_the_session_runs_only_its_own_kind_for_its_lineage(tmp_path, monkeypatch):
    from stages.teacher_modes import token_matched_phase as TM
    from stages.teacher_modes import tool_session as TS
    seen = []

    def refuse(path, kind, root):
        seen.append(kind)
        raise RuntimeError("stop here")
    monkeypatch.setattr(TM, "require_on_pod", refuse)
    args = ["--authorization", str(tmp_path / "a.json"), "--durable-root", str(tmp_path), "--status", str(tmp_path / "s")]
    assert TS.main(args + ["--lineage", "hybrid", "--kind", "tool_generation_sr"]) == 3
    assert TS.main(args + ["--lineage", "thinking_2507", "--kind", "tool_generation_sr"]) == 3   #: refused before binding
    assert seen == ["tool_generation_sr"]
    rec = json.loads((tmp_path / "tool_generation_thinking_2507" / "tool_session.json").read_text())
    assert "not a generation kind of the thinking_2507 lineage" in rec["error"]["error"]


def test_the_extensions_admission_admits_its_content_and_the_base_record_does_not():
    from stages.teacher_modes import tool_source_admission as TSA
    base = TSA.load()
    synth = [{"source": {"dataset": SS.SOURCE, "revision": SS.VERSION}}]
    assert TSA.problems(base, pool_content_sha256=TSA.REVIEWED_CONTENT) == []
    assert any("sources it does not admit" in b for b in TSA.problems(base, pool_content_sha256=TSA.REVIEWED_CONTENT,
                                                                       tasks=synth))
    man = {"pool_id": TSA.EXTENSION_POOL_ID, "content_sha256": TSA.EXTENSION_CONTENT,
           "source_revisions": [["Agent-Ark/Toucan-1.5M", "x"], [SS.SOURCE, SS.VERSION]], "sources": {}, "files": {}}
    import datetime as dt
    rec = TSA.record(man, "0" * 64, decision="quoted", date=dt.datetime.now(dt.timezone.utc).date().isoformat())
    assert TSA.problems(rec, pool_content_sha256=TSA.EXTENSION_CONTENT, tasks=synth) == []
    assert TSA.problems(rec, pool_content_sha256=TSA.REVIEWED_CONTENT)                    #: another content refuses
    assert TSA.problems(dict(rec, conditions=TSA.CONDITIONS), pool_content_sha256=TSA.EXTENSION_CONTENT)
    assert TSA.problems(dict(rec, pool_id="tool_pool@v2"), pool_content_sha256=TSA.EXTENSION_CONTENT)


def test_the_v3_suite_selects_each_source_group_apart():
    from stages.teacher_modes import packed_inputs as PI
    items = [{"task_id": t, "split": "state_eval", "category": "schema_arguments", "tags": {"assistant": [0] * n}}
             for t, n in (("p1", 6), ("s1", 5), ("p2", 6), ("s2", 5))]
    cells = [{"split": "state_eval", "category": "schema_arguments", "group": "pivot", "datasets": [PIVOT],
              "unit": "scored_positions", "target": 6.0, "min_tasks": 1},
             {"split": "state_eval", "category": "schema_arguments", "group": "synth", "datasets": [SS.SOURCE],
              "unit": "scored_positions", "target": 9.0, "min_tasks": 1}]
    plan = {"lineages": {"hybrid": {"modes": {"H-N": {"cells": cells}}}}, "approved_shortfalls": []}
    orders = {"state_eval": ["s1", "p1", "s2", "p2"]}
    with pytest.raises(PI.InputsError, match="source-group cell"):
        PI.tool_cell(items, "state_eval", "H-N", 15.0, plan=plan, orders=orders)
    ds = {"p1": PIVOT, "p2": PIVOT, "s1": SS.SOURCE, "s2": SS.SOURCE}
    picked, got, census = PI.tool_cell(items, "state_eval", "H-N", 15.0, plan=plan, orders=orders, dataset_of=ds)
    assert [x["task_id"] for x in picked] == ["s1", "p1", "s2"] and got == 16.0
    assert census["schema_arguments/pivot"]["tasks"] == 1 and census["schema_arguments/synth"]["scored"] == 10
    assert not PI.unmet(census)
