"""The Stage-2 isolation rules refuse each violation they exist for, on real eval-set-1 items.

Each case takes a REAL protected item (read through `shared.eval_sets`, so a
modified set fails here first) and builds the one-field violation a careless
pool would contain; the rule written for it must fire, and only that rule.
"""
from __future__ import annotations

import json

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import isolation_check as IC
from stages.calibration_volume.tasks import registry as R
from stages.calibration_volume.tasks.common import build_instance

SOURCES = A.read_json(A.CONFIG_DIR / "sources.json")


@pytest.fixture(scope="module")
def fluid():
    return IC.fluid_eval_items()


def training_item(seed_offset=0):
    lo = next(r for r in IC.rules()["rules"] if r["id"] == "R5_shared_generator_exposure")["training_seed_range"][0]
    gen = next(g for g in R.GENERATORS_V2 if R.family_split(g)["calibration"]
               and not g.interactive)
    fam = gen.family(R.family_split(gen)["calibration"][0])
    t = build_instance(gen, fam, "L2", lo + 777 + seed_offset)
    t.update(unit_id=t["task_id"], source="s0_generators_v2_calsplit", source_split="calibration_families")
    return t


def test_a_clean_training_item_passes():
    rep = IC.check_pool([training_item()], sources=SOURCES)
    assert rep["pass"], rep["violations"]


def test_a_held_out_prompt_verbatim_is_refused(fluid):
    item = next(r for r in fluid if not r["interactive"])
    rep = IC.check_pool([{"unit_id": "x", "messages": item["messages"]}])
    assert "R1_held_out_keys" in rep["violations"]


def test_a_reworded_held_out_prompt_is_refused(fluid):
    item = next(r for r in fluid if not r["interactive"] and len(r["messages"][-1]["content"].split()) > 60)
    text = "Please solve this. " + item["messages"][-1]["content"].upper()
    rep = IC.check_pool([{"unit_id": "x", "messages": [{"role": "user", "content": text}]}])
    assert "R2_near_duplicate" in rep["violations"]
    assert "R1_held_out_keys" not in rep["violations"]


def test_an_evaluation_family_is_refused(fluid):
    ev = next(r for r in fluid if r["generator_id"] not in R.EVAL_ONLY_GENERATORS)
    t = training_item()
    t.update(family_id=ev["family_id"], generator_id=ev["generator_id"])
    rep = IC.check_pool([t])
    assert "R3_rule_family" in rep["violations"]


def test_an_evaluation_only_generator_or_its_task_type_is_refused():
    t = training_item()
    t["generator_id"] = sorted(R.EVAL_ONLY_GENERATORS)[0]
    assert "R4_eval_only_generators" in IC.check_pool([t])["violations"]
    ext = {"unit_id": "ext-1", "generator_id": "external_pkg/shortest_path@1",
           "task_type": "graph_paths", "messages": [{"role": "user", "content": "find a path"}]}
    assert "R4_eval_only_generators" in IC.check_pool([ext])["violations"]


def test_a_seed_outside_the_training_range_is_refused_unless_historical():
    t = training_item()
    t["seed"] = 2_000_123
    assert "R5_shared_generator_exposure" in IC.check_pool([t])["violations"]
    assert IC.check_pool([t], historical=True)["pass"]


def test_a_forbidden_source_split_is_refused():
    item = {"unit_id": "g1", "source": "gsm8k_train", "source_split": "test",
            "messages": [{"role": "user", "content": "A unique question about 17 apples and a hat."}]}
    rep = IC.check_pool([item], sources=SOURCES)
    assert "R6_source_split" in rep["violations"]


def test_the_committed_training_pool_still_passes_and_matches_its_manifest():
    tasks_dir = A.REPO / "data/stages/stage-2/reasoning_data/tasks"
    man = json.loads((tasks_dir / "train_tasks_v2.manifest.json").read_text())
    path = tasks_dir / "train_tasks_v2.jsonl"
    assert A.sha256_file(path) == man["sha256"]
    rep = IC.check_pool(A.jsonl(path), sources=SOURCES)
    assert rep["pass"], rep["violations"]


EXTERNAL = {"unit_id": "rg-1", "source": "reasoning_gym", "module": "zebra_puzzles",
            "family_id": "reasoning_gym/zebra_puzzles", "task_type": "positional_constraint_deduction",
            "seed": 1, "messages": [{"role": "user", "content": "Three houses stand in a row; who owns the zebra?"}]}


@pytest.mark.parametrize("change,why", [
    ({"source": None}, "unregistered source None"),
    ({"source": "my_new_scrape"}, "unregistered source 'my_new_scrape'"),
    ({"source": "tool_calling"}, "tool_calling: not trainable"),
    ({"task_type": None}, "reasoning_gym: missing task_type"),
    ({"family_id": ""}, "reasoning_gym: missing family_id"),
    ({"module": "arc_agi"}, "reasoning_gym: module 'arc_agi' is not a candidate module"),
])
def test_missing_provenance_is_a_violation_not_a_pass(change, why):
    rep = IC.check_pool([{**EXTERNAL, **change}], sources=SOURCES)
    assert "R0_provenance" in rep["violations"]
    assert why in rep["results"]["R0_provenance"]["violations"]


def test_a_clean_external_item_passes_and_the_report_bounds_its_claims():
    rep = IC.check_pool([EXTERNAL], sources=SOURCES)
    assert rep["pass"], rep["violations"]
    assert "not that the mechanism" in IC.rules()["what_a_pass_shows"]
    assert "semantic mechanism isolation" in rep["not_established"]


def test_an_arc_lineage_source_is_refused():
    item = {**EXTERNAL, "source": "re_arc", "module": "task_007bbfb7"}
    assert "R7_arc_agi" in IC.check_pool([item], sources=SOURCES)["violations"]


def test_arc_grids_in_a_prompt_are_refused_and_small_coincidences_are_not(monkeypatch):
    import hashlib
    big = "\n".join(" ".join(str((r * 7 + c * 3) % 10) for c in range(6)) for r in range(5))
    pair = ["1 2 0\n3 4 0\n5 0 0", "0 0 5\n0 4 3\n0 2 1"]
    canon = lambda t: "/".join(t.replace(" ", "").split("\n"))  # noqa: E731
    fp = lambda t: hashlib.sha256(canon(t).encode()).hexdigest()[:16]  # noqa: E731
    small = "1 0 0\n0 0 0\n0 0 0"
    sparse = ["2 0 0\n0 0 0\n0 0 0", "2 2 2\n0 0 0\n0 0 0"]   # the historical coincidence's shape
    monkeypatch.setattr(IC, "arc_index", lambda: {"index": {fp(big): ["eval:aaaa"], fp(pair[0]): ["eval:bbbb"],
                                                            fp(pair[1]): ["eval:bbbb"], fp(small): ["eval:cccc"],
                                                            fp(sparse[0]): ["train:dddd"], fp(sparse[1]): ["train:dddd"]},
                                                  "source": {}, "sha256": "x"})
    def pool(text):
        return [{**EXTERNAL, "messages": [{"role": "user", "content": text}]}]
    assert "R7_arc_agi" in IC.check_pool(pool("Input:\n" + big), sources=SOURCES)["violations"]
    assert "R7_arc_agi" in IC.check_pool(pool(f"A:\n{pair[0]}\nB:\n{pair[1]}"), sources=SOURCES)["violations"]
    rep = IC.check_pool(pool("Grid:\n" + small), sources=SOURCES)
    assert rep["pass"] and rep["results"]["R7_arc_agi"]["items_with_any_grid_match"] == 1
    assert IC.check_pool(pool(f"A:\n{sparse[0]}\nB:\n{sparse[1]}"), sources=SOURCES)["pass"]


def test_grids_are_read_in_every_text_encoding():
    assert IC.text_grids("x\n1 2 3\n4 5 6\ny") == ["123/456"]
    assert IC.text_grids("answer 120 / 003 / 450 done") == ["120/003/450"]
    assert IC.text_grids("[[1, 0], [0, 1]]") == ["10/01"]
    assert IC.text_grids("12 7\n3 44") == [] and IC.text_grids("1 2\n3") == []


def test_a_family_that_computes_an_evaluation_rule_is_refused():
    t = training_item()
    t.update(generator_id="string_induction@v2", family_id="str.double_last")
    rep = IC.check_pool([t], sources=SOURCES)
    assert "R9_mechanism_equivalence" in rep["violations"]
    pairs = rep["results"]["R9_mechanism_equivalence"]["pairs"]
    assert {"string_induction@v2/str.double_last", "letter_analogy@v2/letter.double_last"} <= {
        x for p in pairs for x in (p["training_family"], p["evaluation_family"])}


def test_the_mechanism_check_finds_known_equivalences_and_no_false_ones():
    from stages.reasoning_data import mechanism_check as MC
    r = MC.review()
    assert set(r["equivalent_training_families"]) == {
        "string_induction@v2/str.caesar", "string_induction@v2/str.double_last",
        "string_induction@v2/str.duplicate_each"}
    # a calibration rotation is not any held-out grid rule
    assert not any(p["training_family"].endswith("grid.rotate_cw") for p in r["pairs"])
