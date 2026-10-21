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
    t["unit_id"] = t["task_id"]
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
    man = json.loads((tasks_dir / "train_tasks_v1.manifest.json").read_text())
    path = tasks_dir / "train_tasks_v1.jsonl"
    assert A.sha256_file(path) == man["sha256"]
    rep = IC.check_pool(A.jsonl(path), sources=SOURCES)
    assert rep["pass"], rep["violations"]
