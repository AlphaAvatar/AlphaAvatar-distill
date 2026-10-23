"""A corpus candidate is its own identity, and its feasibility is the formal ladder's verdict."""
from __future__ import annotations

import json
import random

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_arm_b as AB
from stages.reasoning_data import corpus_feasibility as CF
from stages.reasoning_data import source_plan as SP

from aadistill.data.nested_ladder import LadderError, LadderUnit, build_ladder

V2 = A.CONFIG_DIR / "corpus_v2_candidate.json"
V3 = A.CONFIG_DIR / "corpus_v3_proposal.json"
TARGETS = [8_600, 16_000, 29_600]


def test_v2_is_the_maintainers_split_under_its_own_id_and_never_the_chosen_one():
    spec = SP.candidate_spec(V2)
    v1 = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    assert spec["corpus_id"] == "stage2.reasoning_master@v2" != v1["corpus_id"]
    assert spec["chosen_variant"]["id"] == "stage2.reasoning_master@v1"
    assert spec["salt"] != v1["salt"]
    cells = AB.cell_shares(spec, spec["corpus_id"])
    assert cells == pytest.approx({
        "primary.task_contained_deduction/s0_generators_v2_calsplit": 0.095,
        "primary.task_contained_deduction/prontoqa": 0.03,
        "primary.task_contained_deduction/reasoning_core": 0.03,
        "primary.task_contained_deduction/reasoning_gym": 0.025,
        "primary.inductive_rule_discovery/s0_generators_v2_calsplit": 0.17,
        "primary.compositional_generalization/s0_generators_v2_calsplit": 0.12,
        "primary.compositional_generalization/reasoning_core": 0.03,
        "primary.compositional_generalization/reasoning_gym": 0.02,
        "primary.abstract_analogical/s0_generators_v2_calsplit": 0.078,
        "primary.abstract_analogical/reasoning_core": 0.002,
        "primary.interactive_adaptation/s0_generators_v2_calsplit": 0.10,
        "supporting.math_reused_gsm8k/corpus_v2_gsm8k_verified": 0.06,
        "supporting.math_new/math_train": 0.03,
        "supporting.math_new/mathematics_dataset": 0.02,
        "supporting.math_new/reasoning_gym": 0.01,
        "supporting.code_algorithmic/mbpp_train": 0.055,
        "supporting.code_algorithmic/reasoning_core": 0.015,
        "supporting.reading_verified/corpus_v2_rag_verified": 0.06,
        "supporting.reading_verified/corpus_v2_multihop_verified": 0.02,
        "supporting.answerability/squad_v2_train_unanswerable": 0.03})
    reasoning = sum(v for k, v in spec["shares"].items() if k.startswith("primary."))
    assert reasoning == pytest.approx(0.70) and sum(spec["shares"].values()) == pytest.approx(1.0)
    # everything the candidate does not declare is @v1's
    for k in ("block_len", "rung_targets", "reserve_tokens", "acceptance", "size_measure"):
        assert spec[k] == v1[k]
    # @v1 itself is untouched: its own variant still resolves to its own split
    assert AB.cell_shares(v1, "stage2.reasoning_master@v1")["primary.abstract_analogical/reasoning_core"] == 0.03


def test_plan_2_keeps_v1s_capability_shares():
    spec, v1 = SP.candidate_spec(V3), A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    assert spec["shares"] == v1["shares"] and spec["corpus_id"] not in (v1["corpus_id"], "stage2.reasoning_master@v2")


def test_neither_candidate_can_be_built_through_the_arm_b_path():
    for path in (V2, V3):
        with pytest.raises(SystemExit):
            AB.main(["--variant", A.read_json(path)["corpus_id"], "--out", "/nonexistent/never-written"])


def _candidate(tmp_path, **edit):
    doc = A.read_json(V2)
    for k, v in edit.items():
        doc[k] = v
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc))
    return p


def test_a_candidate_whose_base_changed_or_that_reuses_an_id_or_breaks_a_sum_refuses(tmp_path):
    base = dict(A.read_json(V2)["derived_from"])
    with pytest.raises(SystemExit, match="not the bytes"):
        SP.candidate_spec(_candidate(tmp_path, derived_from={**base, "spec_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="id of its own"):
        SP.candidate_spec(_candidate(tmp_path, corpus_id="stage2.reasoning_master@v1"))
    shares = {**A.read_json(V2)["shares"], "primary.abstract_analogical": 0.09}
    with pytest.raises(SystemExit):
        SP.candidate_spec(_candidate(tmp_path, shares=shares))
    alloc = json.loads(json.dumps(A.read_json(V2)["source_allocation"]))
    alloc["primary.abstract_analogical"]["reasoning_core"] = 0.003
    with pytest.raises(SystemExit, match="sum"):
        SP.candidate_spec(_candidate(tmp_path, source_allocation=alloc))


def test_the_ladder_salt_comes_from_the_spec_and_v1s_is_unchanged(monkeypatch):
    v1 = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    assert v1["salt"] == AB.SALT
    units = [{"id": "u#t0", "unit_id": "u", "unit_sessions": 1, "stratum": "s", "attributes": {"source": "x"},
              "messages": []}]
    seen = {}
    monkeypatch.setattr(AB.B, "build", lambda sessions, spec, tok: seen.setdefault("spec", spec))
    AB.build(units, {**v1, "salt": "other/v9"}, {"s/x": 1.0}, [10], None)
    assert seen["spec"]["salt"] == "other/v9" and seen["spec"]["allow_composition_drift"] is False
    AB.build(units, v1, {"s/x": 1.0}, [10], None)
    assert seen["spec"]["salt"] == "other/v9"  # setdefault kept the first; v1's own salt is checked above


def _corpus(n_big=40, one_unit_cell=True, seed=0, lo=80):
    rng = random.Random(seed)
    units = [LadderUnit(f"a{i}", "A", rng.randint(lo, 120), family=f"fa{i % 4}") for i in range(n_big * 3)]
    units += [LadderUnit(f"b{i}", "B", rng.randint(lo, 120), family=f"fb{i % 3}") for i in range(n_big)]
    if one_unit_cell:
        units.append(LadderUnit("c0", "C", 100, family="fc"))
    shares = {"A": 0.6, "B": 0.38, "C": 0.02} if one_unit_cell else {"A": 0.6, "B": 0.4}
    return units, shares


def test_a_one_unit_cell_blocks_every_rung_and_is_absent_from_the_validation_tail():
    units, shares = _corpus(n_big=100)
    con, _ = CF.contract(units, shares, salt="t", reserve=1_000, targets=TARGETS)
    c = con["cells"]["C"]
    assert c["runs_dry_at_prefix"] == con["composition_capacity"] < TARGETS[0]
    assert c["blocks_rungs"] == TARGETS and c["reserve_units"] == 0
    assert "C" in con["validation_tail"]["cells_absent"]
    with pytest.raises(LadderError, match="composition capacity"):
        build_ladder(units, shares, TARGETS, salt="t", reserve=1_000)
    found = CF.unsatisfied(con)
    assert any(x.startswith("C: 1 unit(s)") for x in found) and any("ONE unit" in x for x in found)


@pytest.mark.parametrize("seed", range(40))
def test_the_contract_predicts_exactly_what_the_ladder_refuses(seed):
    """Unit sizes from 1 token up, so a cell can run dry a few tokens below a rung."""
    rng = random.Random(seed)
    units, shares = _corpus(n_big=rng.randint(60, 110), one_unit_cell=rng.random() < 0.3, seed=seed, lo=1)
    con, _ = CF.contract(units, shares, salt=f"s{seed}", reserve=1_000, targets=TARGETS)
    blocked = any(v["blocks_rungs"] for v in con["cells"].values())
    reachable = all(v is not None for v in con["rung_prefix_sizes"].values())
    assert CF.builds(units, shares, salt=f"s{seed}", reserve=1_000, targets=TARGETS) == (reachable and not blocked)
    # a rung one token past the capacity: its prefix ends a unit or more beyond, so it is refused
    edge = [con["composition_capacity"] + 1]
    at_edge, _ = CF.contract(units, shares, salt=f"s{seed}", reserve=1_000, targets=edge)
    assert any(v["blocks_rungs"] == edge for v in at_edge["cells"].values())
    assert not CF.builds(units, shares, salt=f"s{seed}", reserve=1_000, targets=edge)


def test_the_minimum_is_minimal_and_the_planning_count_is_at_least_it():
    units, _ = _corpus(n_big=100, one_unit_cell=False)
    shares = {"A": 0.6, "B": 0.4}  # B holds ~10k of the ~12.2k its share of the top rung + reserve needs
    k = CF.minimum_new_units(units, shares, ["B"], salt="m", reserve=1_000, targets=TARGETS, order="ascending")
    assert k["B"] > 0
    ok = lambda n: CF.builds(CF.with_stand_ins(units, {"B": n}, order="ascending"), shares, salt="m",
                             reserve=1_000, targets=TARGETS)
    assert ok(k["B"]) and not ok(k["B"] - 1)
    con, _ = CF.contract(CF.with_stand_ins(units, k, order="ascending"), shares, salt="m", reserve=1_000,
                         targets=TARGETS)
    plan = CF.planning_units(units, shares, k, prefix_top=con["rung_prefix_sizes"][str(TARGETS[-1])],
                             reserve=1_000)
    assert plan["B"] >= k["B"]


def test_tasks_cover_both_the_oversupply_and_the_binomial_chance():
    t = CF.tasks_for(3, 0.15)
    assert t["by_oversupply"] == 30 and t["tasks"] == max(30, t["by_binomial_p90"])
    assert t["chance_of_enough_units"] >= 0.9
    assert CF.tasks_for(11, 0.6721)["tasks"] == 25


def test_independence_finds_a_shared_prompt_and_a_validation_prompt_in_training():
    s = lambda uid, q, t: {"unit_id": uid, "messages": [{"role": "user", "content": q}], "target": t}
    rows = [s("u1", "same", {"r": 1}), s("u2", "same", {"r": 2}), s("u3", "other", {"r": 1})]
    got = CF.independence(rows, reserved={"u2"})
    assert got["units_sharing_a_prompt"] == 2 and got["units_sharing_a_target"] == 2
    assert got["validation_prompts_also_in_training"] == 1
    clean = CF.independence([s("u1", "a", {"r": 1}), s("u2", "b", {"r": 2})], reserved={"u2"})
    assert clean["units_sharing_a_prompt"] == clean["validation_prompts_also_in_training"] == 0


def test_attempt_never_allows_drift_and_reports_the_refusal(monkeypatch):
    seen = {}

    def fake(sessions, spec, shares, targets, tok, **kw):
        seen.update(kw)
        raise LadderError("rung 1: refused")
    monkeypatch.setattr(AB, "build", fake)
    got = CF.attempt([], {"rung_targets": [1]}, {}, None)
    assert got == {"built": False, "refusal": "rung 1: refused"} and not seen.get("drift")
