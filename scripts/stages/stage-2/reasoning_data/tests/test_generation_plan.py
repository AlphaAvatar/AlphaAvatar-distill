"""The paid generation list: only verifiable tasks, drawn without looking at results."""
from __future__ import annotations

from stages.reasoning_data import assets as A
from stages.reasoning_data import generation_plan as GP

SPEC = A.read_json(A.CONFIG_DIR / "corpus_spec.json")


def _ext(module, answer_type, **kw):
    return {"task_id": f"{module}-{kw.get('i', 0)}", "source": "reasoning_gym", "module": module,
            "answer_type": answer_type, "stratum": "primary.compositional_generalization",
            "messages": [{"role": "user", "content": "q"}], "level": kw.get("level", 0)}


def test_a_task_without_a_usable_verifier_never_reaches_the_list():
    assert GP.verifier_of(_ext("rotate_matrix", "matrix"))[0] == "compare:matrix"
    assert GP.verifier_of(_ext("tower_of_hanoi", "checker"))[0] == "checker:reasoning_gym/tower_of_hanoi"
    assert GP.verifier_of(_ext("rotate_matrix", "needs_checker"))[0] is None
    assert GP.verifier_of(_ext("arc_agi", "matrix"))[1] == "arc_agi: not a candidate module"
    assert GP.verifier_of({"source": "x", "verifier": "no_such"})[0] is None
    assert GP.verifier_of({"source": "x"})[0] is None


def test_round_robin_is_balanced_over_groups_and_keeps_pool_order():
    pool = [_ext("a", "text", i=i, level=i % 2) for i in range(6)] + [_ext("b", "text", i=i) for i in range(6)]
    got = GP.round_robin(pool, 6)
    assert [GP.group_of(t) for t in got] == ["a/0", "a/1", "b/0"] * 2
    assert GP.round_robin(pool, 6) == got and len(GP.round_robin(pool, 100)) == 12


def test_a_cell_is_short_only_when_its_supply_cannot_fill_it_in_expectation(tmp_path):
    rows = [{**_ext("rotate_matrix", "matrix", i=i), "task_id": f"t{i}"} for i in range(30)]
    pool = tmp_path / "p.jsonl"
    pool.write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))
    cell = next(c for c in GP.build("stage2.reasoning_master@v1", [pool])["cells"]
                if c["source"] == "reasoning_gym" and c["stratum"] == "primary.compositional_generalization")
    assert cell["generate"] == 30 and cell["below_margin"]
    assert cell["shortfall"] == (cell["expected_units"] < cell["independent_units_estimate"])


def test_the_run_input_carries_what_run_battery_reads():
    t = {"task_id": "m1", "source": "mbpp_train", "stratum": "supporting.code_algorithmic", "family_id": "mbpp",
         "difficulty": "", "verifier": "code_asserts",
         "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]}
    row = GP.run_input(t, SPEC)
    assert row["messages"] == [{"role": "user", "content": "q"}]       # the system message is the run's
    assert {k: row[k] for k in ("generator_id", "family_id", "axis", "interactive")} == {
        "generator_id": "mbpp_train", "family_id": "mbpp", "axis": "supporting.code_algorithmic",
        "interactive": False}


def test_only_the_chosen_variant_gets_a_paid_list(tmp_path):
    import pytest
    from stages.reasoning_data import source_plan as SP
    assert SP.plan("stage2.reasoning_master@v1")["chosen"]
    assert not SP.plan("stage2.reasoning_master@v1-project-only")["chosen"]
    with pytest.raises(SystemExit, match="not the chosen"):
        GP.main(["--variant", "stage2.reasoning_master@v1-project-only", "--pools", str(tmp_path / "x.jsonl"),
                 "--inputs-dir", str(tmp_path)])


def test_the_plan_writes_its_inputs_where_governance_binds_and_the_pod_installs_them():
    from stages.reasoning_data import governance as GOV
    assert GP.INPUTS_DIR == A.REPO / GOV.INPUTS_DIR
