"""The answer comparators parse the WHOLE answer per type (`generation_adapter.COMPARE`).

Accepting a module's own gold and refusing one foreign string proves nothing
about a comparator (the review's four false accepts all passed that), so each
type is held to semantic counterexamples and to legal equivalent forms.
"""
from __future__ import annotations

import json

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import generation_adapter as G

POOL = A.REPO / "data/stages/stage-2/reasoning_data/tasks"


def same(kind, got, gold):
    return G.compares_equal(kind, got, gold)


# the four false accepts the review found, and their relatives -------------------

def test_a_fraction_is_one_number_not_a_two_element_set():
    assert not same("decimal_set", "\\frac{1}{2}", "1, 2")
    assert not same("decimal_set", "1", "1, 2") and not same("decimal_set", "1, 2, 3", "1, 2")
    assert not same("decimal_set", "1, 2, x", "1, 2") and not same("decimal_set", "1, , 2", "1, 2")


def test_a_reshaped_matrix_is_a_different_matrix():
    assert not same("matrix", "[[1,2,3,4]]", "[[1,2],[3,4]]")
    assert not same("matrix", "1 2 3 4", "1 2\n3 4")
    assert not same("matrix", "1 3\n2 4", "1 2\n3 4")                  # transposed
    assert not same("matrix", "1 2\n3 4\n5 6", "1 2\n3 4")
    assert not same("matrix", "The grid is:\n1 2\n3 4", "1 2\n3 4")     # text is not a row of cells


def test_a_number_is_one_number_and_nothing_else():
    assert not same("number", "not 7", "7")
    for bad in ("7 apples", "about 7", "7, 8", "3+4", "7 or 8", "x = 7 and y = 2", "1,000"):
        assert not same("number", bad, "7"), bad
    with pytest.raises(G.Unparseable):
        G.COMPARE["number"]("not 7", "7")


def test_case_is_kept_where_it_is_part_of_the_answer():
    assert not same("text", "CBBA", "cbba") and not same("text", "Faint", "faint")
    assert not same("text", "faint.", "faint")                         # a period is an output byte
    assert not same("repr", "[true]", "[True]") and not same("repr", "['A']", "['a']")
    assert not same("token_seq", "c c a", "C C A")
    assert not same("assignment", "c=4", "C=4")
    assert same("text_ci", "Arnold", "arnold") and same("text_ci", "None.", "None")


def test_order_is_kept_where_it_is_part_of_the_answer():
    assert not same("token_seq", "A B C", "C B A") and not same("token_seq", "A B", "A B B")
    assert not same("repr", "[2, -1]", "[-1, 2]") and not same("repr", "[2.0]", "[2]")
    assert not same("int_set", "0 1", "0 1 3") and not same("int_set", "0 0 1", "0 1")
    assert not same("assignment", "A=1,B=3", "A=1,B=2") and not same("assignment", "A=1", "A=1,B=2")
    assert not same("assignment", "A=1,A=1,B=2", "A=1,B=2")


# legal equivalent forms are accepted ---------------------------------------------

def test_equal_numbers_in_any_literal_form():
    for got in ("0.5", "1/2", "\\frac{1}{2}", "\\dfrac{1}{2}", "x = 0.5", "\\text{0.5}", "0.50", ".5"):
        assert same("number", got, "1/2"), got
    assert same("number", "-3", "-3") and same("number", "−3.0", "-3") and same("number", "-\\frac{6}{2}", "-3")


def test_an_unordered_set_in_any_order_or_bracketing():
    for got in ("1, 2", "2, 1", "{1, 2}", "\\{1, 2\\}", "x = 2, x = 1", "1 and 2", "[2; 1]"):
        assert same("decimal_set", got, "1, 2"), got
    assert same("decimal_set", "-0.4934, 0.0, 1.5472", "1.5472, -0.4934, 0")
    assert same("decimal_set", "1.54725", "1.5472")                   # inside the 4-decimal rounding
    assert not same("decimal_set", "1.5475", "1.5472")
    assert same("int_set", "4 3 1 0", "0 1 3 4") and same("int_set", "[0, 1, 3, 4]", "0 1 3 4")
    assert same("assignment", "B=2, A=1", "A=1,B=2") and same("assignment", "A = 1 , B = 2", "A=1,B=2")


def test_a_matrix_with_other_whitespace_brackets_or_latex_but_the_same_structure():
    gold = "[[0,1],[1,0]]"
    for got in ("[[0, 1], [1, 0]]", "0 1\n1 0", "[[0,1],\n [1,0]]", "\\begin{bmatrix} 0 & 1 \\\\ 1 & 0 \\end{bmatrix}",
                "0 1 \\\\ 1 0"):
        assert same("matrix", got, gold), got
    assert same("token_seq", "9 7 0\n9 1", "9 7 0 9 1") and same("token_seq", "[9, 7, 0, 9, 1]", "9 7 0 9 1")
    assert same("repr", "[-1,2,-1]", "[-1, 2, -1]") and same("repr", '["a", 1]', "['a', 1]")


def test_a_boolean_is_exactly_true_or_false():
    assert same("bool", "\\text{False}", "False") and same("bool", "true", "True") and same("bool", "False.", "False")
    assert not same("bool", "not True", "True") and not same("bool", "True or False", "True")
    with pytest.raises(G.Unparseable):                     # unparseable, not merely unequal
        G.COMPARE["bool"]("not True", "True")


# the adapter and the build ---------------------------------------------------------

def _task(kind, gold):
    return {"task_id": "t", "source": "reasoning_gym", "module": "m", "answer_type": kind, "answer": gold,
            "messages": [{"role": "user", "content": "q"}]}


def test_an_unparseable_answer_is_incorrect_and_says_so():
    c = G.external_answer(_task("number", "7"), "so \\boxed{not 7}")
    assert c.state == "incorrect" and c.reason.startswith("unparseable answer")


def test_one_record_whose_verifier_raises_costs_only_that_candidate(monkeypatch):
    from stages.reasoning_data.tests.test_generation_adapter import TASK, _record

    def boom(task, final):
        raise KeyError("answer")
    monkeypatch.setattr(G, "external_answer", boom)
    c = G.candidate(TASK, _record("cat"), 0)
    assert c.answer == "unverifiable" and not c.accepted_for_training
    assert any("verifier error: KeyError" in r for r in c.reasons)


def test_an_error_outside_the_checks_is_not_swallowed():
    from stages.reasoning_data.tests.test_generation_adapter import TASK, _record
    rec = _record("cat")
    del rec["turns"][0]["generated_tokens"]
    with pytest.raises(KeyError):
        G.candidate(TASK, rec, 0)


@pytest.mark.parametrize("pool", ["external_tasks_v2.jsonl"])
def test_every_kept_gold_passes_end_to_end_and_every_perturbation_is_refused(pool):
    rows = [json.loads(line) for line in open(POOL / pool)]
    kinds = {r["answer_type"] for r in rows}
    assert "number_set" not in kinds and kinds - {"checker"} <= set(G.COMPARE)
    for r in rows:
        if r["answer_type"] == "checker":
            continue
        assert G.external_answer(r, "\\boxed{" + r["answer"] + "}").state == "correct", r["task_id"]
        for w in G.perturbations(r["answer_type"], r["answer"]):
            assert not same(r["answer_type"], w, r["answer"]), (r["module"], w)


def test_the_build_refuses_a_comparator_that_accepts_a_perturbed_gold(tmp_path, monkeypatch):
    from stages.reasoning_data import external_tasks as ET
    export = tmp_path / "export.jsonl"
    export.write_text(json.dumps({
        "source": "reasoning_gym", "module": "bf", "version": "0.1.25", "level": 0, "seed": 7, "index": 0,
        "python_hash_seed": "0", "gold": "faint", "gold_in_prompt": False, "reference_trace": None,
        "prompt": "This is a BF (Brainf*ck) computer program. What is the output?\n\n"
                  ">[-]>[-]<>++++++++++[<++++++++++>-]<++.-----.++++++++.+++++.++++++.<\n\n"
                  "Respond only with the exact output of the program."}) + "\n")
    assert len(ET.build([export])["tasks"]) == 1
    monkeypatch.setitem(G.COMPARE, "text", lambda got, gold: G._text_exact(got).casefold() == G._text_exact(gold).casefold())
    with pytest.raises(SystemExit, match="does not fit"):
        ET.build([export])


def test_a_grid_whose_rows_are_broken_by_a_literal_backslash_n_is_read_as_rows():
    """P1v: the teacher boxed a correct mini_sudoku grid with literal ``\\n``
    (backslash, n) between the rows; the structure is still checked."""
    from stages.reasoning_data import generation_adapter as G
    gold = "2 3 4 1\n4 1 2 3\n3 2 1 4\n1 4 3 2"
    assert G.COMPARE["matrix"]("2 3 4 1\\n4 1 2 3\\n3 2 1 4\\n1 4 3 2", gold)
    assert not G.COMPARE["matrix"]("2 3 4 1\\n4 1 2 3\\n3 2 1 4\\n1 4 2 3", gold)    # a wrong cell
    assert not G.COMPARE["matrix"]("2 3 4 1 4 1 2 3\\n3 2 1 4 1 4 3 2", gold)       # reshaped
    assert not G.COMPARE["matrix"]("2 3\\newline 4 1", "2 3\n4 1")                 # a LaTeX command is not a row break
