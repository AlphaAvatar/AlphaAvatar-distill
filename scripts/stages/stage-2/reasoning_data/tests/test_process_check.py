"""The static process check: a quoted format example is not a conclusion."""
from __future__ import annotations

from stages.reasoning_data.process_check import static_process_check

PROMPT = "Write the output grid as \\boxed{120 / 003 / 450}."


def test_a_format_example_quoted_from_the_prompt_is_ignored():
    r = "The format is \\boxed{120 / 003 / 450}. Applying the rule gives 222/000/000."
    assert static_process_check(r, "222/000/000", "grid", prompt=PROMPT).state == "partial"


def test_a_different_boxed_conclusion_is_a_contradiction():
    r = "Working through it, the answer is \\boxed{7}."
    assert static_process_check(r, "8", "int").state == "contradicted"


def test_the_answer_derived_in_the_reasoning_is_partial_not_verified():
    assert static_process_check("so the total is \\boxed{12}", "12", "int").state == "partial"
    assert static_process_check("x" * 400 + " hence 12", "12", "int").state == "partial"


def test_nothing_shown_is_not_assessed():
    assert static_process_check("I am not sure.", "12", "int").state == "not_assessed"
    assert static_process_check("anything", None, "int").state == "not_assessed"
