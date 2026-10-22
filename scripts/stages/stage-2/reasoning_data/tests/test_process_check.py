"""Text consistency, step verification and the answer are three records, and
none of them is read as more than it shows."""
from __future__ import annotations

from stages.reasoning_data.process_check import NO_STEP_CHECKER, reference_steps, text_consistency

PROMPT = "Write the output grid as \\boxed{120 / 003 / 450}."


def test_a_format_example_quoted_from_the_prompt_is_ignored():
    r = "The format is \\boxed{120 / 003 / 450}. Applying the rule gives 222/000/000."
    assert text_consistency(r, "222/000/000", "grid", prompt=PROMPT).state == "restated"


def test_a_different_last_boxed_value_is_never_a_contradiction():
    """An intermediate quantity boxed inside a derivation is not a conclusion."""
    c = text_consistency("First \\boxed{7} items per row, so the total is 8.", "8", "int")
    assert c.state == "restated" and "contradict" not in c.reason
    after = text_consistency("Total \\boxed{8}; check: rows hold \\boxed{7}.", "8", "int")
    assert after.state == "restated" and "different value is boxed after it" in after.reason
    only_other = text_consistency("Working through it, \\boxed{7}.", "8", "int")
    assert only_other.state == "not_restated" and "intermediates" in only_other.reason


def test_restating_the_answer_is_textual_agreement_only():
    c = text_consistency("so the total is \\boxed{12}", "12", "int")
    assert c.state == "restated" and "not a check of any step" in c.scope
    assert text_consistency("x" * 400 + " hence 12", "12", "int").state == "restated"


def test_nothing_shown_is_not_restated_and_no_answer_is_not_assessed():
    assert text_consistency("I am not sure.", "12", "int").state == "not_restated"
    assert text_consistency("anything", None, "int").state == "not_assessed"


def test_static_generator_steps_are_not_assessed():
    assert NO_STEP_CHECKER.state == "not_assessed"


CONCLUSIONS = ["Max is a wumpus.", "Max is a vumpus.", "Max is not shy."]


def test_reference_steps_all_stated_is_partial_never_verified():
    r = "Max is a yumpus, so Max is a wumpus. Each wumpus is a vumpus: Max is a vumpus. Max is not shy."
    c = reference_steps(r, CONCLUSIONS)
    assert c.state == "partial" and "presence, not validity" in c.scope


def test_reference_steps_negated_is_contradicted_and_missing_is_not_assessed():
    assert reference_steps("Max is not a wumpus. Max is a vumpus. Max is not shy.", CONCLUSIONS).state == "contradicted"
    assert reference_steps("Max is a wumpus. Max is not shy.", CONCLUSIONS).state == "not_assessed"
    assert reference_steps("anything", []).state == "not_assessed"
