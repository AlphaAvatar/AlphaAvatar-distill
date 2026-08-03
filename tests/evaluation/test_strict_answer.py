"""Tests for the strict final-answer rule that replaces last-number extraction."""

from __future__ import annotations

from aadistill.evaluation.strict_answer import (
    extract_final_answer,
    protocol_valid,
    score_numeric,
)

IM_END = "<|im_end|>"


def generation(answer, think="working it out", **over):
    record = {"raw": f"{think}</think>{answer}{IM_END}",
              "degeneration_triggered": False, "natural_termination": True,
              "degeneration_kind": None}
    record.update(over)
    return record


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def test_boxed_wins_over_a_later_marker():
    got, how = extract_final_answer(r"so \boxed{72}. Final Answer: 99")
    assert (got, how) == ("72", "boxed")


def test_explicit_marker_is_accepted_without_a_box():
    assert extract_final_answer("Final Answer: 250") == ("250", "marker")


def test_bare_answer_marker_is_accepted():
    assert extract_final_answer("Answer: 42") == ("42", "marker")


def test_markdown_decorated_marker_is_accepted():
    assert extract_final_answer("**Final Answer:** 18") == ("18", "marker")


def test_a_trailing_number_with_no_marker_is_not_an_answer():
    got, how = extract_final_answer("he pays 100 + 150 = 250")
    assert got is None and how == "no_final_answer"


def test_numbers_inside_a_tool_call_are_never_read():
    answer = '<tool_call>{"name": "calc", "arguments": {"x": 250}}</tool_call>'
    assert extract_final_answer(answer) == (None, "no_final_answer")


def test_marker_inside_a_tool_call_is_stripped_too():
    answer = "<tool_call>Final Answer: 250</tool_call>"
    assert extract_final_answer(answer) == (None, "no_final_answer")


def test_unbalanced_box_falls_through_rather_than_guessing():
    got, how = extract_final_answer(r"\boxed{72 and then nothing")
    assert got is None and how == "no_final_answer"


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
def test_protocol_valid_on_a_clean_generation():
    assert protocol_valid(generation("Final Answer: 7")["raw"]) == (True, "ok")


def test_unterminated_generation_is_protocol_invalid():
    assert protocol_valid("thinking</think>Final Answer: 7")[1] == "not_terminated"


def test_tool_call_in_a_no_tool_task_is_protocol_invalid():
    raw = generation('<tool_call>{"name": "x"}</tool_call>')["raw"]
    assert protocol_valid(raw)[1] == "unexpected_tool_call"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def test_correct_boxed_answer_scores():
    verdict = score_numeric(generation(r"\boxed{250}"), "250")
    assert verdict["correct"] and verdict["reason"] == "ok"


def test_comma_and_currency_forms_normalize():
    assert score_numeric(generation("Final Answer: $1,250.00"), "1250")["correct"]


def test_gold_number_without_a_stated_conclusion_is_incorrect():
    verdict = score_numeric(generation("so 100 + 150 = 250"), "250")
    assert not verdict["correct"] and verdict["reason"] == "no_final_answer"


def test_degenerate_generation_containing_the_gold_is_incorrect():
    record = generation(r"\boxed{250}", degeneration_triggered=True,
                        degeneration_kind="cycle")
    verdict = score_numeric(record, "250")
    assert not verdict["correct"]
    assert verdict["reason"] == "degenerate:cycle"
    # The distinction the flag exists to preserve.
    assert verdict["answer_matches_ignoring_protocol"] is True


def test_protocol_invalid_generation_containing_the_gold_is_incorrect():
    record = {"raw": r"thinking</think>\boxed{250}", "natural_termination": False}
    verdict = score_numeric(record, "250")
    assert not verdict["correct"]
    assert verdict["reason"] == "protocol:not_terminated"


def test_termination_is_reported_separately_from_correctness():
    verdict = score_numeric(generation("Final Answer: 8"), "250")
    assert verdict["natural_termination"] is True
    assert verdict["correct"] is False
    assert verdict["reason"] == "answer_mismatch"
