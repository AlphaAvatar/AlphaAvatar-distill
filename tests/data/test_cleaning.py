"""Rule-level tests for the D1 candidate-cleaning pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "evaluation"))
import degeneration  # noqa: E402

from aadistill.data.cleaning import (  # noqa: E402
    check_completion,
    check_correctness,
    check_serialization,
    check_tool_protocol,
    screen_candidate,
    select_clean,
)

IM_END = "<|im_end|>"


def candidate(index=0, think="reasoning here", answer="The answer is 7.",
              tokens=None, **over):
    raw = f"{think}</think>{answer}{IM_END}"
    base = {
        "index": index, "seed": 1000 + index, "answer": answer, "think": think,
        "raw": raw, "tokens": tokens if tokens is not None else list(range(50)),
        "new_tokens": 50, "hit_cap": False, "over_budget": False,
        "finished": True, "length_limited": False, "correct": True,
        "correctness_verdict": "ok",
    }
    base.update(over)
    return base


def example(group="code_math", source="gsm8k", gold="The answer is 7.",
            tools=None, candidates=None):
    return {"group": group, "source": source, "gold": gold, "tools": tools,
            "candidates": candidates or [candidate()]}


# --------------------------------------------------------------------------
# stage 1 — serialization
# --------------------------------------------------------------------------
def test_serialization_accepts_a_well_formed_candidate():
    assert check_serialization(candidate()) is None


def test_serialization_rejects_empty_answer():
    assert check_serialization(candidate(answer="   ")) == "empty_answer"


def test_serialization_rejects_unclosed_think():
    bad = candidate()
    bad["raw"] = f"thinking forever{IM_END}"
    assert check_serialization(bad) == "think_not_closed"


def test_serialization_rejects_stray_control_marker():
    assert check_serialization(
        candidate(answer="see <|im_start|>system")) == "stray_marker"


def test_serialization_rejects_a_reopened_think_block():
    bad = candidate()
    bad["raw"] = f"a</think>b<think>c</think>d{IM_END}"
    assert check_serialization(bad) == "think_delimiters_invalid"


# --------------------------------------------------------------------------
# stage 2 — correctness
# --------------------------------------------------------------------------
def test_correctness_accepts_the_matching_final_number():
    assert check_correctness(candidate(), example()) is None


def test_correctness_rejects_a_wrong_final_number():
    reason = check_correctness(candidate(answer="The answer is 8."), example())
    assert reason == "correctness:answer_mismatch"


def test_correctness_passes_slices_with_no_mechanical_key():
    # `code` has no answer key; stage 2 must not invent a verdict for it.
    ex = example(group="code_math", source="mbpp", gold="def f(): pass")
    assert check_correctness(candidate(answer="def f(): return 1"), ex) is None


def test_correctness_does_not_apply_a_generic_length_gate():
    """A long but correct derivation must pass (AGENTS.md P3/P10).

    `verify()` would reject this through `hygiene_reason`'s MAX_ANSWER_WORDS;
    stage 2 deliberately calls the answer key alone.
    """
    long_answer = ("step " * 5000) + "The answer is 7."
    assert check_correctness(candidate(answer=long_answer), example()) is None


# --------------------------------------------------------------------------
# stage 3 — tool protocol
# --------------------------------------------------------------------------
TOOLS = [{"type": "function", "function": {
    "name": "get_news", "description": "news",
    "parameters": {"type": "object",
                   "properties": {"country": {"type": "string"}},
                   "required": ["country"]}}}]

CALL = '<tool_call>\n{"name": "get_news", "arguments": {"country": "FR"}}\n</tool_call>'


def test_tool_call_without_a_schema_is_rejected():
    reason = check_tool_protocol(candidate(answer=CALL), example(tools=None))
    assert reason == "unexpected_tool_call"


def test_valid_tool_call_against_a_declared_schema_passes():
    assert check_tool_protocol(
        candidate(answer=CALL), example(tools=TOOLS)) is None


def test_prose_answer_with_a_schema_available_passes():
    assert check_tool_protocol(
        candidate(answer="Here are the headlines."), example(tools=TOOLS)) is None


def test_undeclared_tool_name_is_rejected():
    call = '<tool_call>{"name": "other", "arguments": {}}</tool_call>'
    assert check_tool_protocol(
        candidate(answer=call), example(tools=TOOLS)) == "tool_name_undeclared"


def test_missing_required_argument_is_rejected():
    call = '<tool_call>{"name": "get_news", "arguments": {}}</tool_call>'
    assert check_tool_protocol(
        candidate(answer=call),
        example(tools=TOOLS)) == "tool_required_argument_missing"


def test_unparseable_tool_payload_is_rejected():
    call = "<tool_call>{not json}</tool_call>"
    assert check_tool_protocol(
        candidate(answer=call), example(tools=TOOLS)) == "tool_call_not_json"


# --------------------------------------------------------------------------
# stage 4 — completion
# --------------------------------------------------------------------------
def test_completion_accepts_a_terminated_generation():
    assert check_completion(candidate(), degeneration) is None


@pytest.mark.parametrize("flag,expected", [
    ("hit_cap", "context_limit_reached"),
    ("over_budget", "over_budget"),
    ("length_limited", "length_limited"),
])
def test_completion_rejects_censored_generations(flag, expected):
    assert check_completion(candidate(**{flag: True}), degeneration) == expected


def test_completion_rejects_an_unterminated_generation():
    bad = candidate()
    bad["raw"] = bad["raw"].replace(IM_END, "")
    assert check_completion(bad, degeneration) == "not_terminated"


def test_completion_rejects_a_token_level_cycle():
    cyclic = candidate(tokens=[7, 8, 9] * 400)
    assert check_completion(cyclic, degeneration) == "degenerate:cycle"


# --------------------------------------------------------------------------
# stage 5 — selection
# --------------------------------------------------------------------------
def _by_words(c):
    return len(c["answer"].split())


def test_the_corpus_candidate_is_retained_when_it_survives():
    ex = example(candidates=[
        candidate(0, answer="A much longer but still correct answer is 7."),
        candidate(1, answer="The answer is 7."),
    ])
    verdict = select_clean(ex, degeneration, _by_words, original_index=0)
    assert verdict["retained_original"] is True
    assert verdict["chosen"]["index"] == 0


def test_a_failing_original_is_replaced_by_the_median_survivor():
    """Length enters only among survivors, and picks the middle one.

    Survivor lengths are 12, 7 and 4 words; the median is 7, so the 7-word
    candidate wins — the shortest (4 words, which states the answer without
    deriving it) does not.
    """
    ex = example(candidates=[
        candidate(0, answer="The answer is 8."),                     # wrong
        candidate(1, answer="First add the parts, then divide them, so the answer is 7."),
        candidate(2, answer="Adding the parts gives the answer is 7."),
        candidate(3, answer="The answer is 7."),
    ])
    verdict = select_clean(ex, degeneration, _by_words, original_index=0)
    assert verdict["retained_original"] is False
    assert verdict["chosen"]["index"] == 2
    assert verdict["reasons"][0] == "correctness:answer_mismatch"
    assert verdict["rule"] == "median"


def test_median_of_an_even_survivor_set_ties_on_index():
    """With two survivors the median is their midpoint, so both are equidistant."""
    ex = example(candidates=[
        candidate(0, answer="The answer is 8."),
        candidate(1, answer="After working it through, the answer is 7."),
        candidate(2, answer="The answer is 7."),
    ])
    verdict = select_clean(ex, degeneration, _by_words, original_index=0)
    assert verdict["chosen"]["index"] == 1


def test_shortest_rule_is_still_available_for_comparison():
    ex = example(candidates=[
        candidate(0, answer="The answer is 8."),
        candidate(1, answer="First add the parts, then divide them, so the answer is 7."),
        candidate(2, answer="Adding the parts gives the answer is 7."),
        candidate(3, answer="The answer is 7."),
    ])
    median = select_clean(ex, degeneration, _by_words, original_index=0)
    shortest = select_clean(ex, degeneration, _by_words, original_index=0,
                            rule="shortest")
    assert median["chosen"]["index"] == 2
    assert shortest["chosen"]["index"] == 3
    assert median["chosen"]["index"] != shortest["chosen"]["index"]


def test_an_unknown_selection_rule_fails_loudly():
    ex = example(candidates=[candidate(0, answer="The answer is 8."),
                             candidate(1, answer="The answer is 7.")])
    with pytest.raises(ValueError, match="selection rule"):
        select_clean(ex, degeneration, _by_words, original_index=0, rule="longest")


def test_survivor_lengths_are_recorded_for_the_audit():
    ex = example(candidates=[
        candidate(0, answer="The answer is 8."),
        candidate(1, answer="Adding the parts gives the answer is 7."),
        candidate(2, answer="The answer is 7."),
    ])
    verdict = select_clean(ex, degeneration, _by_words, original_index=0)
    assert set(verdict["survivor_lengths"]) == {1, 2}
    assert 0 not in verdict["survivor_lengths"]


def test_a_retained_original_never_consults_length():
    """The corpus's own candidate wins outright, however long it is."""
    ex = example(candidates=[
        candidate(0, answer="A far longer but entirely correct route to the answer is 7."),
        candidate(1, answer="The answer is 7."),
    ])
    verdict = select_clean(ex, degeneration, _by_words, original_index=0)
    assert verdict["retained_original"] is True
    assert verdict["chosen"]["index"] == 0


def test_no_survivor_yields_no_target():
    ex = example(candidates=[candidate(0, answer="The answer is 8."),
                             candidate(1, answer="The answer is 9.")])
    verdict = select_clean(ex, degeneration, _by_words, original_index=0)
    assert verdict["chosen"] is None
    assert verdict["n_survivors"] == 0


def test_stage_order_reports_the_first_failure_only():
    """A candidate that is both wrong and degenerate reports correctness."""
    bad = candidate(0, answer="The answer is 8.", tokens=[1, 2] * 500)
    assert screen_candidate(bad, example(), degeneration) == \
        "correctness:answer_mismatch"
