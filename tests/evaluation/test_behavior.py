"""Scorer tests for eval_behavior_v0 — synthetic generations, no model.

Load-bearing checks: the think-block contract is read the way the training data
teaches it (prompt opens the block, generation must close it exactly once), the
observed `s2_blocks_v1` failure modes score as failures, tool-call validity
tightens correctly from "emitted" to "exact match", and aggregation never mixes
a group-specific metric into the overall row.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.evaluation.behavior import (
    aggregate,
    behavior_score,
    contains_gold,
    echo_rate,
    final_number,
    format_ok,
    is_refusal,
    parse_tool_calls,
    repetition_rate,
    score_sample,
    score_tool_call,
    split_generation,
)

GOOD = "</think>\n\n2 + 2 equals 4.<|im_end|>"
# The real s2_blocks_v1 failure on "What is 2+2?": never closed the think block,
# so the "answer" was emitted inside it and there is no answer at all.
NEVER_CLOSED = 'Okay, the question is: "What is 2+2?"<|im_end|>'
TRUNCATED = "</think>\n\nThe answer is 4"  # hit max_new_tokens, no terminator


def test_well_formed_generation():
    p = split_generation(GOOD)
    assert p["terminated"] and p["think_closed"] and p["think_immediate"]
    assert p["no_stray_markers"] and format_ok(p)
    assert p["answer"] == "2 + 2 equals 4."


def test_unclosed_think_block_has_no_answer():
    p = split_generation(NEVER_CLOSED)
    assert p["terminated"] is True
    assert p["think_closed"] is False
    assert p["answer"] == ""
    assert format_ok(p) is False


def test_truncated_generation_fails_on_termination_only():
    p = split_generation(TRUNCATED)
    assert p["think_closed"] is True and p["terminated"] is False
    assert format_ok(p) is False
    assert p["answer"] == "The answer is 4"


def test_reopened_think_block_and_stray_markers_fail():
    p = split_generation("</think>\nhi<think>\nmore</think><|im_end|>")
    assert p["think_closed"] is False  # two closes, one re-open
    assert "<think>" in p["stray_markers"]
    p2 = split_generation("</think>\nhi<|im_start|>user\nagain<|im_end|>")
    assert p2["no_stray_markers"] is False and format_ok(p2) is False


def test_thinking_before_answer_is_valid_but_not_immediate():
    p = split_generation("Let me think about it.</think>\n\nParis.<|im_end|>")
    assert format_ok(p) and p["think_closed"]
    assert p["think_immediate"] is False
    assert p["answer"] == "Paris."


def test_echo_rate_flags_question_restatement():
    prompt = "What is the capital of France and why is it famous?"
    assert echo_rate("What is the capital of France", prompt) == 1.0
    assert echo_rate("Paris, for its museums and food.", prompt) == 0.0
    assert echo_rate("", prompt) == 0.0  # too short to judge


def test_repetition_rate_flags_degeneracy():
    assert repetition_rate("a b c a b c a b c") > 0.5
    assert repetition_rate("the quick brown fox jumps over lazy dogs") == 0.0


def test_contains_gold_normalizes_punctuation_and_case():
    assert contains_gold("The answer is Vladislav Lantratov.", "vladislav lantratov")
    assert not contains_gold("The answer is David Hallberg.", "Vladislav Lantratov")
    assert not contains_gold("anything", "")


def test_refusal_detector_covers_gold_phrasings():
    # Verbatim gold refusals from the refusal_uncertainty val split.
    for gold in [
        "That detail isn't in the provided text, so I don't know the answer.",
        "The answer isn't contained in the passage, so I won't guess.",
        "I don't see that information in the context.",
        "The provided context does not contain the information needed.",
        "The passage doesn't cover that, so I'd rather say so than guess.",
        "I cannot answer that from the context.",
    ]:
        assert is_refusal(gold), gold
    assert not is_refusal("The Wayback Machine was migrated in 2009.")


def test_final_number_normalizes():
    assert final_number("He pays 100+150=$250\nThe answer is 250.") == "250"
    assert final_number("So the total is 1,234 dollars") == "1234"
    assert final_number("The answer is 12.50") == "12.5"
    assert final_number("no digits here") is None


TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculate_age",
        "parameters": {"type": "object", "properties": {"date_of_birth": {"type": "string"}},
                       "required": ["date_of_birth"]},
    },
}]
GOLD_CALLS = [{"type": "function",
               "function": {"name": "calculate_age",
                            "arguments": {"date_of_birth": "1990-05-15"}}}]


def wrap(payload: str) -> str:
    return f"</think>\n\n<tool_call>\n{payload}\n</tool_call><|im_end|>"


def test_tool_call_scoring_tightens_correctly():
    exact = wrap('{"name": "calculate_age", "arguments": {"date_of_birth": "1990-05-15"}}')
    s = score_tool_call(split_generation(exact)["answer"], TOOLS, GOLD_CALLS)
    assert all(s.values())

    # Right shape, different argument value: valid but not an exact match.
    other = wrap('{"name": "calculate_age", "arguments": {"date_of_birth": "1991-01-01"}}')
    s = score_tool_call(split_generation(other)["answer"], TOOLS, GOLD_CALLS)
    assert s["tool_args_schema_ok"] and not s["tool_call_exact_match"]

    # Hallucinated function name.
    bad_name = wrap('{"name": "get_age", "arguments": {"date_of_birth": "1990-05-15"}}')
    s = score_tool_call(split_generation(bad_name)["answer"], TOOLS, GOLD_CALLS)
    assert s["tool_call_parsed"] and not s["tool_name_valid"]

    # Missing a required parameter.
    missing = wrap('{"name": "calculate_age", "arguments": {}}')
    s = score_tool_call(split_generation(missing)["answer"], TOOLS, GOLD_CALLS)
    assert s["tool_name_valid"] and not s["tool_args_schema_ok"]

    # Malformed JSON, and no call at all.
    broken = score_tool_call(split_generation(wrap('{"name": '))["answer"], TOOLS, GOLD_CALLS)
    assert broken["tool_call_emitted"] and not broken["tool_call_parsed"]
    none = score_tool_call("I'll look that up for you.", TOOLS, GOLD_CALLS)
    assert not any(none.values())


def test_parse_tool_calls_handles_multiple_blocks():
    answer = wrap('{"name": "a", "arguments": {}}') + wrap('{"name": "b", "arguments": {}}')
    calls = parse_tool_calls(answer)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_score_sample_applies_group_specific_metrics():
    rag = score_sample(
        {"id": "x", "group": "rag_evidence", "prompt_text": "Who won?",
         "gold_answer": "Chordiant"},
        "</think>\n\nChordiant.<|im_end|>")
    assert rag["evidence_hit"] and rag["format_ok"]
    assert "refusal" not in rag and "tool_call_parsed" not in rag

    gsm = score_sample(
        {"id": "y", "group": "code_math", "prompt_text": "cost?",
         "gold_answer": "The answer is 250.", "gsm8k_answer": "250"},
        "</think>\n\nHe pays $250 total.<|im_end|>")
    assert gsm["answer_em"]


def test_truncation_is_recorded_separately_from_termination():
    sample = {"id": "t", "group": "instruction", "prompt_text": "q", "gold_answer": "a"}
    cut = score_sample(sample, TRUNCATED, hit_cap=True)
    assert cut["truncated_at_cap"] and not cut["terminated"] and not cut["format_ok"]
    # Same text, but the model stopped on its own below the cap — a different
    # defect (it emitted no terminator despite having room).
    ran_out = score_sample(sample, TRUNCATED, hit_cap=False)
    assert not ran_out["truncated_at_cap"] and not ran_out["terminated"]
    assert score_sample(sample, GOOD)["truncated_at_cap"] is False


def test_echo_credit_blocks_free_hits_from_parroting():
    # Verbatim shape of the s1@660 failure: the prompt tells the model to say it
    # cannot answer, and the model re-emits the prompt. Raw refusal fires; the
    # credited variant must not.
    prompt = ("Answer the question using only the provided context. If the context "
              "does not contain the answer, say you cannot answer from the context.\n"
              "Context: In 1059, the right of electing the pope was reserved to the "
              "principal clergy of Rome.")
    parrot = ("</think>\n\nAnswer the question using only the provided context. If the "
              "context does not contain the answer, say you cannot answer from the "
              "context.\nContext: In 1059, the right of electing the pope was reserved "
              "to the principal clergy of Rome.<|im_end|>")
    s = score_sample({"id": "p", "group": "refusal_uncertainty",
                      "prompt_text": prompt, "gold_answer": "not in the context"}, parrot)
    assert s["refusal"] is True and s["refusal_credited"] is False
    assert s["answer_is_echo"] is True

    real = score_sample({"id": "r", "group": "refusal_uncertainty",
                         "prompt_text": prompt, "gold_answer": "not in the context"},
                        "</think>\n\nThat isn't stated in the passage.<|im_end|>")
    assert real["refusal"] and real["refusal_credited"] and not real["answer_is_echo"]

    # Same hole for evidence containment: the gold span is inside the context.
    ev = score_sample({"id": "e", "group": "rag_evidence", "prompt_text": prompt,
                       "gold_answer": "principal clergy of Rome"}, parrot)
    assert ev["evidence_hit"] is True and ev["evidence_hit_credited"] is False

    # And an unclosed think block can never be credited (empty answer).
    empty = score_sample({"id": "n", "group": "rag_evidence", "prompt_text": prompt,
                          "gold_answer": "principal clergy of Rome"}, NEVER_CLOSED)
    assert empty["evidence_hit"] is False and empty["evidence_hit_credited"] is False


def test_aggregate_separates_group_metrics_from_overall():
    scored = [
        score_sample({"id": "a", "group": "rag_evidence", "prompt_text": "q",
                      "gold_answer": "Paris"}, "</think>\n\nParis<|im_end|>"),
        score_sample({"id": "b", "group": "refusal_uncertainty", "prompt_text": "q",
                      "gold_answer": "n/a"}, NEVER_CLOSED),
    ]
    agg = aggregate(scored)
    assert agg["overall"]["n"] == 2
    assert agg["overall"]["format_ok"] == 0.5
    # Group-specific metrics stay inside their group.
    assert "evidence_hit" not in agg["overall"]
    assert agg["by_group"]["rag_evidence"]["evidence_hit"] == 1.0
    assert "refusal" in agg["by_group"]["refusal_uncertainty"]


def test_behavior_score_averages_the_axes_it_can_measure():
    """Axes with no samples are skipped, not counted as zeros."""
    scored = [
        score_sample({"id": "a", "group": "rag_evidence", "prompt_text": "q",
                      "gold_answer": "Paris"}, "</think>\n\nParis<|im_end|>"),
        score_sample({"id": "b", "group": "instruction", "prompt_text": "q"}, GOOD),
    ]
    result = behavior_score(scored)
    assert result["axes"]["grounding"] == 1.0 and result["n"]["grounding"] == 1
    assert result["axes"]["format_ok"] == 1.0
    # No refusal / tool / math prompts here: reported as n=0 and left out of the mean.
    for axis in ("refusal", "tool_call", "math"):
        assert result["axes"][axis] is None and result["n"][axis] == 0
    measured = [v for v in result["axes"].values() if v is not None]
    assert result["score"] == round(sum(measured) / len(measured), 4)


def test_behavior_score_gives_silence_no_fluency_credit():
    """A model that emits nothing must not score well for 'not repeating itself'.

    s1@660 answers nothing on 61% of prompts; a naive 1 - rep_3gram term would
    have ranked it above every later checkpoint.
    """
    silent = [score_sample({"id": str(i), "group": "instruction", "prompt_text": "q"},
                           NEVER_CLOSED) for i in range(4)]
    assert behavior_score(silent)["axes"]["fluency"] == 0.0

    speaking = [score_sample({"id": str(i), "group": "instruction", "prompt_text": "q"},
                             GOOD) for i in range(4)]
    assert behavior_score(speaking)["axes"]["fluency"] == 1.0


def test_behavior_score_needs_samples():
    with pytest.raises(ValueError, match="at least one"):
        behavior_score([])


# --- template-aware think handling ----------------------------------------
# Whether the prompt already opened <think> is a property of the CHAT TEMPLATE.
# Judging a self-opening model under the pre-opened rule rejected 100% of
# otherwise-perfect generations (EXPERIMENTS.md 14.2), so both states are
# supported -- and both stay strict.

def test_template_opens_think_reads_the_prompt():
    from aadistill.evaluation.behavior import template_opens_think
    assert template_opens_think("<|im_start|>assistant\n<think>")
    assert template_opens_think("<|im_start|>assistant\n<think>\n")
    assert not template_opens_think("<|im_start|>assistant\n")


def test_split_generation_preopened_rejects_a_second_open():
    from aadistill.evaluation.behavior import split_generation
    ok = split_generation("thinking</think>Answer.<|im_end|>", think_preopened=True)
    assert ok["think_closed"] and ok["answer"] == "Answer."
    # Re-opening an already-open block is a real violation.
    bad = split_generation("<think>thinking</think>Answer.<|im_end|>",
                           think_preopened=True)
    assert not bad["think_closed"]
    assert "<think>" in bad["stray_markers"]


def test_split_generation_self_opening_accepts_and_still_checks():
    from aadistill.evaluation.behavior import split_generation
    ok = split_generation("<think>thinking</think>Answer.<|im_end|>",
                          think_preopened=False)
    assert ok["think_closed"] and ok["answer"] == "Answer."
    assert ok["think"].strip() == "thinking"
    assert "<think>" not in ok["stray_markers"]
    # Missing open, late open, and doubled open all remain violations.
    for raw in ("thinking</think>Answer.<|im_end|>",
                "preamble<think>t</think>A.<|im_end|>",
                "<think>a<think>b</think>A.<|im_end|>"):
        assert not split_generation(raw, think_preopened=False)["think_closed"], raw


def test_protocol_valid_threads_the_state_and_defaults_to_preopened():
    from aadistill.evaluation.strict_answer import protocol_valid
    selfopen = "<think>t</think>A.<|im_end|>"
    assert protocol_valid(selfopen, think_preopened=False)[0]
    assert not protocol_valid(selfopen)[0]          # default = teacher template
    assert protocol_valid("t</think>A.<|im_end|>")[0]


def test_shared_reads_think_preopened_from_the_record():
    from aadistill.evaluation.capability import _shared
    rec = {"raw": "<think>t</think>A.<|im_end|>", "natural_termination": True}
    assert not _shared(rec)["protocol_valid"]                    # legacy default
    assert _shared({**rec, "think_preopened": False})["protocol_valid"]
