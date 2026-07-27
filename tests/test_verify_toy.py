"""Verifier tests — synthetic candidates, no model.

These rules decide what enters the training corpus, so the cases that matter are
the ones where a *wrong* answer could slip through: containment satisfied by an
answer that says the opposite, a math answer that matches the gold's digits but
not its final answer, and an unterminated generation that happens to contain the
gold span.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aadistill.verify import boxed_answer, normalize_math, select, verify

TERMINATED = "<|im_end|>"


def sample(group: str, source: str, gold: str) -> dict:
    return {"id": "x", "group": group, "source": source,
            "messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": gold}]}


def check(group, source, gold, answer, raw=None):
    return verify(sample(group, source, gold), answer, raw or answer + TERMINATED)


def test_span_slices_require_the_gold_span():
    assert check("rag_evidence", "squad_v2", "in the late 1990s",
                 "It happened in the late 1990s.") == (True, "ok")
    assert check("rag_evidence", "squad_v2", "in the late 1990s",
                 "It happened in the 1970s.") == (False, "gold_span_missing")


def test_yes_no_golds_must_lead_the_answer():
    """Containment alone passes for an answer that says the opposite."""
    assert check("multihop_qa", "hotpot_qa", "yes",
                 "No, they are not — the yes/no question is about x.")[0] is False
    assert check("multihop_qa", "hotpot_qa", "yes",
                 "Yes, both films were directed by the same person.") == (True, "ok")


def test_refusal_slice_requires_a_short_refusal():
    assert check("refusal_uncertainty", "squad_v2", "n/a",
                 "The context does not contain this, so I can't answer.") == (True, "ok")
    assert check("refusal_uncertainty", "squad_v2", "n/a",
                 "The answer is 42.") == (False, "not_a_refusal")
    rambling = "I cannot answer this. " + "padding words here " * 40
    assert check("refusal_uncertainty", "squad_v2", "n/a", rambling) == (False, "refusal_too_long")


def test_gsm8k_matches_on_the_final_number_only():
    gold = "He had 5 apples and bought 7 more. The answer is 12."
    assert check("code_math", "gsm8k", gold,
                 "5 + 7 = 12. The answer is 12.") == (True, "ok")
    # Contains the gold's intermediate numbers but reaches the wrong result.
    assert check("code_math", "gsm8k", gold,
                 "5 + 7 = 13. The answer is 13.") == (False, "answer_mismatch")
    assert check("code_math", "gsm8k", gold, "Twelve apples.") == (False, "no_final_number")


def test_openmath_matches_the_boxed_answer_through_latex_noise():
    gold = r"So \[ y = \boxed{2} \]"
    assert check("code_math", "openmath_instruct_2", gold,
                 r"Dividing gives $\boxed{ 2 }$.") == (True, "ok")
    assert check("code_math", "openmath_instruct_2", gold,
                 r"Dividing gives $\boxed{3}$.") == (False, "answer_mismatch")
    assert check("code_math", "openmath_instruct_2", gold,
                 "Dividing gives 2.") == (False, "no_boxed")


def test_hygiene_rejects_before_any_slice_rule():
    gold = "in the late 1990s"
    # An unterminated generation is rejected even though it contains the span.
    assert check("rag_evidence", "squad_v2", gold,
                 "in the late 1990s", raw="in the late 1990s") == (False, "not_terminated")
    assert check("rag_evidence", "squad_v2", gold, "") == (False, "empty")
    assert check("rag_evidence", "squad_v2", gold,
                 "in the late 1990s <|im_start|>") == (False, "stray_marker")


def test_unverifiable_slices_are_refused_not_guessed():
    assert check("code_math", "magicoder_oss", "def f(): pass",
                 "def f(): pass") == (False, "unverifiable_slice")
    assert check("tool_calling", "xlam_fc_60k", "{}", "{}") == (False, "unverifiable_slice")


def test_boxed_answer_handles_nested_braces_and_gives_up_on_unbalanced():
    assert boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert boxed_answer(r"\boxed{1} then \boxed{2}") == "2"  # last wins
    assert boxed_answer(r"\boxed{1") is None
    assert boxed_answer("no box here") is None


def test_normalize_math_ignores_spacing_and_latex_noise():
    assert normalize_math(r"\left( 1,\; 2 \right)") == normalize_math("(1,2)")
    assert normalize_math(r"\dfrac{1}{2}") == normalize_math(r"\frac{1}{2}")


def candidate(index, accepted, words):
    return {"index": index, "accepted": accepted, "answer": " ".join(["w"] * words)}


def test_select_prefers_a_verified_greedy_candidate():
    chosen = select([candidate(0, True, 50), candidate(1, True, 10)])
    assert chosen["index"] == 0


def test_select_falls_back_to_the_median_length_accepted_candidate():
    """Not the shortest: on math that picks the answer with no derivation."""
    chosen = select([candidate(0, False, 5), candidate(1, True, 8),
                     candidate(2, True, 200), candidate(3, True, 60)])
    assert chosen["index"] == 3  # 8 < 60 < 200 -> median is 60 words


def test_select_returns_none_when_nothing_verified():
    assert select([candidate(0, False, 5), candidate(1, False, 9)]) is None


def test_verify_is_exhaustive_over_its_slice_table():
    from aadistill.verify import VERIFIABLE

    for (group, source) in VERIFIABLE:
        accepted, reason = check(group, source, "gold", "gold")
        assert reason != "unverifiable_slice", (group, source)


@pytest.mark.parametrize("gold,answer,expected", [
    ("12", "The answer is 12.", True),
    ("1,200", "The answer is 1200.", True),
])
def test_final_number_normalization_carries_into_verification(gold, answer, expected):
    got = check("code_math", "gsm8k", f"The answer is {gold}.", answer)
    assert got[0] is expected
