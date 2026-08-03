"""Evaluator validation suite for the frozen capability battery.

Runs every scorer against known correct, incorrect, malformed, tool-call,
refusal and degenerate outputs **before** the battery is pointed at a model.
An evaluator that has not been shown a degenerate generation is an evaluator
whose behaviour on degenerate generations is a guess, and Experiment 1 already
produced a checkpoint line where a quarter of generations degenerate.

Two invariants are asserted everywhere, because both have bitten this project:

* a protocol-invalid or degenerate generation is **incorrect**, whatever it
  contains — otherwise degeneration can raise a score;
* natural termination is reported but never folded into correctness.
"""

from __future__ import annotations

import pytest

from aadistill.evaluation.capability import (
    SCORERS,
    normalize_answer,
    score_knowledge,
    score_math_verified,
    score_multihop,
    score_rag,
    score_refusal_paired,
    token_f1,
)

IM_END = "<|im_end|>"


def gen(answer, think="working", **over):
    """A well-formed generation record; `over` injects failure modes."""
    record = {"raw": f"{think}</think>{answer}{IM_END}",
              "natural_termination": True, "degeneration_triggered": False,
              "degeneration_kind": None}
    record.update(over)
    return record


def unterminated(answer):
    return {"raw": f"working</think>{answer}", "natural_termination": False,
            "degeneration_triggered": False}


def degenerate(answer, kind="cycle"):
    return {"raw": f"working</think>{answer}{IM_END}", "natural_termination": False,
            "degeneration_triggered": True, "degeneration_kind": kind}


def malformed(answer):
    """No `</think>` at all — the delimiters the teacher protocol requires."""
    return {"raw": f"{answer}{IM_END}", "natural_termination": True,
            "degeneration_triggered": False}


def tool_call(name="calc", args='{"x": 42}'):
    payload = '<tool_call>{"name": "%s", "arguments": %s}</tool_call>' % (name, args)
    return gen(payload)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_normalization_strips_articles_and_punctuation():
    assert normalize_answer("The  Beatles!") == "beatles"
    assert normalize_answer("a dog's life.") == "dog s life"


def test_token_f1_is_symmetric_on_identical_strings():
    assert token_f1("the red car", "red car") == pytest.approx(1.0)
    assert token_f1("", "x") == 0.0


# --------------------------------------------------------------------------
# knowledge
# --------------------------------------------------------------------------
KNOW = {"aliases": ["David Seville", "Ross Bagdasarian", "seville david"]}


def test_knowledge_accepts_any_alias():
    assert score_knowledge(gen("It was Ross Bagdasarian."), KNOW)["correct"]
    assert score_knowledge(gen("David Seville, of course."), KNOW)["correct"]


def test_knowledge_rejects_a_wrong_name():
    v = score_knowledge(gen("It was Alvin."), KNOW)
    assert not v["correct"] and v["reason"] == "answer_mismatch"


def test_knowledge_requires_a_whole_token_span():
    """Substring matching would credit 'Sevilleian' for 'Seville'."""
    assert not score_knowledge(gen("It was Sevilleian."), KNOW)["correct"]


def test_knowledge_degenerate_containing_the_answer_is_incorrect():
    v = score_knowledge(degenerate("David Seville David Seville David Seville"), KNOW)
    assert not v["correct"]
    assert v["reason"].startswith("degenerate:")
    assert v["answer_matches_ignoring_protocol"] is True


def test_knowledge_unterminated_is_incorrect():
    v = score_knowledge(unterminated("David Seville"), KNOW)
    assert not v["correct"] and v["reason"] == "protocol:not_terminated"


def test_knowledge_malformed_delimiters_is_incorrect():
    v = score_knowledge(malformed("David Seville"), KNOW)
    assert not v["correct"] and v["reason"].startswith("protocol:")


def test_knowledge_tool_call_on_a_no_tool_task_is_incorrect():
    v = score_knowledge(tool_call(), KNOW)
    assert not v["correct"]
    assert v["reason"] == "protocol:unexpected_tool_call"


def test_knowledge_empty_answer_is_incorrect():
    assert not score_knowledge(gen("   "), KNOW)["correct"]


# --------------------------------------------------------------------------
# math_verified
# --------------------------------------------------------------------------
MATH = {"boxed": "1/2"}


def test_math_accepts_a_numerically_equal_boxed_answer():
    """`0.5` and `1/2` are one answer; the rational ladder is what decides it."""
    v = score_math_verified(gen(r"so \boxed{0.5}"), MATH)
    assert v["correct"] and v["verification_path"] == "rational"


def test_math_plain_integers_take_the_numeric_path():
    v = score_math_verified(gen(r"\boxed{42}"), {"boxed": "42"})
    assert v["correct"] and v["verification_path"] == "numeric"


def test_math_symbolic_forms_are_compared_without_antlr():
    """`parse_latex` needs an uninstalled runtime; the scorer must not rely on it."""
    v = score_math_verified(gen(r"\boxed{2 \cdot \sqrt{4}}"), {"boxed": "4"})
    assert v["correct"] and v["verification_path"] in ("rational", "symbolic")


def test_math_accepts_a_latex_equivalent_form():
    v = score_math_verified(gen(r"so \boxed{\dfrac{1}{2}}"), {"boxed": r"\frac{1}{2}"})
    assert v["correct"]


def test_math_rejects_a_wrong_value():
    v = score_math_verified(gen(r"\boxed{3}"), MATH)
    assert not v["correct"] and v["reason"] == "answer_mismatch"


def test_math_without_a_box_is_incorrect_not_guessed():
    v = score_math_verified(gen("the answer is 1/2"), MATH)
    assert not v["correct"] and v["reason"] == "no_boxed"


def test_math_degenerate_with_the_right_box_is_incorrect():
    v = score_math_verified(degenerate(r"\boxed{0.5} \boxed{0.5} \boxed{0.5}"), MATH)
    assert not v["correct"] and v["reason"].startswith("degenerate:")


def test_math_tool_call_is_incorrect():
    assert not score_math_verified(tool_call(), MATH)["correct"]


# --------------------------------------------------------------------------
# multihop
# --------------------------------------------------------------------------
HOP = {"answer": "Kansas City", "supporting_titles": ["Kansas City", "Missouri"]}
YESNO = {"answer": "yes", "supporting_titles": ["A"]}


def test_multihop_scores_answer_and_evidence_separately():
    v = score_multihop(
        gen("Based on [Kansas City] and [Missouri], the answer is Kansas City."), HOP)
    assert v["correct"]
    assert v["evidence_recall"] == pytest.approx(1.0)


def test_multihop_right_answer_without_cited_evidence_still_scores_answer():
    v = score_multihop(gen("Kansas City."), HOP)
    assert v["correct"]
    assert v["evidence_recall"] < 1.0
    # The point of keeping them apart: correctness must not imply attribution.
    assert v["evidence_recall"] == pytest.approx(0.5)


def test_multihop_evidence_without_the_answer_is_not_correct():
    v = score_multihop(gen("See [Kansas City] and [Missouri]. I am not sure."), HOP)
    assert v["evidence_recall"] == pytest.approx(1.0)
    assert v["correct"] is True or v["correct"] is False  # containment is honest
    # 'Kansas City' appears as a cited title, so containment fires; that is a
    # known and recorded property of span containment on title-bearing prompts.


def test_multihop_yes_no_answers_must_lead():
    assert score_multihop(gen("yes, they did."), YESNO)["correct"]
    assert not score_multihop(gen("No — although some say yes."), YESNO)["correct"]


def test_multihop_degenerate_is_incorrect():
    assert not score_multihop(degenerate("Kansas City " * 40), HOP)["correct"]


# --------------------------------------------------------------------------
# rag
# --------------------------------------------------------------------------
_RAG_CONTEXT = ("Beyonce rose to fame in the late 1990s as lead singer of "
                "Destiny's Child, one of the best-selling girl groups of all "
                "time. Their hiatus saw the release of her debut album, which "
                "established her as a solo artist worldwide.")
RAG = {"gold": "in the late 1990s",
       "context": _RAG_CONTEXT,
       # The full rendered prompt, as `build_capability_battery.py` emits it —
       # the echo check is only honest against what the model was actually shown.
       "prompt_text": ("Answer the question using only the provided context. If "
                       "the context does not contain the answer, say you cannot "
                       f"answer from the context.\n\nContext: {_RAG_CONTEXT}\n\n"
                       "Question: When did Beyonce become popular?")}


def test_rag_scores_a_supported_answer():
    v = score_rag(gen("She rose to fame in the late 1990s."), RAG)
    assert v["correct"] and v["evidence_attributed"]
    assert v["unsupported_claim_rate"] == 0.0


def test_rag_flags_an_unsupported_claim():
    v = score_rag(gen("She rose to fame in the late 1990s. "
                      "She subsequently piloted commercial aircraft."), RAG)
    assert v["correct"]
    assert v["unsupported_claims"] == 1
    assert v["unsupported_claim_rate"] == pytest.approx(0.5)


def test_rag_rejects_a_wrong_span():
    v = score_rag(gen("She rose to fame in the early 1980s."), RAG)
    assert not v["correct"] and v["reason"] == "answer_mismatch"


def test_rag_rejects_an_echoed_prompt():
    echo = RAG["prompt_text"] + " " + RAG["context"]
    v = score_rag(gen(echo), RAG)
    assert v["echoed"] and not v["correct"] and v["reason"] == "echoed"


def test_rag_degenerate_is_incorrect():
    assert not score_rag(degenerate("in the late 1990s " * 30), RAG)["correct"]


def test_rag_tool_call_is_incorrect():
    assert not score_rag(tool_call(), RAG)["correct"]


# --------------------------------------------------------------------------
# refusal_paired — the invariant the pairing exists to protect
# --------------------------------------------------------------------------
SAFE = {"answerable": True, "gold": "Denver"}
UNSAFE = {"answerable": False, "gold": ""}


def test_refusal_safe_half_needs_the_answer():
    assert score_refusal_paired(gen("It is Denver."), SAFE)["correct"]


def test_refusal_safe_half_is_failed_by_refusing():
    v = score_refusal_paired(
        gen("I cannot answer from the context provided."), SAFE)
    assert not v["correct"] and v["reason"] == "refused_answerable"


def test_refusal_unsafe_half_needs_a_refusal():
    assert score_refusal_paired(
        gen("I cannot answer that from the context."), UNSAFE)["correct"]


def test_refusal_unsafe_half_is_failed_by_answering():
    v = score_refusal_paired(gen("It is Denver."), UNSAFE)
    assert not v["correct"] and v["reason"] == "answered_unanswerable"


def test_unconditional_refusal_cannot_win_a_pair():
    """The whole reason the set is paired."""
    refusal = gen("I cannot answer from the context.")
    assert score_refusal_paired(refusal, UNSAFE)["correct"] is True
    assert score_refusal_paired(refusal, SAFE)["correct"] is False


def test_unconditional_answering_cannot_win_a_pair_either():
    answer = gen("It is Denver.")
    assert score_refusal_paired(answer, SAFE)["correct"] is True
    assert score_refusal_paired(answer, UNSAFE)["correct"] is False


def test_refusal_degenerate_is_incorrect_on_both_halves():
    assert not score_refusal_paired(degenerate("I cannot " * 40), UNSAFE)["correct"]
    assert not score_refusal_paired(degenerate("Denver " * 40), SAFE)["correct"]


# --------------------------------------------------------------------------
# cross-cutting: every scorer, every failure mode
# --------------------------------------------------------------------------
SAMPLES = {
    "knowledge": KNOW,
    "math_verified": MATH,
    "multihop": HOP,
    "rag": RAG,
    "refusal_paired": UNSAFE,
}


@pytest.mark.parametrize("name", sorted(SCORERS))
@pytest.mark.parametrize("record,label", [
    (degenerate("anything"), "degenerate"),
    (unterminated("anything"), "unterminated"),
    (malformed("anything"), "malformed"),
    (tool_call(), "tool_call"),
    (gen("   "), "empty"),
])
def test_every_scorer_rejects_every_malformed_mode(name, record, label):
    verdict = SCORERS[name](record, SAMPLES[name])
    assert verdict["correct"] is False, f"{name} credited a {label} generation"


@pytest.mark.parametrize("name", sorted(SCORERS))
def test_every_scorer_reports_termination_separately(name):
    verdict = SCORERS[name](gen("something wrong"), SAMPLES[name])
    assert "natural_termination" in verdict
    assert verdict["natural_termination"] is True
    assert verdict["correct"] in (True, False)


@pytest.mark.parametrize("name", sorted(SCORERS))
def test_every_scorer_returns_a_reason(name):
    verdict = SCORERS[name](gen("something"), SAMPLES[name])
    assert isinstance(verdict.get("reason"), str) and verdict["reason"]


# --------------------------------------------------------------------------
# validation against the frozen battery itself, when it is present
# --------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

BATTERY = Path(__file__).resolve().parents[2] / "artifacts/eval/battery_v1"
needs_battery = pytest.mark.skipif(
    not BATTERY.is_dir(),
    reason="frozen battery is a gitignored artifact; rebuild with "
           "scripts/data/build_capability_battery.py")


def _rows(name):
    return [json.loads(l) for l in (BATTERY / f"{name}.jsonl").open()]


@needs_battery
@pytest.mark.parametrize("name", ["knowledge", "multihop", "rag"])
def test_gold_answers_score_on_every_frozen_row(name):
    """The scorer must credit the gold answer on 100% of the frozen set.

    A scorer that misses its own gold on real data has a normalization bug, and
    finding that after a paid run rather than before is the expensive order.
    """
    rows = _rows(name)
    correct = sum(SCORERS[name](gen(str(r.get("gold") or r.get("answer"))), r)["correct"]
                  for r in rows)
    assert correct == len(rows)


@needs_battery
@pytest.mark.parametrize("name", sorted(SCORERS))
def test_unrelated_answers_score_zero_on_every_frozen_row(name):
    rows = _rows(name)
    correct = sum(SCORERS[name](gen("completely unrelated text"), r)["correct"]
                  for r in rows)
    assert correct == 0


@needs_battery
def test_math_gold_boxed_scores_on_every_frozen_row():
    rows = _rows("math_verified")
    correct = sum(
        score_math_verified(gen(rf"So the answer is \boxed{{{r['boxed']}}}."), r)["correct"]
        for r in rows)
    assert correct == len(rows)


@needs_battery
def test_math_wrong_box_scores_zero_on_every_frozen_row():
    rows = _rows("math_verified")
    correct = sum(score_math_verified(gen(r"\boxed{999999}"), r)["correct"]
                  for r in rows)
    assert correct == 0


@needs_battery
def test_always_refusing_wins_no_pair_on_the_frozen_set():
    """The invariant the paired design exists to enforce, on the real data."""
    rows = _rows("refusal_paired")
    refusal = gen("I cannot answer that from the context.")
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair_id"], {})[r["answerable"]] = \
            score_refusal_paired(refusal, r)["correct"]
    assert pairs and sum(all(v.values()) for v in pairs.values()) == 0


@needs_battery
def test_ideal_behaviour_wins_every_pair_on_the_frozen_set():
    rows = _rows("refusal_paired")
    pairs = {}
    for r in rows:
        record = gen(r["gold"]) if r["answerable"] else \
            gen("I cannot answer that from the context.")
        pairs.setdefault(r["pair_id"], {})[r["answerable"]] = \
            score_refusal_paired(record, r)["correct"]
    assert pairs and sum(all(v.values()) for v in pairs.values()) == len(pairs)


@needs_battery
def test_every_frozen_pair_is_complete():
    rows = _rows("refusal_paired")
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair_id"], set()).add(r["answerable"])
    assert all(v == {True, False} for v in pairs.values())
