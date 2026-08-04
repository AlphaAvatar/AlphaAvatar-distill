"""Tests for answer-only generation after gold reasoning (D0.2).

The boundary is a template property, so these use the real chat template through
`render_session` rather than hand-built strings -- the point of the mode is that
it never concatenates or re-encodes text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation.oracle_reasoning import (  # noqa: E402
    OracleBoundaryError, build_oracle_prefix, fits, validate_answer_only,
)

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(TEACHER, revision=REVISION)
    except Exception:                      # offline
        pytest.skip("teacher tokenizer unavailable")


def session(reasoning: str, content: str, data_type: str = "gsm8k") -> dict:
    return {
        "id": f"t-{data_type}", "data_type": data_type,
        "messages": [
            {"role": "system", "content": "You are a helpful Assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "reasoning_content": reasoning,
             "content": content},
        ],
        "tools": None,
    }


def test_prefix_is_template_preopened_and_reconstructs(tok):
    p = build_oracle_prefix(tok, session("Two plus two is four.", "The answer is 4."))
    assert p.prefix_ids + p.gold_answer_ids == p.full_ids          # exact rebuild
    assert p.boundary == len(p.prefix_ids)
    close = tok.convert_tokens_to_ids("</think>")
    assert close in p.prefix_ids and close not in p.gold_answer_ids
    assert tok.convert_tokens_to_ids("<|im_end|>") in p.gold_answer_ids
    assert p.n_reasoning_tokens > 0


def test_structural_span_survives_a_literal_think_close_in_the_reasoning(tok):
    """A trace discussing the literal string must not fool the boundary."""
    tricky = "The model should emit </think> when it is done. So: </think> ends it."
    p = build_oracle_prefix(tok, session(tricky, "The answer is 4."))
    assert p.prefix_ids + p.gold_answer_ids == p.full_ids
    # the boundary is the STRUCTURAL close, i.e. the last one in the prefix
    close = tok.convert_tokens_to_ids("</think>")
    assert p.prefix_ids.count(close) >= 1
    assert close not in p.gold_answer_ids
    assert tok.decode(p.gold_answer_ids).lstrip().startswith("The answer is 4.")


def test_empty_answer_content_is_handled(tok):
    p = build_oracle_prefix(tok, session("Some reasoning here.", ""))
    assert p.prefix_ids + p.gold_answer_ids == p.full_ids
    assert tok.convert_tokens_to_ids("<|im_end|>") in p.gold_answer_ids


def test_no_decode_reencode_round_trip(tok):
    """Every id comes from render_session; re-encoding the text may differ."""
    p = build_oracle_prefix(tok, session("Reasoning — with unicode éè.",
                                         "Answer — also unicode."))
    reencoded = tok(tok.decode(p.full_ids), add_special_tokens=False).input_ids
    # The test is that we do NOT depend on this being equal.
    assert p.prefix_ids + p.gold_answer_ids == p.full_ids
    if reencoded != p.full_ids:
        assert True     # exactly the situation the mode must survive


def test_context_limit_rejects_without_truncating(tok):
    p = build_oracle_prefix(tok, session("word " * 400, "The answer is 4."))
    n = len(p.prefix_ids)
    assert fits(p, n + 512, 512)
    assert not fits(p, n + 10, 512)      # rejected, never truncated
    assert len(p.prefix_ids) == n        # unchanged by the check


def test_answer_only_validator_rules():
    ok = validate_answer_only("The answer is 4.<|im_end|>")
    assert ok["protocol_valid"] and ok["reason"] == "ok"
    # it must NOT require another <think>; re-opening one is leakage
    bad = validate_answer_only("<think>more</think>4<|im_end|>")
    assert not bad["protocol_valid"] and bad["reopened_think"]
    leak = validate_answer_only("stray </think> then 4<|im_end|>")
    assert not leak["protocol_valid"] and leak["reasoning_leakage"]
    empty = validate_answer_only("   <|im_end|>")
    assert not empty["protocol_valid"] and empty["empty_answer"]
    unterminated = validate_answer_only("The answer is 4.")
    assert not unterminated["protocol_valid"]
    assert unterminated["reason"] == "not_terminated"


def test_answer_only_natural_termination_is_reported(tok):
    assert validate_answer_only("4<|im_end|>")["terminated"]
    assert not validate_answer_only("4")["terminated"]


# --- KD role accounting: causal-shift alignment ---------------------------

def test_role_labels_align_to_the_predicted_token_after_the_causal_shift():
    """Entry i of mask[:,1:] scores the prediction of input_ids[:,i+1].

    Labels must therefore describe the TARGET token. Off-by-one here would blame
    </think> for the token before it and silently move mass between roles.
    """
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))
    from audit_kd_decomposition import role_labels

    THINK_CLOSE, IM_END = 900, 901
    #        0    1    2         3    4    5        6
    ids = np.array([10, 11, 12, THINK_CLOSE, 20, 21, IM_END, 0])
    ce = np.array([0, 0, 1, 1, 1, 1, 1, 0], dtype=bool)
    content = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    lab = role_labels(ids, ce, content, THINK_CLOSE, IM_END)
    assert list(lab) == ["prompt_context", "prompt_context", "reasoning",
                         "think_close", "answer_content", "answer_content",
                         "im_end", "excluded_padding"]
    # after the shift the first target is index 1, so the aligned view drops [0]
    shifted = lab[1:]
    assert shifted[2] == "think_close"          # target ids[3]
    assert shifted[5] == "im_end"               # target ids[6]
    assert len(shifted) == len(ids) - 1


def test_role_labels_mark_padding_excluded():
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))
    from audit_kd_decomposition import role_labels
    ids = np.array([10, 11, 0, 0])
    ce = np.array([0, 0, 0, 0], dtype=bool)
    content = np.array([1, 1, 0, 0], dtype=bool)
    lab = role_labels(ids, ce, content, 900, 901)
    assert list(lab) == ["prompt_context", "prompt_context",
                         "excluded_padding", "excluded_padding"]


# --- corpus gold extraction ----------------------------------------------

def test_gold_answer_extracts_the_bare_numeric_answer():
    """`gold` is a full worked solution for the numeric tasks, not an answer.

    Comparing a prediction against the whole solution makes normalize_number
    return None and falls through to a containment test that can never match --
    a scorer reporting 0.0 for a reason unrelated to the model.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
    from run_three_mode_diagnostic import gold_answer
    gsm = {"data_type": "gsm8k",
           "gold": "Natalia sold 48/2 = 24 clips in May.\n"
                   "Natalia sold 48+24 = 72 clips altogether.\nThe answer is 72."}
    assert gold_answer(gsm) == "72"
    boxed = {"data_type": "openmath", "gold": "Work.\n\\[ x = 3 \\]\n\\boxed{-4}"}
    assert gold_answer(boxed) == "-4"
    # non-numeric tasks keep their gold verbatim
    assert gold_answer({"data_type": "rag_evidence",
                        "gold": "in the late 1990s"}) == "in the late 1990s"
    assert gold_answer({"data_type": "gsm8k", "gold": None}) is None
