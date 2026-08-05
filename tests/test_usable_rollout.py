"""Validate the Stage 2/3 primary metric before any ranking is read from it.

Every scorer in this project is checked against deliberately-bad policies on
realistic input before it is trusted. `usable_rollout` gets the same treatment,
including the check that documents its known blind spot: a terse, contentless but
well-formed reply scores a perfect usable rollout.
"""

from __future__ import annotations

import pytest

from aadistill.evaluation import usable_rollout as ur

GOOD_RAW = "reasoning here\n</think>\nThe answer is 42.<|im_end|>"
NO_CLOSE = "reasoning that never closes the block"
REOPENED = "reasoning <think> again\n</think>\nanswer<|im_end|>"


def three_mode(**over):
    rec = {"empty_answer": False, "natural_termination": True,
           "degenerate": False, "context_limit": False, "protocol_valid": True}
    rec.update(over)
    return rec


def behavior_v0(**over):
    rec = {"raw": GOOD_RAW, "natural_termination": True,
           "degeneration_triggered": False, "context_limit_reached": False}
    rec.update(over)
    return rec


# --- schema handling -------------------------------------------------------

def test_detects_both_schemas():
    assert ur.detect_schema(three_mode()) == "three_mode"
    assert ur.detect_schema(behavior_v0()) == "behavior_v0"


def test_unknown_schema_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="unrecognised generation record"):
        ur.detect_schema({"raw": "x", "correct": True})


# --- the conjunction -------------------------------------------------------

def test_all_components_true_is_usable():
    assert ur.usable(three_mode()) is True
    assert ur.usable(behavior_v0()) is True


@pytest.mark.parametrize("field", ["empty_answer", "natural_termination",
                                   "degenerate", "context_limit",
                                   "protocol_valid"])
def test_each_component_alone_makes_it_unusable(field):
    """No component is decorative: flipping any one of the five must fail."""
    bad = {"empty_answer": True, "natural_termination": False,
           "degenerate": True, "context_limit": True,
           "protocol_valid": False}[field]
    assert ur.usable(three_mode(**{field: bad})) is False


@pytest.mark.parametrize("field", ["natural_termination",
                                   "degeneration_triggered",
                                   "context_limit_reached"])
def test_behavior_v0_components_are_load_bearing(field):
    bad = {"natural_termination": False, "degeneration_triggered": True,
           "context_limit_reached": True}[field]
    assert ur.usable(behavior_v0(**{field: bad})) is False


# --- the recomputed components, which is where a rescore can go wrong ------

def test_behavior_v0_recomputes_protocol_validity_from_raw():
    assert ur.components(behavior_v0(raw=NO_CLOSE))["protocol_valid"] is False
    assert ur.components(behavior_v0(raw=REOPENED))["protocol_valid"] is False
    assert ur.components(behavior_v0(raw=GOOD_RAW))["protocol_valid"] is True


def test_behavior_v0_empty_answer_is_detected_from_raw():
    empty = "reasoning only\n</think>\n   \n<|im_end|>"
    assert ur.components(behavior_v0(raw=empty))["non_empty"] is False
    assert ur.usable(behavior_v0(raw=empty)) is False


def test_think_preopened_false_flips_the_verdict_for_self_opening_models():
    """The template-bound defect that once rejected 100% of Qwen3-0.6B output."""
    self_open = "<think>reasoning</think>\nanswer<|im_end|>"
    assert ur.components(behavior_v0(raw=self_open),
                         think_preopened=True)["protocol_valid"] is False
    assert ur.components(behavior_v0(raw=self_open),
                         think_preopened=False)["protocol_valid"] is True


# --- known-bad policies ----------------------------------------------------

def test_terse_contentless_reply_scores_a_perfect_usable_rollout():
    """Documented blind spot, asserted so it cannot be forgotten.

    `usable_rollout` measures trajectory well-formedness, not usefulness. A
    two-token answer is a perfect usable rollout. This is why correctness is a
    separate axis and why `correct_given_usable` exists.
    """
    terse = behavior_v0(raw="k\n</think>\n42<|im_end|>")
    assert ur.usable(terse) is True


def test_degenerate_but_protocol_valid_still_fails():
    """A repetition loop that happens to close its delimiters is not usable."""
    assert ur.usable(three_mode(degenerate=True)) is False


# --- summarize -------------------------------------------------------------

def test_summarize_rates_and_census():
    recs = [three_mode(), three_mode(),
            three_mode(empty_answer=True),
            three_mode(natural_termination=False)]
    s = ur.summarize(recs)
    assert s["n"] == 4
    assert s["usable_rollout_rate"] == 0.5
    assert s["non_empty"] == 0.75
    assert s["natural_termination"] == 0.75
    assert s["no_severe_repetition"] == 1.0
    assert sum(s["first_failure"].values()) == 2


def test_first_failure_census_sums_to_unusable_count():
    recs = [three_mode(empty_answer=True, natural_termination=False,
                       degenerate=True) for _ in range(3)] + [three_mode()]
    s = ur.summarize(recs)
    assert sum(s["first_failure"].values()) == 3
    assert s["first_failure"] == {"non_empty": 3}   # attributed to the first only


def test_empty_input_does_not_crash():
    assert ur.summarize([]) == {"n": 0}
