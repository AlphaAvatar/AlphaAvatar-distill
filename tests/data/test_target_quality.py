"""The teacher-target quality contract: every category kept apart, nothing by length."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.target_quality import (  # noqa: E402
    AcceptancePolicy, Check, assess, select_target, summarize,
)

ENDS = {"natural": ("eos",), "length": ("length",), "degeneration": ("degeneration",)}
STRICT = AcceptancePolicy()


def rec(i=0, *, stop="eos", proto=None, ans="correct", steps=None, obs=None, policy=STRICT,
        unit="u", tokens=None, scope=""):
    return assess(unit, i, stop_reason=stop, protocol_reason=proto, answer=Check(ans, scope=scope),
                  steps=steps, observations=obs, policy=policy, end_reasons=ENDS,
                  generated_tokens=tokens)


def test_a_clean_correct_candidate_is_accepted_with_steps_not_assessed():
    r = rec()
    assert r.accepted_for_training and r.steps == "not_assessed" and r.reasons == []


@pytest.mark.parametrize("kw,reason", [
    ({"stop": "length"}, "ending:length"),
    ({"stop": "degeneration"}, "ending:degeneration"),
    ({"proto": "think_not_closed"}, "protocol:think_not_closed"),
    ({"ans": "incorrect"}, "answer:incorrect"),
    ({"ans": "unverifiable"}, "answer:unverifiable"),
    ({"steps": Check("contradicted", "step 3 misreads a premise")},
     "steps:contradicted:step 3 misreads a premise"),
])
def test_each_failure_is_named_and_rejects(kw, reason):
    r = rec(**kw)
    assert not r.accepted_for_training and reason in r.reasons


def test_all_failures_are_recorded_not_only_the_first():
    r = rec(stop="length", proto="empty_answer", ans="incorrect")
    assert len(r.reasons) == 3
    assert r.length_limited and not r.naturally_terminated and not r.protocol_valid


@pytest.mark.parametrize("state", ["verified", "partial", "not_assessed"])
def test_a_step_state_short_of_contradiction_does_not_reject(state):
    assert rec(steps=Check(state)).accepted_for_training


def test_an_unknown_step_state_is_refused():
    with pytest.raises(ValueError):
        rec(steps=Check("valid"))
    with pytest.raises(ValueError):
        AcceptancePolicy(require_steps=("checked",))


def test_a_group_can_require_its_step_check():
    gated = AcceptancePolicy(require_steps=("partial", "verified"))
    assert rec(steps=Check("partial", scope="every reference conclusion present"), policy=gated).accepted_for_training
    r = rec(policy=gated)
    assert not r.accepted_for_training and r.reasons == ["steps:not_assessed"]


def test_observations_are_recorded_apart_and_reject_only_where_declared():
    obs = {"text_consistency": Check("final_not_restated"), "action_legality": Check("illegal_action", "3 illegal")}
    plain = rec(obs=obs)
    assert plain.accepted_for_training and plain.steps == "not_assessed"
    assert plain.observations == {"action_legality": "illegal_action", "text_consistency": "final_not_restated"}
    legal = AcceptancePolicy(reject_observations={"action_legality": ("illegal_action",)})
    r = rec(obs=obs, policy=legal)
    assert r.reasons == ["action_legality:illegal_action:3 illegal"]
    assert rec(policy=legal).reasons == ["action_legality:missing"]


def test_a_legal_action_sequence_is_not_a_verified_process():
    """Environment replay checks the ACTIONS; the reasoning that chose them is
    not thereby verified, and the record keeps the two apart."""
    r = rec(obs={"action_legality": Check("all_legal")})
    assert r.observations["action_legality"] == "all_legal" and r.steps == "not_assessed"


def test_the_answer_records_what_its_check_covered():
    r = rec(scope="the declared tests passed: not full semantic correctness")
    assert r.answer == "correct" and "not full semantic correctness" in r.answer_scope
    assert summarize([r])["all"]["answer_scope"] == {r.answer_scope: 1}


def test_unverifiable_needs_an_explicit_switch():
    with pytest.raises(ValueError):
        AcceptancePolicy(require_answer=("correct", "unverifiable"))
    open_ = AcceptancePolicy(require_answer=("correct", "unverifiable"), admit_unverifiable=True)
    assert rec(ans="unverifiable", policy=open_).accepted_for_training


def test_unmapped_stop_reason_is_an_error():
    with pytest.raises(ValueError, match="not mapped"):
        rec(stop="mystery")


def test_selection_is_lowest_index_never_length():
    rows = [rec(2, tokens=100), rec(1, tokens=9000), rec(0, ans="incorrect", tokens=500)]
    assert select_target(rows).candidate_index == 1
    assert select_target([rec(0, ans="incorrect")]) is None


def test_summary_counts_units_and_candidates_separately():
    rows = [rec(0, unit="a"), rec(1, unit="a", ans="incorrect"), rec(0, unit="b", stop="length")]
    s = summarize(rows)["all"]
    assert s["candidates"] == 3 and s["units"] == 2
    assert s["accepted_candidates"] == 1 and s["accepted_units"] == 1
    assert s["rejection_reasons"] == {"answer:incorrect": 1, "ending:length": 1}
