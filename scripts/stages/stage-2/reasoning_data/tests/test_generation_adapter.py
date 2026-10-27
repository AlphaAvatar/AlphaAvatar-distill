"""The verified record is the trained record (`generation_adapter`).

A candidate's identity is explicit and unique, the record is the trajectory it
claims, and the selected verdict and its record travel together into the
sessions and the pack.
"""
from __future__ import annotations

import pytest

from stages.reasoning_data import build_master_ladder as B
from stages.reasoning_data import generation_adapter as G
from stages.reasoning_data.tests.test_master_pack import _render_stub, _Tok

from aadistill.evaluation.behavior import split_generation

TASK = {"task_id": "t1", "source": "reasoning_gym", "module": "m", "family_id": "f", "stratum": "primary.x",
        "answer_type": "text", "answer": "cat", "interactive": False,
        "messages": [{"role": "user", "content": "Which animal? Box it."}]}


def _record(answer: str, index=0, **over) -> dict:
    raw = f"Let me think about {answer}.\n</think>\n\nIt is \\boxed{{{answer}}}.<|im_end|>"
    parts = split_generation(raw, think_preopened=True)
    turn = {"turn": 0, "raw": raw, "answer": parts["answer"], "think_preopened": True, "stop_reason": "eos",
            "generated_tokens": 12, "output_token_ids": [len(answer), 7, 9]}
    rec = {"task_id": "t1", "turns": [turn],
           "messages": [{"role": "system", "content": "You are a helpful Assistant."}, *TASK["messages"],
                        {"role": "assistant", "content": parts["answer"], "reasoning_content": parts["think"]}]}
    if index is not None:
        rec["candidate_index"] = index
    rec.update(over)
    return rec


def _adapt(records, tokenizer=None):
    return G.adapt({"t1": TASK}, records, lambda t: t["stratum"], tokenizer=tokenizer)


def test_a_record_without_a_candidate_index_is_refused_not_defaulted():
    with pytest.raises(G.RecordError, match="candidate_index None"):
        _adapt([_record("cat", index=None)])
    with pytest.raises(G.RecordError, match="not an explicit"):
        _adapt([_record("cat", index=True)])


def test_a_repeated_candidate_index_refuses_the_input():
    # a correct and a wrong record under one index: nothing may be assessed, so
    # the wrong one can never ride on the correct one's verdict
    with pytest.raises(G.RecordError, match="appears more than once"):
        _adapt([_record("cat", 0), _record("dog", 0)])


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_the_selected_verdict_and_its_own_record_become_the_sessions(order):
    recs = {0: _record("dog", 0), 1: _record("cat", 1)}
    got = _adapt([recs[i] for i in order])
    (s,) = got["sessions"]
    assert s["target"]["candidate_index"] == 1
    assert s["target"]["record_sha256"] == G.record_sha256(recs[1])
    assert "\\boxed{cat}" in s["messages"][-1]["content"]
    assert got["selected"]["t1"] == s["target"]


def test_messages_that_are_not_the_raw_output_are_refused():
    rec = _record("cat")
    rec["messages"][-1]["content"] = "It is \\boxed{cat}. (edited)"
    with pytest.raises(G.RecordError, match="assistant content is not the raw output's answer"):
        _adapt([rec])
    rec = _record("cat")
    rec["messages"][-1]["reasoning_content"] = "a different chain"
    with pytest.raises(G.RecordError, match="reasoning is not the raw output's reasoning"):
        _adapt([rec])
    rec = _record("cat")
    rec["messages"][1] = {"role": "user", "content": "Another task"}
    with pytest.raises(G.RecordError, match="prompt messages differ"):
        _adapt([rec])


def test_output_token_ids_must_decode_to_the_raw_text():
    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "something else"
    with pytest.raises(G.RecordError, match="do not decode to the raw text"):
        _adapt([_record("cat")], tokenizer=Tok())


def test_an_observation_that_is_not_the_next_user_turn_is_refused():
    task = {**TASK, "task_id": "e1"}
    rec = _record("cat", task_id="e1")
    t0 = dict(rec["turns"][0], observation="f(3) = 5")
    rec["turns"] = [t0, dict(t0, turn=1)]
    asst = rec["messages"][-1]
    rec["messages"] = rec["messages"] + [{"role": "user", "content": "f(3) = 6"}, dict(asst)]
    bad = G.consistency(task, rec)
    assert bad == ["turn 0: the next user turn is not this turn's observation"]
    rec["messages"][-2]["content"] = "f(3) = 5"
    assert G.consistency(task, rec) == []


def test_sessions_of_one_unit_naming_different_targets_are_excluded_whole(monkeypatch):
    monkeypatch.setattr(B, "render_session", _render_stub)
    rows = [{"id": f"u#t{k}", "unit_id": "u", "unit_sessions": 2, "stratum": "s", "family": "f",
             "attributes": {}, "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
             "data_type": "s", "target": {"record_sha256": f"r{k}"}} for k in range(2)]
    rendered, excluded, _ = B.render_all(_Tok(), rows, block_len=512)
    assert not rendered and excluded[0]["kind"] == "inconsistent_target"


def test_a_boxed_text_wrapper_is_the_answer_it_wraps():
    assert G.COMPARE["bool"]("\\text{False}", "False") and not G.COMPARE["bool"]("\\text{True}", "False")
    assert G.COMPARE["text_ci"]("\\text{Alice}", "alice") and not G.COMPARE["text_ci"]("\\text{Bob}", "alice")
    assert G.COMPARE["text"]("\\text{abc}", "abc") and not G.COMPARE["text"]("\\text{ABC}", "abc")
