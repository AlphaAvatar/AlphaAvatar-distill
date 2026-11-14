"""The teacher-reuse prompt check: a missing field is never a match, and equal lengths are never content identity.

The first item-level version compared ``(prompt_tokens, system_source, think_preopened)`` tuples, so two records
that both lacked the fields compared ``(None, None, None) == (None, None, None)`` and passed. These pin the repair:
a field absent or invalid on either side is counted ``not recorded`` and the check does not pass; ``False`` is a
valid ``think_preopened``; and equal prompt fields alone never establish that two prompts have the same content.
"""
from __future__ import annotations

from reasoning_inheritance import teacher_reuse as TR


def _rec(**kw) -> dict:
    base = {"id": "a", "prompt_tokens": 120, "system_source": "default", "think_preopened": False}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not ...}


def test_fields_missing_on_both_sides_do_not_pass():
    theirs = {"a": _rec(prompt_tokens=..., system_source=..., think_preopened=...)}
    ours = {"a": _rec(prompt_tokens=..., system_source=..., think_preopened=...)}
    c = TR.compare_prompts(theirs, ours, fluid=False)
    assert c["compared"] == 0 and c["equal"] == 0
    assert c["not_recorded_teacher"] == 1 and c["not_recorded_other"] == 1
    assert not c["passes"]


def test_one_missing_field_is_not_recorded_rather_than_equal():
    c = TR.compare_prompts({"a": _rec(system_source=...)}, {"a": _rec(system_source=...)}, fluid=False)
    assert c["fields_missing"] == {"system_source": 2} and not c["passes"]


def test_false_is_a_valid_think_preopened_and_distinct_from_missing():
    assert TR.prompt_fields(_rec(think_preopened=False), fluid=False)[1] == []
    both_false = TR.compare_prompts({"a": _rec()}, {"a": _rec()}, fluid=False)
    assert both_false["passes"] and both_false["equal"] == 1
    false_vs_missing = TR.compare_prompts({"a": _rec()}, {"a": _rec(think_preopened=...)}, fluid=False)
    assert not false_vs_missing["passes"] and false_vs_missing["not_recorded_other"] == 1
    assert TR.prompt_fields(_rec(think_preopened=None), fluid=False)[1] == ["think_preopened"]
    assert TR.prompt_fields(_rec(prompt_tokens=True), fluid=False)[1] == ["prompt_tokens"]


def test_an_item_absent_on_the_other_side_fails_the_check():
    c = TR.compare_prompts({"a": _rec(), "b": _rec(id="b")}, {"a": _rec()}, fluid=False)
    assert c["absent_on_the_other_side"] == 1 and not c["passes"]


def test_equal_lengths_alone_do_not_establish_content_identity():
    fields = TR.compare_prompts({"a": _rec()}, {"a": _rec()}, fluid=False)
    assert fields["passes"]
    #: no construction path and no rendering: the fields match, the content is not established
    assert TR.content_identity(False, None, None, None)["status"] == "not_established"
    assert TR.content_identity(False, None, {"established": False}, {"T": fields})["status"] == "not_established"
    #: the derivation, never a "verified" status, is the most the non-fluid records can reach
    assert TR.content_identity(False, None, {"established": True}, {"T": fields})["status"] == "derived_not_recorded"


def test_fluid_prompts_with_equal_lengths_and_different_text_differ():
    def fl(text):
        return {"task_id": "a", "messages": [{"role": "user", "content": text}], "turns": [{"prompt_tokens": 50}]}
    c = TR.compare_prompts({"a": fl("rule one")}, {"a": fl("rule two")}, fluid=True)
    assert c["differ"] == 1 and not c["passes"]
    assert TR.content_identity(True, {"D": c}, None, None)["status"] == "not_established"
    empty = TR.compare_prompts({"a": fl("")}, {"a": fl("")}, fluid=True)
    assert empty["not_recorded_teacher"] == 1 and not empty["passes"]
