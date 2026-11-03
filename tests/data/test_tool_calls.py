"""`aadistill.data.tool_calls`: parsing, schema checks, replay, gold matching, the five episode checks."""
from __future__ import annotations

from aadistill.data.tool_calls import (
    ReplayBackend, ToolCall, canonical_call, parse_tool_calls, schema_problems, step_matches_gold,
    strip_tool_calls, unchecked_keywords, verify_episode,
)

OPEN, C, T = "<tool_call>", "</tool_call>", "</think>"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
    "required": ["city"], "additionalProperties": False}}}]


def call(name, **args):
    return f'{OPEN}\n{{"name": "{name}", "arguments": {__import__("json").dumps(args)}}}\n{C}'


class TestParsing:
    def test_calls_in_order_and_problems_reported(self):
        text = "a " + call("x", k=1) + " b " + call("y") + f" {OPEN}not json{C} {OPEN}{{\"arguments\": {{}}}}{C}"
        calls, bad = parse_tool_calls(text, open_tag=OPEN, close_tag=C)
        assert [c.name for c in calls] == ["x", "y"] and calls[0].arguments == {"k": 1}
        assert len(bad) == 2

    def test_string_arguments_are_decoded_once_and_unclosed_is_a_problem(self):
        calls, bad = parse_tool_calls(f'{OPEN}{{"name": "x", "arguments": "{{\\"a\\": 2}}"}}{C}{OPEN}', open_tag=OPEN, close_tag=C)
        assert calls[0].arguments == {"a": 2} and any("unclosed" in b for b in bad)

    def test_strip_leaves_the_prose(self):
        assert strip_tool_calls("hi " + call("x") + " there", open_tag=OPEN, close_tag=C) == "hi  there"

    def test_canonical_form_ignores_key_order(self):
        assert canonical_call(ToolCall("x", {"a": 1, "b": 2})) == canonical_call({"name": "x", "arguments": {"b": 2, "a": 1}})


class TestSchema:
    def test_the_supported_keywords(self):
        s = TOOLS[0]["function"]["parameters"]
        assert schema_problems(s, {"city": "Paris"}) == []
        assert any("missing required" in p for p in schema_problems(s, {}))
        assert any("unexpected" in p for p in schema_problems(s, {"city": "P", "x": 1}))
        assert any("not one of" in p for p in schema_problems(s, {"city": "P", "unit": "k"}))
        assert any("is not string" in p for p in schema_problems(s, {"city": 3}))
        assert schema_problems({"type": "array", "items": {"type": "integer"}}, [1, True]) != []
        assert schema_problems({"anyOf": [{"type": "string"}, {"type": "null"}]}, None) == []

    def test_unchecked_keywords_are_named_not_assumed(self):
        assert unchecked_keywords({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}}}) == {"minimum"}


class TestReplayAndGold:
    def test_replay_answers_only_recorded_calls(self):
        b = ReplayBackend.from_pairs([({"name": "x", "arguments": {"a": 1}}, "ok")])
        assert b(ToolCall("x", {"a": 1})) == "ok" and b(ToolCall("x", {"a": 2})) is None

    def test_gold_is_an_order_free_multiset(self):
        gold = [[{"name": "x", "arguments": {}}, {"name": "y", "arguments": {}}]]
        assert step_matches_gold([ToolCall("y", {}), ToolCall("x", {})], gold)
        assert not step_matches_gold([ToolCall("x", {}), ToolCall("x", {})], gold)


def _verify(turns, thinking=True, gold=None, final=None):
    return verify_episode(turns, tools=TOOLS, open_tag=OPEN, close_tag=C, think_close=T, expect_think=thinking,
                          gold_steps=gold, final_check=final)


class TestTheFiveChecks:
    def test_an_admitted_episode(self):
        turns = [{"raw": f"plan{T}\n" + call("get_weather", city="Paris"), "stop_reason": "eos", "responses": ["sun"]},
                 {"raw": f"ok{T}\nIt is sunny.", "stop_reason": "eos", "responses": []}]
        v = _verify(turns, gold=[[[{"name": "get_weather", "arguments": {"city": "Paris"}}]], None],
                    final=lambda ts: "sunny" in ts[-1]["raw"])
        assert v.admitted, v.problems

    def test_each_check_refuses_its_failure(self):
        good_call = call("get_weather", city="Paris")
        assert not _verify([{"raw": good_call + T, "stop_reason": "eos", "responses": ["s"]},
                            {"raw": f"x{T}done", "stop_reason": "eos"}]).checks["protocol"]
        assert not _verify([{"raw": f"x{T}done", "stop_reason": "eos"}], thinking=False).checks["protocol"]
        assert not _verify([{"raw": f"x{T}done", "stop_reason": "degeneration"}]).checks["protocol"]
        assert not _verify([{"raw": f"x{T}" + call("nope"), "stop_reason": "eos", "responses": ["s"]},
                            {"raw": f"x{T}d", "stop_reason": "eos"}]).checks["schema"]
        assert not _verify([{"raw": f"x{T}" + good_call, "stop_reason": "eos", "responses": ["s"]},
                            {"raw": f"x{T}d", "stop_reason": "eos"}],
                           gold=[[[{"name": "get_weather", "arguments": {"city": "Rome"}}]]]).checks["gold"]
        assert not _verify([{"raw": f"x{T}" + good_call, "stop_reason": "eos", "responses": [None]}]).checks["execution"]
        assert not _verify([{"raw": f"x{T}" + good_call, "stop_reason": "eos", "responses": ["s"]}]).checks["task_success"]
        assert not _verify([{"raw": f"x{T}d", "stop_reason": "eos"}], final=lambda ts: False).checks["task_success"]
