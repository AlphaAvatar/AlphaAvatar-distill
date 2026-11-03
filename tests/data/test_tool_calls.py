"""`aadistill.data.tool_calls`: parsing, schema checks, replay, gold matching, an episode's verification levels."""
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
        assert unchecked_keywords({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}}}) == set()
        assert unchecked_keywords({"type": "object", "properties": {"n": {"if": {}, "optional": True}},
                                   "$defs": {"d": {"dependentRequired": {}}}}) == {"if", "optional", "dependentRequired"}

    def test_numbers_strings_arrays_and_applicators(self):
        assert schema_problems({"type": "integer", "minimum": 1, "maximum": 3}, 0) != []
        assert schema_problems({"type": "integer", "minimum": 1, "maximum": 3}, 2.0) == []     # 2.0 is an integer
        assert schema_problems({"exclusiveMinimum": 0}, 0) != [] and schema_problems({"multipleOf": 5}, 7) != []
        assert schema_problems({"minLength": 2, "maxLength": 3, "pattern": "^a"}, "abcd") != []
        assert schema_problems({"minLength": 2, "maxLength": 3, "pattern": "^a"}, "ab") == []
        assert schema_problems({"type": "array", "minItems": 2, "uniqueItems": True}, [1, 1]) != []
        assert schema_problems({"prefixItems": [{"type": "string"}], "items": {"type": "integer"}}, ["a", 1, 2]) == []
        assert schema_problems({"prefixItems": [{"type": "string"}], "items": {"type": "integer"}}, ["a", "b"]) != []
        assert schema_problems({"const": "x"}, "y") != [] and schema_problems({"enum": [1, [2]]}, [2]) == []
        assert schema_problems({"oneOf": [{"type": "integer"}, {"type": "number"}]}, 1) != []    # matches both
        assert schema_problems({"allOf": [{"minimum": 0}, {"maximum": 1}]}, 2) != []
        assert schema_problems({"not": {"type": "null"}}, None) != []
        s = {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"type": "string"}},
             "additionalProperties": {"type": "integer"}}
        assert schema_problems(s, {"a": "x", "b": 1}) == [] and schema_problems(s, {"a": 1}) != []
        assert schema_problems(s, {"a": "x", "b": "y"}) != []                  # additionalProperties as a schema
        assert schema_problems({"$ref": "#/$defs/missing"}, 1) != []

    def test_a_schema_with_an_unchecked_keyword_fails_closed(self):
        from aadistill.data.tool_calls import call_problems, tool_index
        tools = tool_index([{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"type": "string", "if": {"const": "a"}}}}}}])
        assert call_problems(ToolCall("f", {"x": "b"}), tools) != []


class TestReplayAndGold:
    def test_replay_answers_only_recorded_calls(self):
        b = ReplayBackend.from_pairs([({"name": "x", "arguments": {"a": 1}}, "ok")])
        assert b(ToolCall("x", {"a": 1})) == "ok" and b(ToolCall("x", {"a": 2})) is None

    def test_gold_is_an_order_free_multiset(self):
        gold = [[{"name": "x", "arguments": {}}, {"name": "y", "arguments": {}}]]
        assert step_matches_gold([ToolCall("y", {}), ToolCall("x", {})], gold)
        assert not step_matches_gold([ToolCall("x", {}), ToolCall("x", {})], gold)


def _verify(turns, thinking=True, gold=None, final=None, level="task_success"):
    return verify_episode(turns, tools=TOOLS, open_tag=OPEN, close_tag=C, think_close=T, expect_think=thinking,
                          gold_steps=gold, final_checks={level: final} if final else None)


class TestTheLevels:
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
        assert _verify([{"raw": f"x{T}" + call("nope"), "stop_reason": "eos", "responses": ["s"]},
                        {"raw": f"x{T}d", "stop_reason": "eos"}]).checks["schema_validity"] is False
        assert _verify([{"raw": f"x{T}" + good_call, "stop_reason": "eos", "responses": ["s"]},
                        {"raw": f"x{T}d", "stop_reason": "eos"}],
                       gold=[[[{"name": "get_weather", "arguments": {"city": "Rome"}}]]]).checks["step_gold"] is False
        v = _verify([{"raw": f"x{T}" + good_call, "stop_reason": "eos", "responses": [None]}])
        assert v.checks["recorded_execution"] is False
        assert _verify([{"raw": f"x{T}" + good_call, "stop_reason": "eos", "responses": ["s"]}]).checks["protocol"] is False
        assert _verify([{"raw": f"x{T}d", "stop_reason": "eos"}], final=lambda ts: False).checks["task_success"] is False

    def test_a_level_without_a_verifier_is_not_checked_never_passed(self):
        v = _verify([{"raw": f"x{T}" + call("get_weather", city="Paris"), "stop_reason": "eos", "responses": ["s"]},
                     {"raw": f"x{T}done", "stop_reason": "eos"}])
        assert v.admitted and v.checks["step_gold"] is None and v.checks["task_success"] is None
        assert v.checks["tool_selection"] is None
        assert v.verified_levels() == ["protocol", "schema_validity", "recorded_execution"]

    def test_a_tool_selection_check_is_not_task_success(self):
        v = _verify([{"raw": f"x{T}" + call("get_weather", city="Paris"), "stop_reason": "eos", "responses": ["s"]},
                     {"raw": f"x{T}done", "stop_reason": "eos"}], final=lambda ts: True, level="tool_selection")
        assert v.checks["tool_selection"] is True and v.checks["task_success"] is None
        assert "task_success" not in v.as_dict()["verified_levels"]

    def test_an_unknown_final_level_is_refused(self):
        import pytest
        with pytest.raises(ValueError):
            _verify([{"raw": f"x{T}d", "stop_reason": "eos"}], final=lambda ts: True, level="success")
