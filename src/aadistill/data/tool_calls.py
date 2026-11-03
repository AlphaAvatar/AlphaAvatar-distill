"""Tool calls in generated text: parsing, checking, replaying, verifying an episode.

The generic half of teacher-native tool-calling data (AGENTS.md P17: every
assistant turn is the teacher's own; upstream assistant turns are never
targets). A task offers tools, a deterministic backend answers the calls the
teacher makes, and a verifier decides whether the episode may become training
data. Nothing here knows a model, a dataset or a chat template: the call
delimiters, the tools, the gold and the backend are all the caller's (P3).

* :func:`parse_tool_calls` reads every ``<open>{json}</close>`` block of a
  turn: each must be a JSON object with a string ``name`` and an object
  ``arguments`` (a JSON-encoded string is decoded once).
* :func:`schema_problems` checks a value against a tool's JSON Schema -- the
  subset tool schemas use (``type``, ``enum``, ``properties``, ``required``,
  ``additionalProperties``, ``items``, ``anyOf``); other keywords are not
  checked and are reported by :func:`unchecked_keywords`, never silently
  treated as satisfied.
* :class:`ReplayBackend` answers a call with the response recorded for its
  canonical form, and nothing otherwise: an off-script call ends the episode.
* :func:`step_matches_gold` compares one turn's calls with the step's allowed
  call sets, order-free (a turn's parallel calls are a multiset).
* :func:`verify_episode` runs the five checks an episode must pass on every turn
  before it enters training: protocol, schema, gold, execution, task success.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ToolCall", "ToolCallError", "parse_tool_calls", "strip_tool_calls", "canonical_call",
    "schema_problems", "unchecked_keywords", "tool_index", "call_problems", "ReplayBackend",
    "step_matches_gold", "EpisodeVerdict", "verify_episode",
]

#: JSON Schema keywords `schema_problems` enforces; anything else is reported, not enforced.
CHECKED_KEYWORDS = frozenset({"type", "enum", "properties", "required", "additionalProperties",
                              "items", "anyOf", "description", "title", "default", "examples",
                              "format", "$schema"})


class ToolCallError(ValueError):
    """A tool call that cannot be read."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


def parse_tool_calls(text: str, *, open_tag: str, close_tag: str) -> tuple[list[ToolCall], list[str]]:
    """``(calls, problems)``: every delimited block of ``text``, in order.

    A block that is not valid JSON, is not an object, lacks a string ``name`` or
    whose ``arguments`` is not an object is a problem, and contributes no call;
    so is an unclosed block. Text outside the blocks is not read.
    """
    calls, problems, at = [], [], 0
    while True:
        i = text.find(open_tag, at)
        if i < 0:
            break
        j = text.find(close_tag, i + len(open_tag))
        if j < 0:
            problems.append(f"unclosed {open_tag} at character {i}")
            break
        body = text[i + len(open_tag):j].strip()
        at = j + len(close_tag)
        try:
            obj = json.loads(body)
        except ValueError as exc:
            problems.append(f"a tool call is not JSON ({exc.msg})")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
            problems.append("a tool call has no string name")
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                problems.append(f"{obj['name']}: arguments are a string that is not JSON")
                continue
        if not isinstance(args, dict):
            problems.append(f"{obj['name']}: arguments are {type(args).__name__}, not an object")
            continue
        calls.append(ToolCall(obj["name"], args))
    if text.find(close_tag, at) >= 0:
        problems.append(f"a {close_tag} without its {open_tag}")
    return calls, problems


def strip_tool_calls(text: str, *, open_tag: str, close_tag: str) -> str:
    """``text`` without its delimited blocks (the assistant message's content
    when the calls are carried as structured ``tool_calls``)."""
    out, at = [], 0
    while True:
        i = text.find(open_tag, at)
        if i < 0:
            out.append(text[at:])
            break
        j = text.find(close_tag, i)
        if j < 0:
            out.append(text[at:])
            break
        out.append(text[at:i])
        at = j + len(close_tag)
    return "".join(out).strip()


def canonical_call(call: ToolCall | Mapping[str, Any]) -> str:
    """One string per call meaning: the name and the arguments with sorted keys."""
    c = call.as_dict() if isinstance(call, ToolCall) else {"name": call["name"],
                                                            "arguments": dict(call.get("arguments") or {})}
    return json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


_TYPES: Mapping[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def schema_problems(schema: Any, value: Any, where: str = "$") -> list[str]:
    """Why ``value`` does not satisfy ``schema`` (empty when it does)."""
    if not isinstance(schema, Mapping) or not schema:
        return []
    if "anyOf" in schema:
        options = [schema_problems(s, value, where) for s in schema["anyOf"]]
        if all(options):
            return [f"{where}: matches none of {len(options)} anyOf options"]
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        known = [x for x in types if x in _TYPES]
        if known and not any(_TYPES[x](value) for x in known):
            return [f"{where}: {type(value).__name__} is not {'/'.join(known)}"]
    out = []
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{where}: {value!r} is not one of {schema['enum']!r}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                out.append(f"{where}: missing required {key!r}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(props))
            if extra:
                out.append(f"{where}: unexpected {extra}")
        for key, sub in props.items():
            if key in value:
                out += schema_problems(sub, value[key], f"{where}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for i, v in enumerate(value):
            out += schema_problems(schema["items"], v, f"{where}[{i}]")
    return out


def unchecked_keywords(schema: Any) -> set[str]:
    """JSON Schema keywords in ``schema`` (recursively) that `schema_problems` does not enforce."""
    out: set[str] = set()
    if isinstance(schema, Mapping):
        out |= {k for k in schema if k not in CHECKED_KEYWORDS}
        for k, v in schema.items():
            if k == "properties" and isinstance(v, Mapping):
                for sub in v.values():
                    out |= unchecked_keywords(sub)
            elif k in ("items",):
                out |= unchecked_keywords(v)
            elif k == "anyOf" and isinstance(v, list):
                for sub in v:
                    out |= unchecked_keywords(sub)
    return out


def tool_index(tools: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """name -> parameters schema, from OpenAI-style ``{"type": "function", "function": {...}}`` envelopes
    (or bare ``{"name", "parameters"}`` entries)."""
    out = {}
    for t in tools:
        f = t.get("function", t)
        out[str(f["name"])] = f.get("parameters") or {}
    return out


def call_problems(call: ToolCall, tools: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Check 2: the call names an offered tool and its arguments satisfy that tool's schema."""
    if call.name not in tools:
        return [f"{call.name}: not an offered tool"]
    return [f"{call.name}: {p}" for p in schema_problems(tools[call.name], dict(call.arguments))]


class ReplayBackend:
    """Recorded responses, keyed by the canonical call; an unrecorded call gets ``None``."""

    def __init__(self, table: Mapping[str, str]):
        self.table = dict(table)

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[Mapping[str, Any], str]]) -> ReplayBackend:
        return cls({canonical_call(c): r for c, r in pairs})

    def __call__(self, call: ToolCall) -> str | None:
        return self.table.get(canonical_call(call))


def step_matches_gold(calls: Sequence[ToolCall], gold: Sequence[Sequence[Mapping[str, Any]]]) -> bool:
    """Whether a turn's calls equal ONE of the allowed call sets, as a multiset of canonical calls."""
    got = Counter(canonical_call(c) for c in calls)
    return any(got == Counter(canonical_call(g) for g in option) for option in gold)


@dataclass
class EpisodeVerdict:
    admitted: bool
    checks: dict[str, bool] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "checks": dict(self.checks), "problems": list(self.problems)}


def verify_episode(turns: Sequence[Mapping[str, Any]], *, tools: Sequence[Mapping[str, Any]],
                   open_tag: str, close_tag: str, think_close: str | None,
                   expect_think: bool, gold_steps: Sequence[Sequence[Sequence[Mapping[str, Any]]]] | None = None,
                   final_check: Callable[[Sequence[Mapping[str, Any]]], bool] | None = None,
                   ) -> EpisodeVerdict:
    """The five checks, every turn (six_category_design.json :: protocol.verification).

    ``turns`` are the generated turns in order, each with ``raw`` (the generated
    text), ``stop_reason`` and ``responses`` (one backend response per call, or
    ``None`` for an off-script call). ``expect_think``: the turn must close its
    think block before any call (thinking mode); otherwise it may not generate
    ``think_close`` at all (a non-thinking mode's think block is the template's).
    ``gold_steps[k]`` lists turn ``k``'s allowed call sets (absent: no gold for
    that turn). ``final_check`` sees every turn and decides task success.
    """
    index = tool_index(tools)
    problems: list[str] = []
    checks = {"protocol": True, "schema": True, "gold": True, "execution": True, "task_success": True}
    if not turns:
        return EpisodeVerdict(False, {k: False for k in checks}, ["no turn"])
    for k, t in enumerate(turns):
        raw = str(t.get("raw") or "")
        if t.get("stop_reason") != "eos":
            checks["protocol"] = False
            problems.append(f"turn {k}: stopped by {t.get('stop_reason')!r}, not its end of turn")
        calls, bad = parse_tool_calls(raw, open_tag=open_tag, close_tag=close_tag)
        if bad:
            checks["protocol"] = False
            problems += [f"turn {k}: {b}" for b in bad]
        if think_close is not None:
            c = raw.find(think_close)
            first = raw.find(open_tag)
            if expect_think and (c < 0 or (first >= 0 and c > first)):
                checks["protocol"] = False
                problems.append(f"turn {k}: the think block is not closed before the first call")
            if not expect_think and c >= 0:
                checks["protocol"] = False
                problems.append(f"turn {k}: generated {think_close} in a non-thinking mode")
        for call in calls:
            p = call_problems(call, index)
            if p:
                checks["schema"] = False
                problems += [f"turn {k}: {x}" for x in p]
        if gold_steps is not None and k < len(gold_steps) and gold_steps[k] is not None:
            if not step_matches_gold(calls, gold_steps[k]):
                checks["gold"] = False
                problems.append(f"turn {k}: the calls are not one of the step's allowed sets")
        responses = list(t.get("responses") or [])
        if calls and (len(responses) != len(calls) or any(r is None for r in responses)):
            checks["execution"] = False
            problems.append(f"turn {k}: the backend did not answer every call")
    last_calls, _ = parse_tool_calls(str(turns[-1].get("raw") or ""), open_tag=open_tag, close_tag=close_tag)
    if last_calls:
        checks["task_success"] = False
        problems.append("the last turn still makes a call")
    elif final_check is not None and not final_check(turns):
        checks["task_success"] = False
        problems.append("the task's final check failed")
    return EpisodeVerdict(all(checks.values()), checks, problems)
