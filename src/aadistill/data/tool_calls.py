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
* :func:`schema_problems` checks a value against a tool's JSON Schema: the
  assertion keywords of `CHECKED_KEYWORDS` (types, enum / const, objects,
  arrays, strings, numbers, the applicators and local ``$ref``), with
  ``format`` an annotation as in draft 2020-12. A schema using any other
  keyword FAILS CLOSED: :func:`unchecked_keywords` names it, and
  :func:`call_problems` reports it as a schema problem, so an unchecked
  constraint is never a schema-validity pass.
* :class:`ReplayBackend` answers a call with the response recorded for its
  canonical form, and nothing otherwise: an off-script call ends the episode.
* :func:`step_matches_gold` compares one turn's calls with the step's allowed
  call sets, order-free (a turn's parallel calls are a multiset).
* :func:`verify_episode` checks an episode on every turn before it enters
  training, level by level (`LEVELS`): protocol, schema validity, recorded
  execution, step gold, tool selection, task success. A level the task carries
  no verifier for is reported ``None`` (not checked), never passed: a check
  that the target tools were called is tool selection, not task success.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ToolCall", "ToolCallError", "parse_tool_calls", "strip_tool_calls", "canonical_call",
    "schema_problems", "unchecked_keywords", "tool_index", "call_problems", "ReplayBackend",
    "step_matches_gold", "EpisodeVerdict", "verify_episode", "LEVELS", "FINAL_LEVELS",
]

#: JSON Schema keywords `schema_problems` enforces (assertions and applicators), and the annotations it may ignore
#: (``format`` is an annotation by default in draft 2020-12); any other keyword makes a schema fail closed.
ASSERTION_KEYWORDS = frozenset({
    "type", "enum", "const", "properties", "required", "additionalProperties", "patternProperties",
    "propertyNames", "minProperties", "maxProperties", "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
    "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "anyOf", "oneOf", "allOf", "not", "$ref", "$defs", "definitions"})
ANNOTATION_KEYWORDS = frozenset({"description", "title", "default", "examples", "format", "$schema", "$id",
                                 "$comment", "deprecated", "readOnly", "writeOnly"})
CHECKED_KEYWORDS = ASSERTION_KEYWORDS | ANNOTATION_KEYWORDS


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
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}
_SUBSCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions")
_SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SUBSCHEMAS = ("additionalProperties", "propertyNames", "not")


def _resolve(ref: str, root: Mapping) -> Mapping | None:
    """A local ``#/$defs/X`` / ``#/definitions/X`` (or ``#``) reference; None for anything else."""
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, Mapping) else None


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True)


def schema_problems(schema: Any, value: Any, where: str = "$", root: Any = None, depth: int = 0) -> list[str]:
    """Why ``value`` does not satisfy ``schema`` (empty when it does). Callers check `unchecked_keywords` first:
    a keyword outside `CHECKED_KEYWORDS` is not evaluated here (`call_problems` refuses such a schema)."""
    if schema is True or schema is None or (isinstance(schema, Mapping) and not schema):
        return []
    if schema is False:
        return [f"{where}: no value is allowed"]
    if not isinstance(schema, Mapping):
        return [f"{where}: the schema is not an object"]
    root = schema if root is None else root
    if depth > 64:
        return [f"{where}: schema references nest too deeply"]
    out: list[str] = []
    if "$ref" in schema:
        target = _resolve(str(schema["$ref"]), root)
        if target is None:
            return [f"{where}: unresolvable $ref {schema['$ref']!r}"]
        out += schema_problems(target, value, where, root, depth + 1)
    for key, test in (("anyOf", any), ("oneOf", None), ("allOf", all)):
        if key in schema:
            res = [not schema_problems(s, value, where, root, depth + 1) for s in schema[key]]
            if key == "oneOf" and sum(res) != 1:
                out.append(f"{where}: matches {sum(res)} of the oneOf options, not exactly one")
            elif key != "oneOf" and not test(res):
                out.append(f"{where}: fails {key}")
    if "not" in schema and not schema_problems(schema["not"], value, where, root, depth + 1):
        out.append(f"{where}: matches a schema it must not")
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        unknown = [x for x in types if x not in _TYPES]
        if unknown:
            return out + [f"{where}: unknown type {unknown}"]
        if not any(_TYPES[x](value) for x in types):
            return out + [f"{where}: {type(value).__name__} is not {'/'.join(types)}"]
    if "enum" in schema and _canon(value) not in {_canon(x) for x in schema["enum"]}:
        out.append(f"{where}: {value!r} is not one of {schema['enum']!r}")
    if "const" in schema and _canon(value) != _canon(schema["const"]):
        out.append(f"{where}: {value!r} is not {schema['const']!r}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        pats = schema.get("patternProperties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                out.append(f"{where}: missing required {key!r}")
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            out.append(f"{where}: fewer than {schema['minProperties']} properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            out.append(f"{where}: more than {schema['maxProperties']} properties")
        for key, v in value.items():
            matched = False
            if key in props:
                matched = True
                out += schema_problems(props[key], v, f"{where}.{key}", root, depth + 1)
            for pat, sub in pats.items():
                if re.search(pat, key):
                    matched = True
                    out += schema_problems(sub, v, f"{where}.{key}", root, depth + 1)
            if "propertyNames" in schema:
                out += schema_problems(schema["propertyNames"], key, f"{where}[name {key!r}]", root, depth + 1)
            if not matched and "additionalProperties" in schema:
                ap = schema["additionalProperties"]
                if ap is False:
                    out.append(f"{where}: unexpected {key!r}")
                elif isinstance(ap, Mapping):
                    out += schema_problems(ap, v, f"{where}.{key}", root, depth + 1)
    if isinstance(value, list):
        prefix = schema.get("prefixItems") or (schema["items"] if isinstance(schema.get("items"), list) else [])
        for i, (sub, v) in enumerate(zip(prefix, value)):
            out += schema_problems(sub, v, f"{where}[{i}]", root, depth + 1)
        rest = schema.get("items") if isinstance(schema.get("items"), (Mapping, bool)) else None
        if rest is not None:
            for i, v in enumerate(value[len(prefix):], start=len(prefix)):
                out += schema_problems(rest, v, f"{where}[{i}]", root, depth + 1)
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(f"{where}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(f"{where}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and len({_canon(v) for v in value}) != len(value):
            out.append(f"{where}: items are not unique")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(f"{where}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{where}: longer than {schema['maxLength']}")
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    out.append(f"{where}: does not match {schema['pattern']!r}")
            except re.error:
                out.append(f"{where}: pattern {schema['pattern']!r} cannot be checked")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        xlo, xhi = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if isinstance(xlo, bool):              #: draft 4: a flag on minimum
            xlo, lo = (lo if xlo else None), (None if xlo else lo)
        if isinstance(xhi, bool):
            xhi, hi = (hi if xhi else None), (None if xhi else hi)
        if lo is not None and value < lo:
            out.append(f"{where}: below the minimum {lo}")
        if hi is not None and value > hi:
            out.append(f"{where}: above the maximum {hi}")
        if xlo is not None and value <= xlo:
            out.append(f"{where}: not above {xlo}")
        if xhi is not None and value >= xhi:
            out.append(f"{where}: not below {xhi}")
        m = schema.get("multipleOf")
        if m and abs(value / m - round(value / m)) > 1e-9:
            out.append(f"{where}: not a multiple of {m}")
    return out


def unchecked_keywords(schema: Any) -> set[str]:
    """Keywords in ``schema`` (recursively, through every subschema) that `schema_problems` does not evaluate."""
    out: set[str] = set()
    if isinstance(schema, Mapping):
        out |= {k for k in schema if k not in CHECKED_KEYWORDS}
        for k, v in schema.items():
            if k in _SUBSCHEMA_MAPS and isinstance(v, Mapping):
                for sub in v.values():
                    out |= unchecked_keywords(sub)
            elif k in _SUBSCHEMA_LISTS and isinstance(v, list):
                for sub in v:
                    out |= unchecked_keywords(sub)
            elif k == "items":
                for sub in (v if isinstance(v, list) else [v]):
                    out |= unchecked_keywords(sub)
            elif k in _SUBSCHEMAS:
                out |= unchecked_keywords(v)
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
    """Schema validity: the call names an offered tool and its arguments satisfy that tool's schema. A schema with
    a keyword the validator does not evaluate fails closed (`unchecked_keywords`)."""
    if call.name not in tools:
        return [f"{call.name}: not an offered tool"]
    unchecked = unchecked_keywords(tools[call.name])
    if unchecked:
        return [f"{call.name}: the schema uses keywords the validator does not check: {sorted(unchecked)}"]
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


#: What an episode's verdict reports, in order. ``protocol``, ``schema_validity`` and ``recorded_execution`` are
#: checked on every episode; ``step_gold`` only where the task carries per-step gold calls; ``tool_selection`` and
#: ``task_success`` only where the task carries a final check of that level.
LEVELS = ("protocol", "schema_validity", "recorded_execution", "step_gold", "tool_selection", "task_success")
FINAL_LEVELS = ("tool_selection", "task_success")


@dataclass
class EpisodeVerdict:
    admitted: bool
    #: level -> True (passed) / False (failed) / None (the task carries no verifier for it)
    checks: dict[str, bool | None] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def verified_levels(self) -> list[str]:
        return [k for k in LEVELS if self.checks.get(k) is True]

    def as_dict(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "checks": dict(self.checks), "problems": list(self.problems),
                "verified_levels": self.verified_levels()}


def verify_episode(turns: Sequence[Mapping[str, Any]], *, tools: Sequence[Mapping[str, Any]],
                   open_tag: str, close_tag: str, think_close: str | None,
                   expect_think: bool, gold_steps: Sequence[Sequence[Sequence[Mapping[str, Any]]]] | None = None,
                   final_checks: Mapping[str, Callable[[Sequence[Mapping[str, Any]]], bool]] | None = None,
                   ) -> EpisodeVerdict:
    """Every level of `LEVELS`, every turn.

    ``turns`` are the generated turns in order, each with ``raw`` (the generated
    text), ``stop_reason`` and ``responses`` (one backend response per call, or
    ``None`` for an off-script call). ``expect_think``: the turn must close its
    think block before any call (thinking mode); otherwise it may not generate
    ``think_close`` at all (a non-thinking mode's think block is the template's).
    The episode must end on a turn without a call. ``gold_steps[k]`` lists turn
    ``k``'s allowed call sets (absent or ``None``: no gold for that turn).
    ``final_checks`` maps a level of `FINAL_LEVELS` to a check that sees every
    turn. An episode is admitted when no level fails; levels without a verifier
    stay ``None`` and are reported as such.
    """
    unknown = set(final_checks or {}) - set(FINAL_LEVELS)
    if unknown:
        raise ValueError(f"final checks of unknown levels {sorted(unknown)}; levels are {FINAL_LEVELS}")
    index = tool_index(tools)
    problems: list[str] = []
    has_gold = gold_steps is not None and any(g is not None for g in gold_steps)
    checks: dict[str, bool | None] = {"protocol": True, "schema_validity": True, "recorded_execution": True,
                                      "step_gold": True if has_gold else None,
                                      **{k: (True if k in (final_checks or {}) else None) for k in FINAL_LEVELS}}
    if not turns:
        return EpisodeVerdict(False, {k: (False if v is not None else None) for k, v in checks.items()}, ["no turn"])
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
                checks["schema_validity"] = False
                problems += [f"turn {k}: {x}" for x in p]
        if has_gold and k < len(gold_steps) and gold_steps[k] is not None:
            if not step_matches_gold(calls, gold_steps[k]):
                checks["step_gold"] = False
                problems.append(f"turn {k}: the calls are not one of the step's allowed sets")
        responses = list(t.get("responses") or [])
        if calls and (len(responses) != len(calls) or any(r is None for r in responses)):
            checks["recorded_execution"] = False
            problems.append(f"turn {k}: the backend did not answer every call")
    last_calls, _ = parse_tool_calls(str(turns[-1].get("raw") or ""), open_tag=open_tag, close_tag=close_tag)
    if last_calls:
        checks["protocol"] = False
        problems.append("the episode ends on a call, not on an answer")
    for level, check in (final_checks or {}).items():
        if not check(turns):
            checks[level] = False
            problems.append(f"the task's {level} check failed")
    return EpisodeVerdict(not any(v is False for v in checks.values()), checks, problems)
