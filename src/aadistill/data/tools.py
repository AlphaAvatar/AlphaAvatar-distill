"""The project's canonical tool representation, and the one conversion into it.

Every tool asset in this project renders tools to the model as an OpenAI-style
envelope:

    {"type": "function",
     "function": {"name": ..., "description": ..., "parameters": ...}}

`parameters` is carried through **verbatim** from the source. That is deliberate
and is what the student was trained on: the Stage-2 mixtures built by
`scripts/data/build_stage2_v1.py` pass the xLAM parameter map straight into the
envelope, so re-shaping it into a JSON Schema here would show the model a tool
description it has never seen.

This function exists because `recovery_search_v1` did **not** use it. That asset
stored the raw xLAM `tools` column — a JSON *string* — and its tool prompts
therefore could not be rendered at all: `apply_chat_template` raises
`ValueError: Tools should either be a JSON schema, or a callable function ...`,
which stopped Stage 3 of the micro-preflight after the permanent controls had
been trained. The conversion was always in the codebase; the battery builder
simply did not call it.

One converter, two callers. A second one written for a single asset is how two
assets come to disagree about what a tool is.
"""

from __future__ import annotations

import json
from typing import Any


class ToolFormatError(ValueError):
    """A tool payload cannot be interpreted as the canonical representation."""


def xlam_tools_to_canonical(tools_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """xLAM `{name, description, parameters}` -> the canonical envelope.

    Byte-for-byte the transformation `build_stage2_v1.build_xlam` applies, and
    the reason this is a function rather than a copy of it.
    """
    return [{"type": "function",
             "function": {"name": tool["name"],
                          "description": tool.get("description", ""),
                          "parameters": tool.get("parameters", {})}}
            for tool in tools_raw
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)]


def parse_xlam_tools(value: Any, *, where: str = "") -> list[dict[str, Any]]:
    """Strictly read a stored xLAM tools payload. Fails closed.

    Accepts the JSON string form the upstream dataset ships, or an already
    decoded list. A malformed payload, a non-list root, or an entry that is not
    a named object raises — a tool prompt whose tools cannot be established is
    not a tool prompt, and silently dropping one changes what the item tests.
    """
    context = f"{where}: " if where else ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ToolFormatError(f"{context}tools is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise ToolFormatError(
            f"{context}tools must be a list at its root, got "
            f"{type(value).__name__}")
    if not value:
        raise ToolFormatError(f"{context}tools is empty")
    for i, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise ToolFormatError(f"{context}tools[{i}] is not an object")
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise ToolFormatError(f"{context}tools[{i}] has no usable name")
        params = tool.get("parameters", {})
        if not isinstance(params, dict):
            raise ToolFormatError(
                f"{context}tools[{i}].parameters is {type(params).__name__}, "
                "not an object")
    return value


def canonical_tool_meaning(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The model-visible meaning of a tool list, independent of its envelope.

    Used to prove that a representation change preserved the tools themselves:
    the same names, descriptions and parameter maps, in the same order. Reads
    both the canonical envelope and the bare xLAM form, so the two can be
    compared directly.
    """
    out = []
    for tool in tools:
        body = tool.get("function") if tool.get("type") == "function" else tool
        body = body if isinstance(body, dict) else {}
        out.append({"name": body.get("name"),
                    "description": body.get("description", ""),
                    "parameters": body.get("parameters", {})})
    return out


def normalize_tools(value: Any, *, where: str = "") -> list[dict[str, Any]]:
    """Read a stored tools payload in **either** representation, fail closed.

    Consumers that only need the tools' *meaning* — the scorer, an audit — must
    not care whether an asset stores the bare xLAM form or the canonical
    envelope. `recovery_search_v1` stored the first, `recovery_search_v2` and
    every Stage-2 mixture store the second, and a consumer that understood only
    one of them read every tool name as `None`: that is exactly how the v2
    migration first drove `tool_name_valid` to 0.0 for every item.

    Strictness is kept where it belongs. A malformed payload, a non-list root,
    an empty list, or an entry with no usable name raises `ToolFormatError`.
    """
    context = f"{where}: " if where else ""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ToolFormatError(f"{context}tools is not valid JSON: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ToolFormatError(
            f"{context}tools must be a non-empty list at its root, got "
            f"{type(value).__name__}")
    meaning = canonical_tool_meaning(value)
    for i, tool in enumerate(meaning):
        if not isinstance(tool["name"], str) or not tool["name"]:
            raise ToolFormatError(f"{context}tools[{i}] has no usable name")
        if not isinstance(tool["parameters"], dict):
            raise ToolFormatError(
                f"{context}tools[{i}].parameters is "
                f"{type(tool['parameters']).__name__}, not an object")
    return meaning
