#!/usr/bin/env python3
"""Toucan-1.5M's SFT config -> `tool_pool@v1` task records (task material only). $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_sources_toucan \\
        --sft-dir artifacts/stages/stage-2/teacher_modes/tool_sources/Agent-Ark__Toucan-1.5M@0df3cf37/SFT --out <tasks.jsonl>

A CANDIDATE source (audit/tool_calling_source_review_v2.json): nothing here admits it to training. A row becomes:

* tools -- the row's OpenAI tool envelopes, unedited; messages -- its first user turn; later user turns (the
  multi-turn subset) arrive after each answer (`later_user_turns`);
* steps -- the recorded trajectory read as steps: the ``tool_call`` messages between two non-call messages are ONE
  step (parallel calls), each paired in order with the ``tool_response`` that follows;
* backend -- a replay table: each recorded call (a Python-literal ``{'name', 'arguments'}``) mapped by its
  canonical form to its response. These are Toucan's REAL MCP executions (June-Sept 2025), failures included. A
  call recorded more than once with DIFFERENT responses is not put in the table: it is listed under
  ``backend.conflicts`` and the pool refuses the task (no response is chosen silently);
* servers -- the atomic MCP server of every offered tool: the tool name's ``<server slug>-`` prefix, the slugs
  learnt from the rows whose target tools name them (``known_slugs``); a tool no slug matches is listed in
  ``unattributed_tools`` and the pool isolates it by its own identity;
* errors -- a response is an error only when it reports one (`error_kind`): a JSON object with a non-null
  ``error``, ``isError: true``, ``success: false`` or ``status: "error"``, or text that opens with an error report;
  ``"error": null`` in a successful result is not one. An error that names a problem with the call's arguments
  (an argument validation error, a missing / invalid argument) is ``invalid_arguments``; any other (the server's
  own output validation, auth, quota, not found, a crash) is ``tool_failure``. The counts are recorded per task;
* category -- irrelevant -> no_tool; an error followed by a successful call of the same tool with other arguments
  -> error_recovery (``corrected_retry``, whatever the error's kind); errors without such a retry ->
  error_recovery (``tool_failure``); the multi-turn subset -> multi_turn; a step with two or more calls ->
  multiple_parallel; two or more sequential steps -> mcp_agent; otherwise tool_selection. Asking the user for a
  missing value has no source here (no gold for it);
* verification -- the irrelevant subset: no call anywhere (tool selection, B); otherwise every target tool called
  (tool selection, ``calls_tools``). No Toucan task carries per-step gold or a task-success verifier, and the
  record says so (``source.verification``).

No upstream assistant turn, reasoning or call text enters a task's context or a target.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.data.tool_calls import canonical_call  # noqa: E402
from stages.teacher_modes import tool_episodes as TE  # noqa: E402

DATASET, REVISION = "Agent-Ark/Toucan-1.5M", "0df3cf37f2abefb380370cfb02eabea2a35ae782"
MAX_TURNS = 10
COLUMNS = ["uuid", "subset_name", "target_tools", "tools", "messages"]
#: text that opens with an error report (case-insensitive, after leading whitespace)
ERROR_OPENING = re.compile(r"^(error\b|an error occurred|failed\b|failure\b|exception\b|traceback\b|mcp ?error|"
                           r"api invoke error|http error|tool '[^']*' execution failed|unauthorized|forbidden)",
                           re.IGNORECASE)
ERROR_ANYWHERE = re.compile(r"McpError|Traceback \(most recent call last\)")
#: an error that names an argument problem
ARGUMENT_ERROR = re.compile(r"validation error|missing (required )?(argument|parameter|field|value)|"
                            r"required (argument|parameter|field|property)|is a required property|field required|"
                            r"invalid (argument|parameter|input|value|params)|unexpected keyword argument|"
                            r"is not of type", re.IGNORECASE)
#: a pydantic validation error of the tool's OUTPUT is the server's own failure, not the call's
OUTPUT_VALIDATION = re.compile(r"validation error for \w*Output\b")


def _json(v):
    return json.loads(v) if isinstance(v, str) else v


def parse_call(content) -> dict | None:
    try:
        c = ast.literal_eval(content) if isinstance(content, str) else content
        args = c.get("arguments", {})
        args = json.loads(args) if isinstance(args, str) else args
        if not isinstance(c.get("name"), str) or not isinstance(args, dict):
            return None
        return {"name": c["name"], "arguments": args}
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return None


def error_kind(response: str) -> str | None:
    """None for a response that reports no error; ``invalid_arguments`` or ``tool_failure`` for one that does."""
    text = str(response).strip()
    is_error = False
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        is_error = (obj.get("isError") is True or obj.get("success") is False
                    or str(obj.get("status", "")).lower() == "error"
                    or obj.get("error") not in (None, "", False, [], {}))
    elif obj is None:
        is_error = bool(ERROR_OPENING.match(text) or ERROR_ANYWHERE.search(text))
    if not is_error:
        return None
    if OUTPUT_VALIDATION.search(text) or not ARGUMENT_ERROR.search(text):
        return "tool_failure"
    return "invalid_arguments"


def steps_of(msgs: list[dict]) -> list[list[tuple[dict | None, str | None]]]:
    """The trajectory's tool steps: each the ``(call, response)`` pairs of one run of consecutive calls."""
    steps, calls, responses = [], None, None
    for m in msgs:
        role = m["role"]
        if role == "tool_call":
            if calls is None or responses:
                if calls is not None:
                    steps.append((calls, responses))
                calls, responses = [], []
            calls.append(parse_call(m["content"]))
        elif role == "tool_response" and calls is not None:
            responses.append(str(m["content"]))
        else:
            if calls is not None:
                steps.append((calls, responses))
            calls, responses = None, None
    if calls is not None:
        steps.append((calls, responses))
    return [[(c, responses[k] if k < len(responses) else None) for k, c in enumerate(calls)]
            for calls, responses in steps]


def _targets(row: dict) -> list[str]:
    return [t.strip().split("::", 1)[-1] for t in str(row.get("target_tools") or "").split(",") if t.strip()]


def _names(tools: list[dict]) -> list[str]:
    return [str(t.get("function", t)["name"]) for t in tools]


def slugs_of_row(row: dict) -> set[str]:
    """The server slugs a row's target tools reveal: an offered name ``<slug>-<target>``."""
    targets = _targets(row)
    out = set()
    for n in _names(_json(row["tools"]) or []):
        for t in targets:
            if n.endswith("-" + t) and len(n) > len(t) + 1:
                out.add(n[: -len(t) - 1])
                break
    return out


def server_of(name: str, known_slugs: Iterable[str]) -> str | None:
    """The longest known slug ``s`` with ``name`` = ``s-<tool>``."""
    return max((s for s in known_slugs if name.startswith(s + "-")), key=len, default=None)


def task_of(row: dict, known_slugs: Iterable[str] = ()) -> dict | None:
    msgs = _json(row["messages"])
    tools = _json(row["tools"]) or []
    users = [m["content"] for m in msgs if m["role"] == "user"]
    if not users:
        return None
    known = set(known_slugs) | slugs_of_row(row)
    servers, unattributed = set(), []
    for n in _names(tools):
        s = server_of(n, known)
        (servers.add(s) if s else unattributed.append(n))
    steps = steps_of(msgs)
    table, seen, conflicts = {}, {}, {}
    errors = {"invalid_arguments": 0, "tool_failure": 0}
    failed, retried = set(), False
    for step in steps:
        for c, r in step:
            if c is None or r is None:
                continue
            key = canonical_call(c)
            if key in seen and seen[key] != r:
                conflicts.setdefault(key, {seen[key]}).add(r)
            seen.setdefault(key, r)
            kind = error_kind(r)
            if kind:
                errors[kind] += 1
                failed.add((c["name"], key))
            elif any(name == c["name"] and k != key for name, k in failed):
                retried = True
    for key, r in seen.items():
        if key not in conflicts:
            table[key] = r
    sub = row["subset_name"]
    targets = sorted(set(_targets(row)))
    if sub == "irrelevant":
        category, detail, final = "no_tool", None, {"kind": "no_call"}
    else:
        final = {"kind": "calls_tools", "values": targets}
        detail = ("corrected_retry" if retried else "tool_failure" if any(errors.values()) else None)
        category = ("error_recovery" if detail else "multi_turn" if sub == "multi-turn" else
                    "multiple_parallel" if any(len(s) >= 2 for s in steps) else
                    "mcp_agent" if len(steps) >= 2 else "tool_selection")
    task = {"task_id": TE.task_id_of(DATASET, REVISION, row["uuid"]), "category": category,
            "category_detail": detail,
            "source": {"dataset": DATASET, "revision": REVISION, "config": "SFT", "row": row["uuid"], "subset": sub,
                       "verification_class": "B" if sub == "irrelevant" else "G2",
                       "recorded": {"steps": len(steps), "calls": sum(len(s) for s in steps),
                                    "max_calls_per_step": max((len(s) for s in steps), default=0), **errors}},
            "family": None, "split": None, "servers": sorted(servers), "unattributed_tools": sorted(unattributed),
            "tools": tools, "messages": [{"role": "user", "content": users[0]}], "later_user_turns": users[1:],
            "backend": {"kind": "replay", "table": table,
                        "conflicts": [{"call": k, "responses": len(v)} for k, v in sorted(conflicts.items())]},
            "final_check": final, "max_turns": MAX_TURNS}
    task["source"]["verification"] = {"levels": TE.verification_levels(task), "step_gold": False,
                                      "task_success_verifier": None}
    return task


def _rows(sft_dir: Path):
    import pyarrow.parquet as pq
    for f in sorted(Path(sft_dir).glob("train-*.parquet")):
        yield from pq.read_table(f, columns=COLUMNS).to_pylist()


def known_slugs(sft_dir: Path) -> set[str]:
    """Pass one: every server slug some row's target tools reveal."""
    out: set[str] = set()
    for row in _rows(sft_dir):
        out |= slugs_of_row(row)
    return out


def convert(sft_dir: Path):
    known = known_slugs(sft_dir)
    for row in _rows(sft_dir):
        t = task_of(row, known)
        if t is not None:
            yield t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sft-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    n = 0
    with open(a.out, "w") as f:
        for t in convert(Path(a.sft_dir)):
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
            n += 1
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
