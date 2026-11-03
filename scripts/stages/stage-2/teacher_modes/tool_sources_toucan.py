#!/usr/bin/env python3
"""Toucan-1.5M's SFT config -> `tool_pool@v1` task records (task material only). $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_sources_toucan \\
        --sft-dir artifacts/stages/stage-2/teacher_modes/tool_sources/Agent-Ark__Toucan-1.5M@0df3cf37/SFT --out <tasks.jsonl>

A CANDIDATE source (audit/tool_calling_source_review_v2.json): nothing here admits it to training. A row becomes:

* tools -- the row's OpenAI tool envelopes, unedited; messages -- its first user turn; later user turns (the
  multi-turn subset) arrive after each answer (`later_user_turns`);
* backend -- a replay table: each recorded call (a ``tool_call`` message, a Python-literal ``{'name', 'arguments'}``)
  mapped by its canonical form to the ``tool_response`` that follows it; a call recorded twice with different
  responses keeps the first. These are Toucan's REAL MCP executions (June-Sept 2025), failures included;
* success -- the irrelevant subset: no call anywhere (B); otherwise every target tool called (G2, `calls_tools`);
* category -- irrelevant -> no_tool; a recorded error response -> errors_missing_parameters; multi-turn ->
  multi_turn; one call -> tool_selection; two -> multiple_parallel; three or more -> mcp_agent;
* family -- the row's MCP servers (from ``target_tools``'s ``Server::tool``), so the split keeps a server's tasks
  together.

No upstream assistant turn, reasoning or call text enters a task's context or a target.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.data.tool_calls import canonical_call  # noqa: E402
from stages.teacher_modes import tool_episodes as TE  # noqa: E402

DATASET, REVISION = "Agent-Ark/Toucan-1.5M", "0df3cf37f2abefb380370cfb02eabea2a35ae782"
MAX_TURNS = 10


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


def task_of(row: dict) -> dict | None:
    msgs = _json(row["messages"])
    tools = _json(row["tools"]) or []
    users = [m["content"] for m in msgs if m["role"] == "user"]
    if not users:
        return None
    table, pending, errors, n_calls = {}, [], 0, 0
    for m in msgs:
        if m["role"] == "tool_call":
            pending.append(parse_call(m["content"]))
            n_calls += 1
        elif m["role"] == "tool_response" and pending:
            c = pending.pop(0)
            if c is not None:
                table.setdefault(canonical_call(c), str(m["content"]))
            errors += "error" in str(m["content"]).lower()
    sub = row["subset_name"]
    targets = [t.strip() for t in str(row.get("target_tools") or "").split(",") if t.strip()]
    servers = sorted({t.split("::", 1)[0] for t in targets if "::" in t})
    if sub == "irrelevant":
        category, final = "no_tool", {"kind": "no_call"}
    else:
        category = ("errors_missing_parameters" if errors else "multi_turn" if sub == "multi-turn" else
                    "tool_selection" if n_calls <= 1 else "multiple_parallel" if n_calls == 2 else "mcp_agent")
        final = {"kind": "calls_tools", "values": sorted({t.split("::", 1)[-1] for t in targets})}
    return {"task_id": TE.task_id_of(DATASET, REVISION, row["uuid"]), "category": category,
            "source": {"dataset": DATASET, "revision": REVISION, "config": "SFT", "row": row["uuid"], "subset": sub,
                       "verification_class": "B" if sub == "irrelevant" else "G2"},
            "family": None, "family_hint": "|".join(servers) if servers else None, "split": None,
            "tools": tools, "messages": [{"role": "user", "content": users[0]}], "later_user_turns": users[1:],
            "backend": {"kind": "replay", "table": table}, "final_check": final, "max_turns": MAX_TURNS}


def convert(sft_dir: Path):
    import pyarrow.parquet as pq
    for f in sorted(Path(sft_dir).glob("train-*.parquet")):
        for row in pq.read_table(f, columns=["uuid", "subset_name", "target_tools", "tools", "messages"]).to_pylist():
            t = task_of(row)
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
