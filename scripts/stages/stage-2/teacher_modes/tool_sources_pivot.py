#!/usr/bin/env python3
"""The NeMo Gym Pivot pair -> `tool_pool@v1` task records, rebuilt so the teacher runs from step one. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_sources_pivot \\
        --which function_calling --jsonl <.../nvidia__Nemotron-RL-Agentic-Function-Calling-Pivot-v1@0ac34ec8/train.jsonl> \\
        --out <tasks.jsonl> [--audit <audit.json>]

CANDIDATE sources (audit/tool_calling_source_review_v2.json): nothing here admits them to training.

A Pivot row is ONE step of an expert trajectory: its ``input`` is the whole history up to that step -- the expert's
own earlier messages, reasoning and calls included -- and ``expected_action`` is the step's next action (one
function call or one message; no row carries parallel calls). A teacher may continue no other model's turns
(P17), so a task is rebuilt from the first user turn only:

* context -- the step-1 row's system text and user message(s), which hold no assistant item; the tools converted to
  chat-template envelopes, unedited;
* gold -- every step of turn 1, in order: a call step's ``expected_action`` (name + arguments, compared exactly after
  canonicalization, `step_matches_gold`), a message step's "no call". Steps count from 1 in Function-Calling and from
  0 in Conversational: turn 1 starts at its lowest step, which must be the row whose context holds only system and
  user items. A trajectory whose turn-1 steps are not all present from that first step, or whose message comes
  before its last step, is skipped and counted;
* backend -- a replay table: each gold call mapped to the ``function_call_output`` that answered the expert's
  identical call in any later row of the trajectory; a gold call without a recorded output skips the trajectory, one
  answered differently in two places is listed under ``backend.conflicts`` (the pool refuses the task);
* later user turns are NOT used: the user simulator wrote them in reply to the expert's message, which the teacher's
  own reply does not reproduce;
* environment -- the connected component of the dataset's tool identities that the task's tools fall in (tools
  co-offered anywhere in the dataset are one environment), so the pool isolates whole environments by split;
* category -- one message step: no_tool; one call step whose arguments are structured (a nested object or array, an
  enum, a number or boolean, or an optional parameter supplied): schema_arguments; one other call step:
  tool_selection; two or more call steps: multi_turn (sequential calls reading tool results). No Pivot row has
  parallel calls or a recorded tool error;
* verification -- protocol, schema validity, recorded execution and step gold; no task-success verifier;
* single-step gold tool calls (Conversational only; maintainer 2026-11-04, a $0 implementation) -- a trajectory
  skipped ONLY because its first gold call has no recorded output (every other rule above applied) becomes, when
  that call's arguments are structured, a ``single_step_gold_tool_call@v1`` task (`tool_episodes.SINGLE_STEP`): the
  same context, gold = that first call alone, no backend, one turn, category ``schema_arguments``; its own task id
  (variant ``turn1-step1``). Recorded execution and task success are unverified (None). A first call that is not
  structured would be a tool_selection decision: outside the approved scope, counted and not converted. A
  trajectory whose first call is answered but a later one is not stays skipped.

The audit records how many trajectories each rule kept or skipped, the categories, the environments, and the
recorded Qwen3-235B per-step pass rates (Conversational only) as a difficulty reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.data.tool_calls import canonical_call  # noqa: E402
from stages.teacher_modes import tool_episodes as TE  # noqa: E402

DATASETS = {
    "function_calling": ("nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
                         "0ac34ec81bb6976d3746407df306a57ce48f0bd8"),
    "conversational": ("nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
                       "bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e"),
}
CONTEXT_ITEMS = ("system", "user")


def position(row: dict) -> tuple[int, int]:
    info = row.get("info") or row.get("meta_info") or {}
    return int(info["turn"]), int(info["step"])


def kind_of(item: dict) -> str:
    return str(item.get("type") or item.get("role"))


def envelope(tool: dict) -> dict:
    """A Responses-API tool as the chat template's ``{"type": "function", "function": {...}}`` envelope."""
    f = {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
    return {"type": "function", "function": f}


def parse_action(action: dict) -> dict | None:
    """A call step's ``{"name", "arguments"}``; None for a message step; raises ValueError for an unreadable call."""
    if action.get("type") == "message":
        return None
    if action.get("type") != "function_call":
        raise ValueError(f"action type {action.get('type')!r}")
    args = action.get("arguments")
    args = json.loads(args) if isinstance(args, str) else args
    if not isinstance(args, dict) or not isinstance(action.get("name"), str):
        raise ValueError("a call without a name or object arguments")
    return {"name": action["name"], "arguments": args}


def structured(call: dict, tools: list[dict]) -> bool:
    """Whether a call's arguments need more than plain strings for required parameters."""
    schema = next((t["function"].get("parameters") or {} for t in tools if t["function"]["name"] == call["name"]), {})
    props, required = schema.get("properties") or {}, set(schema.get("required") or [])
    for k, v in call["arguments"].items():
        p = props.get(k) or {}
        if isinstance(v, (dict, list)) or "enum" in p or k not in required:
            return True
        if isinstance(v, (bool, int, float)) or p.get("type") in ("number", "integer", "boolean"):
            return True
    return False


def read(path: Path) -> tuple[dict, dict, dict, dict]:
    """One pass: per trajectory, its turn-1 steps, its step-1 context, every recorded call's outputs, pass rates."""
    steps, first, outputs, rates = defaultdict(dict), {}, defaultdict(lambda: defaultdict(set)), defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tid = r["trajectory_id"]
            turn, step = position(r)
            items = r["responses_create_params"]["input"]
            calls = {}
            for it in items:
                if kind_of(it) == "function_call":
                    try:
                        args = json.loads(it["arguments"]) if isinstance(it["arguments"], str) else it["arguments"]
                        calls[it["call_id"]] = canonical_call({"name": it["name"], "arguments": args})
                    except (ValueError, KeyError, TypeError):
                        continue
                elif kind_of(it) == "function_call_output" and it.get("call_id") in calls:
                    outputs[tid][calls[it["call_id"]]].add(str(it.get("output")))
            if turn != 1:
                continue
            steps[tid][step] = r["expected_action"]
            if "pass_rate" in r:
                rates[tid][step] = r["pass_rate"]
            if step in (0, 1) and all(kind_of(it) in CONTEXT_ITEMS for it in items):
                first[tid] = {"items": items, "tools": r["responses_create_params"]["tools"], "step": step}
    return steps, first, outputs, rates


def environments(tasks: list[dict]) -> dict[str, str]:
    """task id -> the connected component of tool identities its tools belong to (union-find over co-offering)."""
    from stages.teacher_modes.tool_pool import tool_identities
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ids = {t["task_id"]: [i for _, i in tool_identities(t)] for t in tasks}
    for xs in ids.values():
        for x in xs[1:]:
            a, b = find(xs[0]), find(x)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {tid: (hashlib.sha256(find(xs[0]).encode()).hexdigest()[:16] if xs else "no-tools")
            for tid, xs in ids.items()}


def convert(path: Path, which: str) -> tuple[list[dict], dict]:
    dataset, revision = DATASETS[which]
    steps, first, outputs, rates = read(Path(path))
    skipped, tasks = Counter(), []
    for tid in sorted(steps, key=str):
        got = steps[tid]
        if tid not in first or first[tid]["step"] != min(got):
            skipped["no first-step row (system and user only) for turn 1"] += 1
            continue
        base, top = first[tid]["step"], max(got)
        if sorted(got) != list(range(base, top + 1)):
            skipped["turn-1 steps not contiguous from the first step"] += 1
            continue
        items = first[tid]["items"]
        n = top - base + 1
        try:
            gold = [parse_action(got[k]) for k in range(base, top + 1)]
        except (ValueError, TypeError) as exc:
            skipped[f"an unreadable gold action ({exc})"] += 1
            continue
        if any(g is None for g in gold[:-1]):
            skipped["a message before the last step"] += 1
            continue
        table, conflicts, missing = {}, [], False
        for g in gold:
            if g is None:
                continue
            key = canonical_call(g)
            outs = outputs[tid].get(key)
            if not outs:
                missing = True
                break
            if len(outs) > 1:
                conflicts.append({"call": key, "responses": len(outs)})
            else:
                table[key] = next(iter(outs))
        tools = [envelope(t) for t in first[tid]["tools"]]
        system = next((str(it.get("content") or "") for it in items if kind_of(it) == "system"), "")
        users = [str(it.get("content") or "") for it in items if kind_of(it) == "user"]
        if missing:
            first_unanswered = gold[0] is not None and not outputs[tid].get(canonical_call(gold[0]))
            if which != "conversational" or not first_unanswered:
                skipped["a gold call without a recorded output"] += 1
            elif not structured(gold[0], tools):
                skipped["first gold call without a recorded output, not structured (a tool_selection decision step: "
                        "outside the approved single-step scope)"] += 1
            else:
                tasks.append(single_step(dataset, revision, which, tid, gold[0], n, rates[tid].get(base), system,
                                         tools, users))
            continue
        calls = [g for g in gold if g is not None]
        ends_on_call = gold[-1] is not None
        gold_steps = [[[g]] if g is not None else [[]] for g in gold] + ([None] if ends_on_call else [])
        category = ("no_tool" if not calls else "multi_turn" if len(calls) >= 2 else
                    "schema_arguments" if structured(calls[0], tools) else "tool_selection")
        task = {"task_id": TE.task_id_of(dataset, revision, str(tid), "turn1"), "category": category,
                "category_detail": None,
                "source": {"dataset": dataset, "revision": revision, "config": which, "row": str(tid),
                           "subset": "turn1", "verification_class": "G2",
                           "recorded": {"steps": n, "calls": len(calls), "ends_on_call": ends_on_call,
                                        "step_pass_rates": [rates[tid].get(k) for k in range(base, top + 1)]
                                        if rates.get(tid) else None}},
                "family": None, "split": None, "environment": None, "system": system or None,
                "tools": tools, "messages": [{"role": "user", "content": u} for u in users], "later_user_turns": [],
                "backend": {"kind": "replay", "table": table, "conflicts": conflicts},
                "gold_steps": gold_steps, "final_check": {"kind": "none"}, "max_turns": len(gold_steps)}
        task["source"]["verification"] = {"levels": TE.verification_levels(task), "step_gold": True,
                                          "task_success_verifier": None}
        tasks.append(task)
    env = environments(tasks)
    for t in tasks:
        t["environment"] = f"pivot-{which}:{env[t['task_id']]}"
    comp = Counter(env.values())
    full = [t for t in tasks if TE.episode_shape(t) == TE.FULL_EPISODE]
    single = [t for t in tasks if TE.episode_shape(t) == TE.SINGLE_STEP]
    rated = [t["source"]["recorded"]["step_pass_rates"] for t in full if t["source"]["recorded"]["step_pass_rates"]]
    prod = sorted(_prod(r) for r in rated if all(x is not None for x in r))
    one = sorted(t["source"]["recorded"]["step_pass_rates"][0] for t in single
                 if t["source"]["recorded"]["step_pass_rates"])
    audit = {"dataset": dataset, "revision": revision, "trajectories_with_turn1_rows": len(steps),
             "kept": len(tasks), "skipped": dict(skipped), "category": dict(Counter(t["category"] for t in tasks)),
             "episode_shapes": dict(Counter(TE.episode_shape(t) for t in tasks)),
             "steps": dict(sorted(Counter(t["source"]["recorded"]["steps"] for t in full).items())),
             "ends_on_call": sum(t["source"]["recorded"]["ends_on_call"] for t in full),
             "tasks_with_conflicts": sum(bool(t["backend"]["conflicts"]) for t in full),
             "single_step_qwen3_235b_first_step_pass_rate": (
                 {"tasks": len(one), "median": one[len(one) // 2], "mean": round(sum(one) / len(one), 3)}
                 if one else None),
             "environments": {"count": len(comp), "largest_tasks": comp.most_common(1)[0][1] if comp else 0,
                              "largest_share": round(comp.most_common(1)[0][1] / len(tasks), 4) if tasks else 0},
             "qwen3_235b_episode_pass_rate_product": (
                 {"tasks": len(prod), "median": prod[len(prod) // 2], "p25": prod[len(prod) // 4],
                  "p75": prod[3 * len(prod) // 4]} if prod else None),
             "_": "steps, ends_on_call, conflicts and the episode pass-rate product count the full episodes"}
    return tasks, audit


def single_step(dataset: str, revision: str, which: str, tid, call: dict, steps: int, rate, system: str,
                tools: list[dict], users: list[str]) -> dict:
    """A `tool_episodes.SINGLE_STEP` task: the first gold call alone, verified exactly, never executed."""
    task = {"task_id": TE.task_id_of(dataset, revision, str(tid), "turn1-step1"), "category": "schema_arguments",
            "category_detail": None, "episode_shape": TE.SINGLE_STEP,
            "source": {"dataset": dataset, "revision": revision, "config": which, "row": str(tid),
                       "subset": "turn1-step1", "verification_class": "G2",
                       "recorded": {"steps": 1, "calls": 1, "ends_on_call": True, "trajectory_turn1_steps": steps,
                                    "first_call_recorded_output": False,
                                    "step_pass_rates": [rate] if rate is not None else None}},
            "family": None, "split": None, "environment": None, "system": system or None,
            "tools": tools, "messages": [{"role": "user", "content": u} for u in users], "later_user_turns": [],
            "backend": {"kind": "none"}, "gold_steps": [[[call]]], "final_check": {"kind": "none"}, "max_turns": 1}
    task["source"]["verification"] = {"levels": TE.verification_levels(task), "step_gold": True,
                                      "recorded_execution": None, "task_success_verifier": None}
    bad = TE.task_problems({**task, "split": "training"})
    if bad:
        raise AssertionError(f"{task['task_id']}: {bad}")
    return task


def _prod(xs) -> float:
    out = 1.0
    for x in xs:
        out *= float(x)
    return round(out, 6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--which", required=True, choices=sorted(DATASETS))
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default=None)
    a = ap.parse_args(argv)
    tasks, audit = convert(Path(a.jsonl), a.which)
    with open(a.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    if a.audit:
        Path(a.audit).write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
