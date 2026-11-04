#!/usr/bin/env python3
"""Teacher-native tool-calling episodes: the environment, verification, and what an episode becomes. $0 code.

The protocol is six_category_design.json's (`task_pool.task_record`, `protocol`); this module is its executable
half, shared by every lineage (the hybrid in either mode, Thinking-2507). Nothing here generates or downloads:

* `ToolEnv` -- `run_battery.run_episodes`' environment for one `tool_pool@v1` task: it parses the turn's calls,
  asks the task's deterministic backend for each response (a replay table; never a live API or a simulating LLM),
  and hands back the assistant turn as a runtime re-renders it (structured ``tool_calls``) with the tool messages.
  A turn without a call ends the user turn: the next scripted user message follows, or the episode ends. An
  off-script call (no recorded response) ends the episode, and the verifier refuses it.
* `verify` -- `aadistill.data.tool_calls.verify_episode`'s levels under the mode's think rule (thinking mode closes
  its think block before any call; non-thinking mode never generates ``</think>``): protocol, schema validity and
  recorded execution on every episode; step gold where the task carries gold calls; the task's final check at its
  declared level (`FINAL_CHECK_LEVELS`). `verification_levels` is what a task CAN verify, registered in the pool:
  a check that the target tools were called is tool selection, and no Toucan task verifies task success.
* `documents` -- a verified episode's turns as token-in/token-out training documents, one per turn
  (`token_documents.document_from_turn`): the engine's own prompt ids as context, the generated ids as targets.
* `items` -- a verified episode's turns as calibration / state-evaluation items in `freeze_state_eval`'s form,
  with the tool-calling critical tags ``tool_call_open`` / ``tool_call_close`` beside ``eos``, ``final_answer`` and
  (thinking mode) ``think_close``.

Episode shapes (six_category_design.json :: protocol.episode_shapes; a task names its own, ``full_episode@v1`` when
it names none):

* ``full_episode@v1`` -- the episode runs until a turn without a call (or its turn limit); every call is answered by
  the backend, and the episode must end on an answer.
* ``single_step_gold_tool_call@v1`` (maintainer 2026-11-04, review of 647eea0e: a $0 implementation only) -- ONE
  turn from the first real user input: the teacher's own tool call, which ends the sample at its natural
  termination. The upstream gold call (name + arguments) only verifies it -- exact match after canonicalization,
  plus schema validity and the protocol -- and is never a target. No tool response or final answer exists or is
  required; nothing executes the call, so recorded execution, tool selection and task success stay unverified
  (``None``), never passed. A decision sample, not a completed task; registered for ``schema_arguments`` only, never
  for multi-turn, error-recovery or MCP-agent tasks (`task_problems` refuses it there).

The delimiters and token ids are the teacher family's, read from `TOOL_PROTOCOL` (both teachers share the Qwen3
vocabulary); the core module takes them as arguments.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]

from aadistill.data.tool_calls import (  # noqa: E402
    ReplayBackend, parse_tool_calls, strip_tool_calls, verify_episode,
)

#: The Qwen3 family's tool-call protocol, as its chat template writes it (checked against the real template by the
#: tests): calls are ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` after the think block.
TOOL_PROTOCOL = {"open_tag": "<tool_call>", "close_tag": "</tool_call>", "think_close": "</think>",
                 "open_id": 151657, "close_id": 151658, "think_close_id": 151668,
                 "stop_texts": ("<|im_end|>", "<|endoftext|>")}
TASK_FIELDS = ("task_id", "category", "source", "family", "split", "tools", "messages", "backend", "max_turns")
CATEGORIES = ("tool_selection", "schema_arguments", "no_tool", "multiple_parallel", "multi_turn",
              "error_recovery", "mcp_agent")
#: the level each final-check kind verifies: whether the target tools were called (or none was) is tool selection;
#: only an answer check against the task's gold is task success
FINAL_CHECK_LEVELS = {"no_call": "tool_selection", "calls_tools": "tool_selection", "contains_all": "task_success"}
SPLITS = ("training", "validation", "state_eval", "calibration", "reserve")
FULL_EPISODE = "full_episode@v1"
SINGLE_STEP = "single_step_gold_tool_call@v1"
#: the registered episode shapes (six_category_design.json :: protocol.episode_shapes) and the categories each may
#: carry; a single-step decision is one turn
EPISODE_SHAPES = {FULL_EPISODE: {"categories": CATEGORIES, "max_turns": None},
                  SINGLE_STEP: {"categories": ("schema_arguments",), "max_turns": 1}}


def episode_shape(task: Mapping[str, Any]) -> str:
    return task.get("episode_shape") or FULL_EPISODE


class ToolTaskError(ValueError):
    """A task record that the protocol cannot run."""


def task_problems(task: Mapping[str, Any]) -> list[str]:
    """Why ``task`` is not a runnable `tool_pool@v1` record (empty when it is)."""
    bad = [f"missing {k}" for k in TASK_FIELDS if k not in task]
    if bad:
        return bad
    if task["category"] not in CATEGORIES:
        bad.append(f"category {task['category']!r} is not one of {CATEGORIES}")
    if task["split"] not in SPLITS:
        bad.append(f"split {task['split']!r} is not one of {SPLITS}")
    if not task["tools"] and task["category"] != "no_tool":
        bad.append("no tools offered")
    if not any(m.get("role") == "user" for m in task["messages"]):
        bad.append("no user message")
    if any(m.get("role") == "assistant" for m in task["messages"]):
        bad.append("an upstream assistant turn in the context: the teacher continues no other model's turns")
    backend = task["backend"]
    if backend.get("kind") not in ("replay", "none"):
        bad.append(f"backend {backend.get('kind')!r}: only a replay table or none runs in the episode loop")
    if backend.get("kind") == "replay" and not isinstance(backend.get("table"), Mapping):
        bad.append("a replay backend without its table")
    if int(task["max_turns"]) < 1:
        bad.append("max_turns < 1")
    shape = episode_shape(task)
    if shape not in EPISODE_SHAPES:
        bad.append(f"episode shape {shape!r} is not registered ({sorted(EPISODE_SHAPES)})")
    elif shape == SINGLE_STEP:
        bad += [f"{SINGLE_STEP}: {x}" for x in single_step_problems(task)]
    return bad


def single_step_problems(task: Mapping[str, Any]) -> list[str]:
    """Why a task is not a registered single-step gold tool call: one turn, one gold call to an offered tool, no
    backend, no final check, no later user turn, a registered category."""
    bad = []
    if task["category"] not in EPISODE_SHAPES[SINGLE_STEP]["categories"]:
        bad.append(f"category {task['category']!r}: the shape is registered for "
                   f"{EPISODE_SHAPES[SINGLE_STEP]['categories']} only")
    if int(task["max_turns"]) != 1:
        bad.append("max_turns must be 1")
    if task["backend"].get("kind") != "none":
        bad.append("its call is never executed: backend kind must be 'none'")
    if (task.get("final_check") or {}).get("kind", "none") != "none":
        bad.append("no final check: the sample ends on its call")
    if task.get("later_user_turns"):
        bad.append("no later user turn")
    gold = task.get("gold_steps")
    if not (isinstance(gold, list) and len(gold) == 1 and isinstance(gold[0], list) and len(gold[0]) == 1
            and isinstance(gold[0][0], list) and len(gold[0][0]) == 1):
        bad.append("gold_steps must be one step with one allowed set of exactly one call")
    else:
        offered = {str((t.get("function") or t).get("name")) for t in task["tools"]}
        if str(gold[0][0][0].get("name")) not in offered:
            bad.append("the gold call is not to an offered tool")
    return bad


def backend_of(task: Mapping[str, Any]):
    b = task["backend"]
    if b["kind"] == "replay":
        return ReplayBackend(b["table"])
    return lambda call: None


class ToolEnv:
    """One task's environment for `run_episodes` (`env_factory=ToolEnv.factory`)."""

    def __init__(self, task: Mapping[str, Any]):
        bad = task_problems(task)
        if bad:
            raise ToolTaskError(f"{task.get('task_id')}: " + "; ".join(bad))
        self.task = task
        self.backend = backend_of(task)
        self.max_turns = int(task["max_turns"])
        self.later_users = list(task.get("later_user_turns") or [])
        self.done = False
        self.off_script = False

    @staticmethod
    def factory(task: Mapping[str, Any]):
        return ToolEnv(task) if task.get("tools") is not None and "backend" in task else None

    def step(self, answer: str):
        p = TOOL_PROTOCOL
        calls, _ = parse_tool_calls(answer, open_tag=p["open_tag"], close_tag=p["close_tag"])
        if episode_shape(self.task) == SINGLE_STEP:
            #: the sample ends on this turn: its calls are recorded, never executed or answered
            self.done = True
            if not calls:
                return None
            return {"assistant": {"content": strip_tool_calls(answer, open_tag=p["open_tag"], close_tag=p["close_tag"]),
                                  "tool_calls": [{"type": "function", "function": {"name": c.name,
                                                                                   "arguments": dict(c.arguments)}}
                                                 for c in calls]},
                    "messages": [], "tool_calls": [c.as_dict() for c in calls], "responses": []}
        if not calls:
            if self.later_users:
                nxt = self.later_users.pop(0)
                return {"assistant": {"content": answer}, "messages": [{"role": "user", "content": nxt}],
                        "tool_calls": [], "responses": []}
            self.done = True
            return None
        responses = [self.backend(c) for c in calls]
        if any(r is None for r in responses):
            self.off_script, self.done = True, True
        assistant = {"content": strip_tool_calls(answer, open_tag=p["open_tag"], close_tag=p["close_tag"]),
                     "tool_calls": [{"type": "function", "function": {"name": c.name, "arguments": dict(c.arguments)}}
                                    for c in calls]}
        return {"assistant": assistant,
                "messages": [{"role": "tool", "content": r if r is not None else ""} for r in responses],
                "tool_calls": [c.as_dict() for c in calls], "responses": responses}

    def result(self) -> dict:
        return {"off_script": self.off_script, "done": self.done}


def final_check_level(task: Mapping[str, Any]) -> str | None:
    """The level the task's final check verifies (`FINAL_CHECK_LEVELS`), or None without one."""
    kind = (task.get("final_check") or {}).get("kind", "none")
    if kind == "none":
        return None
    if kind not in FINAL_CHECK_LEVELS:
        raise ToolTaskError(f"{task['task_id']}: final check {kind!r} is not implemented")
    return FINAL_CHECK_LEVELS[kind]


def verification_levels(task: Mapping[str, Any]) -> list[str]:
    """The levels an episode of ``task`` can be verified at (`tool_calls.LEVELS` order)."""
    from aadistill.data.tool_calls import LEVELS
    have = {"protocol", "schema_validity"} | ({"recorded_execution"} if episode_shape(task) != SINGLE_STEP else set())
    if any(g is not None for g in task.get("gold_steps") or []):
        have.add("step_gold")
    if final_check_level(task):
        have.add(final_check_level(task))
    return [k for k in LEVELS if k in have]


def final_check(task: Mapping[str, Any]):
    """The task's final check over its turns: none, no_call (every turn answers without a call), calls_tools (every
    target tool called) or contains_all (the last turn's answer contains every listed string)."""
    spec = task.get("final_check") or {"kind": "none"}
    kind = spec.get("kind", "none")
    if kind == "none":
        return None
    if kind == "no_call":
        p = TOOL_PROTOCOL
        return lambda turns: all(not parse_tool_calls(str(t.get("raw") or ""), open_tag=p["open_tag"],
                                                      close_tag=p["close_tag"])[0] for t in turns)
    if kind == "calls_tools":
        #: every listed tool called at least once over the episode (Toucan's target tools): tool selection only,
        #: never task success; a listed name matches a called name equal to it or ending in "-<name>" (MCP tools
        #: carry their server's slug)
        want = [str(x) for x in spec["values"]]
        p = TOOL_PROTOCOL

        def called(turns):
            names = {c.name for t in turns
                     for c in parse_tool_calls(str(t.get("raw") or ""), open_tag=p["open_tag"],
                                               close_tag=p["close_tag"])[0]}
            return all(any(n == w or n.endswith("-" + w) for n in names) for w in want)
        return called
    if kind == "contains_all":
        want = [str(x) for x in spec["values"]]
        return lambda turns: all(w in str(turns[-1].get("answer") or turns[-1].get("raw") or "") for w in want)
    raise ToolTaskError(f"{task['task_id']}: final check {kind!r} is not implemented")


def verify(task: Mapping[str, Any], episode_turns: Sequence[Mapping[str, Any]], *, thinking: bool):
    """Every level on every turn of one episode (`EpisodeVerdict`). A single-step gold tool call is one turn that
    ends on its call (`verify_episode`'s ``unexecuted_final_call``): protocol, schema validity and the exact gold
    call are checked; recorded execution, tool selection and task success stay None."""
    from aadistill.data.tool_calls import EpisodeVerdict
    p = TOOL_PROTOCOL
    fc, level = final_check(task), final_check_level(task)
    single = episode_shape(task) == SINGLE_STEP
    v = verify_episode(episode_turns, tools=task["tools"], open_tag=p["open_tag"], close_tag=p["close_tag"],
                       think_close=p["think_close"], expect_think=thinking,
                       gold_steps=task.get("gold_steps"), final_checks={level: fc} if fc else None,
                       unexecuted_final_call=single)
    if single:
        bad = [] if len(episode_turns) == 1 else [f"{SINGLE_STEP} has exactly one turn, not {len(episode_turns)}"]
        raw = str((episode_turns[-1] if episode_turns else {}).get("raw") or "")
        tail = raw[raw.rfind(p["close_tag"]) + len(p["close_tag"]):] if p["close_tag"] in raw else ""
        for stop in p["stop_texts"]:
            tail = tail.replace(stop, "")
        if tail.strip():                                     #: V1: nothing but whitespace after the last call
            bad.append(f"{SINGLE_STEP}: text after its call ({tail.strip()[:40]!r}); the sample ends on the call")
        if bad:
            return EpisodeVerdict(False, {**v.checks, "protocol": False}, v.problems + bad)
    return v


def documents(task: Mapping[str, Any], episode_turns: Sequence[Mapping[str, Any]], *, mode: str, stop_ids):
    """One token-in/token-out document per turn of a VERIFIED episode (only the generated ids are targets)."""
    from aadistill.data.token_documents import document_from_turn
    return [document_from_turn(t, doc_id=f"{task['task_id']}|{mode}|t{k}", stop_ids=stop_ids,
                               meta={"task_id": task["task_id"], "category": task["category"],
                                     "family": task["family"], "turn": k, "mode": mode,
                                     "episode_shape": episode_shape(task)})
            for k, t in enumerate(episode_turns)]


def item_of_turn(task: Mapping[str, Any], turn: Mapping[str, Any], k: int, *, mode: str, thinking: bool,
                 domain: str, subtype: str) -> dict | None:
    """`freeze_state_eval.item_of`'s item for one turn, with the tool-calling critical tags; None when the turn is
    not naturally terminated or breaks its mode's think rule."""
    p = TOOL_PROTOCOL
    pr, o = list(turn.get("prompt_token_ids") or []), list(turn.get("output_token_ids") or [])
    if turn.get("stop_reason") != "eos" or not pr or not o:
        return None
    base = len(pr) - 1
    tags = {"assistant": [base + j for j in range(len(o))], "eos": [base + len(o) - 1]}
    if thinking:
        if p["think_close_id"] not in o[:-1]:
            return None
        c = o.index(p["think_close_id"])
        tags.update(think_close=[base + c], reasoning=[base + j for j in range(c)],
                    final_answer=[base + j for j in range(c + 1, len(o) - 1)])
    else:
        if p["think_close_id"] in o:
            return None
        tags["final_answer"] = [base + j for j in range(len(o) - 1)]
    tags["tool_call_open"] = [base + j for j, t in enumerate(o) if t == p["open_id"]]
    tags["tool_call_close"] = [base + j for j, t in enumerate(o) if t == p["close_id"]]
    ids = pr + o
    return {"item_id": f"{task['task_id']}|{mode}|t{k}", "task_id": task["task_id"], "turn": k, "ids": ids,
            "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1, "tags": tags, "domain": domain,
            "subtype": f"{subtype}|{mode}", "type": subtype, "mode": mode, "critical_group": mode,
            "source": task["source"], "family_id": task["family"], "category": task["category"],
            "episode_shape": episode_shape(task)}


def items(task: Mapping[str, Any], episode_turns: Sequence[Mapping[str, Any]], *, mode: str, thinking: bool,
          domain: str = "supporting", subtype: str = "tool_calling@v1") -> list[dict]:
    """A VERIFIED episode's turns as items (every turn, or none: an episode is admitted whole)."""
    out = [item_of_turn(task, t, k, mode=mode, thinking=thinking, domain=domain, subtype=subtype)
           for k, t in enumerate(episode_turns)]
    return [] if any(x is None for x in out) else out


def task_id_of(dataset: str, revision: str, row_key: str, variant: str = "") -> str:
    """The design's task id: one per task, the same for every teacher."""
    return "tc1-" + hashlib.sha256(f"{dataset}@{revision}|{row_key}|{variant}".encode()).hexdigest()[:20]


def read_tasks(path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").split("\n") if x.strip()]
