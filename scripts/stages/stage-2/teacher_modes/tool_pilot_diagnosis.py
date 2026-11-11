#!/usr/bin/env python3
"""Why the tool pilot's episodes were refused: causes kept apart, their overlap, traceable cases. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot_diagnosis \\
        --session <secured pilot directory> --write <run>/evidence/refusal_diagnosis.json

Diagnostic only (maintainer 2026-11-04, review of the pilot): every sealed episode is re-verified with the pod's own
check (`tool_episodes.verify`, `tool_verification@v2`) and its v2 verdict is kept as it was; nothing here admits or
refuses anything. A refused episode's problems are mapped to causes, several per episode when they co-occur:

* ``model_protocol`` -- the model's own output broke the protocol: a stop other than its end of turn, a tool-call
  block that does not parse, the think rule, text after the last call;
* ``schema_invalid`` -- a call names an unoffered tool or its arguments fail the tool's JSON Schema;
* ``replay_no_response`` -- a call the recorded-response table has no answer for; the environment then ends the
  episode on that call, so its "ends on a call" and an unmet target-tool check are CONSEQUENCES (counted under
  ``replay_consequence``), not further model problems;
* ``ends_on_answered_call`` -- the episode ran out of turns still calling, every call answered;
* ``gold_<kind>`` -- the first gold turn that disagrees with the source's recorded call (G2): ``other_tool``,
  ``no_call``, ``value`` (a structured value differs) or ``free_text`` (only string arguments over 30 characters
  differ; whether they mean the same is reviewed case by case, `FREE_TEXT_REVIEW`, never inferred from length);
* ``target_tools_not_called`` -- the final check failed with no replay miss to explain it.

`replay_misses` classifies each unanswered call against the recorded calls of its tool; `cases` gives traceable
representatives (source dataset, revision, row) of the no-call and other-tool refusals with the context the teacher
saw; `difficulty` sets admission beside the source's recorded per-step pass rate of Qwen3-235B (Pivot only).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

from stages.teacher_modes import tool_episodes as TE  # noqa: E402
from stages.teacher_modes import tool_pilot as TPL  # noqa: E402

SCHEMA = "aadistill.stage2.teacher_modes.tool_pilot_refusal_diagnosis/v2"
#: the agent's case-by-case review of the free-text-only gold disagreements (criteria inside); read, never generated
FREE_TEXT_REVIEW = REPO / "logs/stages/stage-2/teacher_modes/audit/tool_pilot_free_text_review.json"
FREE_TEXT_CHARS = 30
MODEL_PROTOCOL = re.compile(r"stopped by|not JSON|no string name|unclosed|without its|arguments are|think block|"
                            r"in a non-thinking mode|text after its last call|has exactly one turn")


def records(session: Path, mode: str, k: int) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(session).glob(f"pilot_{mode}_r1_pass{k}.part*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def calls_of(turn) -> list:
    from aadistill.data.tool_calls import parse_tool_calls
    p = TE.TOOL_PROTOCOL
    return parse_tool_calls(str(turn.get("raw") or ""), open_tag=p["open_tag"], close_tag=p["close_tag"])[0]


def gold_kind(task, turns) -> tuple[str | None, list]:
    """The first gold turn's disagreement and, for ``value`` / ``free_text``, the differing arguments."""
    for k, gold in enumerate(task.get("gold_steps") or []):
        if gold is None:
            continue
        calls = calls_of(turns[k]) if k < len(turns) else []
        got = sorted(((c.name, dict(c.arguments)) for c in calls), key=lambda x: (x[0], json.dumps(x[1], sort_keys=True)))
        if not calls:
            return ("no_call" if any(gold) else None), []
        best = None
        for option in gold:
            want = sorted(((g["name"], dict(g.get("arguments") or {})) for g in option),
                          key=lambda x: (x[0], json.dumps(x[1], sort_keys=True)))
            if [n for n, _ in want] != [n for n, _ in got]:
                continue
            diffs = [(n, a, ga.get(a), xa.get(a)) for (n, ga), (_, xa) in zip(want, got)
                     for a in sorted(set(ga) | set(xa)) if ga.get(a) != xa.get(a)]
            if not diffs:
                return None, []
            if best is None or len(diffs) < len(best):
                best = diffs
        if best is None:
            return "other_tool", []
        free = all(isinstance(g, str) and isinstance(x, str) and len(g) > FREE_TEXT_CHARS for _, _, g, x in best)
        return ("free_text" if free else "value"), best
    return None, []


def causes(task, turns, verdict) -> list[str]:
    out, probs = set(), verdict.problems
    if any(MODEL_PROTOCOL.search(p) for p in probs):
        out.add("model_protocol")
    if verdict.checks.get("schema_validity") is False:
        out.add("schema_invalid")
    miss = any("did not answer every call" in p for p in probs)
    if miss:
        out.add("replay_no_response")
        if any("ends on a call" in p for p in probs) or verdict.checks.get("target_tools_called") is False:
            out.add("replay_consequence")
    elif any("ends on a call" in p for p in probs):
        out.add("ends_on_answered_call")
    if verdict.checks.get("step_gold") is False:
        kind, _ = gold_kind(task, turns)
        out.add(f"gold_{kind or 'other'}")
    if verdict.checks.get("target_tools_called") is False and not miss:
        out.add("target_tools_not_called")
    if any("decision-step episode ends" in p for p in probs) and "gold_no_call" not in out:
        out.add("gold_no_call")
    return sorted(out) or ["unclassified"]


def replay_misses(task, turns) -> Counter:
    c = Counter()
    if (task.get("backend") or {}).get("kind") != "replay":
        return c
    table = [json.loads(k) for k in task["backend"]["table"]]
    for t in turns:
        resp = list(t.get("responses") or [])
        for i, call in enumerate(calls_of(t)):
            if i < len(resp) and resp[i] is not None:
                continue
            same = [k["arguments"] for k in table if k["name"] == call.name]
            if not same:
                c["tool not among the recorded calls"] += 1
                continue
            xa = dict(call.arguments)
            if any(not (set(a) - set(xa)) and (set(xa) - set(a)) and all(a[q] == xa[q] for q in a) for a in same):
                c["an extra argument, the recorded ones equal"] += 1
            elif any(not (set(xa) - set(a)) and (set(a) - set(xa)) and all(a[q] == xa[q] for q in xa) for a in same):
                c["an argument omitted, the rest equal"] += 1
            elif any(set(a) == set(xa) and all((a[q] == xa[q]) or (isinstance(a[q], (int, float)) and
                                                isinstance(xa[q], (int, float)) and float(a[q]) == float(xa[q]))
                                               for q in a) for a in same):
                c["numerically equal values only"] += 1
            else:
                c["different values"] += 1
    return c


def _context(task, turns, mode) -> dict:
    s = task["source"]
    raw = str((turns or [{}])[0].get("raw") or "")
    return {"mode": mode, "task_id": task["task_id"],
            "source": f"{s['dataset']}@{s['revision'][:8]} row {s.get('row')} {s.get('subset') or ''}".strip(),
            "recorded_step_pass_rates": (s.get("recorded") or {}).get("step_pass_rates"),
            "policy_requires_verification": bool(re.search(r"(?i)verif|authenticat", task.get("system") or "")),
            "user": " | ".join(m["content"].strip() for m in task["messages"] if m["role"] == "user")[:500],
            "gold": task.get("gold_steps"), "teacher_calls": [c.as_dict() for c in calls_of(turns[0])] if turns else [],
            "teacher_answer": raw.split("</think>")[-1].strip()[:500]}


def diagnose(session: Path, plan: Path = TPL.PLAN, tasks: Path = TPL.TASKS, *, cases_per_kind: int = 4) -> dict:
    doc, by_id = TPL.load(plan, tasks)
    review = json.loads(FREE_TEXT_REVIEW.read_text()) if FREE_TEXT_REVIEW.is_file() else {"cases": []}
    labels = {(c["mode"], c["task_id"]): c for c in review.get("cases", [])}
    out = {"schema": SCHEMA, "session": str(session), "verification_version": TE.VERIFICATION_VERSION,
           "pilot_plan_sha256": TPL.sha(plan), "modes": {}, "free_text_review": FREE_TEXT_REVIEW.relative_to(REPO).as_posix(),
           "cases": {"no_call": [], "other_tool": []}}
    for mode in sorted(doc["modes"]):
        thinking = mode == "H-T"
        by = defaultdict(lambda: {"sealed": 0, "admitted": 0, "refused": 0, "causes": Counter(), "sole": Counter(),
                                  "combinations": Counter(), "replay_misses": Counter(), "free_text": Counter(),
                                  "target_unmet": Counter()})
        diff = defaultdict(lambda: [0, 0])
        for k in (0, 1):
            for tid, r in records(session, mode, k).items():
                task, turns = by_id[tid], r.get("turns") or []
                key = f"candidate_{k}|{task['source']['dataset']}|{TE.episode_shape(task)}"
                row = by[key]
                row["sealed"] += 1
                v = TE.verify(task, turns, thinking=thinking)
                pr = ((task["source"].get("recorded") or {}).get("step_pass_rates") or [None])[0]
                if k == 0 and pr is not None:
                    b = f"{min(int(pr * 4), 3) * 0.25:.2f}-{min(int(pr * 4), 3) * 0.25 + 0.25:.2f}"
                    diff[f"{TE.episode_shape(task)}|{b}"][0] += 1
                    diff[f"{TE.episode_shape(task)}|{b}"][1] += bool(v.admitted)
                if v.admitted:
                    row["admitted"] += 1
                    continue
                row["refused"] += 1
                cs = causes(task, turns, v)
                row["causes"].update(cs)
                row["combinations"]["+".join(cs)] += 1
                if len([c for c in cs if c != "replay_consequence"]) == 1:
                    row["sole"][next(c for c in cs if c != "replay_consequence")] += 1
                row["replay_misses"].update(replay_misses(task, turns))
                if "target_tools_not_called" in cs:
                    called = [c.name for t in turns for c in calls_of(t)]
                    row["target_unmet"]["no call at all" if not called else "other tools only"] += 1
                if "gold_free_text" in cs:
                    lab = labels.get((mode, tid))
                    row["free_text"][lab["judgment"] if lab else "unreviewed"] += 1
                if k == 0:
                    for kind in ("no_call", "other_tool"):
                        if f"gold_{kind}" in cs and sum(c["mode"] == mode for c in out["cases"][kind]) < cases_per_kind:
                            out["cases"][kind].append(_context(task, turns, mode))
        out["modes"][mode] = {
            "by_candidate_source_shape": {k: {**{x: v[x] for x in ("sealed", "admitted", "refused")},
                                              **{x: dict(v[x].most_common()) for x in
                                                 ("causes", "sole", "combinations", "replay_misses", "free_text",
                                                  "target_unmet")}}
                                          for k, v in sorted(by.items())},
            "admission_by_recorded_step_pass_rate": {k: {"tasks": n, "admitted_first_candidate": a,
                                                         "rate": round(a / n, 4) if n else None}
                                                     for k, (n, a) in sorted(diff.items())}}
    out["reading"] = {
        "replay": "an unanswered call ends the episode on it (ToolEnv: off-script -> done); the recorded table answers "
                  "only the expert's exact calls, so a different but legitimate call is unexecuted, never executed "
                  "from another call's response",
        "gold": "G2 agreement with ONE recorded expert action; the source's own Qwen3-235B pass rates on the same "
                "steps (difficulty) are beside the teacher's admission",
        "verifier": "the v2 verdicts are unchanged; this record adds causes only",
        "adaptation": "Pivot single-step tasks keep the original step-1 context (the system policy and the first "
                      "user message, tools unedited); the representative cases show the context the teacher saw"}
    out["authorizes"] = "nothing"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    doc = diagnose(Path(a.session))
    Path(a.write).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({m: {k: {"refused": v["refused"], "causes": v["causes"]} for k, v in d["by_candidate_source_shape"].items()}
                      for m, d in doc["modes"].items()}, indent=1, ensure_ascii=False)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
