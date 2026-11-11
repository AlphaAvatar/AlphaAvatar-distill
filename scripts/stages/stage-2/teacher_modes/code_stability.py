#!/usr/bin/env python3
"""Code verdicts that move with the string-hash seed: Research A v2's code candidates under several seeds. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.code_stability \\
        --v1-records <v1 scr>/store/generation --v1-identity <v1 study identity> \\
        --records <supplement scr>/store/generation --identity <this study's identity>

`aadistill.data.code_exec` v4 pins the child's hash seed to 0, which made one MBPP candidate's verdict
reproducible (`d934be3a...`: it unpacks ``set(s)`` into two names and passes at seeds 0 and 1, fails at 2 and 3)
without making it right. The maintainer's review (2026-11-01) refuses it and asks for every other code candidate to
be audited for the same instability. Two kinds of code verdict exist in these data:

* MBPP (``verifier`` ``code_asserts``): the candidate's last fenced Python block against the task's asserts, in the
  probed sandbox. `mbpp` re-runs it under every seed of ``HASH_SEEDS`` (and seed 0 twice).
* reasoning_core ``controlled_code_execution``: the prompt is a generated program and the gold is the ``repr`` of
  the value it returns, computed once when the instance was exported (at PYTHONHASHSEED 0). The candidate is
  scored against that gold, so a program whose value depends on the hash seed has no well-defined answer.
  `program` re-runs the prompt's own call under the same seeds and requires the gold every time.

``dynamic_programming`` tasks state their numbers in the prompt and fix ties lexicographically; no Python value of
theirs is hashed, so they are listed and not executed.

`recheck` is the admission's step (token_admission.load): a candidate whose MBPP verdict is unstable is refused and
the task's next candidate is considered, exactly as the frozen verdict would have; a task whose gold is unstable is
refused in both modes. `audit` writes every result (logs/stages/stage-2/teacher_modes/audit/code_stability.json).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

RECORD = REPO / "logs/stages/stage-2/teacher_modes/audit/code_stability.json"
SCHEMA = "aadistill.stage2.teacher_modes.code_stability/v1"
#: seed 0 is the admission's own; seeds 2 and 3 are the ones that exposed `d934be3a...`
HASH_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
PROGRAM_MODULES = ("controlled_code_execution",)
WORKERS = 8
_PROGRAM = re.compile(r"```python\n(.*?)```", re.S)
_CALL = re.compile(r"Call: `([^`]+)`")


def is_mbpp(task: dict) -> bool:
    return task.get("verifier") == "code_asserts"


def is_program(task: dict) -> bool:
    return task.get("source") == "reasoning_core" and task.get("module") in PROGRAM_MODULES


def _prompt(task: dict) -> str:
    return " ".join(m.get("content") or "" for m in task["messages"] if m.get("role") == "user")


def program_parts(task: dict) -> tuple[str, str] | None:
    """(program, call) of a controlled_code_execution prompt, or None when it is not one."""
    text = _prompt(task)
    code, call = _PROGRAM.search(text), _CALL.search(text)
    return (code.group(1), call.group(1)) if code and call else None


def program(task: dict, seeds=HASH_SEEDS) -> dict:
    """Is the task's gold the call's repr under every seed?"""
    from aadistill.data.code_exec import run_asserts_across_seeds
    from stages.reasoning_data import supporting_verify as SV
    parts = program_parts(task)
    if parts is None:
        return {"task_id": task["task_id"], "kind": "program", "stable": False, "passed": False,
                "status": "unparsed", "by_seed": []}
    code, call = parts
    r = run_asserts_across_seeds(code, [f"assert repr({call}) == {str(task['answer'])!r}"], hash_seeds=seeds,
                                 limits=SV.CODE_LIMITS, sandbox=SV._the_sandbox())
    return {"task_id": task["task_id"], "kind": "program", **{k: r[k] for k in ("passed", "stable", "status", "by_seed")}}


def mbpp(task: dict, rec: dict, seeds=HASH_SEEDS) -> dict:
    """The candidate's MBPP verdict under every seed; a record whose turn or code block is missing is not code."""
    from aadistill.data.code_exec import run_asserts_across_seeds
    from stages.reasoning_data import generation_adapter as GA
    from stages.reasoning_data import supporting_verify as SV
    row = {"task_id": task["task_id"], "candidate": rec["candidate_index"], "kind": "mbpp"}
    parts = GA.split_turn(rec["turns"][-1]) if rec["turns"] else None
    code = SV.code_block(parts[1])[0] if parts else None
    if code is None:
        return {**row, "passed": False, "stable": True, "status": "no_code", "by_seed": []}
    r = run_asserts_across_seeds(code, task["test_list"], hash_seeds=seeds, setup=task.get("test_setup_code") or "",
                                 limits=SV.CODE_LIMITS, sandbox=SV._the_sandbox())
    return {**row, **{k: r[k] for k in ("passed", "stable", "status", "by_seed")}}


def _pool(fn, jobs):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(lambda a: fn(*a), jobs))


def recheck(verdicts: dict, tasks: dict, recs: dict, proto: dict, admit) -> dict:
    """The admission's stability step, in place on ``verdicts``: (task, mode) -> refused or moved candidate.

    MBPP: the admitted candidate under every seed; an unstable one is refused and ``admit`` (the frozen verdict)
    runs again on the remaining candidates, whose admitted one is checked the same way. Programs: a task whose gold
    is unstable is refused in both modes. Returns what changed and every check run."""
    checked: dict = {"mbpp": [], "program": []}
    changed: dict = {}
    progs = sorted({tid for (tid, _m), v in verdicts.items() if v["admitted"] and is_program(tasks[tid])})
    bad_gold = set()
    for r in _pool(lambda t: program(tasks[t]), [(t,) for t in progs]):
        checked["program"].append(r)
        if not r["passed"]:
            bad_gold.add(r["task_id"])
    for (tid, mode), v in verdicts.items():
        if tid in bad_gold and v["admitted"]:
            verdicts[(tid, mode)] = {**v, "admitted": False, "docs": [], "unstable_verification": "gold"}
            changed[f"{tid}|{mode}"] = "refused: the gold is not the program's value under every hash seed"
    pending = {k for k, v in verdicts.items() if v["admitted"] and is_mbpp(tasks[k[0]])}
    excluded: dict = {}
    while pending:
        keys = sorted(pending)
        rows = _pool(lambda k: (mbpp(tasks[k[0]], verdicts[k]["record"]), k), [(k,) for k in keys])
        pending = set()
        for r, k in rows:
            checked["mbpp"].append({**r, "mode": k[1]})
            if r["passed"]:
                continue
            excluded.setdefault(k, set()).add(r["candidate"])
            rest = [x for x in recs[k] if x["candidate_index"] not in excluded[k]]
            nv = admit(tasks[k[0]], k[1], rest, proto)
            nv = {**nv, "unstable_verification": "mbpp", "excluded_candidates": sorted(excluded[k])}
            if not nv["admitted"]:
                nv["docs"] = []
            verdicts[k] = nv
            changed[f"{k[0]}|{k[1]}"] = (f"candidate {r['candidate']} refused ({r['status']}); "
                                        + (f"candidate {nv['candidate']} admitted" if nv["admitted"] else "no candidate left"))
            if nv["admitted"]:
                pending.add(k)
    return {"changed": changed, "checked": checked, "hash_seeds": list(HASH_SEEDS)}


def audit(got: dict) -> dict:
    """Every MBPP candidate of both record sets (admitted or not) and every program task of the supply, under every
    seed; the admission's own recheck (`got['code_recheck']`) beside it."""
    tasks, recs = got["tasks"], got["recs"]
    keyed = [(m, tasks[tid], rec) for (tid, m), rs in sorted(recs.items()) if is_mbpp(tasks[tid]) for rec in rs]
    mb = [{**r, "mode": m} for r, (m, _t, _r) in zip(_pool(mbpp, [(t, r) for _m, t, r in keyed]), keyed)]
    prog_ids = sorted({tid for (tid, _m) in recs if is_program(tasks[tid])})
    pr = _pool(lambda t: program(tasks[t]), [(t,) for t in prog_ids])
    dp = sorted({tid for (tid, _m) in recs if tasks[tid].get("source") == "reasoning_core"
                 and tasks[tid].get("module") == "dynamic_programming"})
    return {"mbpp_candidates": mb, "program_tasks": pr, "dynamic_programming_tasks_not_executed": len(dp)}


def _summary(rows: list[dict]) -> dict:
    return {"checked": len(rows), "passed_every_seed": sum(r["passed"] for r in rows),
            "failed_every_seed": sum((not r["passed"]) and r["stable"] for r in rows),
            "unstable": sorted(f"{r['task_id']}|{r.get('mode', '')}|c{r.get('candidate', '')}".rstrip("|c")
                               for r in rows if not r["stable"])}


def task_files(paths) -> list[dict]:
    """Program tasks of a task file (calibration's code prompts, the state evaluation's), for the full search."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                if is_program(t):
                    out.append(t)
    return out


def main(argv=None) -> int:
    from stages.teacher_modes import token_admission as TA
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1-records", required=True)
    ap.add_argument("--v1-identity", required=True)
    ap.add_argument("--records", default=None)
    ap.add_argument("--identity", default=None)
    ap.add_argument("--record", default=str(RECORD))
    a = ap.parse_args(argv)
    got = TA.load(Path(a.v1_records), a.v1_identity, Path(a.records) if a.records else None, a.identity)
    res = audit(got)
    full = task_files([REPO / "data/stages/stage-2/teacher_modes/tasks/calibration_code_v1.jsonl",
                       REPO / "data/stages/stage-2/teacher_modes/tasks/state_eval_hybrid_v1.tasks.jsonl"])
    fs = _pool(lambda t: program(t), [(t,) for t in full])
    from aadistill.data.code_exec import CODE_EXEC_SCHEMA
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "verifier": f"aadistill.data.code_exec.run_asserts_across_seeds ({CODE_EXEC_SCHEMA}), the probed sandbox",
           "hash_seeds": list(HASH_SEEDS), "repeat": "seed 0 runs twice",
           "records": {"v1": a.v1_identity, "supplement": a.identity},
           "summary": {"mbpp_candidates": _summary(res["mbpp_candidates"]),
                       "program_tasks_of_the_training_supply": _summary(res["program_tasks"]),
                       "program_tasks_of_the_full_search_inputs": _summary(fs),
                       "dynamic_programming_tasks_not_executed": res["dynamic_programming_tasks_not_executed"]},
           "admission_recheck": got.get("code_recheck", {}).get("changed"),
           "rows": {"mbpp_candidates": res["mbpp_candidates"], "program_tasks": res["program_tasks"],
                    "full_search_program_tasks": fs},
           "authorizes": "nothing"}
    Path(a.record).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
