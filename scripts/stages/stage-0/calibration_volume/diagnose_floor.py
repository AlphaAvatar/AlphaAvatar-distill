#!/usr/bin/env python3
"""Characterize a floor: is a zero rate format, termination, or capability?

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/diagnose_floor.py \
        --tasks <set.jsonl> --records <label>=<records.jsonl> ... --out <json>

A DIAGNOSTIC, never an endpoint. For each run and axis it separates the ways a
static task can score zero -- the rollout is not usable, the answer is usable
but carries no `\\boxed{}`, or it is boxed and wrong -- and bounds correctness
from ABOVE with a deliberately lenient reading (the last bold span or last line
canonicalizes to the gold answer, or the gold string appears verbatim in a
usable answer). If even the lenient bound is near zero, the zero is not a
format artifact of the strict scorer. Interactive episodes report how they
ended and the rate of turns the environment could not read as an action.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume import score_battery as SB  # noqa: E402
from stages.calibration_volume.tasks.common import CANON, gold_key  # noqa: E402


def lenient(task: dict, answer: str) -> bool:
    g = gold_key(task)
    cands = re.findall(r"\*\*([^*]+)\*\*", answer)[-1:] + [answer.strip().split("\n")[-1]]
    if any(CANON[task["answer_type"]](c) == g for c in cands):
        return True
    norm = lambda s: re.sub(r"\s+", " ", (s or "").lower())  # noqa: E731
    return bool(task["answer"]) and norm(task["answer"]) in norm(answer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--records", nargs="+", required=True, help="label=path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tasks = {t["task_id"]: t for t in (json.loads(l) for l in Path(args.tasks).read_text().splitlines())}
    out = {"schema": "aadistill.stage0.floor_diagnostic/v1", "_not_an_endpoint": True, "runs": {}}
    for spec in args.records:
        label, path = spec.split("=", 1)
        recs = [json.loads(l) for l in Path(path).read_text().splitlines()]
        rows = {r["task_id"]: r for r in SB.score(tasks, recs)}
        by = collections.defaultdict(collections.Counter)
        ends = collections.Counter()
        invalid = turns = 0
        for rec in recs:
            t = tasks[rec["task_id"]]
            r = rows[rec["task_id"]]
            c = by[t["axis"]]
            c["n"] += 1
            if t["interactive"]:
                ends[rec.get("end")] += 1
                env = r.get("env") or {}
                invalid += env.get("invalid_actions", 0)
                turns += max(1, env.get("turns", 0))
                c["success"] += bool(r["correct"])
                continue
            ans = rec["turns"][-1]["answer"] if rec["turns"] else ""
            c["usable"] += bool(r.get("usable"))
            c["boxed"] += bool(r.get("boxed_present"))
            c["strict_correct"] += bool(r["correct"])
            c["lenient_upper_bound"] += bool(r.get("usable")) and lenient(t, ans)
        out["runs"][label] = {
            "records": path,
            "by_axis": {ax: dict(c) for ax, c in sorted(by.items())},
            "interactive": {"ends": dict(ends),
                            "invalid_action_rate": round(invalid / turns, 4) if turns else None},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for label, run in out["runs"].items():
        print(label, {ax: (c.get("strict_correct", c.get("success")), c.get("lenient_upper_bound"),
                           c["n"]) for ax, c in run["by_axis"].items()},
              run["interactive"]["invalid_action_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
