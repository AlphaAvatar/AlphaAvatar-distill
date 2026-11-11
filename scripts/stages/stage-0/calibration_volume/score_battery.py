#!/usr/bin/env python3
"""Score `run_battery.py` records against their task set, per sample and per axis.

    PYTHONPATH=src:scripts python scripts/stages/stage-0/calibration_volume/score_battery.py \
        --tasks <set.jsonl> --records <records.jsonl> --out <rows.jsonl>

Static task: `correct = usable_rollout AND verified answer`. The answer is the
last `\\boxed{}` of the final answer (text after `</think>`), canonicalized by
the task's answer type and compared with the canonicalized gold; this is the
D-series rule (correct only inside a usable rollout), so a correct-looking
answer in a generation that never terminated is not credited.

Interactive task: `correct = the environment's own success`, re-derived here by
REPLAYING the recorded replies through a fresh environment, so the score never
trusts the pod's bookkeeping. Each turn's usable components are reported too.

Pure CPU; runs on the pod and off it, and re-scoring stored records reproduces
the rows byte for byte.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.data.verify import boxed_answer  # noqa: E402
from aadistill.evaluation import usable_rollout  # noqa: E402
from aadistill.evaluation.behavior import record_think_state  # noqa: E402
from stages.calibration_volume.tasks.common import CANON_BY_PROTOCOL, gold_key_for  # noqa: E402
from stages.calibration_volume.tasks.interactive import make_env  # noqa: E402

SCHEMA = "aadistill.stage0.battery_rows/v1"
#: Scoring protocols. v1 is frozen: every historical score was computed with it.
#: v2 (2026-10-19) canonicalizes static answers with `tasks.common.CANON_V2`.
#: An EPISODE is always replayed under the action protocol it was GENERATED
#: with (the record says which; absent means v1), whatever the static protocol:
#: the observations in a transcript came from that environment.
SCORER_IDS = {"v1": "stage0.calibration_volume.score_battery@v1",
              "v2": "stage0.calibration_volume.score_battery@v2"}
SCORER_ID = SCORER_IDS["v1"]


def turn_components(turn: dict) -> dict:
    rec = {k: turn[k] for k in ("raw", "natural_termination", "degeneration_triggered",
                                "context_limit_reached")}
    comp = usable_rollout.components(rec, think_state=record_think_state(turn))
    comp["usable"] = all(comp.values())
    return comp


def score_static(task: dict, rec: dict, protocol: str = "v1") -> dict:
    turn = rec["turns"][-1] if rec["turns"] else None
    if turn is None:
        return {"correct": False, "usable": False, "answer_correct": False,
                "extracted": None, "reason": rec.get("end") or "no_turn"}
    comp = turn_components(turn)
    boxed = boxed_answer(turn["answer"] or "")
    got = CANON_BY_PROTOCOL[protocol][task["answer_type"]](boxed) if boxed is not None else None
    ok = got is not None and got == gold_key_for(task, protocol)
    return {**comp, "answer_correct": ok, "correct": bool(comp["usable"] and ok),
            "extracted": got, "boxed_present": boxed is not None,
            "reason": "ok" if comp["usable"] and ok else
                      ("unusable" if not comp["usable"] else
                       "no_boxed" if boxed is None else "wrong_answer"),
            "generated_tokens": turn["generated_tokens"],
            "stop_reason": turn["stop_reason"]}


def score_episode(task: dict, rec: dict) -> dict:
    env = make_env(task["env"], action_protocol=rec.get("action_protocol", "v1"))
    comps = [turn_components(t) for t in rec["turns"]]
    for t in rec["turns"]:
        if t["stop_reason"] != "eos" or env.done:
            break
        env.step(t["answer"])
    result = env.result()
    pod = rec.get("env_result") or {}
    return {"correct": bool(result["success"]), "replay_matches_pod": all(
        result.get(k) == pod.get(k) for k in result),
            "turns": len(rec["turns"]),
            "usable_turn_rate": round(sum(c["usable"] for c in comps) / len(comps), 4)
            if comps else 0.0,
            "all_turns_usable": bool(comps) and all(c["usable"] for c in comps),
            "env": result, "end": rec.get("end"),
            "generated_tokens": sum(t["generated_tokens"] for t in rec["turns"])}


def score(tasks: dict[str, dict], records: list[dict], protocol: str = "v1") -> list[dict]:
    rows = []
    for rec in sorted(records, key=lambda r: r["task_id"]):
        task = tasks[rec["task_id"]]
        body = (score_episode(task, rec) if task["interactive"]
                else score_static(task, rec, protocol))
        rows.append({"task_id": rec["task_id"], "label": rec["label"],
                     "generator_id": task["generator_id"], "family_id": task["family_id"],
                     "axis": task["axis"], "difficulty": task["difficulty"],
                     "interactive": task["interactive"], **body})
    return rows


def summarize(rows: list[dict]) -> dict:
    def cell(sub):
        n = len(sub)
        c = sum(r["correct"] for r in sub)
        out = {"n": n, "correct": c, "rate": round(c / n, 4) if n else None}
        stat = [r for r in sub if not r["interactive"]]
        if stat:
            out["usable_rate"] = round(sum(r["usable"] for r in stat) / len(stat), 4)
            out["answer_correct_rate"] = round(
                sum(r["answer_correct"] for r in stat) / len(stat), 4)
        return out

    by = collections.defaultdict(list)
    for r in rows:
        by[("axis", r["axis"])].append(r)
        by[("axis_difficulty", f"{r['axis']}|{r['difficulty']}")].append(r)
        by[("generator", r["generator_id"])].append(r)
        by[("generator_difficulty", f"{r['generator_id']}|{r['difficulty']}")].append(r)
        by[("family", f"{r['generator_id']}/{r['family_id']}")].append(r)
    out = {"overall": cell(rows)}
    for (kind, key), sub in sorted(by.items()):
        out.setdefault(kind, {})[key] = cell(sub)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--records", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", default="v1", choices=tuple(SCORER_IDS))
    args = ap.parse_args(argv)
    tasks = {t["task_id"]: t for t in
             (json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip())}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.records:
        records = [json.loads(l) for l in Path(path).read_text().split("\n") if l.strip()]
        rows = score(tasks, records, args.protocol)
        label = records[0]["label"] if records else Path(path).stem
        body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (out_dir / f"{label}.rows.jsonl").write_text(body)
        summary = {"schema": SCHEMA, "scorer": SCORER_IDS[args.protocol], "label": label,
                   "records": str(path),
                   "records_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   "tasks_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
                   "rows_sha256": hashlib.sha256(body.encode()).hexdigest(),
                   "summary": summarize(rows)}
        (out_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        s = summary["summary"]["overall"]
        print(f"{label}: {s['correct']}/{s['n']} correct "
              f"({s['rate']}); by axis " + ", ".join(
                  f"{k.split('_')[0]}={v['rate']}" for k, v in summary["summary"]["axis"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
