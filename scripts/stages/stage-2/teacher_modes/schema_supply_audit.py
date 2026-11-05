#!/usr/bin/env python3
"""The schema_arguments supply of each teacher, recounted from the sealed sessions. $0, dev box, read-only.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_supply_audit \\
        --write logs/stages/stage-2/teacher_modes/audit/schema_arguments_supply_v3.json

Inputs, all already on the dev box and each verified where it is written:

* the canonical pool ``tool_pool@v1`` -- the schema_arguments tasks of every split, their episode shape, source and
  gold calls (streamed; only this category is kept);
* the six import records (`tool_generation_plan.IMPORTS`) -- per mode, every task a session attempted and each
  candidate's outcome. A task listed there is DONE in that mode; the rest of the category is untried supply;
* the admitted documents and items of each mode (`tool_admission` over every imported store) -- the generated ids of
  every admitted turn, split here into reasoning (through ``</think>``), tool-call (``<tool_call>`` .. ``</tool_call>``)
  and the remaining answer tokens; calibration and state-evaluation scored positions from the items' assistant tags.

Reach from untried tasks is a projection from each mode's own measured yield per shape (admitted per attempted task,
with its 90% Wilson interval) times its measured units per admitted episode; it is never a measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SCHEMA = "aadistill.stage2.teacher_modes.schema_arguments_supply/v3"
CATEGORY = "schema_arguments"
POOL = REPO / "data/stages/stage-2/teacher_modes/tool_pool_v1"
ADMISSION = REPO / "artifacts/stages/stage-2/teacher_modes/tool_admission"
ADMITTED = {"H-T": "hybrid_H-T_cs_final", "H-N": "hybrid_H-N_cs_final", "T2507": "thinking_2507_cs_final"}
#: the registered (original) targets: six_category_design.json's tool type x the category's 15%; the hybrid profiles
#: v2 and state_eval_hybrid@v2 cells (Thinking-2507 has no calibration or state-evaluation split in this pool's plan)
TARGETS = {"H-T": {"training": 74012.6745, "calibration": 630.0, "state_eval": 1800.0},
           "H-N": {"training": 74012.6745, "calibration": 630.0, "state_eval": 1800.0},
           "T2507": {"training": 73888.05}}
THINK_END, CALL_OPEN, CALL_CLOSE = 151668, 151657, 151658
Z90 = 1.6448536269514722


def wilson(k: int, n: int, z: float = Z90) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pool_tasks() -> dict[str, dict]:
    from stages.teacher_modes import tool_pool as TP
    pool = json.loads((POOL / "pool.json").read_text())
    out = {}
    for split, meta in pool["files"].items():
        with open(POOL / meta["file"]) as fh:
            for line in fh:
                if f'"category": "{CATEGORY}"' not in line[:200]:
                    continue
                t = json.loads(line)
                if t["category"] == CATEGORY:
                    out[t["task_id"]] = t
    assert pool["content_sha256"], "pool without content identity"
    return out, pool["content_sha256"], TP


def attempts(pool_sha: str) -> dict[str, dict[str, dict]]:
    from stages.teacher_modes import tool_generation_plan as PLAN
    by_mode: dict[str, dict[str, dict]] = defaultdict(dict)
    for path, rec in PLAN.import_records(pool_sha):
        for mode, block in rec["modes"].items():
            for tid, t in block["tasks"].items():
                if t["category"] != CATEGORY:
                    continue
                assert tid not in by_mode[mode], f"{tid} attempted by two records in {mode}"
                by_mode[mode][tid] = dict(t, record=path.name)
    return by_mode


def call_args(gold_steps) -> list[dict]:
    return [c for step in gold_steps or [] if step for alt in step[:1] for c in (alt or [])]


def depth(v) -> int:
    if isinstance(v, dict):
        return 1 + max((depth(x) for x in v.values()), default=0)
    if isinstance(v, list):
        return 1 + max((depth(x) for x in v), default=0)
    return 0


def gold_profile(task: dict) -> dict:
    """What the gold calls of a task ask for, read against the offered tool's JSON Schema."""
    tools = {t["function"]["name"]: t["function"].get("parameters") or {} for t in task["tools"]}
    calls = call_args(task.get("gold_steps"))
    prof = Counter(calls=len(calls))
    for c in calls:
        params = tools.get(c["name"], {})
        props, required = params.get("properties") or {}, set(params.get("required") or [])
        args = c.get("arguments") or {}
        prof["arguments"] += len(args)
        prof["optional_arguments"] += sum(1 for a in args if a not in required)
        prof["nested_or_array_arguments"] += sum(1 for v in args.values() if isinstance(v, (dict, list)))
        prof["enum_arguments"] += sum(1 for a in args if "enum" in (props.get(a) or {}))
        prof["numeric_arguments"] += sum(1 for v in args.values() if isinstance(v, (int, float))
                                         and not isinstance(v, bool))
        prof["max_depth"] = max(prof["max_depth"], depth(args))
    return dict(prof)


def split_output(ids: list[int]) -> dict:
    reasoning = ids.index(THINK_END) + 1 if THINK_END in ids else 0
    tool = 0
    inside = False
    for t in ids[reasoning:]:
        if t == CALL_OPEN:
            inside = True
        if inside:
            tool += 1
        if t == CALL_CLOSE:
            inside = False
    return {"supervised": len(ids), "reasoning": reasoning, "tool_call": tool,
            "answer": len(ids) - reasoning - tool, "calls": ids.count(CALL_OPEN)}


def admitted_units(mode: str, tasks: dict) -> dict[str, dict]:
    """Per admitted schema_arguments episode: its turns' generated-token split or its scored positions."""
    out: dict[str, dict] = defaultdict(lambda: Counter())
    root = ADMISSION / ADMITTED[mode]
    with open(root / "tool_documents.jsonl") as fh:
        for line in fh:
            if f'"category": "{CATEGORY}"' not in line[:400]:
                continue
            d = json.loads(line)
            ep = out[d["task_id"]]
            ep.update(split_output(d["output_ids"]))
            ep["turns"] += 1
    with open(root / "tool_items.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("category") != CATEGORY:
                continue
            ep = out[d["task_id"]]
            ep["scored_positions"] += len(d["tags"]["assistant"])
            ep["calls"] += len(d["tags"].get("tool_call_open", []))
            ep["turns"] += 1
    assert set(out) <= set(tasks), "an admitted episode outside the pool's schema_arguments tasks"
    return out


def describe(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1), "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
            "max": xs[-1], "sum": sum(xs)}


def audit() -> dict:
    tasks, pool_sha, _ = pool_tasks()
    tried = attempts(pool_sha)
    report = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "pool": {"id": "tool_pool@v1", "content_sha256": pool_sha},
              "supply": {}, "modes": {}, "overlap_training_admitted": {}, "authorizes": "nothing"}
    supply = defaultdict(Counter)
    for t in tasks.values():
        supply[t["split"]][t.get("episode_shape", "full_episode@v1")] += 1
        supply[t["split"]]["source:" + t["source"]["dataset"]] += 1
    report["supply"] = {s: dict(c) for s, c in sorted(supply.items())}
    admitted_sets = {}
    for mode in ("H-T", "H-N", "T2507"):
        units = admitted_units(mode, tasks)
        cells = {}
        for split in ("training", "validation", "calibration", "state_eval", "reserve"):
            cell = {}
            for shape in ("full_episode@v1", "single_step_gold_tool_call@v1"):
                pool_ids = [i for i, t in tasks.items()
                            if t["split"] == split and t.get("episode_shape", "full_episode@v1") == shape]
                if not pool_ids:
                    continue
                att = [i for i in pool_ids if i in tried[mode]]
                cands = sum(len(tried[mode][i]["candidates"]) for i in att)
                adm = [i for i in att if any(c["outcome"] == "admitted" for c in tried[mode][i]["candidates"])]
                assert set(adm) == {i for i in att if i in units}, f"{mode}/{split}/{shape}: census vs documents"
                lo, hi = wilson(len(adm), len(att))
                ep = [units[i] for i in adm]
                unit = "scored_positions" if split in ("calibration", "state_eval") else "supervised"
                row = {"supply": len(pool_ids), "attempted": len(att), "untried": len(pool_ids) - len(att),
                       "candidates_generated": cands, "admitted": len(adm),
                       "admitted_per_attempted": round(len(adm) / len(att), 4) if att else None,
                       "wilson90": [round(lo, 4), round(hi, 4)] if att else None,
                       "pass_per_candidate": round(len(adm) / cands, 4) if cands else None,
                       "units": unit, "units_per_admitted": describe([e[unit] for e in ep]),
                       "turns_per_admitted": describe([e["turns"] for e in ep]),
                       "calls_per_admitted": describe([e["calls"] for e in ep])}
                if unit == "supervised":
                    tot = Counter()
                    for e in ep:
                        tot.update({k: e[k] for k in ("supervised", "reasoning", "answer", "tool_call")})
                    row["token_split"] = dict(tot)
                profs = [gold_profile(tasks[i]) for i in adm]
                row["admitted_gold_profile"] = {
                    "calls": sum(p.get("calls", 0) for p in profs),
                    "arguments_per_call": round(sum(p.get("arguments", 0) for p in profs) /
                                                max(1, sum(p.get("calls", 0) for p in profs)), 2),
                    **{f"tasks_with_{k}": sum(1 for p in profs if p.get(k)) for k in
                       ("optional_arguments", "nested_or_array_arguments", "enum_arguments", "numeric_arguments")},
                    "max_depth_histogram": dict(sorted(Counter(p.get("max_depth", 0) for p in profs).items()))}
                row["admitted_environments"] = len({tasks[i]["environment"] for i in adm})
                row["admitted_subsets"] = dict(Counter(tasks[i]["source"]["subset"] for i in adm))
                if att and split in TARGETS[mode] and row["untried"]:
                    mean = statistics.fmean([e[unit] for e in ep]) if ep else 0.0
                    row["untried_reach_projection"] = {
                        "low": round(row["untried"] * lo * mean), "expected": round(row["untried"] * len(adm) /
                                                                                     len(att) * mean),
                        "high": round(row["untried"] * hi * mean)}
                cell[shape] = row
                if split == "training":
                    admitted_sets.setdefault(mode, set()).update(adm)
            total = sum(units[i][("scored_positions" if split in ("calibration", "state_eval") else "supervised")]
                        for i in tasks if tasks[i]["split"] == split and i in units)
            target = TARGETS[mode].get(split)
            cells[split] = {"by_shape": cell, "admitted_units": total, "target": target,
                            "gap": round(max(0.0, target - total), 1) if target is not None else None}
        report["modes"][mode] = cells
    modes = list(admitted_sets)
    for a in range(len(modes)):
        for b in range(a + 1, len(modes)):
            x, y = admitted_sets[modes[a]], admitted_sets[modes[b]]
            report["overlap_training_admitted"][f"{modes[a]}&{modes[b]}"] = {
                "both": len(x & y), modes[a]: len(x), modes[b]: len(y)}
    attempted_sets = {m: {i for i in tried[m] if tasks[i]["split"] == "training"} for m in tried}
    report["attempted_training_tasks"] = {m: len(s) for m, s in attempted_sets.items()}
    report["attempted_training_in_common"] = len(set.intersection(*attempted_sets.values()))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", type=Path)
    args = ap.parse_args(argv)
    report = audit()
    text = json.dumps(report, indent=1, sort_keys=False) + "\n"
    if args.write:
        args.write.write_text(text)
    print(text if not args.write else f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
