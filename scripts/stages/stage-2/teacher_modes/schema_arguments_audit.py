#!/usr/bin/env python3
"""The non-thinking schema_arguments SHORTFALL: option 1 (call-only gold episodes from Pivot Conversational) against
option 2 (WorkBench generated instances), counted at $0. Nothing enters the pool and nothing is generated.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_arguments_audit \\
        --pivot artifacts/stages/stage-2/teacher_modes/tool_sources/nvidia__Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1@bfc7a4dd/train.jsonl \\
        --workbench artifacts/stages/stage-2/teacher_modes/tool_sources/olly-styles__WorkBench@49c7dfd0 \\
        --write logs/stages/stage-2/teacher_modes/audit/schema_arguments_options.json

* option 1 -- the Conversational trajectories `tool_sources_pivot.convert` skips ONLY because their first gold call
  has no recorded output (every other rule of the converter applied), split under the frozen environment rule
  (connected components recomputed over them and the kept tasks together), categorized by
  `tool_sources_pivot.structured`, and the non-thinking length of the gold call turn in the hybrid's tokenizer;
* option 2 -- WorkBench's task files at the pinned commit: templates, write actions per task, and the templates whose
  ground truth is one call needing no lookup (read from their generator logic: ``WORKBENCH_SINGLE_CALL``).

Reach is a PRIOR, not a measurement: candidates x the pool screens' pass share x (1 - (1 - p)^2) (two candidates) x
tokens per item, at three levels of the per-attempt pass rate p; the L40S pilot measures p.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SHORTFALL = REPO / "logs/stages/stage-2/teacher_modes/audit/schema_arguments_shortfall.json"
POOL_AUDIT = REPO / "logs/stages/stage-2/teacher_modes/audit/tool_pool_dry_run_v3.json"
TOKENIZER = REPO / "artifacts/stages/stage-2/teacher_modes/tokenizers/qwen3_4b_1cfa9a72"
WORKBENCH_COMMIT = "49c7dfd00c03d384ec59ea57374f50b766aa5613"
#: the two templates whose single write needs no read first (create_plot's arguments all come from the request);
#: every other template looks up an id, an address or a value before it writes (their generator logic functions)
WORKBENCH_SINGLE_CALL = ("Can you make a {plot_type} chart of {natural_language_metric} since {natural_language_date}?",
                         "Plot the distribution of {natural_language_metric} on {natural_language_date}")
#: per-attempt pass rate of the teacher on a call-only item: low, the Qwen3-235B reference recorded on these rows
#: (rounded), high -- priors until the pilot
PASS = {"pessimistic": 0.15, "expected": 0.32, "optimistic": 0.5}
PIVOT_ENV = "environment:pivot-conversational:"


def call_only(path: Path) -> dict:
    from transformers import AutoTokenizer

    from aadistill.data.tool_calls import canonical_call
    from stages.teacher_modes import tool_pool as TP
    from stages.teacher_modes import tool_sources_pivot as PV
    steps, first, outputs, rates = PV.read(path)
    kept, cand, other = [], [], Counter()
    for tid in sorted(steps, key=str):
        got = steps[tid]
        if tid not in first or first[tid]["step"] != min(got):
            continue
        base, top = first[tid]["step"], max(got)
        if sorted(got) != list(range(base, top + 1)):
            continue
        try:
            gold = [PV.parse_action(got[k]) for k in range(base, top + 1)]
        except (ValueError, TypeError):
            continue
        if any(g is None for g in gold[:-1]):
            continue
        tools = [PV.envelope(t) for t in first[tid]["tools"]]
        t = {"task_id": str(tid), "tools": tools}
        if all(outputs[tid].get(canonical_call(g)) for g in gold if g is not None):
            kept.append(t)
        elif outputs[tid].get(canonical_call(gold[0])):
            other["first call answered, a later one not (a longer prefix episode, not counted)"] += 1
        else:
            cand.append({**t, "call": gold[0], "structured": PV.structured(gold[0], tools), "rate": rates[tid].get(base)})
    env, env_kept = PV.environments(kept + cand), PV.environments(kept)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    by, lens, rate = Counter(), {"schema_arguments": [], "tool_selection": []}, []
    for t in cand:
        cat = "schema_arguments" if t["structured"] else "tool_selection"
        by[f"{TP.split_of(PIVOT_ENV + env[t['task_id']])}/{cat}"] += 1
        body = "<tool_call>\n" + json.dumps(t["call"], ensure_ascii=False) + "\n</tool_call>"
        lens[cat].append(len(tok(body)["input_ids"]) + 1)                     #: + <|im_end|>
        if cat == "schema_arguments" and t["rate"] is not None:
            rate.append(t["rate"])
    moved = sum(TP.split_of(PIVOT_ENV + env[t["task_id"]]) != TP.split_of(PIVOT_ENV + env_kept[t["task_id"]])
                for t in kept)
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else None  # noqa: E731
    return {"source": str(path), "kept_by_the_converter": len(kept), "call_only_candidates": len(cand),
            "not_counted": dict(other), "by_split_category": dict(sorted(by.items())),
            "kept_tasks_whose_split_moves": moved,
            "call_turn_tokens_non_thinking": {c: {"n": len(v), "p25": q(v, .25), "median": q(v, .5), "p75": q(v, .75),
                                                  "mean": round(sum(v) / len(v), 1)} for c, v in lens.items() if v},
            "qwen3_235b_first_step_pass_rate": {"n": len(rate), "median": q(rate, .5),
                                                "mean": round(sum(rate) / len(rate), 3) if rate else None}}


def workbench(d: Path) -> dict:
    out, single = {}, Counter()
    for f in sorted(Path(d).glob("*_tasks_and_outcomes.csv")):
        rows = list(csv.DictReader(f.open()))
        writes, funcs, base = Counter(), Counter(), set()
        for r in rows:
            acts = ast.literal_eval(r["outcome"])
            writes[len(acts)] += 1
            base.add(r["base_template"])
            funcs.update(re.match(r"([\w.]+)\.func\(", a).group(1) for a in acts)
            if r["base_template"] in WORKBENCH_SINGLE_CALL:
                single[r["base_template"]] += 1
        out[f.name.replace("_tasks_and_outcomes.csv", "")] = {
            "tasks": len(rows), "base_templates": len(base), "write_actions_per_task": dict(sorted(writes.items())),
            "write_calls": dict(funcs.most_common())}
    import hashlib
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(d).iterdir()) if f.is_file()}
    return {"commit": WORKBENCH_COMMIT, "files_sha256": files, "license": Path(d, "LICENSE").read_text().splitlines()[0],
            "domains": out, "tasks": sum(v["tasks"] for v in out.values()),
            "base_templates": sum(v["base_templates"] for v in out.values()),
            "single_call_templates": {k: single[k] for k in WORKBENCH_SINGLE_CALL}}


def reach(n: float, screens: float, tokens: float) -> dict:
    return {lvl: round(n * screens * (1 - (1 - p) ** 2) * tokens) for lvl, p in PASS.items()}


def build(pivot: Path, wb: Path) -> dict:
    sf = json.loads(SHORTFALL.read_text())["reach_under_the_plan_priors"]
    pool = json.loads(POOL_AUDIT.read_text())["admitted"]["by_source"]
    admitted = sum(pool["nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"].values())
    screens = admitted / 9051                     #: the converter's kept Conversational tasks the dry pool admitted
    c = call_only(pivot)
    tokens = c["call_turn_tokens_non_thinking"]["schema_arguments"]["mean"]
    add = {s: reach(c["by_split_category"].get(f"{s}/schema_arguments", 0), screens, tokens)
           for s in ("training", "state_eval")}
    now = {"training": sf["H-N/training"], "state_eval": sf["H-N/state_eval"]}
    combined = {s: {lvl: now[s]["reachable"][lvl] + add[s][lvl] for lvl in PASS} for s in add}
    return {"schema": "aadistill.stage2.teacher_modes.schema_arguments_options/v1",
            "question": "maintainer 2026-11-04 (review of fc6fa1d6), item 5: option 1 against option 2 for the "
                        "non-thinking schema_arguments SHORTFALL -- feasibility, verification strength, volume, cost "
                        "and protocol impact; neither is implemented and nothing is generated before the decision",
            "command": "PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_arguments_audit "
                       "--pivot <Pivot Conversational train.jsonl> --workbench <WorkBench files> --write <this>",
            "pass_rate_priors": PASS, "pool_screens_pass_share": round(screens, 4),
            "option_1_call_only": {**c, "reach_H-N_added": add, "reach_H-N_with_today": combined,
                                   "targets": {s: now[s]["target"] for s in now}},
            "option_2_workbench": workbench(wb), "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pivot", required=True)
    ap.add_argument("--workbench", required=True)
    ap.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    doc = build(Path(a.pivot), Path(a.workbench))
    old = json.loads(Path(a.write).read_text()) if Path(a.write).is_file() else {}
    doc["analysis"] = old.get("analysis")                    #: the written comparison, kept across regeneration
    Path(a.write).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc["option_1_call_only"][k] for k in ("by_split_category", "reach_H-N_with_today")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
