#!/usr/bin/env python3
"""H1's result at $0, from the four formal units' evidence: the frozen decision rule, then every capability apart.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1_analysis --evidence <dir>

``<dir>/<unit>/`` holds each unit's extracted evidence (``report/report.json``,
``evaluation/rows/*.rows.jsonl``, ``evaluation/formal_report.json``). Nothing is
rescored and nothing is re-generated: the formal rows the evaluation wrote are
read as they are.

1. IDENTITY. Every unit's report must be formal, name this plan's arm, seed,
   recipe identity, pack and initialization, and its checkpoint must be the
   persisted one; a unit that is missing or not its frozen self refuses.
2. THE DECISION (`h1_statistics.decide`, frozen 2026-10-22): one row per (arm,
   seed, item) on the FORMAL static fluid items of the four static axes;
   cluster = the item's rule family. Interactive episodes are exploratory and
   reported beside it, never inside it.
3. EVERY CAPABILITY APART: each fluid axis by arm and seed; the two generator
   tiers (evaluation-only generators: unseen FORMAT; shared generators'
   held-out families: unseen RULE); the interactive episodes; every supporting
   and guard-rail unit's headline from its formal report; and the rollout
   protocol unit. No pooled score exists and none is made here.
4. ACCOUNTING beside it: U, E and C per arm and seed from `plans/h1_accounting.json`
   (C is reported, not matched).

Writes ``logs/stages/stage-2/reasoning_data/analyses/h1_result.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from shared import eval_sets as ES
from stages.reasoning_data import assets as A
from stages.reasoning_data import h1_statistics as HS
from stages.reasoning_data import unit_driver as UD

SET_ID = "eval-set-1"
FLUID = ("fluid_reasoning_720", "dev_v1")
OUT = A.LOG_DIR / "analyses/h1_result.json"
STATIC_AXES = ("task_contained_deduction", "inductive_rule_discovery", "compositional_generalization",
               "abstract_analogical")


def fluid_meta() -> tuple[dict, set]:
    items = {}
    for line in ES.unit_path(SET_ID, *FLUID).read_text().splitlines():
        it = json.loads(line)
        items[it["task_id"]] = it
    side = json.loads((ES.unit_path(SET_ID, *FLUID).parent / f"{FLUID[1]}.manifest.json").read_text())
    return items, set(side["eval_only_generators"])


def unit_evidence(plan: dict, unit: dict, d: Path) -> dict:
    rep = json.loads((d / "report/report.json").read_text())
    sched = UD.schedule_for(plan, unit["arm"])
    probe = UD.derive_recipe(plan["recipe"], run_name="-", purpose="-", out_dir="-", data_dir="-",
                             student_path="-", seed=unit["seed"], sched=sched,
                             overrides=plan["config"]["recipe"]["overrides"])
    want = UD.unit_inputs(plan, unit, sched, UD.recipe_identity(probe), validation=False)
    if rep.get("formal") is not True or rep.get("inputs") != json.loads(json.dumps(want)):
        raise SystemExit(f"{unit['unit']}: its report is not this frozen unit's ({d / 'report/report.json'})")
    rows = {}
    for f in sorted((d / "evaluation/rows").glob("*.rows.jsonl")):
        rows[f.name.removesuffix(".rows.jsonl")] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return {"report": rep, "rows": rows,
            "formal_report": json.loads((d / "evaluation/formal_report.json").read_text())}


def rate(rows: list[dict]) -> dict:
    n = len(rows)
    c = sum(bool(r.get("correct")) for r in rows)
    return {"n": n, "correct": c, "rate": round(c / n, 4) if n else None}


def analyse(evidence: Path) -> dict:
    plan = UD.load_plan()
    items, eval_only = fluid_meta()
    units = {u["unit"]: {**u, **unit_evidence(plan, u, evidence / u["unit"])} for u in plan["units"]}
    key = "__".join(FLUID)
    decision_rows, by = [], defaultdict(list)
    for name, u in units.items():
        for r in u["rows"][key]:
            it = items[r["id"]]
            if not r.get("formal"):
                continue
            tier = "unseen_format" if it["generator_id"] in eval_only else "unseen_rule"
            by[(u["arm"], u["seed"], it["axis"], tier)].append(r)
            if it.get("interactive"):
                continue
            decision_rows.append({"arm": u["arm"], "seed": u["seed"], "item_id": r["id"],
                                  "cluster": it["family_id"], "axis": it["axis"], "correct": bool(r.get("correct"))})
    decision = HS.decide(decision_rows)
    axes = defaultdict(dict)
    for (arm, seed, axis, tier), rows in sorted(by.items()):
        axes[axis].setdefault(f"{arm}-s{seed}", {})[tier] = rate(rows)
    for axis, cells in axes.items():
        for cell, tiers in cells.items():
            tiers["all"] = rate([r for (a, s, ax, _t), rows in by.items() if ax == axis and f"{a}-s{s}" == cell
                                 for r in rows])
    supporting = {name: {k: v for k, v in (u["formal_report"].get("units") or u["formal_report"]).items()
                         if not k.startswith(FLUID[0])}
                  for name, u in units.items()}
    acc = A.read_json(A.LOG_DIR / "plans/h1_accounting.json")
    return {"schema": "aadistill.stage2.reasoning_data.h1_result/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "question": A.read_json(A.LOG_DIR / "plans/comparison_design.json")["question"],
            "units": {n: {"arm": u["arm"], "seed": u["seed"], "attempts": len(u["report"]["attempts"]),
                          "checkpoint": u["report"]["checkpoint"],
                          "evaluation_descriptor_sha256": u["report"]["evaluation_descriptor_sha256"]}
                      for n, u in units.items()},
            "decision": decision,
            "primary_label": decision["primary"]["label"],
            "by_axis_and_tier": dict(axes),
            "interactive_exploratory": {a: c for a, c in axes.items() if a not in STATIC_AXES},
            "supporting_and_guard_rails": supporting,
            "accounting": {"schedule": acc["schedule"], "exact_per_seed": acc["exact_per_seed"],
                           "primary_basis": acc["primary_basis"],
                           "_": "U is matched within 1%; E follows the passes match_schedule reports; C is NOT matched"},
            "reading": ("the primary result is static correctness B - A on unseen fluid items under the frozen rule; "
                        "lower loss, KL, format or natural termination are never read as reasoning; a floor is "
                        "reported as NOT_RESOLVED/FLOOR, never as equivalence"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args(argv)
    doc = analyse(Path(args.evidence))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    p = doc["decision"]["primary"]
    print(OUT.relative_to(A.REPO), doc["primary_label"], {k: p.get(k) for k in ("estimate", "interval", "flags")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
