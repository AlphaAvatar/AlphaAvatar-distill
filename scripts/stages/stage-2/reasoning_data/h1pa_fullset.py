#!/usr/bin/env python3
"""H1-PA's six units on ALL of eval-set-1, CORRECT and USABLE side by side, at $0.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.reasoning_data.h1pa_fullset \
      --evidence artifacts/stages/stage-2/reasoning_data/h1pa/units/evidence [--json]

Beside `h1pa_analysis` (the preregistered decision), never inside it. Over the
formal items of every unit (the rows files' `formal` flag):

- correct: the registered formal scores (scorer v2) from each unit's
  ``evaluation/formal_report.json``;
- usable: the five-component `aadistill.evaluation.usable_rollout` on every
  stored reply, ``tools_offered`` read from the item. Fluid static items and
  episodes come from `score_battery.py --protocol v2` re-run on the stored
  generations, refused unless it reproduces the pod's per-item correctness;
  episodes report success and all-turns-usable. For the stage3_capability units
  usable is the same standard check applied post hoc (the registered scorer
  reports protocol_valid only).
- instruction / short_realtime have no registered correctness: format_ok is
  reported in its place, labelled. knowledge is outside the formal core;
  legacy_150 is in-sample.

No pooled score is made. Nothing is generated.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import tempfile
from pathlib import Path

from aadistill.evaluation import usable_rollout as UR
from shared import eval_sets as ES
from stages.reasoning_data import assets as A
from stages.reasoning_data import h1pa_driver as D

CAP = "stage3_capability_846"
FLUID = "fluid_reasoning_720__dev_v1"
AXES = ("task_contained_deduction", "inductive_rule_discovery", "compositional_generalization",
        "abstract_analogical")


def jl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def pct(x: int, n: int) -> str:
    return f"{x}/{n} ({100 * x / n:.1f}%)" if n else "-"


def battery(ev: Path, tmp: Path) -> list[dict]:
    """The formal fluid rows, rescored with the battery scorer (which carries usable)."""
    tmp.mkdir(parents=True)
    subprocess.run([str(A.REPO / ".venv/bin/python"), "scripts/stages/stage-0/calibration_volume/score_battery.py",
                    "--tasks", str(ES.unit_path("eval-set-1", "fluid_reasoning_720", "dev_v1")),
                    "--records", str(ev / f"generations/{FLUID}.jsonl"), "--out-dir", str(tmp), "--protocol", "v2"],
                   cwd=A.REPO, env={"PYTHONPATH": "src:scripts"}, capture_output=True, check=True)
    rows = jl(next(tmp.glob("*.rows.jsonl")))
    pod = {r["id"]: r for r in jl(ev / f"rows/{FLUID}.rows.jsonl")}
    if set(pod) != {r["task_id"] for r in rows} or any(
            bool(r["correct"]) != bool(pod[r["task_id"]]["correct"]) for r in rows):
        raise SystemExit(f"{ev}: the battery rescore does not reproduce the pod's fluid rows")
    return [r for r in rows if pod[r["task_id"]]["formal"]]


def usable_rows(ev: Path, gen: str, items: dict) -> dict:
    """{id: usable} over the formal ids of one stored generation file."""
    formal = {r["id"]: r["formal"] for r in jl(ev / f"rows/{gen}.rows.jsonl")}
    return {rec["id"]: UR.usable(rec, think_preopened=bool(rec.get("think_preopened", True)),
                                 tools_offered=bool(items[rec["id"]].get("tools")))
            for rec in jl(ev / f"generations/{gen}.jsonl") if formal.get(rec["id"])}


def unit_table(ev: Path, tmp: Path, items: dict) -> dict:
    U = json.loads((ev / "formal_report.json").read_text())["units"]
    out = {}
    fl = battery(ev, tmp)
    st, ep = [x for x in fl if not x["interactive"]], [x for x in fl if x["interactive"]]
    out["fluid static"] = {"n": len(st), "correct": sum(x["correct"] for x in st), "usable": sum(x["usable"] for x in st)}
    for ax in AXES:
        a = [x for x in st if x["axis"] == ax]
        out[ax] = {"n": len(a), "correct": sum(x["correct"] for x in a), "usable": sum(x["usable"] for x in a)}
    out["interactive episodes"] = {"n": len(ep), "success": sum(x["correct"] for x in ep),
                                   "all_turns_usable": sum(x["all_turns_usable"] for x in ep)}
    for key in ("math_verified", "gsm8k", "multihop", "rag", "knowledge"):
        u = U[f"{CAP}/{key}"]
        us = usable_rows(ev, f"{CAP}__{key}", items[key])
        if len(us) != u["n_items"]:
            raise SystemExit(f"{ev}: {key} has {len(us)} formal replies, the report {u['n_items']}")
        out[key] = {"n": u["n_items"], "correct": round(u["headline_value"] * u["n_items"]),
                    "usable": sum(us.values()), "in_formal_core": u["in_formal_core"]}
        if key == "multihop":
            out[key]["evidence_recall"] = u["metrics"]["evidence_recall"]
        if key == "rag":
            out[key]["evidence_attributed_rate"] = u["metrics"]["evidence_attributed_rate"]
    for key, pos in (("answerability_paired", "answerable_correct"), ("safety_paired", "benign_correct")):
        m = U[f"{CAP}/{key}"]["metrics"]
        us = usable_rows(ev, f"{CAP}__{key}", items[key])
        pairs = collections.defaultdict(list)
        for i, ok in us.items():
            pairs[items[key][i]["pair_id"]].append(ok)
        if len(pairs) != m["pairs"]:
            raise SystemExit(f"{ev}: {key} has {len(pairs)} pairs, the report {m['pairs']}")
        pc = m["pair_correct"]
        out[key] = {"pairs": m["pairs"], "pairs_correct": round(pc * m["pairs"]) if pc <= 1 else pc,
                    "pairs_both_usable": sum(all(v) for v in pairs.values()),
                    "rows": len(us), "rows_usable": sum(us.values()), pos: m[pos]}
    beh = {x["id"]: x for x in jl(ev / f"rows/{CAP}__eval_behavior_v0.rows.jsonl") if x["formal"]}
    us = usable_rows(ev, f"{CAP}__eval_behavior_v0", items["eval_behavior_v0"])
    eb = U[f"{CAP}/eval_behavior_v0"]["metrics"]
    tc = [i for i in beh if beh[i]["group"] == "tool_calling"]
    out["tool_calling"] = {"n": len(tc), "correct": round(eb["by_group"]["tool_calling"]["tool_call_exact_match"] * len(tc)),
                           "usable": sum(us[i] for i in tc)}
    for g in ("instruction", "short_realtime"):
        ids = [i for i in beh if beh[i]["group"] == g]
        out[g] = {"n": len(ids), "correct": None, "format_ok": sum(beh[i]["format_ok"] for i in ids),
                  "usable": sum(us[i] for i in ids)}
    out["protocol integrity"] = {"n": len(beh), "terminated": sum(beh[i]["terminated"] for i in beh),
                                 "usable": sum(us.values())}
    lg = U["stage3_legacy_150/items"]
    lus = usable_rows(ev, "stage3_legacy_150__items", items["legacy"])
    out["legacy_150 (in-sample)"] = {"n": lg["n_items"], "correct": lg["correct"], "usable": sum(lus.values())}
    return out


def render(res: dict) -> str:
    lines = []
    for k in next(iter(res.values())):
        lines.append(k)
        for lab, row in res.items():
            c = row[k]
            if "pairs" in c:
                pos = next(x for x in c if x.endswith("_correct") and x != "pairs_correct")
                s = (f"pairs correct {pct(c['pairs_correct'], c['pairs']):14s} pairs both-usable "
                     f"{pct(c['pairs_both_usable'], c['pairs']):14s} rows usable {pct(c['rows_usable'], c['rows'])}"
                     f"  {pos} {c[pos]:.2f}")
            elif "success" in c:
                s = f"success {pct(c['success'], c['n']):16s} all-turns-usable {pct(c['all_turns_usable'], c['n'])}"
            elif "terminated" in c:
                s = f"terminated {pct(c['terminated'], c['n']):13s} usable {pct(c['usable'], c['n'])}"
            elif c.get("correct") is None:
                s = f"correct n/a (format_ok {pct(c['format_ok'], c['n'])})  usable {pct(c['usable'], c['n'])}"
            else:
                s = f"correct {pct(c['correct'], c['n']):16s} usable {pct(c['usable'], c['n'])}"
            lines.append(f"    {lab:24s} {s}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    items = {u: {r["id"]: r for r in ES.items("eval-set-1", CAP, u)} for u in
             ("math_verified", "gsm8k", "multihop", "rag", "answerability_paired", "safety_paired", "knowledge",
              "eval_behavior_v0")}
    items["legacy"] = {r["id"]: r for r in ES.items("eval-set-1", "stage3_legacy_150", "items")}
    with tempfile.TemporaryDirectory() as tmp:
        res = {u["unit"]: unit_table(Path(args.evidence) / u["unit"] / "evaluation", Path(tmp) / u["unit"], items)
               for u in D.load_plan()["units"]}
    print(json.dumps(res, indent=1) if args.json else render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
