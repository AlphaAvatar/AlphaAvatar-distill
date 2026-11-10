#!/usr/bin/env python3
"""A-MIX-6cat-sr and A1/A2-6cat-sr on eval-set-1's capability units and legacy_150: correct and usable side by side.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.mixed_fullset \
        --evidence <the run's extracted evidence archive> --out <json>

$0, read-only, beside `mixed_analysis` (fluid) and never inside it. As `reasoning_data.h1pa_fullset`, over each
unit's formal rows: correct is the registered formal score (A1/A2 read from their committed, rescored reports); usable
is `aadistill.evaluation.usable_rollout` on every stored reply under the record's own think state (the hybrid
template self-opens in thinking mode and is preclosed in non-thinking mode). Paired sets: pairs correct and pairs
both-usable. instruction / short_realtime have no registered correctness: format_ok stands in, labelled. knowledge is
outside the formal core; legacy_150 is in-sample. No pooled score.
"""
import argparse, collections, glob, json
from pathlib import Path
from aadistill.evaluation import usable_rollout as UR
from shared import eval_sets as ES

CAP = "stage3_capability_846"
def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
items = {u: {r["id"]: r for r in ES.items("eval-set-1", CAP, u)} for u in
         ("math_verified", "gsm8k", "multihop", "rag", "answerability_paired", "safety_paired", "knowledge", "eval_behavior_v0")}
items["legacy"] = {r["id"]: r for r in ES.items("eval-set-1", "stage3_legacy_150", "items")}

def report(arm, seed, ev):
    if arm.startswith("A-MIX"):
        return json.loads((ev / "formal_report.json").read_text())
    return json.loads(Path(sorted(glob.glob(f"logs/stages/stage-2/teacher_modes/runs/*/evidence/units/{arm}-s{seed}/evaluation/formal_report.json"))[-1]).read_text())

def usable_ts(ev, gen, its):
    formal = {r["id"]: r["formal"] for r in jl(ev / f"rows/{gen}.rows.jsonl")}
    out = {}
    for rec in jl(ev / f"generations/{gen}.jsonl"):
        if formal.get(rec["id"]):
            c = UR.components(rec, think_preopened=bool(rec.get("think_preopened", True)),
                              tools_offered=bool(its[rec["id"]].get("tools")), think_state=rec.get("think_state"))
            out[rec["id"]] = all(c.values())
    return out

def unit(arm, seed, ev):
    U = report(arm, seed, ev)["units"]; o = {}
    for k in ("gsm8k", "math_verified", "multihop", "rag", "knowledge"):
        u = U[f"{CAP}/{k}"]; us = usable_ts(ev, f"{CAP}__{k}", items[k])
        assert len(us) == u["n_items"], (arm, k)
        o[k] = (round(u["headline_value"] * u["n_items"]), sum(us.values()), u["n_items"])
    for k in ("answerability_paired", "safety_paired"):
        m = U[f"{CAP}/{k}"]["metrics"]; us = usable_ts(ev, f"{CAP}__{k}", items[k])
        pairs = collections.defaultdict(list)
        for i, ok in us.items(): pairs[items[k][i]["pair_id"]].append(ok)
        pc = m["pair_correct"]
        o[k] = (round(pc * m["pairs"]) if pc <= 1 else pc, sum(all(v) for v in pairs.values()), m["pairs"])
    beh = {x["id"]: x for x in jl(ev / f"rows/{CAP}__eval_behavior_v0.rows.jsonl") if x["formal"]}
    us = usable_ts(ev, f"{CAP}__eval_behavior_v0", items["eval_behavior_v0"])
    eb = U[f"{CAP}/eval_behavior_v0"]["metrics"]
    tc = [i for i in beh if beh[i]["group"] == "tool_calling"]
    o["tool_calling"] = (round(eb["by_group"]["tool_calling"]["tool_call_exact_match"] * len(tc)), sum(us[i] for i in tc), len(tc))
    for g in ("instruction", "short_realtime"):
        ids = [i for i in beh if beh[i]["group"] == g]
        o[g] = ("fmt", sum(beh[i]["format_ok"] for i in ids), sum(us[i] for i in ids), len(ids))
    o["behaviour_all"] = ("term", sum(beh[i]["terminated"] for i in beh), sum(us.values()), len(beh))
    lg = U["stage3_legacy_150/items"]; lus = usable_ts(ev, "stage3_legacy_150__items", items["legacy"])
    o["legacy_150"] = (lg["correct"], sum(lus.values()), lg["n_items"])
    return o

COLS = (("mixed_thinking", "A-MIX-6cat-sr", "evaluation_thinking"), ("A1-6cat-sr", "A1-6cat-sr", "evaluation"),
        ("mixed_non_thinking", "A-MIX-6cat-sr", "evaluation_non_thinking"), ("A2-6cat-sr", "A2-6cat-sr", "evaluation"))
SEEDS = (20260726, 20260801)
FIELDS = {"fmt": ("format_ok", "usable", "n"), "term": ("terminated", "usable", "n")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {lab: {str(s): {} for s in SEEDS} for lab, _, _ in COLS}
    for lab, arm, sub in COLS:
        for s in SEEDS:
            for k, v in unit(arm, s, Path(a.evidence) / f"{arm}-s{s}" / sub).items():
                names = FIELDS[v[0]] if isinstance(v[0], str) else ("correct", "usable", "n")
                res[lab][str(s)][k] = dict(zip(names, v[1:] if isinstance(v[0], str) else v))
    doc = {"schema": "aadistill.stage2.teacher_modes.mixed_fullset/v1", "set": "eval-set-1 v2, formal rows",
           "pairs": "answerability_paired / safety_paired: correct = pairs correct, usable = pairs both usable, n = pairs",
           "not_core": ["knowledge"], "in_sample": ["legacy_150"], "results": res, "authorizes": "nothing"}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
