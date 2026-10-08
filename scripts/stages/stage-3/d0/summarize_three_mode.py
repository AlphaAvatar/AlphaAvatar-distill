#!/usr/bin/env python
"""Rebuild free/oracle summaries from saved generations (CPU, no regeneration).

The driver invokes the three-mode runner twice per arm -- once in the vLLM venv
for free+oracle, once in the training venv for teacher-forced -- and both wrote
`report.json`, so the second clobbered the first. The *generations* are intact,
and summaries are a pure function of them, so nothing needs re-running.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
from run_three_mode_diagnostic import score, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, type=Path)
ap.add_argument("--sessions", type=Path,
                default=REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl")
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()

# Re-score rather than trust the stored `correct`: the rows were scored at
# generation time, and the scorer has since been corrected (free-form QA must not
# be held to the numeric final-answer-marker rule).
sessions = {}
if a.sessions.is_file():
    for line in a.sessions.open():
        d = json.loads(line)
        sessions[d["id"]] = d
res = {}
for mode in ("free", "oracle"):
    f = a.dir / f"{mode}.generations.jsonl"
    if not f.is_file():
        continue
    rows = [json.loads(l) for l in f.open()]
    for r in rows:
        s_ = sessions.get(r["id"])
        if s_ is not None:
            body = r["raw"].split("<|im_end|>")[0]
            if mode == "free":
                from aadistill.evaluation.behavior import split_generation
                body = split_generation(r["raw"], think_preopened=True)["answer"]
            r["correct"] = score(s_, body.strip())
    entry = {"overall": summarize(rows)}
    by = defaultdict(list)
    for r in rows:
        by[r["data_type"]].append(r)
    entry["by_task"] = {t: summarize(v) for t, v in sorted(by.items())}
    if mode == "oracle":
        sp = defaultdict(list)
        for r in rows:
            if r.get("answer_in_reasoning") is not None:
                sp["answer_literally_in_reasoning" if r["answer_in_reasoning"]
                   else "answer_requires_transformation"].append(r)
        entry["numeric_split"] = {k: summarize(v) for k, v in sp.items()}
    res[mode] = entry
a.out.write_text(json.dumps(res, indent=1))
cols = ("n","correct","protocol_valid","natural_termination","empty_answer",
        "repetition","reopened_think","reasoning_leakage","context_limit",
        "answer_tokens_p50")
for mode, e in res.items():
    print(f"\n-- {mode.upper()} --")
    print(f"{'task':14s} " + " ".join(f"{c[:9]:>10s}" for c in cols))
    print(f"{'OVERALL':14s} " + " ".join(f"{str(e['overall'].get(c)):>10s}" for c in cols))
    for t, v in e["by_task"].items():
        print(f"{t:14s} " + " ".join(f"{str(v.get(c)):>10s}" for c in cols))
    for k, v in e.get("numeric_split", {}).items():
        print(f"   {k:36s} n={v['n']:>4d} correct={v['correct']}")
