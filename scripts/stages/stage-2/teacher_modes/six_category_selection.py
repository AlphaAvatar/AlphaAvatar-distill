#!/usr/bin/env python3
"""A1-6cat's and A2-6cat's five existing base-corpus types, re-selected at 1/6 each. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.six_category_selection \\
        --v1-records <v1 scr>/store/generation --v1-identity <id> --records <supplement scr>/store/generation \\
        --identity <id> --write logs/stages/stage-2/teacher_modes/data/six_category_selection.json

six_category_design.json :: base_category: competition math, GSM8K, RAG reading, multi-hop reading and code each
take 1/6 of the base category's 2,960,507 supervised tokens (493,417.83), re-selected from the verified records A1
and A2 were admitted from, with no generation; tool calling, the sixth type, waits for its pool and its teacher-native
generation. The fluid category (2,962,514) is selected exactly as before.

It runs token_admission.py's frozen machinery unchanged -- the same admission verdicts, code-stability and isolation
rechecks, reuse-first supply order and largest-remainder interleave -- on a design whose only change is the base
category's total: 5/6 of 2,960,507, so its five types, at their registered equal shares, land on 1/6 of the full
total each. The tool type's 493,417.83 is recorded as PENDING, never filled from another type.

No pack is written: the documents' shuffle covers the whole training set, so the pack is built once the tool turns
exist. Writes the selected task ids per arm and their accounting. Reads no student result and no evaluation item.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.teacher_modes import build_supplement_tasks as BST  # noqa: E402
from stages.teacher_modes import token_admission as TA  # noqa: E402

SIX = REPO / "configs/stages/stage-2/teacher_modes/six_category_design.json"
SCHEMA = "aadistill.stage2.teacher_modes.six_category_selection/v1"
BASE_TOTAL = 2960507
TYPES = 6


def five_type_design() -> dict:
    d = copy.deepcopy(BST.design())
    types = d["categories"]["base_counterpart"]["types"]
    selected = {k: t for k, t in types.items() if t.get("cells")}
    #: the token-matched design also lists tool_calling at share 0 with no cell (it was left out); quotas() skips it
    if sorted(selected) != ["code", "competition_math", "gsm8k", "multihop_reading", "rag_reading"] or \
            any(abs(t["share"] - 0.2) > 1e-12 for t in selected.values()) or \
            any(t.get("share", 0) for k, t in types.items() if k not in selected):
        raise SystemExit("the token-matched design's base types are not the five equal-share types this assumes")
    d["scale"]["categories"]["base_counterpart"] = BASE_TOTAL * (TYPES - 1) / TYPES
    d["scale"]["per_mode_total"] = d["scale"]["categories"]["base_counterpart"] + d["scale"]["categories"]["fluid"]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1-records", required=True)
    ap.add_argument("--v1-identity", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    d = five_type_design()          #: checked before the admission pass, which takes about an hour
    got = TA.load(Path(a.v1_records), a.v1_identity, Path(a.records), a.identity)
    dec = TA.decide(got, design=d)
    acc = TA.accounting(got, dec, None)
    short = {m: {cat: s["shortfall"] for cat, s in dec["selection"][m].items() if s["shortfall"]}
             for m in TA.MODES}
    per_type = {}
    for arm, row in acc.items():
        cells = row["by_cell"]
        per_type[arm] = {
            "competition_math": cells.get("supporting.math_new/math_train", 0),
            "gsm8k": cells.get("supporting.math_gsm8k/gsm8k_train", 0),
            "rag_reading": cells.get("supporting.reading_verified/squad_v2_train_answerable", 0),
            "multihop_reading": cells.get("supporting.reading_verified/hotpot_qa_train", 0),
            "code": cells.get("supporting.code_algorithmic/mbpp_train", 0)
            + cells.get("supporting.code_algorithmic/reasoning_core", 0),
            "tool_calling": "PENDING: the canonical pool and its teacher-native generation (packed_round_design.json)"}
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "terminal": "FIVE_TYPES_SELECTED" if not any(short.values()) else "SHORTFALL",
           "six_category_design_sha256": A.sha256_file(SIX),
           "token_matched_design_sha256": A.sha256_file(BST.DESIGN),
           "records": {"v1": {"dir": a.v1_records, "identity": a.v1_identity},
                       "supplement": {"dir": a.records, "identity": a.identity}},
           "base_category_target": BASE_TOTAL, "per_type_target": BASE_TOTAL / TYPES,
           "five_type_target": BASE_TOTAL * (TYPES - 1) / TYPES, "fluid_target": d["scale"]["categories"]["fluid"],
           "quotas": dec["quotas"], "mbpp_cap": dec["mbpp_cap"], "shortfall": short,
           "per_type_supervised_tokens": per_type,
           "arms": {arm.replace("A1", "A1-6cat").replace("A2", "A2-6cat"): {
               k: row[k] for k in ("mode", "tasks", "documents", "supervised_tokens", "reasoning_tokens",
                                   "answer_tokens", "prompt_context_tokens", "by_category", "by_axis", "by_cell",
                                   "by_source")} for arm, row in acc.items()},
           "selection": {m: {cat: [t for t, *_ in s["tasks"]] for cat, s in dec["selection"][m].items()}
                         for m in TA.MODES},
           "authorizes": "nothing"}
    Path(a.write).parent.mkdir(parents=True, exist_ok=True)
    Path(a.write).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"terminal": doc["terminal"], "shortfall": short, "per_type": per_type,
                      "arms": {k: {x: v[x] for x in ("tasks", "documents", "supervised_tokens")}
                               for k, v in doc["arms"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
