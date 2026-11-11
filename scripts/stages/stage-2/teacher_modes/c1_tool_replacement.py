#!/usr/bin/env python3
"""C1-6cat: the tool-calling turns in C1's historical pack, and what replacing them leaves untouched. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.c1_tool_replacement \\
        --write logs/stages/stage-2/teacher_modes/audit/c1_tool_replacement.json

Maintainer 2026-11-03 ("Thinking-2507 六类数据对照方案补充"): C1-6cat replaces only the tool-calling data of
D4-Augmented's pack with Thinking-2507-native turns over the shared canonical task pool; every other training and
validation sequence is kept byte for byte, and `D4A_merged_whole` itself is never edited. This audit reads the pack's
own audit rows and:

1. verifies the three files against h1pa.json;
2. classifies every packed sequence as tool-only, tool-free or mixed. A mixed sequence would make the replacement
   touch other data, so the audit refuses one;
3. lists every tool-calling turn to replace (sequence, session id, source, upstream conversation, turn, supervised
   tokens, candidate sha256), training and validation apart;
4. counts what is kept byte for byte (sequences, supervised and non-padding tokens) and the conversations whose
   turns sit on both sides of the training/validation split.

It changes nothing and authorizes nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402

H1PA = REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"
DATASET = "e1_2p96M_plus_fluid"
TOOL = "tool_calling"
#: session-id prefix -> the corpus_v2 source it names (c1_tool_calling_sources.json :: sources)
SOURCES = {"glaive": "glaive_fc_v2", "xlam": "xlam_fc_60k"}
#: the base-corpus category C1's E1 part carries (c1_identity_audit.json :: supervised_tokens_by_part.e1)
BASE_CATEGORY = 2960507
SCHEMA = "aadistill.stage2.teacher_modes.c1_tool_replacement/v1"


def classify(rows: list[dict]) -> dict:
    """Split the pack's sequences into tool-only / tool-free; refuse a sequence that mixes them."""
    tool, kept, mixed = [], [], []
    for r in rows:
        types = {s["data_type"] for s in r["sessions"]}
        if TOOL in types and len(types) > 1:
            mixed.append(r["block"])
        (tool if types == {TOOL} else kept).append(r)
    if mixed:
        raise SystemExit(f"sequences {mixed[:10]} mix tool-calling and other turns: a replacement would touch them")
    return {"tool": tool, "kept": kept}


def turn(row: dict, s: dict) -> list:
    sid = s["session_id"]
    conv, _, t = sid.partition("#")
    return [row["block"], sid, SOURCES[conv.split("-")[0]], conv, t, int(s["supervised_retained"]),
            bool(s.get("truncated")), s["candidate_sha256"]]


def summarize(rows: list[dict]) -> dict:
    sup = sum(int(s["supervised_retained"]) for r in rows for s in r["sessions"])
    return {"sequences": len(rows), "turns": sum(len(r["sessions"]) for r in rows), "supervised_tokens": sup,
            "non_padding_tokens": sum(int(r["unpadded_length"]) for r in rows)}


def by_source(turns: list[list]) -> dict:
    out = defaultdict(lambda: {"conversations": set(), "turns": 0, "supervised_tokens": 0, "truncated_turns": 0})
    for _, _, src, conv, _, sup, cut, _ in turns:
        o = out[src]
        o["conversations"].add(conv)
        o["turns"] += 1
        o["supervised_tokens"] += sup
        o["truncated_turns"] += int(cut)
    return {k: {**v, "conversations": len(v["conversations"])} for k, v in sorted(out.items())}


def audit(rows: list[dict]) -> dict:
    parts = defaultdict(list)
    for r in rows:
        parts[r["part"]].append(r)
    train = parts["e1"] + parts["fluid"]
    tr, va = classify(train), classify(parts["e1_validation"])
    tr_turns = [turn(r, s) for r in tr["tool"] for s in r["sessions"]]
    va_turns = [turn(r, s) for r in va["tool"] for s in r["sessions"]]
    crossing = sorted({t[3] for t in tr_turns} & {t[3] for t in va_turns})
    kept = summarize(tr["kept"])
    replaced = summarize(tr["tool"])
    target = replaced["supervised_tokens"]
    nominal = BASE_CATEGORY / 6
    return {
        "training": {
            "sequences": len(train), "replace": replaced, "keep_byte_for_byte": kept,
            "keep_by_part": {p: summarize([r for r in tr["kept"] if r["part"] == p]) for p in ("e1", "fluid")},
            "replace_by_source": by_source(tr_turns),
            "replace_mean_used_tokens_per_sequence": round(replaced["non_padding_tokens"] / max(1, len(tr["tool"])), 1),
        },
        "validation": {
            "sequences": len(parts["e1_validation"]), "replace": summarize(va["tool"]),
            "keep_byte_for_byte": summarize(va["kept"]), "replace_by_source": by_source(va_turns),
        },
        "conversations_on_both_sides": crossing,
        "replacement_target": {
            "supervised_tokens": target, "nominal_sixth_of_the_base_category": round(nominal, 2),
            "deviation_tokens": round(target - nominal, 2), "deviation_pct": round(100 * (target - nominal) / nominal, 3),
            "rule": "the new tool-calling data is matched to the historical 492,587 first; its deviation from 2,960,507 / 6 "
                    "is reported, never closed by cutting a turn, repeating an example or filling from another category",
        },
        "steps_for_a_new_tool_part": {
            "rule": "T = ceil(3 x (kept training sequences + new tool sequences) / 2) at batch size 2, 3 epochs "
                    "(training_plan_proposal.json); warmup max(10, round(0.05 T)), evaluation every max(25, T // 8), "
                    "save every max(200, T // 2)",
            "kept_training_sequences": kept["sequences"],
            "historical": {"tool_sequences": len(tr["tool"]),
                           "optimizer_steps": math.ceil(3 * (kept["sequences"] + len(tr["tool"])) / 2)},
        },
        "lists": {
            "columns": ["sequence", "session_id", "source", "conversation", "turn", "supervised_tokens", "truncated",
                        "candidate_sha256"],
            "training": tr_turns, "validation": va_turns,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    ds = json.loads(H1PA.read_text())["datasets"][DATASET]
    d = REPO / ds["pack"]
    files = {"blocks.npz": ds["blocks_sha256"], "audit.jsonl": ds["audit_sha256"], "ladder.json": ds["ladder_sha256"]}
    got = {n: A.sha256_file(d / n) for n in files}
    if got != files:
        raise SystemExit(f"{ds['pack']} differs from h1pa.json: {got}")
    rec = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "pack": {"path": ds["pack"], "files_sha256": got, "match": True,
                    "kept_as_is": "D4A_merged_whole is never edited: C1-6cat gets its own pack and manifest"},
           **audit(list(A.jsonl(d / "audit.jsonl"))), "authorizes": "nothing"}
    lists = rec.pop("lists")
    rows = {k: "[\n" + ",\n".join(json.dumps(t) for t in lists[k]) + "\n  ]" for k in ("training", "validation")}
    body = json.dumps({**rec, "lists": {"columns": lists["columns"], "training": "@training", "validation": "@validation"}},
                      indent=1)
    for k, v in rows.items():   # one turn per line: 1,623 rows stay readable and small
        body = body.replace(f'"@{k}"', v)
    Path(a.write).write_text(body + "\n")
    rec["lists"] = lists
    print(json.dumps({k: rec[k] for k in ("training", "validation", "conversations_on_both_sides",
                                          "replacement_target", "steps_for_a_new_tool_part")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
