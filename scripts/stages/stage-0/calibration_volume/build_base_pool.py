#!/usr/bin/env python3
"""Build the Stage-0 base calibration POOL: every eligible teacher-native item.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/build_base_pool.py \
        --teacher-records <teacher_cal.jsonl> --version 1

The pool is what volume treatments SAMPLE from; it is not itself a profile.

Primary items come from the teacher's own calibration sessions on the
calibration-split families (`tasks/registry.py`), kept only when the session is
VERIFIED (static: the scorer's usable-gated boxed answer; interactive: the
environment's success, replayed) and terminated naturally. An interactive
episode is TURN-EXPANDED under the corpus_v2 contract
(`aadistill.data.sessions.final_assistant_loss_mask`): turn k becomes one item
whose context is the episode so far, rendered by the chat template (earlier
reasoning dropped, earlier answers and observations kept), and whose supervised
span is turn k. Every assistant turn is therefore covered by exactly one item,
and `positions.supervised_target_v1` is unchanged. The EPISODE stays the
sampling unit, so its turns are never split across profiles or resamples.

Supporting items come from corpus_v2: verified-correct gsm8k, openmath,
multihop_qa and rag_evidence sessions, outside every session and source prompt
the recovery rung, state_eval@v1 and the D-series batteries use.

Every item carries `domain` = its P10.2 axis (or `supporting`) and `subtype` =
its generator (or corpus type): the DEPTH objective averages per subtype and then
per domain, so these labels are part of the objective and are fixed here, once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.data.sessions import render_session, render_system_block  # noqa: E402
from stages.calibration_volume import score_battery as SB  # noqa: E402

TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
TOKENIZER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
SPECIAL = {"think_open": "<think>", "think_close": "</think>",
           "im_end": "<|im_end|>", "tool_call_close": "</tool_call>"}
SUPPORTING_TYPES = ("gsm8k", "openmath", "multihop_qa", "rag_evidence")
OUT = REPO / "artifacts/stages/stage-0/calibration_volume"


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def tag_positions(ids, body_mask, n_system, special):
    """The e8 builder's frozen tag definition, imported rather than retyped."""
    from stages.e8.build_e8_calibration import tag_positions as _tags
    return _tags(ids, body_mask, n_system, special)


def render_item(tok, special, messages, *, item_id, domain, subtype, meta) -> dict:
    session = {"id": item_id, "messages": messages, "tools": None}
    r = render_session(tok, session, block_len=10 ** 9)
    system_ids = tok(render_system_block(tok, r.system_text, r.tools),
                     add_special_tokens=False).input_ids
    ids = system_ids + r.body_ids
    tags = tag_positions(ids, r.body_mask, len(system_ids), special)
    if not tags.get("assistant"):
        raise ValueError(f"{item_id}: no supervised assistant span")
    return {"item_id": item_id, "domain": domain, "subtype": subtype,
            "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1,
            "templated": True, "ids": ids, "tags": tags,
            "scored_positions": len(tags["assistant"]),
            "reasoning_positions": len(tags.get("reasoning", [])),
            "final_answer_positions": len(tags.get("final_answer", [])), **meta}


def teacher_items(tok, special, records_path: Path, tasks_path: Path) -> tuple[list, dict]:
    tasks = {t["task_id"]: t for t in (json.loads(l) for l in tasks_path.open())}
    records = [json.loads(l) for l in records_path.open()]
    rows = {r["task_id"]: r for r in SB.score(tasks, records)}
    items, census = [], Counter()
    for rec in records:
        task = tasks[rec["task_id"]]
        row = rows[rec["task_id"]]
        ended = rec["turns"][-1]["stop_reason"] if rec["turns"] else "none"
        census[f"{task['axis']}|end={rec['end']}|correct={row['correct']}"] += 1
        if not row["correct"]:
            continue
        if any(t["stop_reason"] != "eos" for t in rec["turns"]):
            census["dropped_nonterminated_turn"] += 1
            continue
        history = rec["messages"]
        assistant_at = [i for i, m in enumerate(history) if m["role"] == "assistant"]
        unit = rec["task_id"]
        base = {"source": "stage0_teacher_native", "unit_id": unit,
                "task_id": unit, "generator_id": task["generator_id"],
                "family_id": task["family_id"],
                "transformation_class": task["transformation_class"],
                "difficulty": task["difficulty"], "axis": task["axis"],
                "pool_class": "primary", "verified": True,
                "lineage": "project procedural generator + teacher-native trajectory",
                "n_turns_in_unit": len(assistant_at)}
        for k, idx in enumerate(assistant_at):
            msgs = history[:idx + 1]
            try:
                items.append(render_item(
                    tok, special, msgs, item_id=f"{unit}#t{k}", domain=task["axis"],
                    subtype=task["generator_id"], meta={**base, "turn": k,
                                                        "session_id": f"{unit}#t{k}"}))
            except ValueError as exc:
                census[f"render_error:{type(exc).__name__}"] += 1
    return items, dict(census)


def excluded_keys(audit_inputs: Path) -> dict[str, set]:
    """Sessions, source prompts and prompt texts the supporting draw must avoid."""
    from stages.calibration_volume import audit_calibration_data as A

    spec = json.loads(audit_inputs.read_text())
    sessions, examples = A.load_corpus(spec["corpus"])
    keys = {"session": set(), "source": set(), "prompt": set()}
    for r in spec["roles"]:
        if r["kind"] == "items" and r["role"].startswith("calibration:"):
            continue  # the historical profiles are not isolation partners of calibration
        if r["kind"] == "items":
            units = A.item_units(REPO / r["path"], sessions, examples, None)
        elif r["kind"] == "recovery_rung":
            units, _ = A.recovery_units(spec["recovery_rung"], sessions, examples)
        else:
            units = A.battery_units(REPO / r["path"], r.get("recursive", False))
        for u in units:
            if u.get("session_id"):
                keys["session"].add(u["session_id"])
            if u.get("source_id"):
                keys["source"].add(str(u["source_id"]).split("#")[0])
            if u.get("prompt_text"):
                keys["prompt"].add(A.text_hash(u["prompt_text"]))
    return keys


def supporting_items(tok, special, corpus: Path, keys: dict, max_tokens: int) -> tuple[list, dict]:
    from stages.calibration_volume.audit_calibration_data import last_user_text, text_hash

    items, census = [], Counter()
    for line in corpus.open():
        s = json.loads(line)
        st = s["data_type"]
        if st not in SUPPORTING_TYPES:
            continue
        if not (s["correctness_verdict"] == "ok" and s["correct"]):
            census[f"{st}|not_verified_correct"] += 1
            continue
        if (s["id"] in keys["session"] or s["source_id"] in keys["source"]
                or text_hash(last_user_text(s["messages"])) in keys["prompt"]):
            census[f"{st}|excluded_by_isolation"] += 1
            continue
        if s["n_rendered_tokens"] > max_tokens:
            census[f"{st}|over_max_tokens"] += 1
            continue
        try:
            items.append(render_item(
                tok, special, s["messages"], item_id=f"{st}/{s['id']}", domain="supporting",
                subtype=st, meta={"source": "corpus_v2", "unit_id": s["source_id"],
                                  "session_id": s["id"], "source_id": s["source_id"],
                                  "axis": "supporting", "pool_class": "supporting",
                                  "verified": True, "generator_id": None,
                                  "family_id": None, "difficulty": None,
                                  "lineage": f"corpus_v2 {s['source']}", "turn": 0,
                                  "n_turns_in_unit": 1,
                                  "candidate_sha256": s.get("candidate_sha256")}))
            census[f"{st}|kept"] += 1
        except ValueError as exc:
            census[f"{st}|render_error:{type(exc).__name__}"] += 1
    return items, dict(census)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-records", required=True)
    ap.add_argument("--tasks", default="data/stages/stage-0/calibration_volume/tasks/calibration_v1.jsonl")
    ap.add_argument("--corpus", default="artifacts/stages/stage-3/corpus_v2/sessions.jsonl")
    ap.add_argument("--audit-inputs", default="configs/stages/stage-0/calibration_volume/audit_inputs.json")
    ap.add_argument("--max-tokens", type=int, required=True,
                    help="the pool's admission bound on rendered item length")
    ap.add_argument("--version", type=int, required=True)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    special = {k: tok.convert_tokens_to_ids(v) for k, v in SPECIAL.items()}

    out_dir = OUT / f"base_pool_v{args.version}"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; a built pool is never rewritten")
    prim, prim_census = teacher_items(tok, special, REPO / args.teacher_records,
                                      REPO / args.tasks)
    over = [i for i in prim if i["n_tokens"] > args.max_tokens]
    over_units = {i["unit_id"] for i in over}
    prim = [i for i in prim if i["unit_id"] not in over_units]
    keys = excluded_keys(REPO / args.audit_inputs)
    supp, supp_census = supporting_items(tok, special, REPO / args.corpus, keys,
                                         args.max_tokens)
    items = prim + supp
    out_dir.mkdir(parents=True)
    body = "".join(json.dumps(i, sort_keys=True) + "\n" for i in items)
    (out_dir / "items.jsonl").write_text(body)
    by = defaultdict(lambda: Counter())
    for i in items:
        key = f"{i['domain']}|{i['subtype']}"
        by[key]["items"] += 1
        by[key]["units"] += 0
        by[key]["scored_positions"] += i["scored_positions"]
        by[key]["reasoning_positions"] += i["reasoning_positions"]
    for key in by:
        by[key]["units"] = len({i["unit_id"] for i in items
                                if f"{i['domain']}|{i['subtype']}" == key})
    manifest = {
        "schema": "aadistill.stage0.base_pool/v1", "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "build_base_pool.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "tokenizer": {"id": TOKENIZER, "revision": TOKENIZER_REVISION},
        "teacher_records": args.teacher_records,
        "teacher_records_sha256": hashlib.sha256((REPO / args.teacher_records).read_bytes()).hexdigest(),
        "tasks_sha256": hashlib.sha256((REPO / args.tasks).read_bytes()).hexdigest(),
        "max_tokens": args.max_tokens,
        "dropped_units_over_max_tokens": len(over_units),
        "items_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "content_sha256": hashlib.sha256("\n".join(
            f"{i['item_id']}:{sha_ids(i['ids'])}" for i in items).encode()).hexdigest(),
        "n_items": len(items), "n_units": len({i["unit_id"] for i in items}),
        "by_stratum": {k: dict(v) for k, v in sorted(by.items())},
        "primary_census": prim_census, "supporting_census": supp_census,
        "exclusion_keys": {k: len(v) for k, v in keys.items()},
        "authorizes": "nothing",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_items", "n_units", "max_tokens",
                                               "dropped_units_over_max_tokens")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
