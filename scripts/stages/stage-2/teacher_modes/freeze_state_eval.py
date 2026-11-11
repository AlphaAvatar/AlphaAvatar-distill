#!/usr/bin/env python3
"""Freeze ``state_eval_hybrid@v1`` from the inputs session's sealed records. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.freeze_state_eval \\
        --records <inputs session store>/inputs_generation --identity <study identity the session sealed under>

The rule (`inputs_protocol.json :: state_evaluation`, maintainer 2026-11-01): per (type, mode), the type's queue in
order, whole naturally terminated one-turn trajectories, until the (type, mode) holds at least ``positions`` scored
positions AND at least ``min_tasks`` tasks; a queue that ends first is a SHORTFALL and nothing is frozen. Nothing is
cut, and correctness is not a criterion.

An item is the trajectory exactly as generated: ``ids`` = prompt ids + output ids; position i predicts token i+1, so
the generated tokens are predicted from position len(prompt) - 1 on. Its tags:
* ``assistant`` -- every position predicting a generated token, the stop token included (the scored positions of
  `positions.supervised_target_v1`);
* ``eos`` -- the position predicting the generated stop token;
* ``think_close`` (thinking mode only) -- the position predicting the generated ``</think>``;
* ``final_answer`` -- the positions predicting the tokens after the think block, the stop token excluded: in
  thinking mode after the generated ``</think>``, in non-thinking mode every generated token (the template's
  preclosed think block is prompt);
* ``reasoning`` (thinking mode only, diagnostic) -- the positions predicting the think span before ``</think>``.
A thinking-mode trajectory without a generated ``</think>``, or a non-thinking one that generates one, breaks its
mode's protocol and is not an item. Each item's ``critical_group`` is its mode, and the manifest declares one group
per mode (`hybrid_state_eval.CRITICAL_GROUPS`), so ``state.critical_token_kl`` weighs the modes 50:50.

Writes the asset (``artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v1/{manifest.json,items.jsonl}``, outside
git) and its record (``logs/stages/stage-2/teacher_modes/audit/state_eval_hybrid_freeze.json``), whose measurement
protocol id is the evaluator's for the search's declared execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402

INPUTS_PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/inputs_protocol.json"
TEACHER = REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json"
OUT = REPO / "artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v1"
RECORD = REPO / "logs/stages/stage-2/teacher_modes/audit/state_eval_hybrid_freeze.json"
SCHEMA = "aadistill.stage2.teacher_modes.state_eval_hybrid_freeze/v1"
MODES = ("H-T", "H-N")
THINK_CLOSE = 151668
SUITE_ID, VERSION = "state_eval_hybrid", 1
LABEL = re.compile(r"^(e\d\d[tn])_(H-T|H-N)_pass0_.*$")


class FreezeError(RuntimeError):
    """The records cannot make the suite."""


def item_of(task: dict, rec: dict, mode: str, stop_ids) -> dict | None:
    """The suite item of one record, or None when it is not naturally terminated or breaks its mode's protocol."""
    turns = rec.get("turns") or []
    if len(turns) != 1:
        return None
    t = turns[0]
    p, o = list(t.get("prompt_token_ids") or []), list(t.get("output_token_ids") or [])
    if t.get("stop_reason") != "eos" or not p or not o or int(o[-1]) not in {int(x) for x in stop_ids}:
        return None
    base = len(p) - 1
    tags = {"assistant": [base + j for j in range(len(o))], "eos": [base + len(o) - 1]}
    if mode == "H-T":
        if THINK_CLOSE not in o[:-1]:
            return None
        c = o.index(THINK_CLOSE)
        tags.update(think_close=[base + c], reasoning=[base + j for j in range(c)],
                    final_answer=[base + j for j in range(c + 1, len(o) - 1)])
    else:
        if THINK_CLOSE in o:
            return None
        tags["final_answer"] = [base + j for j in range(len(o) - 1)]
    ids = p + o
    sub = task["subtype"]
    return {"item_id": f"{task['task_id']}|{mode}", "task_id": task["task_id"], "ids": ids,
            "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1, "tags": tags,
            "domain": task["domain"], "subtype": f"{sub}|{mode}", "type": sub, "mode": mode, "critical_group": mode,
            "source": task.get("source"), "generator_id": task.get("generator_id"), "family_id": task.get("family_id"),
            "seed": task.get("seed"), "row_index": task.get("row_index"), "queue_position": task["queue_position"]}


def records(store: Path, identity: str) -> dict[tuple[str, str], dict]:
    """(task id, mode) -> the sealed state-evaluation record of every complete e-group (one candidate each)."""
    from stages.reasoning_data import generation_driver as GD
    out = {}
    for done in sorted(Path(store).glob("*.complete")):
        label = done.name[:-len(".complete")]
        m = LABEL.match(label)
        if not m:
            continue
        listed = {s["name"]: s["sha256"] for s in json.loads(done.read_text())["shards"]}
        have = {p.name: GD.sha(p) for p in Path(store).glob(f"{label}.part*.jsonl")}
        if have != listed:
            raise FreezeError(f"{label}: the shards present are not the ones its .complete lists")
        for tid, rec in GD.sealed(Path(store), label, identity).items():
            if (tid, m.group(2)) in out:
                raise FreezeError(f"{tid} {m.group(2)}: sealed twice")
            out[(tid, m.group(2))] = rec
    return out


def select(tasks: list[dict], recs: dict, rules: dict, stop_ids) -> tuple[list[dict], dict]:
    """Per (type, mode): whole items in queue order until the positions and the minimum task count are both met."""
    queue = defaultdict(list)
    for t in sorted(tasks, key=lambda t: (t["subtype"], int(t["queue_position"]))):
        queue[t["subtype"]].append(t)
    items, census = [], {}
    need_pos, need_tasks = int(rules["positions"]), int(rules["min_tasks"])
    for sub, q in sorted(queue.items()):
        for mode in MODES:
            got, n, skipped, generated, reached = 0, 0, Counter(), 0, False
            for t in q:
                if got >= need_pos and n >= need_tasks:
                    reached = True
                    break
                rec = recs.get((t["task_id"], mode))
                if rec is None:
                    break                   #: the queue is generated in order: nothing after a gap was generated
                generated += 1
                it = item_of(t, rec, mode, stop_ids)
                if it is None:
                    skipped["not naturally terminated or protocol"] += 1
                    continue
                items.append(it)
                got += len(it["tags"]["assistant"])
                n += 1
            reached = reached or (got >= need_pos and n >= need_tasks)
            census[f"{sub}|{mode}"] = {"scored_positions": got, "tasks": n, "generated_in_order": generated,
                                       "skipped": dict(skipped), "reached": reached,
                                       "tag_positions": {k: sum(len(i["tags"].get(k, [])) for i in items
                                                                if i["subtype"] == f"{sub}|{mode}")
                                                         for k in ("think_close", "final_answer", "eos")}}
    return items, census


def manifest(items: list[dict], census: dict, ident: dict, types_doc: dict, ip: dict, store_identity: str) -> dict:
    from aadistill.initialization.calibration.profiles import mixture_content_sha256
    from stages.teacher_modes import hybrid_state_eval as HSE
    domains = defaultdict(list)
    for t in types_doc["types"]:
        for m in MODES:
            domains[t["domain"]].append(f"{t['subtype']}|{m}")
    return {
        "schema": "aadistill.stage2.teacher_modes.state_eval_hybrid/v1", "suite_id": SUITE_ID, "version": VERSION,
        "domains": {d: subs for d, subs in domains.items()}, "general_domain": None,
        "critical_tags": ["think_close", "final_answer", "eos"], "critical_groups": dict(HSE.CRITICAL_GROUPS),
        "diagnostic_tags": ["assistant", "reasoning"],
        "root_teacher": {"model": ident["model"], "revision": ident["revision"],
                         "weight_shards_sha256": sorted(s["lfs_sha256"] for s in ident["weight_shards"]),
                         "chat_template_sha256": ident["chat_template_sha256"],
                         "vocabulary_sha256": ident["vocabulary_sha256"],
                         "tokenizer_files_sha256": ident["tokenizer_files_sha256"]},
        "modes": {m: {"template_kwargs": {"enable_thinking": m == "H-T"}, "critical_group": m} for m in MODES},
        "rules": {"positions_per_type_and_mode": ip["state_evaluation"]["positions"],
                  "min_tasks_per_type_and_mode": ip["state_evaluation"]["min_tasks"],
                  "selection": "the type's queue in order, whole naturally terminated one-turn trajectories, until both"},
        "inputs_protocol_sha256": A.sha256_file(INPUTS_PROTOCOL),
        "isolation_proof": ip["inputs"]["isolation_proof"],
        "records_identity": store_identity,
        "content_sha256": mixture_content_sha256(items), "n_items": len(items),
        "counts": {k: {"tasks": v["tasks"], "scored_positions": v["scored_positions"]} for k, v in census.items()},
    }


def protocol_id(suite_dir: Path) -> dict:
    """The measurement protocol id of the search's state evaluation on this suite (CPU suffices: every term is a
    declaration or the suite's content)."""
    import dataclasses
    from aadistill.initialization.planning.metrics import StateEvaluator
    from shared import forward_baseline as FB
    from stages.teacher_modes import hybrid_state_eval as HSE
    from stages.teacher_modes import full_search as FS
    ident = json.loads(TEACHER.read_text())
    suite, items, man = HSE.load(suite_dir, ident, REPO / "configs/stages/stage-2/teacher_modes/full_search_types.json")
    terms = FB.session_terms(FB.historical(1, REPO))
    ex = dataclasses.replace(terms["execution"], micro_batch_size=FS.MICRO_BATCH)
    ev = StateEvaluator(suite, items, device="cpu", vocab_size=int(ident["architecture"]["vocab_size"]),
                        suite_content_sha256=man["content_sha256"], position_policy=terms["position_policy"],
                        distribution_support=terms["distribution_support"], execution=ex, numerics=terms["numerics"])
    return {"measurement_protocol_id": ev.measurement_protocol_id, "suite_hash": suite.suite_hash,
            "reduction": ev.reduction.as_dict()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    ip = json.loads(INPUTS_PROTOCOL.read_text())
    proto = json.loads((REPO / ip["base_protocol"]["path"]).read_text())
    tasks = list(A.jsonl(REPO / ip["inputs"]["state_evaluation"]["path"]))
    recs = records(Path(a.records), a.identity)
    items, census = select(tasks, recs, ip["state_evaluation"], proto["teacher"]["stop_ids"])
    short = sorted(k for k, v in census.items() if not v["reached"])
    rec = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "records": {"dir": a.records, "identity": a.identity, "state_evaluation_records": len(recs)},
           "census": census, "shortfall": short}
    if short:
        rec["terminal"] = "SHORTFALL"
        RECORD.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({"terminal": "SHORTFALL", "short": short}, indent=1))
        return 3
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(it, sort_keys=True) + "\n" for it in items)
    (out / "items.jsonl").write_text(body)
    types_doc = json.loads((REPO / ip["inputs"]["types"]["path"]).read_text())
    man = manifest(items, census, json.loads(TEACHER.read_text()), types_doc, ip, a.identity)
    man["types_sha256"] = A.sha256_file(REPO / ip["inputs"]["types"]["path"])
    man["items_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    pid = protocol_id(out)
    rec.update(terminal="FROZEN", asset={"dir": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
                                         "manifest_sha256": A.sha256_file(out / "manifest.json"),
                                         "items_sha256": man["items_sha256"], "content_sha256": man["content_sha256"],
                                         "n_items": len(items)},
               suite_identity=pid, rules=man["rules"],
               by_mode={m: {"items": sum(i["mode"] == m for i in items),
                            "scored_positions": sum(len(i["tags"]["assistant"]) for i in items if i["mode"] == m)}
                        for m in MODES},
               authorizes="nothing")
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("terminal", "asset", "suite_identity", "by_mode")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
