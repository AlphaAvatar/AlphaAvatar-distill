#!/usr/bin/env python3
"""Teacher-native tool-calling episodes -> admitted training documents and search items, per lineage. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_admission \\
        --pool data/stages/stage-2/teacher_modes/tool_pool_v1 --store <tool_session store> --identity <id> \\
        --mode H-T --out <dir>

The tasks are read through `tool_pool.load_pool` (every split file re-verified against pool.json) in each split's
frozen order, so the documents and items come out in the order the session walked and the consumers select by.

Every episode the teacher generated (``run_battery.run_episodes`` in ``generate`` mode, so each turn keeps the
engine's own prompt and output token ids) goes through `tool_episodes.verify` -- protocol, schema validity, recorded
execution, and step gold / tool selection / task success where the task carries that verifier, on every turn -- and
is admitted WHOLE or not at all. An admitted episode becomes, by its task's split:

* training / validation -> one token-in/token-out document per turn (`tool_documents.jsonl`), targets = the
  generated ids only;
* calibration / state_eval -> one item per turn with the tool-calling critical tags (`tool_items.jsonl`), the input
  of the hybrid profiles v2 and of state_eval_hybrid@v2.

The census reports, per split and category: tasks offered, episodes generated, admitted, refusals by level, the
admitted episodes by the levels they were actually verified at (a level without a verifier is never counted as
passed) and by episode shape (a single-step gold tool call is verified at protocol+schema_validity+step_gold), and
the supervised tokens / scored positions admitted. A second candidate (pass k > 0) is read only for a task whose
earlier candidate was refused. Nothing reads a student result or an evaluation item.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import tool_episodes as TE  # noqa: E402

SCHEMA = "aadistill.stage2.teacher_modes.tool_admission/v1"
MODES = {"H-T": True, "H-N": False, "T2507": True}


def admit(tasks: list[dict], episodes: list[dict], *, mode: str, stop_ids) -> dict:
    """Verify and convert. ``episodes`` are records with ``task_id``, ``turns`` and optional ``candidate``."""
    thinking = MODES[mode]
    by_task = {t["task_id"]: t for t in tasks}
    cands = defaultdict(list)
    for ep in episodes:
        cands[ep["task_id"]].append(ep)
    docs, items, census = [], [], defaultdict(lambda: {"offered": 0, "generated": 0, "admitted": 0,
                                                        "refused_by_check": Counter(), "admitted_by_verification":
                                                        Counter(), "admitted_by_shape": Counter(),
                                                        "supervised_tokens": 0, "scored_positions": 0})
    verdicts = {}
    for t in tasks:
        c = census[(t["split"], t["category"])]
        c["offered"] += 1
        for ep in sorted(cands.get(t["task_id"], []), key=lambda e: int(e.get("candidate", 0))):
            c["generated"] += 1
            v = TE.verify(t, ep["turns"], thinking=thinking)
            if not v.admitted:
                for k, ok in v.checks.items():
                    if ok is False:
                        c["refused_by_check"][k] += 1
                verdicts.setdefault(t["task_id"], []).append(v.as_dict())
                continue
            if t["split"] in ("training", "validation"):
                ds = TE.documents(t, ep["turns"], mode=mode, stop_ids=stop_ids)
                docs += [{"doc_id": d.doc_id, "task_id": t["task_id"], "split": t["split"],
                          "category": t["category"], "family": t["family"], "prompt_ids": list(d.prompt_ids),
                          "output_ids": list(d.output_ids)} for d in ds]
                c["supervised_tokens"] += sum(len(d.output_ids) for d in ds)
            elif t["split"] in ("calibration", "state_eval"):
                its = TE.items(t, ep["turns"], mode=mode, thinking=thinking)
                if not its:
                    c["refused_by_check"]["item_protocol"] += 1
                    continue
                items += [{**it, "split": t["split"]} for it in its]
                c["scored_positions"] += sum(len(it["tags"]["assistant"]) for it in its)
            c["admitted"] += 1
            c["admitted_by_verification"]["+".join(v.verified_levels())] += 1
            c["admitted_by_shape"][TE.episode_shape(t)] += 1
            verdicts[t["task_id"]] = [v.as_dict()]
            break
    stray = sorted(set(cands) - set(by_task))
    return {"documents": docs, "items": items, "stray_episodes": stray, "verification_version": TE.VERIFICATION_VERSION,
            "census": {f"{s}|{k}": {**v, "refused_by_check": dict(v["refused_by_check"]),
                                    "admitted_by_verification": dict(v["admitted_by_verification"]),
                                    "admitted_by_shape": dict(v["admitted_by_shape"])}
                       for (s, k), v in sorted(census.items())}}


def v1_tail_audit(generations: dict[str, Path]) -> dict:
    """V1's tail rule (`tool_episodes.text_after_last_call`) on recorded teacher-native turns: per source, the turns
    that make a call, and those with text after their last ``</tool_call>`` -- which `VERIFICATION_VERSION` v2
    refuses and v1 admitted. A $0 measurement of what the rule changes; the turns are read, never used as data."""
    out = {}
    for name, path in generations.items():
        rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        calls, tails = 0, []
        for r in rows:
            tail = TE.text_after_last_call(str(r.get("raw") or ""))
            if tail is None:
                continue
            calls += 1
            if tail.strip():
                tails.append({"id": r.get("id"), "tail": tail.strip()[:120]})
        out[name] = {"file_sha256": __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest(),
                     "turns": len(rows), "turns_with_a_call": calls, "with_text_after_the_last_call": len(tails),
                     "examples": tails[:5]}
    return {"schema": "aadistill.stage2.teacher_modes.v1_tail_audit/v1", "rule": TE.VERIFICATION_VERSION,
            "sources": out, "authorizes": "nothing"}


def episodes_from_store(store: Path, identity: str, mode: str) -> list[dict]:
    """Every sealed episode of ``mode`` in a `tool_session.py` store, with its candidate (pass) index; only
    labels whose ``.complete`` lists exactly the shards present are read."""
    import re
    from stages.reasoning_data import generation_driver as GD
    pat = re.compile(rf"^tool_{re.escape(mode)}_r\d+_pass(\d+)$")       #: tool_session.py's round labels
    out = []
    for done in sorted(Path(store).glob("*.complete")):
        label = done.name[:-len(".complete")]
        m = pat.match(label)
        if not m:
            continue
        listed = {x["name"]: x["sha256"] for x in json.loads(done.read_text())["shards"]}
        have = {p.name: GD.sha(p) for p in Path(store).glob(f"{label}.part*.jsonl")}
        if have != listed:
            raise SystemExit(f"{label}: the shards present are not the ones its .complete lists")
        for tid, rec in GD.sealed(Path(store), label, identity).items():
            out.append({**rec, "task_id": tid, "candidate": int(m.group(1))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1-tail-audit", default=None, help="write V1's tail-rule audit of --generations here and stop")
    ap.add_argument("--generations", action="append", default=[], help="name=recorded generations (jsonl with raw)")
    ap.add_argument("--pool", default=None, help="the tool_pool@v1 directory (verified by tool_pool.load_pool)")
    ap.add_argument("--episodes", default=None, help="run_battery records (jsonl)")
    ap.add_argument("--store", default=None, help="a tool_session.py store (sealed shards), with --identity")
    ap.add_argument("--identity", default=None)
    ap.add_argument("--mode", default=None, choices=sorted(MODES))
    ap.add_argument("--stop-ids", default="151645,151643")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.v1_tail_audit:
        doc = v1_tail_audit({k: Path(v) for k, v in (g.split("=", 1) for g in a.generations)})
        Path(a.v1_tail_audit).write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({k: v["with_text_after_the_last_call"] for k, v in doc["sources"].items()}))
        return 0
    if not (a.pool and a.mode and a.out):
        ap.error("--pool, --mode and --out are required")
    from stages.teacher_modes import tool_pool as TP
    man, splits = TP.load_pool(Path(a.pool))
    tasks = []
    for split in sorted(splits):
        by_id = {t["task_id"]: t for t in splits[split]}
        tasks += [by_id[t] for t in man["orders"][split]]
    bad = {t.get("task_id"): TE.task_problems(t) for t in tasks if TE.task_problems(t)}
    if bad:
        raise SystemExit(f"{len(bad)} task records the protocol cannot run, e.g. {next(iter(bad.items()))}")
    if a.store:
        eps = episodes_from_store(Path(a.store), a.identity, a.mode)
    elif a.episodes:
        eps = [json.loads(x) for x in Path(a.episodes).read_text(encoding="utf-8").split("\n") if x.strip()]
    else:
        raise SystemExit("--episodes or --store is required")
    res = admit(tasks, eps, mode=a.mode, stop_ids=[int(x) for x in a.stop_ids.split(",")])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tool_documents.jsonl").open("w") as f:
        for d in res["documents"]:
            f.write(json.dumps(d) + "\n")
    with (out / "tool_items.jsonl").open("w") as f:
        for it in res["items"]:
            f.write(json.dumps(it) + "\n")
    (out / "tool_admission.json").write_text(json.dumps({"schema": SCHEMA, "mode": a.mode,
                                                         "verification_version": TE.VERIFICATION_VERSION,
                                                         "census": res["census"],
                                                         "stray_episodes": res["stray_episodes"],
                                                         "authorizes": "nothing"}, indent=1) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("offered", "admitted")} for k, v in res["census"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
