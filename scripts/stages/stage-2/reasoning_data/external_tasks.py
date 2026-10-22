#!/usr/bin/env python3
"""Turn an isolated third-party EXPORT into training tasks, refusing what it must. $0.

    # 1. the whitelist the isolated exporter may import, from the candidate modules
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.external_tasks whitelist --out <w.json>
    # 2. in the isolated env:  <venv>/bin/python -I external_export.py --whitelist <w.json> ...
    # 3. the tasks
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.external_tasks build \
        --export <export.jsonl> --out-dir <dir> --name <pool name>

The build reads only the export file, by its sha256 -- never the third-party
library or the data it bundles. Every row is refused unless its module is a
CANDIDATE of `external_modules.json` for its source; then the module's own
rules apply (no gold; the answer verbatim in the prompt where that means the
answer IS the input -- never for choice modules, whose gold is a listed option;
a rotation by a multiple of 360 degrees; a prompt over the size cap), each
drop counted by reason. A kept
row becomes a task with every field isolation rule R0 requires (source,
module, family, task type, seed) plus its stratum and answer comparator; the
project's answer instruction (boxed final answer) is appended to the prompt so
one extraction rule serves every source, and the gold and any reference trace
stay out of it. Each module's declared comparator must accept its own gold and
reject a foreign answer on every kept row, or the build stops. The pool must then pass `isolation_check.check_pool` with
``refuse=True`` -- nothing is written otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import isolation_check as IC

from stages.calibration_volume.tasks.common import ANSWER_INSTRUCTION

EXPORTABLE = ("reasoning_gym", "reasoning_core")
LEVELS = (0, 2)
MAX_PROMPT_CHARS = 6000


def candidates() -> dict[tuple[str, str], dict]:
    return {(m["source"], m["module"]): m for m in A.read_json(A.CONFIG_DIR / "external_modules.json")["modules"]
            if str(m.get("decision", "")).startswith("candidate")}


def whitelist() -> list[dict]:
    return [{"source": s, "module": m, "levels": list(LEVELS)}
            for (s, m) in sorted(candidates()) if s in EXPORTABLE]


def drop_reason(row: dict, mod: dict) -> str | None:
    if row.get("gold") in (None, ""):
        return "no gold"
    if row.get("gold_in_prompt") and mod.get("drop_if_gold_in_prompt"):
        return "the answer is the input (gold verbatim in the prompt)"
    if len(row["prompt"]) > MAX_PROMPT_CHARS:
        return f"prompt over {MAX_PROMPT_CHARS} characters"
    if row["module"] == "rotate_matrix":
        m = re.search(r"by (\d+) degrees", row["prompt"])
        if m and int(m.group(1)) % 360 == 0:
            return "rotation by a multiple of 360 degrees"
    trace = row.get("reference_trace")
    if trace and trace.strip() and trace.strip()[:200] in row["prompt"]:
        return "reference trace in the prompt"
    return None


def task_of(row: dict, mod: dict) -> dict:
    tid = hashlib.sha256("|".join(str(row[k]) for k in ("source", "module", "version", "level", "seed", "index"))
                         .encode()).hexdigest()[:20]
    prompt = row["prompt"].rstrip() + "\n\n" + ANSWER_INSTRUCTION
    task = {"task_id": tid, "unit_id": tid, "source": row["source"], "module": row["module"],
            "family_id": mod["family"], "task_type": mod["task_type"], "stratum": mod["stratum"],
            "answer_type": mod["answer_type"], "answer": str(row["gold"]),
            "seed": row["seed"], "index": row["index"], "level": row["level"], "version": row["version"],
            "interactive": False, "messages": [{"role": "user", "content": prompt}],
            "content_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    if row.get("reference_trace"):
        task["reference_trace_sha256"] = hashlib.sha256(row["reference_trace"].encode()).hexdigest()
    return task


def build(export: Path) -> dict:
    cands = candidates()
    tasks, drops, refused = [], Counter(), Counter()
    seen = set()
    for row in A.jsonl(export):
        mod = cands.get((row.get("source"), row.get("module")))
        if mod is None:
            refused[f"{row.get('source')}/{row.get('module')}: not a candidate module"] += 1
            continue
        why = drop_reason(row, mod)
        if why:
            drops[f"{row['module']}: {why}"] += 1
            continue
        t = task_of(row, mod)
        if t["content_sha256"] in seen:
            drops[f"{row['module']}: duplicate prompt"] += 1
            continue
        seen.add(t["content_sha256"])
        tasks.append(t)
    if refused:
        raise SystemExit(f"the export holds rows the whitelist does not allow: {dict(refused)}")
    # the comparator must accept the module's own gold and reject a foreign answer
    from stages.reasoning_data.generation_adapter import COMPARE
    wrong = Counter(f"{t['source']}/{t['module']}" for t in tasks if t["answer_type"] in COMPARE
                    and not (COMPARE[t["answer_type"]](t["answer"], t["answer"])
                             and not COMPARE[t["answer_type"]]("zzz 999 qqq", t["answer"])))
    if wrong:
        raise SystemExit(f"declared comparator does not fit the module's gold: {dict(wrong)}")
    report = IC.check_pool(tasks)
    if not report["pass"]:
        raise SystemExit(f"isolation failed: {report['violations']}")
    return {"tasks": tasks, "drops": dict(sorted(drops.items())), "isolation": report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("whitelist")
    w.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--export", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--record", default=None, help="also write a small validation record here")
    args = ap.parse_args(argv)
    if args.cmd == "whitelist":
        Path(args.out).write_text(json.dumps(whitelist(), indent=1) + "\n")
        print(args.out, len(whitelist()))
        return 0
    export = Path(args.export)
    got = build(export)
    out = Path(args.out_dir) / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in got["tasks"]))
    iso = got["isolation"]
    manifest = {
        "schema": "aadistill.stage2.reasoning_data.external_tasks/v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "export": {"sha256": A.sha256_file(export), "rows": sum(1 for _ in A.jsonl(export)),
                   "versions": sorted({t["version"] for t in got["tasks"]})},
        "n_tasks": len(got["tasks"]), "drops": got["drops"],
        "by_stratum": dict(sorted(Counter(t["stratum"] for t in got["tasks"]).items())),
        "by_module": dict(sorted(Counter(f"{t['source']}/{t['module']}" for t in got["tasks"]).items())),
        "answer_types": dict(sorted(Counter(t["answer_type"] for t in got["tasks"]).items())),
        "isolation": {"pass": iso["pass"], "violations": iso["violations"],
                      "R7_items_with_any_grid_match": iso["results"]["R7_arc_agi"]["items_with_any_grid_match"],
                      "R2_flagged": iso["results"]["R2_near_duplicate"]["flagged"]},
        "sha256": A.sha256_file(out),
        "approved_for_generation": False,
        "authorizes": "nothing",
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if args.record:
        Path(args.record).write_text(json.dumps({**manifest, "pool_path_not_committed": str(out)}, indent=1) + "\n")
    print(out, manifest["n_tasks"], manifest["by_stratum"], manifest["isolation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
