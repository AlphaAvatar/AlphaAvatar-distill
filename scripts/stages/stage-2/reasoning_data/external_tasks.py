#!/usr/bin/env python3
"""Turn an isolated third-party EXPORT into training tasks, refusing what it must. $0.

    # 1. the whitelist the isolated exporter may import, from the candidate modules
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.external_tasks whitelist --out <w.json>
    #    (--sources narrows it to what one isolated env exports; mathematics_dataset has its own)
    # 2. in each isolated env:  PYTHONHASHSEED=0 <venv>/bin/python -I external_export.py --whitelist <w.json> ...
    # 3. the tasks
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.external_tasks build \
        --export <export.jsonl> [<export.jsonl> ...] --out-dir <dir> --name <pool name>

The build reads only the export files, by their sha256 -- never the third-party
library or the data it bundles. Every row is refused unless its module is a
CANDIDATE of `external_modules.json` for its source; then the module's own
rules apply (no gold; the answer verbatim in the prompt where that means the
answer IS the input -- never for choice modules, whose gold is a listed option;
a rotation by a multiple of 360 degrees; a prompt over the size cap; an item
outside its project checker's domain, `checkers.out_of_domain`, so no paid
candidate is generated for an item that could never be accepted), each
drop counted by reason. A kept
row becomes a task with every field isolation rule R0 requires (source,
module, family, task type, seed) plus its stratum and answer comparator; the
project's answer instruction (boxed final answer) is appended to the prompt so
one extraction rule serves every source, and the gold and any reference trace
stay out of it. Each module's declared comparator -- or, for answer type
``checker``, its project checker (`checkers.CHECKERS`) -- must accept its own
gold and reject a foreign answer on every kept row, or the build stops. A
PrOntoQA task carries ``reference_conclusions`` -- its reference proof's
intermediate conclusions (the chain-of-thought sentences that are not premises,
less the last, which states the query's property and is the label's own check)
-- for the step check, and the build stops unless that check finds every one in
the reference proof itself (``partial``). The pool must then pass
`isolation_check.check_pool` with ``refuse=True`` -- nothing is written
otherwise.
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
from stages.reasoning_data import checkers as CK
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import process_check as PC

from stages.calibration_volume.tasks.common import ANSWER_INSTRUCTION

EXPORTABLE = ("reasoning_gym", "reasoning_core", "prontoqa", "mathematics_dataset")
LEVELS = (0, 2)                     # default; a module row's own `levels` wins
MAX_PROMPT_CHARS = 6000


def candidates() -> dict[tuple[str, str], dict]:
    return {(m["source"], m["module"]): m for m in A.read_json(A.CONFIG_DIR / "external_modules.json")["modules"]
            if str(m.get("decision", "")).startswith("candidate")}


def whitelist(sources=EXPORTABLE) -> list[dict]:
    return [{"source": s, "module": m, "levels": list(row.get("levels", LEVELS))}
            for (s, m), row in sorted(candidates().items()) if s in EXPORTABLE and s in sources]


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
    if mod.get("answer_type") == "checker":
        why = CK.out_of_domain(f"{row['source']}/{row['module']}", row["gold"], row["prompt"])
        if why:
            return f"outside the project checker's domain: {why}"
    return None


def _sentence(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .")


def reference_conclusions(question: str, chain: list[str]) -> list[str]:
    """The chain-of-thought sentences that are not premises of `question`, less
    the last (it states the query's property: the label's check, and quoting the
    query would read as its negation)."""
    premises = {_sentence(s) for s in re.split(r"(?<=\.)\s+", question)}
    return [c for c in chain if _sentence(c) not in premises][:-1]


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
    if row.get("reference_chain"):
        task["reference_conclusions"] = reference_conclusions(row["prompt"].split(" True or false:")[0],
                                                              row["reference_chain"])
    return task


def build(exports: list[Path]) -> dict:
    cands = candidates()
    tasks, drops, refused = [], Counter(), Counter()
    seen = set()
    traces = {}
    for row in (r for e in exports for r in A.jsonl(e)):
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
        traces[t["task_id"]] = row.get("reference_trace") or ""
    if refused:
        raise SystemExit(f"the export holds rows the whitelist does not allow: {dict(refused)}")
    # the comparator / checker must accept the module's own gold and reject a foreign answer
    from stages.reasoning_data.generation_adapter import COMPARE
    missing = sorted({f"{t['source']}/{t['module']}" for t in tasks if t["answer_type"] == "checker"} - set(CK.CHECKERS))
    if missing:
        raise SystemExit(f"answer type 'checker' without a checker: {missing}")

    def fits(t: dict) -> bool:
        if t["answer_type"] == "checker":
            check = CK.CHECKERS[f"{t['source']}/{t['module']}"]
            return (check(t, "\\boxed{" + t["answer"] + "}").state == "correct"
                    and check(t, "\\boxed{zzz 999 qqq}").state != "correct")
        return (COMPARE[t["answer_type"]](t["answer"], t["answer"])
                and not COMPARE[t["answer_type"]]("zzz 999 qqq", t["answer"]))
    wrong = Counter(f"{t['source']}/{t['module']}" for t in tasks
                    if (t["answer_type"] in COMPARE or t["answer_type"] == "checker") and not fits(t))
    if wrong:
        raise SystemExit(f"declared comparator does not fit the module's gold: {dict(wrong)}")
    # the step check must find every reference conclusion in the reference proof itself
    unsound = Counter(f"{t['source']}/{t['module']}" for t in tasks if "reference_conclusions" in t
                      and not (t["reference_conclusions"]
                               and PC.reference_steps(traces[t["task_id"]], t["reference_conclusions"]).state
                               == "partial"))
    if unsound:
        raise SystemExit(f"reference conclusions the reference proof does not state: {dict(unsound)}")
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
    w.add_argument("--sources", nargs="*", default=list(EXPORTABLE))
    b = sub.add_parser("build")
    b.add_argument("--export", nargs="+", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--record", default=None, help="also write a small validation record here")
    args = ap.parse_args(argv)
    if args.cmd == "whitelist":
        wl = whitelist(tuple(args.sources))
        Path(args.out).write_text(json.dumps(wl, indent=1) + "\n")
        print(args.out, len(wl))
        return 0
    exports = [Path(e) for e in args.export]
    got = build(exports)
    out = Path(args.out_dir) / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in got["tasks"]))
    iso = got["isolation"]
    manifest = {
        "schema": "aadistill.stage2.reasoning_data.external_tasks/v2",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exports": [{"sha256": A.sha256_file(e), "rows": sum(1 for _ in A.jsonl(e)),
                     "sources": sorted({r["source"] for r in A.jsonl(e)}),
                     "python_hash_seed": sorted({str(r.get("python_hash_seed")) for r in A.jsonl(e)})}
                    for e in exports],
        "versions": {s: sorted({t["version"] for t in got["tasks"] if t["source"] == s})
                     for s in sorted({t["source"] for t in got["tasks"]})},
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
