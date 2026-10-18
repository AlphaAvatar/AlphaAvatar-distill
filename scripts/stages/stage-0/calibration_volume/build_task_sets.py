#!/usr/bin/env python3
"""Build a role's procedural task set (dev, final, or calibration prompts).

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-0/calibration_volume/build_task_sets.py \
        --role dev --per-cell 16 --version 1

A CELL is one (generator, difficulty). Each cell gets `--per-cell` instances
(`--per-cell-interactive` for interactive generators), cycling through the
role's families in a fixed order, so every family of the role is represented
and no family is chosen by looking at results. Seeds come from the role's own
range (`registry.SEED_RANGES`), so roles cannot share an instance.

Writes `<role>_v<N>.jsonl` and its manifest next to it, and refuses to write a
set that shares a family, a content hash or a prompt with any already-built set
of a role it must be isolated from. Nothing here loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.evaluation.isolation import key_overlaps, require_disjoint  # noqa: E402
from stages.calibration_volume.tasks import registry as R  # noqa: E402
from stages.calibration_volume.tasks.common import DIFFICULTIES, build_instance  # noqa: E402

OUT_DIR = REPO / "data/stages/stage-0/calibration_volume/tasks"
TASKS_DIR = Path(__file__).resolve().parent / "tasks"
SCHEMA = "aadistill.stage0.task_set/v1"

#: Which roles must be isolated from which, and on what keys. Evaluation must
#: share no FAMILY with calibration; no two roles may share an instance.
ISOLATION = {
    ("calibration", "dev"): ("family", "content", "prompt"),
    ("calibration", "final"): ("family", "content", "prompt"),
    ("dev", "final"): ("content", "prompt"),
}


def generator_source_digest() -> dict:
    files = sorted(TASKS_DIR.glob("*.py"))
    entries = [{"path": str(p.relative_to(REPO)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    digest = hashlib.sha256("".join(f"{e['path']}:{e['sha256']}\n"
                                    for e in entries).encode()).hexdigest()
    return {"digest": digest, "files": entries}


def prompt_key(task: dict) -> str:
    text = " ".join((task["messages"][0]["content"] or "").split()).lower()
    return hashlib.sha256(text.encode()).hexdigest()


#: `prompt` is required-disjoint for STATIC tasks only. An interactive task's
#: opening message shows only its start state and budget, so two episodes with
#: different hidden rules can open with the same words by design; their content
#: hash (which includes the hidden environment) still must not collide, and the
#: opening-message overlap is reported rather than required to be zero.
KEYS = {
    "family": lambda t: f"{t['generator_id']}/{t['family_id']}",
    "content": lambda t: t["content_sha256"],
    "prompt": lambda t: None if t["interactive"] else prompt_key(t),
    "interactive_opening": lambda t: prompt_key(t) if t["interactive"] else None,
}


def build(role: str, per_cell: int, per_cell_interactive: int,
          generators: list[str] | None) -> list[dict]:
    split_role = "calibration" if role == "calibration" else "evaluation"
    lo, hi = R.SEED_RANGES[role]
    tasks, seen = [], set()
    for gen in R.GENERATORS:
        if generators and gen.generator_id not in generators:
            continue
        fams = R.family_split(gen)[split_role]
        if not fams:
            continue
        n = per_cell_interactive if gen.interactive else per_cell
        for d_i, difficulty in enumerate(DIFFICULTIES):
            made, i = 0, 0
            while made < n:
                fam = gen.family(fams[i % len(fams)])
                seed = lo + d_i * 100_000 + i
                if seed >= hi:
                    raise SystemExit(f"{gen.generator_id}/{difficulty}: seed range exhausted")
                i += 1
                t = build_instance(gen, fam, difficulty, seed)
                if t["content_sha256"] in seen:
                    continue
                seen.add(t["content_sha256"])
                t["role"] = role
                tasks.append(t)
                made += 1
    return tasks


def existing_sets(exclude: Path) -> dict[str, list[dict]]:
    """Every built set a new one must be isolated from: the stage's own task
    sets, and every task set registered as a permanent evaluation set (dev_v1
    lives in data/eval-set/ since 2026-10-18)."""
    from shared import eval_sets
    out: dict[str, list[dict]] = {}
    paths = sorted(OUT_DIR.glob("*_v*.jsonl")) + eval_sets.registered_files(SCHEMA)
    for path in paths:
        if path == exclude:
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if rows:
            out.setdefault(rows[0]["role"], []).extend(rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, choices=sorted(R.SEED_RANGES))
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--per-cell", type=int, required=True)
    ap.add_argument("--per-cell-interactive", type=int, default=None)
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    out = OUT_DIR / f"{args.role}_v{args.version}.jsonl"
    from shared import eval_sets
    if any(p.name == out.name for p in eval_sets.registered_files(SCHEMA)):
        raise SystemExit(f"{out.name} is a registered evaluation set (data/eval-set/); a "
                         "registered set is never rebuilt -- bump --version")
    if out.exists():
        raise SystemExit(f"{out.relative_to(REPO)} exists; a built set is never "
                         "rewritten -- bump --version")
    per_int = args.per_cell_interactive if args.per_cell_interactive is not None \
        else args.per_cell
    tasks = build(args.role, args.per_cell, per_int, args.generators)

    others = existing_sets(out)
    roles = {args.role: tasks, **{r: rows for r, rows in others.items() if r != args.role}}
    report = key_overlaps(roles, KEYS)
    required = [(a, b, k) for (a, b), keys in ISOLATION.items()
                if args.role in (a, b) and a in roles and b in roles for k in keys]
    checked = require_disjoint(report, required)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks)
    out.write_text(body)
    counts = {
        "tasks": len(tasks),
        "by_axis": dict(Counter(t["axis"] for t in tasks)),
        "by_generator": dict(Counter(t["generator_id"] for t in tasks)),
        "by_difficulty": dict(Counter(t["difficulty"] for t in tasks)),
        "by_family": dict(Counter(f"{t['generator_id']}/{t['family_id']}" for t in tasks)),
        "interactive": sum(t["interactive"] for t in tasks),
    }
    manifest = {
        "schema": SCHEMA,
        "role": args.role,
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(["scripts/stages/stage-0/calibration_volume/build_task_sets.py",
                             *(argv if argv is not None else sys.argv[1:])]),
        "note": args.note,
        "file": str(out.relative_to(REPO)),
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "content_ids_sha256": hashlib.sha256(
            "\n".join(sorted(t["content_sha256"] for t in tasks)).encode()).hexdigest(),
        "per_cell": args.per_cell, "per_cell_interactive": per_int,
        "seed_range": list(R.SEED_RANGES[args.role]),
        "split_salt": R.SPLIT_SALT, "cal_fraction": R.CAL_FRACTION,
        "eval_only_generators": sorted(R.EVAL_ONLY_GENERATORS),
        "family_split": R.split_table(),
        "generator_source": generator_source_digest(),
        "counts": counts,
        "isolation": {"checked": checked, "overlaps": report},
        "license": "project-generated procedural tasks; no external dataset, no ARC-AGI material",
        "authorizes": "nothing",
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out.relative_to(REPO)}: {counts['tasks']} tasks "
          f"({counts['interactive']} interactive); isolation checks {len(checked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
