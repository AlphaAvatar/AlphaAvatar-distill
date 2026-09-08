#!/usr/bin/env python3
"""Record a snapshot of an experiment's live executable closure.

    PYTHONPATH=src:scripts python scripts/architecture/derive_closure.py --write

The derivation itself lives in `aadistill.governance.closure` and runs live
wherever an executable identity is needed, so nothing depends on this script
having been run. What this writes is a *snapshot*: a committed record of what
the set looked like at a point in time, useful for reviewing how the closure
moves across a change. It is compared against, never trusted — a recorded file
list goes stale on the first edit, and one that still loads reports a confident
identity for the wrong set of files.

The entry points are owned by each experiment, not by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.governance.closure import compare, derive  # noqa: E402

OUT_DIR = "configs/experiments"


def _phase_c1() -> tuple[tuple[str, ...], tuple[str, ...]]:
    from experiments.phase_c1.authorization import (
        C1_DECLARED_INPUTS, C1_ENTRY_POINTS)
    return C1_ENTRY_POINTS, C1_DECLARED_INPUTS


#: experiment id -> callable returning (entry points, declared non-python inputs)
EXPERIMENTS = {"phase_c1": _phase_c1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="phase_c1", choices=sorted(EXPERIMENTS))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    entries, declared = EXPERIMENTS[args.experiment]()
    doc = derive(REPO, args.experiment, entries, declared)
    print(f"{args.experiment}: {doc['n_files']} files, digest {doc['digest'][:16]}…")

    by_area: dict[str, int] = {}
    for row in doc["files"]:
        parts = row["path"].split("/")
        key = parts[1] if parts[0] in ("src", "scripts") else parts[0]
        by_area[key] = by_area.get(key, 0) + 1
    print("  by area: " + ", ".join(f"{k}={v}" for k, v in sorted(by_area.items())))

    out = REPO / OUT_DIR / args.experiment / "executable_closure.json"
    if out.is_file():
        drift = compare(doc, json.loads(out.read_text()))
        if drift["digest_matches"]:
            print("  snapshot: current")
        else:
            print(f"  snapshot drift: +{len(drift['added_files'])} file(s), "
                  f"-{len(drift['removed_files'])} file(s), "
                  f"{len(drift['changed_files'])} edited")
            for path in drift["added_files"][:8]:
                print(f"    + {path}")
            for path in drift["removed_files"][:8]:
                print(f"    - {path}")
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
