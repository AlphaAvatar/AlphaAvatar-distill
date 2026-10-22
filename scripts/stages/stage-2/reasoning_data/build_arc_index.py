#!/usr/bin/env python3
"""Fingerprints of every ARC-AGI-1 grid, for isolation rule R7. $0, CPU.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.build_arc_index \
        --wheel <arckit-0.1.0-py3-none-any.whl>

The reference is the ARC-AGI-1 data arckit 0.1.0 bundles (``arc1.json``: the
400 training and 400 evaluation tasks; ``kaggle2024.json``: the ARC Prize 2024
copies), read from the pinned wheel -- whose sha256 is checked -- without
installing or importing it. Only FINGERPRINTS are kept: a 16-hex sha256 prefix
of each grid's canonical form, mapped to the split and task ids it occurs in.
A grid cannot be rebuilt from its fingerprint, so the index carries no ARC
content into the repository; the wheel itself stays outside it.

Writes `data/stages/stage-2/reasoning_data/isolation/arc_grid_index.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from stages.reasoning_data import assets as A

OUT = A.REPO / "data/stages/stage-2/reasoning_data/isolation/arc_grid_index.json"
WHEEL_SHA256 = "5a22619309d2c29b04796eadf2a58792e563d6b811ee021e5d8ca97880ebbe80"
FILES = ("arckit/data/arc1.json", "arckit/data/kaggle2024.json")


def canonical(grid) -> str:
    """Rows of digits joined by '/': the same form `arc_grids` extracts from text."""
    return "/".join("".join(str(int(c)) for c in row) for row in grid)


def fingerprint(grid) -> str:
    return hashlib.sha256(canonical(grid).encode()).hexdigest()[:16]


def task_grids(task: dict):
    for pair in [*task.get("train", []), *task.get("test", [])]:
        for k in ("input", "output"):
            if pair.get(k) is not None:
                yield pair[k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheel", required=True)
    args = ap.parse_args(argv)
    wheel = Path(args.wheel)
    got = A.sha256_file(wheel)
    if got != WHEEL_SHA256:
        raise SystemExit(f"wheel sha256 {got} is not the pinned {WHEEL_SHA256}")
    index: dict[str, set] = defaultdict(set)
    files, counts = {}, defaultdict(int)
    with zipfile.ZipFile(wheel) as z:
        for name in FILES:
            raw = z.read(name)
            files[name] = hashlib.sha256(raw).hexdigest()
            for split, tasks in json.loads(raw).items():
                for tid, task in tasks.items():
                    counts[split] += 1
                    for g in task_grids(task):
                        index[fingerprint(g)].add(f"{split}:{tid}")
    doc = {"schema": "aadistill.stage2.reasoning_data.arc_grid_index/v1",
           "source": {"package": "arckit 0.1.0 (PyPI)", "wheel_sha256": WHEEL_SHA256, "files_sha256": files},
           "canonical_form": "rows of single digits joined by '/' (build_arc_index.canonical)",
           "fingerprint": "sha256(canonical)[:16]",
           "tasks_read": dict(counts), "grids": len(index),
           "index": {h: sorted(v) for h, v in sorted(index.items())},
           "use": "isolation rule R7 only (isolation_check.arc_matches); never training, calibration or development data"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(OUT.relative_to(A.REPO), doc["tasks_read"], doc["grids"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
