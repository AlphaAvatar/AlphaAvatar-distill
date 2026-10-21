#!/usr/bin/env python3
"""Calibrate the code verifier on MBPP's human reference solutions. $0, CPU.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.check_code_verifier

Before any teacher program is judged by `aadistill.data.code_exec`, the
verifier must accept the dataset's own correct programs and reject broken
ones. This runs every non-test MBPP task (train, validation, prompt -- the test
split, tasks 11-510, is never read) from the local HF cache: the reference
solution must pass its own asserts, and a mutant whose first `return` returns
None must fail them. The reference code is human-written, so running it on the
dev box is safe; teacher-written code is verified on the generation pod.
Writes `logs/stages/stage-2/reasoning_data/audit/code_verifier_check.json`.
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A

from aadistill.data.code_exec import Limits, run_asserts

OUT = A.LOG_DIR / "audit/code_verifier_check.json"
CACHE = Path.home() / ".cache/huggingface/hub/datasets--google-research-datasets--mbpp/snapshots"
SPLITS = ("train", "validation", "prompt")
LIMITS = Limits(wall_seconds=10, cpu_seconds=5, memory_mb=512)


def main() -> int:
    import pyarrow.parquet as pq

    snap = sorted(glob.glob(str(CACHE / "*")))[-1]
    rows, failing = {}, []
    for split in SPLITS:
        path = Path(snap) / "full" / f"{split}-00000-of-00001.parquet"
        tasks = pq.read_table(path).to_pylist()
        gold_pass = mutant_reject = mutants = 0
        for t in tasks:
            setup = t.get("test_setup_code") or ""
            if run_asserts(t["code"], t["test_list"], setup=setup, limits=LIMITS)["passed"]:
                gold_pass += 1
            else:
                failing.append({"split": split, "task_id": t["task_id"]})
            if "return" in t["code"]:
                mutants += 1
                broken = t["code"].replace("return", "return None and", 1)
                mutant_reject += not run_asserts(broken, t["test_list"], setup=setup, limits=LIMITS)["passed"]
        rows[split] = {"tasks": len(tasks), "reference_passes": gold_pass,
                       "mutants": mutants, "mutants_rejected": mutant_reject,
                       "file_sha256": A.sha256_file(path)}
    doc = {"schema": "aadistill.stage2.reasoning_data.code_verifier_check/v1",
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": f"google-research-datasets/mbpp @ {Path(snap).name} (local HF cache); test split not read",
           "verifier": "aadistill.data.code_exec.run_asserts", "limits": LIMITS.__dict__,
           "per_split": rows, "reference_failures": failing,
           "reading": "a task whose reference fails its own tests is unverifiable and is excluded from the code stratum; a mutant that survives shows the dataset's tests cannot see that change",
           "authorizes": "nothing"}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), {k: (v["reference_passes"], v["tasks"]) for k, v in rows.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
