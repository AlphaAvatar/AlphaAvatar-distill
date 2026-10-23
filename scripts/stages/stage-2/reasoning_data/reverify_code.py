#!/usr/bin/env python3
"""Re-verify one run's saved MBPP teacher programs under the current code verifier. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.reverify_code \
        --run-id h1gen_20261022_190653 --store <secured store> --previous-commit <sha>

`aadistill.data.code_exec` v2's assert driver ran the candidate and its tests
in one interpreter through the live ``builtins.exec`` and kept its report in
module globals, so a candidate could rebind ``exec``/``compile``, its
namespace's ``__builtins__`` or the report itself and be recorded as passing
tests that never ran (maintainer review, 2026-10-23). v3 binds what the
verdict depends on before any candidate code runs and fails a run whose
program or tests rebound a builtin, the builtins the tests resolve through, a
trace or profile hook, or left a thread running (``tampered``).

Nothing is regenerated and no record is changed. Every saved record of a
``code_asserts`` task is judged twice inside the same probed sandbox: by the
verifier as it stood at ``--previous-commit`` (read from git: the one P1v and
the corpus feasibility checks used) and by the one on disk, with the task's
own tests, setup and `supporting_verify.CODE_LIMITS`. The result is a new
record with its own identity -- both verifiers' source hashes, the store's
shards, every candidate's verdict under each, the candidates whose verdict
moved, and each candidate's acceptance under the current verifier.

Writes ``logs/stages/stage-2/reasoning_data/audit/code_reverification_<run>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import code_sandbox as CS
from stages.reasoning_data import generation_adapter as G
from stages.reasoning_data import governance as GOV
from stages.reasoning_data import p1v
from stages.reasoning_data import supporting_verify as SV

from aadistill.data import code_exec as CE

VERIFIER = "src/aadistill/data/code_exec.py"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def previous_verifier(commit: str):
    """`code_exec` exactly as it stood at `commit`, loaded under its own name."""
    src = subprocess.run(["git", "show", f"{commit}:{VERIFIER}"], cwd=A.REPO, capture_output=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp(prefix="aad_prev_verifier_")) / "code_exec_previous.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("code_exec_previous", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["code_exec_previous"] = mod
    spec.loader.exec_module(mod)
    return mod, _sha(src)


def _verdict(res: dict) -> dict:
    return {k: res.get(k) for k in ("passed", "status", "tests_run", "tests_passed", "tests_total", "detail")}


def reverify(run_id: str, store: Path, commit: str) -> dict:
    auth = GOV.load(A.REPO / GOV.authorization_path(run_id))
    plan = A.read_json(A.REPO / GOV.plan_of(auth))
    recs, ins = p1v.inputs(store, auth.plan_hash)
    tasks = {t["task_id"]: t for p in plan["pools"] for t in A.jsonl(A.REPO / p["path"])}
    prev, prev_sha = previous_verifier(commit)
    sandbox, probe = CS.verified_sandbox()
    rows = []
    for rec in sorted((r for r in recs if tasks[r["task_id"]].get("verifier") == "code_asserts"),
                      key=lambda r: (r["task_id"], r["candidate_index"])):
        task = tasks[rec["task_id"]]
        row = {"task_id": rec["task_id"], "candidate_index": rec["candidate_index"],
               "record_sha256": G.record_sha256(rec)}
        parts = A.split_raw(rec["turns"][-1].get("raw", "")) if rec["turns"] else None
        code = SV.code_block(parts[1])[0] if parts else None
        if code is None:
            row["not_executed"] = "think not closed" if parts is None else "no fenced code block in the final answer"
        else:
            kw = {"setup": task.get("test_setup_code") or "", "limits": SV.CODE_LIMITS, "sandbox": sandbox}
            row["code_sha256"] = _sha(code.encode())
            row["previous"] = _verdict(prev.run_asserts(code, task["test_list"], **kw))
            row["current"] = _verdict(CE.run_asserts(code, task["test_list"], **kw))
        row["accepted_under_current"] = G.candidate(task, rec, rec["candidate_index"]).accepted_for_training
        rows.append(row)
    ran = [r for r in rows if "current" in r]
    moved = [{"task_id": r["task_id"], "candidate_index": r["candidate_index"],
              "previous": r["previous"]["status"], "current": r["current"]["status"],
              "detail": r["current"]["detail"]} for r in ran if r["previous"]["passed"] != r["current"]["passed"]]
    return {
        "schema": "aadistill.stage2.reasoning_data.code_reverification/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id, "launch_identity_sha256": auth.plan_hash,
        "store": {"shards": ins["shards"], "records": ins["records"],
                  "shards_sha256": _sha(json.dumps(ins["shard_sha256"], sort_keys=True).encode())},
        "verifiers": {"previous": {"commit": commit, "path": VERIFIER, "source_sha256": prev_sha,
                                   "schema": prev.CODE_EXEC_SCHEMA},
                      "current": {"path": VERIFIER, "source_sha256": _sha((A.REPO / VERIFIER).read_bytes()),
                                  "schema": CE.CODE_EXEC_SCHEMA}},
        "script_sha256": _sha(Path(__file__).read_bytes()),
        "sandbox": probe, "limits": vars(SV.CODE_LIMITS),
        "summary": {"code_asserts_records": len(rows), "executed": len(ran),
                    "not_executed": len(rows) - len(ran),
                    "previous_passed": sum(r["previous"]["passed"] for r in ran),
                    "current_passed": sum(r["current"]["passed"] for r in ran),
                    "verdicts_moved": len(moved),
                    "accepted_under_current": sum(r["accepted_under_current"] for r in rows)},
        "moved": moved, "rows": rows,
        "reading": ("the saved records are judged again, not regenerated; a moved verdict is the previous "
                    "verifier's error on that record, and every later reader of the run's accepted units "
                    "(p1v.verify_run, corpus_feasibility) re-runs the current verifier itself"),
        "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--previous-commit", required=True, help="the commit whose code_exec judged the run before")
    args = ap.parse_args(argv)
    doc = reverify(args.run_id, Path(args.store), args.previous_commit)
    out = A.LOG_DIR / f"audit/code_reverification_{args.run_id}.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(out.relative_to(A.REPO), json.dumps(doc["summary"]), [m["task_id"] for m in doc["moved"]][:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
