#!/usr/bin/env python3
"""On-pod driver of H1's teacher generation (P1): the plan's inputs, two candidate passes.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        -m stages.reasoning_data.generation_driver --plan <generation_plan.json> \
        --inputs-dir <dir> --authorization <authorization.json> --status <file> --out <dir>

Generation writes TEXT: teacher programs are never executed here (the
verifying sandbox is on the dev box, `code_sandbox.py`), so this pod's
credentials, network and volume never meet model-written code.

1. CONTRACT: the authorization is loaded under `POLICY`, must grant
   ``teacher_generation`` and be bound to this plan file's sha256; every input
   and pool file must hash to what the plan records. Nothing runs otherwise.
2. PASS 0: each input group (one per system message) through the Stage-0
   generator, `run_battery.py --mode generate` (the teacher's recommended
   sampling, a seeded request per turn, real environment loops), at the plan's
   session limit, seed base ``base + 0``.
3. Every record of a pass is TAGGED with ``candidate_index`` = the pass and
   the pass's seed base and input/raw hashes, written beside the raw output,
   and the group's ``.done`` marker (its sha256) is written at once: that is
   what the launcher's poll hook pulls off the pod (P8.2.1), so a finished pass
   survives a later failure.
4. PASS 1: the second candidate, only for tasks whose first failed, judged here
   by the adapter's own `candidate()` (pure Python: the battery scorer, the
   environment replay, comparators and checkers); a task whose verifier needs
   the dev-box sandbox (`generation_plan.DEV_BOX_ONLY`) gets none.
5. TERMINAL: one evidence record, ``MARKER:ALL_DONE`` or ``MARKER:RUN_FAILED``
   and the exit code, in one place. A failed group is recorded and the others
   go on; nothing already marked done is touched. Teardown is the launcher's.

A rerun on the same --out skips every group whose tagged records and marker
are present and hash-consistent: finished generation is never paid for twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import generation_plan as GP

from aadistill.governance.authorization import ActionPolicy, AuthorizationError, SpendAuthorization

VLLM_PY = "/opt/vllm/bin/python"
RUNNER = "scripts/stages/stage-0/calibration_volume/run_battery.py"
SEED_BASE = 20261022
UNIT_OK, UNIT_FAILED = "UNIT_OK", "UNIT_FAILED"      # contain neither session marker
POLICY = ActionPolicy(
    policy_id="h1_teacher_generation", allowed=frozenset({"teacher_generation"}),
    wire_schema="aadistill.stage2.reasoning_data.generation_authorization/v1",
    plan_hash_key="generation_plan_sha256",
    enforcement="generation_driver runs only the plan this authorization is bound to")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize(path: Path, plan_sha: str) -> dict:
    auth = SpendAuthorization.load(path, policy=POLICY)
    if not auth.allows("teacher_generation"):
        raise AuthorizationError(f"{path} does not grant teacher_generation")
    auth.require_plan(plan_sha)
    return {"authorization_id": auth.authorization_id, "hard_cap_usd": auth.hard_cap_usd}


def tag(raw: Path, out: Path, k: int, seed_base: int, input_sha: str) -> int:
    meta = {"index": k, "seed_base": seed_base, "input_sha256": input_sha, "raw_records_sha256": sha(raw)}
    n = 0
    with open(out, "w") as f:
        for rec in A.jsonl(raw):
            f.write(json.dumps({**rec, "candidate_index": k, "generation_pass": meta}, ensure_ascii=False) + "\n")
            n += 1
    return n


def second_pass(tasks: dict[str, dict], tagged: Path) -> list[str]:
    """Task ids whose first candidate the adapter rejects, where a second
    candidate is allowed (the verdict is computable here)."""
    from stages.reasoning_data.generation_adapter import candidate
    retry = []
    for rec in A.jsonl(tagged):
        task = tasks[rec["task_id"]]
        if GP.verifier_of(task)[0] in GP.DEV_BOX_ONLY:
            continue
        if not candidate(task, rec, rec["candidate_index"]).accepted_for_training:
            retry.append(rec["task_id"])
    return retry


def run_battery(cmd: list[str], log: Path) -> int:
    """The generator executes no model output, so it keeps the session's
    environment (hub cache and access), as battery_driver's runs do."""
    with log.open("w") as lf:
        return subprocess.run(cmd, cwd=A.REPO, stdout=lf, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": "src:scripts"}).returncode


def main(argv=None, *, run=run_battery, authorize=authorize) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="teacher dir; default: the protocol's hub id + revision")
    args = ap.parse_args(argv)
    out, status = Path(args.out), Path(args.status)
    out.mkdir(parents=True, exist_ok=True)

    def mark(m: str) -> None:
        with status.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    record = {"schema": "aadistill.stage2.reasoning_data.generation_session/v1",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "groups": [], "passes": {}}
    evidence = out / "generation_session.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=1) + "\n")

    try:
        plan_path = Path(args.plan)
        plan = json.loads(plan_path.read_text())
        record["contract"] = {"plan_sha256": sha(plan_path), **authorize(Path(args.authorization), sha(plan_path))}
        inputs = Path(args.inputs_dir)
        for f in plan["inputs"]:
            if sha(inputs / f["file"]) != f["sha256"]:
                raise SystemExit(f"{f['file']} does not hash to the plan's {f['sha256'][:12]}")
        tasks = {}
        for p in plan["pools"]:
            if sha(A.REPO / p["path"]) != p["sha256"]:
                raise SystemExit(f"{p['path']} does not hash to the plan's {p['sha256'][:12]}")
            tasks.update({t["task_id"]: t for t in A.jsonl(A.REPO / p["path"])})
        proto = A.read_json(A.CONFIG_DIR / "generation_protocol.json")["teacher"]
        record["contract"]["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        record["contract"] = {"status": "failed", "error": repr(exc)}
        save()
        mark("RUN_FAILED")
        return 2
    save()
    mark("CONTRACT_OK")
    model = ["--model", args.model] if args.model else ["--model", proto["model"], "--revision", proto["revision"]]

    retry: dict[str, list[str]] = {}
    for k in (0, 1):
        for f in plan["inputs"]:
            label = f"pass{k}_{Path(f['file']).stem}"
            tagged, done = out / f"{label}.jsonl", out / f"{label}.done"
            entry = {"label": label, "pass": k, "seed_base": SEED_BASE + k}
            t0 = time.time()
            try:
                src = inputs / f["file"]
                if k == 1:
                    if label.replace("pass1_", "pass0_") not in retry:
                        continue                       # its first pass did not finish
                    ids = set(retry[label.replace("pass1_", "pass0_")])
                    if not ids:
                        continue
                    src = out / f"{label}.input.jsonl"
                    src.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                                           for r in A.jsonl(inputs / f["file"]) if r["task_id"] in ids))
                if done.is_file() and tagged.is_file() and done.read_text().strip() == sha(tagged):
                    entry.update(status="ok", resumed=True, records_sha256=sha(tagged))
                else:
                    raw = out / f"{label}.raw.jsonl"
                    cmd = [VLLM_PY, RUNNER, *model, "--label", label, "--tasks", str(src), "--mode", "generate",
                           "--session-limit", str(plan["session_limit"]), "--seed-base", str(SEED_BASE + k),
                           "--system", f["system"], "--out", str(raw)]
                    rc = run(cmd, out / f"{label}.log")
                    if rc != 0 or not raw.is_file():
                        raise RuntimeError(f"run_battery exited {rc}; see {label}.log")
                    entry["records"] = tag(raw, tagged, k, SEED_BASE + k, sha(src))
                    entry["records_sha256"] = sha(tagged)
                    done.write_text(entry["records_sha256"] + "\n")
                    entry["status"] = "ok"
                if k == 0:
                    retry[label] = second_pass(tasks, tagged)
                    entry["second_candidate_tasks"] = len(retry[label])
                entry["seconds"] = round(time.time() - t0, 1)
                mark(f"{UNIT_OK}:{label}")
            except BaseException as exc:  # noqa: BLE001
                entry.update(status="failed", error=repr(exc), traceback=traceback.format_exc()[-3000:],
                             seconds=round(time.time() - t0, 1))
                mark(f"{UNIT_FAILED}:{label}")
            record["groups"].append(entry)
            save()
    failed = [g["label"] for g in record["groups"] if g["status"] != "ok"]
    record.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), failed=failed,
                  terminal="ALL_DONE" if not failed else "RUNS_FAILED")
    save()
    mark("ALL_DONE" if not failed else "RUN_FAILED")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
