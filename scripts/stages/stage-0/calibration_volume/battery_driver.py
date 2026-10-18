#!/usr/bin/env python3
"""On-pod driver of the Stage-0 battery session (dev calibration + teacher data).

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/battery_driver.py \
        --auth <authorization.json> --plan-hash <h> --status <status file>

Runs the plan's runs in order, each as its own `run_battery.py` subprocess under
the vLLM interpreter (one engine per model, released between models), scores
every eval run with `score_battery.py`, and writes one evidence record. A run
that finishes leaves `<label>.done` beside its records, which is what the
launcher's poll hook reads to pull it off the pod at once (P8.2.1). A failed
run is recorded and the driver continues with the next one: one model failing
must not cost the others' measurements. The session fails only if a run the
formal design depends on is missing; every outcome is in the record.
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

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume import governance as G  # noqa: E402

AUDIT = "artifacts/audit/stage0_calvol_battery"
VLLM_PY = "/opt/vllm/bin/python"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
        name, mem, drv = [x.strip() for x in out.split(",")]
        return {"name": name, "memory": mem, "driver": drv}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", required=True)
    ap.add_argument("--plan-hash", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="labels (rehearsal)")
    args = ap.parse_args(argv)

    audit = REPO / AUDIT
    (audit / "records").mkdir(parents=True, exist_ok=True)
    (audit / "scores").mkdir(parents=True, exist_ok=True)
    status = Path(args.status)

    def mark(m: str) -> None:
        with status.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    record = {"schema": "aadistill.stage0.calibration_volume.battery_record/v1",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       capture_output=True, text=True).stdout.strip(),
              "gpu": gpu(), "runs": [], "stages": {}}
    evidence = audit / "battery.json"

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=1) + "\n")

    try:
        auth = G.load_for("battery", args.auth)
        auth.require_plan(args.plan_hash)
        auth.require_plan(G.plan_hash("battery", REPO))
        plan = G.plan_doc("battery", REPO)
        for role, t in plan["tasks"].items():
            if sha(REPO / t["path"]) != t["sha256"]:
                raise SystemExit(f"{t['path']} does not hash to the plan's {t['sha256']}")
        record["stages"]["A_contract"] = {"status": "ok", "plan_hash": args.plan_hash,
                                          "authorization_id": auth.authorization_id}
    except BaseException as exc:  # noqa: BLE001
        record["stages"]["A_contract"] = {"status": "failed", "error": repr(exc)}
        save()
        mark("RUN_FAILED")
        return 2
    save()
    mark("CONTRACT_OK")

    students = {s["label"]: s for s in plan["students"]}
    env = {**os.environ, "PYTHONPATH": "src:scripts"}
    for run in plan["runs"]:
        label = run["label"]
        if args.only and label not in args.only:
            continue
        out = audit / "records" / f"{label}.jsonl"
        entry = {"label": label, "mode": run["mode"], "tasks": run["tasks"]}
        t0 = time.time()
        try:
            if run["model"] == "teacher":
                model = plan["teacher"]["id"]
                extra = ["--revision", plan["teacher"]["revision"]]
            else:
                s = students[run["model"].split(":", 1)[1]]
                model = str(REPO / s["dir"])
                for f in s["relay_files"]:
                    p = REPO / s["dir"] / Path(f["path"]).name
                    if sha(p) != f["sha256"]:
                        raise RuntimeError(f"{p} does not hash to {f['sha256']}")
                extra = []
            cmd = [VLLM_PY, "scripts/stages/stage-0/calibration_volume/run_battery.py",
                   "--model", model, "--label", label,
                   "--tasks", plan["tasks"][run["tasks"]]["path"],
                   "--mode", run["mode"], "--out", str(out), *extra]
            if run.get("trained_context"):
                cmd += ["--trained-context", str(run["trained_context"])]
            if run.get("session_limit"):
                cmd += ["--session-limit", str(run["session_limit"])]
            if run.get("seed_base") is not None:
                cmd += ["--seed-base", str(run["seed_base"])]
            log = audit / "records" / f"{label}.log"
            with log.open("w") as lf:
                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf,
                                    stderr=subprocess.STDOUT).returncode
            entry.update({"rc": rc, "seconds": round(time.time() - t0, 1)})
            if rc != 0 or not out.is_file():
                raise RuntimeError(f"run_battery exited {rc}; see {log.name}")
            entry["records_sha256"] = sha(out)
            entry["summary"] = json.loads(out.with_suffix(".summary.json").read_text())
            if run["mode"] == "eval":
                rc = subprocess.run(
                    [sys.executable, "scripts/stages/stage-0/calibration_volume/score_battery.py",
                     "--tasks", plan["tasks"][run["tasks"]]["path"], "--records", str(out),
                     "--out-dir", str(audit / "scores")], cwd=REPO, env=env,
                    capture_output=True, text=True)
                entry["score_rc"] = rc.returncode
                entry["score_stdout"] = rc.stdout[-2000:]
            entry["status"] = "ok"
            (audit / "records" / f"{label}.done").write_text(entry["records_sha256"] + "\n")
            mark(f"RUN_OK:{label}")
        except BaseException as exc:  # noqa: BLE001
            entry.update({"status": "failed", "error": repr(exc),
                          "traceback": traceback.format_exc()[-4000:],
                          "seconds": round(time.time() - t0, 1)})
            mark(f"RUN_FAILED_UNIT:{label}")
        record["runs"].append(entry)
        save()

    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failed = [r["label"] for r in record["runs"] if r["status"] != "ok"]
    record["terminal"] = "ALL_DONE" if not failed else "RUNS_FAILED"
    record["failed_runs"] = failed
    save()
    mark("ALL_DONE" if not failed else "RUN_FAILED")
    return 0 if not failed else 3


if __name__ == "__main__":
    raise SystemExit(main())
