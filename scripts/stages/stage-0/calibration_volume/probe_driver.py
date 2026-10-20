#!/usr/bin/env python3
"""On-pod driver of one Top-4 probe session: per unit, recovery -> eval-set-1.

    cd /workspace/aad && PYTHONPATH=src:scripts /opt/train/bin/python \
        scripts/stages/stage-0/calibration_volume/probe_driver.py \
        --run-id <run> --auth <authorization.json> --plan probes_d1 \
        --status <file> --volume /runpod-volume

One unit of the 2026-10-20 extension (`plans/top4_extension.json`) is one
candidate -- a dose's distinct Top-k leaf, staged from the relay by the setup
and re-hashed here -- recovered by the frozen recipe at one seed and measured on
every item of eval-set-1. The recovery and the measurement are the dose
session's own (`dose_driver.train_recovery`, `dose_driver.evaluate_all`), so a
probe differs from a dose's Top-1 unit only in its student, seed and GPU.

DURABILITY: each unit's recovered student and generations go to the session's
network volume as they finish. A replacement session evaluates a trained student
rather than retraining it and rescores finished generations rather than
regenerating them (P8.4); a unit with a report on the volume is not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume.dose_driver import (PREREG, SET_ID, evaluate_all,  # noqa: E402
                                                    gpu, sha, sync, train_recovery)

AUDIT = "artifacts/audit/stage0_calvol_probes"
WORK = "artifacts/stages/stage-0/calibration_volume/probes"
#: what a finished unit keeps on the volume beside its recovered student
UNIT_RECORDS = ("report.json", "rows", "recovery_products.json", "recovery_config.json",
                "recovery.log", "train_log.jsonl", "eval_fluid.log", "eval_capability.log")


def main(argv=None) -> int:
    from stages.calibration_volume import governance as G

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--volume", required=True)
    args = ap.parse_args(argv)
    plan = G.plan_doc(args.plan)
    prereg = json.loads((REPO / PREREG).read_text())
    audit = REPO / AUDIT
    audit.mkdir(parents=True, exist_ok=True)
    status = Path(args.status)
    vol = Path(args.volume) / "stage0_calvol" / prereg["study_id"] / "probes"
    env = {**os.environ, "PYTHONPATH": "src:scripts"}
    record = {"schema": "aadistill.stage0.calibration_volume.probe_session/v1",
              "run_id": args.run_id, "plan": args.plan, "volume_dir": str(vol),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       capture_output=True, text=True).stdout.strip(),
              "gpu": gpu(), "units": {}, "status": "STARTED"}

    def mark(m: str) -> None:
        with status.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        (audit / "probe_session.json").write_text(json.dumps(record, indent=1, default=str) + "\n")

    mark("DRIVER_START")
    try:
        auth = G.load_for("probes", args.auth)
        auth.require_plan(G.plan_hash(args.plan))
        record["authorization_id"] = auth.authorization_id
        from aadistill.evaluation import eval_registry as ER
        from shared import eval_sets as ES
        if ER.current_identity(ER.entry(ES.registry(), SET_ID)) != prereg["evaluation"]["eval_set_identity"]:
            raise RuntimeError(f"{SET_ID} is not at its preregistered identity")
        ES.manifest(SET_ID)                                  # bytes vs identity
        if subprocess.run(["mountpoint", "-q", args.volume]).returncode != 0:
            raise RuntimeError(f"no network volume mounted at {args.volume}")
        vol.mkdir(parents=True, exist_ok=True)
        save()
        for u in plan["units"]:
            label = u["label"]
            out, vdir = audit / label, vol / label
            out.mkdir(parents=True, exist_ok=True)
            entry = record["units"].setdefault(label, {"t0": time.time()})
            mark(f"UNIT_START:{label}")
            if (vdir / "out" / "report.json").is_file():
                sync("pull", vdir / "out", out, out / "sync.log")
                entry.update(status="ok", restored_complete=True)
            else:
                prod_path = vdir / "products.json"
                prod = json.loads(prod_path.read_text()) if prod_path.is_file() else {}
                rec_out = REPO / WORK / label / "recovery"
                if prod.get("recovery", {}).get("complete"):
                    from stages.phase_c1.autoinit_c1_driver import trained_model_dir
                    sync("pull", vdir / "recovery", rec_out, out / "sync.log")
                    model_dir = trained_model_dir(rec_out)
                    if sha(model_dir / "model.safetensors") != prod["recovery"]["model_sha256"]:
                        raise RuntimeError(f"{label}: the recovered student on the volume does "
                                           "not hash to its record")
                    entry["recovery"] = "restored from the volume"
                else:
                    cand = REPO / u["candidate_dir"]
                    shards = sorted(cand.glob("*.safetensors"))
                    if len(shards) != 1 or sha(shards[0]) != u["checkpoint_sha256"]:
                        raise RuntimeError(f"{label}: {cand} does not hold the candidate shard "
                                           f"{u['checkpoint_sha256'][:12]}")
                    cfg = out / "recovery_config.json"
                    model_dir = train_recovery(
                        cand, rec_out, cfg, out / "recovery.log", env=env, run_name=label,
                        seed=int(u["seed"]),
                        purpose=(f"Stage-0 Top-4 extension: {u['treatment']} rank {u['rank']} "
                                 f"(quality {u['quality_order']}) at seed {u['seed']}; only run "
                                 "identity, pack path, student_path and seed differ from the recipe"))
                    if (rec_out / "train_log.jsonl").is_file():
                        shutil.copyfile(rec_out / "train_log.jsonl", out / "train_log.jsonl")
                    prod["recovery"] = {"complete": True,
                                        "model_sha256": sha(model_dir / "model.safetensors"),
                                        "model_bytes": (model_dir / "model.safetensors").stat().st_size,
                                        "config_sha256": sha(cfg), "state_id": u["state_id"],
                                        "seed": int(u["seed"])}
                    entry["recovery_pushed"] = sync("push", rec_out, vdir / "recovery", out / "sync.log")
                    vdir.mkdir(parents=True, exist_ok=True)
                    prod_path.write_text(json.dumps(prod, indent=1) + "\n")
                (out / "recovery_products.json").write_text(json.dumps(prod["recovery"], indent=1) + "\n")
                rep, runs = evaluate_all(model_dir, out, vdir / "generations", label, prereg, env)
                (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
                for junk in ("package", "uncapped_in", "uncapped_out"):
                    shutil.rmtree(out / junk, ignore_errors=True)
                staged = out.parent / f".{label}.records"
                shutil.rmtree(staged, ignore_errors=True)
                staged.mkdir()
                for name in UNIT_RECORDS:
                    src = out / name
                    if src.is_dir():
                        shutil.copytree(src, staged / name)
                    elif src.is_file():
                        shutil.copyfile(src, staged / name)
                sync("push", staged, vdir / "out", out / "sync.log")
                shutil.rmtree(staged, ignore_errors=True)
                shutil.rmtree(rec_out, ignore_errors=True)   # on the volume; the next unit needs the disk
                entry.update(status="ok", generated=runs,
                             fluid_formal=rep["units"]["fluid_reasoning_720/dev_v1"].get("correct"))
            #: a unit restored from the volume was secured by the session that made
            #: it; marking it here sends the launcher after files it never pulled
            if not entry.get("restored_complete"):
                (audit / "records").mkdir(exist_ok=True)
                (audit / "records" / f"{label}.done").write_text(sha(out / "report.json") + "\n")
            entry["seconds"] = round(time.time() - entry.pop("t0"), 1)
            save()
            mark(f"UNIT_OK:{label}")
        record["status"] = "COMPLETE"
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                             "traceback": traceback.format_exc()[-6000:]}
    finally:
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        mark("ALL_DONE" if record["status"] == "COMPLETE" else "RUN_FAILED")
    return 0 if record["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
