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

DURABILITY: each unit's recovered student (with its config and logs) and its
generations go to the session's network volume as they finish, the unit's
report.json last. A replacement session evaluates a trained student rather than
retraining it and rescores finished generations rather than regenerating them
(P8.4); a unit with a report on the volume is not rerun, only brought home again,
since the session that made it may have died before its launcher fetched it.
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
#: what a finished unit keeps on the volume beside its recovered student;
#: report.json is pushed last, so its presence there means the rest landed
UNIT_RECORDS = ("rows", "recovery_products.json", "recovery_config.json",
                "recovery.log", "train_log.jsonl", "eval_fluid.log", "eval_capability.log")
#: pushed as soon as the recovery finishes: a replacement session that restores
#: the student restores these with it rather than losing them with the pod
RECOVERY_RECORDS = ("recovery_config.json", "recovery.log", "train_log.jsonl")


def push_files(out: Path, names, dst: Path, log: Path) -> dict:
    """Push `names` from `out` to `dst` through a staging dir; the result has
    "error" when nothing durable can be assumed."""
    staged = out.parent / f".{out.name}.push"
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir()
    for name in names:
        src = out / name
        if src.is_dir():
            shutil.copytree(src, staged / name)
        elif src.is_file():
            shutil.copyfile(src, staged / name)
    res = sync("push", staged, dst, log)
    shutil.rmtree(staged, ignore_errors=True)
    if "copied" not in res and "error" not in res:
        res = {**res, "error": "volume_sync reported no result"}
    return res


def push_student(rec_out: Path, dst: Path, log: Path) -> dict:
    """Push the recovered student as a replacement restores it -- the model and the
    pointer to it -- and not the trainer state, which nothing after training reads
    (P8.4; 3.5 of a recovery's 5.6 GB)."""
    from stages.phase_c1.autoinit_c1_driver import trained_model_dir
    model_dir = trained_model_dir(rec_out)
    staged = rec_out.parent / ".student.push"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(model_dir, staged / model_dir.relative_to(rec_out), copy_function=os.link)
    for name in ("checkpoints/latest.txt", "train_log.jsonl"):
        if (rec_out / name).is_file():
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
            os.link(rec_out / name, staged / name)
    res = sync("push", staged, dst, log)
    shutil.rmtree(staged, ignore_errors=True)
    if "copied" not in res and "error" not in res:
        res = {**res, "error": "volume_sync reported no result"}
    return res


def restore_recovery(vdir: Path, rec_out: Path, out: Path, want: str) -> Path:
    """The recovered student from the volume, verified against its record."""
    from stages.phase_c1.autoinit_c1_driver import trained_model_dir
    sync("pull", vdir / "recovery", rec_out, out / "sync.log")
    sync("pull", vdir / "recovery_records", out, out / "sync.log")
    model_dir = trained_model_dir(rec_out)
    if sha(model_dir / "model.safetensors") != want:
        raise RuntimeError("the recovered student on the volume does not hash to its record")
    return model_dir


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
                #: the session that made it may have died before its launcher
                #: fetched the records, so this one brings them home too
                sync("pull", vdir / "out", out, out / "sync.log")
                sync("pull", vdir / "generations", out / "generations", out / "sync.log")
                entry.update(status="ok", restored_complete=True)
            else:
                prod_path = vdir / "products.json"
                prod = json.loads(prod_path.read_text()) if prod_path.is_file() else {}
                rec_out = REPO / WORK / label / "recovery"
                model_dir = None
                if prod.get("recovery", {}).get("complete"):
                    try:
                        model_dir = restore_recovery(vdir, rec_out, out,
                                                     prod["recovery"]["model_sha256"])
                        entry["recovery"] = "restored from the volume"
                    except Exception as exc:  # noqa: BLE001 -- not durable after all: retrain
                        entry["recovery_restore_failed"] = repr(exc)[:300]
                        prod.pop("recovery", None)
                if model_dir is None:
                    #: generations on the volume came from a student this unit no
                    #: longer has; a new one never inherits them (no pooling)
                    if (vdir / "generations").is_dir():
                        moved = vdir / f"generations.superseded.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                        (vdir / "generations").rename(moved)
                        entry["generations_superseded"] = moved.name
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
                    pushed = [push_student(rec_out, vdir / "recovery", out / "sync.log"),
                              push_files(out, RECOVERY_RECORDS, vdir / "recovery_records",
                                         out / "sync.log")]
                    entry["recovery_pushed"] = pushed
                    #: only a recovery that reached the volume is one a replacement may restore
                    if not any("error" in p for p in pushed):
                        vdir.mkdir(parents=True, exist_ok=True)
                        prod_path.write_text(json.dumps(prod, indent=1) + "\n")
                (out / "recovery_products.json").write_text(json.dumps(prod["recovery"], indent=1) + "\n")
                rep, runs = evaluate_all(model_dir, out, vdir / "generations", label, prereg, env)
                (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
                for junk in ("package", "uncapped_in", "uncapped_out"):
                    shutil.rmtree(out / junk, ignore_errors=True)
                pushed = push_files(out, UNIT_RECORDS, vdir / "out", out / "sync.log")
                if "error" not in pushed:
                    pushed = push_files(out, ("report.json",), vdir / "out", out / "sync.log")
                entry["records_pushed"] = pushed
                shutil.rmtree(rec_out, ignore_errors=True)   # on the volume; the next unit needs the disk
                entry.update(status="ok", generated=runs,
                             fluid_formal=rep["units"]["fluid_reasoning_720/dev_v1"].get("correct"))
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
