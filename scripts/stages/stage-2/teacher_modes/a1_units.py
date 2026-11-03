#!/usr/bin/env python3
"""On-pod driver of A1's two units (Research A v2, A1 phases C and D): each seed trained from the hybrid full search's
Top-1, then formally evaluated, seed 20260726 first.

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-2/teacher_modes/a1_units.py --authorization <a.json> \\
        --durable-root <volume>/teacher_modes --status <file>

The units are `training_plan_a1.json`'s, written on the dev box from the committed search record: each config is
the frozen recipe (E1's, three epochs, batch size 2 of 8,192-token packed sequences, AdamW 5e-5, CE 0.25 + KD 1.0 at
the assistant positions with Top-200 + tail, per-turn isolation, the hybrid in thinking mode as the online teacher)
with its ``lineage`` block. Before a step runs:
* the hybrid's tokenizer files are staged from its snapshot and refused unless each file's sha256 is the pinned one;
* the launch admission (`training_plan.admit`) checks every lineage fact the pod can observe -- the Top-1's bytes and
  effective config on the volume, the search record's root and checkpoint, the cached teacher snapshot, the pack, the
  tokenizer through the trainer's own contract -- and the trainer checks them again, the loaded model after loading.

STATE, from the durable root alone, so a replacement session continues: a unit with its formal report is done; a
unit whose training completed (``run_completion.json``) is evaluated; a unit with save points resumes at its latest
(the checkpoint's lineage stamp checked); otherwise it trains from the Top-1. A unit's training stops at its deadline
(``--unit-deadline-hours``); a stopped or failed unit stops the session -- the second seed never starts on a failure
of the first, and is never skipped because of the first one's result.

The evaluation is the frozen eval-set-1 v2 with nucleus_v1 (`h1pa_driver.evaluate`), in thinking mode with the
hybrid's tokenizer and template; every item of every unit is generated.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/training_plan_a1.json"
PREREG = REPO / "logs/stages/stage-2/teacher_modes/plans/token_matched_preregistration.json"
SCHEMA = "aadistill.stage2.teacher_modes.a1_units/v1"
TEMPLATE = {"enable_thinking": True}
#: the packed round's arms (`--arm`, `--plan`, `--kind`): each in its own mode and template; C1-6cat's teacher is
#: Thinking-2507, whose template opens the think block itself. A1 (the defaults) is unchanged.
ARM_TEMPLATE = {"A1": TEMPLATE, "A1-6cat": {"enable_thinking": True}, "A2-6cat": {"enable_thinking": False},
                "C1-6cat": {}}
HYBRID_ARMS = frozenset({"A1", "A1-6cat", "A2-6cat"})


def units(plan: dict, arm: str = "A1") -> list[dict]:
    """The plan's units of ``arm`` in their order (seed 20260726 first)."""
    us = [u for u in plan["units"] if u["arm"] == arm]
    return sorted(us, key=lambda u: plan["seed_order"].index(u["seed"]))


def evaluation_spec() -> dict:
    pr = json.loads(PREREG.read_text())["evaluation"]
    return {"set": pr["set"]["eval_set_id"], "manifest_sha256": pr["set"]["manifest_sha256"], "trained_context": 8192,
            "action_protocol": pr["set"]["action_protocol"],
            "decoding": {"path": pr["decoding"]["path"], "sha256": pr["decoding"]["sha256"]}}


def state(out_dir: Path, eval_dir: Path) -> str:
    if (eval_dir / "formal_report.json").is_file():
        return "done"
    if (out_dir / "run_completion.json").is_file() or list(out_dir.glob("run_completion_resume_step*.json")):
        return "trained"
    if (out_dir / "checkpoints" / "latest.txt").is_file():
        return "resume"
    return "fresh"


def final_model(out_dir: Path, steps: int) -> Path:
    d = out_dir / "checkpoints" / f"step_{steps:06d}" / "model"
    if not (d / "model.safetensors").is_file():
        raise RuntimeError(f"{d} holds no final weights")
    return d


def train(cfg_path: Path, *, resume: bool, deadline_s: float, log: Path) -> int:
    cmd = [sys.executable, "scripts/shared/training/train_stage3.py", "--config", str(cfg_path)]
    if resume:
        cmd += ["--resume", "latest"]
    env = {**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}
    with open(log, "a") as fh:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            return p.wait(timeout=deadline_s)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
            return 124


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--authorization", required=True)
    ap.add_argument("--durable-root", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--unit-deadline-hours", type=float, required=True)
    ap.add_argument("--kind", default="train_a1")
    ap.add_argument("--plan", default=str(PLAN.relative_to(REPO)))
    ap.add_argument("--arm", default="A1", choices=sorted(ARM_TEMPLATE))
    ap.add_argument("--units-dir", default="v2_units")
    a = ap.parse_args(argv)
    durable, status = Path(a.durable_root), Path(a.status)
    record = {"schema": SCHEMA, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": {},
              "arm": a.arm, "kind": a.kind}
    evidence = durable / ("a1_units.json" if a.arm == "A1" else f"{a.arm}_units.json")

    def mark(m: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} MARKER:{m}\n")
        print(f"MARKER:{m}", flush=True)

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=1, default=str) + "\n")

    ok = True
    try:
        from stages.teacher_modes import token_matched_phase as TM
        from stages.teacher_modes import training_plan as TP
        record["authorization"] = TM.require_on_pod(Path(a.authorization), a.kind, REPO).get("authorization_id")
        plan = json.loads((REPO / a.plan).read_text())
        from shared.pod.teacher_cache import local_snapshot
        us = units(plan, a.arm)
        if not us:
            raise RuntimeError(f"{a.plan} holds no unit of arm {a.arm}")
        t = plan.get("teacher") or us[0]["config"]["teacher"]
        snap = Path(local_snapshot(t["model_id"], t["revision"]))
        if a.arm in HYBRID_ARMS:
            record["tokenizer"] = TP.stage_tokenizers(snap)
        mark("CONTRACT_OK")
        for u in us:
            name, cfg = u["unit"], u["config"]
            row = record["units"].setdefault(name, {})
            out_dir, eval_dir = Path(cfg["out_dir"]), durable / a.units_dir / name / "evaluation"
            out_dir.mkdir(parents=True, exist_ok=True)
            st = state(out_dir, eval_dir)
            row["state_at_start"] = st
            if st == "done":
                continue
            if st in ("fresh", "resume"):
                TP.admit([{"unit": name, "config": cfg}], snapshots={t["model_id"]: snap})
                row["admitted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                cfg_path = out_dir / "config.json"
                cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
                save()
                mark(f"TRAIN_START:{name}")
                t0 = time.time()
                rc = train(cfg_path, resume=st == "resume", deadline_s=a.unit_deadline_hours * 3600,
                           log=out_dir / "train_stdout.log")
                row.update(train_rc=rc, train_seconds=round(time.time() - t0, 1))
                save()
                if rc != 0:
                    raise RuntimeError(f"{name}: training exited {rc} ({'deadline' if rc == 124 else 'failure'})")
                mark(f"TRAINED:{name}")
            steps = int(cfg["schedule"]["total_steps"])
            model_dir = final_model(out_dir, steps)
            from aadistill.infrastructure.manifest import sha256_file
            row["final_model_safetensors_sha256"] = sha256_file(model_dir / "model.safetensors")
            save()
            mark(f"EVAL_START:{name}")
            from stages.reasoning_data import h1pa_driver as HD
            ident = json.loads((REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json").read_text())
            if a.arm in HYBRID_ARMS:
                pkg = {"tokenizer_source": str(REPO / TP.TOKENIZERS["hybrid"]),
                       "sidecar_sha256": {n: ident["files_sha256"][n] for n in ("tokenizer.json", "tokenizer_config.json")}}
            else:                       #: Thinking-2507's tokenizer, the one its lineage trained with
                src = REPO / TP.TOKENIZERS["thinking_2507"]
                pkg = {"tokenizer_source": str(src),
                       "sidecar_sha256": {n: sha256_file(src / n) for n in ("tokenizer.json", "tokenizer_config.json")}}
            t0 = time.time()
            rep, _files = HD.evaluate({"evaluation": evaluation_spec(), "out": eval_dir, "progress": eval_dir / "progress",
                                       "label": name, "model_dir": model_dir, "package": pkg,
                                       "template_kwargs": ARM_TEMPLATE[a.arm],
                                       "env": {**os.environ, "PYTHONPATH": "src:scripts:scripts/stages/stage-1"}})
            row.update(eval_seconds=round(time.time() - t0, 1), formal_report=str(eval_dir / "formal_report.json"))
            save()
            mark(f"UNIT_DONE:{name}")
    except BaseException as exc:  # noqa: BLE001
        ok = False
        record["error"] = {"error": repr(exc)[:2000], "traceback": traceback.format_exc()[-6000:]}
    record.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  terminal="ALL_DONE" if ok else "RUN_FAILED")
    save()
    mark("ALL_DONE" if ok else "RUN_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
