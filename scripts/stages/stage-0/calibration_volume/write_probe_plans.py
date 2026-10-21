#!/usr/bin/env python3
"""Write the Top-4 extension's probe session plans from the finalists manifest. $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/write_probe_plans.py \
        --manifest <finalists_manifest.json> --treatments d1 d2 d4 \
        --volume-id <id> --data-center US-TX-3 [--unit-minutes recovery=65 eval=90]

One plan per dose (`configs/stages/stage-0/calibration_volume/probes_<dose>_session.json`):
its units from `finalists.probe_units` under the extension's registered seeds,
each candidate staged from the relay by sha256, and the session priced from
per-unit minutes. The minutes are an estimate until a probe has run on the
L40S (`plans/top4_extension.json :: budget.rule` re-prices from the first
measured unit); the plan says which.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

EXT = "logs/stages/stage-0/calibration_volume/plans/top4_extension.json"
PLANS = REPO / "configs/stages/stage-0/calibration_volume"


def plan_for(manifest: dict, treatment: str, ext: dict, *, volume: dict,
             unit_minutes: dict, basis: str) -> dict:
    from stages.calibration_volume.finalists import probe_units
    units = probe_units(manifest, treatment, ext["units"]["seeds"])
    n = len(units)
    prereg = json.loads((REPO / ext["relation_to_the_preregistration"]["preregistration"]).read_text())
    teacher = json.loads((PLANS / f"dose_{treatment}_session.json").read_text())["teacher"]
    return {
        "schema": "aadistill.stage0.calibration_volume.session_plan/v1",
        "session_kind": "probes", "name": f"probes_{treatment}", "treatment": treatment,
        "purpose": (f"Stage-0 Top-4 extension, dose {treatment}: {n} units, each one candidate "
                    "recovered by the frozen recipe at one seed and measured on every item of "
                    "eval-set-1 (plans/top4_extension.json)."),
        "extension": EXT, "extension_sha256": hashlib.sha256((REPO / EXT).read_bytes()).hexdigest(),
        "study_id": prereg["study_id"],
        "teacher_revision": teacher["revision"], "teacher": teacher,
        "units": units,
        "setup_minutes": 20,
        "budget_phases": {f"{k}_x{n}": round(float(v) * n, 1) for k, v in unit_minutes.items()},
        "unit_minutes": unit_minutes, "unit_minutes_basis": basis,
        "contingency_fraction": 0.2, "reserve_minutes": 30,
        "hardware": {"gpu": "NVIDIA L40S", "container_disk_gb": 200, "network_volume": volume},
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--treatments", nargs="+", required=True)
    ap.add_argument("--volume-id", required=True)
    ap.add_argument("--data-center", required=True)
    ap.add_argument("--volume-gb", type=int, default=100)
    ap.add_argument("--volume-record", default=None)
    ap.add_argument("--unit-minutes", nargs="*", default=["recovery=65", "eval=90", "pushes=5"])
    ap.add_argument("--basis", default=("estimate: recovery measured on an L40S for this recipe "
                                        "(about 60 min); eval-set-1 on an L40S not yet measured, "
                                        "taken at about 1.8x the A100's 49 min"))
    args = ap.parse_args(argv)
    ext = json.loads((REPO / EXT).read_text())
    manifest = json.loads(args.manifest.read_text())
    minutes = {k: float(v) for k, v in (x.split("=", 1) for x in args.unit_minutes)}
    volume = {"id": args.volume_id, "data_center": args.data_center, "size_gb": args.volume_gb,
              "mount": "/runpod-volume", "record": args.volume_record}
    for t in args.treatments:
        doc = plan_for(manifest, t, ext, volume=volume, unit_minutes=minutes, basis=args.basis)
        path = PLANS / f"probes_{t}_session.json"
        path.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{path.relative_to(REPO)}: {len(doc['units'])} units, "
              f"{sum(doc['budget_phases'].values()):.0f} work minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
