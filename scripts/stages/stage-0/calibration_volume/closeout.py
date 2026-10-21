#!/usr/bin/env python3
"""Close one Stage-0 calibration-volume run: outcome, money, provider state.

    # a run abandoned before any provider resource existed
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/closeout.py \
        --run-id <id> --abandoned "<why>"

    # a session that ran
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/closeout.py \
        --run-id <id> --store <scr>/store --disk-gb 150

`money.all_in_usd` is what `derive_budget.project_sessions` and this
programme's envelope book: the session's GPU cost plus its container disk
($0.10/GB-month over the elapsed minutes), ceiled to the 4-decimal quantum.
A session whose record shows a provider resource but no cost refuses: an
unknown is never booked as zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume import governance as G  # noqa: E402

DISK_USD_PER_GB_MONTH = 0.10
ARTIFACT_ROOT = Path("artifacts/stages/stage-0/calibration_volume")
#: A record larger than this stays in the artifact store; its hash is in `units`.
EVIDENCE_MAX_BYTES = 2_000_000


def ceil4(x: float) -> float:
    return math.ceil(x * 10_000 - 1e-9) / 10_000


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--phase", default="prerequisites")
    ap.add_argument("--abandoned", default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--disk-gb", type=float, default=None)
    ap.add_argument("--outcome", default=None, help="one-line outcome statement")
    args = ap.parse_args(argv)

    run = REPO / G.run_dir(args.run_id)
    out = run / "closeout" / "outcome.json"
    if out.exists():
        raise SystemExit(f"{out.relative_to(REPO)} exists; a closeout is written once")
    doc = {"schema": "aadistill.stage0.calibration_volume.closeout/v1",
           "run_id": args.run_id, "experiment_id": G.EXPERIMENT_ID,
           "stage_id": G.STAGE_ID, "phase": args.phase,
           "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if args.abandoned:
        doc.update({"status": "ABANDONED_BEFORE_LAUNCH", "outcome": args.abandoned,
                    "provider_resource_created": False,
                    "money": {"all_in_usd": 0.0, "_basis": "no provider resource existed"}})
    else:
        sess = json.loads((run / "runtime/session.json").read_text())
        created = bool(sess.get("provider_resource_created"))
        cost = sess.get("cost") or {}
        if created and cost.get("actual_usd") is None:
            raise SystemExit("the session created a provider resource and records no "
                             "cost; reconcile against the provider before closing")
        gpu = float(cost.get("actual_usd") or 0.0)
        minutes = float(cost.get("elapsed_minutes") or 0.0)
        disk = (args.disk_gb or 0.0) * DISK_USD_PER_GB_MONTH * minutes / (60 * 24 * 30.4375)
        fps = sess.get("final_pod_state") or {}
        doc.update({
            "status": "SESSION_ENDED", "terminal": sess.get("terminal"),
            "passed": sess.get("passed"), "outcome": args.outcome,
            "provider_resource_created": created, "pod_id": sess.get("pod_id"),
            "provider_confirms_gone": sess.get("provider_confirms_gone"),
            "final_pod_state": fps,
            "money": {"all_in_usd": ceil4(gpu + disk), "gpu_usd": gpu,
                      "disk_usd": round(disk, 4), "elapsed_minutes": minutes,
                      "price_per_hour": cost.get("price_per_hour"),
                      "authorized_usd": cost.get("authorized_usd"),
                      "_basis": "GPU actual from runtime/session.json :: cost plus "
                                "container disk at $0.10/GB-month over the elapsed "
                                "minutes, ceiled to the 4-decimal quantum"},
            "session_record_sha256": sha(run / "runtime/session.json"),
        })
        if args.store:
            store = Path(args.store)
            dest = REPO / ARTIFACT_ROOT / args.run_id
            ev = run / "evidence"
            ev.mkdir(parents=True, exist_ok=True)
            units = {}
            for unit in sorted(p for p in store.iterdir() if p.is_dir()):
                # a search unit nests its journal under search/, so walk the tree
                files = {}
                for f in sorted(x for x in unit.rglob("*") if x.is_file()):
                    rel = f.relative_to(unit)
                    files[str(rel)] = {"sha256": sha(f), "bytes": f.stat().st_size}
                    (dest / unit.name / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest / unit.name / rel)
                # small, reviewable evidence goes into git; raw records stay outside
                for name in ("score.summary.json", f"{unit.name}.summary.json",
                             "search_record.json", "structure.json",
                             # a formal dose unit: one directory per stage
                             "dose_record.json", "search/search_record.json",
                             "search/stage1_selection.json", "recovery/recovery_products.json",
                             "recovery/recovery_config.json", "eval/report.json",
                             # the Top-4 extension: a dose's finalists, a probe unit
                             "finalists/finalists_manifest.json", "report.json",
                             "recovery_products.json", "recovery_config.json"):
                    if (unit / name).is_file() and (unit / name).stat().st_size <= EVIDENCE_MAX_BYTES:
                        (ev / unit.name / name).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(unit / name, ev / unit.name / name)
                units[unit.name] = files
            for name in ("battery.json", "search_session.json", "bench.json", "probe_session.json"):
                if (store / name).is_file():
                    shutil.copy2(store / name, ev / name)
            doc["units"] = units
            doc["artifacts_root"] = str((ARTIFACT_ROOT / args.run_id))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out.relative_to(REPO)}: {doc['status']} ${doc['money']['all_in_usd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
