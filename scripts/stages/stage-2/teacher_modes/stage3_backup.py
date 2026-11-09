#!/usr/bin/env python3
"""The Stage-3 students' final weights, copied off the search volume onto the dev box's persistent disk. $0 here.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.stage3_backup --plan
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.stage3_backup --verify

Maintainer 2026-11-09 (UTC), section 5: the six Stage-3 -sr students' final fp32 weights and configs, which exist only
on ``thprcgh7s9``, get an independent copy on the dev box (a cloud disk outside RunPod; a temporary cache is not a
backup), every file checked by sha256. The volume's region (US-TX-3) has no S3 API, so its bytes leave only through a
pod: the copy rides on the next session that attaches the volume (kind ``train_sr_mix``), pulled by its polls while it
trains (`pull_unit`, one unit per poll) and finished by its closeout -- never a reason to hold a billing pod, since
the volume's copy stands either way. That session's own two units are copied the same way once trained.

Per unit: ``model.safetensors``, ``config.json`` and ``generation_config.json`` of the final save point's ``model/`` and
the save point's ``resume_identity.json``; ``trainer_state.pt`` is not copied (no exact-continuation consumer:
maintainer 2026-11-09). A file is received only when its sha256 on the dev box equals the pod's for the same file,
and the weights' equals the sha256 the unit's own record committed (`RECORD`). Each unit gets a ``receipt.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

RECORD = REPO / "logs/stages/stage-2/teacher_modes/storage/stage3_devbox_backup.json"
SCHEMA = "aadistill.stage2.teacher_modes.stage3_devbox_backup/v1"
#: the dev box's out-of-tree durable store (docs/REPO_LAYOUT.md), and the units' records on the search volume
DEST_ROOT = Path("/home/ecs-user/aad-artifacts/stage3_sr/final_weights")
RUNS = REPO / "logs/stages/stage-2/teacher_modes/runs"
SOURCE_RUNS = ("a_train_sr_a1_20261107_104149", "a_train_sr_a2_20261107_174531", "a_train_sr_c1_20261108_104542")
MODEL_FILES = ("model.safetensors", "config.json", "generation_config.json")
#: measured pod -> dev box (memory: neither-transport-carries-a-checkpoint), for the estimate only
DOWNLINK_MB_S = 11.5
VOLUME = "thprcgh7s9"


def units_from_records(runs=SOURCE_RUNS) -> dict:
    """unit -> its final weights on the volume, from the LAST committed record that names them."""
    out = {}
    for run in runs:
        for f in sorted((RUNS / run / "evidence").glob("*_units.json")):
            for name, row in json.loads(f.read_text())["units"].items():
                if row.get("final_model_path") and row.get("final_model_safetensors_sha256"):
                    out[name] = {"model_dir": row["final_model_path"],
                                 "model_safetensors_sha256": row["final_model_safetensors_sha256"],
                                 "record": str(f.relative_to(REPO))}
    return out


def plan(dest_root: Path = DEST_ROOT) -> dict:
    from stages.teacher_modes import packed_training_plan as PTP
    us = units_from_records()
    total = len(us) * PTP.MODEL_BYTES
    return {"schema": SCHEMA, "decided_by": "maintainer 2026-11-09 (UTC), 'Research A 新增 Hybrid Mixed Stage 3 与 "
                                            "Checkpoint 精简', section 5",
            "volume": VOLUME, "destination_root": str(dest_root),
            "files_per_unit": [f"model/{n}" for n in MODEL_FILES] + ["resume_identity.json"],
            "not_copied": {"trainer_state.pt": "no exact-continuation consumer (closed training)"},
            "units": us, "weights_bytes": total,
            "transfer_estimate_minutes": round(total / (DOWNLINK_MB_S * 1e6) / 60, 1),
            "carried_by": "kind train_sr_mix: its polls while it trains (token_matched_phase.backup_on_poll), its "
                          "closeout for what is left; evidence only, never a reason to hold the pod",
            "independence": "the dev box's cloud disk is outside RunPod; the copy is a second store, the volume's "
                            "copy stays until the maintainer retires it",
            "authorizes": "nothing"}


def remote_files(model_dir: str) -> dict[str, str]:
    """published relative path -> remote path of one unit's files."""
    parent = str(Path(model_dir).parent)
    out = {f"model/{n}": f"{model_dir}/{n}" for n in MODEL_FILES}
    out["resume_identity.json"] = f"{parent}/resume_identity.json"
    return out


def receipt_path(dest_root: Path, unit: str) -> Path:
    return Path(dest_root) / unit / "receipt.json"


def received(dest_root: Path, unit: str, sha256: str) -> bool:
    p = receipt_path(dest_root, unit)
    if not p.is_file():
        return False
    r = json.loads(p.read_text())
    return bool(r.get("verified")) and r["files"].get("model/model.safetensors", {}).get("sha256") == sha256


def pull_unit(ctx, unit: str, row: dict, dest_root: Path = DEST_ROOT) -> dict:
    """One unit's files from the pod to the dev box (`phase._pull`: scp, then sha256 against the pod's own
    sha256sum of the same file), its receipt written. Never raises; a failure is the receipt's ``error``."""
    from stages.teacher_modes import phase as P1
    out = {"unit": unit, "verified": False}
    try:
        rem = remote_files(row["model_dir"])
        lines = P1._listing(ctx, "sha256sum " + " ".join(rem.values())) or []
        on_pod = {}
        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                on_pod[parts[1]] = parts[0]
        dest = Path(dest_root) / unit / Path(row["model_dir"]).parent.name
        files, t0 = {}, time.time()
        for rel, remote in rem.items():
            want = on_pod.get(remote)
            if want is None:
                raise RuntimeError(f"{remote}: the pod lists no sha256 for it")
            ok = P1._pull(ctx, remote, dest / rel, want)
            files[rel] = {"sha256": want, "bytes": (dest / rel).stat().st_size if (dest / rel).is_file() else None,
                          "received": ok}
        weights = files["model/model.safetensors"]["sha256"]
        out.update(files=files, destination=str(dest), seconds=round(time.time() - t0, 1),
                   source={"volume": VOLUME, "model_dir": row["model_dir"],
                           "pod": getattr(getattr(ctx, "target", None), "pod_id", None)},
                   weights_match_the_record=weights == row["model_safetensors_sha256"],
                   verified=all(f["received"] for f in files.values()) and weights == row["model_safetensors_sha256"])
    except Exception as exc:  # noqa: BLE001 -- the volume's copy stands; a later poll or session tries again
        out["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        p = receipt_path(dest_root, unit)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1) + "\n")
    except OSError:
        pass
    return out


def owed(record: dict, session_units: dict | None = None, dest_root: Path | None = None) -> dict:
    """unit -> row still owed a verified dev-box copy: the record's units, then the session's own trained units (its
    units record's ``final_model_path`` and sha256)."""
    dest_root = Path(dest_root or record["destination_root"])
    rows = dict(record["units"])
    for name, r in ((session_units or {}).get("units") or {}).items():
        if r.get("final_model_path") and r.get("final_model_safetensors_sha256"):
            rows[name] = {"model_dir": r["final_model_path"], "model_safetensors_sha256": r["final_model_safetensors_sha256"]}
    return {u: r for u, r in rows.items() if not received(dest_root, u, r["model_safetensors_sha256"])}


def verify_local(record: dict | None = None) -> dict:
    """Every received unit re-hashed on the dev box against its receipt (and the weights against the record)."""
    record = record or json.loads(RECORD.read_text())
    dest_root, out = Path(record["destination_root"]), {}
    for unit, row in record["units"].items():
        p = receipt_path(dest_root, unit)
        if not p.is_file():
            out[unit] = "not received"
            continue
        r = json.loads(p.read_text())
        bad = []
        for rel, f in r.get("files", {}).items():
            q = Path(r["destination"]) / rel
            h = hashlib.sha256()
            with open(q, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != f["sha256"]:
                bad.append(rel)
        if r["files"]["model/model.safetensors"]["sha256"] != row["model_safetensors_sha256"]:
            bad.append("weights vs the record")
        out[unit] = "verified" if not bad else f"MISMATCH {bad}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="write the backup record from the committed unit records")
    ap.add_argument("--verify", action="store_true", help="re-hash every received copy on the dev box")
    a = ap.parse_args(argv)
    if a.plan:
        doc = plan()
        RECORD.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({k: doc[k] for k in ("weights_bytes", "transfer_estimate_minutes")} | {"units": len(doc["units"])}))
    if a.verify:
        print(json.dumps(verify_local(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
