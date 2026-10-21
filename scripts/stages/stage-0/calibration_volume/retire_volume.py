#!/usr/bin/env python3
"""Retire one of the study's network volumes: tombstone, delete, verify, true up.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-0/calibration_volume/retire_volume.py \
        --netvol-run netvol_<utc> (--audit-run volaudit_<utc> | --probe-runs probes_<..> ...) \
        [--delete]

AGENTS.md 2.5 (standing retirement of historical checkpoint weights). Every unit
whose weights the volume holds is terminal and analysed, its evaluation evidence
is durable in git and in the out-of-tree store, and its identities are recorded.
This writes the tombstone into the volume's run directory BEFORE the delete, so
a delete that fails leaves the intent; after the provider no longer lists the
volume, the tombstone is rewritten as DELETED and the volume's closeout is trued
up from its reservation to the actual storage cost.

Members come from one of two sources, and the tombstone says which:
* --audit-run: a volume_audit.py listing taken on a pod attached to the volume
  (every file, its size, the sha256 of the small files and of the students);
* --probe-runs: derived from probe_driver.py's push sets and each probe
  session's record, for a probe volume when no pod could be had in its
  datacenter. The driver writes nothing else there, and probe_session.json
  shows whether any unit restored or superseded anything.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = REPO / "logs/stages/stage-0/calibration_volume/runs"
#: where closeout.py keeps each run's store (gitignored); an audit's fetched files go there too
ARTIFACTS = "artifacts/stages/stage-0/calibration_volume"
SCHEMA = "aadistill.stage0.calibration_volume.volume_tombstone/v1"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def student_records() -> dict[str, str]:
    """model_sha256 -> the committed record that names it."""
    out = {}
    for p in sorted(RUNS.glob("*/evidence/**/recovery_products.json")):
        sha = load(p).get("model_sha256")
        if sha:
            out.setdefault(sha, rel(p))
    for p in sorted(RUNS.glob("dose_*/evidence/*/dose_record.json")):
        sha = (load(p).get("stages", {}).get("R_recovery") or {}).get("model_sha256")
        if sha:
            out.setdefault(sha, rel(p))
    return out


def classify(path: str) -> str:
    if path.endswith("model.safetensors") and re.search(r"(recovery|package)/", path):
        return "recovered student weights"
    if path.endswith("trainer_state.pt"):
        return "recovery optimizer state"
    if re.search(r"(search_unit|finalists)/.*\.(safetensors|pt|bin)$", path):
        return "search state / finalist weights"
    if path.endswith((".safetensors", ".pt", ".bin")):
        return "other weights"
    return "record or small file"


def members_from_audit(audit_run: str) -> tuple[dict, list[dict]]:
    ev_path = RUNS / audit_run / "evidence/volume_audit.json"
    ev = load(ev_path)
    if ev.get("status") != "AUDITED":
        raise SystemExit(f"{audit_run} did not audit the volume ({ev.get('status')})")
    known = student_records()
    members = []
    for f in ev["files"]:
        m = {"path": f["path"], "bytes": f["bytes"], "sha256": f.get("sha256"),
             "class": classify(f["path"])}
        if f.get("fetched"):
            m["local_copy"] = f"{ARTIFACTS}/{audit_run}/files/{f['path']}"
        if m["class"] == "recovered student weights":
            m["identity_record"] = known.get(f.get("sha256") or "")
        members.append(m)
    students = [m for m in members if m["class"] == "recovered student weights"]
    unmatched = [m["path"] for m in students if not m.get("identity_record")]
    if unmatched:
        raise SystemExit(f"students on the volume that no committed record names: {unmatched}")
    if not (REPO / ARTIFACTS / audit_run / "files").is_dir():
        raise SystemExit(f"{ARTIFACTS}/{audit_run}/files is missing: keep the fetched files first")
    src = {"kind": "listing", "audit_run": audit_run, "record": rel(ev_path),
           "pod_id": ev.get("pod_id"), "listed_utc": ev.get("ended_utc"),
           "fetched_into": [f"{ARTIFACTS}/{audit_run}/files", ev["store"]]}
    return src, members


def members_from_probes(probe_runs: list[str]) -> tuple[dict, list[dict]]:
    members = []
    for run in probe_runs:
        sess = load(RUNS / run / "evidence/probe_session.json")
        if sess.get("status") != "COMPLETE":
            raise SystemExit(f"{run} is {sess.get('status')}, not COMPLETE")
        vdir = sess["volume_dir"].split("/stage0_calvol/", 1)[1]
        store = load(RUNS / run / "closeout/outcome.json").get("artifacts_root")
        for label, u in sess["units"].items():
            odd = {"recovery_restore_failed", "generations_superseded", "restored_complete",
                   "recovery"} & set(u)
            if odd or u.get("status") != "ok":
                raise SystemExit(f"{run}/{label}: {sorted(odd) or u.get('status')} -- list it instead")
            prod = load(RUNS / run / "evidence" / label / "recovery_products.json")
            student, records = u["recovery_pushed"]
            base = f"stage0_calvol/{vdir}/{label}"
            members += [
                {"path": f"{base}/recovery/", "files": student["copied"], "bytes": student["bytes"],
                 "class": "recovered student weights",
                 "contents": "checkpoints/<tag>/model/ (model.safetensors + its config files), "
                             "checkpoints/latest.txt, train_log.jsonl",
                 "model_sha256": prod["model_sha256"], "model_bytes": prod["model_bytes"],
                 "identity_record": rel(RUNS / run / "evidence" / label / "recovery_products.json")},
                {"path": f"{base}/recovery_records/", "files": records["copied"],
                 "bytes": records["bytes"], "class": "record or small file",
                 "local_copy": f"{store}/{label}/"},
                {"path": f"{base}/products.json", "class": "record or small file",
                 "contents": '{"recovery": <recovery_products.json>}',
                 "local_copy": f"{store}/{label}/recovery_products.json (the same object)"},
                {"path": f"{base}/generations/", "class": "record or small file",
                 "local_copy": f"{store}/{label}/generations/"},
                {"path": f"{base}/out/", "class": "record or small file",
                 "contents": "rows/, report.json, recovery_products.json, recovery_config.json, "
                             "recovery.log, train_log.jsonl, eval logs",
                 "local_copy": f"{store}/{label}/"},
            ]
    src = {"kind": "derived", "probe_runs": probe_runs,
           "rule": "scripts/stages/stage-0/calibration_volume/probe_driver.py writes, per unit, "
                   "exactly recovery/ (push_student), recovery_records/, products.json, "
                   "generations/ (evaluate_all) and out/ (UNIT_RECORDS, then report.json); each "
                   "session record shows every unit trained here, none restored or superseded"}
    return src, members


def references(volume_id: str) -> list[str]:
    r = subprocess.run(["git", "grep", "-l", volume_id], cwd=REPO, capture_output=True, text=True)
    return sorted(r.stdout.split())


def provider_lists(volume_id: str) -> bool:
    r = subprocess.run(["runpodctl", "network-volume", "list"], capture_output=True, text=True,
                       timeout=120, check=True)
    return any(v.get("id") == volume_id for v in json.loads(r.stdout))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--netvol-run", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-run")
    g.add_argument("--probe-runs", nargs="+")
    ap.add_argument("--why", required=True, help="why nothing reads these bytes any more")
    ap.add_argument("--reconstruction", required=True)
    ap.add_argument("--surviving-evidence", nargs="+", required=True)
    ap.add_argument("--listing-attempts", nargs="*", default=[],
                    help="volume_audit runs that could not list this volume (no capacity)")
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args(argv)

    run_dir = RUNS / a.netvol_run
    sess = load(run_dir / "runtime/session.json")
    vid = sess["volume_id"]
    src, members = (members_from_audit(a.audit_run) if a.audit_run
                    else members_from_probes(a.probe_runs))
    if a.listing_attempts:
        src["listing_attempts"] = {
            r: load(RUNS / r / "closeout/outcome.json")["status"] for r in a.listing_attempts}
    by_class: dict[str, dict] = {}
    for m in members:
        c = by_class.setdefault(m["class"], {"entries": 0, "bytes": 0})
        c["entries"] += 1
        c["bytes"] += m.get("bytes") or 0
    refs = references(vid)
    tomb = {"schema": SCHEMA, "volume_id": vid, "name": sess["name"],
            "data_center_id": sess["data_center_id"], "size_gb": sess["size_gb"],
            "created_utc": sess["created_utc"], "status": "INTENT", "intent_utc": now(),
            "deleted_utc": None,
            "deletion_criterion": "AGENTS.md 2.5 standing retirement of historical checkpoint "
                                  "weights: every unit is terminal and analysed, its evaluation "
                                  "evidence and identities are durable, no active or declared "
                                  "consumer reads the bytes, and none is an incumbent, release "
                                  "or retained anchor",
            "why": a.why,
            "references_at_deletion": refs,
            "members_source": src, "totals_by_class": by_class,
            "members": members,
            "reconstruction": a.reconstruction,
            "canonical_surviving_evidence": a.surviving_evidence,
            "never_deleted": "logs and evidence: the records and generations the volume held "
                             "have the local copies named per member"}
    out = run_dir / "closeout/tombstone.json"
    out.write_text(json.dumps(tomb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {rel(out)} (INTENT): {len(members)} members, "
          f"{sum(c['bytes'] for c in by_class.values()) / 1e9:.2f} GB; {len(refs)} references")
    if not a.delete:
        return 0

    if not provider_lists(vid):
        raise SystemExit(f"the provider does not list {vid}; nothing to delete")
    r = subprocess.run(["runpodctl", "network-volume", "delete", vid], capture_output=True,
                       text=True, timeout=300)
    print(f"delete rc={r.returncode}: {(r.stdout + r.stderr).strip()[:300]}")
    gone = not provider_lists(vid)
    if not gone:
        raise SystemExit(f"{vid} is still listed after the delete; the tombstone stays INTENT")
    deleted = now()
    tomb.update(status="DELETED", deleted_utc=deleted,
                delete_command=f"runpodctl network-volume delete {vid}",
                provider_confirms_gone=True)
    out.write_text(json.dumps(tomb, indent=1, ensure_ascii=False) + "\n")

    oc_path = run_dir / "closeout/outcome.json"
    oc = load(oc_path)
    pb = sess["price_basis"]
    t0 = datetime.fromisoformat(sess["created_utc"].replace("Z", "+00:00"))
    hours = (datetime.fromisoformat(deleted) - t0).total_seconds() / 3600
    usd = math.ceil(sess["size_gb"] * pb["usd_per_gb_month"] * hours / pb["hours_per_month"]
                    * 1e4) / 1e4
    oc.update(recorded_utc=deleted, status="VOLUME_DELETED",
              outcome=(f"Network volume {vid} ({sess['size_gb']} GB, {sess['data_center_id']}) "
                       f"was DELETED at {deleted} under the AGENTS.md 2.5 standing retirement; "
                       f"closeout/tombstone.json names what it held and where the evidence lives. "
                       f"Its reservation is trued up to the actual storage cost."),
              money={"all_in_usd": usd, "hours": round(hours, 3),
                     "reserved_usd_replaced": oc["money"]["all_in_usd"],
                     "_basis": f"ACTUAL: {sess['size_gb']} GB x ${pb['usd_per_gb_month']}/GB-month "
                               f"x {hours:.3f} h / {pb['hours_per_month']} h = ${usd:.4f} "
                               f"(created {sess['created_utc']} -> deleted {deleted}), ceiled to "
                               "the 4-decimal quantum"},
              tombstone="closeout/tombstone.json")
    oc_path.write_text(json.dumps(oc, indent=1, ensure_ascii=False) + "\n")
    print(f"{vid} DELETED, provider confirms gone; {hours:.2f} h -> ${usd:.4f} "
          f"(reservation ${oc['money']['reserved_usd_replaced']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
