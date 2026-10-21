#!/usr/bin/env python3
"""Carry the study's network-volume tombstones into the project's tombstone inventory.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-0/calibration_volume/register_volume_tombstones.py

`retire_volume.py` wrote each volume's tombstone into its own run directory.
The project's deletion rule (logs/maintenance/inventories/checkpoint_registry.json
:: deletion_rule) names ONE inventory of retired weights,
`logs/maintenance/inventories/checkpoint_tombstones.json`, and the checkpoint
registry links a logical artifact to its tombstone through it. This adds one
entry per (volume, weight class) -- the per-file detail stays with the volume's
own tombstone, which every entry names -- and re-derives the inventory's totals.
Idempotent: an entry with the same canonical_id is replaced, never duplicated.
Small files are not weights and were fetched before the delete; they get no entry.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = "logs/stages/stage-0/calibration_volume/runs"
INVENTORY = REPO / "logs/maintenance/inventories/checkpoint_tombstones.json"
NETVOL_RUNS = ("netvol_20261019_051206", "netvol_20261020_093832",
               "netvol_20261020_233114")
WEIGHT_CLASSES = {
    "recovered student weights": ("recovered_students",
                                  "recovered (Stage-3-recipe) students, measured on all of eval-set-1"),
    "search state / finalist weights": ("search_states",
                                        "materialized D1 full-search states (d4's journal states and finalists)"),
    "recovery optimizer state": ("optimizer_state",
                                 "a completed recovery's optimizer state (trainer_state.pt)"),
}


def entries_for(run: str) -> list[dict]:
    rel = f"{RUNS}/{run}/closeout/tombstone.json"
    t = json.loads((REPO / rel).read_text())
    listed = t["members_source"]["kind"] == "listing"
    out = []
    for cls, (slug, role) in WEIGHT_CLASSES.items():
        members = [m for m in t["members"] if m["class"] == cls]
        if not members:
            continue
        ids = {m["path"]: m.get("sha256") or m.get("model_sha256") for m in members}
        size = sum(m["bytes"] for m in members)
        out.append({
            "canonical_id": f"stage0_calvol_{t['volume_id']}_{slug}",
            "historical_paths": [f"runpod-network-volume:{t['volume_id']}/{m['path']}"
                                 for m in members],
            "relay_path": None,
            "relay_hash_verified": None,
            "store": (f"RunPod network volume {t['volume_id']} ({t['data_center_id']}, "
                      f"{t['size_gb']} GB); the volume itself is deleted"),
            "retention_tier": "TIER_3_RECORD_ONLY",
            "experiment": "Stage-0 calibration_volume",
            "seed": None,
            "scientific_role": role,
            "weights_sha256": None,
            "member_sha256": {p: s for p, s in ids.items() if s},
            "member_identity_basis": (
                "hashed on the volume by the pre-deletion listing "
                f"({t['members_source'].get('audit_run')})" if listed else
                "DERIVED / NOT DIRECTLY VERIFIED: no pod could attach to the volume "
                "before deletion; members and model sha256 come from the probe "
                "sessions' own recovery_products.json, not from a listing"),
            "config_sha256": None,
            "size_bytes": size,
            "size_human": f"{size / 2**30:.3f} GiB",
            "n_files": sum(m.get("files", 1) for m in members),
            "members": [m["path"] for m in members],
            "deleted_utc": t["deleted_utc"],
            "deletion_criterion": "agents_2_5_standing_retirement",
            "deletion_criterion_meaning": t["deletion_criterion"],
            "references_at_deletion": t["references_at_deletion"],
            "deterministic_reconstruction_possible": slug == "search_states",
            "reconstruction_recipe": t["reconstruction"],
            "expected_reconstruction_cost": "stated per class in reconstruction_recipe",
            "canonical_surviving_evidence": [rel, *t["canonical_surviving_evidence"]],
            "frozen_results_location": "logs/stages/stage-0/calibration_volume/design/"
                                       "formal_analysis.json; design/extension_analysis.json",
            "withdrawn": False,
            "withdrawn_reason": None,
            "withdrawn_utc": None,
            "reason_physical_weights_deleted": t["why"],
        })
    return out


def main() -> int:
    doc = json.loads(INVENTORY.read_text())
    new = [e for run in NETVOL_RUNS for e in entries_for(run)]
    ids = {e["canonical_id"] for e in new}
    doc["tombstones"] = [t for t in doc["tombstones"] if t["canonical_id"] not in ids] + new
    ts = doc["tombstones"]
    active = [x for x in ts if not x.get("withdrawn")]
    withdrawn = [x for x in ts if x.get("withdrawn")]
    tot = doc["totals"]
    tot["tombstones_total"] = len(ts)
    tot["active"] = len(active)
    tot["active_retired_bytes"] = sum(x.get("size_bytes") or 0 for x in active)
    tot["active_retired_human"] = f"{tot['active_retired_bytes'] / 2**30:.3f} GiB"
    tot["withdrawn"] = len(withdrawn)
    tot["withdrawn_ids"] = sorted(x["canonical_id"] for x in withdrawn)
    tot["withdrawn_bytes"] = sum(x.get("size_bytes") or 0 for x in withdrawn)
    sized = sum(1 for x in active if x.get("size_bytes"))
    tot["_size_coverage"] = (
        f"{sized} of {len(active)} active entries carry a size; the older ones "
        "predate the field and are not back-filled, so the byte total is a LOWER "
        "BOUND on what has been retired.")
    INVENTORY.write_text(json.dumps(doc, indent=1) + "\n")
    for e in new:
        print(f"{e['canonical_id']}: {e['n_files']} files, {e['size_human']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
