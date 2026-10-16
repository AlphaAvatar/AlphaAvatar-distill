"""The D-series family's live navigation/status record.

`logs/stages/stage-1/families/d_series/current.json` — DERIVED, not authored.
The family's design record owns the allocation rule, the manifest owns the
realized bytes, `verify_batteries.py` owns identity; this file is the one
machine-readable pointer that says which of those is live, so a reader (or a
launch gate) starts here instead of inferring readiness from a historical
analysis. The stale-battery mistake this guards against: a capacity record
truthfully reporting the ORIGINAL pools exhausted was read as "D1 has no
battery" while the realized family sat verified on disk.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/families/d_series/current.py --write

AUTHORIZES NOTHING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

OUT = "logs/stages/stage-1/families/d_series/current.json"
SCHEMA = "aadistill.family_current/v1"


def build(repo_root: Path = REPO) -> dict[str, Any]:
    from shared.run_layout import resolve_historical
    from stages.d_series import battery_family as family
    from stages.d_series import build_batteries as builder

    record = json.loads((repo_root / family.RECORD).read_text())
    manifest = json.loads((repo_root / builder.MANIFEST).read_text())

    battery_root = repo_root / builder.OUT
    files_present = sum(
        1 for rel in manifest["output_files"] if (battery_root / rel).is_file())

    return {
        "schema": SCHEMA,
        "_contract": (
            "DERIVED navigation for the current state of the D-series "
            "behavioural family. Every value names its owner; edit the "
            "owner, then regenerate this file. AUTHORIZES NOTHING."),
        "family_id": family.FAMILY_ID,
        "kind": "experiment_family",
        "stage_id": 1,
        "serves": ["phase_d1", "phase_d2", "phase_d3"],
        "owners": {
            "scripts": "scripts/stages/stage-1/families/d_series",
            "logs": "logs/stages/stage-1/families/d_series",
            "design_record": family.RECORD,
            "realized_manifest": builder.MANIFEST,
            "battery_bytes": builder.OUT,
            "verifier": "scripts/stages/stage-1/families/d_series/verify_batteries.py",
        },
        "status": record.get("status"),
        "allocation_rule_id": family.allocation_rule_id(),
        "family_content_id": manifest.get("family_content_id"),
        "roles": {
            role: {
                "n_prompts": spec.get("n_prompts"),
                "n_scorable": spec.get("n_scorable"),
                "item_ids_sha256": spec.get("item_ids_sha256"),
            }
            for role, spec in sorted(manifest.get("roles", {}).items())
        },
        "realized_files_present": {
            "present": files_present,
            "expected": len(manifest.get("output_files", {})),
            "_presence_is_not_identity": (
                "run the verifier for byte identity; this is navigation"),
        },
        "superseded_for_readiness": {
            "d1_evidence_capacity": resolve_historical(
                "logs/stages/stage-1/phase_d1/analyses/d1_evidence_capacity.json",
                repo_root),
            "_why": (
                "that record answers how many further batteries the ORIGINAL "
                "C1 pools could yield (zero); the realized family superseded "
                "it for readiness on 2026-10-03 and it is kept as provenance"),
        },
        "shared_evaluation": _shared_evaluation(repo_root),
        "objectives": _objectives(repo_root),
        "authorizes": "nothing",
    }


def _objectives(repo_root: Path) -> dict[str, Any] | None:
    """Where D2's and D3's registered objectives live. Navigation only."""
    from stages.d_series import objectives

    path = repo_root / objectives.CONTRACT
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    out = {
        "contract": objectives.CONTRACT,
        "contract_hash": record["contract_hash"],
        "design": "logs/stages/stage-1/families/d_series/plans/"
                  "d2_d3_objective_design.md",
        "d2_objective_hash": record["d2"]["objective_hash"],
    }
    #: The D3 this contract describes (the student-confidence objective) was
    #: CANCELLED by the maintainer on 2026-10-16 and replaced by a separately
    #: registered experiment. The contract is a frozen record and keeps its
    #: own text; the navigation reads the CURRENT D3 from its registration.
    from stages.phase_d3 import d3_design as D

    reg_path = repo_root / D.REGISTRATION
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text())
        out["d3_in_this_contract"] = {
            "objective_hash": record["d3"]["objective_hash"],
            "status": reg["replaces"]["status"]}
        out["d3"] = {"name": reg["name"], "registration": D.REGISTRATION,
                     "registration_hash": reg["registration_hash"],
                     "status": reg["status"].split(" -- ")[0]}
    else:
        out["d3_status"] = record["d3"]["status"].split(":")[0]
    return out


def _shared_evaluation(repo_root: Path) -> dict[str, Any] | None:
    """Where the shared protocol lives and which identity it carries.

    Navigation only: the protocol record owns its content and the audit owns
    its verification. Absent before the protocol was written.
    """
    from stages.d_series import shared_evaluation as shared

    path = repo_root / shared.RECORD
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    return {
        "protocol_id": f"{record['protocol_id']}@v{record['version']}",
        "protocol_hash": record["protocol_hash"],
        "record": shared.RECORD,
        "audit": shared.AUDIT,
        "design": "logs/stages/stage-1/families/d_series/plans/"
                  "d2_d3_shared_evaluation_design.md",
        "reuses_roles": sorted(r["reuses_battery"]["role"]
                               for r in record["rungs"].values()),
        "reserved_roles": sorted(record["reserved_independent_batteries"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    doc = build()
    body = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    out = REPO / OUT
    if args.write:
        out.write_text(body)
        print(f"wrote {OUT}")
    else:
        same = out.is_file() and out.read_text() == body
        print(f"{OUT}: {'current' if same else 'STALE (run with --write)'}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
