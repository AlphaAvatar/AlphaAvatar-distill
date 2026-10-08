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
        "authorizes": "nothing",
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
