"""D1's live navigation/status record: `logs/stages/stage-1/phase_d1/current.json`.

DERIVED, not authored. Every value here is read from a named owner — the
frozen design, the retention decision, the realized family, the run tree, the
repository snapshot — and this file exists so that "what is D1's state right
now" has one machine-readable answer that cannot drift from its owners:
regenerating it is part of the convergence chain, and a historical analysis
can no longer masquerade as a live blocker simply by being found first.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d1/current.py --write

AUTHORIZES NOTHING. It is navigation; the owners it points at are the facts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

OUT = "logs/stages/stage-1/phase_d1/current.json"
SCHEMA = "aadistill.experiment_current/v1"


def build(repo_root: Path = REPO) -> dict[str, Any]:
    from stages.phase_d1 import behavioural as bhv
    from stages.phase_d1 import d1_session as ds

    design = ds.design(repo_root)
    open_blockers = ds.open_blockers(repo_root)

    retention_rel = ("logs/stages/stage-1/phase_d1/decisions/"
                     "post_search_finalist_retention.json")
    retention = json.loads((repo_root / retention_rel).read_text())
    members = retention["the_frozen_behavioural_finalists"]["members"]

    arms_present = {
        q: (Path(src) / next(m["state_id"] for m in members
                             if m["quality_position"] == int(q[1:]))).is_dir()
        for q, src in sorted(bhv.ARM_SOURCES.items())
    }

    runs_root = repo_root / "logs/stages/stage-1/phase_d1/runs"
    runs = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
    complete = [r for r in runs
                if (runs_root / r / "closeout" / "outcome.json").is_file()]

    snapshot = json.loads((repo_root / "logs/state/current.json").read_text())

    return {
        "schema": SCHEMA,
        "_contract": (
            "DERIVED navigation for the current state of phase_d1. Every "
            "value names its owner; edit the owner, then regenerate this "
            "file (it runs in the convergence chain). AUTHORIZES NOTHING."),
        "experiment_id": "phase_d1",
        "stage_id": 1,
        "owners": {
            "scripts": "scripts/stages/stage-1/phase_d1",
            "logs": "logs/stages/stage-1/phase_d1",
            "artifacts": "artifacts/stages/stage-1/phase_d1",
            "family": "logs/stages/stage-1/families/d_series",
            "repo_snapshot": "logs/state/current.json",
        },
        "design": {
            "path": ds.DESIGN_PATH,
            "design_hash": ds.design_hash(repo_root),
            "status": design.get("status"),
            "open_blockers": list(open_blockers),
            "_blockers_are_derived": "d1_session.open_blockers()",
        },
        "evidence": {
            "battery_family": ("logs/stages/stage-1/families/d_series/"
                               "analyses/autoinit_d_series_battery_family.json"),
            "roles_required": ["d1_screening", "d1_confirmation"],
            "status_owner": "the family's current.json",
        },
        "candidates": {
            "decision": retention_rel,
            "rule_applied": retention["the_frozen_behavioural_finalists"].get(
                "rule_applied"),
            "finalist_bytes_present": arms_present,
            "_presence_is_not_identity": (
                "a directory existing is navigation; identity is the digest "
                "check `behavioural.require_arms_present()` performs"),
        },
        "runs": {
            "root": "logs/stages/stage-1/phase_d1/runs",
            "all": runs,
            "with_closeout": complete,
        },
        "active_run": None,
        "next_scientific_action": snapshot.get("next"),
        "_next_owner": "logs/state/current.json :: next",
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
