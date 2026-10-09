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

    family_cur = json.loads((repo_root / "logs/stages/stage-1/families/"
                             "d_series/current.json").read_text())
    capacity = json.loads((repo_root / "logs/stages/stage-1/phase_d1/"
                           "analyses/d1_evidence_capacity.json").read_text())
    roles = family_cur.get("roles", {})
    family_ready = (family_cur.get("status") == "BUILT / VERIFIED"
                    and "d1_screening" in roles and "d1_confirmation" in roles)
    capacity_is_historical = (capacity.get("record_role") == "historical_analysis"
                              and capacity.get("live_state") is False)
    def _is_active(r: str) -> bool:
        """A run is active only if its own records say it has not ended:
        no closeout, and a runtime session record without a `terminal`
        field. Early aborts keep runtime evidence and a terminal marker;
        counting them as active is the false-liveness this derivation
        replaces."""
        if (runs_root / r / "closeout").is_dir():
            return False
        sess = runs_root / r / "runtime" / "session.json"
        if not sess.is_file():
            return False
        return "terminal" not in json.loads(sess.read_text())

    active = [r for r in runs if _is_active(r)]

    readiness = {
        "_contract": (
            "The battery-readiness answers a reviewer needs, each DERIVED "
            "from its named owner at generation time. The realized family "
            "manifest is the readiness authority; the capacity analysis is "
            "historical and can never reopen this block."),
        "d_series_family": family_cur.get("family_id"),
        "family_status": family_cur.get("status"),
        "d1_screening_battery": "AVAILABLE" if "d1_screening" in roles else "MISSING",
        "d1_confirmation_battery": ("AVAILABLE" if "d1_confirmation" in roles
                                    else "MISSING"),
        "original_c1_pool_capacity": (
            "EXHAUSTED, HISTORICAL ONLY" if capacity_is_historical
            else "see the capacity analysis — its record_role is not historical"),
        "d1_evidence_blocker": ("CLOSED" if family_ready and not open_blockers
                                else "OPEN"),
        "d1_screening": ("ACTIVE: " + ", ".join(active) if active else
                         "PAUSED — no open blocker, no outstanding one-use "
                         "authorization, no active run; the gate is the "
                         "maintainer decision in logs/state/current.json :: next"),
        "_owners": {
            "family_status": "logs/stages/stage-1/families/d_series/current.json",
            "battery_roles": family_cur.get("owners", {}).get("realized_manifest"),
            "pool_capacity": ("logs/stages/stage-1/phase_d1/analyses/"
                              "d1_evidence_capacity.json (record_role: "
                              "historical_analysis; superseded for readiness by "
                              "the family manifest)"),
            "blockers": "d1_session.open_blockers()",
            "pause": "logs/state/current.json :: next",
        },
    }

    gates = {
        "_contract": (
            "Four different kinds of 'can D1 run?', kept distinct because "
            "conflating them has already produced a false blocker once."),
        "formal_funding": ("priced in the frozen design "
                           f"({ds.DESIGN_PATH} :: pricing); the budget terms "
                           "live with the C1 grant the design cites"),
        "one_use_execution_authorization": (
            "none outstanding — a chain is built fresh per attempt and "
            "consumed by it (P12.1)"),
        "open_scientific_blockers": list(open_blockers),
        "maintainer_pause": snapshot.get("next"),
    }

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
        "readiness": readiness,
        "gates": gates,
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
