#!/usr/bin/env python3
"""Verify, at `$0`, every input the D2 screening rung consumes but does not make.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/verify_d2_screening_inputs.py [--write]

1. the four INITIALIZATIONS: each candidate's bytes in the search's durable
   store recompute all four recorded identities;
2. the IMPORTED UNITS of the shared screening rung (B, the comparator; q2,
   descriptive): result self-hash, durable result == committed result,
   per-sample rows and generations against their recorded sha256, and a
   RE-SCORE of the stored generations with the pinned scorer that must
   reproduce the per-sample rows byte for byte -- the shared protocol's own
   `_verify_unit`, so the reuse check is the protocol's, not a second one;
3. B's TRAINING identity equals the protocol's recovery block on every field
   but the initialization -- what makes a fresh D2 probe and an imported B unit
   a pair;
4. the session contract derives and hashes (location-free).

Nothing is retrained, regenerated or re-evaluated on a GPU, and no checkpoint
of B or q2 is needed (both are retired).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_REL = "logs/stages/stage-1/phase_d2/analyses/d2_screening_inputs.json"


def verify(repo_root: Path = REPO_ROOT, *, rescore: bool = True,
           evidence_root: Path | None = None) -> dict[str, Any]:
    from stages.d_series import shared_evaluation as SE
    from stages.phase_d2 import d2_screening as SC

    record = SC.protocol(repo_root)
    arms = SC.require_arms_present(repo_root)
    units = [u for u in SE.historical_units(repo_root, evidence_root)
             if u.rung == SC.SHARED_RUNG]
    block = record["recovery"]
    imported = []
    for unit in units:
        store = SE.durable_store(unit.rung, repo_root, evidence_root)
        checked = SE._verify_unit(unit, store, repo_root, rescore=rescore)
        training_differs = sorted(
            k for k in SE.TRAINING_KEYS
            if k not in ("initialization_digest", "seed")
            and unit.training.get(k) != block.get(k))
        imported.append({
            "probe_id": unit.probe_id, "arm": unit.arm, "seed": unit.seed,
            "role": ("comparator" if unit.arm == "B"
                     else "descriptive only (winner's curse)"),
            "durable_store": str(store),
            "result_verified": checked["result_verified"],
            "checks": checked["checks"],
            "training_matches_the_recovery_block": not training_differs,
            "training_fields_differing": training_differs,
            "evaluation_protocol_hash": unit.evaluation[
                "evaluation_protocol_hash"],
            "evaluation_matches_the_rung": unit.evaluation[
                "evaluation_protocol_hash"]
            == record["rungs"][SC.SHARED_RUNG]["evaluation_protocol_hash"],
            "correct_overall": unit.result["correct_overall"],
            "usable_rollout_rate": unit.result["usable_rollout_rate"],
            "checkpoint_retired": not checked["checks"].get(
                "checkpoint_bytes_present", False),
        })
    b = [u for u in imported if u["arm"] == "B"]
    contract = SC.session_contract(repo_root)
    ok = (all(u["result_verified"] and u["training_matches_the_recovery_block"]
              and u["evaluation_matches_the_rung"] for u in imported)
          and len(b) == len(SC.seeds(repo_root)))
    return {
        "schema": "aadistill.phase_d2.screening_inputs/v1",
        "recorded_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "all_verified": ok,
        "initializations": arms,
        "imported_units": imported,
        "rescored_with_the_pinned_scorer": rescore,
        "contract_hash": SC.contract_hash(contract),
        "shared_protocol_hash": record["protocol_hash"],
        "tree_matches_the_protocol": SE.tree_matches(record, repo_root),
        "_not_done": ("nothing about B or q2 is retrained, regenerated or "
                      "re-evaluated, and no logits or confidence are computed "
                      "for them; their checkpoints are retired"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-rescore", action="store_true")
    args = ap.parse_args(argv)
    doc = verify(rescore=not args.no_rescore)
    if args.write:
        out = REPO_ROOT / OUT_REL
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT_REL}")
    print(json.dumps({k: doc[k] for k in ("all_verified", "contract_hash")}
                     | {"imported": [(u["probe_id"], u["result_verified"],
                                      u["checks"].get(
                                          "rescore_reproduces_per_sample"))
                                     for u in doc["imported_units"]]},
                     indent=1))
    return 0 if doc["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
