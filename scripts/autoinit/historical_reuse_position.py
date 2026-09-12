#!/usr/bin/env python3
"""The four conclusions about historical probe reuse, stated together.

    PYTHONPATH=src:scripts python scripts/autoinit/historical_reuse_position.py \
        --out logs/experiments/shared/analyses/autoinit_historical_reuse_position.json

Each of these is already established somewhere, and separately they read as
though one of them must be wrong:

1. the historical probe BYTES are valid -- every retained checkpoint still
   re-derives the artifact digest its probe recorded;
2. the probes RECONSTRUCT under their historical contract -- seed, battery and
   attested protocol hash all match what Phase A recorded;
3. the numbers are not in question -- 570 frozen Phase-A samples re-scored
   through the pre- and post-migration trees are byte-identical;
4. and live reuse under scoring contract v3 is nevertheless **REFUSED**.

They are simultaneously true, and this file exists so that nobody has to
reconcile them from four documents and a docstring. The identity rule refuses a
probe whose recorded scorer digest is not the live one. It is deliberately
conservative and it is about IDENTITY, not about arithmetic: demonstrating that
the arithmetic is unchanged is not the same as demonstrating that a superseded
contract may be admitted, and turning the first into the second would change
what counts as a reusable scientific observation.

**This script relaxes nothing.** It calls the same `verify()` the pre-provider
gate calls, asserts that its verdict is still REFUSED, and fails if it is not.
Conclusion 3 is read from the equivalence record's own fields rather than
restated in prose, so a future edit that weakened that evidence would change
this record instead of leaving a confident sentence behind.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

EQUIVALENCE = REPO / "logs/maintenance/inventories/architecture_scoring_equivalence.json"

#: The one check that may fail. Anything else means the probes themselves
#: stopped reconstructing, which is a different and much worse finding.
LIVE_CONTRACT_CHECK = "scoring_contract_matches_live"


def _rel(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise.

    `Path.relative_to` RAISES on a path outside the base, which turns a
    successful run into a traceback after the work is already done -- and
    makes the function untestable against a fixture in a tmp dir.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def position(verify_fn=None, equivalence_path: Path = EQUIVALENCE) -> dict:
    """Derive all four conclusions. Nothing here is asserted from prose."""
    if verify_fn is None:
        from verify_historical_probe_reuse import verify as verify_fn

    r = verify_fn()
    probes = r["probes"]

    def every(check: str) -> bool:
        return bool(probes) and all(p["checks"][check] for p in probes)

    historical_checks = sorted(
        {c for p in probes for c in p["checks"]} - {LIVE_CONTRACT_CHECK})

    bytes_valid = every("artifact_digest_re_derives_from_bytes") and all(
        p["recomputed_artifact_digest"] == p["recorded_artifact_digest"]
        and p["recomputed_artifact_digest"] for p in probes)
    reconstructs = all(every(c) for c in historical_checks)
    only_the_contract_fails = all(
        p["failed"] == [LIVE_CONTRACT_CHECK] for p in r["failures"])

    eq = json.loads(equivalence_path.read_text())
    numbers_unchanged = (eq["all_scores_identical"] is True
                         and eq["total_samples"] > 0)

    live_refused = r["reuse_verified"] is False and bool(r["failures"])

    conclusions = {
        "1_historical_probe_bytes_are_valid": {
            "verdict": "PASS" if bytes_valid else "FAIL",
            "how": ("every probe's student_artifact_digest re-derives from the "
                    "retained checkpoint bytes and equals the recorded value"),
            "n_probes": r["n_probes"],
        },
        "2_probes_reconstruct_under_their_historical_contract": {
            "verdict": "PASS" if reconstructs else "FAIL",
            "how": ("every check except the live-contract identity passes for "
                    "every probe"),
            "checks": historical_checks,
        },
        "3_behavior_v0_relocation_outputs_are_byte_identical": {
            "verdict": "PASS" if numbers_unchanged else "FAIL",
            "how": ("the 570-sample comparison re-scored frozen Phase-A "
                    "generations through the pre- and post-migration trees"),
            "record": _rel(equivalence_path),
            "total_samples": eq["total_samples"],
            "paths_compared": eq["paths_compared"],
            "all_scores_identical": eq["all_scores_identical"],
            "coverage_limitation": eq["coverage"]["limitation"],
        },
        "4_live_reuse_under_scoring_contract_v3": {
            "verdict": "REFUSED" if live_refused else "NOT REFUSED",
            "how": (f"{len(r['failures'])} of {r['n_probes']} probes fail "
                    f"{LIVE_CONTRACT_CHECK}, and nothing else"),
            "only_the_live_contract_check_fails": only_the_contract_fails,
            "why_this_is_not_a_defect": (
                "the identity rule refuses a probe whose recorded scorer digest "
                "is not the live one. The scorer relocated and the contract "
                "legitimately moved v2 -> v3; conclusion 3 shows the arithmetic "
                "did not move with it."),
            "why_it_is_not_relaxed": (
                "admitting a superseded contract because equivalence was "
                "demonstrated would change what counts as a reusable scientific "
                "observation. That is a maintainer decision, not a migration "
                "one. No old->new equivalence bypass exists, the historical "
                "probes are not rewritten to v3, and their historical scoring "
                "identity is unchanged."),
        },
    }

    consistent = (bytes_valid and reconstructs and only_the_contract_fails
                  and numbers_unchanged and live_refused)
    return {
        "schema": "aadistill.autoinit.historical_reuse_position/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": (
            "The four conclusions about historical probe reuse, each DERIVED "
            "here rather than cited: valid bytes, reconstruction under the "
            "historical contract, byte-identical relocation outputs, and a "
            "REFUSED live reuse. Regenerable; relaxes nothing; authorizes "
            "nothing."),
        "conclusions": conclusions,
        "all_four_hold_simultaneously": consistent,
        "live_scoring_contract_digest": r["live_scoring_contract_digest"],
        "probes_dir_digest": r["probes_dir_digest"],
        "reuse_verified": r["reuse_verified"],
        "_what_would_change_this": (
            "a maintainer decision to admit a superseded scoring contract, or "
            "new evidence that the probes stopped reconstructing. Neither is a "
            "migration action."),
        "authorizes": "nothing",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/experiments/shared/analyses/autoinit_historical_reuse_position.json")
    args = ap.parse_args()

    doc = position()
    for name, c in doc["conclusions"].items():
        print(f"  {c['verdict']:12} {name}")
    print(f"\nall four hold simultaneously: {doc['all_four_hold_simultaneously']}")

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    # Non-zero if the four do NOT hold together: that would mean either a probe
    # stopped reconstructing or the refusal silently lapsed.
    return 0 if doc["all_four_hold_simultaneously"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
