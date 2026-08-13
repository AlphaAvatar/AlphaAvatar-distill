"""Issue the micro-preflight spend authorization against the rehearsed harness.

    PYTHONPATH=src python scripts/autoinit/issue_authorization.py

Run this AFTER the harness rehearsal passes and the harness is committed. It
binds the authorization to three things the launcher then enforces: the preflight
plan hash, the harness source digest, and the session commit. Re-issuing is the
correct response to editing the harness -- and re-rehearsing first is the point.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.authorization import (  # noqa: E402
    MICRO_PREFLIGHT_AUTHORIZATION, harness_source_digest,
)
from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_micro_preflight_authorization.json")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, cwd=REPO_ROOT).stdout.strip())
    harness = harness_source_digest(REPO_ROOT)
    auth = replace(MICRO_PREFLIGHT_AUTHORIZATION,
                   authorized_session_commit=commit,
                   harness_source_digest=harness["digest"],
                   provenance_commit=f"{commit}{'+dirty' if dirty else ''}")
    auth.require_plan(PREFLIGHT_PLAN_V1.plan_hash)
    auth.require_harness(REPO_ROOT)
    payload = auth.as_dict()
    (REPO_ROOT / args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "authorization_sha256": payload["authorization_sha256"],
        "harness_source_digest": harness["digest"],
        "authorized_session_commit": commit,
        "working_tree_dirty": dirty,
        "expected_usd": auth.expected_usd, "hard_cap_usd": auth.hard_cap_usd,
        "phase_a_authorized": auth.allows_phase_a,
    }, indent=2))


if __name__ == "__main__":
    main()
