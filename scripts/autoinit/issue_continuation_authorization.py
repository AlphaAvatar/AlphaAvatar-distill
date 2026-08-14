"""Issue the narrow characterization-continuation authorization.

    PYTHONPATH=src python scripts/autoinit/issue_continuation_authorization.py

Run AFTER the continuation rehearsal passes and the harness is committed. It
binds the authorization to the continuation plan hash, the CONTINUATION harness
digest — its own launcher, driver and plan module, not the preflight's files —
and the session commit. Editing any of those is meant to invalidate it: re-run
the rehearsal, re-commit, re-issue.

It authorizes characterization only. Nothing in this artifact can permit
training, retraining a control, or Phase A.
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

from aadistill.autoinit.authorization import harness_source_digest  # noqa: E402
from aadistill.autoinit.continuation import (  # noqa: E402
    CONTINUATION_AUTHORIZATION, CONTINUATION_HARNESS_SOURCE_FILES_V1,
    CONTINUATION_PLAN_V1,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_continuation_authorization.json")
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse to issue against a dirty working tree, because "
                         "the pod checks out a commit and would not run the "
                         "uncommitted edits this digest covers")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, cwd=REPO_ROOT).stdout.strip())
    if dirty and args.require_clean:
        raise SystemExit(
            "refusing to issue: the working tree is dirty. The pod checks out "
            f"{commit} from a bundle, so uncommitted edits would not be the "
            "code that runs, while the harness digest would claim they were.")
    harness = harness_source_digest(
        REPO_ROOT, files=CONTINUATION_HARNESS_SOURCE_FILES_V1)
    auth = replace(CONTINUATION_AUTHORIZATION,
                   authorized_session_commit=commit,
                   harness_source_digest=harness["digest"],
                   provenance_commit=f"{commit}{'+dirty' if dirty else ''}")
    auth.require_plan(CONTINUATION_PLAN_V1.plan_hash)
    auth.require_harness(REPO_ROOT)
    payload = auth.as_dict()
    (REPO_ROOT / args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "authorization_sha256": payload["authorization_sha256"],
        "plan_hash": CONTINUATION_PLAN_V1.plan_hash,
        "harness_source_digest": harness["digest"],
        "harness_source_files": [f["path"] for f in harness["files"]],
        "authorized_session_commit": commit,
        "working_tree_dirty": dirty,
        "expected_usd": auth.expected_usd, "hard_cap_usd": auth.hard_cap_usd,
        "authorized_stages": list(auth.authorized_stages),
        "phase_a_authorized": auth.allows_phase_a,
    }, indent=2))


if __name__ == "__main__":
    main()
