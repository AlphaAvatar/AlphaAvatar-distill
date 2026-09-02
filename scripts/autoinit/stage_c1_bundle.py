#!/usr/bin/env python3
"""Put the authorization-carrying C1 session commit where the pod can fetch it.

    PYTHONPATH=src python scripts/autoinit/stage_c1_bundle.py \
        --session-commit <sha> [--dry-run]

The step whose absence cost C1 attempt 1 `$0.0786` and a 404. It was documented
in `scripts/pod/AGENTS.md` and depended on an operator remembering it; this makes
it a command with a name, and `bundle_staged_gate` then refuses to launch without
its result.

**Ordering is not negotiable.** The bundle must contain the commit that carries
the authorization, and the authorization cannot be committed before it exists, so
the only correct sequence is:

    repair base -> issue -> commit ONLY the authorization -> THAT commit is the
    session commit -> bundle it -> upload -> gate -> provider

A bundle built for the pre-authorization base and reused afterwards checks out a
tree with no `logs/autoinit_c1_authorization.json` in it. The gate catches that,
but it should never be reached.

This step MUTATES the relay. The pre-provider gate is read-only by design, so
that what it verifies is the object the pod would actually fetch rather than a
side effect of the check itself.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.c1_bundle import (  # noqa: E402
    RELAY_REPO, C1BundleError, build_bundle, canonical_bundle_name,
    canonical_repo_path, stage_bundle,
)

AUTH_PATH = "logs/autoinit_c1_authorization.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and verify locally; upload nothing")
    ap.add_argument("--out", default="logs/autoinit_c1_bundle.json")
    args = ap.parse_args()

    commit = args.session_commit.strip()
    known = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True, cwd=REPO_ROOT)
    if known.returncode != 0:
        raise SystemExit(f"{commit} is not a commit in this repository")

    carries = subprocess.run(["git", "show", f"{commit}:{AUTH_PATH}"],
                             capture_output=True, cwd=REPO_ROOT)
    if carries.returncode != 0:
        raise SystemExit(
            f"refusing to stage: {commit[:12]}… does not carry {AUTH_PATH}. The "
            "bundle must contain the AUTHORIZATION-CARRYING commit; a bundle for "
            "the pre-authorization base checks out a tree the driver cannot run.")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            if args.dry_run:
                built = build_bundle(REPO_ROOT, commit,
                                     work / canonical_bundle_name(commit))
                built["upload"] = "DRY RUN — nothing uploaded"
            else:
                built = stage_bundle(REPO_ROOT, commit, workdir=work)
        except C1BundleError as exc:
            raise SystemExit(f"refusing to stage: {exc}") from exc
        record = {
            "schema": "aadistill.autoinit.c1_bundle/v1",
            "session_commit": commit,
            "canonical_bundle_name": built["canonical_name"],
            "relay_repo": RELAY_REPO,
            "relay_path": canonical_repo_path(commit),
            "sha256": built["sha256"],
            "bytes": built["bytes"],
            "carries_authorization": AUTH_PATH,
            "upload": built["upload"],
            "verify": built["verify"],
            "note": ("the pre-provider bundle_staged_gate downloads this exact "
                     "object, verifies it, checks it out and requires the "
                     "resulting HEAD, authorization and harness digest to match. "
                     "This record is the local half; the gate is the remote half."),
        }
        # The bundle itself is large and out of tree by policy; only its identity
        # is committed.
        (REPO_ROOT / args.out).write_text(json.dumps(record, indent=1) + "\n")

    print(json.dumps({k: record[k] for k in
                      ("session_commit", "canonical_bundle_name", "relay_path",
                       "sha256", "bytes", "upload")}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
