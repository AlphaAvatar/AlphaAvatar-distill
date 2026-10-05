#!/usr/bin/env python3
"""Build, verify and upload the bundle a D1 pod fetches. MUTATES THE RELAY.

    PYTHONPATH=src:scripts python scripts/autoinit/stage_d1_bundle.py \
        --session-commit <sha> --run-id d1_search_20261006_120000

**D1 had neither this command nor a gate that would have missed it.** The
prepared chain's readiness record named `transfer/aad_autoinit_6c1dd8d9.bundle`
as a verified fact; the relay holds 563 `transfer/` objects and over a hundred
`aad_autoinit_*.bundle`s, and that was not one of them. A launch would have
created a pod, paid for startup and died at `SETUP_RC=1` on a 404 -- which is C1
attempt 1 exactly, the failure `aadistill.infrastructure.bundle_transport` was
written to make impossible.

The split is the shared module's: preparation (here) may mutate the relay; the
launcher's `bundle_staged_gate` is read-only and verifies what preparation left.
`stage_bundle` refuses to overwrite a different object already under the
canonical name, because a launcher may already have verified those bytes.

The record this writes stays UNCOMMITTED inside a launch window. The
session-commit gate runs with `check_lineage=True`, which permits exactly one
path to differ between the authorized base and the session commit -- the
authorization artifact -- so committing this record would make the lineage check
refuse the launch it was prepared for.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.bundle_transport import (  # noqa: E402
    BundleTransportError, build_bundle, stage_bundle,
)
from experiments.phase_d1 import d1_session as D1S  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402


def governance_path(run_id: str, name: str) -> str:
    return (f"{rel_run_dir(D1S.EXPERIMENT_ID, run_id, D1S.STAGE_ID)}"
            f"/governance/{name}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--run-id", required=True,
                    help="the run whose authorization the bundle must carry")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and verify locally; upload nothing")
    ap.add_argument("--out", default=None,
                    help="explicit output path; defaults to this run's "
                         "governance/bundle.json")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    commit = args.session_commit.strip()
    transport = D1S.transport()

    known = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True, cwd=REPO_ROOT)
    if known.returncode != 0:
        raise SystemExit(f"{commit} is not a commit in this repository")

    auth_rel = governance_path(args.run_id, "authorization.json")
    out_rel = args.out or governance_path(args.run_id, "bundle.json")

    #: THE BUNDLE MUST CARRY THE AUTHORIZATION, checked against the COMMIT and
    #: not the worktree: a file present locally and uncommitted is not in the
    #: bundle, and the pod would check out a tree whose driver has no grant to
    #: load.
    carries = subprocess.run(["git", "show", f"{commit}:{auth_rel}"],
                             capture_output=True, cwd=REPO_ROOT)
    if carries.returncode != 0:
        raise SystemExit(
            f"refusing to stage: {commit[:12]} does not carry {auth_rel}. The "
            "bundle must contain the AUTHORIZATION-CARRYING commit; a bundle "
            "for the pre-authorization base checks out a tree with no "
            "authorization in it.")

    with tempfile.TemporaryDirectory(prefix="d1-bundle-") as tmp:
        work = Path(tmp)
        try:
            if args.dry_run:
                built = build_bundle(
                    transport, REPO_ROOT, commit,
                    work / transport.bundle_name(commit))
                built["upload"] = "DRY RUN - nothing uploaded"
            else:
                built = stage_bundle(transport, REPO_ROOT, commit, workdir=work)
        except BundleTransportError as exc:
            raise SystemExit(f"refusing to stage: {exc}") from None

        record = {
            "schema": "aadistill.phase_d1.bundle/v1",
            "run_id": args.run_id,
            "stage_id": D1S.STAGE_ID,
            "session_commit": commit,
            "canonical_bundle_name": built["canonical_name"],
            "relay_repo": transport.relay_repo,
            "relay_path": transport.repo_path(commit),
            "sha256": built["sha256"],
            "bytes": built["bytes"],
            "heads": built.get("heads", ""),
            "carries_authorization": auth_rel,
            "upload": built["upload"],
            "verify": built["verify"],
            "transport": transport.label,
            "note": ("the pre-provider bundle_staged_gate downloads this exact "
                     "object, verifies its sha256 against the value here, "
                     "checks out the session commit from it, confirms it "
                     "carries this run's authorization bytes and re-digests the "
                     "AUTHORIZED executable set inside the checkout. This "
                     "record is the local half; the gate is the remote half. It "
                     "stays UNCOMMITTED inside a launch window: committing it "
                     "adds a path to the session lineage diff and "
                     "session_commit_gate refuses."),
        }
        #: Only the bundle's IDENTITY is recorded. The object itself is large
        #: and lives on the relay, out of tree by policy.
        out = REPO_ROOT / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")

    print(json.dumps({k: record[k] for k in
                      ("session_commit", "canonical_bundle_name", "relay_path",
                       "sha256", "bytes", "upload")}, indent=1))
    print(f"wrote {out_rel}  (UNCOMMITTED by design; see the note in it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
