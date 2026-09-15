#!/usr/bin/env python3
"""Put the authorization-carrying C2 session commit where the pod can fetch it.

    PYTHONPATH=src python scripts/autoinit/stage_c2_bundle.py \
        --run-id attempt2 --session-commit <sha> [--dry-run]

The step whose absence cost C1 attempt 1 `$0.0786` and a 404. `bundle_staged_gate`
in the C2 launcher then refuses to launch without its result.

**Ordering is not negotiable.** The bundle must contain the commit that carries
the authorization, and the authorization cannot be committed before it exists, so
the only correct sequence is:

    grant committed -> launch-bound readiness sweep -> commit ONLY the record
    -> issue -> commit ONLY the authorization -> THAT commit is the session
    commit -> bundle it -> upload -> gates -> provider

A bundle built for the pre-authorization base and reused afterwards checks out a
tree with no authorization in it. The gate catches that, but it should never be
reached.

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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.phase_c2.bundle import (  # noqa: E402
    C2_TRANSPORT, BundleTransportError, build_bundle, canonical_bundle_name,
    canonical_repo_path, stage_bundle,
)
from experiments.phase_c2.session import (  # noqa: E402
    c2_authorization_path, c2_run_path,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-commit", required=True)
    #: REQUIRED. The bundle record belongs to the run whose authorization the
    #: bundle carries; there is no repository-level path for it.
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage-id", default=None,
                    help="the run's declared stage. Defaults to the stage the "
                         "experiment's own configuration declares.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and verify locally; upload nothing")
    ap.add_argument("--out", default=None,
                    help="explicit output path; defaults to this run's "
                         "governance/bundle.json")
    args = ap.parse_args()

    cfg = json.loads(
        (REPO_ROOT / "configs/experiments/phase_c2/authorization.json").read_text())
    stage_id = args.stage_id or cfg["stage_id"]
    auth_rel = c2_authorization_path(args.run_id, stage_id)
    if args.out is None:
        args.out = c2_run_path(args.run_id, "bundle_record", stage_id)

    commit = args.session_commit.strip()
    known = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True, cwd=REPO_ROOT)
    if known.returncode != 0:
        raise SystemExit(f"{commit} is not a commit in this repository")

    carries = subprocess.run(["git", "show", f"{commit}:{auth_rel}"],
                             capture_output=True, cwd=REPO_ROOT)
    if carries.returncode != 0:
        raise SystemExit(
            f"refusing to stage: {commit[:12]}… does not carry {auth_rel}. The "
            "bundle must contain the AUTHORIZATION-CARRYING commit; a bundle "
            "for the pre-authorization base checks out a tree the driver cannot "
            "run.")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            if args.dry_run:
                built = build_bundle(C2_TRANSPORT, REPO_ROOT, commit,
                                     work / canonical_bundle_name(commit))
                built["upload"] = "DRY RUN — nothing uploaded"
            else:
                built = stage_bundle(commit, workdir=work, repo_root=REPO_ROOT)
        except BundleTransportError as exc:
            raise SystemExit(f"refusing to stage: {exc}") from exc
        record = {
            "schema": "aadistill.autoinit.c2_bundle/v1",
            "run_id": args.run_id,
            "stage_id": stage_id,
            "session_commit": commit,
            "canonical_bundle_name": built["canonical_name"],
            "relay_repo": C2_TRANSPORT.relay_repo,
            "relay_path": canonical_repo_path(commit),
            "sha256": built["sha256"],
            "bytes": built["bytes"],
            "carries_authorization": auth_rel,
            "upload": built["upload"],
            "verify": built["verify"],
            "note": ("the pre-provider bundle_staged_gate downloads this exact "
                     "object, verifies it, checks it out and requires the "
                     "resulting HEAD, authorization bytes and executable digest "
                     "to match. This record is the local half; the gate is the "
                     "remote half. It stays UNCOMMITTED inside a launch window: "
                     "committing it adds a third path to the session lineage "
                     "diff and session_commit_gate refuses."),
        }
        #: The bundle itself is large and out of tree by policy; only its
        #: identity is recorded.
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")

    print(json.dumps({k: record[k] for k in
                      ("session_commit", "canonical_bundle_name", "relay_path",
                       "sha256", "bytes", "upload")}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
