#!/usr/bin/env python3
"""Put the authorization-carrying BEHAVIOURAL session commit where a pod can fetch it.

    PYTHONPATH=src:scripts python \\
        scripts/autoinit/stage_c2_behavioural_bundle.py \\
        --run-id attempt1 --session-commit <sha> [--dry-run]

Each session stages through its own module because the record it writes is the
one that session's `bundle_staged_gate` reads. A bundle staged through another
session's stager lands on the shared relay under the right name carrying a
record the gate reads as somebody else's.

**The record's key is the one the consumer reads.** This gate calls
`BT.roundtrip(local_bundle_sha256=record["bundle_sha256"])`. A sibling stager
writes `sha256`, and a record copied from it would parse, look right, and fail
at the gate with a `KeyError` after the whole chain had been issued.

**Ordering is not negotiable.** The bundle must contain the commit that carries
the authorization, and the authorization cannot be committed before it exists:

    grant committed -> launch-bound readiness sweep -> commit ONLY the record
    -> issue -> commit ONLY the authorization -> THAT commit is the session
    commit -> bundle it -> upload -> gates -> provider

**The record stays UNCOMMITTED inside a launch window.** Committing it adds a
third path to the session lineage diff and `session_commit_gate` refuses: the
authorized base and the session commit may differ by the authorization and
nothing else.

This step MUTATES the relay. The pre-provider gate is read-only by design, so
that what it verifies is the object a pod would actually fetch rather than a
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
for _extra in ("src", "scripts"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from experiments.phase_c2.behavioural_bundle import (  # noqa: E402
    BEHAVIOURAL_TRANSPORT, BundleTransportError, build_bundle,
    canonical_bundle_name, canonical_repo_path, stage_bundle,
)
from experiments.run_layout import rel_run_dir  # noqa: E402

EXPERIMENT_ID = "phase_c2_behavioural"
STAGE_ID = "1"


def governance_path(run_id: str, name: str, stage_id: str) -> str:
    return f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id)}/governance/{name}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and verify locally; upload nothing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    stage_id = args.stage_id or STAGE_ID
    auth_rel = governance_path(args.run_id, "authorization.json", stage_id)
    out_rel = args.out or governance_path(args.run_id, "bundle.json", stage_id)

    commit = args.session_commit.strip()
    known = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True, cwd=REPO_ROOT)
    if known.returncode != 0:
        raise SystemExit(f"{commit} is not a commit in this repository")

    #: The bundle must CARRY the authorization. Checked against the commit
    #: rather than the worktree: a file present locally and uncommitted would
    #: not be in the bundle, and the pod would check out a tree the driver
    #: cannot run.
    carries = subprocess.run(["git", "show", f"{commit}:{auth_rel}"],
                             capture_output=True, cwd=REPO_ROOT)
    if carries.returncode != 0:
        raise SystemExit(
            f"refusing to stage: {commit[:12]}… does not carry {auth_rel}. The "
            "bundle must contain the AUTHORIZATION-CARRYING commit; a bundle "
            "for the pre-authorization base checks out a tree with no "
            "authorization in it.")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            if args.dry_run:
                built = build_bundle(BEHAVIOURAL_TRANSPORT, REPO_ROOT, commit,
                                     work / canonical_bundle_name(commit))
                built["upload"] = "DRY RUN — nothing uploaded"
            else:
                built = stage_bundle(commit, workdir=work, repo_root=REPO_ROOT)
        except BundleTransportError as exc:
            raise SystemExit(f"refusing to stage: {exc}") from exc

        record = {
            "schema": "aadistill.autoinit.c2_behavioural_bundle/v1",
            "run_id": args.run_id,
            "stage_id": stage_id,
            "session_commit": commit,
            #: THE KEY `bundle_staged_gate` READS. Not `sha256`.
            "bundle_sha256": built["sha256"],
            "canonical_bundle_name": built["canonical_name"],
            "relay_repo": BEHAVIOURAL_TRANSPORT.relay_repo,
            "relay_path": canonical_repo_path(commit),
            "bytes": built["bytes"],
            "heads": built.get("heads", ""),
            "carries_authorization": auth_rel,
            "upload": built["upload"],
            "verify": built["verify"],
            "transport": BEHAVIOURAL_TRANSPORT.label,
            "note": ("the pre-provider bundle_staged_gate downloads this exact "
                     "object, verifies its sha256, checks out the session "
                     "commit from it and re-digests the AUTHORIZED executable "
                     "set inside the checkout. This record is the local half; "
                     "the gate is the remote half. It stays UNCOMMITTED inside "
                     "a launch window: committing it adds a third path to the "
                     "session lineage diff and session_commit_gate refuses."),
        }
        out = REPO_ROOT / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")

    print(json.dumps({k: record[k] for k in
                      ("session_commit", "canonical_bundle_name", "relay_path",
                       "bundle_sha256", "bytes", "upload")}, indent=1))
    print(f"wrote {out_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
