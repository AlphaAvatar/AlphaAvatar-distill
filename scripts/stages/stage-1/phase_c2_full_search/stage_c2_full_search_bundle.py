#!/usr/bin/env python3
"""Put the authorization-carrying FULL-SEARCH session commit where a pod can fetch it.

    PYTHONPATH=src:scripts python \
        scripts/autoinit/stage_c2_full_search_bundle.py \
        --run-id attempt1 --session-commit <sha> [--dry-run]

The step whose absence cost C1 attempt 1 `$0.0786` and a 404, and the step this
session did not have: `stage_c2_bundle.py` is bound to SEARCH-1's transport and
writes Search-1's record schema, so a full-search bundle staged through it would
be uploaded under the shared relay convention with a record the full-search
`bundle_staged_gate` reads as another session's. One registry-sized file is the
whole cost of the separation, and its absence is the same class of gap as the
missing `SESSION_KIND=c2_full_search` setup dispatch branch that review found.

**Ordering is not negotiable.** The bundle must contain the commit that carries
the authorization, and the authorization cannot be committed before it exists,
so the only correct sequence is:

    grant committed -> launch-bound readiness sweep -> commit ONLY the record
    -> issue -> commit ONLY the authorization -> THAT commit is the session
    commit -> bundle it -> upload -> gates -> provider

A bundle built for the pre-authorization base and reused afterwards checks out a
tree with no authorization in it. The gate catches that, but it should never be
reached.

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

from experiments.phase_c2.full_search_bundle import (  # noqa: E402
    FULL_SEARCH_TRANSPORT, BundleTransportError, build_bundle,
    canonical_bundle_name, canonical_repo_path, stage_bundle,
)
from experiments.run_layout import rel_run_dir  # noqa: E402

EXPERIMENT_ID = "phase_c2_full_search"


def governance_path(run_id: str, name: str, stage_id: str) -> str:
    """This run's governance area, through the shared run layout.

    Derived rather than formatted, so a layout change moves the authorization,
    the readiness record and this record together.
    """
    return f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id)}/governance/{name}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
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
    args = ap.parse_args(argv)

    cfg = json.loads((REPO_ROOT / "configs/experiments/phase_c2"
                      / "full_search_authorization.json").read_text())
    stage_id = args.stage_id or str(cfg["stage_id"])
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
                built = build_bundle(FULL_SEARCH_TRANSPORT, REPO_ROOT, commit,
                                     work / canonical_bundle_name(commit))
                built["upload"] = "DRY RUN — nothing uploaded"
            else:
                built = stage_bundle(commit, workdir=work, repo_root=REPO_ROOT)
        except BundleTransportError as exc:
            raise SystemExit(f"refusing to stage: {exc}") from exc

        record = {
            "schema": "aadistill.autoinit.c2_full_search_bundle/v1",
            "run_id": args.run_id,
            "stage_id": stage_id,
            "session_commit": commit,
            "canonical_bundle_name": built["canonical_name"],
            "relay_repo": FULL_SEARCH_TRANSPORT.relay_repo,
            "relay_path": canonical_repo_path(commit),
            "sha256": built["sha256"],
            "bytes": built["bytes"],
            "heads": built.get("heads", ""),
            "carries_authorization": auth_rel,
            "upload": built["upload"],
            "verify": built["verify"],
            "transport": FULL_SEARCH_TRANSPORT.label,
            "note": ("the pre-provider bundle_staged_gate downloads this exact "
                     "object, verifies its sha256, checks out the session "
                     "commit from it and re-digests the AUTHORIZED executable "
                     "set inside the checkout. This record is the local half; "
                     "the gate is the remote half. It stays UNCOMMITTED inside "
                     "a launch window: committing it adds a third path to the "
                     "session lineage diff and session_commit_gate refuses."),
        }
        #: The bundle itself is large and out of tree by policy; only its
        #: identity is recorded.
        out = REPO_ROOT / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=1) + "\n")

    print(json.dumps({k: record[k] for k in
                      ("session_commit", "canonical_bundle_name", "relay_path",
                       "sha256", "bytes", "upload")}, indent=1))
    print(f"wrote {out_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
