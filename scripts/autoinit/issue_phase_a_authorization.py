"""Issue the Phase-A authorization. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/issue_phase_a_authorization.py \
        --grant logs/<a one-use grant document>.json --require-clean

Run AFTER the rehearsal passes and the harness is committed.

**The grant is an input, not a constant.** `src/aadistill/autoinit/phase_a.py`
carries the authorization SCHEMA — caps, stages, stage conditions, scope — and
nothing about a particular permission. Until 2026-08-18 it also carried the
attempt-7 grant prose: which attempt it covered, the cumulative spend at
approval, what it did not authorize. That is a one-use maintainer decision living
in executable source, where it goes stale silently and still reads as though it
applies. It now arrives in `--grant`, and issuing without one is refused.

It binds the grant to four identities, and editing any of them is meant to
invalidate it:

* the **session plan** hash (`PHASE_A_PLAN_V1`);
* the **science plan** hash, read from the frozen plan on disk rather than from a
  constant, so a threshold that moved after freezing cannot be authorized;
* the **Phase-A harness digest** — its own launcher, driver, search module and
  plan module, not the preflight's file set;
* the **session commit**, which is what the pod checks out from the bundle.

The grant timestamp is real wall-clock UTC taken at issue time. It is not a
constant and must not be back-dated: the continuation burned a paid session on a
stale binding, and an authorization whose timestamp predates the harness it
covers is exactly that failure in a different costume.

Issuing is not launching. The artifact authorizes spend and stages; starting the
run is a separate, explicit decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.phase_a import (  # noqa: E402
    GRANT_PROSE_REQUIRED, PHASE_A_AUTHORIZATION,
    PHASE_A_HARNESS_SOURCE_FILES_V1, PHASE_A_PLAN_V1, phase_a_harness_digest,
)

FROZEN_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"

#: What a grant document must say. Nothing here is derived: a maintainer writes
#: it, and everything the ISSUER can establish for itself — the timestamp, the
#: committed base, the harness digest, both plan hashes — is deliberately absent
#: so a grant cannot assert an identity it did not check.
GRANT_FIELDS = ("granted_by", "covers", "cumulative_spend_at_approval_usd",
                "cumulative_cap_usd", "does_not_authorize")


def load_grant(path: Path) -> dict:
    grant = json.loads(path.read_text())
    # Presence, not truthiness. A cumulative spend of 0.0 is a real figure, and
    # `if not grant.get(f)` would refuse it as missing — the class of bug that
    # makes a validator reject the one case it most needs to accept.
    missing = [f for f in GRANT_FIELDS
               if f not in grant
               or (isinstance(grant[f], str) and not grant[f].strip())
               or grant[f] is None]
    if missing:
        raise SystemExit(
            f"refusing to issue: {path} is missing {missing}. A grant states who "
            "permitted what, at what cumulative spend, and what it does NOT "
            "authorize; an artifact issued without those cannot be audited later.")
    for derived in ("granted_utc", "authorized_session_commit",
                    "harness_source_digest", "plan_hash", "science_plan_hash",
                    "authorization_sha256"):
        if derived in grant:
            raise SystemExit(
                f"refusing to issue: {path} sets {derived!r}, which this script "
                "derives. A grant that asserts an identity it did not compute is "
                "how a stale binding gets authorized.")
    return grant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_phase_a_authorization.json")
    ap.add_argument("--grant", required=True,
                    help="a one-use grant document: who permitted what, at what "
                         "cumulative spend, and what it does not authorize")
    ap.add_argument("--frozen-plan", default=FROZEN_PLAN)
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse to issue against a dirty working tree, because "
                         "the pod checks out a commit and would not run the "
                         "uncommitted edits this digest claims to cover")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()
    dirty_paths = subprocess.run(["git", "status", "--porcelain"],
                                 capture_output=True, text=True,
                                 cwd=REPO_ROOT).stdout.strip()
    # The authorization artifact itself is written by this script and cannot be
    # committed before it exists, so its own path never counts as dirt.
    dirt = [ln for ln in dirty_paths.splitlines()
            if args.out not in ln]
    if dirt and args.require_clean:
        raise SystemExit(
            "refusing to issue: the working tree is dirty in "
            f"{[ln.strip() for ln in dirt][:6]}. The pod checks out {commit} "
            "from a bundle, so uncommitted edits would not be the code that "
            "runs, while the harness digest would claim they were.")

    grant_path = Path(args.grant)
    if not grant_path.is_absolute():
        grant_path = REPO_ROOT / grant_path
    if not grant_path.is_file():
        raise SystemExit(
            f"refusing to issue: no grant document at {grant_path}. Phase A is "
            "not authorized by the existence of this script.")
    grant = load_grant(grant_path)
    try:
        grant_name = str(grant_path.relative_to(REPO_ROOT))
    except ValueError:                      # a grant outside the repository
        grant_name = str(grant_path)

    frozen = json.loads((REPO_ROOT / args.frozen_plan).read_text())
    science_plan_hash = frozen["plan_hash"]

    harness = phase_a_harness_digest(
        REPO_ROOT, files=PHASE_A_HARNESS_SOURCE_FILES_V1)
    granted = datetime.now(timezone.utc)
    if PHASE_A_AUTHORIZATION.granted_by != GRANT_PROSE_REQUIRED:
        raise SystemExit(
            "refusing to issue: the authorization SCHEMA carries grant prose. "
            "A grant is a one-use decision and does not belong in executable "
            "source; put it in the --grant document.")
    granted_by = "\n\n".join([
        grant["granted_by"].strip(),
        f"Covers: {grant['covers']}",
        (f"Cumulative spend at approval "
         f"${float(grant['cumulative_spend_at_approval_usd']):.4f} against a "
         f"cumulative cap of ${float(grant['cumulative_cap_usd']):.2f}."),
        f"Does NOT authorize: {grant['does_not_authorize']}",
        (f"Grant document: {grant_name} "
         f"sha256 {hashlib.sha256(grant_path.read_bytes()).hexdigest()}"),
    ])
    auth = replace(
        PHASE_A_AUTHORIZATION,
        granted_by=granted_by,
        authorization_id=f"autoinit.phase_a.{granted:%Y-%m-%dT%H%MZ}",
        granted_utc=granted.strftime("%Y-%m-%dT%H:%M:%SZ"),
        science_plan_hash=science_plan_hash,
        authorized_session_commit=commit,
        harness_source_digest=harness["digest"],
        provenance_commit=f"{commit}{'+dirty' if dirt else ''}")

    # Every gate the launcher and driver will apply, applied here first, so a
    # broken artifact never reaches a pod.
    auth.require_plan(PHASE_A_PLAN_V1.plan_hash)
    auth.require_science_plan(science_plan_hash)
    auth.require_harness(REPO_ROOT)
    for stage in range(6):
        auth.require_stage(stage)
    assert auth.allows_phase_a is True
    assert auth.automatic_followon_start is False

    payload = auth.as_dict()
    (REPO_ROOT / args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "authorization_id": auth.authorization_id,
        "granted_utc": auth.granted_utc,
        "grant_document": grant_name,
        "grant_document_sha256": hashlib.sha256(grant_path.read_bytes()).hexdigest(),
        "authorization_sha256": payload["authorization_sha256"],
        "session_plan_hash": PHASE_A_PLAN_V1.plan_hash,
        "science_plan_hash": science_plan_hash,
        "harness_source_digest": harness["digest"],
        "harness_source_files": [f["path"] for f in harness["files"]],
        "authorized_session_commit": commit,
        "working_tree_dirty": bool(dirt),
        "expected_usd": auth.expected_usd,
        "hard_cap_usd": auth.hard_cap_usd,
        "per_launch_hard_usd": auth.per_launch_hard_usd,
        "authorized_stages": list(auth.authorized_stages),
        "phase_a_authorized": auth.allows_phase_a,
        "automatic_followon_start": auth.automatic_followon_start,
        "launched": False,
    }, indent=2))


if __name__ == "__main__":
    main()
