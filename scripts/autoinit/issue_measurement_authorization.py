"""Issue the bounded causal-depth measurement authorization. Zero cost.

    PYTHONPATH=src python scripts/autoinit/issue_measurement_authorization.py \
        --grant logs/<a one-use grant document>.json --require-clean

The same grant/schema split the Phase-A issuer uses, for the same reason: who
permitted what, at what cumulative spend, is a one-use maintainer decision, and
putting it in executable source is how it goes stale silently while still reading
as though it applies.

This issues a **SpendAuthorization**, whose `allows_phase_a` is a hard `False`.
The measurement therefore cannot start Phase A whatever it is pointed at, and
that is a property of the type rather than a promise in a comment.

Issuing is not launching.
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

from aadistill.autoinit.authorization import harness_source_digest  # noqa: E402
from aadistill.autoinit.measurement import (  # noqa: E402
    MEASUREMENT_AUTHORIZATION, MEASUREMENT_PLAN_V1,
)
from aadistill.autoinit.phase_a import GRANT_PROSE_REQUIRED  # noqa: E402

#: What the measurement session actually executes. NOT the Phase-A set: this
#: session runs its own launcher and its own job, and digesting Phase-A's files
#: would certify code this run never touches while leaving the measurement job
#: itself unmeasured.
MEASUREMENT_HARNESS_FILES: tuple[str, ...] = (
    "scripts/pod/autoinit_measurement_launch.py",
    "scripts/autoinit/measure_causal_depth_runtime.py",
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/autoinit_science_inputs.py",
    "scripts/pod/watchdog.py",
    "scripts/pod/collect_artifacts.py",
    "src/aadistill/autoinit/authorization.py",
    "src/aadistill/autoinit/measurement.py",
    "src/aadistill/autoinit/operators/depth.py",
    "src/aadistill/init/contribution.py",
    "src/aadistill/infrastructure/session.py",
    "src/aadistill/infrastructure/session_prechecks.py",
    "src/aadistill/infrastructure/session_runner.py",
)

GRANT_FIELDS = ("granted_by", "covers", "cumulative_spend_at_approval_usd",
                "cumulative_cap_usd", "does_not_authorize")


def load_grant(path: Path) -> dict:
    grant = json.loads(path.read_text())
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
                    "harness_source_digest", "plan_hash", "authorization_sha256"):
        if derived in grant:
            raise SystemExit(
                f"refusing to issue: {path} sets {derived!r}, which this script "
                "derives. A grant that asserts an identity it did not compute is "
                "how a stale binding gets authorized.")
    return grant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_measurement_authorization.json")
    ap.add_argument("--grant", required=True,
                    help="a one-use grant document: who permitted what, at what "
                         "cumulative spend, and what it does not authorize")
    ap.add_argument("--require-clean", action="store_true")
    args = ap.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = [ln for ln in subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
        cwd=REPO_ROOT).stdout.strip().splitlines() if args.out not in ln]
    if dirty and args.require_clean:
        raise SystemExit(
            "refusing to issue: the working tree is dirty in "
            f"{[l.strip() for l in dirty][:6]}. The pod checks out {commit} from "
            "a bundle, so uncommitted edits would not be the code that runs "
            "while the harness digest would claim they were.")

    grant_path = Path(args.grant)
    if not grant_path.is_absolute():
        grant_path = REPO_ROOT / grant_path
    if not grant_path.is_file():
        raise SystemExit(
            f"refusing to issue: no grant document at {grant_path}. A paid "
            "measurement is not authorized by the existence of this script.")
    grant = load_grant(grant_path)

    if MEASUREMENT_AUTHORIZATION.granted_by != GRANT_PROSE_REQUIRED:
        raise SystemExit(
            "refusing to issue: the authorization SCHEMA carries grant prose.")

    harness = harness_source_digest(REPO_ROOT, files=MEASUREMENT_HARNESS_FILES)
    granted = datetime.now(timezone.utc)
    granted_by = "\n\n".join([
        grant["granted_by"].strip(),
        f"Covers: {grant['covers']}",
        (f"Cumulative spend at approval "
         f"${float(grant['cumulative_spend_at_approval_usd']):.4f} against a "
         f"cumulative cap of ${float(grant['cumulative_cap_usd']):.2f}."),
        f"Does NOT authorize: {grant['does_not_authorize']}",
        (f"Grant document: {grant_path.relative_to(REPO_ROOT)} "
         f"sha256 {hashlib.sha256(grant_path.read_bytes()).hexdigest()}"),
    ])
    auth = replace(
        MEASUREMENT_AUTHORIZATION,
        granted_by=granted_by,
        authorization_id=f"autoinit.measurement.{granted:%Y-%m-%dT%H%MZ}",
        granted_utc=granted.strftime("%Y-%m-%dT%H:%M:%SZ"),
        authorized_session_commit=commit,
        harness_source_digest=harness["digest"],
        harness_source_files=MEASUREMENT_HARNESS_FILES,
        provenance_commit=f"{commit}{'+dirty' if dirty else ''}")

    auth.require_plan(MEASUREMENT_PLAN_V1.plan_hash)
    auth.require_harness(REPO_ROOT)
    auth.require_stage(0)
    assert auth.allows_phase_a is False, "a measurement may not authorize Phase A"

    payload = auth.as_dict()
    (REPO_ROOT / args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "authorization_id": auth.authorization_id,
        "granted_utc": auth.granted_utc,
        "grant_document": str(grant_path.relative_to(REPO_ROOT)),
        "grant_document_sha256": hashlib.sha256(grant_path.read_bytes()).hexdigest(),
        "authorization_sha256": payload["authorization_sha256"],
        "plan_hash": MEASUREMENT_PLAN_V1.plan_hash,
        "harness_source_digest": harness["digest"],
        "harness_source_files": [f["path"] for f in harness["files"]],
        "authorized_session_commit": commit,
        "working_tree_dirty": bool(dirty),
        "expected_usd": auth.expected_usd,
        "hard_cap_usd": auth.hard_cap_usd,
        "per_launch_hard_usd": auth.per_launch_hard_usd,
        "authorized_stages": list(auth.authorized_stages),
        "phase_a_authorized": auth.allows_phase_a,
        "launched": False,
    }, indent=2))


if __name__ == "__main__":
    main()
