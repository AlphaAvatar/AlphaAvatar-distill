"""Issue the recovery-continuation authorization. Zero cost; launches nothing.

    PYTHONPATH=src python \
        scripts/autoinit/issue_recovery_continuation_authorization.py \
        --grant logs/<a one-use continuation grant>.json --require-clean

**Why this is not `--out` on the Phase-A issuer.** That issuer binds
`PHASE_A_HARNESS_SOURCE_FILES_V1`, which measures the full launcher, driver and
beam search and contains neither continuation file. Pointing it at a different
output path would have produced an artifact whose harness digest did not measure
the executable the paid continuation actually runs — green, and certifying
something it never read. It would also have carried the search's $23.0484
ceiling into a session priced at $16.7456.

Full Phase A and the recovery continuation are **distinct operational
harnesses**, independently measured. This script differs from its Phase-A sibling
in exactly three ways:

* the harness digest covers the continuation closure — the Phase-A set minus the
  unreachable search, plus the continuation launcher, driver, strict Stage-1
  importer, device handoff, leaf durability and the schema module;
* the price is **derived from `continuation_budget()`** at issue time rather than
  read from a constant, so the artifact cannot claim a ceiling the launcher's own
  pricing would not produce;
* the grant must say it is a continuation grant, so a Phase-A grant document
  cannot be reused to authorize this session by accident.

Everything else is deliberately identical: the same frozen session plan hash, the
same science plan hash read from the frozen plan on disk, the same session-commit
binding, the same real wall-clock timestamp, the same refusal to issue against a
dirty tree.

Issuing is not launching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pod"))

from aadistill.autoinit.phase_a import PHASE_A_PLAN_V1  # noqa: E402
from aadistill.autoinit.recovery_continuation import (  # noqa: E402
    CONTINUATION_GRANT_PROSE_REQUIRED, RECOVERY_CONTINUATION_AUTHORIZATION,
    RECOVERY_CONTINUATION_HARNESS_FILES_V1, RecoveryContinuationAuthorization,
    recovery_continuation_harness_digest,
)

FROZEN_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"
OUT = "logs/autoinit_recovery_continuation_authorization.json"

#: What a continuation grant document must say. Nothing here is derived: a
#: maintainer writes it, and everything the ISSUER can establish for itself — the
#: timestamp, the committed base, the harness digest, both plan hashes, and the
#: price — is deliberately absent, so a grant cannot assert an identity or a
#: ceiling it did not check.
GRANT_FIELDS = ("granted_by", "covers", "cumulative_spend_at_approval_usd",
                "cumulative_cap_usd", "does_not_authorize", "grant_type")

GRANT_TYPE = "recovery_continuation"


def load_grant(path: Path) -> dict:
    grant = json.loads(path.read_text())
    # Presence, not truthiness: a cumulative spend of 0.0 is a real figure.
    missing = [f for f in GRANT_FIELDS
               if f not in grant
               or (isinstance(grant[f], str) and not grant[f].strip())
               or grant[f] is None]
    if missing:
        raise SystemExit(
            f"refusing to issue: {path} is missing {missing}. A grant states who "
            "permitted what, at what cumulative spend, and what it does NOT "
            "authorize; an artifact issued without those cannot be audited later.")
    if grant["grant_type"] != GRANT_TYPE:
        raise SystemExit(
            f"refusing to issue: {path} is a {grant['grant_type']!r} grant, not "
            f"a {GRANT_TYPE!r} one. A grant for the full Phase-A session "
            "approved a beam search at a search's price; it does not carry over "
            "to a session that imports Stage 1, and reusing it would let one "
            "maintainer decision authorize two different runs.")
    for derived in ("granted_utc", "authorized_session_commit",
                    "harness_source_digest", "plan_hash", "science_plan_hash",
                    "expected_usd", "hard_cap_usd", "per_launch_hard_usd",
                    "authorization_sha256"):
        if derived in grant:
            raise SystemExit(
                f"refusing to issue: {path} sets {derived!r}, which this script "
                "derives. A grant that asserts an identity or a price it did not "
                "compute is how a stale binding gets authorized.")
    return grant


def derived_pricing(args) -> dict:
    """Price the continuation from the launcher's own `continuation_budget()`.

    Not a transcription. `$14.9233` and `$16.7456` appear in this project's prose
    because they are what this call returns; if the step-time model, the probe
    counts or the contingency move, the artifact moves with them, and a written
    constant would not.

    Priced against an unbounded authorization on purpose: the ceiling is the
    *output* of this function, and passing the number being derived as the limit
    that constrains the derivation would make it unfalsifiable.
    """
    import autoinit_phase_a_launch as launcher

    plan = launcher.continuation_budget(args).plan(
        price_per_hour=args.max_price, authorized_usd=float("inf"))
    # The 4-dp CEILING, for the reason the Phase-A cap is one: the runner prices
    # against `hard_cap_usd` and `plan()` refuses `hard_terminate > authorized`,
    # so a cap rounded DOWN would make the launcher refuse its own plan by a
    # fraction of a cent.
    return {
        "expected_usd": round(plan.expected_usd, 4),
        "hard_cap_usd": math.ceil(plan.hard_terminate_usd * 1e4) / 1e4,
        "soft_stop_usd": round(plan.soft_stop_usd, 4),
        "expected_minutes": round(plan.expected_minutes, 2),
        "price_per_hour": args.max_price,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--grant", required=True,
                    help="a one-use continuation grant document: who permitted "
                         "what, at what cumulative spend, and what it does not "
                         "authorize")
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
    dirt = [ln for ln in dirty_paths.splitlines() if args.out not in ln]
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
            f"refusing to issue: no grant document at {grant_path}. The recovery "
            "continuation is not authorized by the existence of this script.")
    grant = load_grant(grant_path)
    try:
        grant_name = str(grant_path.relative_to(REPO_ROOT))
    except ValueError:                      # a grant outside the repository
        grant_name = str(grant_path)

    frozen = json.loads((REPO_ROOT / args.frozen_plan).read_text())
    science_plan_hash = frozen["plan_hash"]

    harness = recovery_continuation_harness_digest(REPO_ROOT)
    if any(f["path"] in ("scripts/autoinit/phase_a_search.py",)
           for f in harness["files"]):
        raise SystemExit(
            "refusing to issue: the search module is inside the continuation "
            "harness. Either the session can reach a search — in which case this "
            "authorization is the wrong one — or the file set is wrong.")

    granted = datetime.now(timezone.utc)
    if RECOVERY_CONTINUATION_AUTHORIZATION.granted_by != CONTINUATION_GRANT_PROSE_REQUIRED:
        raise SystemExit(
            "refusing to issue: the authorization SCHEMA carries grant prose. "
            "A grant is a one-use decision and does not belong in executable "
            "source; put it in the --grant document.")

    # The launcher's real parser, so the price is the price the launcher would
    # compute. Transcribing defaults here is the class of defect that makes a
    # validator disagree with the thing it validates.
    import autoinit_recovery_continuation_launch as cont
    launch_args = cont.build_parser().parse_args(
        ["--scr", "/tmp/pricing-only", "--session-commit", commit or "0" * 40,
         "--bundle", "pricing-only.bundle"])
    pricing = derived_pricing(launch_args)

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
        RECOVERY_CONTINUATION_AUTHORIZATION,
        granted_by=granted_by,
        authorization_id=f"autoinit.recovery_continuation.{granted:%Y-%m-%dT%H%MZ}",
        granted_utc=granted.strftime("%Y-%m-%dT%H:%M:%SZ"),
        science_plan_hash=science_plan_hash,
        expected_usd=pricing["expected_usd"],
        hard_cap_usd=pricing["hard_cap_usd"],
        #: One session, so the per-launch limit IS the cap — the check that
        #: stopped a single run spending a cumulative allowance.
        per_launch_hard_usd=pricing["hard_cap_usd"],
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
    assert auth.allows_beam_search is False
    assert auth.authorizes_recovery_continuation is True
    assert auth.automatic_followon_start is False
    # And the priced plan must fit inside the ceiling just derived from it.
    auth.require_within_cap(pricing["hard_cap_usd"])

    out_path = REPO_ROOT / args.out
    payload = auth.as_dict()
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    # Round-trip through the loader that the launcher will use, from the bytes
    # just written. An artifact that this script can build but the launcher
    # cannot load is a failure discovered at $0 here or on a paid pod later.
    reloaded = RecoveryContinuationAuthorization.load(out_path)
    assert reloaded.hard_cap_usd == auth.hard_cap_usd
    assert reloaded.harness_source_files == RECOVERY_CONTINUATION_HARNESS_FILES_V1
    reloaded.require_harness(REPO_ROOT)

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
        "harness_covers_search": False,
        "authorized_session_commit": commit,
        "working_tree_dirty": bool(dirt),
        "pricing": pricing,
        "expected_usd": auth.expected_usd,
        "hard_cap_usd": auth.hard_cap_usd,
        "per_launch_hard_usd": auth.per_launch_hard_usd,
        "authorized_stages": list(auth.authorized_stages),
        "phase_a_authorized": auth.allows_phase_a,
        "allows_beam_search": auth.allows_beam_search,
        "authorizes_recovery_continuation": auth.authorizes_recovery_continuation,
        "automatic_followon_start": auth.automatic_followon_start,
        "launched": False,
    }, indent=2))


if __name__ == "__main__":
    main()
