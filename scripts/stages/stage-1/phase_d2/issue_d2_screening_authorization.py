#!/usr/bin/env python3
"""Issue D2's ONE-USE behavioural authorization (screening or the shared
confirmation), from a logged command.

    ... --rung confirmation --grant logs/budget/approvals/
        autoinit_d2_confirmation_grant.json --run-id d2_confirmation_<utc>

    PYTHONPATH=src:scripts python \
        scripts/stages/stage-1/phase_d2/issue_d2_screening_authorization.py \
        --grant logs/budget/approvals/autoinit_d2_screening_grant.json \
        --run-id d2_screening_<utc>

The D2 search issuer's shape, for the screening type. Everything of substance
is `d2_screening_authorization.build_payload`'s: it refuses an unapproved
preregistration, a grant without a cumulative envelope, a phase outside the
formal book, an envelope the earlier subruns have narrowed below the session's
price, and a ceiling the four money conditions do not fund; it quotes the live
rate and derives the contract hash on this tree. This command reads the grant,
refuses a dirty tree, takes HEAD as the session commit, writes the artifact
ONCE and reads it back through the type the pod loads.

ISSUING IS NOT LAUNCHING. It creates no resource and starts nothing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from stages.phase_d2 import d2_screening as SC  # noqa: E402
from stages.phase_d2 import d2_screening_authorization as A  # noqa: E402
from stages.phase_d2.autoinit_d2_screening_launch import auth_path_for  # noqa: E402


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grant", required=True, type=Path,
                    help="the maintainer's screening grant document")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rung", default="screening",
                    choices=("screening", "confirmation"))
    ap.add_argument("--porcelain", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.grant.is_file():
        raise SystemExit(f"no grant at {args.grant}")
    grant = json.loads(args.grant.read_text())
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit("refusing to issue against a dirty tree:\n" + dirty)
    commit = git("rev-parse", "HEAD")
    granted_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rel = auth_path_for(args.run_id)
    out = REPO_ROOT / rel
    if out.exists():
        raise SystemExit(f"{rel} already exists. An authorization is one-use: "
                         "issue into a new run id.")
    try:
        payload = A.build_payload(grant=grant, session_commit=commit,
                                  granted_utc=granted_utc, run_id=args.run_id,
                                  repo_root=REPO_ROOT, rung=args.rung)
    except A.D2ScreeningRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    contract = A.contract_module(args.rung)
    kind = (A.D2ConfirmationAuthorization if args.rung == "confirmation"
            else A.D2ScreeningAuthorization)
    reloaded = kind.load(out)
    reloaded.require_plan(SC.preregistration(REPO_ROOT)["preregistration_hash"])
    reloaded.require_run_id(args.run_id)
    reloaded.require_session_commit(commit)
    reloaded.require_harness(REPO_ROOT)
    reloaded.require_contract(contract.contract_hash(
        contract.session_contract(REPO_ROOT)))
    screening = args.rung == "screening"
    for claim, ok in (
            ("allows recovery training", reloaded.allows_recovery_training),
            ("authorizes its rung", reloaded.authorizes_d2_screening
             if screening else reloaded.authorizes_d2_confirmation),
            ("refuses the other rung", not (reloaded.authorizes_d2_confirmation
                                            if screening else
                                            reloaded.authorizes_d2_screening)),
            ("refuses a beam", not reloaded.allows_beam_search),
            ("refuses a fresh confirmation",
             not reloaded.authorizes_fresh_confirmation),
            ("refuses D3", not reloaded.authorizes_d3),
            ("refuses re-selection",
             not reloaded.authorizes_candidate_reselection),
            ("refuses B/q2 retraining",
             not reloaded.authorizes_reference_retraining),
            ("refuses promotion", not reloaded.authorizes_promotion),
            ("refuses automatic follow-on",
             not reloaded.automatic_followon_start)):
        if not ok:
            raise SystemExit(f"the issued artifact fails its own check: {claim}")
    quote = payload["money"]["live_quote"]
    summary = {
        "path": rel, "authorization_sha256": payload["authorization_sha256"],
        "session_commit": commit, "run_id": payload["run_id"],
        "granted_utc": payload["granted_utc"],
        "live_rate_usd_per_hour": quote["usd_per_hour"],
        "expected_usd": payload["expected_usd"],
        "hard_cap_usd": payload["hard_cap_usd"],
        "hard_runtime_minutes": payload["hard_runtime_minutes"],
        "envelope_remaining_usd": payload["money"]["envelope_remaining_usd"],
        "contract_hash": payload["contract_hash"],
        "harness_digest": payload["harness_source_digest"]}
    print(json.dumps(summary, indent=None if args.porcelain else 1))
    if not args.porcelain:
        print(f"authorizes ONE D2 {args.rung} session "
              f"({reloaded.n_probes} probes); NOT B/q2 retraining, NOT a "
              "promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
