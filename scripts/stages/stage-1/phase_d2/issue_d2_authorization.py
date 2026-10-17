#!/usr/bin/env python3
"""Issue D2's ONE-USE formal search authorization, from a logged command.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d2/issue_d2_authorization.py \
        --grant logs/budget/approvals/autoinit_d2_grant.json \
        --run-id d2_search_<utc> --prediction-rows scored_v1

D1's issuer, for D2's type. Everything of substance is
`d2_authorization.build_payload`'s: it refuses an unapproved preregistration
revision, a row selection without its own adopted evidence, a phase outside
the formal book, and a ceiling the four money conditions do not fund; it quotes
the live rate and derives the ceiling from the price of the DECLARED row
selection. This is the command around it: read the maintainer's grant, refuse
a dirty tree, take HEAD as the session commit, write the artifact once, and
read it back through the TYPE the pod loads.

ISSUING IS NOT LAUNCHING. It creates no resource and starts nothing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from stages.d_series.programme_closeout import refuse_closed_authority  # noqa: E402
from stages.phase_d2 import d2_authorization as A  # noqa: E402
from stages.phase_d2 import d2_session as S  # noqa: E402
from stages.phase_d2.autoinit_d2_launch import auth_path_for  # noqa: E402


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
                    help="the maintainer's grant document")
    #: `run_id` is a field of the bound config_hash: an authorization issued
    #: for one run id cannot authorize a launch under another.
    ap.add_argument("--run-id", required=True)
    #: The row selection is part of the bound identity AND of the price.
    ap.add_argument("--prediction-rows", required=True,
                    choices=list(S.PREDICTION_ROWS))
    ap.add_argument("--porcelain", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.grant.is_file():
        raise SystemExit(f"no grant at {args.grant}")
    #: The D-series is CLOSED (maintainer decision 2026-10-17): a closed
    #: authority issues nothing, whatever else about it would still pass.
    refuse_closed_authority(args.grant, repo_root=REPO_ROOT)
    grant = json.loads(args.grant.read_text())

    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit(
            "refusing to issue against a dirty tree:\n" + dirty
            + "\nThe authorized session commit must describe exactly what the "
              "pod checks out, and the harness digest is derived from the "
              "working tree.")
    commit = git("rev-parse", "HEAD")
    #: THE REAL UTC at issuance: the live rate inside `build_payload` is
    #: quoted now, and `granted_utc` dates it.
    granted_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rel = auth_path_for(args.run_id)
    out = REPO_ROOT / rel
    #: ONE-USE MEANS ONE ARTIFACT, checked before the live quote.
    if out.exists():
        raise SystemExit(f"{rel} already exists. An authorization is one-use: "
                         "issue into a new run id.")
    try:
        payload = A.build_payload(
            grant=grant, session_commit=commit, granted_utc=granted_utc,
            run_id=args.run_id, prediction_rows=args.prediction_rows,
            repo_root=REPO_ROOT)
    except A.D2AuthorizationRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    #: READ BACK THROUGH THE LOADING TYPE, and the identity the driver asserts.
    reloaded = A.D2Authorization.load(out)
    reloaded.require_plan(S.preregistration_hash(REPO_ROOT))
    reloaded.require_run_id(args.run_id)
    reloaded.require_session_commit(commit)
    reloaded.require_harness(REPO_ROOT)
    for claim, ok in (
            ("authorizes_d2_search", reloaded.authorizes_d2_search),
            ("allows_beam_search", reloaded.allows_beam_search),
            ("refuses recovery", not reloaded.allows_recovery),
            ("refuses behavioural", not reloaded.allows_behavioural),
            ("refuses automatic follow-on",
             not reloaded.automatic_followon_start),
            ("binds the declared rows",
             reloaded.prediction_rows == args.prediction_rows),
            ("prices the declared rows",
             reloaded.pricing.get("prediction_rows") == args.prediction_rows)):
        if not ok:
            raise SystemExit(f"the issued artifact fails its own check: {claim}")
    with tempfile.TemporaryDirectory(prefix="d2-issue-") as tmp:
        session = S.build_session(workdir=Path(tmp), run_id=args.run_id,
                                  device=A.FORMAL_DEVICE,
                                  prediction_rows=args.prediction_rows,
                                  repo_root=REPO_ROOT)
        contract = S.assert_session_contract(session, REPO_ROOT)
    for field in ("config_hash", "measurement_protocol_id", "objective_hash",
                  "ranking_policy_hash"):
        if payload[field] != contract[field]:
            raise SystemExit(
                f"the issued artifact binds {field}={payload[field][:16]} and "
                f"the session this tree builds carries {contract[field][:16]}; "
                "a launch would be refused in stage A, on a billing pod")

    quote = payload["money"]["live_quote"]
    summary = {
        "path": rel, "authorization_sha256": payload["authorization_sha256"],
        "session_commit": commit, "run_id": payload["run_id"],
        "granted_utc": payload["granted_utc"],
        "prediction_rows": payload["prediction_rows"],
        "live_rate_usd_per_hour": quote["usd_per_hour"],
        "expected_usd": payload["expected_usd"],
        "hard_cap_usd": payload["hard_cap_usd"],
        "hard_ceiling_minutes": payload["pricing"]["hard_ceiling_minutes"],
        "pricing_in_force": payload["pricing"]["pricing_in_force"],
        "config_hash": payload["config_hash"],
        "harness_digest": payload["harness_source_digest"]}
    print(json.dumps(summary, indent=None if args.porcelain else 1))
    if not args.porcelain:
        top_k = int(S.preregistration(REPO_ROOT)["retention"]["k"])
        print(f"authorizes ONE D2 formal search and a committed Top-{top_k}; "
              "NOT recovery, NOT behavioural work, NOT a promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
