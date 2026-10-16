#!/usr/bin/env python3
"""Issue the new D3's ONE-USE authorization (structural or probes), from a
logged command.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/issue_d3_authorization.py \
        --rung structural --grant logs/budget/approvals/autoinit_d3_structural_grant.json \
        --run-id d3_structural_<utc>

Everything of substance is `d3_authorization.build_payload`'s: it refuses an
unapproved registration, a grant without a cumulative envelope, a phase outside
the formal book, an envelope earlier subruns have narrowed below the session's
price plus its teardown reserve, and a ceiling the four money conditions do not
fund; it quotes the live rate and derives the bound contract on this tree. This
command reads the grant, refuses a dirty tree, takes HEAD as the session
commit, writes the artifact ONCE and reads it back through the type the pod
loads.

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

from stages.phase_d3 import d3_authorization as A  # noqa: E402
from stages.phase_d3 import d3_design as D  # noqa: E402
from shared.run_layout import rel_run_dir  # noqa: E402


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def auth_path_for(run_id: str) -> str:
    return (f"{rel_run_dir(D.EXPERIMENT_ID, run_id, D.STAGE_ID)}/governance/"
            "authorization.json")


def bound_contract(rung: str) -> str:
    if rung == "structural":
        return D.registered_treatment_spec(REPO_ROOT).spec_hash
    from stages.phase_d3 import d3_probes as P

    return P.contract_hash(P.session_contract(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grant", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rung", required=True, choices=tuple(A.RUNG_GOVERNANCE))
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
                                  rung=args.rung, repo_root=REPO_ROOT)
    except A.D3AuthorizationRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    reloaded = A.TYPES[args.rung].load(out)
    reloaded.require_plan(D.registration_hash(REPO_ROOT))
    reloaded.require_run_id(args.run_id)
    reloaded.require_session_commit(commit)
    reloaded.require_harness(REPO_ROOT)
    reloaded.require_contract(bound_contract(args.rung))
    structural = args.rung == "structural"
    for claim, ok in (
            ("authorizes its rung", reloaded.authorizes_d3_structural
             if structural else reloaded.authorizes_d3_probes),
            ("refuses the other rung", not (reloaded.authorizes_d3_probes
                                            if structural else
                                            reloaded.authorizes_d3_structural)),
            ("trains only on the probes rung",
             (not reloaded.authorizes_recovery_training) if structural
             else reloaded.allows_recovery_training),
            ("refuses a beam", not reloaded.authorizes_beam_search),
            ("refuses reopening D1/D2", not reloaded.authorizes_reopen_d1_d2),
            ("refuses the cancelled D3", not reloaded.authorizes_cancelled_d3),
            ("refuses q2/B retraining",
             not reloaded.authorizes_reference_retraining),
            ("refuses the screening seeds",
             not reloaded.authorizes_screening_seeds),
            ("refuses a fresh confirmation",
             not reloaded.authorizes_fresh_confirmation),
            ("refuses promotion", not reloaded.authorizes_promotion),
            ("refuses automatic follow-on",
             not reloaded.authorizes_automatic_followon_start)):
        if not ok:
            raise SystemExit(f"the issued artifact fails its own check: {claim}")
    quote = payload["money"]["live_quote"]
    print(json.dumps({
        "path": rel, "authorization_sha256": payload["authorization_sha256"],
        "session_commit": commit, "run_id": payload["run_id"],
        "granted_utc": payload["granted_utc"],
        "live_rate_usd_per_hour": quote["usd_per_hour"],
        "expected_usd": payload["expected_usd"],
        "hard_cap_usd": payload["hard_cap_usd"],
        "hard_runtime_minutes": payload["hard_runtime_minutes"],
        "envelope_remaining_usd": payload["money"]["envelope_remaining_usd"],
        "contract_hash": payload["contract_hash"],
        "harness_digest": payload["harness_source_digest"]},
        indent=None if args.porcelain else 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
