#!/usr/bin/env python3
"""Write a D1 BEHAVIOURAL launch-bound readiness record. DERIVED. AUTHORIZES
NOTHING.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d1/write_d1_behavioural_launch_readiness.py \
        --run-id d1_behavioural_<utc> --session-commit <sha> --bundle <canonical>

Every field is READ from the thing that owns it -- the issued authorization,
the design, the live behavioural closure, the transport's name derivation, the
budget deriver -- and never restated. The launcher's `launch_readiness_gate`
compares this record against the live invocation and refuses a stale one at
`$0`. Like the bundle record, it stays UNCOMMITTED inside a launch window: the
session-commit gate's lineage check permits only the authorization artifact to
differ from the authorized base.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod",
              "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from stages.phase_d1 import behavioural as B  # noqa: E402
from stages.phase_d1 import behavioural_authorization as BA  # noqa: E402
from stages.phase_d1 import d1_authorization as A  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d1.behavioural_governance import (  # noqa: E402
    D1BehaviouralAuthorization,
)
from shared.run_layout import rel_run_dir  # noqa: E402

SCHEMA = "aadistill.phase_d1.behavioural_launch_readiness/v1"


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    return out.stdout.strip() if out.returncode == 0 else ""


def governance_path(run_id: str, name: str) -> str:
    return (f"{rel_run_dir(D1S.EXPERIMENT_ID, run_id, D1S.STAGE_ID)}"
            f"/governance/{name}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True,
                    help="must be the canonical name derived from the session "
                         "commit; an alias is refused here as well as at "
                         "launch")
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    commit = args.session_commit.strip()

    auth_rel = governance_path(args.run_id, "authorization.json")
    auth_path = REPO_ROOT / auth_rel
    if not auth_path.is_file():
        raise SystemExit(
            f"no authorization at {auth_rel}; issue it first with "
            "scripts/stages/stage-1/phase_d1/"
            "issue_d1_behavioural_authorization.py")
    auth = D1BehaviouralAuthorization.load(auth_path)
    auth.require_plan(B.design(REPO_ROOT)["design_hash"])
    auth.require_run_id(args.run_id)

    #: LINEAGE, not equality: the authorization is granted against the clean
    #: pre-authorization HEAD and the launch commit is that base plus the
    #: artifact itself. The launcher's own gate uses this relation; it is
    #: imported rather than restated.
    from aadistill.infrastructure.session_prechecks import (
        lineage_from_authorized_base,
    )

    lineage = lineage_from_authorized_base(
        REPO_ROOT, auth.authorized_session_commit, commit, auth_rel)
    if not lineage["ok"]:
        raise SystemExit(
            f"refusing to write readiness: {lineage['reason']}\n"
            "The session commit must be the authorized base plus this run's "
            "authorization and nothing else.")

    canonical = D1S.require_canonical_bundle(args.bundle, commit)
    closure = BA.behavioural_current_executable(REPO_ROOT)
    money = A.live_money(REPO_ROOT)
    design = B.design(REPO_ROOT)
    blockers = list(design["open_blockers"])
    cell = BA.session_cell(auth.rung, REPO_ROOT)

    bundle_rel = governance_path(args.run_id, "bundle.json")
    bundle_record = None
    if (REPO_ROOT / bundle_rel).is_file():
        bundle_record = json.loads((REPO_ROOT / bundle_rel).read_text())

    ceiling = float(auth.hard_cap_usd)
    raw_auth = json.loads(auth_path.read_text())
    record = {
        "schema": SCHEMA,
        "_contract": (
            "LAUNCH-BOUND. Every fact a behavioural launch consumes, derived "
            "at $0 immediately before it. It AUTHORIZES NOTHING: the one-use "
            "artifact does that, and this is what launch_readiness_gate "
            "checks the live invocation against."),
        "_generated_by": ("scripts/stages/stage-1/phase_d1/"
                          "write_d1_behavioural_launch_readiness.py"),
        "authorizes": "nothing",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "run_id": args.run_id,
        "stage_id": D1S.STAGE_ID,
        "session_commit": commit,
        "rung": auth.rung,
        "advancing_candidate": (auth.advancing_candidate
                                if auth.rung == "confirmation" else None),
        "design_hash": design["design_hash"],
        "open_blockers": blockers,
        "authorization": {
            "path": auth_rel,
            "authorization_sha256": raw_auth["authorization_sha256"],
            "granted_utc": auth.granted_utc,
            "hard_cap_usd": ceiling,
            "expected_usd": float(auth.expected_usd),
            "run_id": auth.run_id,
            "rung": auth.rung,
            "contract_hash": auth.contract_hash,
            "battery_role": auth.battery_role,
            "battery_content_id": auth.battery_content_id,
            "seeds": list(auth.seeds),
            "preregistration_sha256": raw_auth.get("preregistration_sha256"),
            "replay_plan_sha256": raw_auth.get("replay_plan_sha256"),
        },
        "harness": {
            "digest": closure["digest"],
            "n_files": closure["n_files"],
            "entry_points": list(closure["entry_points"]),
        },
        "lineage": lineage,
        "bundle": {
            "_derived": "from the session commit; an alias fails at $0",
            "name": canonical,
            "repo_path": D1S.canonical_bundle_path(commit),
            "relay_repo": D1S.transport().relay_repo,
            "local_record": bundle_rel if bundle_record else None,
            "local_sha256": (bundle_record or {}).get("sha256"),
            "staged": bundle_record is not None,
            "_verified_by": ("autoinit_d1_behavioural_launch."
                             "bundle_staged_gate, which downloads this object "
                             "and round-trips it"),
        },
        "derived_session": {
            "hard_ceiling_usd": ceiling,
            "hard_ceiling_minutes": float(auth.hard_runtime_minutes or 0.0),
            "accepted_cell": cell,
        },
        "live_price": (auth.money or {}).get("live_quote"),
        "four_conditions": (
            "ALL FOUR PASS" if not A.check_the_four_conditions(
                ceiling=ceiling, money=money)
            else A.check_the_four_conditions(ceiling=ceiling, money=money)),
        "balances_after_this_session": {
            "formal_remaining_usd": round(
                money["formal_remaining_usd"] - ceiling, 4),
            "package_remaining_usd": round(
                money["package_remaining_usd"] - ceiling, 4),
            "project_remaining_usd": round(
                money["project_remaining_usd"] - ceiling, 4),
        },
        "evidence_owed": {
            "what": ("every scored probe's result, per-sample rows and raw "
                     "generations, plus the session record with its contract, "
                     "materialization gates, ranking and selection"),
            "gate": ("ArtifactPolicy.products_secured requires each scored "
                     "probe's result and per-sample rows off-pod; the "
                     "generations additionally travel in the required final "
                     "archive"),
            "checkpoints": ("NOT owed (P8.4): a scored probe's weights have "
                            "no downstream consumer, and the arms "
                            "rematerialize deterministically from their "
                            "pinned paths"),
        },
        "no_resource_exists": True,
        "launch_invocation": {
            "_must_use": commit,
            "command": (
                "PYTHONPATH=src:scripts .venv/bin/python "
                "scripts/stages/stage-1/phase_d1/"
                "autoinit_d1_behavioural_launch.py "
                f"--scr <scr> --session-commit {commit} --bundle {canonical} "
                f"--run-id {args.run_id} --rung {auth.rung}"),
            "head_at_preparation": git("rev-parse", "HEAD"),
        },
    }

    out_rel = args.out or governance_path(args.run_id, "launch_readiness.json")
    out = REPO_ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    print(f"wrote {out_rel}  (UNCOMMITTED by design)")
    print(f"  run_id / rung    {record['run_id']} / {record['rung']}")
    print(f"  session commit   {commit}")
    print(f"  design           {record['design_hash'][:16]}")
    print(f"  closure          {closure['digest'][:16]} "
          f"({closure['n_files']} files)")
    print(f"  authorization    "
          f"{record['authorization']['authorization_sha256'][:16]} "
          f"granted {auth.granted_utc}")
    print(f"  contract         {auth.contract_hash[:16]}")
    print(f"  ceiling          ${ceiling:.4f}")
    print(f"  four conditions  {record['four_conditions']}")
    print(f"  bundle           {canonical} staged={record['bundle']['staged']}")
    print(f"  open blockers    {blockers or 'none'}")
    print("  AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
