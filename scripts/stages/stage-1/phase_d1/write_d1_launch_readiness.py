#!/usr/bin/env python3
"""Write D1's launch-bound readiness record. DERIVED, and AUTHORIZES NOTHING.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d1/write_d1_launch_readiness.py \
        --run-id d1_search_20261006_120000 \
        --session-commit <sha> --bundle aad_autoinit_<sha8>.bundle

**The previous readiness record had no generator at all.** It was written by
hand, which is why nothing could regenerate it, nothing checked it, and it
drifted without anybody noticing: it recorded a verified canonical bundle that
does not exist on the relay, and a closure digest for a tree the repairs then
superseded. A record that states facts no command derived is not evidence.

Every field here is READ from the thing that owns it -- the authorization
artifact, the design, the live closure, the transport's name derivation, the
budget ledger -- and never restated. The launcher's `launch_readiness_gate`
then compares this record against the live invocation and refuses a stale one at
`$0`.

It is an ADMISSION record, not a second authorization. The one-use artifact
authorizes; this pins what the launch was supposed to be so a reviewer (and the
gate) can tell whether the launch that happened is the one that was prepared.
Like the bundle record, it stays UNCOMMITTED inside a launch window, because the
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
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from stages.phase_d1 import d1_authorization as A  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from shared.run_layout import rel_run_dir  # noqa: E402

SCHEMA = "aadistill.phase_d1.launch_readiness/v1"


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    return out.stdout.strip() if out.returncode == 0 else ""


def governance_path(run_id: str, name: str) -> str:
    return (f"{rel_run_dir(D1S.EXPERIMENT_ID, run_id, D1S.STAGE_ID)}"
            f"/governance/{name}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True,
                    help="must be the canonical name derived from the session "
                         "commit; an alias is refused here as well as at launch")
    ap.add_argument("--arm", default=D1S.TREATMENT_ARM, choices=list(D1S.ARMS))
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    commit = args.session_commit.strip()

    #: THE AUTHORIZATION, through the loading type. A readiness record for a
    #: grant that will not load describes a launch that cannot happen.
    auth_rel = governance_path(args.run_id, "authorization.json")
    auth_path = REPO_ROOT / auth_rel
    if not auth_path.is_file():
        raise SystemExit(
            f"no authorization at {auth_rel}; issue it first with "
            "scripts/stages/stage-1/phase_d1/issue_d1_authorization.py")
    auth = A.D1Authorization.load(auth_path)
    auth.require_plan(D1S.design_hash(REPO_ROOT))
    auth.require_run_id(args.run_id)
    auth.require_treatment_arm()

    #: THE LAUNCH COMMIT IS CHECKED BY LINEAGE, NOT BY EQUALITY.
    #:
    #: `require_session_commit` asserts `commit == authorized_session_commit`,
    #: and that can only hold at ISSUANCE: the authorization is granted against
    #: the clean PRE-authorization HEAD, because the artifact cannot be
    #: committed before it exists. The commit a pod checks out is therefore
    #: always a later one -- the base plus the authorization itself -- and this
    #: function asked for equality and got the refusal it deserved.
    #:
    #: The right relation is the one the launcher's own gate uses, so this
    #: imports it rather than restating it: descends from the authorized base,
    #: and nothing but the authorization path differs.
    from aadistill.infrastructure.session_prechecks import (
        lineage_from_authorized_base,
    )

    lineage = lineage_from_authorized_base(
        REPO_ROOT, auth.authorized_session_commit, commit, auth_rel)
    if not lineage["ok"]:
        raise SystemExit(
            f"refusing to write readiness: {lineage['reason']}\n"
            "The session commit must be the authorized base plus this run's "
            "authorization and nothing else; session_commit_gate would refuse "
            "the launch at $0.")

    #: THE CANONICAL BUNDLE NAME, derived from the commit and compared with what
    #: the launch will pass. An alias fails here, at $0.
    canonical = D1S.require_canonical_bundle(args.bundle, commit)

    #: THE LIVE CLOSURE, and the closure AT THE BOUND COMMIT. Both, because they
    #: answer different questions: the first is what would execute now, and the
    #: second is what the pod will actually check out.
    closure = A.d1_current_executable(REPO_ROOT)
    at_commit = _closure_at_commit(commit, auth.harness_source_files)

    money = A.live_money(REPO_ROOT)
    design = D1S.design(REPO_ROOT)
    blockers = list(D1S.open_blockers(REPO_ROOT))

    #: THE BUNDLE RECORD, when preparation has run. Read, never asserted: the
    #: launcher's gate does the remote round trip, and this records what the
    #: local half said.
    bundle_rel = governance_path(args.run_id, "bundle.json")
    bundle_record = None
    if (REPO_ROOT / bundle_rel).is_file():
        bundle_record = json.loads((REPO_ROOT / bundle_rel).read_text())

    ceiling = float(auth.hard_cap_usd)
    record = {
        "schema": SCHEMA,
        "_contract": (
            "LAUNCH-BOUND. Every fact a launch consumes, derived at $0 "
            "immediately before it, with the identity it is bound to. It "
            "AUTHORIZES NOTHING: the one-use artifact does that, and this is "
            "what launch_readiness_gate checks the live invocation against."),
        "_generated_by": "scripts/stages/stage-1/phase_d1/write_d1_launch_readiness.py",
        "authorizes": "nothing",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "run_id": args.run_id,
        "stage_id": D1S.STAGE_ID,
        "session_commit": commit,
        "arm": args.arm,
        "design_hash": design["design_hash"],
        "open_blockers": blockers,
        "authorization": {
            "path": auth_rel,
            "authorization_sha256": json.loads(
                auth_path.read_text())["authorization_sha256"],
            "granted_utc": auth.granted_utc,
            "authorized_stages": list(auth.authorized_stages),
            "hard_cap_usd": ceiling,
            "expected_usd": float(auth.expected_usd),
            #: THE IDENTITIES STAGE A WILL ASSERT. Recorded so a reviewer can
            #: see that the grant and the session agree without rebuilding both.
            "run_id": auth.run_id,
            "config_hash": auth.config_hash,
            "measurement_protocol_id": auth.measurement_protocol_id,
            "suite_content_sha256": auth.suite_content_sha256,
        },
        "harness": {
            "digest": closure["digest"],
            "n_files": closure["n_files"],
            "entry_points": list(closure["entry_points"]),
            "digest_at_bound_commit": at_commit,
            "unchanged": at_commit == closure["digest"],
        },
        #: HOW THE LAUNCH COMMIT RELATES TO THE AUTHORIZED BASE. Recorded as the
        #: shared gate derives it, because equality is the wrong relation and
        #: asking for it is a refusal every correct launch would hit.
        "lineage": lineage,
        "bundle": {
            "_derived": "from the session commit; an alias fails at $0",
            "name": canonical,
            "repo_path": D1S.canonical_bundle_path(commit),
            "relay_repo": D1S.transport().relay_repo,
            "local_record": bundle_rel if bundle_record else None,
            "local_sha256": (bundle_record or {}).get("sha256"),
            "staged": bundle_record is not None,
            "_verified_by": ("autoinit_d1_launch.bundle_staged_gate, which "
                             "downloads this object and round-trips it. This "
                             "record states the name and the local digest; it "
                             "does NOT claim the remote object was verified."),
        },
        "derived_session": A.session_ceiling(REPO_ROOT),
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
        "staged_science_inputs": _science_inputs(),
        "products_owed": {
            "what": "the committed Top-2 checkpoints",
            "gate": ("ArtifactPolicy.products_secured requires BOTH, "
                     "identity-verified at the destination, read from the "
                     "authoritative stage1_selection.json in <scr>/store"),
            "why": ("the next D1 stage trains recovery probes FROM these "
                    "initializations and the screening price does not fund "
                    "rebuilding two complete structural paths"),
        },
        "evidence_layout": {
            "driver_out": f"artifacts/audit/{_audit_dirname()}",
            "_one_authoritative_layout": (
                "the driver's --out, the relay's evidence spec, the report "
                "fetch and both artifact specs all name this directory"),
        },
        "no_resource_exists": True,
        "launch_invocation": {
            "_must_use": commit,
            "command": _launch_command(args.run_id, commit, canonical, args.arm),
            "head_at_preparation": git("rev-parse", "HEAD"),
            "why": ("the authorization binds the commit whose closure it "
                    "measured, and session_commit_gate refuses any other. "
                    "Launching at the bound commit runs exactly the tree that "
                    "was verified."),
        },
    }

    out_rel = args.out or governance_path(args.run_id, "launch_readiness.json")
    out = REPO_ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    print(f"wrote {out_rel}  (UNCOMMITTED by design)")
    print(f"  run_id           {record['run_id']}")
    print(f"  session commit   {commit}")
    print(f"  design           {record['design_hash'][:16]}")
    print(f"  closure          {closure['digest'][:16]} "
          f"({closure['n_files']} files) · at bound commit "
          f"{str(at_commit)[:16]} · unchanged "
          f"{record['harness']['unchanged']}")
    print(f"  authorization    {record['authorization']['authorization_sha256'][:16]} "
          f"granted {auth.granted_utc}")
    print(f"  config_hash      {auth.config_hash[:16]}")
    print(f"  ceiling          ${ceiling:.4f}")
    print(f"  four conditions  {record['four_conditions']}")
    print(f"  bundle           {canonical} staged={record['bundle']['staged']}")
    print(f"  open blockers    {blockers or 'none'}")
    print("  AUTHORIZES NOTHING.")
    return 0


def _audit_dirname() -> str:
    from stages.phase_d1 import autoinit_d1_launch as L

    return L.AUDIT_DIRNAME


def _launch_command(run_id: str, commit: str, bundle: str, arm: str) -> str:
    return ("PYTHONPATH=src:scripts .venv/bin/python "
            "scripts/stages/stage-1/phase_d1/autoinit_d1_launch.py "
            f"--scr <scr> --session-commit {commit} --bundle {bundle} "
            f"--run-id {run_id} --arm {arm}")


def _closure_at_commit(commit: str, files) -> str | None:
    """The authorized file set's digest AT the bound commit, from git blobs.

    Derived with the shared formula rather than a local one; a second digest
    rule is two rules as soon as either is edited.
    """
    from aadistill.governance.closure import digest_of
    import hashlib

    entries = []
    for rel in sorted(files):
        blob = subprocess.run(["git", "show", f"{commit}:{rel}"],
                              capture_output=True, cwd=REPO_ROOT)
        if blob.returncode != 0:
            return None
        entries.append({"path": rel,
                        "sha256": hashlib.sha256(blob.stdout).hexdigest()})
    return digest_of(entries) if entries else None


def _science_inputs() -> dict:
    """What the launcher will scp, and the pin each one is verified against."""
    from stages.phase_d1 import autoinit_d1_launch as L

    out = {"local_assets": [a.as_env_entry() for a in L.SCIENCE_ASSETS],
           "teacher_revision": D1S.root_teacher_identity(REPO_ROOT)["revision"]}
    try:
        out["state_eval"] = D1S.verify_state_eval_bytes(REPO_ROOT)
    except Exception as exc:                                      # noqa: BLE001
        out["state_eval_error"] = str(exc)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
