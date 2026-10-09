#!/usr/bin/env python3
"""Issue a D1 behavioural ONE-USE authorization. Reproducibly, from a command.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d1/issue_d1_behavioural_authorization.py \
        --grant logs/budget/approvals/autoinit_d1_behavioural_screening_grant.json \
        --run-id d1_behavioural_<utc> --rung screening

Everything of substance is `behavioural_authorization.build_payload`'s, not
this file's. This is the command around it: read the maintainer's grant, refuse
a dirty tree and a future-dated grant, take HEAD as the session commit, let
`build_payload` quote the live rate and derive the ceiling against the four
budget conditions, write the artifact once, and read it back through the TYPE
the pod's dispatch branch will load.

A CONFIRMATION issuance additionally requires `--selection-record` -- the
secured screening evidence -- and `--advancing-candidate` must equal the arm
that record's mechanical selection advanced. The candidate is screening's
output; it is read, never chosen.

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
for extra in ("src", "scripts", "scripts/autoinit", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.governance.grant import refuse_a_future_dated_grant  # noqa: E402

from stages.phase_d1 import behavioural as B  # noqa: E402
from stages.phase_d1 import behavioural_authorization as BA  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d1.behavioural_governance import (  # noqa: E402
    D1BehaviouralAuthorization,
)
from shared.run_layout import rel_run_dir  # noqa: E402


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def authorization_path_for(run_id: str) -> str:
    return (f"{rel_run_dir(D1S.EXPERIMENT_ID, run_id, D1S.STAGE_ID)}"
            "/governance/authorization.json")


def advanced_arm_from(selection_record: Path) -> str:
    """The arm screening's mechanical selection advanced. READ AND RECOMPUTED.

    Takes the SECURED SCREENING SESSION EVIDENCE (`d1_behavioural.json`) and
    trusts none of its conclusions: an arbitrary JSON file containing
    `advanced: true` and an arm label is not evidence of a selection. Four
    things are required, each from the document's own data:

    * it is a COMPLETED screening session record, by schema, rung and status;
    * its contract hash is the one THIS tree derives for the screening rung,
      so the field it measured is the frozen field;
    * the mechanical selection RECOMPUTES from its per-probe scored rows,
      through the same frozen `rank_screening` + `advance_one` the pod ran;
    * the recomputation AGREES with the recorded selection -- arm and outcome
      both -- and advanced somebody.

    A disagreement anywhere is refused, never resolved: the rule is frozen,
    so two answers means the record is not what it claims to be.
    """
    from aadistill.infrastructure.manifest import sha256_json

    doc = json.loads(selection_record.read_text())
    if doc.get("schema") != "aadistill.phase_d1.behavioural_session/v1":
        raise SystemExit(
            f"{selection_record} declares schema {doc.get('schema')!r}; the "
            "selection is read only from the secured screening session "
            "evidence, never from a document that merely contains a verdict")
    if doc.get("rung") != "screening":
        raise SystemExit(
            f"{selection_record} is a {doc.get('rung')!r} record; only the "
            "screening rung produces an advancing candidate")
    if doc.get("status") != "COMPLETE":
        raise SystemExit(
            f"{selection_record} records status {doc.get('status')!r}; an "
            "incomplete screening selected nothing anyone may confirm")
    want_contract = sha256_json(B.session_contract("screening", REPO_ROOT))
    if doc.get("contract_hash") != want_contract:
        raise SystemExit(
            f"{selection_record} ran under contract "
            f"{str(doc.get('contract_hash'))[:16]} and this tree derives "
            f"{want_contract[:16]} for the screening rung; the field it "
            "measured is not the frozen field this confirmation would extend")

    rows = [{"arm": p["arm"], "seed": p["seed"],
             "correct_overall": p["correct_overall"],
             "usable_rollout_rate": p["usable_rollout_rate"]}
            for p in doc.get("probes", []) if p.get("scored")]
    recomputed = B.advance_one(B.rank_screening(rows, REPO_ROOT))
    recorded = doc.get("selection") or {}
    if not recomputed.get("advanced"):
        raise SystemExit(
            f"{selection_record}: the frozen rule recomputes "
            f"{recomputed.get('outcome')!r} over the record's own scored "
            "rows; a confirmation cannot be issued for a screening that "
            "advanced nobody")
    if (recorded.get("arm") != recomputed.get("arm")
            or bool(recorded.get("advanced")) != bool(
                recomputed.get("advanced"))):
        raise SystemExit(
            f"{selection_record} records selection "
            f"{recorded.get('arm')!r}/{recorded.get('outcome')!r} and the "
            f"frozen rule recomputes {recomputed.get('arm')!r}/"
            f"{recomputed.get('outcome')!r} from the same rows. Two answers "
            "from one frozen rule means the record is not what it claims to "
            "be; refused, not resolved.")
    return str(recomputed["arm"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grant", required=True, type=Path,
                    help="the maintainer's grant document")
    ap.add_argument("--run-id", required=True,
                    help="the run this authorization is issued for; one-use")
    ap.add_argument("--rung", required=True,
                    choices=("screening", "confirmation"))
    ap.add_argument("--advancing-candidate", default=None,
                    help="confirmation only; must equal the arm the "
                         "--selection-record advanced")
    ap.add_argument("--selection-record", type=Path, default=None,
                    help="confirmation only: the secured screening evidence "
                         "whose mechanical selection names the candidate")
    ap.add_argument("--gpu", default="NVIDIA L40S",
                    help="the card whose live securePrice is quoted")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="issue against a dirty tree. Almost never right.")
    ap.add_argument("--porcelain", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.grant.is_file():
        raise SystemExit(f"no grant at {args.grant}")
    grant = json.loads(args.grant.read_text())

    #: A hand-written grant's date field says `utc`; hold it to that.
    refuse_a_future_dated_grant(str(grant.get("granted_utc", "")))

    advancing = args.advancing_candidate or None
    if args.rung == "confirmation":
        if args.selection_record is None or not args.selection_record.is_file():
            raise SystemExit(
                "a confirmation issuance requires --selection-record: the "
                "advancing candidate is screening's output and is read from "
                "its secured evidence, never typed")
        advanced = advanced_arm_from(args.selection_record)
        if advancing and advancing != advanced:
            raise SystemExit(
                f"--advancing-candidate says {advancing!r} and the screening "
                f"selection record advanced {advanced!r}; the record wins and "
                "the disagreement is refused rather than resolved")
        advancing = advanced
    elif args.selection_record is not None or advancing:
        raise SystemExit(
            "a screening issuance takes no advancing candidate and no "
            "selection record; screening is what produces them")

    if not args.allow_dirty:
        dirty = git("status", "--porcelain")
        if dirty:
            raise SystemExit(
                "refusing to issue against a dirty tree:\n" + dirty
                + "\nThe authorized session commit must describe exactly what "
                  "the pod checks out, and the harness digest is derived from "
                  "the working tree.")

    commit = git("rev-parse", "HEAD")
    granted_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rel = authorization_path_for(args.run_id)
    out = REPO_ROOT / rel
    #: ONE-USE MEANS ONE ARTIFACT. Checked BEFORE the live quote, so a refusal
    #: here does not leave a consumed grant looking fresh.
    if out.exists():
        raise SystemExit(
            f"{rel} already exists. An authorization is one-use: issue into a "
            "new run id, or move the existing artifact aside deliberately.")

    try:
        payload = BA.build_payload(
            grant=grant, session_commit=commit, granted_utc=granted_utc,
            run_id=args.run_id, rung=args.rung,
            advancing_candidate=advancing, repo_root=REPO_ROOT)
    except BA.D1BehaviouralIssuanceRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    #: READ BACK THROUGH THE LOADING TYPE. An artifact the pod's dispatch
    #: branch would refuse is not an authorization, and discovering that here
    #: is free.
    reloaded = D1BehaviouralAuthorization.load(out)
    reloaded.require_plan(B.design(REPO_ROOT)["design_hash"])
    reloaded.require_rung(args.rung)
    reloaded.require_run_id(args.run_id)
    if reloaded.authorized_session_commit != commit:
        raise SystemExit("the issued artifact binds a different session commit")
    reloaded.require_harness(REPO_ROOT)
    if args.rung == "confirmation":
        if reloaded.require_advancing_candidate() != advancing:
            raise SystemExit(
                "the issued artifact names a different advancing candidate")
    for claim, ok in (
            ("allows recovery training", reloaded.allows_recovery_training),
            ("authorizes behavioural selection",
             reloaded.authorizes_behavioural_selection),
            ("refuses a beam", not reloaded.allows_beam_search),
            ("refuses a search", not reloaded.authorizes_d1_search),
            ("refuses reselection",
             not reloaded.authorizes_candidate_reselection),
            ("refuses incumbent remeasurement",
             not reloaded.authorizes_incumbent_remeasurement),
            ("refuses automatic follow-on",
             not reloaded.automatic_followon_start)):
        if not ok:
            raise SystemExit(f"the issued artifact fails its own check: {claim}")

    #: AND THE CONTRACT THE DRIVER WILL ASSERT, recomputed from this tree and
    #: required equal -- closing the loop between the issuer and the pod
    #: rather than trusting that both call the same helper.
    from aadistill.infrastructure.manifest import sha256_json

    contract = B.session_contract(args.rung, REPO_ROOT,
                                  advancing_candidate=advancing)
    if sha256_json(contract) != reloaded.contract_hash:
        raise SystemExit(
            "the issued artifact binds a contract hash this tree does not "
            "reproduce; the driver would refuse it on a billing pod")

    quote = payload["money"]["live_quote"]
    if args.porcelain:
        print(json.dumps({
            "path": rel,
            "authorization_sha256": payload["authorization_sha256"],
            "session_commit": commit, "run_id": payload["run_id"],
            "rung": payload["rung"],
            "granted_utc": payload["granted_utc"],
            "live_rate_usd_per_hour": quote["usd_per_hour"],
            "hard_cap_usd": payload["hard_cap_usd"],
            "contract_hash": payload["contract_hash"],
            "preregistration_sha256": payload["preregistration_sha256"],
            "harness_digest": payload["harness_source_digest"]}))
        return 0
    print(f"wrote {rel}")
    print(f"  authorization_sha  {payload['authorization_sha256']}")
    print(f"  granted_utc        {payload['granted_utc']}  (real clock)")
    print(f"  run_id / rung      {payload['run_id']} / {payload['rung']}")
    if advancing:
        print(f"  advancing          {advancing}  (read from the selection "
              "record)")
    print(f"  session commit     {commit}")
    print(f"  design / plan      {payload['design_hash'][:16]}")
    print(f"  contract           {payload['contract_hash'][:16]}")
    print(f"  preregistration    {payload['preregistration_sha256'][:16]}")
    print(f"  battery            {payload['battery_role']} "
          f"({payload['battery_content_id'][:16]})")
    print(f"  seeds              {payload['seeds']}")
    print(f"  harness            {payload['harness_source_digest'][:16]} "
          f"({payload['harness']['n_files']} files, derived)")
    print(f"  live rate          ${quote['usd_per_hour']}/h ({quote['_quoted']})")
    print(f"  ceiling            ${payload['hard_cap_usd']:.4f} "
          f"(expected ${payload['expected_usd']:.4f} over "
          f"{payload['hard_runtime_minutes']:.2f} ceiling minutes)")
    print(f"  four conditions    {payload['money']['four_conditions']}")
    print(f"  authorizes ONE {args.rung} session; NOT a beam, NOT a "
          "re-selection, NOT a promotion, NOT the other rung.")
    print("  ISSUING IS NOT LAUNCHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
