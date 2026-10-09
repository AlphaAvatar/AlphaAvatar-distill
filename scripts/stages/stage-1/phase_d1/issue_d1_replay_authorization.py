#!/usr/bin/env python3
"""Issue the one-use authorization for D1's finalist rematerialization.

    PYTHONPATH=src:scripts python \
        scripts/stages/stage-1/phase_d1/issue_d1_replay_authorization.py --rate 1.09

Run it AFTER the launch-bound readiness sweep is recorded and committed, and
BEFORE the bundle is staged: the bundle must carry the authorization, so the
authorization must exist first, and the readiness record and the authorization
are therefore separate commits.

THE CEILING IS THE CAMPAIGN'S REMAINDER, NOT ITS CEILING. The first four
attempts were issued at the full $3.50 each because the artifact was written by
hand with the task ceiling in it. That is the defect this script exists to
close: AGENTS.md P12.1 says the budget is cumulative across every resource and
subrun and a rerun does not reset it, so a fifth attempt may be authorized for

    ceiling - settled spend - teardown reserve

read from the campaign ledger, which owns every draw. Four attempts at a
hand-copied $3.50 would have been $14 against a $3.50 campaign.

It binds four things the launcher re-derives and refuses on disagreement: the
session commit, the executable digest over the declared harness files, the
preflight plan hash, and the money.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.governance.authorization import (  # noqa: E402
    harness_source_digest,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

AUTH_REL = "logs/budget/approvals/autoinit_d1_replay_authorization.json"
CAMPAIGN_REL = ("logs/stages/stage-1/phase_d1/validations/"
                "finalist-rematerialization/v1/campaign.json")

#: Every file whose bytes decide what this session runs or what it is
#: authorized to do. Hand-declared rather than import-walked, because the
#: replay's closure is four modules and two configs and a walk would add a
#: dependency on the walker for no coverage this cannot state.
HARNESS_FILES: tuple[str, ...] = (
    "scripts/pod/autoinit_d1_replay_launch.py",
    "scripts/pod/autoinit_d1_replay_driver.py",
    "scripts/experiments/stage-1/phase_d1/replay_specs.py",
    "scripts/experiments/stage-1/phase_d1/d1_session.py",
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/collect_artifacts.py",
    "configs/stages/stage-1/phase_d1/d1_replay_artifacts.json",
    "configs/stages/stage-1/phase_d1/d1_replay_artifacts_failed.json",
    "logs/stages/stage-1/phase_d1/decisions/post_search_finalist_retention.json",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=REPO_ROOT).stdout.strip()


def campaign_money(repo_root: Path) -> dict[str, float]:
    """What the ledger says is left. `booked_usd` is RECOMPUTED from the
    subruns rather than trusted, because a roll-up total that drifts from its
    components is a silent overspend."""
    doc = json.loads((repo_root / CAMPAIGN_REL).read_text())
    subruns = doc.get("subruns") or []
    settled = round(sum(float(row["cost_usd"]) for row in subruns), 4)
    stated = round(float(doc["booked_usd"]), 4)
    if abs(settled - stated) > 1e-6:
        raise SystemExit(
            f"the campaign states booked_usd=${stated} and its {len(subruns)} "
            f"subruns sum to ${settled}. One of them is stale; authorizing "
            "against either without resolving that would spend money nobody "
            "has accounted for.")
    ceiling = float(doc["ceiling_usd"])
    reserve = float(doc.get("teardown_reserve_usd", 0.0))
    from_campaign = round(ceiling - settled - reserve, 4)

    #: AND THE ALLOWANCE THAT ACTUALLY FUNDS IT. A campaign ceiling is a bound
    #: this task accepted; it is not money. The 2026-10-07 amendment raised
    #: this ceiling to $10.0000 while the GPU engineering allowance it is
    #: charged to held $3.4959, so the ceiling now EXCEEDS its own funding
    #: source -- and a cap derived from the ceiling alone would authorize more
    #: than the book can pay. Taking the minimum is the whole point: whichever
    #: bound binds, binds.
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from maintenance.consolidation.derive_budget import derive

    allowance = derive(REPO_ROOT)["engineering"]
    from_allowance = round(float(allowance["remaining_usd"]) - reserve, 4)
    available = min(from_campaign, from_allowance)
    return {"ceiling_usd": ceiling, "settled_usd": settled,
            "teardown_reserve_usd": reserve,
            "available_from_campaign_usd": from_campaign,
            "allowance_remaining_usd": round(float(allowance["remaining_usd"]), 4),
            "available_from_allowance_usd": from_allowance,
            "available_usd": available,
            "bound_by": ("campaign ceiling" if from_campaign <= from_allowance
                         else "GPU engineering allowance"),
            "n_prior_subruns": len(subruns)}


def priced_session(rate: float) -> dict[str, float]:
    """The session's own cost model, priced at the live rate.

    Taken from the launcher's `BudgetSpec` through the real planner, so the
    number this artifact caps and the number the launcher computes come from
    one place. A hand-typed expectation here is how a cap and a plan disagree.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))
    from stages.phase_d1 import autoinit_d1_replay_launch as L

    args = L.build_parser().parse_args([
        "--scr", "/tmp/d1-replay-pricing",
        "--session-commit", "0" * 40,
        "--bundle", "pricing.bundle",
        "--run-id", "pricing",
    ])
    #: `authorized_usd` is deliberately the CAMPAIGN REMAINDER and not the
    #: campaign ceiling: `plan_session` raises when the hard threshold exceeds
    #: what it was given, so pricing against the remainder makes an unfittable
    #: session fail here, at $0, rather than in the launcher after the sweep.
    plan = L.spec(args).budget.plan(
        price_per_hour=rate,
        authorized_usd=campaign_money(REPO_ROOT)["available_usd"])
    return {
        "expected_minutes": round(plan.expected_minutes, 2),
        "soft_minutes": round(plan.soft_stop_minutes, 2),
        "hard_minutes": round(plan.hard_terminate_minutes, 2),
        "expected_usd": round(plan.expected_usd, 4),
        "hard_usd": round(plan.hard_terminate_usd, 4),
    }


def build_record(*, rate: float, commit: str, money: dict[str, float],
                 priced: dict[str, float], version: int,
                 granted_by: str) -> dict:
    #: THE DIGEST COMES FROM THE FUNCTION THAT VERIFIES IT, not from a formula
    #: written here. The first version of this issuer computed its own --
    #: `sha256_json({rel: sha256_json({"bytes": hex})})` -- which is a perfectly
    #: deterministic hash of the same bytes and is NOT the hash
    #: `SpendAuthorization.require_harness` computes. So the runner refused
    #: before the dry run had reached a single gate:
    #:
    #:     the harness on disk digests to 246c846cddac… but this authorization
    #:     was granted against e436d89ea23d…
    #:
    #: Caught at $0 by the gate whose entire job this is, which is the right
    #: place -- but it is the "import the contract, do not describe it" failure
    #: exactly: a hand-written restatement of a hash agrees with itself and
    #: with nothing else.
    harness = harness_source_digest(REPO_ROOT, files=HARNESS_FILES)
    files = [entry["path"] for entry in harness["files"]]
    digest = harness["digest"]

    from aadistill.initialization.planning.recovery import (
        PreflightPlan, PreflightStage,
    )
    sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))
    from stages.phase_d1 import autoinit_d1_replay_launch as L

    args = L.build_parser().parse_args([
        "--scr", "/tmp/d1-replay-pricing", "--session-commit", "0" * 40,
        "--bundle", "pricing.bundle", "--run-id", "pricing"])
    spec = L.spec(args)
    assert isinstance(spec.plan_id, str) and spec.plan_hash
    assert PreflightPlan and PreflightStage      # imported for the contract

    record = {
        "schema": "aadistill.autoinit.spend_authorization/v1",
        "authorization_id": "autoinit.v1.d1_replay",
        "version": version,
        "granted_utc": datetime.now(timezone.utc).isoformat(),
        "granted_by": granted_by,
        "plan_id": spec.plan_id,
        "preflight_plan_hash": spec.plan_hash,
        "expected_usd": priced["expected_usd"],
        "hard_cap_usd": money["available_usd"],
        "authorized_stages": [0],
        "stage_conditions": {
            "0": ("all eight operator steps are pinned to the digest the "
                  "search recorded; the ROOT STATE is derived from the "
                  "mechanism and resolved against the recorded step-0 "
                  "configs before any weights load; a mismatch after that "
                  "derivation holds is terminal for that leaf and is NOT "
                  "retried"),
        },
        "scope_note": (
            "ONE pod, two digest-pinned fixed paths, each leaf fetched and "
            "re-identified off-pod the moment it reconstructs. Decides "
            "nothing, measures nothing, trains nothing. Does NOT authorize a "
            "beam search, recovery training, behavioural screening or "
            "confirmation."),
        "phase_a_authorized": False,
        "automatic_phase_a_start": False,
        "authorized_session_commit": commit,
        "harness_source_digest": digest,
        "harness_source_files": files,
        "per_launch_hard_usd": money["available_usd"],
        "provenance_commit": commit,
        "rate_usd_per_hour": rate,
        "one_use": True,
        "max_provider_resources": 1,
        "money": {
            **money,
            "priced": priced,
            "_why_the_cap_is_not_the_ceiling": (
                f"{money['n_prior_subruns']} paid subruns have settled "
                f"${money['settled_usd']} of the ${money['ceiling_usd']} "
                "campaign ceiling. P12.1: the budget is cumulative across "
                "every resource and subrun and a rerun does not reset it, so "
                "this attempt is capped at ceiling - settled - teardown "
                f"reserve = ${money['available_usd']}."),
            "_the_priced_hard_fits": (
                f"${priced['hard_usd']} hard against ${money['available_usd']} "
                "available"),
        },
        "campaign": CAMPAIGN_REL,
        "enforcement": (
            "the launcher loads this artifact and refuses to create a pod "
            "whose priced hard threshold exceeds hard_cap_usd, refuses a "
            "stage not in authorized_stages, verifies the provider account "
            "balance before creating anything, and has no code path to "
            "Phase A"),
    }
    if priced["hard_usd"] > money["available_usd"] + 1e-9:
        raise SystemExit(
            f"the session prices at ${priced['hard_usd']} hard "
            f"({priced['hard_minutes']} min at ${rate}/h) and only "
            f"${money['available_usd']} remains in the campaign. Re-price the "
            "session or return for a decision -- do NOT issue a cap the plan "
            "cannot fit under.")
    record["authorization_sha256"] = sha256_json(record)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, required=True,
                    help="the live securePrice this artifact is issued at")
    ap.add_argument("--version", type=int, default=None,
                    help="defaults to the existing artifact's version + 1")
    ap.add_argument("--granted-by", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the money and the pricing, write nothing")
    args = ap.parse_args(argv)

    money = campaign_money(REPO_ROOT)
    priced = priced_session(args.rate)
    print(f"campaign  ceiling ${money['ceiling_usd']:.4f}  settled "
          f"${money['settled_usd']:.4f}  reserve "
          f"${money['teardown_reserve_usd']:.4f}  ->  "
          f"${money['available_from_campaign_usd']:.4f}")
    print(f"allowance remaining ${money['allowance_remaining_usd']:.4f}  ->  "
          f"${money['available_from_allowance_usd']:.4f}")
    print(f"available ${money['available_usd']:.4f}  "
          f"(bound by the {money['bound_by']})")
    print(f"session   expected {priced['expected_minutes']:.1f} min "
          f"${priced['expected_usd']:.4f}  ·  soft "
          f"{priced['soft_minutes']:.1f} min  ·  hard "
          f"{priced['hard_minutes']:.1f} min ${priced['hard_usd']:.4f}")
    if args.dry_run:
        return 0

    if git("status", "--porcelain") and not args.allow_dirty:
        raise SystemExit(
            "refusing to issue against a dirty tree: the artifact binds a "
            "commit, and uncommitted work is not in it. Commit the readiness "
            "record first, then issue.")

    existing = REPO_ROOT / AUTH_REL
    version = args.version
    if version is None:
        version = (int(json.loads(existing.read_text()).get("version", 0)) + 1
                   if existing.is_file() else 1)
    granted_by = args.granted_by or (
        json.loads(existing.read_text())["granted_by"] if existing.is_file()
        else "maintainer decision")

    record = build_record(rate=args.rate, commit=git("rev-parse", "HEAD"),
                          money=money, priced=priced, version=version,
                          granted_by=granted_by)
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps(record, indent=1) + "\n")
    print(f"\nwrote {AUTH_REL}  version {version}  "
          f"hard_cap ${record['hard_cap_usd']:.4f}  "
          f"commit {record['authorized_session_commit'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
