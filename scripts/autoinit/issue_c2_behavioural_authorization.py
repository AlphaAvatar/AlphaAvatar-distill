#!/usr/bin/env python3
"""Issue the one-use authorization for the Phase-C2 behavioural campaign.

    PYTHONPATH=src:scripts python \
        scripts/autoinit/issue_c2_behavioural_authorization.py \
        --run-id attempt1 --rate 1.09

Run this AFTER the grant is committed and the launch-bound readiness sweep has
been recorded and committed, and BEFORE the bundle is staged. The ordering is
not a style preference: the bundle must carry the authorization, so the
authorization must exist first, and `session_commit_and_lineage` permits
exactly one tracked path to differ between the authorized base and the session
commit — so the readiness record and the authorization are separate commits.

**The rate is LIVE and the amounts are derived from it.** `authorization_terms`
re-prices the reviewed decomposition at whatever `gpuTypes.securePrice` says
now and returns the five amounts distinctly. Nothing here reads
`QUOTED_RATE_USD_PER_HOUR`: an authorization carrying a dollar window derived
from a stale quote is a window that does not bound the bill. The scientific
work is NOT rescaled either way — a dearer card does not shorten the
experiment and a cheaper one does not extend it.

It binds the artifact to five things the launcher re-derives and refuses on
disagreement: the session commit, the executable closure digest, the plan hash,
the campaign, and the money.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402
from experiments.phase_c2.session import C2ResourceScope  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402

EXPERIMENT_ID = "phase_c2_behavioural"
STAGE_ID = "1"

#: How much the live quote may move the dollar authorization before this stops
#: and returns to the maintainer. The grant's rule is "materially changes"; a
#: cent on a $33 ceiling is not material and a re-quote that moves it by more
#: than this is not a number anybody reviewed.
MATERIAL_USD = 0.05


def governance_path(run_id: str, name: str) -> str:
    return f"{rel_run_dir(EXPERIMENT_ID, run_id, STAGE_ID)}/governance/{name}"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=REPO_ROOT).stdout.strip()


def build_authorization_record(*, grant: dict, terms: dict, commit: str,
                               dirty: bool, live: dict, plan_hash: str,
                               run_id: str) -> dict:
    """Assemble the document. Extracted so a `$0` test can round-trip it.

    BUILT as the object and then serialised, never hand-written: the loader
    maps `plan_hash` from `phase_a_session_plan_hash` and `science_plan_hash`
    from `phase_a_science_plan_hash`, and it reconciles the five amounts
    against each other. A document assembled from what an issuer felt it needed
    has parsed as nothing the loader could read, after a whole chain was built
    on it.
    """
    auth = BG.BehaviouralAuthorization(
        authorization_id=BG.PLAN_ID,
        granted_utc=datetime.now(timezone.utc).isoformat(),
        granted_by=grant["granted_by"],
        plan_id=BG.PLAN_ID,
        plan_hash=plan_hash,
        #: The plan IS the science here: the frozen protocol, the six arms'
        #: identities, the batteries and the recovery recipe. There is no
        #: second document to hash, so the two are the same object rather than
        #: one being invented.
        science_plan_hash=plan_hash,
        expected_usd=float(terms["expected_all_in_usd"]),
        #: GPU money. The runner compares the planned hard threshold — which is
        #: GPU dollars — against this, and the loader refuses a document where
        #: the two disagree.
        hard_cap_usd=float(terms["gpu_hard_usd"]),
        authorized_stages=BG.AUTHORIZED_STAGES,
        stage_conditions={
            "prepare": ("verifies the teacher by shard hash and materializes "
                        "the arms the remaining probes need, each along its "
                        "pinned path and each gated on its exact recorded "
                        "identity. A mismatch is a scientific finding and "
                        "stops the session; no probe starts."),
            "screen": ("six fresh screening probes on the one frozen screening "
                       "seed, each announced for durability the moment it "
                       "finishes and pulled off-pod while the next trains."),
            "rank": ("scores the screening battery through its own pinned "
                     "scorer, ranks by paired delta against B, and advances "
                     "EXACTLY one by the frozen tie-break. No verdict leaves "
                     "this stage."),
            "confirm": ("six fresh confirmation probes — the advanced "
                        "candidate and B, paired across the three frozen "
                        "seeds. Cannot begin until every screening probe is "
                        "trained AND scored."),
            "decide": ("applies the frozen Phase-C rule at C2's own bootstrap "
                       "seed and STOPS. GO, NO_GO and INCONCLUSIVE are all "
                       "complete results."),
        },
        scope_note=grant["covers"],
        authorized_session_commit=commit,
        harness_source_digest=live["digest"],
        harness_source_files=tuple(row["path"] for row in live["files"]),
        resource_scope=C2ResourceScope(
            run_id=run_id,
            issuances_permitted=1,
            launch_attempts_permitted=1,
            provider_resources_permitted=int(grant["max_provider_resources"]),
            one_billing_resource_at_a_time=True),
        provenance_commit=f"{commit}{'+dirty' if dirty else ''}",
        campaign_id=BG.CAMPAIGN_ID,
        rate_usd_per_hour=float(terms["rate_usd_per_hour"]),
        hard_runtime_minutes=float(terms["hard_runtime_minutes"]),
        gpu_hard_usd=float(terms["gpu_hard_usd"]),
        disk_hard_usd=float(terms["disk_hard_usd"]),
        all_in_hard_usd=float(terms["all_in_hard_usd"]),
        campaign_all_in_hard_usd=float(terms["campaign_all_in_hard_usd"]),
    )
    record = auth.as_dict()
    record["run_id"] = run_id
    record["stage_id"] = STAGE_ID
    record["one_use"] = True
    record["max_provider_resources"] = int(grant["max_provider_resources"])
    record["grant_path"] = governance_path(run_id, "grant.json")
    record["terms_basis"] = terms
    record["_two_ceilings"] = (
        "all_in_hard_usd bounds THIS SESSION: it is derived from the live rate "
        "and the frozen runtime, and it is what the watchdog, the window and "
        "every spend check inside the pod are built from. "
        "campaign_all_in_hard_usd bounds the CAMPAIGN cumulatively across every "
        "attempt and provider resource; it is a maintainer number, not a "
        "derived one. `campaign_continuation_gate` sums each predecessor's GPU "
        "actual plus its derived container disk, adds this session's remaining "
        "planned work, and compares that total to the CAMPAIGN ceiling. "
        "Raising the campaign ceiling funds another attempt. It buys no "
        "runtime, no disk, no probes, no seeds and no scientific scope: those "
        "come from the frozen record through the session ceiling alone.")
    #: Recomputed LAST, over everything above, because the loader verifies the
    #: document it is given and not the object that produced it.
    record.pop("authorization_sha256", None)
    record["authorization_sha256"] = sha256_json(record)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rate", type=float, required=True,
                    help="the LIVE gpuTypes.securePrice this artifact is "
                         "issued at — never communityPrice, which the launcher "
                         "does not pay")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="issue against a dirty tree. The launcher will refuse "
                         "the result; this exists for local inspection only.")
    args = ap.parse_args(argv)

    dirty = bool(git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "refusing to issue against a dirty tree: the artifact binds a "
            "commit, and uncommitted work is not in it. Commit the readiness "
            "record first, then issue.")

    commit = git("rev-parse", "HEAD")
    grant_rel = governance_path(args.run_id, "grant.json")
    grant_path = REPO_ROOT / grant_rel
    if not grant_path.is_file():
        raise SystemExit(f"no grant at {grant_rel}; nothing authorizes this run")
    grant = json.loads(grant_path.read_text())

    if grant.get("campaign_id") != BG.CAMPAIGN_ID:
        raise SystemExit(
            f"the grant names campaign {grant.get('campaign_id')!r} and this "
            f"tree declares {BG.CAMPAIGN_ID!r}.")

    #: A GRANT IS WRITTEN BY HAND AND ITS DATE FIELD SAYS `utc`. attempt3's
    #: said 2026-09-21 while the session ran on 2026-09-20 UTC — a local
    #: timezone date in a UTC field, which nothing detected because nothing
    #: read it. It distorted no money there, since run costs are attributed
    #: from closeouts through the run index. But a grant dated after the work
    #: it authorizes is not a record anyone can reason from, and the check is
    #: one comparison against the clock.
    granted = str(grant.get("granted_utc", ""))[:10]
    today = datetime.now(timezone.utc).date().isoformat()
    if granted > today:
        raise SystemExit(
            f"the grant is dated {granted} and today is {today} UTC. A grant "
            "cannot be issued from the future: either the date was written "
            "from a local timezone — which is what happened to attempt3 — or "
            "the clock is wrong. Fix the grant before issuing against it.")

    approved = grant["approved_money"]
    if float(args.rate) > float(approved["max_rate_usd_per_hour"]):
        raise SystemExit(
            f"the live rate ${args.rate}/h is above the approved maximum "
            f"${approved['max_rate_usd_per_hour']}/h. The ceiling was derived "
            "from that rate; at a higher one the same window costs more than "
            "was approved. Re-derive and return for a decision — the "
            "experiment is NOT shortened to fit a price increase.")

    #: THE CAMPAIGN CEILING IS NOT DERIVED. It is the maintainer's cumulative
    #: number for the whole campaign and it is read, never computed: no rate,
    #: no runtime and no amount of remaining work may move it. It is required
    #: explicitly, because a grant that names only one ceiling cannot say which
    #: of the two it meant, and guessing would silently fund a session at the
    #: campaign number.
    if "campaign_all_in_usd" not in approved:
        raise SystemExit(
            "the grant's approved_money names no campaign_all_in_usd. Since "
            "the two ceilings diverged, a grant must state BOTH the cumulative "
            "campaign ceiling and this session's all_in_usd; one number cannot "
            "say which it is, and reading the session ceiling as the campaign "
            "one would under-fund the campaign while reading it the other way "
            "would hand a single session the campaign's whole budget.")
    campaign_ceiling = float(approved["campaign_all_in_usd"])
    if abs(campaign_ceiling - BG.CAMPAIGN_ALL_IN_CEILING_USD) > MATERIAL_USD:
        raise SystemExit(
            f"the grant approves a ${campaign_ceiling} cumulative campaign "
            f"ceiling and this reviewed tree declares "
            f"${BG.CAMPAIGN_ALL_IN_CEILING_USD}. The campaign ceiling is a "
            "maintainer decision, so the two must agree: a grant is written by "
            "hand, and this is exactly where a ceiling typo would enter "
            "unreviewed. If the maintainer raised it, the raise belongs in the "
            "reviewed tree too.")

    #: DERIVED AT THE LIVE RATE, then compared to what was approved. A quote
    #: that moves the dollar authorization materially is a number nobody
    #: reviewed, and the grant's own rule is to stop and return. This compares
    #: the SESSION ceilings: the campaign ceiling is not a function of the rate,
    #: so a live quote can never justify moving it.
    terms = BG.authorization_terms(REPO_ROOT, rate_usd_per_hour=args.rate,
                                   campaign_all_in_hard_usd=campaign_ceiling)
    drift = abs(float(terms["all_in_hard_usd"]) - float(approved["all_in_usd"]))
    if drift > MATERIAL_USD:
        raise SystemExit(
            f"at the live rate ${args.rate}/h this session's ceiling derives "
            f"to ${terms['all_in_hard_usd']} and the grant approves "
            f"${approved['all_in_usd']} — a ${drift:.4f} change. That is a "
            "materially different dollar authorization from the reviewed "
            "basis. STOP and return to the maintainer before creating a "
            "provider resource.")

    live = BG.current_executable(REPO_ROOT)
    plan_hash = BG.plan_hash(REPO_ROOT)

    #: The reviewed identities, re-derived and CHECKED. The grant records them
    #: so a reader can see what was reviewed; nothing trusts that copy.
    reviewed = grant.get("reviewed_baseline") or {}
    for label, mine, theirs in (
            ("plan hash", plan_hash, reviewed.get("plan_hash")),
            ("executable closure", live["digest"],
             reviewed.get("executable_closure")),
            ("closure file count", live["n_files"],
             reviewed.get("closure_files"))):
        if theirs is not None and mine != theirs:
            raise SystemExit(
                f"the reviewed {label} is {theirs} and this tree derives "
                f"{mine}. The tree has moved since the review; an "
                "authorization issued over it would bind work nobody reviewed.")

    record = build_authorization_record(
        grant=grant, terms=terms, commit=commit, dirty=dirty, live=live,
        plan_hash=plan_hash, run_id=args.run_id)

    out_rel = governance_path(args.run_id, "authorization.json")
    out = REPO_ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")

    #: Loaded back through the REAL loader before anything is reported as
    #: issued. The loader reconciles the five amounts, the campaign and the
    #: schema; a document that cannot be read is not an authorization.
    reloaded = BG.BehaviouralAuthorization.load(out)

    print(json.dumps({
        "wrote": out_rel,
        "campaign_id": reloaded.campaign_id,
        "authorized_session_commit": commit,
        "plan_hash": plan_hash,
        "harness_digest": live["digest"],
        "harness_n_files": live["n_files"],
        "rate_usd_per_hour": reloaded.rate_usd_per_hour,
        "hard_runtime_minutes": reloaded.hard_runtime_minutes,
        "gpu_hard_usd": reloaded.gpu_hard_usd,
        "disk_hard_usd": reloaded.disk_hard_usd,
        "all_in_hard_usd": reloaded.all_in_hard_usd,
        "campaign_all_in_hard_usd": reloaded.campaign_all_in_hard_usd,
        "expected_all_in_usd": terms["expected_all_in_usd"],
        "loaded_back": True,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
