#!/usr/bin/env python3
"""Issue the one-use authorization for the replay-only reconstruction.

    PYTHONPATH=src:scripts python \
        scripts/autoinit/issue_c2_replay_authorization.py --run-id attempt1

Run this AFTER the grant is committed and the launch-bound readiness sweep has
been recorded and committed, and BEFORE the bundle is staged. The ordering is
not a style preference — the bundle must carry the authorization, so the
authorization must exist first, and `session_commit_and_lineage` permits exactly
one tracked path to differ between the authorized base and the session commit,
so the readiness record and the authorization are separate commits.

It binds the artifact to four things the launcher then re-derives and refuses on
disagreement:

* the **session commit**, so an authorization cannot travel to another tree;
* the **executable digest**, derived live over the replay's own closure;
* the **plan hash**, which for this session IS the source binding — the five
  digest-pinned paths and the attempt-3 commit they came from. An authorization
  issued against a different set of pins cannot run this session;
* the **money**, GPU and separately billed disk kept apart.

A dirty tree is refused by default. Attempt 15 of another experiment skipped the
commit-only-the-readiness-record step and was refused at `$0` for it, which is
the cheap place, but the refusal is still a wasted invocation.
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

from experiments.phase_c2 import replay as RG  # noqa: E402
from experiments.phase_c2 import replay_specs as RS  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402

EXPERIMENT_ID = "phase_c2_replay"
STAGE_ID = "1"


def governance_path(run_id: str, name: str) -> str:
    return f"{rel_run_dir(EXPERIMENT_ID, run_id, STAGE_ID)}/governance/{name}"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=REPO_ROOT).stdout.strip()


def build_authorization_record(*, grant, approved, commit, dirty, rate,
                               binding, live, plan_hash,
                               run_id: str = "attempt1") -> dict:
    """Assemble the document. Extracted so a $0 test can round-trip it.

    The assembly and the loader were written independently and did not
    agree; the disagreement surfaced in a dry run, after the sweep, the
    authorization and the bundle had all been built on it. A test can now
    build and load one without an issued artifact on the tree — which
    matters, because the sweep runs BEFORE the authorization exists.
    """
    #: BUILT as the object and then serialised, never hand-written. The
    #: loader maps `plan_hash` from `phase_a_session_plan_hash` and
    #: `science_plan_hash` from `phase_a_science_plan_hash`, and the
    #: constructor requires eleven fields; a document assembled from what this
    #: session felt it needed parsed as nothing the loader could read, and the
    #: dry run refused it after the whole chain had been issued.
    #: The campaign REMAINDER, derived from the recorded closeouts. Computed
    #: here rather than passed in, so every caller of this builder — the issuer
    #: and the $0 round-trip test — prices the same way.
    money = RG.remaining_usd()
    if money["this_attempt_gpu_usd"] <= 0:
        raise ValueError(
            f"the campaign has spent ${money['campaign_spent_usd']:.4f} of "
            f"${money['campaign_all_in_usd']:.2f}; nothing remains to "
            "authorize. A new ceiling is a maintainer decision.")

    auth = RG.ReplayAuthorization(
        authorization_id=RG.PLAN_ID,
        granted_utc=datetime.now(timezone.utc).isoformat(),
        granted_by=grant["granted_by"],
        plan_id=RG.PLAN_ID,
        #: The plan IS the source binding: the five digest-pinned paths and the
        #: attempt-3 commit they came from.
        plan_hash=plan_hash,
        #: No separate science plan. This session produces no measurement, so
        #: the two hashes are the same object rather than one being invented.
        science_plan_hash=plan_hash,
        expected_usd=float(approved.get("expected_usd", money["this_attempt_gpu_usd"])),
        #: The campaign REMAINDER, not the campaign ceiling. Eight attempts at
        #: the ceiling would be $38 against a $4.77 campaign.
        hard_cap_usd=money["this_attempt_gpu_usd"],
        authorized_stages=RG.AUTHORIZED_STAGES,
        stage_conditions={
            "bind_identities": ("binds the source binding, builds the five "
                                "pinned specs and checks disk headroom. Loads "
                                "no model."),
            "reconstruct": ("materializes each pinned path in order, gating "
                            "every intermediate on its recorded digest, and "
                            "secures each finished leaf before the next path "
                            "starts. A digest mismatch is TERMINAL."),
        },
        scope_note=(
            "ONE replay-only reconstruction of the five checkpoints behind "
            "attempt 3's frozen Top-5 (selection "
            f"{RS.SELECTION_SHA256[:12]}…), from session commit "
            f"{RS.ATTEMPT3_SESSION_COMMIT[:12]}…. It decides nothing: no beam, "
            "no ranking, no selection, no selection-bearing evaluation, no "
            "control comparison, no training and no behavioural work."),
        authorized_session_commit=commit,
        harness_source_digest=live["digest"],
        harness_source_files=tuple(row["path"] for row in live["files"]),
        provenance_commit=f"{commit}{'+dirty' if dirty else ''}",
    )
    record = auth.as_dict()
    #: Facts the TYPE does not carry, added beside it rather than in place of
    #: it. The separately billed disk is here because `hard_cap_usd` is GPU
    #: money and a ceiling that folded the two together would hide one of them.
    record["run_id"] = run_id
    record["stage_id"] = STAGE_ID
    record["rate_usd_per_hour"] = rate
    record["one_use"] = True
    record["max_provider_resources"] = 1
    record["money"] = {
        **money,
        "gpu_hard_usd": money["this_attempt_gpu_usd"],
        "disk_usd": money["disk_usd"],
        "all_in_usd": money["remaining_all_in_usd"],
        "_why_two_numbers": (
            "the provider bills container disk separately from the GPU. "
            "`hard_cap_usd` is GPU money, which is what the budget planner and "
            "the watchdog spend against; the disk is real and is recorded "
            "beside it rather than folded in."),
    }
    record["grant_path"] = governance_path(run_id, "grant.json")
    record["source_binding"] = binding
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
                    help="the live securePrice this artifact is issued at")
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

    #: The grant states the money a person approved. The issuer does NOT take
    #: the ceiling from its own module and hope they agree — it checks, and a
    #: disagreement is a refusal rather than a silent preference for one of them.
    approved = grant["approved_money"]
    for field, mine in (("gpu_hard_usd", RG.GPU_HARD_USD),
                        ("disk_usd", RG.DISK_USD),
                        ("all_in_usd", RG.ALL_IN_USD)):
        if abs(float(approved[field]) - mine) > 1e-9:
            raise SystemExit(
                f"the grant approves {field}=${approved[field]} and this tree "
                f"declares ${mine}. One of them is stale; an authorization that "
                "chose between them silently would spend money nobody approved.")

    if float(args.rate) > float(approved["max_rate_usd_per_hour"]):
        raise SystemExit(
            f"the live rate ${args.rate}/h is above the approved maximum "
            f"${approved['max_rate_usd_per_hour']}/h. The ceiling was derived "
            "from that rate; at a higher one the same window costs more than "
            "was approved. Re-derive and return for a decision.")

    #: The pins must still describe this tree, checked here as well as at the
    #: gate: an authorization issued over moved operators would be an artifact
    #: authorizing a replay that cannot succeed.
    binding = RS.source_binding(REPO_ROOT)
    live = RG.current_executable(REPO_ROOT)
    plan_hash = RG.plan_hash(REPO_ROOT)

    record = build_authorization_record(
        grant=grant, approved=approved, commit=commit, dirty=dirty,
        rate=args.rate, binding=binding, live=live, plan_hash=plan_hash,
        run_id=args.run_id)

    out_rel = governance_path(args.run_id, "authorization.json")
    out = REPO_ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")

    money = RG.remaining_usd(REPO_ROOT)
    print(json.dumps({
        "wrote": out_rel,
        "authorized_session_commit": commit,
        "plan_hash": plan_hash[:16] + "…",
        "harness_digest": live["digest"][:16] + "…",
        "harness_n_files": live["n_files"],
        "gpu_hard_usd": money["this_attempt_gpu_usd"],
        "all_in_usd": money["remaining_all_in_usd"],
        "campaign_spent_usd": money["campaign_spent_usd"],
        "rate": args.rate,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
