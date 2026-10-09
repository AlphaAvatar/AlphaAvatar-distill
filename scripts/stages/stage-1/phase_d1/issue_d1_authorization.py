#!/usr/bin/env python3
"""Issue D1's ONE-USE formal search authorization. Reproducibly, from a command.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d1/issue_d1_authorization.py \
        --grant logs/budget/approvals/autoinit_d1_grant.json \
        --run-id d1_search_20261006_120000

**This script did not exist.** The committed
`runs/d1_search_20261005_173304/governance/authorization.json` was produced by
no logged command, which is the thing AGENTS.md P4 forbids: an experiment must
be reproducible from a logged command, and a governance artifact assembled by
hand cannot be re-derived, re-checked, or shown to have hashed what it claims.
Its `granted_utc` was `2026-10-06T00:00:00Z` -- a timestamp in the FUTURE of
the moment it was written -- which no `datetime.now()` can produce and which
therefore cannot support the one claim the package contract attaches to it:
that the rate was quoted live, immediately before issuance.

Everything of substance is `d1_authorization.build_payload`'s, not this file's.
This is the command around it: read the maintainer's grant, refuse a dirty
tree, take HEAD as the session commit, let `build_payload` quote the live rate
and derive the ceiling, write the artifact once, and read it back through the
TYPE that will load it on the pod.

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
for extra in ("src", "scripts", "scripts/autoinit"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from stages.phase_d1 import d1_authorization as A  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grant", required=True, type=Path,
                    help="the maintainer's grant document")
    #: REQUIRED, and it is not bookkeeping: `run_id` is a field of
    #: `SearchConfig.as_dict()` and therefore of the `config_hash` this
    #: authorization binds. An authorization issued for one run id cannot
    #: authorize a launch under another.
    ap.add_argument("--run-id", required=True,
                    help="the run this authorization is issued for; it enters "
                         "the bound config_hash")
    ap.add_argument("--gpu", default="NVIDIA L40S",
                    help="the card whose live securePrice is quoted")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="issue against a dirty tree. Almost never right: the "
                         "authorized session commit must describe exactly what "
                         "the pod checks out.")
    ap.add_argument("--porcelain", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.grant.is_file():
        raise SystemExit(f"no grant at {args.grant}")
    grant = json.loads(args.grant.read_text())

    if not args.allow_dirty:
        dirty = git("status", "--porcelain")
        if dirty:
            raise SystemExit(
                "refusing to issue against a dirty tree:\n" + dirty
                + "\nThe authorized session commit must describe exactly what "
                  "the pod checks out, and the harness digest is derived from "
                  "the working tree.")

    commit = git("rev-parse", "HEAD")

    #: THE REAL UTC, from the clock, at the moment of issuance. Not a constant,
    #: not rounded to midnight, and not in the future: the live rate quoted
    #: inside `build_payload` is quoted now, and `granted_utc` is what dates it.
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
        payload = A.build_payload(
            grant=grant, session_commit=commit, granted_utc=granted_utc,
            run_id=args.run_id, repo_root=REPO_ROOT)
    except A.D1AuthorizationRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    #: READ BACK THROUGH THE LOADING TYPE. An artifact the pod's dispatch branch
    #: would refuse is not an authorization, and discovering that here is free.
    reloaded = A.D1Authorization.load(out)
    reloaded.require_plan(D1S.design_hash(REPO_ROOT))
    reloaded.require_run_id(args.run_id)
    reloaded.require_treatment_arm()
    reloaded.require_session_commit(commit)
    reloaded.require_harness(REPO_ROOT)
    for claim, ok in (
            ("authorizes_d1_search", reloaded.authorizes_d1_search),
            ("allows_beam_search", reloaded.allows_beam_search),
            ("refuses recovery", not reloaded.allows_recovery),
            ("refuses behavioural", not reloaded.allows_behavioural),
            ("refuses automatic follow-on",
             not reloaded.automatic_followon_start)):
        if not ok:
            raise SystemExit(f"the issued artifact fails its own check: {claim}")

    #: AND THE IDENTITY THE DRIVER WILL ASSERT. `config_hash` is the one that
    #: could not be matched before; checking it here closes the loop between the
    #: issuer and stage A rather than trusting that both call the same helper.
    import tempfile

    D1S._register_frozen_operators()
    with tempfile.TemporaryDirectory(prefix="d1-issue-") as tmp:
        session = D1S.build_session(
            arm=A.FORMAL_ARM, workdir=Path(tmp), run_id=args.run_id,
            device=A.FORMAL_DEVICE, repo_root=REPO_ROOT)
        contract = D1S.assert_session_contract(session, REPO_ROOT)
    for field, got in (("config_hash", contract["config_hash"]),
                       ("measurement_protocol_id",
                        contract["measurement_protocol_id"]),
                       ("suite_content_sha256",
                        contract["suite_content_sha256"])):
        if payload[field] != got:
            raise SystemExit(
                f"the issued artifact binds {field}={payload[field][:16]} and "
                f"the session this tree builds for run {args.run_id!r} on "
                f"{A.FORMAL_DEVICE} carries {got[:16]}. A launch would be "
                "refused in stage A, on a billing pod.")

    quote = payload["money"]["live_quote"]
    if args.porcelain:
        print(json.dumps({
            "path": rel,
            "authorization_sha256": payload["authorization_sha256"],
            "session_commit": commit, "run_id": payload["run_id"],
            "granted_utc": payload["granted_utc"],
            "live_rate_usd_per_hour": quote["usd_per_hour"],
            "hard_cap_usd": payload["hard_cap_usd"],
            "config_hash": payload["config_hash"],
            "harness_digest": payload["harness_source_digest"]}))
        return 0
    print(f"wrote {rel}")
    print(f"  authorization_sha  {payload['authorization_sha256']}")
    print(f"  granted_utc        {payload['granted_utc']}  (real clock)")
    print(f"  run_id             {payload['run_id']}")
    print(f"  session commit     {commit}")
    print(f"  arm                {payload['arm']}")
    print(f"  design / plan      {payload['design_hash'][:16]}")
    print(f"  protocol           {payload['measurement_protocol_id']}")
    print(f"  config_hash        {payload['config_hash']}")
    print(f"  suite content      {payload['suite_content_sha256'][:16]}")
    print(f"  harness            {payload['harness_source_digest'][:16]} "
          f"({payload['harness']['n_files']} files, derived)")
    print(f"  live rate          ${quote['usd_per_hour']}/h "
          f"(stock {quote.get('stock_status')}, {quote['_quoted']})")
    print(f"  ceiling            ${payload['hard_cap_usd']:.4f} "
          f"(expected ${payload['expected_usd']:.4f} over "
          f"{payload['money']['derived_session']['hard_ceiling_minutes']:.2f} "
          "ceiling minutes)")
    print(f"  four conditions    {payload['money']['four_conditions']}")
    print(f"  stages             {'/'.join(payload['authorized_stages'])}")
    #: READ from the design, not typed. This line said "Top-2" for a round
    #: after the maintainer widened the finalist set to 4.
    top_k = int(D1S.design()["behavioural_design"]["top_k"])
    print(f"  authorizes ONE D1 formal beam and a committed Top-{top_k}; "
          "NOT recovery, "
          "NOT behavioural work, NOT a promotion decision.")
    print("  ISSUING IS NOT LAUNCHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
