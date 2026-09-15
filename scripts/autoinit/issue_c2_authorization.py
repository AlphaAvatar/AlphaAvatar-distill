#!/usr/bin/env python3
"""Issue the ONE-USE Phase-C2 Search-1 authorization. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/issue_c2_authorization.py \
        --run-id attempt2 \
        --grant logs/stages/stage-1/phase_c2/runs/attempt2/governance/grant.json

The grant is an **input**, not a constant. `experiments/phase_c2/session.py`
carries the authorization *type* — the hard-`False` scope properties, the
schema-refusing loader, the ceiling derivation — and nothing about a particular
permission. A one-use maintainer decision living in executable source goes stale
silently and still reads as though it applies.

**This cannot become a Phase-A, a C1 or a recovery grant.**
`C2Authorization.allows_phase_a` and `.allows_recovery_training` are hard
`False` properties with no field to set, and `load` refuses any artifact whose
schema is not the C2 one. That is a property of the type, not a promise here.

What it binds, each of it RE-DERIVED from the tree and refused on disagreement:

* the **session commit**, the clean pre-authorization HEAD the pod checks out;
* the **live derived C2 executable closure**, digest and file set, which the
  launcher's own gates re-derive from that commit's blobs and from the bundle;
* the **C2 plan hash**, covering the session contract AND the configured search
  space, so a grant cannot survive a change to either;
* the **session-contract hash** as the science plan;
* the **pricing record's own hash** and the exact $15.0446 ceiling it derives;
* the **frozen baseline B** by spec hash and artifact digest, re-derived by
  constructing the fixed path rather than read from a constant;
* a valid **launch-bound C2 readiness record**, PASS, for this run against this
  tree.

Issuing is not launching.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
#: NOT `scripts/experiments/phase_c2` on sys.path. The sibling
#: `scripts/experiments/phase_c1` holds `packaging.py`, which shadows the
#: third-party `packaging` distribution and breaks the next transformers import
#: in the process; reaching experiment modules through the package is the
#: convention that avoids it, and `scripts` on the path already allows it.

from experiments.phase_c2.authorization_payload import (  # noqa: E402
    C2AuthorizationRefused, build_c2_authorization_payload, load_config,
)
from experiments.phase_c2.session import C2Authorization  # noqa: E402


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grant", required=True, type=Path)
    #: REQUIRED, unlike C1's. C2 has no repository-level authorization path to
    #: fall back to: the artifact is owned by the run from the start, which is
    #: where C1 arrived after a pointer, a history file and a migration.
    ap.add_argument("--run-id", required=True,
                    help="the run this authorization is issued for. The "
                         "artifact is written into that run's governance area, "
                         "so a later issuance cannot overwrite it.")
    ap.add_argument("--stage-id", default=None,
                    help="the run's declared stage. Defaults to the stage the "
                         "experiment's own configuration declares.")
    ap.add_argument("--out", default=None,
                    help="explicit output path. Defaults to this run's "
                         "governance/authorization.json.")
    #: DEFAULT ON. C1 made the clean-tree check opt-in and attempt 15 aborted at
    #: $0 because it was not passed: the launch-bound record was still
    #: uncommitted when the authorization was issued, so the authorized base
    #: predated both, and the lineage gate correctly refused a two-path diff. An
    #: option that must be remembered to be safe is a default in the wrong
    #: position.
    ap.add_argument("--allow-dirty", action="store_true",
                    help="issue against a dirty tree. Almost never right: the "
                         "authorized session commit must describe exactly what "
                         "the pod checks out, and anything uncommitted now "
                         "becomes a second path in the lineage diff later.")
    ap.add_argument("--porcelain", action="store_true")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT)
    stage_id = args.stage_id or cfg["stage_id"]

    #: The grant is read here and VALIDATED inside the builder, so the CLI and a
    #: test candidate are refused for the same reasons by the same code.
    grant = json.loads(args.grant.read_text())

    #: The two effects that make this an ISSUANCE rather than a computation, and
    #: why they stay in the CLI: refusing a dirty tree, and reading the clock.
    #: `build_c2_authorization_payload` does neither.
    if not args.allow_dirty:
        dirty = git("status", "--porcelain")
        if dirty:
            raise SystemExit(
                "refusing to issue against a dirty tree:\n" + dirty
                + "\nThe authorized session commit must describe exactly what "
                  "the pod checks out.")

    commit = git("rev-parse", "HEAD")
    try:
        payload = build_c2_authorization_payload(
            grant=grant, session_commit=commit,
            granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            run_id=args.run_id, stage_id=stage_id, repo_root=REPO_ROOT,
            grant_path=str(args.grant))
    except C2AuthorizationRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None

    if args.out is None:
        from experiments.phase_c2.session import c2_authorization_path

        args.out = c2_authorization_path(args.run_id, stage_id)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    #: An issuance never writes over an existing authorization. One-use means
    #: one artifact; replacing one in place is how a consumed authorization
    #: becomes indistinguishable from a fresh one.
    if out.exists():
        raise SystemExit(
            f"{args.out} already exists. An authorization is one-use: issue "
            "into a new run, or move the existing artifact aside deliberately.")
    out.write_text(json.dumps(payload, indent=1) + "\n")

    #: Round-trip through the REAL loader: an artifact this issuer wrote but the
    #: driver cannot load is worse than none. Every assertion below is a
    #: property the pod depends on.
    reloaded = C2Authorization.load(out)
    bound = payload["bound"]
    assert reloaded.hard_cap_usd == float(cfg["accepted_pricing"]["hard_ceiling_usd"])
    assert reloaded.plan_hash == bound["c2_plan_hash"]
    assert reloaded.science_plan_hash == bound["c2_session_contract_hash"]
    assert reloaded.harness_source_digest == bound["c2_harness_digest"]
    assert len(reloaded.harness_source_files) == bound["c2_harness_n_files"]
    assert reloaded.authorized_session_commit == commit
    assert reloaded.allows_phase_a is False
    assert reloaded.allows_recovery_training is False
    assert reloaded.authorizes_c2_search1 is True

    if args.porcelain:
        print(json.dumps({"authorization_id": payload["authorization_id"],
                          "authorization_sha256": payload["authorization_sha256"],
                          "authorized_session_commit": commit,
                          "run_id": args.run_id, "stage_id": stage_id}))
    else:
        money = cfg["accepted_pricing"]
        print(f"wrote {args.out}")
        print(f"  owned by run       {args.run_id} (stage {stage_id})")
        print(f"  authorization_id   {payload['authorization_id']}")
        print(f"  authorization_sha  {payload['authorization_sha256']}")
        print(f"  session commit     {commit}")
        print(f"  executable closure {bound['c2_harness_digest']} "
              f"({bound['c2_harness_n_files']} files, derived)")
        print(f"  plan               {bound['c2_plan_hash']}")
        print(f"  session contract   {bound['c2_session_contract_hash']}")
        print(f"  pricing            {bound['pricing_sha256']}")
        print(f"  baseline B         spec {bound['baseline']['spec_hash'][:16]} "
              f"artifact {bound['baseline']['artifact_digest'][:16]}")
        print(f"  readiness          {bound['readiness'].get('record')} "
              f"{bound['readiness'].get('record_kind')}")
        print(f"  ceiling            ${payload['hard_cap_usd']:.4f}  "
              f"(expected ${money['expected_usd']:.4f}, "
              f"soft stop ${money['soft_stop_usd']:.4f})")
        print(f"  cumulative         "
              f"${float(grant['budget_context_at_approval']['cumulative_spend_usd']):.4f}"
              f" spent of ${money['cumulative_cap_usd']:.4f}")
        print("  ISSUING IS NOT LAUNCHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
