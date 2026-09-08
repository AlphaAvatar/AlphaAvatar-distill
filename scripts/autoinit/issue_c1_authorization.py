#!/usr/bin/env python3
"""Issue the ONE-USE Phase-C1 authorization. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/issue_c1_authorization.py \
        --grant logs/autoinit_c1_grant.json --require-clean

Same contract as the Phase-A, Phase-B and continuation issuers, and the same
reason for it: the grant is an **input**, not a constant. `c1_authorization.py`
carries the authorization *type* -- the hard-`False` scope properties, the
harness file set, the ceiling derivation -- and nothing about a particular
permission. A one-use maintainer decision living in executable source goes stale
silently and still reads as though it applies.

**This cannot become a Phase-A or a search grant.**
`C1Authorization.allows_phase_a` and `.allows_beam_search` are hard `False`
properties with no field to set, and `load` refuses any artifact whose schema is
not the C1 one. That is a property of the type, not a promise in this docstring.

What this binds, and what invalidates it if edited:

* the **session commit**, the clean pre-authorization HEAD the pod checks out;
* the **C1 harness digest**, over the declared file set, which
  `session_commit_gate` independently re-derives from that commit's own blobs;
* the **C1 isolation plan hash**, rebuilt from the committed frozen identities
  rather than transcribed, with the treatment operator explicitly registered
  first -- importing its module does not register it;
* the **C0 preregistration digest** as the science plan;
* the **execution preregistration**, by its own self-verified hash;
* the **hard ceiling**, cross-checked against `logs/phase_c1_pricing.json`;
* the **battery**, **teacher binding** and **scoring contract** identities the
  session will measure under.

Every one of those is DERIVED here and refused if the grant asserts it. A grant
that asserts an identity it did not compute is not evidence of anything.

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
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/phase_c1"))

from experiments.phase_c1.authorization import C1Authorization  # noqa: E402
from authorization_payload import (  # noqa: E402
    C1AuthorizationRefused, build_c1_authorization_payload, load_config,
)

OUT = "logs/autoinit_c1_authorization.json"


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
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse to issue against a dirty tree; the authorized "
                         "commit must describe what the pod will check out")
    ap.add_argument("--porcelain", action="store_true")
    args = ap.parse_args()

    #: The grant is read here and VALIDATED inside the builder, so the CLI and
    #: a test candidate are refused for the same reasons by the same code.
    grant = json.loads(args.grant.read_text())

    #: The two effects that make this an ISSUANCE rather than a computation, and
    #: which is why they stay in the CLI: refusing a dirty tree, and reading the
    #: clock. `build_c1_authorization_payload` does neither.
    if args.require_clean:
        dirty = git("status", "--porcelain")
        if dirty:
            raise SystemExit(
                "refusing to issue against a dirty tree:\n" + dirty
                + "\nThe authorized session commit must describe exactly what the "
                  "pod checks out.")

    commit = git("rev-parse", "HEAD")
    try:
        payload = build_c1_authorization_payload(
            grant=grant, session_commit=commit,
            granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            repo_root=REPO_ROOT, grant_path=str(args.grant))
    except C1AuthorizationRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None

    out = REPO_ROOT / args.out
    out.write_text(json.dumps(payload, indent=1) + "\n")

    # Round-trip through the real loader: an artifact this issuer wrote but the
    # driver cannot load is worse than none.
    reloaded = C1Authorization.load(out)
    assert reloaded.hard_cap_usd == float(
        load_config(REPO_ROOT)["accepted_pricing"]["hard_ceiling_usd"])
    assert reloaded.plan_hash == payload["bound"]["isolation_plan_hash"]
    assert reloaded.allows_phase_a is False
    assert reloaded.allows_beam_search is False

    if args.porcelain:
        print(json.dumps({"authorization_id": payload["authorization_id"],
                          "authorization_sha256": payload["authorization_sha256"],
                          "authorized_session_commit": commit}))
    else:
        bound = payload["bound"]
        print(f"wrote {args.out}")
        print(f"  authorization_id   {payload['authorization_id']}")
        print(f"  authorization_sha  {payload['authorization_sha256']}")
        print(f"  session commit     {commit}")
        print(f"  harness            {bound['c1_harness_digest']} "
              f"({bound['c1_harness_n_files']} files)")
        print(f"  isolation plan     {bound['isolation_plan_hash']}")
        print(f"  preregistration    {bound['execution_preregistration']}")
        print(f"  scoring            {bound['scoring_contract']['contract']} "
              f"{bound['scoring_contract']['digest'][:16]}")
        money = load_config(REPO_ROOT)["accepted_pricing"]
        print(f"  ceiling            ${payload['hard_cap_usd']:.4f}  "
              f"(floor ${money['planning_floor_usd']:.4f}, "
              f"soft stop ${money['soft_stop_usd']:.4f})")
        print(f"  cumulative         "
              f"${float(grant['cumulative_spend_at_approval_usd']):.4f}"
              f" spent of ${money['cumulative_cap_usd']:.4f}")
        print("  ISSUING IS NOT LAUNCHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
