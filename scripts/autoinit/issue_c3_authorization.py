#!/usr/bin/env python3
"""Issue the ONE-USE Phase-C3 authorization. Zero cost; launches nothing.

    PYTHONPATH=src python scripts/autoinit/issue_c3_authorization.py \
        --grant logs/budget/approvals/autoinit_c3_grant.json

Same contract as the Phase-A, Phase-B, continuation and C1 issuers, and the
same reason for it: the grant is an **input**, not a constant. A one-use
maintainer decision living in executable source goes stale silently and still
reads as though it applies.

**This cannot become a C1, Phase-A or search grant.**
`C3Authorization.allows_phase_a`, `.allows_beam_search`,
`.allows_arm_elimination` and `.authorizes_c1_isolation` are hard `False`
properties with no field to set, and `load` refuses any artifact whose schema
is not the C3 one. That is a property of the type, not a promise here.

**The envelope is not the grant.** The 2026-09-28 amendment set a `$30.00`
per-session envelope and made the provider price a live input. This issuer
refuses a grant that asks for anything other than the ceiling DERIVED from
the live `securePrice` record -- issuing at the envelope when the derived
figure is `$22.1452` would be a 36% over-authorization that every downstream
gate would accept.

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
#: NOT `scripts/experiments/phase_c3` on sys.path -- the sibling phase_c1
#: directory holds `packaging.py`, which shadows the third-party `packaging`
#: distribution and breaks the next transformers import in the process.
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.phase_c3.authorization import C3Authorization  # noqa: E402
from experiments.phase_c3.authorization_payload import (  # noqa: E402
    C3AuthorizationRefused, build_c3_authorization_payload,
)

#: The RUN's grant, not the repository-level one. `grant_provenance_gate`
#: resolves the recorded path and requires it to be this run's
#: governance/grant.json: a grant belongs to one attempt.
DEFAULT_GRANT = "logs/budget/approvals/autoinit_c3_grant.json"
DEFAULT_OUT = "logs/budget/approvals/autoinit_c3_authorization.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grant", default=DEFAULT_GRANT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--session-commit", default=None,
                    help="defaults to the current HEAD")
    ap.add_argument("--porcelain", action="store_true")
    args = ap.parse_args(argv)

    grant_path = REPO_ROOT / args.grant
    if not grant_path.is_file():
        print(f"no grant at {args.grant}", file=sys.stderr)
        return 2
    grant = json.loads(grant_path.read_text())

    #: A dirty tree cannot be authorized: the pod checks out the session
    #: commit, so an uncommitted edit is code the authorization did not
    #: measure and the pod would never run.
    if git("status", "--porcelain").strip():
        print("the working tree is dirty; commit before issuing so the "
              "authorization binds what the pod will actually check out",
              file=sys.stderr)
        return 2
    commit = args.session_commit or git("rev-parse", "HEAD").strip()

    try:
        payload = build_c3_authorization_payload(
            grant=grant, session_commit=commit,
            granted_utc=datetime.now(timezone.utc).isoformat(),
            repo_root=REPO_ROOT, grant_path=args.grant)
    except C3AuthorizationRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True,
                              ensure_ascii=False) + "\n")

    #: Round-trip through the REAL loader: an artifact this issuer wrote but
    #: the driver cannot load is worse than none.
    reloaded = C3Authorization.load(out)
    assert reloaded.hard_cap_usd == payload["hard_cap_usd"]
    assert reloaded.authorizes_c3_isolation is True
    assert reloaded.authorizes_c1_isolation is False
    assert reloaded.allows_beam_search is False
    assert reloaded.allows_arm_elimination is False

    if args.porcelain:
        print(json.dumps({
            "authorization_id": payload["authorization_id"],
            "authorization_sha256": payload["authorization_sha256"],
            "authorized_session_commit": commit}))
        return 0

    b = payload["bound"]
    lp = b["live_pricing"]
    print(f"wrote {args.out}")
    print(f"  authorization_id   {payload['authorization_id']}")
    print(f"  authorization_sha  {payload['authorization_sha256']}")
    print(f"  session commit     {commit}")
    print(f"  harness            {payload['c3_harness_digest']} "
          f"({payload['c3_harness_n_files']} files)")
    print(f"  session contract   {b['session_contract_hash']}")
    print(f"  preregistration    {b['preregistration_sha256']}")
    print(f"  arms               {b['arm_ids']}")
    print(f"  seeds              {b['recovery_seeds']} "
          f"(bootstrap {b['bootstrap_seed']})")
    print(f"  primary contrast   {b['primary_contrast']}")
    print(f"  battery            {b['battery']['asset_id']} "
          f"{b['battery']['content_sha256'][:16]}")
    print(f"  scoring            {b['scoring_contract']['contract']} "
          f"{b['scoring_contract']['digest'][:16]}")
    print(f"  live securePrice   ${lp['gpu_rate_usd_per_hour']:.4f}/h "
          f"-> billed ${lp['billed_rate_usd_per_hour']:.6f}/h")
    print(f"  ceiling            ${payload['hard_cap_usd']:.4f} DERIVED "
          f"(envelope ${lp['per_session_envelope_usd']:.2f} — not the grant)")
    print(f"  expected           ${payload['expected_usd']:.4f}")
    print(f"  scope              {payload['_scope']['probes']} probes = "
          f"{payload['_scope']['arms']} arms x {payload['_scope']['seeds']} seeds")
    print("  ISSUING IS NOT LAUNCHING. C4 is not authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
