#!/usr/bin/env python3
"""Issue A3's ONE-USE authorization from its grant. Derives; never transcribes.

    PYTHONPATH=src:scripts python scripts/autoinit/issue_a3_authorization.py \
        --grant logs/stages/stage-1/phase_a3/runs/<run>/governance/grant.json \
        --out   logs/stages/stage-1/phase_a3/runs/<run>/governance/authorization.json

The grant says what a maintainer decided. Everything else — the ceiling, the
stages, the harness digest, the design hash, the identities — is derived from
the tree this authorization will bind, and **refused if the grant asserts it**.

Three refusals that exist because the mistake has been made:

* a DIRTY TREE. The pod checks out the session commit, so an uncommitted edit
  is code the authorization did not measure and the pod will never run.
* a FUTURE-DATED grant. C2-behavioural's attempt3 carried a local-timezone
  date in a UTC field for a session that ran the day before, and nothing
  detected it because nothing read it.
* an artifact this issuer writes but the real loader cannot read. The payload
  is round-tripped through `A3Authorization.load` before it is accepted.
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

from aadistill.governance.grant import (  # noqa: E402
    GrantRefused, refuse_a_future_dated_grant,
)
from experiments.phase_a3.a3_authorization import A3Authorization  # noqa: E402
from experiments.phase_a3.a3_authorization_payload import (  # noqa: E402
    A3AuthorizationRefused, build_a3_authorization_payload,
)

DEFAULT_GRANT = "logs/budget/approvals/autoinit_a3_grant.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grant", default=DEFAULT_GRANT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-commit", default=None,
                    help="defaults to the current HEAD")
    ap.add_argument("--porcelain", action="store_true")
    args = ap.parse_args(argv)

    grant_path = REPO_ROOT / args.grant
    if not grant_path.is_file():
        print(f"no grant at {args.grant}", file=sys.stderr)
        return 2
    grant = json.loads(grant_path.read_text())

    if git("status", "--porcelain").strip():
        print("the working tree is dirty; commit before issuing so the "
              "authorization binds what the pod will actually check out",
              file=sys.stderr)
        return 2
    commit = args.session_commit or git("rev-parse", "HEAD").strip()

    try:
        refuse_a_future_dated_grant(grant.get("granted_utc", ""))
    except GrantRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    try:
        payload = build_a3_authorization_payload(
            grant=grant, session_commit=commit,
            granted_utc=datetime.now(timezone.utc).isoformat(),
            repo_root=REPO_ROOT, grant_path=args.grant)
    except A3AuthorizationRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True,
                              ensure_ascii=False) + "\n")

    #: Round-trip through the REAL loader. An artifact this issuer wrote but
    #: the driver cannot load is worse than none.
    reloaded = A3Authorization.load(out)
    assert reloaded.hard_cap_usd == payload["hard_cap_usd"]
    assert reloaded.authorizes_a3 is True
    assert reloaded.authorizes_c1_isolation is False
    assert reloaded.authorizes_c3_isolation is False
    assert reloaded.allows_beam_search is False
    assert reloaded.allows_arm_elimination is False
    assert reloaded.allows_control_retraining is False
    assert reloaded.allows_on_pod_decision is False

    if args.porcelain:
        print(json.dumps({
            "authorization_id": payload["authorization_id"],
            "authorization_sha256": payload["authorization_sha256"],
            "authorized_session_commit": commit}))
        return 0

    b = payload["bound"]
    lp, d = b["live_pricing"], b["design"]
    print(f"issued {payload['authorization_id']} at {commit[:12]}")
    print(f"  money      expected ${lp['expected_usd']:.4f}, hard "
          f"${lp['hard_ceiling_usd']:.4f} on {lp['gpu']} at "
          f"${lp['rate_usd_per_hour']}/h, {len(lp['limits_checked'])} limits")
    print(f"  design     {d['design_sha256'][:12]}, {d['arms']} arm x "
          f"{len(d['seeds'])} seeds = {d['probes']} probes, aggregation "
          f"{'off pod' if d['aggregation_off_pod'] else 'ON POD'}")
    print(f"  identities parent {b['identities']['parent'][:12]}, incumbent "
          f"{b['identities']['incumbent'][:12]}, a_bsz3 "
          f"{b['identities']['a_bsz3']} (measured)")
    print(f"  controls   {b['controls']['source_run']} seeds "
          f"{b['controls']['seeds']}, retrained "
          f"{b['controls']['retrained']}, hashes verified")
    print(f"  harness    {payload['harness_source_digest'][:12]} over "
          f"{len(payload['harness_source_files'])} files")
    print(f"  stages     {list(payload['authorized_stages'])}")
    print(f"  sha256     {payload['authorization_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
