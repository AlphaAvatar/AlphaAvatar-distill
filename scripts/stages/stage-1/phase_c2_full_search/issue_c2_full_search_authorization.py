#!/usr/bin/env python3
"""Issue the full-search authorization from a maintainer grant.

    PYTHONPATH=src:scripts python \
      scripts/autoinit/issue_c2_full_search_authorization.py \
      --grant <path> --run-id attempt1

Thin by design. Everything that DECIDES anything lives in
`experiments.phase_c2.full_search_authorization`, so the CLI and a test
candidate are refused for the same reasons by the same code. Three effects stay
here because they are effects rather than computations: refusing a dirty tree,
reading the clock, and re-quoting the live rate.

**Not any other C2 issuer.** `issue_c2_authorization.py` asserts
`authorizes_c2_search1 is True` and prices Search-1's restricted beam;
`issue_c2_baseline_completion_authorization.py` prices one rebuild. A
full-search grant flowing through either would be validated against another
session's code, permission and price -- and would pass.

**The rate is re-quoted here, and the ceiling is not.** The pricing record's
basis is planning evidence with a date on it, so `--verify-rate` queries the
provider and refuses when the live securePrice no longer matches what the grant
approved against. The ceiling itself is DERIVED by the assembler from the
pricing record's minutes at the grant's rate -- same work, current price. Beam
width 6 is the standing design and nothing in this tool can narrow it.

**The chronology this belongs to**, which the tool enforces rather than
documents: the grant is committed; on that clean tree the FULL-SEARCH
`launch_bound` sweep runs; only the readiness record is committed; on the
resulting clean HEAD this issuer runs. So the readiness check inside the
assembler is a refusal, and the clean-tree check here defaults ON -- C1 attempt
15 aborted at `$0` because it was opt-in and not passed, leaving the authorized
base predating both the record and the artifact.

ISSUING IS NOT LAUNCHING.
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

from experiments.phase_c2 import full_search as FSG  # noqa: E402
from experiments.phase_c2.full_search_authorization import (  # noqa: E402
    FullSearchAuthorizationRefused, build_payload, load_config,
)


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def authorization_path_for(run_id: str, stage_id: str) -> str:
    from experiments.run_layout import rel_run_dir

    return (f"{rel_run_dir('phase_c2_full_search', run_id, stage_id)}"
            "/governance/authorization.json")


def live_secure_price(gpu: str) -> float:
    """The provider's CURRENT securePrice for the card, or refuse.

    `securePrice`, never `communityPrice`: `pod create` provisions secure, and
    reading the community floor has under-reported this project's cost twice --
    once in a figure a maintainer then acted on.

    A failure here is a REFUSAL rather than a fallback to the priced basis. An
    issuer that could not reach the provider does not know what a session will
    cost, and issuing against a remembered number is how a ceiling comes to
    describe last week's price.
    """
    import os

    from aadistill.infrastructure.provider import RunPodProvider, read_api_key

    key = os.environ.get("RUNPOD_API_KEY") or read_api_key(
        os.path.expanduser("~/.runpod/config.toml"))
    try:
        doc = RunPodProvider(key)._gql(
            'query { gpuTypes(input:{id:"%s"}) { id securePrice } }' % gpu)
        rows = (doc.get("data") or {}).get("gpuTypes") or []
        price = rows[0]["securePrice"]
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(
            f"cannot re-quote {gpu} securePrice ({type(exc).__name__}: {exc}). "
            "Refusing to issue against the pricing record's remembered basis: "
            "an hour-old price is not a price.") from None
    if price is None:
        raise SystemExit(f"the provider quotes no securePrice for {gpu}")
    return round(float(price), 4)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grant", required=True, type=Path)
    #: REQUIRED. The artifact is owned by the run from the start, so a later
    #: issuance cannot overwrite it and a launcher cannot find somebody else's.
    ap.add_argument("--run-id", required=True,
                    help="the run this authorization is issued for")
    ap.add_argument("--stage-id", default=None,
                    help="defaults to the stage the experiment's config declares")
    ap.add_argument("--gpu", default="NVIDIA L40S",
                    help="the card whose live securePrice is re-quoted")
    #: DEFAULT ON, both of them. See the module docstring.
    ap.add_argument("--skip-rate-quote", action="store_true",
                    help="do not re-quote the live rate. Almost never right: "
                         "the ceiling is derived from the grant's rate, and a "
                         "grant whose rate is stale prices the wrong session.")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="issue against a dirty tree. Almost never right: the "
                         "authorized session commit must describe exactly what "
                         "the pod checks out.")
    ap.add_argument("--porcelain", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(REPO_ROOT)
    stage_id = args.stage_id or cfg["stage_id"]

    grant = json.loads(args.grant.read_text())

    if not args.allow_dirty:
        dirty = git("status", "--porcelain")
        if dirty:
            raise SystemExit(
                "refusing to issue against a dirty tree:\n" + dirty
                + "\nThe authorized session commit must describe exactly what "
                  "the pod checks out.")

    #: The rate the grant approved against, checked against the live one BEFORE
    #: anything is written. The assembler derives the ceiling from the grant's
    #: rate; this is what makes that rate current rather than remembered.
    approved_rate = None
    live_rate = None
    if not args.skip_rate_quote:
        money = grant.get("approved_money") or {}
        if "price_basis_usd_per_hour" not in money:
            raise SystemExit(
                "the grant states no approved_money.price_basis_usd_per_hour, "
                "so there is no rate to check the live quote against")
        approved_rate = round(float(money["price_basis_usd_per_hour"]), 4)
        live_rate = live_secure_price(args.gpu)
        if live_rate != approved_rate:
            raise SystemExit(
                f"the live {args.gpu} securePrice is ${live_rate}/h and the "
                f"grant approved against ${approved_rate}/h. The ceiling is "
                "derived from the rate, so issuing now would authorize a plan "
                "priced at a rate nobody is charging.\n"
                f"  at ${live_rate}/h the beam-6 ceiling is "
                f"${FSG.derive_ceiling_usd(live_rate, REPO_ROOT):.4f}\n"
                "Re-derive the grant's approved_money at the live rate. Do NOT "
                "narrow beam width 6 to absorb a price change.")

    commit = git("rev-parse", "HEAD")
    try:
        payload = build_payload(
            grant=grant, session_commit=commit,
            granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            run_id=args.run_id, stage_id=stage_id, repo_root=REPO_ROOT,
            grant_path=str(args.grant))
    except FullSearchAuthorizationRefused as exc:
        raise SystemExit(f"refusing to issue: {exc}") from None

    rel = authorization_path_for(args.run_id, stage_id)
    out = REPO_ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    #: An issuance never writes over an existing authorization. One-use means
    #: one artifact; replacing one in place is how a consumed authorization
    #: becomes indistinguishable from a fresh one.
    if out.exists():
        raise SystemExit(
            f"{rel} already exists. An authorization is one-use: issue into a "
            "new run, or move the existing artifact aside deliberately.")
    out.write_text(json.dumps(payload, indent=1) + "\n")

    #: Read back through the TYPE that will load it on the pod. An artifact this
    #: loader refuses is not an authorization, and discovering that here costs
    #: nothing.
    reloaded = FSG.FullSearchAuthorization.load(out)
    for claim, ok in (("authorizes_c2_full_search",
                       reloaded.authorizes_c2_full_search),
                      ("refuses Search-1", not reloaded.authorizes_c2_search1),
                      ("refuses baseline completion",
                       not reloaded.authorizes_c2_baseline_completion),
                      ("refuses behavioural selection",
                       not reloaded.authorizes_behavioural_selection)):
        if not ok:
            raise SystemExit(f"the issued artifact fails its own check: {claim}")

    bound = payload["bound"]
    if args.porcelain:
        print(json.dumps({"path": rel,
                          "authorization_sha256": payload["authorization_sha256"],
                          "session_commit": commit,
                          "live_rate_usd_per_hour": live_rate,
                          "hard_cap_usd": payload["hard_cap_usd"]}))
        return 0
    print(f"wrote {rel}")
    print(f"  authorization_sha  {payload['authorization_sha256']}")
    print(f"  session commit     {commit}")
    print(f"  search closure     {bound['full_search_harness_digest']} "
          f"({bound['full_search_harness_n_files']} files, derived)")
    print(f"  plan               {bound['full_search_plan_hash']}")
    print(f"  pricing            {bound['full_search_pricing_sha256']}")
    print(f"  joint space        {bound['joint_space_total_leaves']} leaves "
          f"({bound['joint_space_decomposed_leaves']} decomposed) over "
          f"{len(bound['joint_space_allowed_impls'])} implementations, "
          f"profiles {'unpinned' if bound['joint_space_impl_profiles_are_unpinned'] else 'PINNED'}")
    print(f"  beam               width {bound['beam_width']} warmup "
          f"{bound['beam_warmup_levels']} ({bound['beam_schedule_id']})")
    print(f"  ranking policy     {bound['ranking_policy_hash'][:16]}…")
    if live_rate is not None:
        print(f"  live rate          ${live_rate}/h (re-quoted, matches the grant)")
    print(f"  ceiling            ${payload['hard_cap_usd']:.4f}  "
          f"(expected ${bound['expected_usd']:.4f} at "
          f"${bound['price_per_hour_usd']}/h over "
          f"{bound['hard_ceiling_minutes']} ceiling minutes)")
    print(f"  readiness          {payload['readiness']['path']} "
          f"{payload['readiness']['kind']}")
    print(f"  terminates at      {payload['terminates_at']}")
    print("  authorizes ONE full joint beam and a committed Top-5; NOT "
          "Search-1, NOT a baseline rebuild, NOT behavioural selection.")
    print("  ISSUING IS NOT LAUNCHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
