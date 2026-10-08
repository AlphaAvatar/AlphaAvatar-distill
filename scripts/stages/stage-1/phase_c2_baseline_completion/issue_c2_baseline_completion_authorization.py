#!/usr/bin/env python3
"""Issue the baseline-completion authorization from a maintainer grant.

    PYTHONPATH=src:scripts python \
      scripts/autoinit/issue_c2_baseline_completion_authorization.py \
      --grant <path> --run-id attempt5

Thin by design. Everything that DECIDES anything lives in
`experiments.phase_c2.baseline_completion_authorization`, so the CLI and a test
candidate are refused for the same reasons by the same code. Two effects stay
here because they are effects rather than computations: refusing a dirty tree,
and reading the clock.

**Not `issue_c2_authorization.py`.** That issuer asserts
`authorizes_c2_search1 is True`, derives the Search-1 closure and checks the
$15.0446 beam ceiling. A completion grant flowing through it would be validated
against another session's code, permission and price -- and would pass.

**The chronology this belongs to**, which the tool enforces rather than
documents: the grant is committed; on that clean tree the COMPLETION
`launch_bound` sweep runs; only the readiness record is committed; on the
resulting clean HEAD this issuer runs. So the readiness check inside the
assembler is a refusal, and the clean-tree check here defaults ON -- C1 attempt
15 aborted at `$0` because it was opt-in and not passed, leaving the authorized
base predating both the record and the artifact.
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

from experiments.phase_c2 import baseline_completion as BC  # noqa: E402
from experiments.phase_c2.baseline_completion_authorization import (  # noqa: E402
    CompletionAuthorizationRefused, build_payload, load_config,
)


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         cwd=REPO_ROOT)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def authorization_path_for(run_id: str, stage_id: str) -> str:
    from experiments.run_layout import rel_run_dir

    return (f"{rel_run_dir('phase_c2_baseline_completion', run_id, stage_id)}"
            "/governance/authorization.json")


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
    #: DEFAULT ON. See the module docstring.
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

    commit = git("rev-parse", "HEAD")
    try:
        payload = build_payload(
            grant=grant, session_commit=commit,
            granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            run_id=args.run_id, stage_id=stage_id, repo_root=REPO_ROOT,
            grant_path=str(args.grant))
    except CompletionAuthorizationRefused as exc:
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
    reloaded = BC.BaselineCompletionAuthorization.load(out)
    if not reloaded.authorizes_c2_baseline_completion:
        raise SystemExit("the issued artifact does not authorize completion")
    if reloaded.authorizes_c2_search1:
        raise SystemExit("the issued artifact claims it can authorize a beam")

    if args.porcelain:
        print(json.dumps({"path": rel,
                          "authorization_sha256": payload["authorization_sha256"],
                          "session_commit": commit}))
        return 0
    print(f"wrote {rel}")
    print(f"  authorization_sha  {payload['authorization_sha256']}")
    print(f"  session commit     {commit}")
    print(f"  completion closure {payload['bound']['completion_harness_digest']} "
          f"({payload['bound']['completion_harness_n_files']} files, derived)")
    print(f"  plan               {payload['bound']['completion_plan_hash']}")
    print(f"  pricing            {payload['bound']['completion_pricing_sha256']}")
    print(f"  baseline B         spec {payload['bound']['baseline_spec_hash'][:16]} "
          f"artifact {payload['bound']['baseline_artifact_digest'][:16]}")
    print(f"  frozen candidates  {payload['bound']['frozen_candidates_self_sha256']}")
    print(f"  ceiling            ${payload['hard_cap_usd']:.4f}  "
          f"(expected ${payload['bound']['expected_usd']:.4f} at "
          f"${payload['bound']['price_per_hour_usd']}/h)")
    print(f"  readiness          {payload['readiness']['record']} "
          f"{payload['readiness']['record_kind']}")
    print("  authorizes baseline completion; NOT Search-1.")
    print("  ISSUING IS NOT LAUNCHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
