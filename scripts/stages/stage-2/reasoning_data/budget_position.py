#!/usr/bin/env python3
"""H1's budget position: the records, the provider, and every unexplained dollar. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.budget_position \
        --snapshot logs/budget/provider/runpod_billing_<stamp>.json

Reads a COMMITTED read-only provider snapshot (`provider_billing.py fetch`) and
the derived project books (`derive_budget.py`), then attributes every billed pod
no run record created, from existing evidence only, in three passes:

1. the tracked tree, excluding the billing evidence itself
   (`provider_billing.mentioned`);
2. this project's untracked launch logs on the dev box (`artifacts/`,
   `/home/ecs-user/aad-scratch`), text files only;
3. git history over the window the remaining pods were billed in -- records the
   2026-07-31 consolidation removed still name the pods they ran.

A pod a project record names is project ACTIVITY; whether its dollars are inside
a maintainer-stated anchor is a separate question this cannot answer, so the
money is bounded, never assumed: records (lower) and provider (upper). Writes
`logs/stages/stage-2/reasoning_data/budget/h1_budget_position.json`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from stages.reasoning_data import assets as A

from maintenance.consolidation import provider_billing as PB

OUT = A.LOG_DIR / "budget/h1_budget_position.json"
LAUNCH_ROOTS = (A.REPO / "artifacts", Path("/home/ecs-user/aad-scratch"))
TEXT = ("*.json", "*.jsonl", "*.md", "*.txt", "*.log", "*.out", "*.sh", "pod_id")


def untracked_mentions(ids: list[str]) -> dict[str, list[str]]:
    if not ids:
        return {}
    pat = A.LOG_DIR / ".ids.tmp"
    pat.write_text("\n".join(ids) + "\n")
    try:
        args = ["grep", "-rlF", "-f", str(pat)] + [f"--include={g}" for g in TEXT]
        files = subprocess.run(args + [str(r) for r in LAUNCH_ROOTS if r.is_dir()],
                               capture_output=True, text=True).stdout.split()
    finally:
        pat.unlink()
    out: dict[str, list[str]] = {}
    for f in files:
        text = Path(f).read_text(errors="ignore")
        for i in ids:
            if i in text:
                out.setdefault(i, []).append(f)
    return {i: sorted(v)[:3] for i, v in out.items()}


def history_mentions(ids: list[str], first: str, last: str) -> dict[str, dict]:
    if not ids:
        return {}
    since = (date.fromisoformat(first) - timedelta(days=7)).isoformat()
    until = (date.fromisoformat(last) + timedelta(days=30)).isoformat()
    revs = subprocess.run(["git", "-C", str(A.REPO), "rev-list", "--all", f"--since={since}",
                           f"--until={until}"], capture_output=True, text=True).stdout.split()
    out: dict[str, dict] = {}
    left = set(ids)
    for c in reversed(revs):                      # oldest first: the record that ran it
        if not left:
            break
        hits = subprocess.run(["git", "-C", str(A.REPO), "grep", "-o", "-F",
                               *sum((["-e", i] for i in sorted(left)), []), c],
                              capture_output=True, text=True).stdout.splitlines()
        for h in hits:
            _, path, pid = h.split(":", 2)
            if pid in left:
                out[pid] = {"commit": c, "file": path}
                left.discard(pid)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args(argv)
    snap_path = A.REPO / args.snapshot
    rep = PB.report(A.REPO, snap_path)
    snap = A.read_json(snap_path)
    unnamed = {r["pod_id"]: r for r in rep["pods_unattributed"]}
    launch = untracked_mentions(sorted(unnamed))
    rest = sorted(set(unnamed) - set(launch))
    days = [unnamed[p]["first_day"] for p in rest] or ["2026-01-01"]
    hist = history_mentions(rest, min(days), max(days))
    still = sorted(set(rest) - set(hist))

    def usd(ids):
        return round(sum(unnamed[i]["usd"] for i in ids), 4)

    recon = A.read_json(A.S0 / "closeout/budget_reconciliation.json")
    cap = recon["limits_are_different_things"]["project_cumulative_cap_usd"]["value"]
    records = rep["records_cumulative_usd"]
    provider = rep["provider_billed_usd"]["total"]
    account = snap["account"]
    doc = {
        "schema": "aadistill.stage2.reasoning_data.budget_position/v1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot": rep["snapshot"],
        "provider_queried_utc": snap["fetched_utc"],
        "categories": {
            "confirmed_project_spend_records_usd": records,
            "confirmed_non_project_spend_usd": 0.0,
            "_confirmed_non_project": "none established: the sibling AlphaAvatar checkout and the out-of-tree store name none of the unattributed pods",
            "provider_billed_all_time_usd": provider,
            "provider_minus_records_usd": rep["provider_minus_records_usd"],
            "by_period": recon["project"]["by_period"],
            "open_resource_liabilities": {"pods": len(account["pods"]), "network_volumes": len(account["network_volumes"]),
                                          "spend_per_hour_usd": account["current_spend_per_hr"], "usd": 0.0},
            "provider_balance_usd": account["client_balance_usd"],
            "_provider_balance": "money on the account, not a grant and not project headroom",
        },
        "pods_no_run_record_created": {
            "count": len(unnamed), "usd": usd(unnamed),
            "named_by_untracked_project_launch_logs": {"count": len(launch), "usd": usd(launch),
                                                       "pods": {p: {"usd": unnamed[p]["usd"], "first_day": unnamed[p]["first_day"], "files": launch[p]} for p in sorted(launch)}},
            "named_by_project_records_in_git_history": {"count": len(hist), "usd": usd(hist),
                                                        "pods": {p: {"usd": unnamed[p]["usd"], "first_day": unnamed[p]["first_day"], **hist[p]} for p in sorted(hist)}},
            "unnamed": {"count": len(still), "usd": usd(still),
                        "first_day": min((unnamed[p]["first_day"] for p in still), default=None),
                        "last_day": max((unnamed[p]["first_day"] for p in still), default=None),
                        "pods": still, "status": "OPEN"},
            "_reading": "named = this project's own records ran the pod (activity confirmed); it does not show whether those dollars sit inside the maintainer-stated 2026-09-11 anchor",
        },
        "bounds": {
            "project_cap_usd": cap,
            "headroom_if_records_are_complete_usd": round(cap - records, 4),
            "headroom_if_all_provider_billing_is_project_usd": round(cap - provider, 4),
            "conservative_headroom_usd": round(min(cap - records, cap - provider), 4),
        },
        "for_the_maintainer": [
            f"were the {len(still)} unnamed pods (${usd(still)}, billed {min(days)} to {max(days)}) this project's? Nothing in the repository, its history or the dev box names them",
            "does the 2026-09-11 anchor ($267.8998, maintainer-stated) include all account spend before that day? The provider billed $37.93 more through it",
            "until both are answered, any H1 spend needs a cap or envelope that covers the conservative bound (the cap is exceeded by the provider total)",
        ],
        "authorizes": "nothing",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), doc["bounds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
