#!/usr/bin/env python3
"""What the PROVIDER billed, set beside what the run records booked.

    # fetch a snapshot (read-only provider calls, $0) and commit it as evidence
    PYTHONPATH=src python scripts/maintenance/consolidation/provider_billing.py \
        fetch --start 2026-06-01 --out logs/budget/provider/runpod_billing_<date>.json
    # compare a committed snapshot with the records
    PYTHONPATH=src python scripts/maintenance/consolidation/provider_billing.py \
        report --snapshot logs/budget/provider/runpod_billing_<date>.json

Every balance in this repository is derived from run records
(`derive_budget.py`), which is right for what each book is PERMITTED to spend
and blind to anything the records do not hold. The provider's own billing
history is the independent check: each billed pod is attributed to the run
record that names it, and what no record names is reported as unattributed --
never assumed to belong to the project, and never assumed not to.

Network-volume billing arrives as daily totals with no volume id, so it can be
compared with booked volume runs only over a window, not per volume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REST = "https://rest.runpod.io/v1"
SCHEMA = "aadistill.provider_billing_snapshot/v1"


def fetch(api_key: str, start: str, end: str) -> dict:
    """Day-bucketed pod and network-volume billing, plus the account's live
    state. Read-only. A non-browser User-Agent is required: the default Python
    one is refused by the provider's edge (HTTP 403, error 1010)."""
    def get(path: str):
        req = urllib.request.Request(
            REST + path, headers={"Authorization": f"Bearer {api_key}",
                                  "User-Agent": "curl/8.0",
                                  "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())

    window = f"bucketSize=day&startTime={start}T00:00:00Z&endTime={end}T23:59:59Z"
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from aadistill.infrastructure.provider import RunPodProvider
    prov = RunPodProvider(api_key)
    return {
        "schema": SCHEMA,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": {"endpoint": REST, "window": window},
        "account": {**prov.account_balance().as_dict(),
                    **{k: v for k, v in prov.account_resources().as_dict().items()
                       if k != "known"}},
        "pods": get(f"/billing/pods?{window}"),
        "networkvolumes": get(f"/billing/networkvolumes?{window}"),
    }


def run_records(root: Path) -> dict[str, list[dict]]:
    """pod id -> the runs that CREATED it: a session record or closeout whose own
    `pod_id`, or one of whose draws, is that pod."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    logs = root / "logs"
    for path in sorted([*logs.rglob("runtime/session.json"),
                        *logs.rglob("closeout/outcome.json")]):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        pods = {doc.get("pod_id")} | {d.get("pod_id") for d in doc.get("draws") or []
                                     if isinstance(d, dict)}
        run_dir = path.parent.parent
        for pid in pods - {None, ""}:
            out[pid][run_dir.as_posix()] = {
                "run_dir": run_dir.relative_to(root).as_posix(),
                "run_id": doc.get("run_id") or run_dir.name}
    return {pid: list(runs.values()) for pid, runs in out.items()}


#: Files that list billed pods BECAUSE they are the billing evidence: the
#: snapshots themselves and every reconciliation written from them. A pod they
#: name is not thereby known to the repository -- committing the list of
#: unattributed pods once made all 53 of them read as "mentioned".
BILLING_EVIDENCE = (":(exclude)logs/budget/provider/**",
                    ":(exclude)**/budget_reconciliation.json",
                    ":(exclude)**/provider_attribution.json")

#: A file that cites a billing snapshot was written FROM the billing evidence --
#: every reconciliation records the snapshot it read (`snapshot_identity`) --
#: so whatever pods it lists, it attributes none of them. This is what keeps a
#: NEW reconciliation, wherever it is committed, from changing attribution: the
#: path list above only knows the reports that existed when it was written (the
#: H1 budget position, committed after it, made all 53 pods read as mentioned).
SNAPSHOT_CITATIONS = ("logs/budget/provider/", SCHEMA)


def _git_grep(root: Path, *args: str) -> list[str]:
    import subprocess
    out = subprocess.run(["git", "grep", *args], cwd=root, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line]


def mentioned(root: Path, ids: list[str]) -> set[str]:
    """Which ids appear in a tracked file that is not billing evidence and does
    not cite a billing snapshot. A pod a record mentions without having created
    it (a hand-written ledger line, an older layout) is at least not unknown to
    the repository; a pod only the billing evidence or a report derived from it
    names stays unattributed."""
    if not ids:
        return set()
    pattern = sum((["-e", i] for i in ids), [])
    files = _git_grep(root, "-l", "-F", *pattern, "--", ".", *BILLING_EVIDENCE)
    if not files:
        return set()
    literal = [f":(literal){f}" for f in files]
    derived = set(_git_grep(root, "-l", "-F",
                            *sum((["-e", c] for c in SNAPSHOT_CITATIONS), []), "--", *literal))
    keep = [f":(literal){f}" for f in files if f not in derived]
    return set(_git_grep(root, "-h", "-o", "-F", *pattern, "--", *keep)) if keep else set()


def billed_by_pod(snapshot: dict) -> dict[str, dict]:
    pods: dict[str, dict] = {}
    for row in snapshot["pods"]:
        rec = pods.setdefault(row["podId"], {"usd": 0.0, "days": []})
        rec["usd"] += float(row["amount"])
        rec["days"].append(row["time"][:10])
    return pods


def volumes_by_day(snapshot: dict) -> dict[str, float]:
    return {row["startDate"][:10]: float(row["amount"])
            for row in snapshot["networkvolumes"]}


def snapshot_identity(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def report(root: Path, snapshot_path: Path) -> dict:
    """Project-wide: provider total against the derived project cumulative,
    and every billed pod no record names."""
    snap = json.loads(snapshot_path.read_text())
    pods = billed_by_pod(snap)
    created = run_records(root)
    loose = [p for p in pods if p not in created]
    named = mentioned(root, loose)
    unattributed = sorted(
        ({"pod_id": p, "usd": round(pods[p]["usd"], 4), "first_day": min(pods[p]["days"])}
         for p in loose if p not in named), key=lambda r: (r["first_day"], r["pod_id"]))
    by_month: dict[str, float] = defaultdict(float)
    for r in unattributed:
        by_month[r["first_day"][:7]] += r["usd"]
    sys.path.insert(0, str(root / "scripts"))
    from maintenance.consolidation.derive_budget import derive
    project = derive(root)["project"]
    pod_total = round(sum(p["usd"] for p in pods.values()), 4)
    vol_total = round(sum(volumes_by_day(snap).values()), 4)
    return {
        "snapshot": snapshot_identity(snapshot_path, root),
        "provider_billed_usd": {"pods": pod_total, "network_volumes": vol_total,
                                "total": round(pod_total + vol_total, 4)},
        "records_cumulative_usd": project.get("cumulative_spend_usd"),
        "provider_minus_records_usd": round(
            pod_total + vol_total - float(project.get("cumulative_spend_usd") or 0), 4),
        "pods_billed": len(pods),
        "pods_created_by_a_run_record": sum(1 for p in pods if p in created),
        "pods_only_mentioned": sorted(p for p in loose if p in named),
        "pods_unattributed": unattributed,
        "unattributed_usd_by_first_month": {k: round(v, 4)
                                            for k, v in sorted(by_month.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    f.add_argument("--out", required=True)
    f.add_argument("--runpod-config", default="~/.runpod/config.toml")
    r = sub.add_parser("report")
    r.add_argument("--snapshot", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from aadistill.infrastructure.provider import read_api_key
        key = os.environ.get("RUNPOD_API_KEY") or read_api_key(
            os.path.expanduser(a.runpod_config))
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fetch(key, a.start, a.end), indent=1) + "\n")
        print(out)
        return 0
    print(json.dumps(report(REPO_ROOT, Path(a.snapshot).resolve()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
