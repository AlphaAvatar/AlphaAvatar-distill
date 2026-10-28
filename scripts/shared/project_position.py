#!/usr/bin/env python3
"""The project's committed spend as the provider bills it, engineering-campaign subruns included ($0).

    # the position and whether requested hard caps fit it, from a snapshot or read live (read-only)
    PYTHONPATH=src:scripts python scripts/shared/project_position.py \\
        --snapshot logs/budget/provider/runpod_billing_<ts>.json --request A-E=3.50 --request B-E=3.00
    PYTHONPATH=src:scripts python scripts/shared/project_position.py --live --request A-E=3.50

`mixed_governance.position` is the rule every issuer since the mixed
calibration used: provider-billed pods and volumes, plus recorded sessions the
provider has not posted yet, plus live volumes' reservations. It finds
unposted sessions only in ``runs/*/runtime/session.json``. An engineering
campaign records its paid sessions as subruns of a ``campaign.json``
(``validations/``, ``pilots/``, ``investigations/``), which it never read, so an
engineering pod still inside the posting window was missing from "committed"
(accounting audit 2026-10-27, finding 3). ``position`` here is that rule plus
the campaign subruns:

* every subrun with a pod must have a CONFIRMED teardown, whatever its age and
  whether or not the provider has billed it: its own ``teardown_confirmed:
  true``, or an entry in a teardown reconciliation (below). Anything else
  refuses the position -- a missing bill is never read as a pod that is gone
  (maintainer, 2026-10-28), and neither is a posted one;
* a subrun whose pod the provider has billed is in the bill already;
* one the provider has not billed, ended less than ``POSTING_WINDOW_HOURS``
  ago, is a liability at its booked ``cost_usd``;
* one older than the window was never charged; it is listed, not counted.

A teardown reconciliation (``logs/budget/reconciliations/*.json``, schema
``RECONCILIATION_SCHEMA``, written by ``--write-reconciliation``) confirms the
teardown of subruns whose records predate the field: it cites a committed
provider snapshot by path and sha256, and lists each pod with the snapshot's
billed amount. It is re-checked at every read -- the snapshot must still hash
to the cited value, its resource listing must have been read, and every listed
pod must be absent from it -- so a hand-edited or stale entry refuses rather
than confirms.

A subrun's end is its ``ended_utc`` when it records one (new campaign records
should), else the commit that first recorded its pod id, else the record's
modification time. The project cap is the newest grant's
(`derive_budget.latest_project_cap`) unless the caller passes one. The account
balance is reported beside the position and never read as headroom.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.governance.authorization import AuthorizationError  # noqa: E402

CAMPAIGN_SCHEMA = "aadistill.engineering_campaign/v1"
RECONCILIATION_SCHEMA = "aadistill.campaign_teardown_reconciliation/v1"
RECONCILIATIONS = "logs/budget/reconciliations"


def _mg():
    from stages.calibration_volume import mixed_governance as MG
    return MG


def _utc(s: str | None) -> datetime | None:
    if not s:
        return None
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _ended(root: Path, path: Path, sub: dict) -> tuple[datetime, str]:
    if sub.get("ended_utc"):
        return _utc(sub["ended_utc"]), "subrun ended_utc"
    try:
        out = subprocess.run(["git", "log", "--reverse", "--format=%cI", "-S", str(sub["pod_id"]), "--",
                              str(path.relative_to(root))], cwd=root, capture_output=True, text=True,
                             timeout=60).stdout.split()
    except (OSError, subprocess.SubprocessError):
        out = []
    if out:
        return _utc(out[0]), "the commit that first recorded the pod"
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc), "the record's modification time"


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconciled_teardowns(root: Path = REPO) -> dict[str, dict]:
    """pod id -> the reconciliation entry that confirms its teardown, each re-checked
    against the snapshot it cites (module docstring). Refuses on any entry that does not hold."""
    root = Path(root)
    out: dict[str, dict] = {}
    for path in sorted((root / RECONCILIATIONS).glob("*.json")) if (root / RECONCILIATIONS).is_dir() else []:
        doc = json.loads(path.read_text())
        if doc.get("schema") != RECONCILIATION_SCHEMA:
            continue
        rel = str(path.relative_to(root))
        snap_p = root / doc["snapshot"]["path"]
        if not snap_p.is_file() or _sha256(snap_p) != doc["snapshot"]["sha256"]:
            raise AuthorizationError(f"{rel}: its snapshot {doc['snapshot']['path']} is missing or no longer the "
                                     "cited bytes; the teardowns it confirms are unconfirmed")
        snap = json.loads(snap_p.read_text())
        acct = snap.get("account") or {}
        if acct.get("resources_known") is not True or not isinstance(acct.get("pods"), list):
            raise AuthorizationError(f"{rel}: its snapshot's resource listing was not read; it confirms nothing")
        live = {p.get("id") for p in acct["pods"]}
        for row in doc["teardowns"]:
            if row["pod_id"] in live:
                raise AuthorizationError(f"{rel}: pod {row['pod_id']} is in its snapshot's inventory; not torn down")
            out[row["pod_id"]] = {"record": rel, "snapshot_fetched_utc": snap.get("fetched_utc")}
    return out


def campaign_subruns(root: Path = REPO, *, posted: set[str], now: datetime,
                     expect_pods: tuple[str, ...] = (), window_hours: float | None = None,
                     reconciled: dict[str, dict] | None = None) -> dict:
    """Every engineering-campaign subrun with a pod, against the provider's posted pods.

    Refuses when any subrun's teardown is unconfirmed (module docstring), before its
    posting or its age is looked at."""
    root = Path(root)
    window = _mg().POSTING_WINDOW_HOURS if window_hours is None else window_hours
    reconciled = reconciled_teardowns(root) if reconciled is None else reconciled
    unposted, never, scanned, subruns = [], [], 0, 0
    confirmed = {"subrun record": 0, "reconciliation": 0}
    for path in sorted((root / "logs").rglob("campaign.json")):
        doc = json.loads(path.read_text())
        if doc.get("schema") != CAMPAIGN_SCHEMA:
            continue
        scanned += 1
        for sub in doc.get("subruns") or []:
            pod = sub.get("pod_id")
            if not pod:
                continue
            subruns += 1
            if pod in expect_pods:
                continue
            where = f"{path.relative_to(root)} :: {sub.get('subrun_id')}"
            if sub.get("teardown_confirmed") is True:
                confirmed["subrun record"] += 1
            elif pod in reconciled:
                confirmed["reconciliation"] += 1
            else:
                raise AuthorizationError(f"{where}: pod {pod}'s teardown is not confirmed (no teardown_confirmed, "
                                         "no reconciliation entry); a missing or posted bill does not show it gone. "
                                         "Reconcile first")
            if pod in posted:
                continue
            ended, basis = _ended(root, path, sub)
            row = {"campaign": str(path.relative_to(root)), "subrun_id": sub.get("subrun_id"), "pod_id": pod,
                   "usd": sub.get("cost_usd"), "ended_utc": ended.isoformat(), "ended_basis": basis,
                   "age_hours": round((now - ended).total_seconds() / 3600, 1)}
            if row["age_hours"] >= window:
                never.append(row)
                continue
            if not isinstance(row["usd"], (int, float)):
                raise AuthorizationError(f"{row['campaign']} :: {row['subrun_id']}: pod {pod} is unposted and "
                                         "records no cost")
            unposted.append(row)
    return {"campaigns_scanned": scanned, "subruns_with_a_pod": subruns, "teardown_confirmed_by": confirmed,
            "unposted": unposted, "never_posted_not_counted": never}


def write_reconciliation(root: Path, snapshot_path: Path, out: Path) -> dict:
    """Confirm, from a committed provider snapshot, the teardown of every campaign subrun whose
    record does not state it: each pod must be absent from the snapshot's resource listing (which
    must have been read); its billed amount there is recorded beside its booked cost. Refuses when
    any such pod is still listed. Read-only towards the provider."""
    root, snapshot_path = Path(root), Path(snapshot_path)
    snap = json.loads(snapshot_path.read_text())
    acct = snap.get("account") or {}
    if acct.get("resources_known") is not True or not isinstance(acct.get("pods"), list):
        raise AuthorizationError("the snapshot's resource listing was not read; it confirms no teardown")
    live = {p.get("id") for p in acct["pods"]}
    billed: dict[str, float] = {}
    for r in snap["pods"]:
        billed[r.get("podId")] = round(billed.get(r.get("podId"), 0.0) + float(r["amount"]), 4)
    rows = []
    for path in sorted((root / "logs").rglob("campaign.json")):
        doc = json.loads(path.read_text())
        if doc.get("schema") != CAMPAIGN_SCHEMA:
            continue
        for sub in doc.get("subruns") or []:
            pod = sub.get("pod_id")
            if not pod or sub.get("teardown_confirmed") is True:
                continue
            if pod in live:
                raise AuthorizationError(f"pod {pod} ({path.relative_to(root)}) is still in the provider's inventory")
            rows.append({"campaign": str(path.relative_to(root)), "subrun_id": sub.get("subrun_id"), "pod_id": pod,
                         "booked_usd": sub.get("cost_usd"), "provider_billed_usd": billed.get(pod)})
    shared = sorted({r["pod_id"] for r in rows if sum(x["pod_id"] == r["pod_id"] for x in rows) > 1})
    doc = {"schema": RECONCILIATION_SCHEMA,
           "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "snapshot": {"path": str(snapshot_path.resolve().relative_to(root.resolve())),
                        "sha256": _sha256(snapshot_path), "fetched_utc": snap.get("fetched_utc")},
           "rule": ("each pod below is a campaign subrun whose record states no teardown_confirmed; the provider "
                    "resource listing in the cited snapshot was read and does not contain it, which is the same "
                    "provider query a launcher's teardown confirmation polls. Re-checked at every read by "
                    "shared.project_position.reconciled_teardowns"),
           "teardowns": rows,
           "attribution_notes": [f"pod {p} is named by more than one subrun; the provider bills it once "
                                 f"(${billed.get(p)}), so at most one of those records is its true cost. The "
                                 "position counts the provider's bill, so committed spend is not understated; the "
                                 "attribution stays unresolved" for p in shared],
           "authorizes": "nothing"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def position(root: Path = REPO, *, snapshot: dict, now: datetime | None = None,
             expect_pods: tuple[str, ...] = (), cap_usd: float | None = None) -> dict:
    """`mixed_governance.position` with engineering-campaign subruns counted (module docstring)."""
    from maintenance.consolidation import derive_budget as DB

    root = Path(root)
    MG = _mg()
    base = MG.position(root, snapshot=snapshot, now=now, expect_pods=expect_pods)
    now = now or _utc(snapshot.get("fetched_utc")) or datetime.now(timezone.utc)
    posted = {r.get("podId") for r in snapshot["pods"]}
    camp = campaign_subruns(root, posted=posted, now=now, expect_pods=expect_pods)
    committed = round(base["committed_usd"] + sum(float(u["usd"]) for u in camp["unposted"]), 4)
    cap = float(cap_usd) if cap_usd is not None else float(DB.latest_project_cap(root, None)["cap_usd"])
    return {**base, "campaign_subruns": camp, "committed_usd": committed, "project_cap_usd": cap,
            "project_remaining_usd": round(cap - committed, 4),
            "rule": "shared.project_position: mixed_governance.position + engineering-campaign subruns"}


def readiness(pos: dict, requests: dict[str, float]) -> dict:
    """Whether hard caps fit the project position and the account balance, alone and together."""
    bal = float(pos.get("balance_usd") or 0.0)
    rows = {}
    for name, cap in requests.items():
        rows[name] = {"hard_cap_usd": cap,
                      "fits_project_cap": pos["committed_usd"] + cap <= pos["project_cap_usd"] + 1e-9,
                      "fits_account_balance": cap <= bal + 1e-9}
    total = round(sum(requests.values()), 4)
    together = {"hard_caps_usd": total,
                "committed_after_usd": round(pos["committed_usd"] + total, 4),
                "fits_project_cap": pos["committed_usd"] + total <= pos["project_cap_usd"] + 1e-9,
                "fits_account_balance": total <= bal + 1e-9}
    return {"requests": rows, "together": together,
            "live_pods": pos["account_pods"], "live_volumes": pos["account_volumes"],
            "balance_usd": bal, "_balance": "reported beside the position, never read as headroom or as a grant"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--snapshot", help="a provider_billing snapshot JSON")
    src.add_argument("--live", action="store_true", help="read the provider now (read-only, $0)")
    ap.add_argument("--request", action="append", default=[], metavar="NAME=USD")
    ap.add_argument("--cap", type=float, default=None, help="override the newest grant's project cap")
    ap.add_argument("--write-reconciliation", default=None, metavar="OUT",
                    help="with --snapshot: confirm, from it, the teardown of subruns whose records do not state it")
    a = ap.parse_args(argv)
    if a.write_reconciliation:
        if not a.snapshot:
            ap.error("--write-reconciliation reads a committed --snapshot")
        doc = write_reconciliation(REPO, Path(a.snapshot), Path(a.write_reconciliation))
        print(json.dumps({"teardowns": len(doc["teardowns"]), "attribution_notes": doc["attribution_notes"]}, indent=1))
        return 0
    snap = _mg().provider_snapshot() if a.live else json.loads(Path(a.snapshot).read_text())
    pos = position(REPO, snapshot=snap, cap_usd=a.cap)
    req = {k: float(v) for k, v in (r.split("=", 1) for r in a.request)}
    out = {"fetched_utc": pos["fetched_utc"], "provider_billed_usd": pos["provider_billed_usd"],
           "committed_usd": pos["committed_usd"], "project_cap_usd": pos["project_cap_usd"],
           "project_remaining_usd": pos["project_remaining_usd"],
           "unposted_sessions": pos["unposted_sessions"],
           "campaign_subruns": {k: v for k, v in pos["campaign_subruns"].items() if k != "never_posted_not_counted"}
                               | {"never_posted_not_counted": len(pos["campaign_subruns"]["never_posted_not_counted"])},
           "storage": [{k: v for k, v in s.items() if k != "netting"} for s in pos["storage"]],
           "rule": pos["rule"]}
    if req:
        out["readiness"] = readiness(pos, req)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
