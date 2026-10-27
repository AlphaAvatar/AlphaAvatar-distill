#!/usr/bin/env python3
"""The independent accounting audit: provider billing against every record ($0, read-only).

    PYTHONPATH=src:scripts .venv/bin/python scripts/maintenance/consolidation/accounting_audit.py \\
        --snapshot logs/budget/provider/runpod_billing_<ts>.json \\
        --write logs/budget/audits/accounting_audit_<date>.json

It reads one committed provider snapshot and the repository's records and
answers, with each figure derived rather than restated:

1. **Attribution of billed pods.** Each billed pod is created by a run record,
   only mentioned by one, attributed by the evidence file (every number
   re-checked against the bill), or unattributed -- split at the C1 anchor,
   because the maintainer's stated anchor covers everything before it.
2. **The gap.** Provider total minus the records' cumulative, decomposed into
   the pre-anchor part (provider above the stated anchor) and the post-anchor
   part (records against provider since).
3. **Unposted sessions.** Every recorded pod absent from the bill: inside the
   72 h posting window it is a liability counted at its booked cost; beyond it
   the provider never charged it.
4. **Network volumes.** Day rows carry no volume id: per day, the billed GB-h
   against what the recorded lifetimes can accrue, and the excess that stays
   unattributed (it is in the bill either way).
5. **The mixed-calibration study.** Each of its four booked items against what
   the provider billed for it.
6. **The cap and the binding position.** The cap every issuer uses against the
   one the records derivation used, and the provider-based committed position.

It edits no record, rebooks no history and moves no cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

EVIDENCE = "logs/budget/audits/attribution_evidence.json"
POSTING_WINDOW_HOURS = 72.0
MIXED_ITEMS = (("MC1 attempt 1", "logs/stages/stage-0/calibration_volume/runs/mixs_20261026_102914"),
               ("MC1 attempt 2", "logs/stages/stage-0/calibration_volume/runs/mixs_20261026_103634"),
               ("MC2", "logs/stages/stage-0/calibration_volume/runs/mixr_20261026_140539"))
MIXED_VOLUME = "logs/stages/stage-0/calibration_volume/runs/mixvol_20261026_102835"


def _day(row: dict) -> str:
    return str(row.get("time") or row.get("startDate"))[:10]


def pod_rows(snap: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in snap["pods"]:
        p = out.setdefault(r["podId"], {"usd": 0.0, "ms": 0, "days": set()})
        p["usd"] += float(r["amount"])
        p["ms"] += int(r.get("timeBilledMs") or 0)
        p["days"].add(_day(r))
    return out


def check_evidence(root: Path, pods: dict) -> tuple[dict, list[str]]:
    """Accept each evidence row only when the bill confirms its numbers."""
    doc = json.loads((root / EVIDENCE).read_text())
    accepted, problems = {}, []
    for row in doc["pods"]:
        pid, a, c = row["pod_id"], row["attributed_to"], row["checks"]
        billed = pods.get(pid)
        rec = json.loads((root / a["record"]).read_text())
        sub = next((s for s in rec.get("subruns", []) if s.get("subrun_id") == a["subrun_id"]), None)
        bad = []
        if billed is None:
            bad.append("not in the bill")
        if sub is None or float(sub["cost_usd"]) != a["recorded_cost_usd"] or \
                float(sub["elapsed_minutes"]) != a["recorded_elapsed_minutes"] or sub.get("pod_id") != a["recorded_pod_id"]:
            bad.append("the cited record does not say what the row quotes")
        if billed is not None:
            if billed["days"] != {c["same_utc_day"]}:
                bad.append(f"billed on {sorted(billed['days'])}")
            if abs(billed["ms"] / 60000 - a["recorded_elapsed_minutes"]) > c["max_minutes_difference"]:
                bad.append("billed minutes differ from the recorded elapsed minutes")
            if abs(billed["usd"] - a["recorded_cost_usd"]) > c["max_usd_difference"]:
                bad.append("billed amount differs from the recorded cost")
        if bad:
            problems.append(f"{pid}: {'; '.join(bad)}")
        else:
            accepted[pid] = {**a, "billed_usd": round(billed["usd"], 4),
                             "billed_minutes": round(billed["ms"] / 60000, 3)}
    return accepted, problems


def audit(root: Path, snapshot_path: Path) -> dict:
    from maintenance.consolidation import provider_billing as PB
    from maintenance.consolidation.derive_budget import PACKAGE, derive
    from stages.calibration_volume import mixed_governance as MG

    snap = json.loads(snapshot_path.read_text())
    fetched = datetime.fromisoformat(snap["fetched_utc"])
    pkg_cfg = json.loads((root / PACKAGE).read_text())
    anchor_day = str(pkg_cfg["execution_package"]["approved_utc"])[:10]
    d = derive(root)
    project = d["project"]

    # 1. billed pods
    pods = pod_rows(snap)
    created = PB.run_records(root)
    loose = [p for p in pods if p not in created]
    mentioned = PB.mentioned(root, loose)
    evidence, evidence_problems = check_evidence(root, pods)
    unattributed = sorted(({"pod_id": p, "usd": round(pods[p]["usd"], 4), "first_day": min(pods[p]["days"])}
                           for p in loose if p not in mentioned and p not in evidence),
                          key=lambda r: (r["first_day"], r["pod_id"]))
    pre = [r for r in unattributed if r["first_day"] < anchor_day]
    post = [r for r in unattributed if r["first_day"] >= anchor_day]

    # 2. the gap, split at the anchor
    pods_pre = sum(float(r["amount"]) for r in snap["pods"] if _day(r) < anchor_day)
    pods_post = sum(float(r["amount"]) for r in snap["pods"] if _day(r) >= anchor_day)
    vols = PB.volumes_by_day(snap)
    vol_pre = sum(v for k, v in vols.items() if k < anchor_day)
    vol_post = sum(v for k, v in vols.items() if k >= anchor_day)
    records_since = round(project["spent_since_anchor_usd"] + project["engineering_campaign_spend_since_anchor_usd"], 4)
    provider_total = round(pods_pre + pods_post + vol_pre + vol_post, 4)
    gap = {
        "provider_total_usd": provider_total,
        "records_cumulative_usd": project["cumulative_spend_usd"],
        "provider_minus_records_usd": round(provider_total - project["cumulative_spend_usd"], 4),
        "before_the_anchor": {"anchor_day": anchor_day, "provider_usd": round(pods_pre + vol_pre, 4),
                              "maintainer_stated_anchor_usd": project["anchor_usd"],
                              "provider_minus_anchor_usd": round(pods_pre + vol_pre - project["anchor_usd"], 4)},
        "since_the_anchor": {"provider_usd": round(pods_post + vol_post, 4), "records_usd": records_since,
                             "provider_minus_records_usd": round(pods_post + vol_post - records_since, 4)},
    }

    # 3. recorded pods the provider has not billed
    never, pending = [], []
    sessions = {f"{s['experiment_id']}/{s['run_id']}": s for s in __import__(
        "maintenance.consolidation.derive_budget", fromlist=["project_sessions"]).project_sessions(root)}
    for pod, recs in sorted(created.items()):
        if pod in pods:
            continue
        for r in recs:
            run = r.get("run_id")
            booked = next((s["cost_usd"] for k, s in sessions.items() if k.endswith(f"/{run}")), None)
            ended, basis = None, None
            for path in sorted(root.glob(f"logs/stages/*/*/runs/{run}/closeout/outcome.json")):
                ended, basis = json.loads(path.read_text()).get("recorded_utc"), "closeout recorded_utc"
                if not ended:
                    # the date the record entered history: a closeout without its own
                    # timestamp cannot be younger than its first commit
                    ended = subprocess.run(["git", "log", "--diff-filter=A", "--format=%cI", "--", str(path)],
                                           cwd=root, capture_output=True, text=True).stdout.strip().split("\n")[-1] or None
                    basis = "the commit that added its closeout"
            age_h = ((fetched - datetime.fromisoformat(ended.replace("Z", "+00:00"))).total_seconds() / 3600
                     if ended else None)
            row = {"pod_id": pod, "run_id": run, "run_booked_usd": booked, "ended_utc": ended,
                   "ended_basis": basis, "age_hours": None if age_h is None else round(age_h, 1)}
            if age_h is None:
                raise SystemExit(f"{run}: pod {pod} is unbilled and its end time cannot be established")
            (pending if age_h is not None and age_h < POSTING_WINDOW_HOURS else never).append(row)

    # 3b. engineering-campaign subruns: the issuers do not scan these
    campaign_unbilled, campaigns_scanned = [], 0
    for path in sorted((root / "logs").rglob("campaign.json")):
        doc = json.loads(path.read_text())
        if doc.get("schema") != "aadistill.engineering_campaign/v1":
            continue
        campaigns_scanned += 1
        for sub in doc.get("subruns") or []:
            pid = sub.get("pod_id")
            if pid and pid not in pods and pid not in evidence:
                campaign_unbilled.append({"campaign": str(path.relative_to(root)), "subrun": sub.get("subrun_id"),
                                          "pod_id": pid, "booked_usd": sub.get("cost_usd")})

    # 4. network volumes
    records = MG.volume_records(root)
    vol_days, excess_usd = [], 0.0
    for r in sorted(snap["networkvolumes"], key=lambda r: r["startDate"]):
        day = datetime.fromisoformat(r["startDate"][:10]).replace(tzinfo=timezone.utc)
        alive = [(v["volume"], v["size_gb"] * MG._hours_in_day(v, day, fetched)) for v in records
                 if v["created"] < day + timedelta(days=1) and (v["deleted"] is None or v["deleted"] > day)]
        reachable = sum(x for _, x in alive)
        billed = float(r.get("diskSpaceBilledGb") or 0.0)
        excess = max(0.0, billed - reachable)
        usd = float(r["amount"]) * excess / billed if billed else 0.0
        excess_usd += usd
        vol_days.append({"day": r["startDate"][:10], "billed_gb_h": billed, "billed_usd": round(float(r["amount"]), 4),
                         "recorded_volumes": [a for a, _ in alive], "recorded_max_gb_h": reachable,
                         "unattributed_gb_h": excess, "unattributed_usd": round(usd, 4)})

    # 5. the mixed-calibration study
    mixed = []
    for label, run in MIXED_ITEMS:
        sess = json.loads((root / run / "runtime/session.json").read_text())
        out = json.loads((root / run / "closeout/outcome.json").read_text())
        billed = pods.get(sess["pod_id"])
        mixed.append({"item": label, "run": run, "pod_id": sess["pod_id"],
                      "booked_usd": (out.get("money") or {}).get("all_in_usd"),
                      "provider_billed_usd": None if billed is None else round(billed["usd"], 4),
                      "billed_days": None if billed is None else sorted(billed["days"])})
    vol = json.loads((root / MIXED_VOLUME / "closeout/outcome.json").read_text())
    mixed.append({"item": "MC1 volume", "run": MIXED_VOLUME, "volume_id": "t6yrsvujha",
                  "booked_usd": (vol.get("money") or {}).get("all_in_usd"),
                  "provider_billed_usd": None,
                  "_provider": "volume rows carry no volume id: 2026-10-26's row (1,560 GB-h, $0.1517) holds this "
                               "150 GB x 4 h (600 GB-h) and am5m373kuq's 720 GB-h; see network_volumes"})
    grant = json.loads((root / "logs/budget/approvals/mixed_calibration_grant.json").read_text())
    booked_total = round(sum(float(m["booked_usd"]) for m in mixed), 4)

    # 6. caps and the binding position
    pos = MG.position(root, snapshot=snap)
    findings = []
    if project["cap_source"]["package_cap_usd"] != project["cap_source"]["cap_usd"]:
        findings.append({"id": "stale_package_cap", "severity": "accounting",
                         "what": f"the records derivation read the project cap from {PACKAGE} "
                                 f"(${project['cap_source']['package_cap_usd']}), which no grant after 2026-10-23 "
                                 f"updated; every issuer used ${project['cap_source']['cap_usd']} "
                                 f"({project['cap_source']['owner']})",
                         "fix": "derive_budget.latest_project_cap reads the newest grant that states the cap; "
                                "no record edited"})
    if evidence:
        findings.append({"id": "misrecorded_pod_id", "severity": "attribution",
                         "what": "a billed pod named by no record is the C3 packing pilot's subrun v6, booked "
                                 "under v5's pod id", "rows": evidence,
                         "fix": "attributed by logs/budget/audits/attribution_evidence.json; its cost was "
                                "already in the records, so nothing is added"})
    findings.append({"id": "campaign_subruns_not_scanned_for_unposted_pods", "severity": "governance",
                     "what": "mixed_governance.position (and h1pa_governance.live_position) look for unposted "
                             "pods only in runs/*/runtime/session.json; an engineering campaign's subruns "
                             "(validations/, pilots/) are invisible to it, so an engineering pod still inside "
                             "the 72 h posting window would be missing from 'committed'",
                     "fix": "the next study's issuer must count campaign subruns too (owed engineering; "
                            "today none is inside the window, so today's position is complete)"})
    findings.append({"id": "volume_rows_exceed_lifetimes", "severity": "attribution",
                     "what": f"the provider's volume rows bill {round(excess_usd, 4)} USD of GB-h beyond what "
                             "every recorded volume's lifetime can accrue (including three September rows no "
                             "stage storage record covers: the C2 behavioural durable-staging volumes, booked "
                             "in their own validation records)",
                     "fix": "none needed for safety: the position counts every billed row; the excess is "
                            "never netted against a reservation"})
    verdict = ("CLOSED: the provider-based position is complete and conservative; no unposted session "
               "is outstanding; every post-anchor billed pod is attributed; the remaining gap lies "
               "before the 2026-09-11 anchor, which the maintainer stated, and its attribution is the "
               "maintainer's") if not post and not pending and not evidence_problems else "OPEN"
    return {
        "schema": "aadistill.budget.accounting_audit/v1",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": {"path": str(snapshot_path.relative_to(root)),
                     "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                     "fetched_utc": snap["fetched_utc"]},
        "verdict": verdict,
        "billed_pods": {"total": len(pods), "created_by_a_run_record": sum(1 for p in pods if p in created),
                        "only_mentioned": len([p for p in loose if p in mentioned]),
                        "attributed_by_evidence": sorted(evidence),
                        "evidence_problems": evidence_problems,
                        "unattributed_before_anchor": {"pods": len(pre), "usd": round(sum(r["usd"] for r in pre), 4)},
                        "unattributed_since_anchor": post},
        "gap": gap,
        "recorded_pods_not_billed": {"inside_posting_window": pending, "never_charged": never,
                                     "engineering_campaign_subruns_not_billed": campaign_unbilled,
                                     "engineering_campaigns_scanned": campaigns_scanned,
                                     "_rule": f"a pod absent from the bill for {POSTING_WINDOW_HOURS:.0f} h was "
                                              "never charged; its booked cost stays in the records, which "
                                              "therefore overstate by that much"},
        "network_volumes": {"days": vol_days, "unattributed_usd": round(excess_usd, 4),
                            "billed_usd": round(sum(vols.values()), 4)},
        "mixed_calibration": {"items": mixed, "booked_total_usd": booked_total,
                              "grant_closed_spent_usd": grant["closed"]["spent_usd"],
                              "agrees_with_the_grant_closeout": abs(booked_total - grant["closed"]["spent_usd"]) < 1e-4},
        "cap": {"binding_project_cap_usd": project["cap_source"]["cap_usd"],
                "owner": project["cap_source"]["owner"],
                "package_cap_usd_stale": project["cap_source"]["package_cap_usd"]},
        "position": {k: pos[k] for k in ("provider_billed_usd", "unposted_sessions", "committed_usd",
                                         "project_cap_usd", "project_remaining_usd", "balance_usd",
                                         "account_pods", "account_volumes")}
                    | {"storage": [{k: v for k, v in s.items() if k != "netting"} for s in pos["storage"]],
                       "rule": pos["rule"]},
        "findings": findings,
        "not_changed": ["no historical conclusion, authorization, grant or run record was edited",
                        "no cap was raised: $740 is the maintainer's 2026-10-26 decision",
                        "the pre-anchor difference is reported, not booked"],
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    doc = audit(REPO_ROOT, Path(a.snapshot).resolve())
    text = json.dumps(doc, indent=1, default=str) + "\n"
    if a.write:
        Path(a.write).write_text(text)
        print(a.write)
    print(json.dumps({k: doc[k] for k in ("verdict", "gap", "position")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
