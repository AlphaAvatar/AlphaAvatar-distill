#!/usr/bin/env python3
"""The Stage-0 calibration-volume programme's money, reconciled three ways.

    PYTHONPATH=src python scripts/stages/stage-0/calibration_volume/write_budget_reconciliation.py \
        --snapshot logs/budget/provider/runpod_billing_2026-10-21.json

Writes `closeout/budget_reconciliation.json`:

* what each run BOOKED (its closeout) against what the PROVIDER billed for the
  pod it created (the committed billing snapshot);
* what each authorization was checked against WHEN it was issued -- the
  envelope every authorization embeds -- so whether the cap in force was kept is
  read from the record of that moment, not recomputed against today's grant;
* when each budget amendment was recorded (the grant's git history) and when
  cumulative new spend actually reached the original $90.

Nothing here edits a closeout, a grant or a cap. Differences are reported.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))

from maintenance.consolidation.derive_budget import _closeout_cost  # noqa: E402
from maintenance.consolidation.provider_billing import (  # noqa: E402
    billed_by_pod, report as project_report, snapshot_identity, volumes_by_day,
)

EXP = "logs/stages/stage-0/calibration_volume"
GRANT = "logs/budget/approvals/stage0_calibration_volume_programme_grant.json"
OUT = f"{EXP}/closeout/budget_reconciliation.json"

#: The maintainer's messages in the executing session. They are not repository
#: records: the times come from that session's transcript and are stated as
#: such. Each was recorded in the grant by the commit `amendments()` finds.
MAINTAINER_MESSAGES = {
    90.0: {"words": "当前 RunPod 账户据维护者确认约有 $90 可用余额。这 $90 是从当前状态"
                    "开始的全部新增支出硬上限",
           "utc": "2026-10-19T04:43:24Z"},
    30.0: {"words": ["d4 也做，上限改加 $25 (Recommended)",
                     "Top1也要补缺失的种子，额度多增加5美元"],
           "utc": "2026-10-20T05:17:42Z (+$25); the +$5 followed before the "
                  "amendment commit, its own timestamp was lost to context "
                  "compaction"},
    6.0: {"words": "并行，加6美金", "utc": "2026-10-20T14:49:14Z"},
}
AUTHORIZED_AT = "2026-10-19T04:43:24Z"
ORIGINAL_NEW_SPEND_CAP = 90.0


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def amendments(grant: dict) -> list[dict]:
    """Each ceiling the grant has held, and the commit that first recorded it."""
    out = []
    for ceiling in sorted({107.5099, 137.5099, 143.5099}):
        line = git("log", "--reverse", "--format=%h %cI",
                   f"-S\"programme_ceiling_usd\": {ceiling}", "--", GRANT).splitlines()
        h, when = line[0].split()
        out.append({"programme_ceiling_usd": ceiling, "recorded_commit": h,
                    "recorded_utc": utc(when).astimezone(timezone.utc)
                    .isoformat(timespec="seconds")})
    added = {107.5099: 90.0, 137.5099: 30.0, 143.5099: 6.0}
    for a in out:
        a["added_usd"] = added[a["programme_ceiling_usd"]]
        a["maintainer_message"] = MAINTAINER_MESSAGES[a["added_usd"]]
    assert float(grant["programme_ceiling_usd"]) == out[-1]["programme_ceiling_usd"]
    return out


def runs() -> list[dict]:
    rows = []
    for d in sorted((REPO / EXP / "runs").iterdir()):
        out = load(d / "closeout/outcome.json")
        sess = load(d / "runtime/session.json")
        auth = load(d / "governance/authorization.json")
        booked, _ = _closeout_cost(out)
        pod = sess.get("pod_id") or out.get("pod_id")
        start = sess.get("created_utc")
        if pod and not start:
            hit = [t["utc"] for t in sess.get("timeline") or []
                   if pod in t.get("msg", "") and "creat" in t.get("msg", "")]
            start = hit[0] if hit else None
        env = json.loads((auth.get("stage_conditions") or {}).get("envelope") or "{}")
        row = {"run_id": d.name, "phase": out.get("phase"),
               "status": out.get("status"), "booked_usd": booked,
               "pod_id": pod, "volume_id": sess.get("volume_id"),
               "start_utc": start,
               "end_utc": (sess.get("finished_utc") or out.get("recorded_utc")),
               "provider_resource_created": out.get("provider_resource_created")}
        if auth:
            held = sum(o["hard_cap_usd"] for o in env.get("outstanding", []))
            committed = round(env["programme_spent_usd"] + held
                              + float(auth["hard_cap_usd"]), 4)
            row["authorization"] = {
                "granted_utc": auth["granted_utc"],
                "hard_cap_usd": float(auth["hard_cap_usd"]),
                "programme_ceiling_in_force_usd": env["programme_ceiling_usd"],
                "programme_spent_at_issue_usd": env["programme_spent_usd"],
                "held_beside_it_usd": round(held, 4),
                "committed_at_issue_usd": committed,
                "fits_ceiling_in_force": committed <= env["programme_ceiling_usd"]}
        rows.append(row)
    return rows


def accrual(rows: list[dict]) -> list[tuple[datetime, datetime, float]]:
    """Each post-authorization charge spread evenly over its own lifetime."""
    out = []
    for r in rows:
        if not r["booked_usd"] or not r["start_utc"] or not r["end_utc"]:
            continue
        a, b = utc(r["start_utc"]), utc(r["end_utc"])
        if a >= utc(AUTHORIZED_AT) and b > a:
            out.append((a, b, r["booked_usd"]))
    return out


def accrued(spans, t: datetime) -> float:
    return sum(usd * min(1.0, max(0.0, (t - a) / (b - a))) for a, b, usd in spans)


def crossing(spans, level: float) -> str | None:
    t, end = utc(AUTHORIZED_AT), max(b for _, b, _ in spans)
    while t <= end:
        if accrued(spans, t) >= level:
            return t.isoformat(timespec="minutes")
        t += timedelta(minutes=1)
    return None


def concurrency(issued: list[dict]) -> dict:
    """When running two sessions at once was recorded, and the issues that used it."""
    h, when = git("log", "--reverse", "--format=%h %cI",
                  '-S"max_concurrent_sessions": 2', "--", GRANT).splitlines()[0].split()
    at = utc(when).astimezone(timezone.utc)
    beside = [{"run_id": r["run_id"], "granted_utc": r["authorization"]["granted_utc"],
               "after_it_was_recorded": utc(r["authorization"]["granted_utc"]) >= at}
              for r in issued if r["authorization"]["held_beside_it_usd"]]
    return {"max_concurrent_sessions_2_recorded_commit": h,
            "recorded_utc": at.isoformat(timespec="seconds"),
            "maintainer_message": {"words": "最多同时两个 (Recommended)",
                                   "utc": "2026-10-20T09:34:07Z"},
            "issued_beside_a_live_session": beside}


def project_cap() -> float:
    sys.path.insert(0, str(REPO / "scripts"))
    from maintenance.consolidation.derive_budget import derive
    return float(derive(REPO)["project"]["cap_usd"])


def by_period(snap: dict) -> list[dict]:
    """Provider day buckets against the records' own cumulative figures at
    three points: the C1 package anchor (the maintainer-stated project total of
    2026-09-11), the eve of this programme, and now. Day buckets are UTC, so
    each boundary is good to within one day's spend."""
    days: dict[str, float] = {}
    for row in snap["pods"]:
        days[row["time"][:10]] = days.get(row["time"][:10], 0.0) + float(row["amount"])
    for d, v in volumes_by_day(snap).items():
        days[d] = days.get(d, 0.0) + v

    def through(day: str) -> float:
        return round(sum(v for d, v in days.items() if d <= day), 4)

    points = [("2026-09-11", 267.8998, "the C1 package anchor, "
               "cumulative_spend_at_approval_usd (maintainer-stated)"),
              ("2026-10-17", 500.1472, "$517.6571 at the 2026-10-19 authorization "
               "less this programme's 2026-10-18 runs ($17.5099)"),
              ("2026-10-21", 625.5798, "derive_budget.py, now")]
    out, prev = [], (0.0, 0.0)
    for day, records, what in points:
        prov = through(day)
        out.append({"through_utc_day": day, "provider_cumulative_usd": prov,
                    "records_cumulative_usd": records, "records_are": what,
                    "provider_minus_records_usd": round(prov - records, 4),
                    "period_provider_usd": round(prov - prev[0], 4),
                    "period_records_usd": round(records - prev[1], 4)})
        prev = (prov, records)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", required=True)
    a = ap.parse_args(argv)
    snap_path = (REPO / a.snapshot).resolve()
    snap = json.loads(snap_path.read_text())
    billed = billed_by_pod(snap)
    grant = json.loads((REPO / GRANT).read_text())
    amend = amendments(grant)
    rows = runs()
    for r in rows:
        if r["pod_id"]:
            got = billed.get(r["pod_id"])
            r["provider_billed_usd"] = None if got is None else round(got["usd"], 4)
            r["provider_minus_booked_usd"] = (
                None if got is None else round(got["usd"] - (r["booked_usd"] or 0), 4))
    before = [r for r in rows if r["end_utc"] and utc(r["end_utc"]) < utc(AUTHORIZED_AT)]
    after = [r for r in rows if r not in before]
    vols = volumes_by_day(snap)
    vol_window = {d: round(v, 4) for d, v in vols.items() if d >= AUTHORIZED_AT[:10]}

    def side(rs, with_volumes: bool) -> dict:
        pods = [r for r in rs if r["pod_id"]]
        netvol = [r for r in rs if r["volume_id"] and not r["pod_id"]]
        prov_pods = sum(r.get("provider_billed_usd") or 0 for r in pods)
        prov_vol = sum(vol_window.values()) if with_volumes else 0.0
        return {
            "runs": len(rs),
            "booked_usd": round(sum(r["booked_usd"] or 0 for r in rs), 4),
            "booked_pods_usd": round(sum(r["booked_usd"] or 0 for r in pods), 4),
            "booked_network_volumes_usd": round(sum(r["booked_usd"] or 0 for r in netvol), 4),
            "provider_pods_usd": round(prov_pods, 4),
            "provider_network_volumes_usd": round(prov_vol, 4),
            "provider_total_usd": round(prov_pods + prov_vol, 4),
            "pods_not_in_provider_billing": [
                {"run_id": r["run_id"], "pod_id": r["pod_id"], "booked_usd": r["booked_usd"]}
                for r in pods if r.get("provider_billed_usd") is None]}

    pre, post = side(before, False), side(after, True)
    spans = accrual(after)
    amend_view = [{**x, "new_spend_accrued_by_then_usd": round(
        accrued(spans, utc(x["recorded_utc"])), 4)} for x in amend]
    issued = [r for r in rows if "authorization" in r]
    for r in issued:
        au = r["authorization"]
        if utc(au["granted_utc"]) < utc(AUTHORIZED_AT):
            continue    # under the pre-authorization grant, whose envelope it embeds
        enough = next(x for x in amend
                      if x["programme_ceiling_usd"] >= au["committed_at_issue_usd"])
        au["ceiling_it_needed_recorded_utc"] = enough["recorded_utc"]
        au["issued_after_that_ceiling_was_recorded"] = (
            utc(au["granted_utc"]) >= utc(enough["recorded_utc"]))
    authorized_new = sum(x["added_usd"] for x in amend)
    doc = {
        "schema": "aadistill.stage0.calibration_volume.budget_reconciliation/v1",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"provider_snapshot": snapshot_identity(snap_path, REPO),
                    "grant": GRANT, "runs": f"{EXP}/runs/*",
                    "generator": "scripts/stages/stage-0/calibration_volume/"
                                 "write_budget_reconciliation.py"},
        "authorization_boundary_utc": AUTHORIZED_AT,
        "historical_spend_before_authorization": {
            "project_cumulative_at_authorization_usd": 517.6571,
            "_project_figure": "derive_budget.py at 2026-10-19, the figure the "
                               "maintainer's authorization cites; it includes "
                               "the programme's 2026-10-18 runs below",
            "programme_2026_10_18": pre},
        "actual_spending_after_authorization": post,
        "authorized_limits": {
            "original_new_spend_cap_usd": ORIGINAL_NEW_SPEND_CAP,
            "new_spend_authorized_in_total_usd": authorized_new,
            "programme_ceiling_usd": float(grant["programme_ceiling_usd"]),
            "formal_volume_ceiling_usd": grant["phases"]["formal_volume"]["ceiling_usd"],
            "per_session_cap_usd": grant["per_session_cap_usd"],
            "max_concurrent_sessions": grant["phases"]["formal_volume"]
            ["max_concurrent_sessions"],
            "amendments": amend_view,
            "concurrency": concurrency(issued)},
        "cap_in_force_at_every_issue": {
            "authorizations": len(issued),
            "all_fit_the_ceiling_in_force": all(
                r["authorization"]["fits_ceiling_in_force"] for r in issued),
            "relied_on_an_amendment_beyond_the_original_90": [
                r["run_id"] for r in issued
                if r["authorization"]["committed_at_issue_usd"] > amend[0]["programme_ceiling_usd"]],
            "every_issue_came_after_the_ceiling_it_needed_was_recorded": all(
                r["authorization"]["issued_after_that_ceiling_was_recorded"]
                for r in issued
                if "issued_after_that_ceiling_was_recorded" in r["authorization"]),
            "largest_hard_cap_usd": max(r["authorization"]["hard_cap_usd"] for r in issued)},
        "new_spend_reached_the_original_90_utc": crossing(spans, ORIGINAL_NEW_SPEND_CAP),
        "_accrual": ("each post-authorization charge spread evenly from its pod's "
                     "creation (a volume's creation) to its end; an estimate of "
                     "WHEN money was spent, not of how much"),
        "overrun": {
            "against_the_cap_in_force_usd": round(max(
                0.0, post["booked_usd"] - authorized_new,
                post["provider_total_usd"] - authorized_new), 4),
            "against_the_original_90_alone_usd": {
                "booked": round(post["booked_usd"] - ORIGINAL_NEW_SPEND_CAP, 4),
                "provider": round(post["provider_total_usd"] - ORIGINAL_NEW_SPEND_CAP, 4)}},
        "the_missing_108": {
            "what_it_was": ("NOT historical spend: the 28 post-authorization runs above "
                            "(booked $%.4f). Their closeouts existed and logs/index.json "
                            "did not list them; derive_budget.project_sessions read runs "
                            "only through the index, so the project total stayed at the "
                            "pre-authorization figure while this programme's own gate "
                            "(governance.programme_spent, which reads closeouts) was right"
                            % post["booked_usd"]),
            "root_cause": ("the run index is navigation rebuilt by hand "
                           "(record_run_index.py); no closeout writer updates it, and the "
                           "project balance depended on it"),
            "prevention": {
                "code": ["scripts/maintenance/consolidation/derive_budget.py :: "
                         "_sessions_the_index_misses (closeouts and open sessions on disk; "
                         "counted once; named)",
                         "src/aadistill/infrastructure/provider.py :: reconcile_account, "
                         "src/aadistill/infrastructure/session_runner.py :: "
                         "check_account_resources (on by default, before create)",
                         "scripts/shared/pod/stage_c2_probes_to_volume.py :: "
                         "require_reconciled_account; scripts/shared/pod/canary.py :: "
                         "account_reconciled",
                         "scripts/maintenance/consolidation/provider_billing.py (provider "
                         "billing beside the records)"],
                "tests": ["tests/docs/test_log_organisation.py :: "
                          "TestTheProjectTotalDoesNotDependOnTheIndex",
                          "tests/infrastructure/test_account_balance_gate.py :: "
                          "TestTheAccountHoldsNothingTheRecordsDoNotName, "
                          "TestTheRunPodListingIsParsedNotAssumed"]}},
        "retroactively_discovered_historical_spend": {
            "provider_minus_records_usd": None,
            "status": "UNVERIFIED attribution: see project.pods_unattributed",
            "_meaning": ("money the provider billed this account that no repository "
                         "record holds; none of it falls in this programme"),
        },
        "remaining": {
            "project_per_records_usd": None,
            "programme_usd": round(float(grant["programme_ceiling_usd"])
                                   - pre["booked_usd"] - post["booked_usd"], 4),
            "_not_a_permission": "no new paid work is authorized by any remainder",
        },
        "resource_terminal_state": {
            "fetched_utc": snap["fetched_utc"],
            "pods": snap["account"].get("pods"),
            "network_volumes": snap["account"].get("network_volumes"),
            "current_spend_per_hr": snap["account"].get("current_spend_per_hr"),
            "client_balance_usd": snap["account"].get("client_balance_usd")},
        "runs": rows,
        "project": {**project_report(REPO, snap_path),
                    "by_period": by_period(snap)},
    }
    proj = doc["project"]
    doc["retroactively_discovered_historical_spend"]["provider_minus_records_usd"] = (
        proj["provider_minus_records_usd"])
    doc["retroactively_discovered_historical_spend"]["if_all_project_spend_over_cap_usd"] = (
        round(proj["provider_billed_usd"]["total"] - project_cap(), 4))
    doc["remaining"]["project_per_records_usd"] = round(
        project_cap() - proj["records_cumulative_usd"], 4)
    (REPO / OUT).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
