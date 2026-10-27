"""The project position counts engineering-campaign subruns (accounting audit 2026-10-27, finding 3).

A campaign subrun the provider has not billed, inside the posting window, is a
liability at its booked cost; without a confirmed teardown it refuses the
position outright; past the window it was never charged. The position on the
repository's own records and committed snapshot is the issuers' rule plus the
campaign subruns, against the newest grant's cap.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aadistill.governance.authorization import AuthorizationError
from shared import project_position as PP

REPO = Path(__file__).resolve().parents[3]
SNAPSHOT = REPO / "logs/budget/provider/runpod_billing_2026-10-27T1011Z.json"
NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def _campaign(root: Path, subruns: list[dict], where="logs/stages/stage-x/exp/validations/v/v1") -> Path:
    d = root / where
    d.mkdir(parents=True)
    p = d / "campaign.json"
    p.write_text(json.dumps({"schema": PP.CAMPAIGN_SCHEMA, "campaign_id": "t", "subruns": subruns}))
    return p


def _sub(pod, hours_ago, usd=0.5, confirmed=True, **kw):
    return {"subrun_id": f"s_{pod}", "pod_id": pod, "cost_usd": usd, "teardown_confirmed": confirmed,
            "ended_utc": (NOW - timedelta(hours=hours_ago)).isoformat(), **kw}


def test_an_unposted_subrun_inside_the_window_is_counted_and_an_old_one_is_not(tmp_path):
    _campaign(tmp_path, [_sub("billed", 2), _sub("fresh", 5, usd=1.25), _sub("old", 200), {"subrun_id": "x"}])
    out = PP.campaign_subruns(tmp_path, posted={"billed"}, now=NOW)
    assert out["campaigns_scanned"] == 1 and out["subruns_with_a_pod"] == 3
    assert [r["pod_id"] for r in out["unposted"]] == ["fresh"] and out["unposted"][0]["usd"] == 1.25
    assert [r["pod_id"] for r in out["never_posted_not_counted"]] == ["old"]
    assert out["unposted"][0]["ended_basis"] == "subrun ended_utc"


def test_an_unposted_subrun_without_a_confirmed_teardown_refuses(tmp_path):
    _campaign(tmp_path, [_sub("live?", 1, confirmed=False)])
    with pytest.raises(AuthorizationError, match="teardown is not confirmed"):
        PP.campaign_subruns(tmp_path, posted=set(), now=NOW)
    # the pod the caller is about to own is not a stray
    assert PP.campaign_subruns(tmp_path, posted=set(), now=NOW, expect_pods=("live?",))["unposted"] == []


def test_an_unposted_subrun_without_a_cost_refuses(tmp_path):
    _campaign(tmp_path, [_sub("p", 1, usd=None)])
    with pytest.raises(AuthorizationError, match="records no cost"):
        PP.campaign_subruns(tmp_path, posted=set(), now=NOW)


def test_a_subrun_without_ended_utc_is_dated_by_its_record(tmp_path):
    p = _campaign(tmp_path, [{"subrun_id": "s", "pod_id": "p", "cost_usd": 0.1, "teardown_confirmed": True}])
    out = PP.campaign_subruns(tmp_path, posted=set(), now=datetime.now(timezone.utc))
    assert out["unposted"][0]["ended_basis"] == "the record's modification time"     # no git here
    assert p.is_file()


def test_other_schemas_are_not_campaigns(tmp_path):
    d = tmp_path / "logs/a"
    d.mkdir(parents=True)
    (d / "campaign.json").write_text(json.dumps({"schema": "something/v1", "subruns": [_sub("p", 1)]}))
    assert PP.campaign_subruns(tmp_path, posted=set(), now=NOW)["campaigns_scanned"] == 0


def test_the_position_on_the_repository_is_the_issuers_rule_plus_campaign_subruns():
    from stages.calibration_volume import mixed_governance as MG

    snap = json.loads(SNAPSHOT.read_text())
    base = MG.position(REPO, snapshot=snap)
    pos = PP.position(REPO, snapshot=snap)
    camp = pos["campaign_subruns"]
    assert camp["campaigns_scanned"] >= 18 and camp["subruns_with_a_pod"] >= 55
    assert pos["committed_usd"] == round(base["committed_usd"] + sum(u["usd"] for u in camp["unposted"]), 4)
    assert pos["project_cap_usd"] == 740.0            # the newest grant's, not a constant here
    r = PP.readiness(pos, {"A-E": 3.5, "B-E": 3.0})
    assert r["together"]["hard_caps_usd"] == 6.5
    assert r["together"]["committed_after_usd"] == round(pos["committed_usd"] + 6.5, 4)
    # a request larger than the headroom does not fit, whatever the balance says
    big = PP.readiness(pos, {"X": pos["project_remaining_usd"] + 0.01})
    assert not big["requests"]["X"]["fits_project_cap"] and big["requests"]["X"]["fits_account_balance"]


def test_position_adds_what_the_campaign_scan_counts(monkeypatch):
    """The composition line itself: an unposted campaign subrun raises committed by its cost."""
    snap = json.loads(SNAPSHOT.read_text())
    before = PP.position(REPO, snapshot=snap)
    row = {"campaign": "c", "subrun_id": "s", "pod_id": "p", "usd": 1.2345}
    monkeypatch.setattr(PP, "campaign_subruns", lambda *a, **k: {"campaigns_scanned": 1, "subruns_with_a_pod": 1,
                                                                   "unposted": [row], "never_posted_not_counted": []})
    after = PP.position(REPO, snapshot=snap)
    assert after["committed_usd"] == round(before["committed_usd"] + 1.2345, 4)
    assert after["project_remaining_usd"] == round(before["project_remaining_usd"] - 1.2345, 4)
