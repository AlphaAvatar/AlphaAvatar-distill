"""A session may be funded beside another only when that one is named, live and
booked at its hard cap, and never beyond the phase's max_concurrent_sessions
(maintainer decision of 2026-10-20: at most two formal_volume sessions)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone

import pytest

from aadistill.governance.authorization import AuthorizationError
from stages.calibration_volume import governance as G


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) + "\n")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / G.PROGRAMME_GRANT).parent.mkdir(parents=True)
    shutil.copyfile(G.REPO / G.PROGRAMME_GRANT, tmp_path / G.PROGRAMME_GRANT)
    _write(tmp_path / G.run_dir("closed") / "closeout/outcome.json",
           {"phase": "formal_volume", "money": {"all_in_usd": 10.0}})
    return tmp_path


def _open(repo, run_id, *, hard=39.5, hours_left=12.0):
    after = datetime.now(timezone.utc) + timedelta(hours=hours_left)
    _write(repo / G.run_dir(run_id) / "runtime/session.json",
           {"provider_resource_created": True, "pod_id": f"pod-{run_id}",
            "terminate_after_utc": after.strftime("%Y-%m-%dT%H:%M:%SZ")})
    _write(repo / G.authorization_path(run_id),
           {"hard_cap_usd": hard, "stage_conditions": {"phase": "formal_volume"}})


def test_an_open_run_nobody_names_still_refuses(repo):
    _open(repo, "d4")
    with pytest.raises(AuthorizationError, match="reconcile"):
        G.remaining_envelope("formal_volume", repo)


def test_a_named_live_run_is_booked_at_its_hard_cap(repo):
    base = G.remaining_envelope("formal_volume", repo)
    _open(repo, "d4", hard=39.5)
    env = G.remaining_envelope("formal_volume", repo, concurrent_with=["d4"])
    assert env["phase_remaining_usd"] == pytest.approx(base["phase_remaining_usd"] - 39.5)
    assert env["programme_remaining_usd"] == pytest.approx(base["programme_remaining_usd"] - 39.5)
    assert [o["pod_id"] for o in env["outstanding"]] == ["pod-d4"]


def test_a_named_run_past_its_terminate_time_refuses(repo):
    _open(repo, "d4", hours_left=-1)
    with pytest.raises(AuthorizationError, match="no live pod"):
        G.remaining_envelope("formal_volume", repo, concurrent_with=["d4"])


def test_two_live_beside_a_third_exceed_the_phase_limit(repo):
    _open(repo, "d4")
    _open(repo, "p1", hard=26.5)
    with pytest.raises(AuthorizationError, match="max_concurrent_sessions = 2"):
        G.remaining_envelope("formal_volume", repo, concurrent_with=["d4", "p1"])


def test_a_phase_without_the_allowance_stays_one_at_a_time(repo):
    _open(repo, "d4")
    with pytest.raises(AuthorizationError, match="max_concurrent_sessions = 1"):
        G.remaining_envelope("pilot", repo, concurrent_with=["d4"])


def test_a_closed_or_unknown_run_cannot_be_named(repo):
    with pytest.raises(AuthorizationError, match="not open runs"):
        G.remaining_envelope("formal_volume", repo, concurrent_with=["closed"])
