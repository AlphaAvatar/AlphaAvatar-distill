"""`SessionRunner.create()`: the acquisition window, the billed window and an answer that names no pod.

Three things went wrong, or could, on the path from "ask for a pod" to "own a pod":

* the provider-side ``--terminate-after`` was computed ONCE before the retry
  loop, so every minute queued for capacity came off a late pod's backstop
  (a_tool_pilot_20261104_140656 queued 18 minutes);
* a create whose answer carried no id -- a timeout, a transport error, an
  unreadable reply -- was read as "no pod", which is the inference that
  abandons a billing resource;
* the datacenter could only be constrained through a network volume.

The REAL `create()` runs here with the provider and the CLI faked.
"""
from __future__ import annotations

import json
import subprocess
import types
from datetime import datetime
from pathlib import Path

import pytest

from aadistill.infrastructure import session_runner as SR
from aadistill.infrastructure.provider import AccountResources
from aadistill.infrastructure.session import ExecutionCommands

REPO = Path(__file__).resolve().parents[2]
CAPACITY = '{"error":"There are no longer any instances available with the requested specifications."}'


class _Provider:
    def __init__(self, listings=()):
        self.listings = list(listings)
        self.listed = 0

    def account_resources(self):
        self.listed += 1
        if not self.listings:
            return AccountResources(known=False, error="unreachable")
        return self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]


def _runner(tmp_path, monkeypatch, answers, *, provider=None, clock=None, **args):
    """`answers`: one per create call -- a string (stdout), or an exception to raise."""
    r = object.__new__(SR.SessionRunner)
    r.provider = provider or _Provider()
    r.cli, r.repo_root, r.pod_id, r.price, r.start_epoch, r.draw = "runpodctl", REPO, "", 1.09, 0.0, 1
    r.ev, r._watchdog_for = {}, ""
    r.scr = tmp_path
    r.say = lambda m: r.ev.setdefault("said", []).append(m)
    r.save = lambda: None
    r.plan = types.SimpleNamespace(hard_terminate_minutes=90.0, hard_terminate_usd=2.0)
    r.auth = types.SimpleNamespace(hard_cap_usd=2.5)
    r.spec = types.SimpleNamespace(
        session_id="s-tool_pilot",
        commands=ExecutionCommands(watchdog="scripts/shared/pod/watchdog.py",
                                   setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
                                   artifact_collector="scripts/shared/pod/collect_artifacts.py",
                                   remote_python="/opt/train/bin/python", workspace_root="/workspace",
                                   checkout_root="/workspace/aad", min_cuda_version="13.0"),
        teardown=types.SimpleNamespace(require_provider_confirmation=True))
    r.a = types.SimpleNamespace(**{"create_attempts": len(answers), "create_retry_seconds": 120.0,
                                   "gpu": "NVIDIA L40S", "max_price": 1.19, "image": "img", "disk_gb": 150,
                                   "out": "unused.json", **args})
    sent, now = [], [clock or 1_800_000_000.0]
    queue = list(answers)

    def fake_run(cmd, **kw):
        sent.append(cmd)
        a = queue.pop(0)
        if isinstance(a, BaseException):
            raise a
        return subprocess.CompletedProcess(cmd, 0 if a.startswith('{"id"') else 1, a, "")

    monkeypatch.setattr(SR.subprocess, "run", fake_run)
    monkeypatch.setattr(SR.subprocess, "Popen", lambda cmd, **kw: types.SimpleNamespace(pid=4242))
    monkeypatch.setattr(SR.time, "time", lambda: now[0])

    def sleep(s):
        now[0] += s
    monkeypatch.setattr(SR.time, "sleep", sleep)
    return r, sent


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_the_backstop_counts_from_the_pod_not_from_the_queue(tmp_path, monkeypatch):
    r, sent = _runner(tmp_path, monkeypatch, [CAPACITY, CAPACITY, json.dumps({"id": "pod1", "costPerHr": 1.09})])
    assert r.create() is True
    t = [datetime.strptime(_flag(c, "--terminate-after"), "%Y-%m-%dT%H:%M:%SZ") for c in sent]
    assert (t[1] - t[0]).total_seconds() == 120 and (t[2] - t[1]).total_seconds() == 120
    assert r.ev["billed_window_deadline_utc"] == _flag(sent[2], "--terminate-after")
    assert [a["outcome"] for a in r.ev["acquisition"]["attempts"]] == ["capacity", "capacity", "created"]
    assert r.pod_id == "pod1" and r.ev["provider_resource_created"] is True


def test_a_cli_without_the_flag_is_not_handed_it(tmp_path, monkeypatch):
    r, sent = _runner(tmp_path, monkeypatch, [json.dumps({"id": "pod1", "cost": 1.09})])
    r.cli_facts = {"pod_create_terminate_after": False}
    assert r.create() is True
    assert "--terminate-after" not in sent[0] and r.ev["terminate_after_utc"] is None
    assert r.ev["billed_window_deadline_utc"] and r.price == 1.09


def test_datacenter_groups_rotate_across_attempts_without_a_volume(tmp_path, monkeypatch):
    r, sent = _runner(tmp_path, monkeypatch, [CAPACITY] * 3, data_center_ids="EU-NL-1,US-TX-3; US-CA-2")
    assert r.create() is False
    assert [_flag(c, "--data-center-ids") for c in sent] == ["EU-NL-1,US-TX-3", "US-CA-2", "EU-NL-1,US-TX-3"]
    assert SR.data_center_groups(" ;A, B;;C ") == ["A,B", "C"] and SR.data_center_groups(None) == []


def test_a_volume_keeps_its_own_datacenter(tmp_path, monkeypatch):
    r, sent = _runner(tmp_path, monkeypatch, [CAPACITY], network_volume_id="vol", volume_mount_path="/v",
                      data_center_ids="EUR-IS-2")
    r.ws = "/workspace"
    r.create()
    assert sent[0].count("--data-center-ids") == 1 and _flag(sent[0], "--data-center-ids") == "EUR-IS-2"


def test_an_answer_without_an_id_is_settled_by_the_account_and_a_found_pod_is_owned(tmp_path, monkeypatch):
    mine = AccountResources(known=True, pods=({"kind": "pod", "id": "podX", "name": "aadistill-s-tool_pilot",
                                               "desiredStatus": "RUNNING", "costPerHr": 1.09},))
    r, sent = _runner(tmp_path, monkeypatch, [subprocess.TimeoutExpired("runpodctl", 300)], provider=_Provider([mine]))
    assert r.create() is True
    assert r.pod_id == "podX" and r.ev["provider_resource_created"] is True and r._watchdog_for == "podX"
    assert r.ev["acquisition"]["attempts"][0]["outcome"] == "adopted"
    assert "TimeoutExpired" in (tmp_path / "create_raw_d1_a1.txt").read_text()


def test_an_unreadable_account_after_an_ambiguous_create_stops_everything(tmp_path, monkeypatch):
    r, sent = _runner(tmp_path, monkeypatch, ['{"error":"upstream 502","code":"create_uncertain"}', CAPACITY],
                      provider=_Provider([]))
    assert r.create() is False
    assert len(sent) == 1 and r.ev["provider_state_unknown"] is True and r.pod_id == ""


def test_a_repeated_non_capacity_refusal_is_deterministic_and_stops(tmp_path, monkeypatch):
    clean = AccountResources(known=True, pods=())
    bad = '{"error":"invalid gpu id","code":"bad_request","status":400}'
    r, sent = _runner(tmp_path, monkeypatch, [bad, bad, bad], provider=_Provider([clean]))
    assert r.create() is False
    assert len(sent) == 2 and [a["outcome"] for a in r.ev["acquisition"]["attempts"]] == ["refused", "refused"]


@pytest.mark.parametrize("text,want", [
    ('{"id":"p1","costPerHr":1.09}', ("p1", 1.09)), ('{"id":"p1","cost":0.99}', ("p1", 0.99)),
    (CAPACITY + "\nUsage:\n  runpodctl pod create", ("", None)), ('noise "id": "p2" noise', ("p2", None))])
def test_the_create_answer_is_parsed_for_both_cli_generations(text, want):
    assert SR.answer_pod(text, text) == want
    assert bool(SR.CAPACITY_REFUSAL.search(CAPACITY)) and not SR.CAPACITY_REFUSAL.search('{"error":"invalid gpu id"}')
