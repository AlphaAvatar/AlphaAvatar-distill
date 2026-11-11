"""`SessionRunner.create()`: the acquisition window, the billed window and an answer that names no pod.

What went wrong, or could, on the path from "ask for a pod" to "own a pod":

* the provider-side ``--terminate-after`` was computed ONCE before the retry loop, so every minute queued for
  capacity came off a late pod's backstop (a_tool_pilot_20261104_140656 queued 18 minutes);
* a create whose answer carried no id -- a timeout, a transport error, ``create_uncertain`` -- was read as "no pod";
  ONE empty listing was then taken as proof, although a pod can appear after its create returned;
* only the first matching pod would have been registered, and the bill counted from when the answer came back.

The REAL `create()` runs here with the provider, the CLI and the clock faked.
"""
from __future__ import annotations

import json
import subprocess
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aadistill.infrastructure import session_runner as SR
from aadistill.infrastructure.provider import AccountResources, PodState
from aadistill.infrastructure.session import ExecutionCommands

REPO = Path(__file__).resolve().parents[2]
CAPACITY = '{"error":"There are no longer any instances available with the requested specifications."}'
UNCERTAIN = '{"error":"the pod may have been created, check runpodctl pod list","code":"create_uncertain"}'
T0 = 1_800_000_000.0


class _Provider:
    """``script``: one entry per account listing -- "empty", "unknown", or a list of request indexes (0 = the
    session's first request) whose pods are live; the last entry repeats. ``created_at`` offsets a pod's
    provider creation time from its request's start."""

    def __init__(self, runner, script=("empty",), created_at=30.0):
        self.r, self.script, self.calls, self.created_at = runner, list(script), 0, created_at
        self.terminated, self.gone, self.keep_alive = [], set(), False

    def _pods(self, idx):
        names = list(self.r._requests)
        return [{"kind": "pod", "id": f"pod{i}", "name": names[i], "desiredStatus": "RUNNING", "costPerHr": 1.09}
                for i in idx if i < len(names) and f"pod{i}" not in self.gone]

    def account_resources(self):
        e = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if e == "unknown":
            return AccountResources(known=False, error="unreachable")
        return AccountResources(known=True, pods=tuple(self._pods([] if e == "empty" else e)))

    def pod_record(self, pid):
        i = int(pid[3:]) if pid.startswith("pod") and pid[3:].isdigit() else 0
        reqs = list(self.r._requests.values())
        start = reqs[i]["epoch"] if i < len(reqs) else T0
        return {"id": pid, "createdAt": datetime.fromtimestamp(start + self.created_at, timezone.utc).isoformat(),
                "dataCenterId": "EU-RO-1", "costPerHr": 1.09}

    def get(self, pid):
        return PodState(pod_id=pid, exists=pid not in self.gone, desired_status="TERMINATED" if pid in self.gone else "RUNNING")

    def terminate(self, pid):
        self.terminated.append(pid)
        self.gone.add(pid)
        return []


def _runner(tmp_path, monkeypatch, answers, *, script=("empty",), **args):
    """`answers`: one per create call -- a string (stdout) or an exception to raise."""
    r = object.__new__(SR.SessionRunner)
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
                                   "out": "unused.json", "reconcile_window_seconds": 60.0, **args})
    r.provider = _Provider(r, script)
    sent, now, queue = [], [T0], list(answers)

    def fake_run(cmd, **kw):
        if "remove" in cmd:                                                 #: the CLI's delete reaches the provider
            r.provider.terminated.append(cmd[-1])
            if not r.provider.keep_alive:
                r.provider.gone.add(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        sent.append(cmd)
        a = queue.pop(0)
        if isinstance(a, BaseException):
            raise a
        return subprocess.CompletedProcess(cmd, 0 if a.startswith('{"id"') else 1, a, "")

    monkeypatch.setattr(SR.subprocess, "run", fake_run)
    monkeypatch.setattr(SR.subprocess, "Popen", lambda cmd, **kw: types.SimpleNamespace(pid=4242))
    monkeypatch.setattr(SR.time, "time", lambda: now[0])
    monkeypatch.setattr(SR.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    return r, sent, now


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_the_backstop_counts_from_the_pod_not_from_the_queue(tmp_path, monkeypatch):
    r, sent, now = _runner(tmp_path, monkeypatch, [CAPACITY, CAPACITY, json.dumps({"id": "pod2", "costPerHr": 1.09})])
    assert r.create() is True
    t = [datetime.strptime(_flag(c, "--terminate-after"), "%Y-%m-%dT%H:%M:%SZ") for c in sent]
    assert (t[1] - t[0]).total_seconds() == 120 and (t[2] - t[1]).total_seconds() == 120
    assert [a["outcome"] for a in r.ev["acquisition"]["attempts"]] == ["capacity", "capacity", "created"]
    assert len({_flag(c, "--name") for c in sent}) == 3                 #: every request its own identity


def test_the_start_is_the_providers_creation_time_else_the_request(tmp_path, monkeypatch):
    """Cost, deadline and watchdog count from one instant: the provider's createdAt, read right after the create
    (with the datacenter), never the later moment the answer arrived; without a record, the request's start."""
    r, sent, now = _runner(tmp_path, monkeypatch, [json.dumps({"id": "pod0", "costPerHr": 1.09})])
    assert r.create() is True
    req = list(r._requests.values())[0]["epoch"]
    assert r.start_epoch == req + 30.0 and r.ev["start"]["basis"] == "provider createdAt"
    assert r.ev["pod_record"]["dataCenterId"] == "EU-RO-1"
    r2, _, _ = _runner(tmp_path / "b", monkeypatch, [json.dumps({"id": "pod0", "costPerHr": 1.09})])
    (tmp_path / "b").mkdir()
    r2.provider.pod_record = lambda pid: None
    assert r2.create() is True
    assert r2.start_epoch == list(r2._requests.values())[0]["epoch"] and r2.ev["start"]["basis"] == "request start"


def test_a_cli_without_the_flag_is_not_handed_it(tmp_path, monkeypatch):
    r, sent, _ = _runner(tmp_path, monkeypatch, [json.dumps({"id": "pod0", "cost": 1.09})])
    r.cli_facts = {"pod_create_terminate_after": False}
    assert r.create() is True
    assert "--terminate-after" not in sent[0] and r.ev["terminate_after_utc"] is None
    assert r.ev["billed_window_deadline_utc"]


def test_datacenter_groups_rotate_across_attempts_without_a_volume(tmp_path, monkeypatch):
    r, sent, _ = _runner(tmp_path, monkeypatch, [CAPACITY] * 3, data_center_ids="EU-NL-1,US-TX-3; US-CA-2")
    assert r.create() is False
    assert [_flag(c, "--data-center-ids") for c in sent] == ["EU-NL-1,US-TX-3", "US-CA-2", "EU-NL-1,US-TX-3"]
    assert SR.data_center_groups(" ;A, B;;C ") == ["A,B", "C"] and SR.data_center_groups(None) == []


def test_a_volume_keeps_its_own_datacenter(tmp_path, monkeypatch):
    r, sent, _ = _runner(tmp_path, monkeypatch, [CAPACITY], network_volume_id="vol", volume_mount_path="/v",
                         data_center_ids="EUR-IS-2")
    r.ws = "/workspace"
    r.create()
    assert sent[0].count("--data-center-ids") == 1 and _flag(sent[0], "--data-center-ids") == "EUR-IS-2"


# --- the three risks: a delayed creation, empty then present, several matches ----------------------------------------

def test_a_delayed_creation_is_owned_although_the_first_listings_are_empty(tmp_path, monkeypatch):
    """The create times out; the account reads empty twice, then the request's pod appears: it is owned, from its
    request's start window, and no second create is made."""
    r, sent, now = _runner(tmp_path, monkeypatch, [subprocess.TimeoutExpired("runpodctl", 300), CAPACITY],
                           script=("empty", "empty", [0]))
    assert r.create() is True
    assert len(sent) == 1 and r.pod_id == "pod0" and r._watchdog_for == "pod0"
    assert r.ev["acquisition"]["attempts"][0]["outcome"] == "adopted"


def test_one_empty_listing_never_settles_an_uncertain_create(tmp_path, monkeypatch):
    r, sent, now = _runner(tmp_path, monkeypatch, [UNCERTAIN, CAPACITY], script=("empty", "unknown"))
    assert r.create() is False
    assert len(sent) == 1 and r.ev["provider_state_unknown"] is True and r.pod_id == ""


def test_a_pod_that_appears_after_the_window_is_owned_before_any_second_create(tmp_path, monkeypatch):
    """Empty across the whole window (so the next attempt may proceed), then the first request's pod shows up in the
    listing read before the next create: owned, and the next create is never sent."""
    r, sent, now = _runner(tmp_path, monkeypatch, [UNCERTAIN, CAPACITY],
                           script=("empty", "empty", "empty", "empty", [0]))
    assert r.create() is True
    assert len(sent) == 1 and r.pod_id == "pod0"
    assert r.ev["acquisition"]["attempts"][-1]["outcome"] == "adopted late"


def test_every_resource_of_the_sessions_requests_is_registered_and_the_extras_released(tmp_path, monkeypatch):
    """The first create is uncertain and reads empty across the window; the second returns pod1; the listing then
    shows the first request's pod0 as well. pod1 is owned; pod0 is released with the provider's confirmation and
    billed from ITS request's start, and that cost is in the session's total."""
    r, sent, now = _runner(tmp_path, monkeypatch, [UNCERTAIN, json.dumps({"id": "pod1", "costPerHr": 1.09})],
                           script=("empty", "empty", "empty", "empty", "empty", [0, 1]))
    assert r.create() is True
    assert r.pod_id == "pod1" and len(sent) == 2
    extra = r.ev["extra_resources"]
    assert [e["pod_id"] for e in extra] == ["pod0"] and extra[0]["confirmed_not_billing"] is True
    assert r.provider.terminated == ["pod0"] and extra[0]["usd"] > 0
    r.release_and_confirm = lambda pid, why: {"confirmed_not_billing": True, "exists": False, "desired_status": None,
                                              "billing": False}
    r.teardown_now("end")
    assert r.ev["cost"]["extra_resources_usd"] == extra[0]["usd"]
    assert r.ev["cost"]["actual_usd"] == pytest.approx(r.usd() + extra[0]["usd"], abs=1e-4)


def test_two_pods_listed_for_one_uncertain_request_keep_the_earliest_and_release_the_other(tmp_path, monkeypatch):
    r, sent, now = _runner(tmp_path, monkeypatch, [UNCERTAIN], script=("empty", [0, 1]))
    real_pods = r.provider._pods

    def two(idx):                                                          #: the provider made two for one name
        p = real_pods([0])
        return p + [{**p[0], "id": "podX"}] if p else []
    r.provider._pods = two
    assert r.create() is True
    assert r.pod_id == "pod0" and [e["pod_id"] for e in r.ev["extra_resources"]] == ["podX"]


def test_an_unconfirmed_release_of_an_extra_tears_down_and_stops(tmp_path, monkeypatch):
    r, sent, now = _runner(tmp_path, monkeypatch, [UNCERTAIN, json.dumps({"id": "pod1", "costPerHr": 1.09})],
                           script=("empty", "empty", "empty", "empty", "empty", [0, 1]))
    r.provider.keep_alive = True                                            #: the extra never goes
    assert r.create() is False
    assert r.ev["provider_state_unknown"] is True and r.ev["extra_resources"][0]["confirmed_not_billing"] is False


def test_a_repeated_non_capacity_refusal_is_deterministic_and_stops(tmp_path, monkeypatch):
    bad = '{"error":"invalid gpu id","code":"bad_request","status":400}'
    r, sent, _ = _runner(tmp_path, monkeypatch, [bad, bad, bad])
    assert r.create() is False
    assert len(sent) == 2 and [a["outcome"] for a in r.ev["acquisition"]["attempts"]] == ["refused", "refused"]


@pytest.mark.parametrize("text,want", [
    ('{"id":"p1","costPerHr":1.09}', ("p1", 1.09)), ('{"id":"p1","cost":0.99}', ("p1", 0.99)),
    (CAPACITY + "\nUsage:\n  runpodctl pod create", ("", None)), ('noise "id": "p2" noise', ("p2", None))])
def test_the_create_answer_is_parsed_for_both_cli_generations(text, want):
    assert SR.answer_pod(text, text) == want
    assert bool(SR.CAPACITY_REFUSAL.search(CAPACITY)) and not SR.CAPACITY_REFUSAL.search(UNCERTAIN)
