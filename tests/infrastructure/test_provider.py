"""The provider client's transport details, one of which was a live defect.

RunPod's edge returns **403 Forbidden** to the default `Python-urllib/3.x`
User-Agent on every GraphQL query, including ones that succeed byte-for-byte
from curl. Found 2026-08-09 while preparing the control-plane canary, before any
pod existed.

Why a header mattered: `PodState.billing` treats an unanswered poll as "still
billing", which is the safe direction for *deciding* to terminate — but it also
means the watchdog's **verification** poll could never confirm a pod was gone.
Every session would have ended in `TERMINATION_FAILED` against a pod that had in
fact died. The launchers never hit it because they shell out to curl.
"""

import json
from unittest import mock

import pytest

from aadistill.infrastructure.provider import (
    USER_AGENT, PodState, RunPodProvider, SimulatedProvider, read_api_key,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_the_user_agent_is_not_the_urllib_default():
    assert USER_AGENT
    assert not USER_AGENT.lower().startswith("python-urllib"), (
        "RunPod 403s the urllib default; a poll that cannot answer makes the "
        "watchdog unable to verify a pod is gone")
    assert "aadistill" in USER_AGENT


def test_every_graphql_request_sends_it():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.headers)
        captured["url"] = req.full_url
        return FakeResponse({"data": {"pod": None}})

    provider = RunPodProvider("test-key")
    with mock.patch("urllib.request.urlopen", fake_urlopen):
        provider.get("pod-1")
    # urllib title-cases header names.
    assert captured["headers"].get("User-agent") == USER_AGENT
    assert captured["headers"].get("Content-type") == "application/json"
    assert "api_key=test-key" in captured["url"]


def test_a_null_pod_is_the_only_response_meaning_gone():
    provider = RunPodProvider("k")
    with mock.patch("urllib.request.urlopen",
                    lambda req, timeout=None: FakeResponse({"data": {"pod": None}})):
        st = provider.get("gone")
    assert st.exists is False and st.billing is False


def test_a_graphql_error_is_not_evidence_of_absence():
    provider = RunPodProvider("k")
    with mock.patch("urllib.request.urlopen",
                    lambda req, timeout=None: FakeResponse(
                        {"errors": [{"message": "rate limited"}]})):
        st = provider.get("pod-1")
    assert st.exists is True and st.billing is True and st.error


def test_a_transport_failure_is_not_evidence_of_absence():
    provider = RunPodProvider("k")

    def boom(req, timeout=None):
        raise OSError("connection reset")

    with mock.patch("urllib.request.urlopen", boom):
        st = provider.get("pod-1")
    assert st.billing is True
    assert "OSError" in st.error


def test_a_running_pod_is_billing():
    provider = RunPodProvider("k")
    payload = {"data": {"pod": {"id": "p", "desiredStatus": "RUNNING",
                                "costPerHr": 0.99, "runtime": {"uptimeInSeconds": 60}}}}
    with mock.patch("urllib.request.urlopen",
                    lambda req, timeout=None: FakeResponse(payload)):
        st = provider.get("p")
    assert st.exists and st.billing and st.runtime_ready
    assert st.cost_per_hr == 0.99


def test_an_exited_pod_is_not_billing():
    assert PodState("p", exists=True, desired_status="EXITED").billing is False
    assert PodState("p", exists=True, desired_status="TERMINATED").billing is False
    assert PodState("p", exists=True, desired_status="RUNNING").billing is True


def test_termination_tries_the_cli_first_then_graphql():
    """The CLI path is the one this project has actually used; GraphQL is the
    unverified fallback and must be reported as such."""
    provider = RunPodProvider("k", runpodctl="/nonexistent/runpodctl")
    with mock.patch("urllib.request.urlopen",
                    lambda req, timeout=None: FakeResponse({"data": {"podTerminate": None}})):
        attempts = provider.terminate("p")
    assert [a.method for a in attempts] == ["runpodctl remove pod",
                                            "graphql podTerminate"]
    assert attempts[0].ok is False, "a nonexistent binary cannot succeed"
    assert attempts[0].verified_transport is True
    assert attempts[1].verified_transport is False, (
        "the mutation has never run against the live endpoint; the journal must "
        "say so")
    assert attempts[1].ok is True


def test_a_broken_cli_path_is_how_the_canary_forces_the_fallback():
    """No provider state is altered to induce the failure — only this process's
    view of the CLI."""
    provider = RunPodProvider("k", runpodctl="/nonexistent/runpodctl-canary")
    attempt = provider._terminate_cli("p")
    assert attempt.ok is False
    assert "FileNotFoundError" in attempt.error or "NotADirectoryError" in attempt.error


def test_read_api_key_strips_single_quotes(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("apikey = 'rpa_SECRET'\n")
    assert read_api_key(str(p)) == "rpa_SECRET"


def test_read_api_key_refuses_a_file_without_one(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("something = 'else'\n")
    with pytest.raises(ValueError, match="no apikey"):
        read_api_key(str(p))


def test_an_empty_key_is_refused():
    with pytest.raises(ValueError, match="empty RunPod API key"):
        RunPodProvider("   ")


def test_the_simulator_matches_the_real_client_surface():
    """The simulator stands in for the real thing in the failure replay; if the
    surfaces drift, that test stops meaning anything."""
    sim, real = SimulatedProvider("p"), RunPodProvider("k")
    for name in ("get", "terminate"):
        assert callable(getattr(sim, name)) and callable(getattr(real, name))
    st = sim.get("p")
    assert isinstance(st, PodState)
    assert {a.method for a in sim.terminate("p")} == {"simulated"}
