"""Every launcher control-plane call must survive a flaky control plane.

Recovery-continuation attempt 1 (2026-08-21) passed every gate, created a pod,
and died **27 seconds later** on `URLError: SSL UNEXPECTED_EOF` in the readiness
poll — $0.01, no stage run. Measured immediately afterwards at $0, the RunPod
GraphQL endpoint was failing **5 of 20 requests (25%)**.

`SessionRunner` had three direct `provider._gql()` calls, and `_gql` raises.
`provider.get()` — the one path that was already resilient — carries the reason
in its own docstring: *"Never raises. A watchdog that dies on a transient 502 is
not a backstop."* The launcher was never given the same treatment, and two of
the three sites run **while a pod is billing**.

These tests drive the **real methods** against a scripted control plane. A test
that only asserted `try:` appears in the source would pass against a handler
that swallows the error and returns the wrong answer, which is the more
expensive bug.
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure import session_runner as SR  # noqa: E402
from aadistill.infrastructure.provider import (  # noqa: E402
    TRANSIENT_TRANSPORT, Observation, PodState, SimulatedProvider,
)

RUNNER_SRC = REPO / "src/aadistill/infrastructure/session_runner.py"

PORTS_OK = {"pod": {"runtime": {"ports": [
    {"ip": "1.2.3.4", "publicPort": "10192", "privatePort": 22, "type": "tcp"}]}}}
IMAGE_OK = {"pod": {"imageName": "runpod/pytorch:1.1.0", "machine": {"podHostId": "h"}}}


class ScriptedProvider:
    """A control plane that answers from a script, and counts what it was asked.

    `script` is a list of `Observation`s; the last one repeats forever, so
    "transient, transient, then fine" and "broken for good" are both one line.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.terminated = []

    def observe(self, query: str) -> Observation:
        obs = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return obs

    def get(self, pod_id):
        return PodState(pod_id=pod_id, exists=True, desired_status="RUNNING")

    def terminate(self, pod_id):
        self.terminated.append(pod_id)
        return []


def blip(error="URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING]>"):
    return Observation(ok=False, transient=True, error=error)


def answer(data):
    return Observation(ok=True, data=data)


def runner(provider, **over):
    """A `SessionRunner` with only what these three methods touch.

    Built without `__init__` on purpose: `__init__` verifies the harness digest
    against a real authorization, and this change *invalidates* the consumed
    continuation authorization by design. The methods under test are the real
    ones either way.
    """
    r = object.__new__(SR.SessionRunner)
    r.provider = provider
    r.pod_id = "podtest"
    r.price = 0.99
    r.start_epoch = time.time()
    r.ev = {"timeline": []}
    r.say = lambda m: r.ev.setdefault("said", []).append(m)
    r.a = SimpleNamespace(
        gpu="NVIDIA L40S", max_price=0.99,
        create_attempts=over.pop("create_attempts", 4),
        create_retry_seconds=0.0,
        startup_limit_min=over.pop("startup_limit_min", 15.0),
        image="runpod/pytorch:1.1.0")
    for k, v in over.items():
        setattr(r, k, v)
    return r


SLEPT: list[float] = []


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    """These methods sleep between retries; the sleeps are not what is tested.

    Recorded rather than discarded, because *how much* deadline a failed
    observation consumes is itself a property worth pinning.
    """
    SLEPT.clear()
    monkeypatch.setattr(SR.time, "sleep", lambda s: SLEPT.append(s))


# --- the classification is shared, not re-derived ---------------------------

def test_the_transient_set_is_what_get_already_recognised():
    """One named tuple, used by `get()` and by `observe()`. The failure mode is
    one caller classifying differently from another."""
    for exc in (OSError, ValueError, TimeoutError):
        assert exc in TRANSIENT_TRANSPORT
    import urllib.error
    assert urllib.error.URLError in TRANSIENT_TRANSPORT
    src = (REPO / "src/aadistill/infrastructure/provider.py").read_text()
    assert src.count("except TRANSIENT_TRANSPORT") >= 2, (
        "a caller is catching its own hand-written tuple again")


def test_observe_never_raises_on_any_transient_class(monkeypatch):
    """Driven through the real `RunPodProvider.observe`, one exception class at
    a time — not a mocked-out stand-in for it."""
    from aadistill.infrastructure.provider import RunPodProvider
    import urllib.error

    p = RunPodProvider("k" * 20)
    for exc in (urllib.error.URLError("ssl eof"), OSError("reset by peer"),
                ValueError("Expecting value: line 1"), TimeoutError("deadline")):
        monkeypatch.setattr(p, "_gql", lambda q, e=exc: (_ for _ in ()).throw(e))
        obs = p.observe("query { x }")
        assert obs.ok is False and obs.transient is True
        assert type(exc).__name__ in obs.error


def test_a_graphql_errors_body_is_not_an_answer_of_no(monkeypatch):
    """`get()` has always refused to read a query error as absence. `observe`
    must classify it the same way, or a caller would treat 'the server declined'
    as 'there are no ports'."""
    from aadistill.infrastructure.provider import RunPodProvider

    p = RunPodProvider("k" * 20)
    monkeypatch.setattr(p, "_gql", lambda q: {"errors": [{"message": "boom"}]})
    obs = p.observe("query { x }")
    assert obs.ok is False and obs.transient is True and "boom" in obs.error
    # And the same body through `get` still reports the pod as present.
    st = p.get("podtest")
    assert st.exists is True and st.billing is True


# --- 1. check_gpu_offered: never create a pod, never crash, all at $0 -------

def test_pricing_recovers_from_transient_failures_then_succeeds():
    prov = ScriptedProvider([blip(), blip(), answer(
        {"gpuTypes": [{"id": "NVIDIA L40S", "securePrice": 0.79,
                       "lowestPrice": {"stockStatus": "High"}}]})])
    r = runner(prov)
    assert r.check_gpu_offered() is True
    assert r.price == 0.79
    assert prov.calls == 3
    retries = r.ev["control_plane_retries"]
    assert [x["where"] for x in retries] == ["check_gpu_offered"] * 2
    assert all(x["billing"] is False for x in retries), (
        "pricing runs before any pod exists; nothing here is billing")


def test_pricing_that_never_answers_aborts_cleanly_without_a_pod():
    prov = ScriptedProvider([blip()])
    r = runner(prov, create_attempts=3)
    assert r.check_gpu_offered() is False           # returns, does not raise
    assert prov.calls == 3
    assert prov.terminated == [], "no pod exists to terminate, and none was made"
    said = " ".join(r.ev["said"])
    assert "No pod was created" in said
    assert "not offered" not in said, (
        "an unanswered price query is unknown, not evidence the GPU is absent")


def test_pricing_does_not_crash_through_the_launcher():
    """The exact shape of attempt 1's death, one step earlier: the exception
    must not escape into `run()`."""
    prov = ScriptedProvider([blip("OSError: [Errno 104] Connection reset by peer")])
    r = runner(prov, create_attempts=2)
    assert r.check_gpu_offered() is False


def test_pricing_still_refuses_a_gpu_that_is_genuinely_too_expensive():
    """Resilience must not have blunted the check it wraps."""
    prov = ScriptedProvider([answer(
        {"gpuTypes": [{"id": "NVIDIA L40S", "securePrice": 4.20,
                       "lowestPrice": {"stockStatus": "Low"}}]})])
    r = runner(prov)
    assert r.check_gpu_offered() is False
    assert "above the priced" in " ".join(r.ev["said"])


def test_pricing_still_refuses_a_gpu_that_is_genuinely_not_offered():
    prov = ScriptedProvider([answer({"gpuTypes": []})])
    r = runner(prov)
    assert r.check_gpu_offered() is False
    assert "not offered" in " ".join(r.ev["said"])


# --- 2. wait_endpoint: unknown is not absence, and the pod is billing -------

def test_the_endpoint_poll_survives_the_failure_that_killed_attempt_1():
    """One `SSL: UNEXPECTED_EOF` used to end the session. It must now be a
    logged retry that keeps polling."""
    prov = ScriptedProvider([blip(), blip(), blip(), answer(PORTS_OK)])
    r = runner(prov)
    assert r.wait_endpoint(time.time() + 600) == ("1.2.3.4", "10192")
    assert prov.calls == 4
    retries = r.ev["control_plane_retries"]
    assert len(retries) == 3
    assert all(x["where"] == "wait_endpoint" for x in retries)
    assert all(x["billing"] is True for x in retries), (
        "this poll runs against a pod that is already costing money")


def test_a_long_run_of_failures_still_recovers_within_the_deadline():
    """At a measured 25% failure rate a real run sees streaks. Twenty in a row
    must not be treated as an answer."""
    prov = ScriptedProvider([*[blip()] * 20, answer(PORTS_OK)])
    r = runner(prov)
    assert r.wait_endpoint(time.time() + 600) == ("1.2.3.4", "10192")
    assert len(r.ev["control_plane_retries"]) == 20


def test_a_failed_observation_costs_exactly_one_poll_interval():
    """Found by mutation: deleting the `continue` after a failed observation
    still loops and still recovers, so every outcome-level test above passed —
    but the failure then falls into the port scan with no data, sleeps a second
    time, and advances the progress counter. Under a flaky control plane that
    spends the startup deadline at **twice** the intended rate and reports
    startup progress that never happened.
    """
    prov = ScriptedProvider([blip(), blip(), blip(), answer(PORTS_OK)])
    r = runner(prov)
    assert r.wait_endpoint(time.time() + 600) == ("1.2.3.4", "10192")
    assert SLEPT == [10, 10, 10], (
        f"three failures should cost three poll intervals, not {SLEPT}")
    # And a failure is not a poll: the "starting (Ns)" progress line counts
    # observations that actually answered.
    assert not any("starting (" in m for m in r.ev["said"])


def test_the_progress_counter_tracks_answers_not_attempts():
    """Six *answered* polls with no ports yet is genuine startup progress and
    should say so; six unanswered ones are not."""
    no_ports = answer({"pod": {"runtime": None}})
    prov = ScriptedProvider([*[no_ports] * 6, answer(PORTS_OK)])
    r = runner(prov)
    assert r.wait_endpoint(time.time() + 600) == ("1.2.3.4", "10192")
    assert any("starting (60s)" in m for m in r.ev["said"])


def test_a_failed_observation_is_never_read_as_the_pod_being_gone():
    """The whole point. A caller that concluded 'no ports, therefore gone' would
    abandon a billing pod."""
    prov = ScriptedProvider([blip()])
    r = runner(prov)
    assert r.wait_endpoint(time.time() + 0.05) is None      # deadline, not gone
    assert prov.terminated == []
    said = " ".join(r.ev["said"])
    assert "gone" not in said.lower()


def test_persistent_failure_terminates_cleanly_at_the_existing_deadline():
    prov = ScriptedProvider([blip()])
    r = runner(prov)
    t0 = time.time()
    assert r.wait_endpoint(t0 + 0.2) is None
    assert time.time() - t0 < 30, "it must stop at the deadline it was given"


def test_the_endpoint_poll_uses_the_callers_deadline_not_a_fresh_one():
    """`setup_on_draw` owns one startup bound for the whole draw. If this method
    re-derived its own, the operator's `startup_limit_min` would silently cover
    twice the wall clock."""
    prov = ScriptedProvider([blip()])
    r = runner(prov, startup_limit_min=999.0)
    t0 = time.time()
    assert r.wait_endpoint(t0 + 0.2) is None
    assert time.time() - t0 < 30


def test_setup_on_draw_shares_one_startup_deadline_between_both_observations():
    """Structural, because the alternative is two `startup_limit_min` windows."""
    tree = ast.parse(RUNNER_SRC.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "setup_on_draw")
    body = ast.unparse(fn)
    assert "startup_deadline = time.time() + self.a.startup_limit_min * 60" in body
    assert "self.wait_endpoint(startup_deadline)" in body
    assert "self.read_image_digest(target, startup_deadline)" in body


# --- 3. read_image_digest: fail closed, never an unverified identity --------

class _Target:
    def run(self, cmd, timeout=None):
        return SimpleNamespace(stdout="580.126.09\n", returncode=0)


def test_the_image_observation_recovers_from_transient_failures():
    prov = ScriptedProvider([blip(), blip(), answer(IMAGE_OK)])
    r = runner(prov)
    digest = r.read_image_digest(_Target(), time.time() + 600)
    assert digest == "runpod/pytorch:1.1.0@580.126.09"
    assert len(r.ev["control_plane_retries"]) == 2
    assert all(x["billing"] is True for x in r.ev["control_plane_retries"])


def test_an_unconfirmable_image_fails_closed_rather_than_defaulting():
    """It used to fall back to `self.a.image` — the image we ASKED for. That is
    what we requested, not what is running, and it must never reach a
    reproducibility record as though it were an observation."""
    prov = ScriptedProvider([blip()])
    r = runner(prov)
    with pytest.raises(SR.ImageIdentityUnavailable):
        r.read_image_digest(_Target(), time.time() + 0.05)


def test_a_provider_that_reports_no_image_name_also_fails_closed():
    prov = ScriptedProvider([answer({"pod": {"imageName": None}})])
    r = runner(prov)
    with pytest.raises(SR.ImageIdentityUnavailable) as exc:
        r.read_image_digest(_Target(), time.time() + 600)
    assert "not what is running" in str(exc.value)


def test_the_requested_image_is_never_substituted_for_the_observed_one():
    """Mutation-proof for the specific fallback that was removed."""
    src = RUNNER_SRC.read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "read_image_digest")
    body = ast.unparse(fn)
    assert 'imageName' in body
    assert "self.a.image" not in body.split('"""')[-1], (
        "the requested image is back in the observation path")


# --- 4. the driver cannot start without a confirmed image identity ----------

def test_a_failed_image_identity_stops_the_draw_before_setup_runs():
    """`setup_on_draw` must return a non-ok outcome, and it must not be one of
    the redrawable ones: the control plane, not this host, is what failed."""
    tree = ast.parse(RUNNER_SRC.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "setup_on_draw")
    body = ast.unparse(fn)
    assert "except ImageIdentityUnavailable" in body
    assert "return 'no_image_identity'" in body
    # It is raised before the setup script is RUN. Not before it is copied:
    # the scp lands earlier in the method and copying a file executes nothing,
    # so `autoinit_preflight_setup.sh` is the wrong landmark for this claim.
    # `parse_setup_probe` only has anything to read once setup has run.
    assert body.index("no_image_identity") < body.index("parse_setup_probe")
    assert body.index("no_image_identity") < body.index("setup_timeout_s")


def test_no_image_identity_is_not_a_redrawable_outcome():
    fn = next(n for n in ast.walk(ast.parse(RUNNER_SRC.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    body = ast.unparse(fn)
    assert "('cold', 'no_endpoint')" in body or '"cold", "no_endpoint"' in body
    assert "no_image_identity" not in body, (
        "an unconfirmable image must tear down, not redraw onto another host")


def test_an_aborting_outcome_tears_down_before_returning():
    fn = next(n for n in ast.walk(ast.parse(RUNNER_SRC.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    body = ast.unparse(fn)
    assert "self.teardown_now" in body
    assert body.index("ABORT after draw") < body.index("self.teardown_now")


# --- 5. the class cannot be silently reintroduced ---------------------------

def test_the_runner_makes_no_raw_gql_call():
    """The guard the maintainer asked for. A future single-shot `_gql` in a paid
    path is exactly attempt 1, and would otherwise pass every other test here.

    `observe()` is the only sanctioned entry point: it never raises and it
    classifies a declined answer as unknown rather than as 'no'.
    """
    offenders = []
    for node in ast.walk(ast.parse(RUNNER_SRC.read_text())):
        if isinstance(node, ast.Attribute) and node.attr == "_gql":
            offenders.append(getattr(node, "lineno", "?"))
    assert not offenders, (
        f"session_runner.py calls provider._gql directly at line(s) {offenders}. "
        "Use provider.observe(), which never raises and reports a declined "
        "answer as unknown state rather than as a negative answer.")


def test_every_control_plane_read_in_the_runner_goes_through_observe_or_get():
    """The positive half: the calls exist and they use the resilient paths."""
    src = RUNNER_SRC.read_text()
    assert src.count("self.provider.observe(") == 3, (
        "expected exactly the three launcher queries: price, ports, image")
    assert "self.provider.get(" in src


def test_the_simulated_provider_can_rehearse_a_flaky_control_plane():
    """`poll_errors` now drives `observe` too, so a dry run can rehearse the
    failure that cost a pod rather than only pod-state flakiness."""
    sim = SimulatedProvider("p", poll_errors=2)
    assert sim.observe("q").ok is False
    assert sim.observe("q").ok is False
    assert sim.observe("q").ok is True
