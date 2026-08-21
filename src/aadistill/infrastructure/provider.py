"""Provider-level view of a pod: does it exist, and can it be made to stop.

Deliberately narrow. The watchdog that depends on this must keep working when
SSH is blocked, the remote driver is hung, the orchestrator log is silent,
training has crashed, or artifact collection has failed — so nothing here may
touch the pod itself. Every function talks to the provider's control plane and
to nothing else.

Two transports, chosen for what this project has actually verified (P14):

* **Polling** goes over the RunPod GraphQL API. Every launcher since E2 reads
  `pod(input:{podId:…}) { runtime { ports … } }` this way, so the query shape,
  the auth-in-query-string form and the `runtime: null`-means-starting semantics
  are all confirmed by use.
* **Termination** goes through `runpodctl remove pod` first, because that is the
  call every session in this project has actually made. The GraphQL
  `podTerminate` mutation is implemented as a fallback and is marked
  `verified=False` in the journal: it comes from RunPod's public schema and has
  never been exercised against the live endpoint from this repo.

The ground truth for "did the pod stop" is neither of those. It is a subsequent
**poll** showing the pod gone or exited. A termination call that returns 200 and
leaves a billing pod is exactly the failure E6b's `--terminate-after` was: a
deadline set on the pod at 00:28:47 that was still `RUNNING` at 00:34. Never
trust the request; confirm the state.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

# RunPod's edge rejects the default `Python-urllib/3.x` User-Agent with **403
# Forbidden** — every query, including ones that succeed byte-for-byte from
# curl. Found on 2026-08-09 while preparing the control-plane canary, before any
# pod existed.
#
# It matters more than a header usually would. `PodState.billing` treats an
# unanswered poll as "still billing", which is the safe direction for deciding
# to terminate — but it also means the watchdog's *verification* poll could
# never confirm the pod was gone. Every session would have ended in
# `TERMINATION_FAILED` with a pod that had actually died. The launchers never
# hit this because they shell out to curl.
USER_AGENT = "aadistill-watchdog/1.0 (+https://github.com/AlphaAvatar/AlphaAvatar-distill)"

# Provider states in which a pod is no longer accruing GPU time. `EXITED` is
# included: a stopped pod still holds disk, but the GPU meter — the only thing
# these thresholds are denominated in — has stopped.
GONE_STATUSES = frozenset({"TERMINATED", "EXITED", "DEAD"})

#: What counts as "the control plane did not answer", as opposed to "it answered
#: and the answer is no". Named once and used by every caller, because the whole
#: failure mode is one caller classifying differently from another.
#:
#: `URLError` covers the TLS and connection-reset family; `OSError` its parent,
#: for a socket that dies without a URL wrapper; `ValueError` covers
#: `JSONDecodeError`, which is a truncated body and therefore also a transport
#: symptom; `TimeoutError` is the deadline. Recovery-continuation attempt 1
#: (2026-08-21) died on `SSL: UNEXPECTED_EOF_WHILE_READING` against an endpoint
#: measured at 25% failure, in a launcher path that caught none of these.
TRANSIENT_TRANSPORT: tuple[type[BaseException], ...] = (
    urllib.error.URLError, OSError, ValueError, TimeoutError)


@dataclass(frozen=True)
class Observation:
    """One control-plane answer, or the absence of one. Never an exception.

    The distinction that matters is `ok`: **a failed observation is unknown
    state, not a negative answer.** A caller that treats `ok=False` as "no ports
    yet" merely wastes a poll; a caller that treats it as "the pod is gone"
    abandons a billing pod, which is why `PodState.billing` reports unknown as
    still billing and why this type refuses to collapse the two.
    """

    ok: bool
    data: dict | None = None
    error: str | None = None
    #: True when the failure is the transport or a malformed body — retryable.
    #: A GraphQL `errors` array is also transient by this definition: it is the
    #: server declining to answer, and `get()` has always refused to read it as
    #: absence.
    transient: bool = False

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True)
class PodState:
    """What the control plane says about one pod at one instant."""

    pod_id: str
    exists: bool
    desired_status: str | None = None
    runtime_ready: bool = False
    cost_per_hr: float | None = None
    error: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def billing(self) -> bool:
        """Is this pod plausibly still costing money?

        Unknown counts as billing. An API error, a timeout or an unparseable
        response must never be read as "the pod is gone" — that is the same
        inference as reading a silent orchestrator log as an idle session, and
        it fails in the direction that costs money.
        """
        if self.error is not None:
            return True
        if not self.exists:
            return False
        if self.desired_status is None:
            return True
        return self.desired_status.upper() not in GONE_STATUSES


@dataclass(frozen=True)
class TerminationAttempt:
    """One attempt to stop a pod, recorded whether or not it worked."""

    method: str
    verified_transport: bool
    ok: bool
    returncode: int | None = None
    response: str = ""
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "verified_transport": self.verified_transport,
            "ok": self.ok,
            "returncode": self.returncode,
            "response": self.response[:2000],
            "error": self.error,
        }


class PodProvider(Protocol):
    """The whole surface the watchdog is allowed to depend on."""

    def get(self, pod_id: str) -> PodState: ...

    def observe(self, query: str) -> Observation: ...

    def terminate(self, pod_id: str) -> list[TerminationAttempt]: ...


class RunPodProvider:
    """RunPod control plane: GraphQL for state, CLI-then-GraphQL to terminate."""

    def __init__(self, api_key: str, *, timeout: float = 30.0,
                 runpodctl: str | None = None) -> None:
        if not api_key.strip():
            raise ValueError("empty RunPod API key")
        self._key = api_key.strip()
        self.timeout = timeout
        self._cli = runpodctl or shutil.which("runpodctl")

    # -- transport ---------------------------------------------------------
    def _gql(self, query: str) -> dict:
        req = urllib.request.Request(
            f"{RUNPOD_GRAPHQL}?api_key={self._key}",
            data=json.dumps({"query": query}).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": USER_AGENT},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def observe(self, query: str) -> Observation:
        """One `_gql` attempt, classified exactly the way `get()` classifies it.

        `get()` has always been the resilient path — *"Never raises. A watchdog
        that dies on a transient 502 is not a backstop."* — but it answers one
        fixed question about one pod. Every other launcher query went straight to
        `_gql`, which raises, so the same control-plane blip that `get()` absorbs
        killed the launcher outright.

        This is that classification, decoupled from that one query, so a caller
        asking anything else gets the same treatment rather than a second,
        divergent copy of the rules.
        """
        try:
            body = self._gql(query)
        except TRANSIENT_TRANSPORT as exc:
            return Observation(ok=False, transient=True,
                               error=f"{type(exc).__name__}: {exc}")
        if body.get("errors"):
            # The server declining to answer. Not an answer of "no".
            return Observation(ok=False, transient=True,
                               error=json.dumps(body["errors"])[:500])
        return Observation(ok=True, data=body.get("data") or {})

    # -- state -------------------------------------------------------------
    def get(self, pod_id: str) -> PodState:
        """Never raises. A watchdog that dies on a transient 502 is not a backstop."""
        query = (
            "query { pod(input:{podId:\"%s\"}) { id desiredStatus costPerHr "
            "runtime { uptimeInSeconds } } }" % pod_id
        )
        try:
            body = self._gql(query)
        except TRANSIENT_TRANSPORT as exc:
            return PodState(pod_id=pod_id, exists=True,
                            error=f"{type(exc).__name__}: {exc}")
        if body.get("errors"):
            # A query error is not evidence of absence.
            return PodState(pod_id=pod_id, exists=True,
                            error=json.dumps(body["errors"])[:500], raw=body)
        pod = (body.get("data") or {}).get("pod")
        if pod is None:
            # RunPod returns a null pod for an id it no longer knows. This is
            # the only response that means "gone".
            return PodState(pod_id=pod_id, exists=False,
                            desired_status="TERMINATED", raw=body)
        cost = pod.get("costPerHr")
        return PodState(
            pod_id=pod_id,
            exists=True,
            desired_status=pod.get("desiredStatus"),
            runtime_ready=bool(pod.get("runtime")),
            cost_per_hr=float(cost) if cost is not None else None,
            raw=body,
        )

    # -- termination -------------------------------------------------------
    def terminate(self, pod_id: str) -> list[TerminationAttempt]:
        """Try every transport available and report all of them.

        Returns the attempts rather than a bare bool because the caller journals
        them: when a pod survives a termination round, which transports were
        tried and what they said is the whole diagnostic.
        """
        attempts: list[TerminationAttempt] = []
        if self._cli:
            attempts.append(self._terminate_cli(pod_id))
            if attempts[-1].ok:
                return attempts
        attempts.append(self._terminate_gql(pod_id))
        return attempts

    def _terminate_cli(self, pod_id: str) -> TerminationAttempt:
        try:
            proc = subprocess.run(
                [self._cli, "remove", "pod", pod_id],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return TerminationAttempt(
                method="runpodctl remove pod", verified_transport=True,
                ok=False, error=f"{type(exc).__name__}: {exc}")
        return TerminationAttempt(
            method="runpodctl remove pod", verified_transport=True,
            ok=proc.returncode == 0, returncode=proc.returncode,
            response=(proc.stdout + proc.stderr).strip())

    def _terminate_gql(self, pod_id: str) -> TerminationAttempt:
        query = "mutation { podTerminate(input:{podId:\"%s\"}) }" % pod_id
        try:
            body = self._gql(query)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            return TerminationAttempt(
                method="graphql podTerminate", verified_transport=False,
                ok=False, error=f"{type(exc).__name__}: {exc}")
        return TerminationAttempt(
            method="graphql podTerminate", verified_transport=False,
            ok=not body.get("errors"), response=json.dumps(body)[:2000],
            error=json.dumps(body["errors"])[:500] if body.get("errors") else None)


class SimulatedProvider:
    """An in-memory control plane, for dry runs and for the failure simulator.

    It is not a mock in the testing sense — the watchdog CLI accepts it via
    ``--simulate`` so a session's thresholds can be rehearsed end to end without
    creating a pod. The failure knobs exist because the behaviours worth
    rehearsing are the ones that cost money: a termination call that succeeds
    while the pod keeps running is precisely E6b's `--terminate-after`.
    """

    def __init__(self, pod_id: str, *, exists: bool = True,
                 desired_status: str = "RUNNING", cost_per_hr: float = 0.99,
                 terminate_failures: int = 0, ignored_terminations: int = 0,
                 poll_errors: int = 0) -> None:
        self.pod_id = pod_id
        self.exists = exists
        self.desired_status = desired_status
        self.cost_per_hr = cost_per_hr
        self.terminate_failures = terminate_failures
        # Terminations the provider accepts and then does not act on.
        self.ignored_terminations = ignored_terminations
        self.poll_errors = poll_errors
        self.calls: list[str] = []

    def get(self, pod_id: str) -> PodState:
        self.calls.append("get")
        if self.poll_errors > 0:
            self.poll_errors -= 1
            return PodState(pod_id=pod_id, exists=True,
                            error="SimulatedError: control plane unreachable")
        if not self.exists:
            return PodState(pod_id=pod_id, exists=False,
                            desired_status="TERMINATED")
        return PodState(pod_id=pod_id, exists=True,
                        desired_status=self.desired_status, runtime_ready=True,
                        cost_per_hr=self.cost_per_hr)

    def observe(self, query: str) -> Observation:
        """Shares `poll_errors` with `get`, so one knob rehearses a control
        plane that is flaky for every question, not only for pod state."""
        self.calls.append("observe")
        if self.poll_errors > 0:
            self.poll_errors -= 1
            return Observation(ok=False, transient=True,
                               error="SimulatedError: control plane unreachable")
        return Observation(ok=True, data={})

    def terminate(self, pod_id: str) -> list[TerminationAttempt]:
        self.calls.append("terminate")
        if self.terminate_failures > 0:
            self.terminate_failures -= 1
            return [TerminationAttempt(
                method="simulated", verified_transport=True, ok=False,
                returncode=1, error="SimulatedError: termination call failed")]
        if self.ignored_terminations > 0:
            self.ignored_terminations -= 1
            return [TerminationAttempt(
                method="simulated", verified_transport=True, ok=True,
                returncode=0, response="accepted (pod keeps running)")]
        self.exists = False
        self.desired_status = "TERMINATED"
        return [TerminationAttempt(
            method="simulated", verified_transport=True, ok=True,
            returncode=0, response="terminated")]


def read_api_key(config_path: str) -> str:
    """RunPod CLI config values are single-quoted under a lowercase `apikey`.

    Stripping only `"` returns a key with a leading quote and GraphQL answers
    `{"error":{}}` — a failure that reads like an auth problem and is a parsing
    problem (`scripts/pod/AGENTS.md`).
    """
    import re

    text = open(config_path).read()
    m = re.search(r"apikey\s*=\s*(.+)", text, re.I)
    if not m:
        raise ValueError(f"no apikey in {config_path}")
    return m.group(1).strip().strip('"').strip("'")
