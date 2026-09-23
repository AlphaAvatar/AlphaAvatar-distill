"""`stage_c2_probes_to_volume.py` — the transport tool, and what it refuses.

Everything expensive about this tool is a refusal. It creates one provider
resource and then moves 22 GiB through it for many hours, so each thing it
declines to start is worth more than anything it does: the run that discovered
the host-load gate spent seventy minutes reaching 6% of one probe before its
rate decayed to ten days' worth.

The provider types here are the REAL `PodState` and `TerminationAttempt`. Three
defects in the teardown recorder — a `detail` field, a `status` field, and an
exception raised while recording a teardown that had already happened — all
came from inventing attributes on types nobody read, so a fake shaped by hand
would reproduce the mistake rather than catch it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.infrastructure.provider import (  # noqa: E402
    PodState, TerminationAttempt,
)

import stage_c2_probes_to_volume as S  # noqa: E402


class _Args:
    def __init__(self, **kw) -> None:
        defaults = {"max_usd": 3.0, "price_cap": 0.4, "max_load_per_cpu": 8.0,
                    "floor_mb_per_second": 0.15, "floor_probe_mib": 32,
                    "probe_timeout_min": 180.0, "volume_gb": 40}
        for k, v in {**defaults, **kw}.items():
            setattr(self, k, v)


class _Provider:
    """The provider surface the tool uses, answering with the real types."""

    def __init__(self, *, states: list[PodState], attempts=None,
                 terminate_raises: bool = False) -> None:
        self.states = list(states)
        self.attempts = attempts if attempts is not None else [
            TerminationAttempt(method="runpodctl remove pod",
                               verified_transport=True, ok=True, returncode=0,
                               response="removed")]
        self.terminate_raises = terminate_raises
        self.terminated: list[str] = []

    def terminate(self, pod_id):
        if self.terminate_raises:
            raise RuntimeError("control plane exploded")
        self.terminated.append(pod_id)
        return self.attempts

    def get(self, pod_id):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


def _pod(provider, *, run=None, **kw):
    p = S.StagingPod(_Args(**kw), provider, {})
    p.pod_id = "podA"
    p.price = 0.06
    p.start_epoch = S.time.time() - 60
    p.endpoint = ("1.2.3.4", "2222")
    if run is not None:
        p.run = run                                    # type: ignore[method-assign]
    return p


def _result(stdout="", rc=0):
    return type("R", (), {"returncode": rc, "stdout": stdout, "stderr": ""})()


# ---------------------------------------------------------------------------
# the host-load gate
# ---------------------------------------------------------------------------

def test_the_load_limit_is_a_backstop_and_not_a_filter():
    """It began as a filter at 8.0 per vCPU and had to stop being one.

    Load in the sixties is the NORM for the cheapest CPU flavour in this
    datacenter, and the draw that measured 65 moved 1.5 GB successfully before
    it decayed. A gate at 8.0 refused the next draw in forty seconds and would
    have refused every draw that works. What decides is throughput; this only
    catches the absurd.
    """
    assert S.main.__doc__ is None or True
    import stage_c2_probes_to_volume as mod
    src = (REPO / "scripts/autoinit/stage_c2_probes_to_volume.py").read_text()
    assert '"--max-load-per-cpu", type=float, default=100.0' in src, (
        "the load limit is back to filtering draws that would have worked")
    #: A draw at the observed norm passes.
    pod = _pod(_Provider(states=[PodState("podA", exists=False)]),
               max_load_per_cpu=100.0,
               run=lambda cmd, timeout=0: _result("65.17 56.75 56.34 74/6555 444\n2\n"))
    health = pod.require_healthy_host()
    assert health["load_per_cpu"] == 32.59
    #: And the mechanism still refuses when a caller sets a real limit.
    strict = _pod(_Provider(states=[PodState("podA", exists=False)]),
                  max_load_per_cpu=8.0,
                  run=lambda cmd, timeout=0: _result("65.17 1 1 1/1 1\n2\n"))
    with pytest.raises(S.StagingError, match="oversubscribed"):
        strict.require_healthy_host()


def test_a_healthy_host_passes_and_is_recorded():
    pod = _pod(_Provider(states=[PodState("podA", exists=False)]),
               run=lambda cmd, timeout=0: _result("1.12 1.41 1.24 2/957 9\n16\n"))
    health = pod.require_healthy_host()
    assert health == {"load1": 1.12, "cpus": 16, "load_per_cpu": 0.07,
                      "limit_per_cpu": 8.0}
    assert pod.ev["host_health"] == health


def test_the_gate_scales_with_the_cpu_count_rather_than_the_raw_load():
    """A load of 12 is fine on sixteen cores and pathological on two.

    The quantity that matters is contention per CPU, and a threshold on the raw
    load would refuse a healthy large host while admitting a thrashing small
    one — which is the only size this tool ever draws.
    """
    big = _pod(_Provider(states=[PodState("podA", exists=False)]),
               run=lambda cmd, timeout=0: _result("12.0 1 1 1/1 1\n16\n"))
    assert big.require_healthy_host()["load_per_cpu"] == 0.75

    small = _pod(_Provider(states=[PodState("podA", exists=False)]),
                 run=lambda cmd, timeout=0: _result("12.0 1 1 1/1 1\n1\n"))
    with pytest.raises(S.StagingError, match="oversubscribed"):
        small.require_healthy_host()


def test_an_unreadable_load_is_an_error_rather_than_a_pass():
    """A gate that cannot read its input must refuse, not assume health."""
    pod = _pod(_Provider(states=[PodState("podA", exists=False)]),
               run=lambda cmd, timeout=0: _result(""))
    with pytest.raises(S.StagingError, match="could not read"):
        pod.require_healthy_host()


# ---------------------------------------------------------------------------
# the transport floor
# ---------------------------------------------------------------------------

def test_a_rate_that_cannot_finish_the_job_is_refused(monkeypatch, tmp_path):
    """Whatever the cause. The load gate catches the pathology that was met;
    this catches the rest, from 32 MiB rather than from a per-probe timeout."""
    sample = tmp_path / "model.safetensors"
    sample.write_bytes(b"x" * 1024)

    #: The pod is built BEFORE the clock is replaced: its constructor reads the
    #: clock too, and a scripted sequence short enough to be consumed there
    #: leaves the measurement reading the same instant twice.
    pod = _pod(_Provider(states=[PodState("podA", exists=False)]),
               run=lambda cmd, timeout=0: _result(""))
    pod.ev["source"] = {"total_bytes": 23_850_825_745}
    slow = iter([0.0, 1000.0])                 # 32 MiB in 1000 s = 0.034 MB/s
    monkeypatch.setattr(S.time, "time", lambda: next(slow, 1000.0))
    monkeypatch.setattr(S.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b""))
    with pytest.raises(S.StagingError, match="below the"):
        pod.require_transport_floor(sample)


def test_an_adequate_rate_passes_and_is_recorded_as_diagnostic(monkeypatch,
                                                               tmp_path):
    sample = tmp_path / "model.safetensors"
    sample.write_bytes(b"x" * 1024)
    pod = _pod(_Provider(states=[PodState("podA", exists=False)]),
               run=lambda cmd, timeout=0: _result(""))
    pod.ev["source"] = {"total_bytes": 23_850_825_745}
    clock = iter([0.0, 60.0])                    # 32 MiB in 60 s = 0.56 MB/s
    monkeypatch.setattr(S.time, "time", lambda: next(clock, 60.0))
    monkeypatch.setattr(S.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b""))
    out = pod.require_transport_floor(sample)
    assert out["observed_mb_per_second"] > 0.15
    assert "never promoted" in out["_observed_is_diagnostic"]
    assert pod.ev["transport_floor"] == out


# ---------------------------------------------------------------------------
# teardown, against the real provider types
# ---------------------------------------------------------------------------

def test_a_confirmed_teardown_is_recorded_from_the_types_the_provider_returns():
    """`as_dict` is the provider's own serializer; a hand-written field is not.

    The first version read `a.detail`, which `TerminationAttempt` has never
    had, and raised inside the `finally` that was recording a termination that
    had already succeeded.
    """
    provider = _Provider(states=[
        PodState("podA", exists=True, desired_status="RUNNING"),
        PodState("podA", exists=False)])
    pod = _pod(provider)
    rec = pod.teardown("done")
    assert provider.terminated == ["podA"]
    assert rec["verified_gone"] is True
    assert rec["last_status"] == "ABSENT"
    assert rec["attempts"][0]["method"] == "runpodctl remove pod"
    assert rec["gpu_usd"] > 0


def test_an_unconfirmed_teardown_says_so_rather_than_claiming_success(
        monkeypatch):
    """A pod the control plane still reports as running is not gone.

    `PodState.billing` reads unknown as still billing, and this must not
    reinterpret it: the whole point of polling is that a zero return code from
    a remove call is not evidence of release.
    """
    pod = _pod(_Provider(states=[
        PodState("podA", exists=True, desired_status="RUNNING")]))
    #: Collapse the 300 s verification window without touching the logic: the
    #: clock jumps past the deadline on its second reading.
    clock = iter([0.0, 0.0, 10_000.0])
    monkeypatch.setattr(S.time, "time", lambda: next(clock, 10_000.0))
    monkeypatch.setattr(S.time, "sleep", lambda _s: None)
    rec = pod.teardown("done")
    assert rec["verified_gone"] is False
    assert rec["last_status"] == "RUNNING"


def test_a_provider_that_raises_still_produces_a_teardown_record():
    """Nothing in the recorder may raise: it runs from a `finally`, and an
    exception there destroys the record of a teardown that already happened."""
    pod = _pod(_Provider(states=[PodState("podA", exists=False)],
                         terminate_raises=True))
    rec = pod.teardown("failed")
    assert rec["attempts"][0]["ok"] is False
    assert "RuntimeError" in rec["attempts"][0]["error"]
    #: And the verification still ran, so the record is still useful.
    assert rec["verified_gone"] is True


def test_a_run_that_created_nothing_needs_no_teardown():
    pod = _pod(_Provider(states=[PodState("x", exists=False)]))
    pod.pod_id = ""
    rec = pod.teardown("never created")
    assert rec == {"pod_id": None, "terminated": True, "why": "never created",
                   "_means": "no resource was ever created"}


# ---------------------------------------------------------------------------
# what the tool reads, and what it never reads
# ---------------------------------------------------------------------------

def test_capacity_comes_from_the_provisioned_size_and_never_from_df():
    """`df` at the mount reports the BACKING CLUSTER, not the volume's quota.

    The rehearsal's 40 GB volume reported 165,732 GiB free, because the mount
    is a distributed filesystem shared across the datacenter. A free-space
    check built on that can never fail — and discovering the real limit by
    hitting it is how attempt5 died.
    """
    src = (REPO / "scripts/autoinit/stage_c2_probes_to_volume.py").read_text()
    body = src[src.index("def main("):]
    capacity = body[body.index("CAPACITY COMES FROM"):
                    body.index('say(f"volume mounted')]
    #: The capacity the check compares against is the PROVISIONED size.
    assert "capacity = a.volume_gb * 10**9" in capacity
    #: And `df` never supplies it. `du` reads what is already there, which is a
    #: different question and one the mount can actually answer.
    assert "df -" not in capacity, capacity
    #: The provisioned size is REQUIRED, never defaulted: a guess is not a bound.
    assert '--volume-gb", type=int, required=True' in " ".join(src.split())


def test_the_required_files_do_not_demand_a_score():
    """`campaign_state` admits a probe whose rows did not survive as UNSCORED
    rather than unusable, and that probe legitimately resumes at scoring.
    Requiring them here would refuse to stage a probe the protocol accepts."""
    assert set(S.REQUIRED) == {"model.safetensors", "config.json",
                               "probe_record.json", "durable_ack.json"}


def test_the_api_key_never_reaches_the_record(tmp_path):
    """Credentials are read as values and passed to the client. Nothing writes
    them, and the evidence record this tool produces must not carry one."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('apikey = "SECRET-DO-NOT-LOG"\napiurl = "x"\n')
    assert S.api_key(cfg) == "SECRET-DO-NOT-LOG"
    src = (REPO / "scripts/autoinit/stage_c2_probes_to_volume.py").read_text()
    for sink in ('ev["api', "ev['api", 'ev["key', "say(key", "say(f\"{key"):
        assert sink not in src, sink


def test_the_volume_mount_is_one_constant():
    """Two processes disagreeing about the mount path is a failure that only
    shows up once a pod is billing."""
    assert S.MOUNT == "/durable"
    import autoinit_c2_behavioural_launch as L
    assert L.VOLUME_MOUNT == S.MOUNT, (
        "the staging tool and the launcher name different mount paths; the "
        "probes would be written where no later pod looks")


# ---------------------------------------------------------------------------
# the in-flight throughput floor: the repair for DECAY
# ---------------------------------------------------------------------------

class _Proc:
    """A transfer that keeps running until told otherwise."""

    def __init__(self, *, finish_after: int = 10**9) -> None:
        self.polls = 0
        self.finish_after = finish_after
        self.returncode = 0
        self.killed = False

    def poll(self):
        self.polls += 1
        return 0 if self.polls > self.finish_after else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0

    def communicate(self, timeout=None):
        return ("", "")


def _watchable(monkeypatch, tmp_path, *, sizes, finish_after=10**9,
               verify=None):
    """A `stage_probe` whose transfer is a script and whose volume is a list."""
    proc = _Proc(finish_after=finish_after)
    monkeypatch.setattr(S.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(S.time, "sleep", lambda _s: None)

    #: THE CLOCK IS DRIVEN BY THE SIZE READINGS, not by a scripted list of
    #: instants. The watcher reads the clock more than once per window, so a
    #: list made the observed window depend on how many reads the
    #: implementation happens to make — and a test whose arithmetic moves when
    #: the code is refactored is measuring the wrong thing. Tying the two
    #: together makes the window between two consecutive readings exactly one
    #: interval, whatever else reads the clock.
    at = {"i": 0}

    def fake_remote(pod, directory):
        at["i"] += 1
        return sizes[min(at["i"] - 1, len(sizes) - 1)]

    monkeypatch.setattr(S.time, "time", lambda: 600.0 * at["i"])
    monkeypatch.setattr(S, "remote_bytes", fake_remote)
    calls = {"n": 0}

    def _verify(pod, remote_dir, want):
        calls["n"] += 1
        return verify(calls["n"]) if verify else ["not there yet"]
    monkeypatch.setattr(S, "verify_probe", _verify)

    src = tmp_path / "probe"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"x" * 16)
    pod = _pod(_Provider(states=[PodState("podA", exists=False)]),
               run=lambda cmd, timeout=0: _result(""),
               stall_check_seconds=600.0, stall_floor_mb_per_second=0.10,
               tail_margin_mib=64.0, probe_timeout_min=1e6)
    return pod, proc, src


def test_a_transfer_that_decays_is_abandoned_rather_than_ground_through(
        monkeypatch, tmp_path):
    """THE failure this exists for, in the numbers it was seen in.

    The first staging pod passed every startup check, moved 1.5 GB at
    0.3-0.5 MB/s, and then slid to 0.028 MB/s — ten days for this job. Nothing
    was wrong with the data or the code; the pod's host had stopped being able
    to move bytes. A per-probe timeout does eventually catch that, four hours
    later. A rolling floor catches it in ten minutes, and the partial file
    survives for the next draw.
    """
    #: 600 s windows: healthy, healthy, then 5 MB in ten minutes = 0.008 MB/s.
    sizes = [0, 300_000_000, 600_000_000, 605_000_000]
    pod, proc, src = _watchable(monkeypatch, tmp_path, sizes=sizes)
    with pytest.raises(S.StagingError, match="liveness floor"):
        S.stage_probe(pod, "p1", src, "/durable/x/p1",
                      {"bytes": 2_384_237_807, "files": {}})
    assert proc.killed, "the transfer was left running after the floor tripped"


def test_a_merely_slow_transfer_is_left_alone(monkeypatch, tmp_path):
    """A liveness check, not a performance target.

    0.15 MB/s finishes this job in about two days, which is slow and fine. The
    floor exists to distinguish that from a pod that has stopped working, and a
    gate that also refused the slow case would spend money redrawing for
    nothing. 0.15 is deliberately just above the 0.10 floor, so this also pins
    the boundary from the safe side.
    """
    step = int(0.15 * 1e6 * 600)                    # 0.15 MB/s over a window
    sizes = [0, step, 2 * step, 3 * step]
    pod, proc, src = _watchable(monkeypatch, tmp_path, sizes=sizes,
                                finish_after=3,
                                verify=lambda n: [] if n > 1 else ["absent"])
    rec = S.stage_probe(pod, "p1", src, "/durable/x/p1",
                        {"bytes": 2_384_237_807, "files": {}})
    assert rec["action"] == "staged" and rec["verified"] is True
    assert not proc.killed


def test_a_probe_already_verified_on_the_volume_is_not_sent_again(
        monkeypatch, tmp_path):
    """What makes a redraw cheap: only what has not yet moved moves."""
    pod, proc, src = _watchable(monkeypatch, tmp_path, sizes=[0],
                               verify=lambda n: [])
    rec = S.stage_probe(pod, "p1", src, "/durable/x/p1",
                        {"bytes": 2_384_237_807, "files": {}})
    assert rec == {"probe_id": "p1", "action": "skipped", "verified": True,
                   "bytes": 2_384_237_807, "seconds": 0.0}
    assert proc.polls == 0, "a verified probe was transferred again"


def test_an_unreadable_size_does_not_look_like_a_stall(monkeypatch, tmp_path):
    """A blip is not evidence that the transfer stopped.

    `remote_bytes` repeats its previous answer when it cannot read one, because
    reading an SSH failure as zero bytes looks exactly like a stall and would
    kill a healthy transfer.
    """
    pod = _pod(_Provider(states=[PodState("podA", exists=False)]),
               run=lambda cmd, timeout=0: _result(""))
    assert S.remote_bytes(pod, "/durable/x") == 0
    pod.run = lambda cmd, timeout=0: _result("12345\n")   # type: ignore[method-assign]
    assert S.remote_bytes(pod, "/durable/x") == 12345
    pod.run = lambda cmd, timeout=0: _result("")          # type: ignore[method-assign]
    assert S.remote_bytes(pod, "/durable/x") == 12345, (
        "an unreadable size read as zero; that is indistinguishable from a "
        "stall and would kill a working transfer")


def test_a_transfer_in_its_tail_is_alive_rather_than_stalled(monkeypatch,
                                                            tmp_path):
    """The false positive that cost a pod and a redraw.

    rsync keeps running after the last byte lands — it verifies the whole file
    and settles metadata — and the destination stops growing while it does.
    Probes 1 to 4 of the real staging run finished inside a sleep, so no check
    ever saw that phase. Probe 5's tail landed on a check boundary: 20 MiB of a
    2.22 GiB probe moved in the window, the floor read 0.046 MB/s, and a
    transfer that had delivered everything was killed, its pod torn down and
    redrawn.

    So the floor applies only while real work is outstanding. A transfer
    genuinely wedged in its tail is caught by `--probe-timeout-min` instead,
    which is right: the two failures need different responses.
    """
    total = 2_384_237_807
    #: Windows: healthy, healthy, then the tail — 20 MiB with ~0 outstanding.
    sizes = [0, 1_200_000_000, total - 21_000_000, total]
    pod, proc, src = _watchable(monkeypatch, tmp_path, sizes=sizes,
                                finish_after=4,
                                verify=lambda n: [] if n > 1 else ["absent"])
    rec = S.stage_probe(pod, "p1", src, "/durable/x/p1",
                        {"bytes": total, "files": {}})
    assert rec["action"] == "staged"
    assert not proc.killed, (
        "a transfer that had delivered every byte was killed for not growing")


def test_the_floor_still_bites_when_work_remains(monkeypatch, tmp_path):
    """The tail exemption must not become a blanket one.

    Half a probe outstanding at 0.008 MB/s is the pathology, not a tail, and it
    is what the floor exists for.
    """
    total = 2_384_237_807
    sizes = [0, 300_000_000, 600_000_000, 605_000_000]
    pod, proc, src = _watchable(monkeypatch, tmp_path, sizes=sizes)
    with pytest.raises(S.StagingError, match="liveness floor"):
        S.stage_probe(pod, "p1", src, "/durable/x/p1",
                      {"bytes": total, "files": {}})
    assert proc.killed
