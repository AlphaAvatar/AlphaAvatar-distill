"""The E6b session, replayed against the hardened stack.

On 2026-08-08 four things went wrong together and each hid the next:

1. the launcher's driver-start ssh did not detach, and blocked for 434 minutes;
2. so the launcher never reached its polling loop and never tore the pod down;
3. so the watcher, which tailed the orchestrator **log**, saw silence and read
   seven hours of a billing pod as seven hours of nothing happening;
4. and RunPod's `--terminate-after`, the documented last-resort layer since E4,
   did not fire — deadline 00:28:47, pod `RUNNING` at 00:34.

Cost: $7.68 against a $7.12 authorization. Collateral: the machine-readable
training event streams for both arms, which existed only on the pod and were
absent from the one bundling command that ran before teardown.

These tests drive the whole sequence through the real modules — a local shell
standing in for the pod, an in-memory control plane standing in for RunPod — and
assert that each layer now catches what it missed. No GPU is involved: this is
control flow, and control flow does not need to be paid for.
"""

import json
import os
import signal
import time
from dataclasses import dataclass

import pytest

from aadistill.infrastructure.artifact_gate import (
    ArtifactSpec, build_manifest, create_archive, evaluate_teardown,
    verify_archive, verify_extracted,
)
from aadistill.infrastructure.budget import StepTime, plan_session
from aadistill.infrastructure.log_relay import LogRelay, RelaySpec
from aadistill.infrastructure.provider import SimulatedProvider
from aadistill.infrastructure.remote import (
    CommandResult, JobSpec, LocalShellTarget, RemoteLaunchError, probe,
    start_detached,
)
from aadistill.infrastructure.watchdog import (
    Journal, MarkerSnapshot, SessionWatcher, Watchdog, WatchdogPolicy,
)

E6B_PRICE = 0.99
E6B_STEPS = 2916


class HangingStartTarget:
    """The E6b pod: the bootstrap runs, and the start channel never closes."""

    def __init__(self, inner: LocalShellTarget):
        self.inner = inner
        self.hung_calls = 0

    def run(self, command, *, timeout):
        if command.startswith("set -u"):
            self.hung_calls += 1
            self.inner.run(command, timeout=timeout)
            time.sleep(min(timeout, 1.0))
            return CommandResult(124, "", "", timed_out=True)
        return self.inner.run(command, timeout=timeout)


class BlockedSSHTarget:
    """Every connection hangs — the pod is unreachable, not absent."""

    def run(self, command, *, timeout):
        return CommandResult(124, "", "", timed_out=True)


@dataclass
class Session:
    tmp: object
    pod_root: object
    provider: SimulatedProvider
    target: LocalShellTarget
    started_epoch: float


@pytest.fixture
def session(tmp_path):
    pod_root = tmp_path / "pod"
    (pod_root / "workspace").mkdir(parents=True)
    return Session(tmp_path, pod_root, SimulatedProvider("pod-e6b"),
                   LocalShellTarget(pod_root), time.time())


def driver_spec(session, command="sleep 600", job_id="e6b_driver"):
    return JobSpec(
        job_id=job_id,
        workdir=str(session.pod_root / "workspace"),
        command=command,
        job_dir=str(session.pod_root / "workspace" / "jobs"),
        log_path=str(session.pod_root / "workspace" / "e6b_run.log"),
        status_path=str(session.pod_root / "workspace" / "e6b.status"),
    )


def kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# --------------------------------------------------------------------------
# 1. the launcher
# --------------------------------------------------------------------------

def test_successful_detached_launch(session):
    spec = driver_spec(session)
    t0 = time.monotonic()
    job = start_detached(session.target, spec, start_timeout=20,
                         verify_timeout=10)
    assert time.monotonic() - t0 < 30
    assert job.start_channel_closed and job.pid > 0
    assert probe(session.target, job)[0] == "ALIVE"
    kill(job.pid)


def test_the_e6b_blocked_launcher_no_longer_blocks(session):
    """The originating failure. The driver runs; the launcher comes back."""
    hanging = HangingStartTarget(session.target)
    spec = driver_spec(session)

    t0 = time.monotonic()
    job = start_detached(hanging, spec, start_timeout=2, verify_timeout=5)
    elapsed = time.monotonic() - t0

    assert hanging.hung_calls == 1
    assert elapsed < 30, (
        f"the launcher took {elapsed:.1f}s against a 600s driver; E6b took "
        "434 minutes here")
    assert job.start_channel_closed is False
    assert job.confirmed_by == "descriptor_probe"
    assert probe(session.target, job)[0] == "ALIVE"
    kill(job.pid)


def test_a_remote_driver_that_dies_at_startup_is_not_polled_for(session):
    """A driver that fails at import must not become hours of polling."""
    spec = driver_spec(session, command="python3 -c 'import nonexistent_module'")
    job = start_detached(session.target, spec, start_timeout=20,
                         verify_timeout=10, verify_attempts=8, verify_delay=0.3)
    deadline = time.monotonic() + 10
    liveness = "ALIVE"
    while time.monotonic() < deadline:
        liveness, _ = probe(session.target, job)
        if liveness != "ALIVE":
            break
        time.sleep(0.2)
    assert liveness.startswith("EXITED:") and liveness != "EXITED:0", (
        "the poller must be able to tell a crashed driver from a running one "
        "without a terminal marker ever being written")


def test_blocked_ssh_is_a_launch_error_not_a_silent_poll(session):
    with pytest.raises(RemoteLaunchError, match="could not be confirmed"):
        start_detached(BlockedSSHTarget(), driver_spec(session),
                       start_timeout=0.5, verify_timeout=0.5,
                       verify_attempts=2, verify_delay=0)


# --------------------------------------------------------------------------
# 2. the watcher, and the silence
# --------------------------------------------------------------------------

def test_a_silent_orchestrator_over_a_billing_pod_is_not_idle(session):
    """Point 3 of the failure chain, as an assertion.

    Nothing has written a marker in seven hours because the launcher is blocked.
    The verdict must still be that money is being spent.
    """
    watcher = SessionWatcher()
    verdict = watcher.assess(
        session.provider.get("pod-e6b"),
        MarkerSnapshot(last_marker=None, age_seconds=None,
                       orchestrator_log_age_seconds=7 * 3600))
    assert verdict.billing is True
    assert verdict.state == "LIVE_AND_BILLING_NO_MARKERS"
    assert "out of band" in verdict.recommended_action


def test_the_watchdog_terminates_a_pod_whose_orchestrator_is_silent(session):
    """The independent layer, with everything else inert.

    No SSH, no markers, no launcher. The pod is 480 minutes old against a 431
    minute limit — E6b's actual overrun shape — and the watchdog is the only
    thing still working.
    """
    clock = [session.started_epoch + 480 * 60]
    policy = WatchdogPolicy(
        pod_id="pod-e6b", session_start_epoch=session.started_epoch,
        price_per_hour=E6B_PRICE, hard_terminate_minutes=431.0,
        authorized_usd=7.12, verify_delay_seconds=0.0)
    journal = Journal(session.tmp / "watchdog.jsonl")
    dog = Watchdog(session.provider, policy, journal,
                   clock=lambda: clock[0], sleep=lambda s: None)

    assert dog.run(max_ticks=3) == "pod_gone"
    assert session.provider.exists is False

    events = [r["event"] for r in journal.records()]
    assert "hard_limit_reached" in events
    assert "terminate_attempt" in events
    assert "terminate_verify" in events
    assert "terminated" in events
    limit = next(r for r in journal.records()
                 if r["event"] == "hard_limit_reached")
    assert limit["elapsed_minutes"] == pytest.approx(480.0, abs=0.1)
    assert limit["accrued_usd"] == pytest.approx(7.92, abs=0.01)


def test_the_watchdog_retries_a_termination_the_provider_ignores(session):
    """`--terminate-after` accepted a deadline and the pod ran on for 6 minutes."""
    provider = SimulatedProvider("pod-e6b", ignored_terminations=3)
    policy = WatchdogPolicy(
        pod_id="pod-e6b", session_start_epoch=session.started_epoch - 600 * 60,
        price_per_hour=E6B_PRICE, hard_terminate_minutes=431.0,
        authorized_usd=7.12, verify_polls=1, verify_delay_seconds=0.0)
    journal = Journal(session.tmp / "watchdog.jsonl")
    dog = Watchdog(provider, policy, journal, sleep=lambda s: None)

    assert dog.tick().action == "verified_gone"
    attempts = [r for r in journal.records() if r["event"] == "terminate_attempt"]
    assert len(attempts) == 4
    assert all(a["any_ok"] for a in attempts[:3]), (
        "each ignored call reported success; only the verification poll knew")


def test_a_provider_that_cannot_terminate_escalates_loudly(session):
    provider = SimulatedProvider("pod-e6b", terminate_failures=99)
    policy = WatchdogPolicy(
        pod_id="pod-e6b", session_start_epoch=session.started_epoch - 600 * 60,
        price_per_hour=E6B_PRICE, hard_terminate_minutes=431.0,
        authorized_usd=7.12, terminate_rounds=3, verify_polls=1,
        verify_delay_seconds=0.0, escalate_after_minutes=0.0)
    journal = Journal(session.tmp / "watchdog.jsonl")
    dog = Watchdog(provider, policy, journal, sleep=lambda s: None)

    assert dog.run(max_ticks=2) == "termination_failed"
    records = journal.records()
    assert any(r["event"] == "TERMINATION_FAILED" for r in records)
    assert any("human" in str(r.get("note", "")) for r in records)


# --------------------------------------------------------------------------
# 3. the artifacts
# --------------------------------------------------------------------------

def write_arm(session, arm, *, steps=30, event_stream=True):
    d = session.pod_root / "workspace" / "aad" / "artifacts" / "stage3" / arm
    d.mkdir(parents=True, exist_ok=True)
    if event_stream:
        with open(d / "train_log.jsonl", "w") as f:
            for i in range(steps):
                f.write(json.dumps({"event": "train_step", "step": i,
                                    "loss": 3.0 / (i + 1)}) + "\n")
    (d / "run_manifest.json").write_text(json.dumps({"config_sha256": arm}))
    return d


def artifacts_root(session):
    return session.pod_root / "workspace" / "aad" / "artifacts"


SPECS = (
    ArtifactSpec("event_stream", "stage3/e6b_*/train_log.jsonl"),
    ArtifactSpec("run_manifest", "stage3/e6b_*/run_manifest.json"),
)


def test_missing_structured_logs_block_teardown(session):
    """E6b's actual artifact loss, caught while the pod is still alive."""
    write_arm(session, "e6b_p2_r2960k_sa", event_stream=False)
    write_arm(session, "e6b_p2_r2960k_sb", event_stream=False)

    manifest = build_manifest(artifacts_root(session), SPECS)
    assert not manifest.ok
    assert manifest.missing[0]["artifact_class"] == "event_stream"

    decision = evaluate_teardown({
        "training_complete": True, "evaluation_complete": True,
        "artifact_manifest_created": True, "required_files_present": False})
    assert not decision.allowed
    assert decision.failed_check == "required_files_present"


def test_the_relay_keeps_the_event_stream_when_the_pod_is_deleted(session):
    """Defence in depth: even a teardown that loses the bundle keeps the events."""
    arm = write_arm(session, "e6b_p2_r2960k_sa", steps=291)
    spec = RelaySpec(remote_path=str(arm / "train_log.jsonl"),
                     local_name="e6b_sa.train_log.jsonl")
    relay = LogRelay(session.target, (spec,), session.tmp / "durable")
    assert relay.sync_once().ok

    # The pod goes away — with the bundle still on it, as in E6b.
    import shutil
    shutil.rmtree(session.pod_root)
    session.provider.terminate("pod-e6b")

    events = relay.recovered_events(spec)
    assert len(events) == 291
    assert events[-1]["step"] == 290


def test_partial_transfer_and_hash_mismatch_are_both_caught(session):
    write_arm(session, "e6b_p2_r2960k_sa")
    write_arm(session, "e6b_p2_r2960k_sb")
    root = artifacts_root(session)
    manifest = build_manifest(root, SPECS)
    assert manifest.ok

    archive = create_archive(manifest, session.tmp / "e6b.tar.gz")
    assert not verify_archive(archive, manifest)

    # A transfer that dropped one file.
    store = session.tmp / "store"
    for entry in manifest.entries[:-1]:
        p = store / entry.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((root / entry.path).read_bytes())
    problems = verify_extracted(store, manifest)
    assert problems and "not retrieved" in problems[0]

    # And one that corrupted a file.
    last = manifest.entries[-1]
    (store / last.path).parent.mkdir(parents=True, exist_ok=True)
    (store / last.path).write_text("truncated")
    problems = verify_extracted(store, manifest)
    assert any("hash mismatch" in p for p in problems)


def test_safe_teardown_runs_the_whole_sequence(session):
    """The good path, end to end, with the pod deleted only at the end."""
    write_arm(session, "e6b_p2_r2960k_sa")
    write_arm(session, "e6b_p2_r2960k_sb")
    root = artifacts_root(session)

    state = {"training_complete": True, "evaluation_complete": True}

    manifest = build_manifest(root, SPECS)
    state["artifact_manifest_created"] = True
    state["required_files_present"] = manifest.ok
    # The arms have finished; nothing is being appended, so the streams are
    # final rather than snapshots.
    state["final_streams_quiescent"] = manifest.final_streams_quiescent

    archive = create_archive(manifest, session.tmp / "e6b.tar.gz")
    state["archive_created"] = archive.is_file()
    state["archive_contents_verified"] = not verify_archive(archive, manifest)

    store = session.tmp / "store"
    import tarfile
    with tarfile.open(archive) as tar:
        tar.extractall(store, filter="data")
    state["transfer_complete"] = True
    state["local_hashes_verified"] = not verify_extracted(store, manifest)
    state["checkpoint_hashes_matched"] = True
    # E6b owed no off-pod products beyond its archive, so the session answers
    # the products-secured question with "none owed" rather than skipping it —
    # an unreported check counts as False and would block this teardown.
    state["required_products_secured"] = True
    state["report_inputs_verified"] = True

    decision = evaluate_teardown(state)
    assert decision.allowed and not decision.emergency

    session.provider.terminate("pod-e6b")
    assert session.provider.exists is False


def test_emergency_budget_teardown_deletes_and_records_the_loss(session):
    """The cost watchdog outranks the artifact gate, and says so in the record."""
    write_arm(session, "e6b_p2_r2960k_sa", event_stream=False)
    manifest = build_manifest(artifacts_root(session), SPECS)
    assert not manifest.ok

    decision = evaluate_teardown(
        {"training_complete": True, "evaluation_complete": True,
         "artifact_manifest_created": True, "required_files_present": False,
         "final_streams_quiescent": False},
        emergency_budget=True,
        emergency_reason="hard limit 431 min reached; $7.12 authorization",
        incomplete_event_streams=("stage3/e6b_p2_r2960k_sa/train_log.jsonl",))
    assert decision.allowed and decision.emergency
    assert decision.failed_check == "required_files_present"
    assert "LOST" in decision.reason
    assert "archive_created" in decision.reason, (
        "the record must name every check that did not run")
    assert "THE FINAL EVENT STREAM IS INCOMPLETE" in decision.reason
    assert decision.incomplete_event_streams == (
        "stage3/e6b_p2_r2960k_sa/train_log.jsonl",)

    session.provider.terminate("pod-e6b")
    assert session.provider.exists is False


# --------------------------------------------------------------------------
# 4. the arithmetic that would have prevented all of it
# --------------------------------------------------------------------------

def test_the_e6b_plan_is_rejected_at_its_real_step_time():
    """Priced at 4.15 s/step, E6b never fits $7.12 — and it says so up front."""
    from aadistill.infrastructure.budget import BudgetError

    with pytest.raises(BudgetError) as exc:
        plan_session(
            price_per_hour=E6B_PRICE, authorized_usd=7.12, arms=2,
            steps_per_arm=E6B_STEPS,
            step_time=StepTime(4.15, "E6b arms, L40S, 2026-08-08"),
            setup_minutes=25.0, eval_minutes_per_arm=10.0, transfer_minutes=20.0,
            artifact_recovery_reserve_minutes=30.0)
    assert "shortfall" in str(exc.value)


def test_the_soft_stop_leaves_the_reserve_for_collection():
    plan = plan_session(
        price_per_hour=E6B_PRICE, authorized_usd=9.50, arms=2,
        steps_per_arm=E6B_STEPS,
        step_time=StepTime(4.15, "E6b arms, L40S, 2026-08-08"),
        setup_minutes=25.0, eval_minutes_per_arm=10.0, transfer_minutes=20.0,
        artifact_recovery_reserve_minutes=30.0)

    # A second arm may not start if it would eat the reserve, even though it
    # would finish before the hard limit. E6b's driver re-priced against the
    # authorization and started an arm that left nothing for teardown.
    arm_minutes = E6B_STEPS * 4.15 / 60
    assert plan.may_start(plan.expected_minutes - arm_minutes, arm_minutes)
    assert not plan.may_start(plan.soft_stop_minutes - 5, arm_minutes)
    assert plan.phase_at(plan.soft_stop_minutes + 1) == "artifact_recovery"
