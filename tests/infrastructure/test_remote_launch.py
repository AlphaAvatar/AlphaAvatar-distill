"""The launcher must not stay attached to the job it starts.

E6b's launcher blocked on its driver-start ssh for 434 minutes. It never reached
its polling loop, so it never tore the pod down at completion, and the watcher —
which tailed the orchestrator log — saw silence and read it as an idle session.
One blocked call disabled the completion teardown and the monitoring together.

These tests hold `start_detached` to a wall-clock bound against jobs that outlive
them, including the exact E6b shape where the start channel never closes at all.
"""

import json
import time
from pathlib import Path

import pytest

from aadistill.infrastructure.remote import (
    CommandResult, JobSpec, LocalShellTarget, RemoteLaunchError,
    bootstrap_script, probe, start_detached,
)

# The bound the launcher is entitled to assume. The jobs below run for minutes;
# if the implementation waits for them, these fail rather than hang.
RETURN_BOUND_S = 30.0


def make_spec(root: Path, command: str, job_id: str = "driver") -> JobSpec:
    return JobSpec(
        job_id=job_id,
        workdir=str(root / "work"),
        command=command,
        job_dir=str(root / "jobs"),
        log_path=str(root / "work" / f"{job_id}.log"),
        status_path=str(root / "work" / f"{job_id}.status"),
    )


@pytest.fixture
def target(tmp_path):
    (tmp_path / "work").mkdir()
    return LocalShellTarget(tmp_path)


def test_start_returns_while_the_job_is_still_running(tmp_path, target):
    """The regression test: a 600-second job must not delay the launcher."""
    spec = make_spec(tmp_path, "sleep 600")

    t0 = time.monotonic()
    job = start_detached(target, spec, start_timeout=20, verify_timeout=10)
    elapsed = time.monotonic() - t0

    assert elapsed < RETURN_BOUND_S, (
        f"start_detached took {elapsed:.1f}s for a job that runs for 600s; the "
        "launcher is attached to the driver, which is the E6b failure")
    assert job.pid > 0
    assert probe(target, job)[0] == "ALIVE", "the job should still be running"

    # And it is genuinely detached: killing it is the caller's job, not a
    # side effect of the launcher returning.
    import os
    import signal
    os.kill(job.pid, signal.SIGKILL)


def test_start_returns_even_when_the_start_channel_never_closes(tmp_path):
    """E6b exactly: the bootstrap runs, and the channel hangs open anyway.

    The launcher must still resume orchestration, and must still come away with
    the durable job identity — which it gets from the descriptor file, not from
    the call that hung.
    """
    real = LocalShellTarget(tmp_path)
    (tmp_path / "work").mkdir(exist_ok=True)
    spec = make_spec(tmp_path, "sleep 600")

    class HangingStartTarget:
        """Runs the bootstrap for real, then refuses to close the channel."""

        def __init__(self):
            self.start_calls = 0

        def run(self, command, *, timeout):
            if command.startswith("set -u"):          # the bootstrap
                self.start_calls += 1
                real.run(command, timeout=timeout)     # the work happens
                time.sleep(min(timeout, 2.0))          # and then it hangs
                return CommandResult(124, "", "", timed_out=True)
            return real.run(command, timeout=timeout)

    hanging = HangingStartTarget()
    t0 = time.monotonic()
    job = start_detached(hanging, spec, start_timeout=2, verify_timeout=10)
    elapsed = time.monotonic() - t0

    assert hanging.start_calls == 1
    assert elapsed < RETURN_BOUND_S
    assert job.start_channel_closed is False
    assert job.confirmed_by == "descriptor_probe", (
        "when the start channel hangs, confirmation must come from the "
        "out-of-band descriptor probe")
    assert job.pid > 0

    import os
    import signal
    os.kill(job.pid, signal.SIGKILL)


def test_descriptor_is_durable_on_the_pod(tmp_path, target):
    """The job identity survives on disk, so a later connection can recover it."""
    spec = make_spec(tmp_path, "sleep 60")
    job = start_detached(target, spec, start_timeout=20, verify_timeout=10)

    descriptor = json.loads(Path(job.descriptor_path).read_text())
    assert descriptor["job_id"] == "driver"
    assert int(descriptor["pid"]) == job.pid
    assert Path(job.pid_path).read_text().strip() == str(job.pid)
    assert descriptor["log_path"] == job.log_path

    import os
    import signal
    os.kill(job.pid, signal.SIGKILL)


def test_child_records_its_own_pid_not_the_wrappers(tmp_path, target):
    """`$!` is the wrapper's pid under some setsid invocations; `$$` never is.

    A pid that belongs to a wrapper which has already exited makes every later
    liveness probe report a dead driver, and the poller tears down a healthy
    run.
    """
    spec = make_spec(tmp_path, "sleep 60")
    job = start_detached(target, spec, start_timeout=20, verify_timeout=10)

    import os
    import signal
    os.kill(job.pid, 0)          # the recorded pid is a live process
    cmdline = Path(f"/proc/{job.pid}/cmdline").read_bytes().decode(errors="replace")
    assert "sleep" in cmdline, (
        f"pid {job.pid} is alive but is not the job (cmdline {cmdline!r})")
    os.kill(job.pid, signal.SIGKILL)


def test_output_goes_to_the_log_not_the_channel(tmp_path, target):
    spec = make_spec(tmp_path, "echo hello-from-the-driver; sleep 30")
    job = start_detached(target, spec, start_timeout=20, verify_timeout=10)
    deadline = time.monotonic() + 10
    text = ""
    while time.monotonic() < deadline:
        text = Path(job.log_path).read_text()
        if "hello-from-the-driver" in text:
            break
        time.sleep(0.2)
    assert "hello-from-the-driver" in text

    import os
    import signal
    os.kill(job.pid, signal.SIGKILL)


def test_a_short_job_that_finishes_during_verification_is_not_an_error(tmp_path, target):
    spec = make_spec(tmp_path, "true")
    job = start_detached(target, spec, start_timeout=20, verify_timeout=10,
                         verify_attempts=3, verify_delay=0.2)
    assert job.confirmed_by in {"descriptor_probe", "descriptor_probe_exited"}
    assert Path(job.exit_path).read_text().strip() == "0"


def test_a_job_that_never_starts_raises_rather_than_polling_forever(tmp_path):
    """A driver that dies at import must not become hours of polling."""
    spec = make_spec(tmp_path, "sleep 600")

    class DeadTarget:
        def run(self, command, *, timeout):
            return CommandResult(255, "", "ssh: connect to host port 22: "
                                          "Connection refused")

    with pytest.raises(RemoteLaunchError, match="could not be confirmed"):
        start_detached(DeadTarget(), spec, start_timeout=1, verify_timeout=1,
                       verify_attempts=2, verify_delay=0)


def test_bootstrap_quotes_every_path(tmp_path):
    """Paths with spaces are not the risk; unquoted expansion is."""
    spec = make_spec(tmp_path, "sleep 1")
    script = bootstrap_script(spec)
    assert "$(ls" not in script and "`" not in script, (
        "the bootstrap must not use command substitution over globs; that is "
        "the construct that silently shortened E3/E4/E5 artifact bundles")
    assert "< /dev/null" in script
    assert "setsid" in script
    assert "echo $$ >" in script


def test_env_is_forwarded_into_the_detached_job(tmp_path, target):
    spec = JobSpec(
        job_id="envjob", workdir=str(tmp_path / "work"),
        command="printf '%s' \"$TEACHER_REVISION\" > "
                f"{tmp_path}/work/teacher.txt",
        job_dir=str(tmp_path / "jobs"),
        log_path=str(tmp_path / "work" / "envjob.log"),
        status_path=str(tmp_path / "work" / "envjob.status"),
        env={"TEACHER_REVISION": "768f209d"},
    )
    start_detached(target, spec, start_timeout=20, verify_timeout=10,
                   verify_attempts=5, verify_delay=0.3)
    deadline = time.monotonic() + 10
    written = ""
    while time.monotonic() < deadline:
        p = tmp_path / "work" / "teacher.txt"
        if p.is_file():
            written = p.read_text()
            if written:
                break
        time.sleep(0.2)
    assert written == "768f209d", (
        "a variable the driver needs must survive detachment; E6b's setup died "
        "on a KeyError for exactly this class of omission")
