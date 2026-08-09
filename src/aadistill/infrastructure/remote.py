"""Start a long remote job without staying attached to it.

E6b's launcher started its driver with the intended incantation —
``setsid nohup … > log 2>&1 < /dev/null & disown`` — and the ssh call blocked for
the whole 434-minute run anyway. The invocation was byte-identical to E6's, which
had returned in 74 seconds. So the lesson is not "use setsid": E6b already did.
The lesson is that **whether the channel closes is not under the launcher's
control**, and a launcher whose orchestration depends on ssh returning has a
single point of failure it cannot inspect.

This module removes the dependency:

1. The remote bootstrap detaches the job *and writes a durable descriptor*
   (`job.json`, pidfile) on the pod before it exits. The descriptor is the
   contract, not the ssh exit status.
2. The start call is run under a hard local timeout. If the channel hangs, the
   timeout fires and the launcher keeps going.
3. Either way, a **separate short ssh** reads the descriptor back. That call —
   not the start call — is what confirms the job is running.

So the launcher resumes orchestration within `start_timeout + verify_timeout`
whether the start channel closes in 74 seconds or never closes at all. That
bound is the property `tests/infrastructure/test_remote_launch.py` asserts
against a driver that outlives the test by minutes.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol

DESCRIPTOR_PREFIX = "AADISTILL_JOB "


class RemoteLaunchError(RuntimeError):
    """The job could not be confirmed running on the pod."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class ShellTarget(Protocol):
    """Somewhere a shell command can be run. An SSH pod, or a local directory."""

    def run(self, command: str, *, timeout: float) -> CommandResult: ...


class SSHTarget:
    """A pod reached over ssh.

    `-n` matters: without it ssh inherits the launcher's stdin, and a launcher
    running under `nohup … &` can hand the remote shell a stdin that never
    reaches EOF. That is one of the few mechanisms by which a correctly
    backgrounded remote command still holds the channel open.
    """

    def __init__(self, host: str, port: int | str, *, user: str = "root",
                 options: tuple[str, ...] = (
                     "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null",
                     "-o", "ConnectTimeout=30",
                     "-o", "ServerAliveInterval=30")) -> None:
        self.host = host
        self.port = str(port)
        self.user = user
        self.options = options

    def argv(self, command: str) -> list[str]:
        return ["ssh", "-n", "-p", self.port, *self.options,
                f"{self.user}@{self.host}", command]

    def run(self, command: str, *, timeout: float) -> CommandResult:
        try:
            proc = subprocess.run(self.argv(command), capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                returncode=124,
                stdout=(exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode(errors="replace")
                if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                timed_out=True)
        except OSError as exc:
            return CommandResult(returncode=125, stdout="",
                                 stderr=f"{type(exc).__name__}: {exc}")
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)


class LocalShellTarget:
    """A directory on this machine, driven through the same shell contract.

    This is the pod simulator's shell. It runs the *same* bootstrap text the
    real path sends over ssh, so the simulation exercises the actual script
    rather than a paraphrase of it — the gap that let E6's pod die on a test
    gate its inputs could not satisfy.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, command: str, *, timeout: float) -> CommandResult:
        try:
            proc = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True,
                timeout=timeout, cwd=self.root, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            return CommandResult(
                124,
                out.decode(errors="replace") if isinstance(out, bytes) else out,
                err.decode(errors="replace") if isinstance(err, bytes) else err,
                timed_out=True)
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)


@dataclass(frozen=True)
class JobSpec:
    """What to run, and where its durable paths live on the pod."""

    job_id: str
    workdir: str
    command: str
    job_dir: str
    log_path: str
    status_path: str
    env: dict[str, str] | None = None

    @property
    def pid_path(self) -> str:
        return f"{self.job_dir}/{self.job_id}.pid"

    @property
    def descriptor_path(self) -> str:
        return f"{self.job_dir}/{self.job_id}.job.json"

    @property
    def exit_path(self) -> str:
        return f"{self.job_dir}/{self.job_id}.exit"


@dataclass(frozen=True)
class RemoteJob:
    """A confirmed-running remote job, and every path needed to follow it."""

    job_id: str
    pid: int
    workdir: str
    command: str
    log_path: str
    status_path: str
    pid_path: str
    descriptor_path: str
    exit_path: str
    started_utc: str
    start_channel_closed: bool
    confirmed_by: str

    def as_dict(self) -> dict:
        return asdict(self)


def bootstrap_script(spec: JobSpec) -> str:
    """The remote shell text that detaches the job and records it.

    Three details carry weight.

    * The child writes **its own** pid (`echo $$` inside the detached shell)
      rather than the parent reading `$!`. `setsid` forks when it is already a
      process-group leader and does not otherwise, so `$!` is the job's pid in
      some invocations and its wrapper's in others. `$$` is always the job's.
    * The wrapper records the exit status to `exit_path` when the job finishes,
      so "gone with no terminal marker" is distinguishable from "still running"
      without asking the process table.
    * Every descriptor write happens **before** the bootstrap prints, so a
      caller that never sees the printed line can still read the file.
    """
    env_prefix = ""
    if spec.env:
        # `export`, not a `VAR=x cmd` prefix. A prefix assignment is applied
        # after the shell has already expanded `$VAR` on that same line, so
        # `TEACHER_REVISION=abc python -c "...$TEACHER_REVISION..."` forwards an
        # empty string — the same class of silent-empty-variable failure that
        # killed the E6b setup at INIT_READY.
        env_prefix = "".join(
            f"export {k}={shlex.quote(v)}; " for k, v in sorted(spec.env.items()))
    inner = (
        f"echo $$ > {shlex.quote(spec.pid_path)}; "
        f"{env_prefix}{spec.command}; "
        f"echo $? > {shlex.quote(spec.exit_path)}"
    )
    return f"""set -u
mkdir -p {shlex.quote(spec.job_dir)} {shlex.quote(spec.workdir)}
cd {shlex.quote(spec.workdir)}
rm -f {shlex.quote(spec.pid_path)} {shlex.quote(spec.exit_path)}
touch {shlex.quote(spec.log_path)}
setsid bash -c {shlex.quote(inner)} >> {shlex.quote(spec.log_path)} 2>&1 < /dev/null &
for _ in $(seq 1 50); do
  [ -s {shlex.quote(spec.pid_path)} ] && break
  sleep 0.1
done
JOB_PID=$(cat {shlex.quote(spec.pid_path)} 2>/dev/null || echo "")
printf '{{"job_id":"%s","pid":"%s","workdir":"%s","log_path":"%s","status_path":"%s","started_utc":"%s"}}\\n' \\
  {shlex.quote(spec.job_id)} "$JOB_PID" {shlex.quote(spec.workdir)} \\
  {shlex.quote(spec.log_path)} {shlex.quote(spec.status_path)} \\
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > {shlex.quote(spec.descriptor_path)}
echo "{DESCRIPTOR_PREFIX}$(cat {shlex.quote(spec.descriptor_path)})"
exit 0
"""


def read_descriptor_script(spec: JobSpec) -> str:
    """A short, cheap command that reports the job's durable state.

    Deliberately separate from the bootstrap. This is the call that decides
    whether the launcher may proceed, and it must be able to succeed on a
    connection made *after* the start channel misbehaved.
    """
    return (
        f"if [ -s {shlex.quote(spec.descriptor_path)} ]; then "
        f"echo \"{DESCRIPTOR_PREFIX}$(cat {shlex.quote(spec.descriptor_path)})\"; "
        f"fi; "
        f"PID=$(cat {shlex.quote(spec.pid_path)} 2>/dev/null || echo ''); "
        f"if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then echo ALIVE; "
        f"elif [ -s {shlex.quote(spec.exit_path)} ]; then "
        f"echo \"EXITED:$(cat {shlex.quote(spec.exit_path)})\"; "
        f"else echo UNKNOWN; fi"
    )


def _parse_descriptor(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith(DESCRIPTOR_PREFIX):
            try:
                return json.loads(line[len(DESCRIPTOR_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def start_detached(
    target: ShellTarget,
    spec: JobSpec,
    *,
    start_timeout: float = 120.0,
    verify_timeout: float = 60.0,
    verify_attempts: int = 5,
    verify_delay: float = 2.0,
    sleep=time.sleep,
) -> RemoteJob:
    """Start `spec` detached and return only once it is confirmed running.

    The return is bounded by `start_timeout + verify_attempts * (verify_timeout
    + verify_delay)` regardless of how long the job itself runs. A start channel
    that never closes costs `start_timeout` and nothing else.

    Raises `RemoteLaunchError` when the descriptor cannot be read back or the
    process is not alive — the launcher must not proceed to poll for a job that
    was never started, which is the failure mode where a pod bills for hours
    against a driver that died at import time.
    """
    started = target.run(bootstrap_script(spec), timeout=start_timeout)
    channel_closed = not started.timed_out

    descriptor = _parse_descriptor(started.stdout) if channel_closed else None
    confirmed_by = "start_channel" if descriptor else ""

    last = ""
    for attempt in range(1, verify_attempts + 1):
        probe = target.run(read_descriptor_script(spec), timeout=verify_timeout)
        last = (probe.stdout + probe.stderr).strip()
        found = _parse_descriptor(probe.stdout)
        if found:
            descriptor = found
        if "ALIVE" in probe.stdout:
            confirmed_by = "descriptor_probe"
            break
        if "EXITED:" in probe.stdout and descriptor:
            # The job ran and finished inside the verification window. That is a
            # legitimate outcome for a short job and must not be an error.
            confirmed_by = "descriptor_probe_exited"
            break
        if attempt < verify_attempts:
            sleep(verify_delay)

    if not descriptor or not confirmed_by:
        raise RemoteLaunchError(
            f"job {spec.job_id} could not be confirmed on the target. "
            f"start channel: rc={started.returncode} timed_out={started.timed_out} "
            f"stderr={started.stderr.strip()[:400]!r}; "
            f"last probe: {last[:400]!r}. Refusing to poll for a job that may "
            "never have started.")

    pid_raw = str(descriptor.get("pid", "")).strip()
    try:
        pid = int(pid_raw)
    except ValueError as exc:
        raise RemoteLaunchError(
            f"job {spec.job_id} descriptor carries no usable pid "
            f"({pid_raw!r}); the durable job identity is the whole point of "
            "detaching") from exc

    return RemoteJob(
        job_id=spec.job_id, pid=pid, workdir=spec.workdir, command=spec.command,
        log_path=spec.log_path, status_path=spec.status_path,
        pid_path=spec.pid_path, descriptor_path=spec.descriptor_path,
        exit_path=spec.exit_path,
        started_utc=str(descriptor.get("started_utc", "")),
        start_channel_closed=channel_closed, confirmed_by=confirmed_by)


def probe(target: ShellTarget, job: RemoteJob, *,
          timeout: float = 60.0) -> tuple[str, str]:
    """(liveness, raw) for a running job, where liveness is ALIVE/EXITED:n/UNKNOWN.

    `UNKNOWN` deliberately does not mean dead. It means the pidfile and the exit
    file disagree with each other or are both absent, which on a pod that is
    still billing is a reason to look, not a reason to stop looking.
    """
    spec = JobSpec(job_id=job.job_id, workdir=job.workdir, command=job.command,
                   job_dir=str(Path(job.pid_path).parent),
                   log_path=job.log_path, status_path=job.status_path)
    result = target.run(read_descriptor_script(spec), timeout=timeout)
    out = result.stdout
    if "ALIVE" in out:
        return "ALIVE", out
    for line in out.splitlines():
        if line.startswith("EXITED:"):
            return line.strip(), out
    return "UNKNOWN", out
