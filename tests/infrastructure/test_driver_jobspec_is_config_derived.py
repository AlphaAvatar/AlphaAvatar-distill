"""The driver's `JobSpec` is built from the session's declared image layout.

`SessionRunner` used to hold `WS = "/workspace"` and `REPO = "/workspace/aad"`
as module constants. They were removed so a second image could be supported by
declaring `ExecutionCommands` rather than patching this module's globals, and
the note at the top of `session_runner.py` says every consumer reads `self.ws`
and `self.repo` now.

**One did not.** `run()` built the driver job with `workdir=REPO`, a name that no
longer exists, three lines above an `env` that correctly used `self.repo`. It
raises `NameError` — *after* the provider resource is created, *after* setup
completes and *after* `materialize_inputs` returns, which is the most expensive
moment in the session at which to discover a typo.

Why nothing caught it: the removal was verified against the nineteen f-strings
that build remote commands, and this is a keyword argument. And the tests that
drive the real acquisition loop
(`tests/pod/test_c1_one_provider_resource.py`) all stub `setup_on_draw` to a
*failure* outcome, so `run()` returns before reaching this line. The success
path through `run()` had no execution coverage at all.

So this drives the REAL `run()` to the submission boundary, with only the
provider, SSH and remote-process boundaries faked, and asserts the job's paths
against **two different declared layouts** — a single layout cannot distinguish
"read from the config" from "happens to match the default".
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure import session_runner as SR
from aadistill.infrastructure.provider import PodState
from aadistill.infrastructure.session import ExecutionCommands

#: Two deployments, sharing no path component. Anything the runner emits that
#: is not derived from the layout it was handed will match one of these and not
#: the other.
LAYOUT_A = dict(workspace_root="/workspace", checkout_root="/workspace/aad",
                remote_python="/opt/train/bin/python", min_cuda_version="13.0")
LAYOUT_B = dict(workspace_root="/srv/run", checkout_root="/srv/run/checkout",
                remote_python="/usr/local/venv/bin/python3",
                min_cuda_version="12.4")


class _StopAtSubmission(Exception):
    """Raised by the faked transport once the real `JobSpec` exists.

    The point of interest is the job the runner CONSTRUCTED, so the test stops
    there rather than faking the relay, the artifact gate and teardown as well.
    Everything before this point is production code.
    """


class _Provider:
    def get(self, pod_id):
        return PodState(pod_id=pod_id, exists=True, desired_status="RUNNING")

    def terminate(self, pod_id):
        return []


def _runner(monkeypatch, layout: dict, tmp_path: Path):
    """The real `SessionRunner.run`, faked only at its external I/O."""
    r = object.__new__(SR.SessionRunner)
    r.provider = _Provider()
    r.cli = "runpodctl"
    r.pod_id = ""
    r.price = 1.09
    r.start_epoch = 0.0
    r.ev = {}
    r.scr = tmp_path / "scr"
    r.scr.mkdir(parents=True, exist_ok=True)
    r.say = lambda m: r.ev.setdefault("said", []).append(m)
    r.save = lambda: None
    r.make_plan = lambda: True
    r.run_prechecks = lambda: True
    r.launch_watchdog = lambda: Path("/dev/null")
    r.teardown_now = lambda why: r.ev.setdefault("teardown", []).append(why)
    r.plan = types.SimpleNamespace(hard_terminate_minutes=834.0)
    #: Set by `setup_on_draw` in production; the fake below returns "ok"
    #: without doing the SSH, so the endpoint is supplied here.
    r.endpoint = ("10.0.0.1", "22")
    #: `context()` is production code and reads both of these. They carry no
    #: path, so they cannot influence what this test asserts.
    r.auth = types.SimpleNamespace(hard_cap_usd=15.1475)
    r.image_digest = "sha256:probe"

    #: The REAL `ExecutionCommands`, not a double: a stub that names only the
    #: fields it needs goes stale the moment the type gains one.
    commands = ExecutionCommands(
        watchdog="scripts/pod/watchdog.py",
        setup_script="scripts/pod/autoinit_preflight_setup.sh",
        artifact_collector="scripts/pod/collect_artifacts.py", **layout)
    ws = layout["workspace_root"]
    r.spec = types.SimpleNamespace(
        commands=commands, session_id="layout-probe",
        driver_job_id="probe_driver",
        driver_command=lambda ctx, plan: f"{layout['remote_python']} drive.py",
        run_log_path=f"{ws}/probe_run.log",
        status_path=f"{ws}/probe.status",
        materialize_inputs=lambda ctx: True)
    r.ws = commands.workspace_root
    r.repo = commands.checkout_root

    r.a = types.SimpleNamespace(
        create_attempts=1, host_draws=1, create_retry_seconds=300.0,
        gpu="NVIDIA L40S", max_price=1.09, image="img", disk_gb=200)

    captured: dict = {}

    def fake_run(cmd, **kw):
        if "create" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, '{"id":"pod1","costPerHr":1.09}', "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_start_detached(target, job, **kw):
        captured["job"] = job
        raise _StopAtSubmission

    monkeypatch.setattr(SR.subprocess, "run", fake_run)
    monkeypatch.setattr(SR.time, "sleep", lambda s: None)
    monkeypatch.setattr(SR.SessionRunner, "setup_on_draw",
                        lambda self, draw: "ok")
    monkeypatch.setattr(SR, "start_detached", fake_start_detached)
    #: `context()` is production code and builds an `SSHTarget`; nothing in
    #: this test connects, so the constructor is all that runs.
    return r, captured


def _submit(monkeypatch, layout, tmp_path):
    r, captured = _runner(monkeypatch, layout, tmp_path)
    with pytest.raises(_StopAtSubmission):
        r.run()
    assert "job" in captured, "the runner never reached driver submission"
    return captured["job"]


@pytest.mark.parametrize("layout", [LAYOUT_A, LAYOUT_B],
                         ids=["workspace_aad", "srv_run_checkout"])
def test_the_driver_job_paths_come_from_the_declared_layout(monkeypatch, layout,
                                                            tmp_path):
    """Executed, not read off the source. This is the line that raised."""
    job = _submit(monkeypatch, layout, tmp_path)

    assert job.workdir == layout["checkout_root"]
    assert job.env["PYTHONPATH"] == f"{layout['checkout_root']}/src"
    assert job.job_dir == f"{layout['workspace_root']}/jobs"
    assert job.command.startswith(layout["remote_python"])


def test_the_two_layouts_produce_different_jobs(monkeypatch, tmp_path):
    """The guard against a value that merely happens to match one default.

    Every path asserted above must actually differ between the two
    deployments, or the parametrized test could pass on a hard-coded constant.
    """
    a = _submit(monkeypatch, LAYOUT_A, tmp_path / "a")
    b = _submit(monkeypatch, LAYOUT_B, tmp_path / "b")

    assert a.workdir != b.workdir
    assert a.job_dir != b.job_dir
    assert a.env["PYTHONPATH"] != b.env["PYTHONPATH"]
    #: And neither leaks the other's roots anywhere in the descriptor.
    for job, mine, theirs in ((a, LAYOUT_A, LAYOUT_B), (b, LAYOUT_B, LAYOUT_A)):
        blob = f"{job.workdir} {job.job_dir} {job.env} {job.command}"
        assert theirs["checkout_root"] not in blob
        assert theirs["workspace_root"] not in blob
        assert mine["checkout_root"] in blob
