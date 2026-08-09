"""Static and CLI-level guards for the post-E6b session contract.

The unit tests in `tests/infrastructure/` prove the modules behave. These prove
the *scripts a session actually runs* are wired to them — which is the gap that
produced E6b in the first place: the behaviour was understood, and the launcher
that ran did something else.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POD = REPO / "scripts/pod"
PY = sys.executable
ENV_SRC = {"PYTHONPATH": str(REPO / "src")}


def run(args, **kw):
    import os
    env = {**os.environ, **ENV_SRC}
    return subprocess.run([PY, *args], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120, **kw)


# --------------------------------------------------------------------------
# The banned construct
# --------------------------------------------------------------------------

# `$(ls …)` inside a quoted ssh command is expanded by the remote login shell.
# When the glob matches nothing the substitution is empty, `tar` bundles a
# smaller set, exits 0, and the digest of the incomplete bundle verifies. These
# three launchers drove completed experiments and are frozen as records of how
# those runs were driven (P4); they are named individually so that a launcher
# *derived* from one of them is caught rather than inheriting the exemption.
FROZEN_RECORD_LAUNCHERS = {
    "e3_launch.sh": "E3 complete 2026-08-06; checkpoints deleted 2026-08-09",
    "e4_launch.sh": "E4 complete 2026-08-07",
    "e5_launch.sh": "E5 complete 2026-08-08",
}


def launchers():
    return sorted(POD.glob("*_launch.sh"))


def test_launchers_are_discovered():
    assert launchers(), "the lint below would be vacuous"


@pytest.mark.parametrize("path", launchers(), ids=lambda p: p.name)
def test_no_command_substitution_over_globs_in_new_launchers(path):
    text = path.read_text()
    offenders = [line.strip() for line in text.splitlines()
                 if "$(ls " in line or "`ls " in line]
    if path.name in FROZEN_RECORD_LAUNCHERS:
        pytest.skip(f"frozen record: {FROZEN_RECORD_LAUNCHERS[path.name]}")
    assert not offenders, (
        f"{path.name} expands a glob through command substitution inside an "
        f"ssh command: {offenders}. A pattern that matches nothing yields an "
        "empty substitution and a silently short archive. Use "
        "scripts/pod/collect_artifacts.py, which expands the spec in Python "
        "and reports what it could not find.")


@pytest.mark.parametrize("name", sorted(FROZEN_RECORD_LAUNCHERS))
def test_the_frozen_allowlist_only_names_launchers_that_exist(name):
    """A stale exemption is an exemption that will be inherited by accident."""
    assert (POD / name).is_file(), (
        f"{name} is exempted from the glob lint but no longer exists; drop it "
        "from FROZEN_RECORD_LAUNCHERS rather than leaving a name that a future "
        "script could take.")


# --------------------------------------------------------------------------
# The scripts exist and run
# --------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["start_job.py", "watchdog.py",
                                    "collect_artifacts.py"])
def test_the_hardening_entry_points_are_runnable(script):
    result = run([str(POD / script), "--help"])
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_watchdog_runs_end_to_end_against_the_simulated_control_plane(tmp_path):
    """A session can rehearse its thresholds without creating a pod."""
    journal = tmp_path / "watchdog.jsonl"
    result = run([
        str(POD / "watchdog.py"), "--pod-id", "sim-1",
        "--session-start-epoch", "0",          # long past every limit
        "--price-per-hour", "0.99", "--hard-minutes", "1",
        "--authorized-usd", "9.00", "--journal", str(journal),
        "--verify-delay-seconds", "0", "--once", "--simulate",
    ])
    assert result.returncode == 0, result.stderr
    events = [json.loads(l)["event"] for l in journal.read_text().splitlines()]
    assert "hard_limit_reached" in events
    assert "terminated" in events


def test_start_job_refuses_to_forward_an_empty_variable():
    """E6b's setup died on a missing TEACHER_REVISION after paying for setup."""
    result = run([
        str(POD / "start_job.py"), "--host", "127.0.0.1", "--port", "22",
        "--job-id", "j", "--workdir", "/tmp", "--command", "true",
        "--log", "/tmp/j.log", "--status", "/tmp/j.status",
        "--env", "TEACHER_REVISION=",
    ])
    assert result.returncode == 2
    assert "refusing to forward a blank value" in result.stderr


def test_collect_artifacts_reports_a_missing_event_stream_and_fails(tmp_path):
    root = tmp_path / "artifacts" / "stage3" / "e7_arm_sa"
    root.mkdir(parents=True)
    (root / "run_manifest.json").write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([
        {"artifact_class": "event_stream",
         "pattern": "stage3/e7_*/train_log.jsonl"},
        {"artifact_class": "run_manifest",
         "pattern": "stage3/e7_*/run_manifest.json"},
    ]))

    result = run([
        str(POD / "collect_artifacts.py"), "manifest",
        "--root", str(tmp_path / "artifacts"), "--spec", str(spec),
        "--out", str(tmp_path / "manifest.json"),
    ])
    assert result.returncode == 5
    assert "MISSING event_stream" in result.stdout
    assert (tmp_path / "manifest.json").is_file(), (
        "the manifest is written even when incomplete; it is the record of "
        "what was looked for")


def test_collect_artifacts_full_cycle(tmp_path):
    arm = tmp_path / "artifacts" / "stage3" / "e7_arm_sa"
    arm.mkdir(parents=True)
    (arm / "train_log.jsonl").write_text('{"event":"train_step","step":0}\n')
    (arm / "run_manifest.json").write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([
        {"artifact_class": "event_stream",
         "pattern": "stage3/e7_*/train_log.jsonl"},
        {"artifact_class": "run_manifest",
         "pattern": "stage3/e7_*/run_manifest.json"},
    ]))
    manifest = tmp_path / "manifest.json"
    archive = tmp_path / "bundle.tar.gz"

    assert run([str(POD / "collect_artifacts.py"), "manifest", "--root",
                str(tmp_path / "artifacts"), "--spec", str(spec),
                "--out", str(manifest)]).returncode == 0
    assert run([str(POD / "collect_artifacts.py"), "archive", "--manifest",
                str(manifest), "--out", str(archive)]).returncode == 0
    assert run([str(POD / "collect_artifacts.py"), "verify-archive",
                "--manifest", str(manifest),
                "--archive", str(archive)]).returncode == 0

    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "training_complete": True, "evaluation_complete": True,
        "artifact_manifest_created": True, "required_files_present": True,
        "archive_created": True, "archive_contents_verified": True,
        "transfer_complete": True, "local_hashes_verified": True,
        "checkpoint_hashes_matched": True, "report_inputs_verified": True}))
    gate = run([str(POD / "collect_artifacts.py"), "gate", "--state",
                str(state)])
    assert gate.returncode == 0
    assert json.loads(gate.stdout)["allowed"] is True


def test_the_gate_cli_blocks_when_a_check_is_missing(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"training_complete": True}))
    result = run([str(POD / "collect_artifacts.py"), "gate", "--state",
                  str(state)])
    assert result.returncode == 5
    assert json.loads(result.stdout)["failed_check"] == "evaluation_complete"


def test_the_archive_step_refuses_an_incomplete_manifest(tmp_path):
    arm = tmp_path / "artifacts" / "stage3" / "e7_arm_sa"
    arm.mkdir(parents=True)
    (arm / "run_manifest.json").write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([
        {"artifact_class": "event_stream",
         "pattern": "stage3/e7_*/train_log.jsonl"}]))
    manifest = tmp_path / "manifest.json"
    run([str(POD / "collect_artifacts.py"), "manifest", "--root",
         str(tmp_path / "artifacts"), "--spec", str(spec), "--out",
         str(manifest)])

    blocked = run([str(POD / "collect_artifacts.py"), "archive", "--manifest",
                   str(manifest), "--out", str(tmp_path / "b.tar.gz")])
    assert blocked.returncode == 5
    assert "refusing to archive an incomplete manifest" in blocked.stdout
