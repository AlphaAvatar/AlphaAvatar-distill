"""`SessionRunner.save` must create the directory it writes into.

`save()` is the only thing that puts the session record on the dev box, and it
runs on every path: immediately after `create()` registers a pod id, after each
draw, at the dry-run stop, from `teardown_now`, and from `run_session`'s
closeout. It did

    (self.repo_root / self.a.out).write_text(...)

with no `mkdir`, so a missing parent directory did not fail where `--out` was
configured. It raised `FileNotFoundError` out of the FIRST `save()` after the
provider returned a pod id — the moment the record matters most, and the one
place an exception costs money rather than time.

It survived because every launcher so far passed an `--out` whose parent already
existed. The D1 launcher resolves its default to
`runs/<run_id>/runtime/session.json`, which is where the shared run layout puts
a session record and which nothing creates in advance; its first real
pre-provider dry run reached `DRY_RUN_GATES_PASSED` with all six gates green and
then died here.

Generic, not D1's: the hazard belongs to any session whose record lands anywhere
that does not already exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aadistill.infrastructure.session_runner import SessionRunner  # noqa: E402


def _saver(repo_root: Path, out: str) -> SessionRunner:
    """A runner with nothing but what `save` reads. Nothing else is reachable."""
    runner = SessionRunner.__new__(SessionRunner)
    runner.repo_root = repo_root
    runner.a = type("A", (), {"out": out})()
    runner.ev = {"session_id": "probe", "terminal": "PROBE"}
    return runner


def test_it_creates_a_missing_parent_directory(tmp_path):
    runner = _saver(tmp_path, "logs/runs/a_run/runtime/session.json")
    runner.save()
    written = tmp_path / "logs/runs/a_run/runtime/session.json"
    assert written.is_file()
    assert json.loads(written.read_text())["session_id"] == "probe"


def test_it_creates_several_missing_levels(tmp_path):
    """The run layout nests: stage, experiment, run, area."""
    runner = _saver(
        tmp_path, "logs/stages/stage-1/an_experiment/runs/a_run/runtime/s.json")
    runner.save()
    assert (tmp_path / "logs/stages/stage-1/an_experiment/runs/a_run/runtime/"
            "s.json").is_file()


def test_it_overwrites_an_existing_record(tmp_path):
    """`save()` is called many times per session and each call must land: the
    record is rewritten on every state change, not appended to."""
    runner = _saver(tmp_path, "logs/r/session.json")
    runner.save()
    runner.ev["terminal"] = "ALL_DONE"
    runner.save()
    body = json.loads((tmp_path / "logs/r/session.json").read_text())
    assert body["terminal"] == "ALL_DONE"


def test_an_existing_directory_is_not_an_error(tmp_path):
    """`exist_ok`: the second and later saves of a session find it there."""
    (tmp_path / "logs/r").mkdir(parents=True)
    runner = _saver(tmp_path, "logs/r/session.json")
    runner.save()
    runner.save()
    assert (tmp_path / "logs/r/session.json").is_file()


def test_the_repair_is_what_makes_this_pass(tmp_path, monkeypatch):
    """The mutation check: with the `mkdir` removed, a missing parent raises.

    Without this the three tests above would pass against a `save()` that
    merely happened to be called where the directory existed, and the
    regression would not be pinned to the repair.
    """
    runner = _saver(tmp_path, "logs/runs/a_run/runtime/session.json")

    def save_without_mkdir(self) -> None:
        (self.repo_root / self.a.out).write_text(json.dumps(self.ev) + "\n")

    monkeypatch.setattr(SessionRunner, "save", save_without_mkdir)
    with pytest.raises(FileNotFoundError):
        runner.save()
