"""`--dry-run` must stop before a provider resource exists.

Several launchers advertised the flag as "run every $0 gate and stop before
provider creation" and NOTHING consulted it: `SessionRunner.run` went straight
from the prechecks to `create()`. A dry run whose gates all passed created a
real pod and billed for it, and the flag was only ever "safe" because some gate
happened to refuse first.

That is the kind of defect a test cannot find by reading a passing run, so this
one drives the runner to the boundary and asserts it stops there.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aadistill.infrastructure.session_runner import SessionRunner  # noqa: E402


class _Stop(Exception):
    """Raised by the fake `create` so a leak is unmistakable."""


def _runner_at_the_boundary(dry_run: bool):
    """A runner whose plan and prechecks pass and whose `create` is a tripwire."""
    runner = SessionRunner.__new__(SessionRunner)
    runner.ev = {}
    runner.said: list[str] = []
    runner.a = type("A", (), {"dry_run": dry_run, "host_draws": 1})()
    runner.say = runner.said.append
    runner.save = lambda: None
    runner.make_plan = lambda: True
    runner.run_prechecks = lambda: True

    def create():
        raise _Stop("create() was reached")

    runner.create = create
    return runner


def test_a_dry_run_stops_before_create_is_reached():
    runner = _runner_at_the_boundary(dry_run=True)
    assert SessionRunner.run(runner) is False
    assert runner.ev["dry_run"] is True
    assert runner.ev["provider_resource_created"] is False
    assert runner.ev["terminal"] == "DRY_RUN_GATES_PASSED"
    assert any("DRY RUN" in line for line in runner.said)


def test_a_dry_run_is_not_reported_as_a_pass():
    """It proves the chain is launchable, not that the session succeeded."""
    runner = _runner_at_the_boundary(dry_run=True)
    assert SessionRunner.run(runner) is False
    note = runner.ev["_dry_run_is_not_a_pass"].lower()
    assert "not evidence the session succeeded" in note
    assert "no pod existed" in note


def test_without_the_flag_the_runner_still_reaches_create():
    """Guards the guard: if this passed too, the test above would prove nothing
    and the flag could be doing anything at all."""
    runner = _runner_at_the_boundary(dry_run=False)
    try:
        SessionRunner.run(runner)
    except _Stop:
        return
    raise AssertionError("create() was not reached without --dry-run")


def test_a_namespace_without_the_flag_behaves_exactly_as_before():
    """`getattr` with a default, so a launcher that never declared the flag is
    unchanged rather than silently turned into a dry run."""
    runner = _runner_at_the_boundary(dry_run=False)
    del runner.a.__class__.dry_run
    try:
        SessionRunner.run(runner)
    except _Stop:
        return
    raise AssertionError("a namespace without dry_run did not reach create()")


def test_the_check_sits_after_the_prechecks_not_before_them():
    """A dry run that skipped the gates would report 'gates passed' having run
    none of them — the opposite of what the flag is for."""
    source = inspect.getsource(SessionRunner.run)
    gates = source.index("run_prechecks")
    flag = source.index('"dry_run"')
    assert gates < flag, "the dry-run check must come after the prechecks"
