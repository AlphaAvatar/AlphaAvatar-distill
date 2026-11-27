"""The launcher polls for at least as long as the plan lets the pod run (`session_runner.poll_lifetime`).

A shared launcher's fixed 600 min default ended a 3,420 min training session at 605 min, mid-training: the poll loop
reached its limit and the launcher collected and tore the pod down while the driver still ran.
"""
from types import SimpleNamespace

from aadistill.infrastructure.session_runner import poll_lifetime


def test_an_unset_limit_is_derived_from_the_hard_bound():
    args = SimpleNamespace(poll_seconds=60.0, poll_limit_min=None)
    ok, why = poll_lifetime(args, 3420.0)
    assert ok, why
    assert args.poll_limit_min == 3422.0          # the hard bound plus two 60 s polls


def test_a_limit_below_the_hard_bound_is_refused_before_any_pod():
    args = SimpleNamespace(poll_seconds=60.0, poll_limit_min=600.0)
    ok, why = poll_lifetime(args, 3420.0)
    assert not ok and "3420.0" in why
    assert args.poll_limit_min == 600.0           # refused, never silently raised


def test_a_limit_that_clears_the_bound_is_kept():
    args = SimpleNamespace(poll_seconds=60.0, poll_limit_min=1967.87)
    ok, _ = poll_lifetime(args, 1925.87)
    assert ok and args.poll_limit_min == 1967.87
    ok, _ = poll_lifetime(SimpleNamespace(poll_seconds=60.0, poll_limit_min=1926.0), 1925.87)
    assert not ok                                 # inside the bound but short of the two polls that see the pod go


def test_a_namespace_without_the_flag_derives_it():
    args = SimpleNamespace(poll_seconds=90.0)
    ok, _ = poll_lifetime(args, 100.0)
    assert ok and args.poll_limit_min == 103.0
