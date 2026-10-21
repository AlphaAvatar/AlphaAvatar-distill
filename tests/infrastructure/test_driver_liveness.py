"""An unanswered driver probe on a billing pod is not a dead driver.

`remote.probe` reads UNKNOWN when nothing answers, and its contract says that
on a pod still billing this is a reason to keep looking. The poll loop ended on
the first one: on 2026-10-19 a formal search lost its pod to about 30 s of ssh
failure while the provider still reported the pod RUNNING and billing.
"""
from __future__ import annotations

from aadistill.infrastructure.session_runner import (
    UNANSWERED_DRIVER_GRACE_S, driver_terminal,
)


def test_a_crashed_driver_ends_the_session_at_once():
    assert driver_terminal("EXITED:1", None, 0.0) == ("DRIVER_EXITED:1", None)
    assert driver_terminal("EXITED:0", 5.0, 6.0)[0] == "DRIVER_EXITED:0"


def test_an_unanswered_driver_is_given_the_grace_then_ends():
    stop, since = driver_terminal("UNKNOWN", None, 100.0)
    assert (stop, since) == (None, 100.0)
    stop, since = driver_terminal("UNKNOWN", since, 100.0 + UNANSWERED_DRIVER_GRACE_S - 1)
    assert (stop, since) == (None, 100.0)
    assert driver_terminal("UNKNOWN", since, 100.0 + UNANSWERED_DRIVER_GRACE_S)[0] == "DRIVER_UNKNOWN"


def test_an_answer_resets_the_streak():
    _, since = driver_terminal("UNKNOWN", None, 0.0)
    assert driver_terminal("ALIVE", since, 30.0) == (None, None)
    stop, since = driver_terminal("UNKNOWN", None, UNANSWERED_DRIVER_GRACE_S + 10)
    assert stop is None and since == UNANSWERED_DRIVER_GRACE_S + 10
