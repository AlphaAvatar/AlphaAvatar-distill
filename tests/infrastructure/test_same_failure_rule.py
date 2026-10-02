"""A deterministic paid failure may not be retried unchanged.

The rule existed in `AGENTS.md` and nothing executed it, so A3's acquisition
loop created twenty-one paid pods discovering one missing setup variable.
These are the cases that decide whether a provider resource may be created:
the same signature with nothing changed must REFUSE, and a provider transient
must not be mistaken for one.

Table-driven over the predicate rather than over a launcher, because the
predicate is the thing with the rule in it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.failure_signature import (  # noqa: E402
    latest_signature, same_unchanged_failure, signature_of, transient_reason,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    same_failure_gate,
)

FROZEN_EXPECT = (
    "[04:43:29] offline install completed in 10s\n"
    "MARKER:TRAIN_ENV\n"
    "/workspace/autoinit_preflight_setup.sh: line 362: SESSION_FROZEN_EXPECT: "
    "a session declaring ASSETS_READY must name its frozen-asset expectation "
    "document\nSETUP_RC=1\n")
PLAN_HASH = FROZEN_EXPECT.replace("SESSION_FROZEN_EXPECT",
                                  "SESSION_PLAN_HASH")
CAPACITY = ('[20:53:33] attempt 1: create failed — {"error":"There are no '
            'longer any instances available with the requested '
            'specifications. Please refresh and try again."}\n'
            "[20:53:33] ABORT: could not create a pod\n")
CLEAN = "[05:02:21] draw 1: running setup\nMARKER:SETUP_DONE\nALL_DONE\n"


@pytest.mark.parametrize("text,expected", [
    (FROZEN_EXPECT, "SETUP_REQUIRED_VAR:SESSION_FROZEN_EXPECT"),
    (PLAN_HASH, "SETUP_REQUIRED_VAR:SESSION_PLAN_HASH"),
    ("error: unrecognized arguments: --scr /tmp/x",
     "ARGPARSE_UNRECOGNIZED:--scr"),
    ("error: the following arguments are required: --max-price",
     "ARGPARSE_REQUIRED:--max-price"),
    ("[20:50:05] ABORT at $0: the readiness_gate refused",
     "GATE_REFUSED:readiness_gate"),
    ("LAUNCHER ERROR: FileNotFoundError: no such file",
     "LAUNCHER_EXCEPTION:FileNotFoundError"),
    (CAPACITY, None),
    (CLEAN, None),
    ("", None),
])
def test_the_signature_is_what_it_should_be(text, expected):
    assert signature_of(text) == expected


def test_two_missing_variables_are_not_the_same_failure():
    """The kept token is what makes the class useful.

    Collapsing every `${VAR:?}` refusal into one signature would make fixing
    `SESSION_FROZEN_EXPECT` look like it addressed a later, different missing
    variable -- and the gate would then refuse a legitimate retry.
    """
    assert signature_of(FROZEN_EXPECT) != signature_of(PLAN_HASH)


@pytest.mark.parametrize("text,reason", [
    (CAPACITY, "PROVIDER_CAPACITY"),
    ("[12:00:00] draw 2: cold host, exit 90", "COLD_HOST"),
    ("[12:00:00] ssh never became reachable within the startup limit",
     "UNREACHABLE"),
    (FROZEN_EXPECT, None),
])
def test_transients_are_named_rather_than_implied(text, reason):
    """So "retry under backoff" and "do not create another resource" are
    distinguished by a positive classification, not by a missing match."""
    assert transient_reason(text) == reason


def test_a_transient_beside_a_gate_line_is_still_a_transient():
    """A capacity refusal logged after an informational gate line must not
    read as a deterministic gate refusal -- that would stop the backoff
    policy the maintainer explicitly preserved."""
    mixed = "[20:50:05] readiness_gate PASS\n" + CAPACITY
    assert signature_of(mixed) is None
    assert transient_reason(mixed) == "PROVIDER_CAPACITY"


@pytest.mark.parametrize("prev,cur,changed,expected", [
    ("A:x", "A:x", False, True),    # the twenty-one-pod case
    ("A:x", "A:x", True, False),    # repaired
    ("A:x", "B:y", False, False),   # a different failure
    (None, "A:x", False, False),    # nothing to repeat
    ("A:x", None, False, False),
])
def test_the_predicate(prev, cur, changed, expected):
    assert same_unchanged_failure(prev, cur,
                                  corrective_change=changed) is expected


def test_latest_signature_takes_the_most_recent_one_that_exists():
    """A capacity refusal after a deterministic failure must not erase it:
    the deterministic one is still the thing a retry would repeat."""
    assert latest_signature([CLEAN, FROZEN_EXPECT, CAPACITY]) == (
        "SETUP_REQUIRED_VAR:SESSION_FROZEN_EXPECT")
    assert latest_signature([FROZEN_EXPECT, PLAN_HASH]) == (
        "SETUP_REQUIRED_VAR:SESSION_PLAN_HASH")
    assert latest_signature([CLEAN, CAPACITY]) is None


# --- the gate, which is what actually stops a create call ------------------

def _ctx():
    return types.SimpleNamespace(evidence={})


def test_the_gate_refuses_an_unchanged_repetition():
    gate = same_failure_gate(
        lambda: [("a3_attempt27", FROZEN_EXPECT),
                 ("a3_attempt28", FROZEN_EXPECT)],
        corrective_change=lambda: (False, "nothing is committed beyond HEAD"))
    ctx = _ctx()
    ok, why = gate(ctx)
    assert ok is False
    assert "SETUP_REQUIRED_VAR:SESSION_FROZEN_EXPECT" in why
    assert "a3_attempt28" in why, "the message must name the attempt repeated"
    assert "may not be retried unchanged" in why
    ev = ctx.evidence["same_failure_rule"]
    assert ev["previous_deterministic_signature"] == (
        "SETUP_REQUIRED_VAR:SESSION_FROZEN_EXPECT")
    assert ev["corrective_change"] is False


def test_the_gate_allows_a_repaired_retry():
    gate = same_failure_gate(
        lambda: [("a3_attempt28", FROZEN_EXPECT)],
        corrective_change=lambda: (True, "2 tracked paths changed"))
    ok, why = gate(_ctx())
    assert ok is True and "is addressed" in why


def test_the_gate_allows_a_capacity_retry_even_with_nothing_changed():
    """The behaviour the maintainer explicitly preserved: a provider
    transient follows the backoff policy, and nothing about it is a
    repetition to forbid."""
    gate = same_failure_gate(
        lambda: [("a3_attempt4", CAPACITY), ("a3_attempt5", CAPACITY)],
        corrective_change=lambda: (False, "nothing changed"))
    ok, why = gate(_ctx())
    assert ok is True
    assert "no previous attempt failed deterministically" in why


def test_the_gate_allows_the_first_attempt_of_all():
    gate = same_failure_gate(lambda: [],
                             corrective_change=lambda: (True, "no prior"))
    ok, _ = gate(_ctx())
    assert ok is True


def test_a_different_deterministic_failure_is_not_a_repetition():
    """Fixing one missing variable and hitting the NEXT one is progress, and
    the rule must not read it as repetition -- otherwise the first real
    repair would deadlock the chain."""
    gate = same_failure_gate(
        lambda: [("a3_attempt28", FROZEN_EXPECT), ("a3_attempt29", PLAN_HASH)],
        corrective_change=lambda: (False, "nothing changed"))
    ok, why = gate(_ctx())
    #: The signature it compares is the LATEST one, and the current attempt
    #: has not run yet -- so a prior-vs-prior comparison must not fire. What
    #: refuses a repetition is the next attempt ending the same way again.
    assert ok is False, why
    assert "SESSION_PLAN_HASH" in why, (
        "the newest deterministic failure is the one a retry would repeat")
