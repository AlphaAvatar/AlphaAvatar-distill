"""The normalized signature of a paid attempt's deterministic failure.

A deterministic paid failure may not be retried unchanged. Enforcing that
needs one thing the repository did not have: a way to say whether THIS
attempt failed the same way as the LAST one. A3 answered that question
twenty-one times with twenty-one pods, because the only available comparison
was a human reading two logs.

So a signature is derived from the failure text: the class of failure, plus
the one identifying token that distinguishes it from a sibling of the same
class. `SETUP_REQUIRED_VAR:SESSION_FROZEN_EXPECT` is the same failure in every
run, while timestamps, pod ids, prices, paths and durations are not part of
it -- those are what made two identical failures look different.

**What is deliberately NOT a deterministic signature**: provider capacity, a
cold host, an unreachable endpoint, a transport timeout. Those are transient
acquisition failures, they are expected to differ run to run, and the existing
backoff/reacquisition policy handles them. `signature_of` returns `None` for
them, which is how the caller tells "retry under backoff" from "do not create
another resource".

The reusable core carries the MECHANISM. It names no experiment, no stage, no
session kind and no repository path; the patterns are failure classes of the
shared setup/launch path, which is deployment-level infrastructure, and a
caller that recognizes a new class adds it here with its own class name.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

#: (class, pattern, group to keep). The kept group is the identifying token,
#: so two failures of one class with different subjects are different
#: signatures -- a missing `SESSION_FROZEN_EXPECT` and a missing
#: `SESSION_PLAN_HASH` are not the same failure and must not be collapsed.
_DETERMINISTIC: tuple[tuple[str, re.Pattern[str], int], ...] = (
    #: `: "${VAR:?message}"` in the shared setup script.
    ("SETUP_REQUIRED_VAR",
     re.compile(r"setup\.sh: line \d+: ([A-Z_][A-Z0-9_]*): "), 1),
    #: A declared setup step that ran and refused.
    ("SETUP_MARKER_FAILED", re.compile(r"\bmark(?:er)?\s+\"?([A-Z_]+_FAILED)"), 1),
    #: The shared setup's own numbered exits.
    ("SETUP_EXIT", re.compile(r"\bSETUP_RC=(\d+)"), 1),
    #: argparse, on either side of a launcher/driver or shell/CLI seam.
    ("ARGPARSE_UNRECOGNIZED",
     re.compile(r"error: unrecognized arguments: (--[a-z][a-z0-9-]*)"), 1),
    ("ARGPARSE_REQUIRED",
     re.compile(r"error: the following arguments are required: "
                r"(--[a-z][a-z0-9-]*)"), 1),
    #: A pre-provider gate refusing. The gate's NAME is the token.
    ("GATE_REFUSED", re.compile(r"ABORT at \$0: .*?\b(\w+_gate)\b"), 1),
    #: A driver stage raising a terminal integrity stop.
    ("DRIVER_INTEGRITY", re.compile(r"\b(A3_INTEGRITY_FAILURE|"
                                    r"A3_REPLAY_MISMATCH|"
                                    r"FixedPathDigestMismatch)\b"), 1),
    #: A python exception that reached the top of the launcher.
    ("LAUNCHER_EXCEPTION",
     re.compile(r"LAUNCHER ERROR: (\w+(?:Error|Exception))"), 1),
)

#: Transient acquisition failures. Present so the distinction is explicit and
#: testable rather than implied by the absence of a deterministic match.
_TRANSIENT: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PROVIDER_CAPACITY",
     re.compile(r"no longer any instances available|"
                r"could not create a pod|insufficient capacity",
                re.I)),
    ("COLD_HOST", re.compile(r"\bcold host\b|exit 90\b|setup gate timed out",
                             re.I)),
    ("UNREACHABLE", re.compile(r"ssh never became reachable|"
                               r"TCP 22 never|startup limit", re.I)),
)


def transient_reason(text: str) -> str | None:
    """The transient-acquisition class this text shows, if any."""
    for name, pattern in _TRANSIENT:
        if pattern.search(text):
            return name
    return None


def signature_of(text: str) -> str | None:
    """`CLASS:token` for a deterministic failure, else `None`.

    `None` means either "this attempt did not fail deterministically" or
    "this attempt failed in a way that is expected to differ between runs".
    Both answers permit a retry under the ordinary policy; only a non-None
    signature that MATCHES the previous attempt's forbids creating another
    resource.

    A transient class wins over a deterministic one when both appear, because
    a capacity refusal logged beside a gate's informational line is a capacity
    refusal.
    """
    if not text:
        return None
    if transient_reason(text):
        return None
    for name, pattern, group in _DETERMINISTIC:
        m = pattern.search(text)
        if m:
            return f"{name}:{m.group(group)}"
    return None


def same_unchanged_failure(previous: str | None, current: str | None,
                           *, corrective_change: bool) -> bool:
    """Would retrying now be a repetition rather than a repair?

    True only when both attempts carry the SAME non-None signature and nothing
    corrective changed. Expressed as its own function so the rule is one
    readable predicate with its own tests, rather than a condition spread
    across a launcher.
    """
    if not previous or not current:
        return False
    if previous != current:
        return False
    return not corrective_change


def latest_signature(texts: Sequence[str]) -> str | None:
    """The signature of the most recent attempt that produced one."""
    for text in reversed(list(texts)):
        sig = signature_of(text)
        if sig:
            return sig
    return None
