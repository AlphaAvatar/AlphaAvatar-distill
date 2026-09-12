"""This project's response rule for the GPU repeatability measurement.

The application layer. `aadistill.initialization.planning.ranking` owns what an
`EpsilonResponseRule` IS and how it evaluates a measured range; it used to hold
the instance too, and the instance's prose named Phase A -- so a generic ranking
module told every reader which of THIS project's phases would not start.

**The four sentences are transcribed verbatim.** They are recorded in
`logs/cross-stage/phase_a/plans/autoinit_phase_a_preregistration.json`, which is frozen, so the rule must
keep producing exactly these bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.planning.ranking import (  # noqa: E402
    PARETO_V1, EpsilonResponseRule)

#: Frozen against the shipped policy's smallest declared epsilon.
EPSILON_RESPONSE_V1 = EpsilonResponseRule(
    declared_epsilon=min(PARETO_V1.epsilon.values()),
    proceed_note=("proceed: the measured range is below the declared epsilon, "
                  "which stands unchanged"),
    stop_note=("STOP: the measured range reaches or exceeds the declared "
               "epsilon. Do NOT materialize a new epsilon automatically. "
               "Mark the preflight as requiring review and do not start "
               "Phase A; a beam tolerance derived from one profiling run "
               "is not a scientific tolerance."),
    if_below_note="epsilon stands unchanged",
    if_at_or_above_note=("no automatic re-derivation; preflight requires "
                         "review and Phase A is blocked"),
    blocked_key="phase_a_blocked",
)

__all__ = ["EPSILON_RESPONSE_V1"]
