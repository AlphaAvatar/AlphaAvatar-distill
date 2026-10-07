"""The 2026-10-06 stop, against the journal that run actually left behind.

D1's formal search `d1_search_20261006_101725` stopped at 449.8 minutes against
its own 1125.5-minute hard threshold. The launcher logged

    the pod is gone — the watchdog acted

and `finish_emergency` recorded

    the watchdog terminated the pod at the hard threshold (1126 min / $20.45)

while the watchdog's own journal recorded `action: none` on all 446 ticks with
`over_hard_limit: false` throughout. The real cause was an exhausted RunPod
account balance. Two records asserted a cause neither had checked.

`observed_stop_cause` is the generic repair and its behaviour is regressed in
core, at `tests/infrastructure/test_stop_cause_is_classified.py`. THIS file is
the half that is D1's: one run's journal, one run's elapsed minutes, one run's
threshold.

WHY IT MOVED HERE. The assertion used to sit in core behind

    @pytest.mark.skipif(not REAL_JOURNAL.is_file(), ...)

and the journal lives under `logs/.../runs/`, which is gitignored. So on a pod
and on any fresh checkout it skipped silently — and the skip-predicate audit
flagged it as the core suite's one unaccounted predicate: the signal is
classified, and nothing says the two machines must decide it the same way. A
regression that can vanish without anyone noticing is not a regression.
AGENTS.md §2.8a puts a closed run's own state with its experiment, which is
also where the fixture exists.

The journal is kept as the fixture rather than hand-built because a synthetic
one would only ever prove the classifier agrees with its author.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.provider import PodState  # noqa: E402
from aadistill.infrastructure.session_runner import (  # noqa: E402
    STOP_BY_PROVIDER, observed_stop_cause,
)

RUN_ID = "d1_search_20261006_101725"
REAL_JOURNAL = REPO / (
    f"logs/stages/stage-1/phase_d1/runs/{RUN_ID}/runtime/"
    "watchdog_zhk120whedjsg9.jsonl")

#: The run's own numbers, from its closeout. Not re-derived here.
ELAPSED_MINUTES = 449.81
HARD_TERMINATE_MINUTES = 1125.545

GONE = PodState(pod_id="zhk120whedjsg9", exists=False,
                desired_status="TERMINATED")


class TestTheRealIncident:

    def test_the_journal_this_asserts_against_is_present(self):
        """Stated as its own assertion rather than as a skip.

        In this suite the journal is evidence that is supposed to be here, so
        its absence is a finding about the checkout — not a reason to pass
        quietly. That inversion is the whole point of the move.
        """
        assert REAL_JOURNAL.is_file(), (
            f"{REAL_JOURNAL.relative_to(REPO)} is missing. It is the run's own "
            "watchdog journal and the only evidence that the watchdog did not "
            "act; restore it from the run's artifacts before trusting any "
            "attribution of the 2026-10-06 stop.")

    def test_the_stop_is_not_attributed_to_the_watchdog(self):
        got = observed_stop_cause(
            GONE, watchdog_journal=REAL_JOURNAL,
            elapsed_minutes=ELAPSED_MINUTES,
            hard_terminate_minutes=HARD_TERMINATE_MINUTES)
        assert got["cause"] == STOP_BY_PROVIDER
        assert got["watchdog_took_an_action"] is False
        assert got["over_hard_limit"] is False

    def test_the_journal_is_read_and_not_assumed(self):
        """A classifier that never opened the file would report zero ticks and
        still reach the same verdict above, so the tick count is what shows it
        looked."""
        got = observed_stop_cause(
            GONE, watchdog_journal=REAL_JOURNAL,
            elapsed_minutes=ELAPSED_MINUTES,
            hard_terminate_minutes=HARD_TERMINATE_MINUTES)
        assert got["watchdog_ticks"] > 400

    def test_the_elapsed_time_was_well_inside_the_threshold(self):
        """The arithmetic the launcher's claim required to be false. 449.81 of
        1125.545 is 40% of the bound, so "terminated at the hard threshold"
        was not merely unchecked — it was contradicted by two numbers the
        launcher already had."""
        assert ELAPSED_MINUTES / HARD_TERMINATE_MINUTES < 0.5

    @pytest.mark.parametrize("missing", [None, "/nowhere/watchdog.jsonl"])
    def test_without_the_journal_the_cause_is_not_claimed_either_way(
            self, missing):
        """The failure mode generalised: no journal must not become a default
        attribution in the other direction."""
        got = observed_stop_cause(
            GONE, watchdog_journal=missing,
            elapsed_minutes=ELAPSED_MINUTES,
            hard_terminate_minutes=HARD_TERMINATE_MINUTES)
        assert got["watchdog_took_an_action"] is False
        assert got["watchdog_ticks"] == 0
