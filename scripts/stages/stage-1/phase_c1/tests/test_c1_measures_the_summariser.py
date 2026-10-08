"""C1's harness must measure the pytest-outcome summariser.

Split from `tests/integration/test_cpu_test_outcome_evidence.py` by the
2026-10-03 convergence round. The summariser's own behaviour — parsing JUnit,
preserving every skip reason, refusing a pod whose suite passed but whose skip
set is not the certified one — is generic and stays in core. Whether C1's
declared harness source set INCLUDES it is C1's wiring: a grant that did not
measure a file able to refuse its pod would be a grant over the wrong bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def test_the_summariser_is_inside_the_measured_harness():
    """C1's measured harness included the summariser — a historical fact the
    frozen declaration carries at its freeze-time spelling. The 2026-10-08
    migration moved the file; the spelling stays, and its current address
    resolves through the historical-path table."""
    from shared.run_layout import resolve_historical
    from stages.phase_c1.authorization import C1_HARNESS_SOURCE_FILES_V1

    frozen = "scripts/pod/summarize_pytest_outcomes.py"
    assert frozen in C1_HARNESS_SOURCE_FILES_V1
    assert (REPO / resolve_historical(frozen, REPO)).is_file()
