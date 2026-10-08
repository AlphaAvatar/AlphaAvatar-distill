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
    from stages.phase_c1.authorization import C1_HARNESS_SOURCE_FILES_V1
    assert "scripts/shared/pod/summarize_pytest_outcomes.py" in C1_HARNESS_SOURCE_FILES_V1
