"""Phase A's authorization constant is a SCHEMA, not a grant.

Split from `tests/integration/test_session_architecture.py` by the 2026-10-03
convergence round. The invariant — that no authorization constant anywhere in the
experiments tree carries attempt-specific grant prose — stays in core and now
DISCOVERS every such constant rather than naming three. These assertions are about
Phase A's constant in particular: that its `granted_by` is the required-prose
sentinel and that its issue-time fields are still placeholders.

`plan.py` carried attempt-7's grant inside the constant, where it still read as
current after the attempt was over.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def test_the_phase_a_authorization_schema_carries_no_grant():
    from stages.phase_a.plan import (
        GRANT_PROSE_REQUIRED,
        PHASE_A_AUTHORIZATION,
    )

    assert PHASE_A_AUTHORIZATION.granted_by == GRANT_PROSE_REQUIRED, (
        "the Phase-A authorization schema carries grant prose again")
    for attr in ("granted_utc", "science_plan_hash"):
        assert getattr(PHASE_A_AUTHORIZATION, attr) == "PLACEHOLDER", attr
    assert PHASE_A_AUTHORIZATION.authorized_session_commit is None
    assert PHASE_A_AUTHORIZATION.harness_source_digest is None
