"""The continuation shares Phase A's science and not its session.

Split from `tests/integration/test_session_architecture.py` by the 2026-10-03
convergence round. The structural rules — every spec validates, specs are frozen,
no spec subclasses the runner, every session declares its own operational fields —
stay in core and hold over every launcher. This is the continuation's own claim
about itself: the same frozen plan, a different session in every field that names
a run, its own authorization type and harness, and a budget with Stage-1's search
phase and both reserves removed so it cannot be priced as if it ran one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests"))

from support.session_specs import all_specs  # noqa: E402


def test_the_recovery_continuation_shares_the_science_and_not_the_session():
    """The one intentional `plan_hash` collision, asserted rather than allowed.

    Dropping `plan_hash` from the uniqueness list above removes a check; this
    replaces it with a stronger and more specific one. The continuation must
    share the **full Phase-A** plan hash — nothing was rewritten to pretend
    Phase A always began at Stage 2 — while being a different operational
    session in every respect that decides what runs, what it costs, and what
    permits it.
    """
    by_name = {name: spec for name, _m, _a, spec in all_specs()}
    cont = by_name["autoinit_recovery_continuation_launch"]
    phase_a = by_name["autoinit_phase_a_launch"]

    # Same science, deliberately and exactly.
    assert cont.plan_hash == phase_a.plan_hash
    assert cont.plan_id == phase_a.plan_id
    assert cont.plan_hash == (
        "9377a2dc61f21790dd111d72a5de0e039ea1d31afef2d09e18c98a0b0cc2a0aa"), (
        "the frozen Phase-A session plan moved")

    # Different operational session, in every field that names a run.
    for field in ("session_id", "schema", "status_path", "run_log_path",
                  "authorization_path", "driver_job_id"):
        assert getattr(cont, field) != getattr(phase_a, field), field

    # Its own authorization TYPE and harness, not Phase A's.
    from stages.recovery_continuation.session import RECOVERY_CONTINUATION_HARNESS_FILES_V1, RecoveryContinuationAuthorization
    from stages.phase_a.plan import PHASE_A_HARNESS_SOURCE_FILES_V1, PhaseAAuthorization
    assert cont.authorization_loader == RecoveryContinuationAuthorization.load
    assert phase_a.authorization_loader == PhaseAAuthorization.load
    assert (set(RECOVERY_CONTINUATION_HARNESS_FILES_V1)
            != set(PHASE_A_HARNESS_SOURCE_FILES_V1))
    assert ("scripts/stages/stage-1/phase_a/phase_a_search.py"
            not in RECOVERY_CONTINUATION_HARNESS_FILES_V1)

    # Its own budget: the Stage-1 search phase and both Stage-1 reserves are
    # gone, so it cannot be priced as if it were running one.
    plan = cont.budget.plan(price_per_hour=0.99, authorized_usd=16.7456)
    assert not [p for p in plan.breakdown if p.name == "stage1_beam_search"]
    assert plan.soft_stop_reserves == ()
    assert plan.hard_terminate_usd == pytest.approx(16.7456, abs=1e-4)
    full = phase_a.budget.plan(price_per_hour=0.99, authorized_usd=23.0484)
    assert plan.hard_terminate_usd < full.hard_terminate_usd

    # And it says so about itself.
    assert cont.evidence_fields["runs_a_search"] is False
    assert phase_a.evidence_fields.get("runs_a_search") is not False
