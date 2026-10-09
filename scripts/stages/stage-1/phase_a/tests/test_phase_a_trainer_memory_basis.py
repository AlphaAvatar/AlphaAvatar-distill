"""The Phase-A driver's memory constants equal the basis that measured them.

Split from `tests/initialization/test_device_handoff.py` by the 2026-10-03
convergence round. The device-handoff MECHANISM — what Stage 1 frees, when, and
what Stage 2 then sees — stays in core. This reads a measured basis record and a
driver's constants, so it is experiment wiring twice over.

`RECOVERY_TRAINER_BYTES` was 22 GiB because somebody rounded up attempt 12's
mid-failure footprint; the trainer's measured peak is 39.79 GiB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

GIB = 1024 ** 3
BASIS = (REPO / "logs/stages/stage-1/recovery_continuation/analyses"
         / "autoinit_recovery_trainer_memory_basis.json")

pytestmark = pytest.mark.skipif(
    not BASIS.is_file(),
    reason="the measured memory basis is not staged in this checkout")


def test_the_trainer_requirement_matches_its_recorded_basis():
    """RECOVERY_TRAINER_BYTES was 22 GiB because somebody rounded up attempt
    12's mid-failure footprint. The trainer's measured peak is 39.79 GiB."""
    import json

    sys.path.insert(0, str(REPO / "scripts/pod"))
    from stages.phase_a import autoinit_phase_a_driver as drv

    basis = json.loads(
        (REPO / "logs/stages/stage-1/recovery_continuation/analyses/autoinit_recovery_trainer_memory_basis.json").read_text())
    terms = basis["conversion_to_device_bytes"]["terms_gib"]
    assert drv.RECOVERY_TRAINER_PEAK_ALLOCATED_GIB == terms["peak_allocated"]
    assert drv.RECOVERY_TRAINER_RESERVED_SLACK_GIB == terms["allocator_reserved_slack"]
    assert drv.RECOVERY_TRAINER_NON_TORCH_GIB == terms["non_pytorch_overhead"]
    assert drv.RECOVERY_TRAINER_BYTES == basis["conversion_to_device_bytes"]["need_bytes"]
    assert drv.RECOVERY_TRAINER_BYTES > 39 * GIB, (
        "the requirement is below the trainer's measured peak again")
