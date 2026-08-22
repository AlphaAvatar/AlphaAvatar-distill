"""Hand the card from Stage 1 to Stage 2, and prove what was freed.

Attempt 12 died six seconds after Stage 1 succeeded: the driver ran the search
in-process and still held 24.05 GiB when Stage 2 spawned the trainer needing
17.97 GiB on a 44.39 GiB card.

**What that 24.05 GiB was is a question this module answers rather than assumes.**
`InitializationState` carries metadata, not CUDA models, and PyTorch's caching
allocator does not return freed blocks to the driver — so a process can hold tens
of GiB *reserved* with almost nothing *allocated*. Those are different diseases.
Calling the second a model leak would send the next session hunting a bug that is
not there, which is why the record distinguishes them and the verdict is derived
from `allocated`, not from `reserved`.

These run on CPU with an injected torch stand-in, because the arithmetic and the
verdict are what need pinning; the CUDA calls themselves need a device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.device_handoff import (  # noqa: E402
    DeviceHandoffError, LIVE_RETENTION_LIMIT_BYTES, complete_release, cuda_memory,
    require_headroom, require_released,
)

GIB = 2 ** 30


class FakeTorch:
    """A torch stand-in whose memory figures move only when they should."""

    def __init__(self, *, allocated, reserved, free, total=44 * GIB,
                 frees_on_empty_cache=True):
        self.state = {"allocated": allocated, "reserved": reserved, "free": free,
                      "total": total}
        self._frees = frees_on_empty_cache
        self.empty_cache_calls = 0

        outer = self

        class _cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_allocated(_d=0):
                return outer.state["allocated"]

            @staticmethod
            def memory_reserved(_d=0):
                return outer.state["reserved"]

            @staticmethod
            def mem_get_info(_d=0):
                return outer.state["free"], outer.state["total"]

            @staticmethod
            def empty_cache():
                outer.empty_cache_calls += 1
                if not outer._frees:
                    return
                # Only the CACHED part can be returned: reserved falls to
                # allocated, and the driver sees that much more free.
                cached = outer.state["reserved"] - outer.state["allocated"]
                outer.state["reserved"] -= cached
                outer.state["free"] += cached

        self.cuda = _cuda


def test_allocator_reservation_is_not_called_a_leak():
    """Attempt 12's shape, if the 24 GiB was cached blocks: 1 GiB live, 24 GiB
    reserved. After the release the card is usable and nothing was leaking."""
    torch = FakeTorch(allocated=1 * GIB, reserved=24 * GIB, free=20 * GIB)
    before = cuda_memory(torch_mod=torch)
    rec = complete_release(before, torch_mod=torch)

    assert torch.empty_cache_calls == 1
    assert rec["live_retention"] is False
    assert "allocator reservation" in rec["verdict"]
    assert "leak" in rec["verdict"]           # names it, to deny it
    assert rec["freed_reserved_bytes"] == 23 * GIB
    assert rec["gained_free_bytes"] == 23 * GIB
    assert rec["after"]["free_bytes"] == 43 * GIB


def test_a_genuine_live_retention_is_named_as_one():
    """The other disease: memory still ALLOCATED after the release. Here
    `empty_cache` can return nothing because nothing is merely cached."""
    torch = FakeTorch(allocated=24 * GIB, reserved=24 * GIB, free=20 * GIB)
    before = cuda_memory(torch_mod=torch)
    rec = complete_release(before, torch_mod=torch)

    assert rec["live_retention"] is True
    assert "still ALLOCATED" in rec["verdict"]
    assert rec["freed_reserved_bytes"] == 0
    assert rec["after"]["allocated_bytes"] == 24 * GIB


def test_the_verdict_reads_allocated_not_reserved():
    """A large `reserved` with a small `allocated` must NOT read as retention —
    the misdiagnosis this module exists to prevent.

    `frees_on_empty_cache=False` is what makes this test able to fail: when the
    allocator DOES return everything, `reserved` collapses to `allocated` and
    reading either gives the same verdict, so the two are indistinguishable. An
    allocator that keeps its blocks separates them — 64 MiB live against 30 GiB
    reserved. Reading `reserved` would call that a 30 GiB leak.
    """
    torch = FakeTorch(allocated=64 * 2**20, reserved=30 * GIB, free=14 * GIB,
                      frees_on_empty_cache=False)
    before = cuda_memory(torch_mod=torch)
    rec = complete_release(before, torch_mod=torch)

    assert rec["after"]["reserved_bytes"] == 30 * GIB      # nothing returned
    assert rec["after"]["allocated_bytes"] == 64 * 2**20   # and almost nothing live
    assert rec["live_retention"] is False, (
        "a caching allocator holding 30 GiB with 64 MiB live was called a leak")
    assert "allocator reservation" in rec["verdict"]


def test_the_record_carries_both_sides_and_the_deltas():
    torch = FakeTorch(allocated=2 * GIB, reserved=20 * GIB, free=24 * GIB)
    before = cuda_memory(torch_mod=torch)
    rec = complete_release(before, torch_mod=torch)
    for side in ("before", "after"):
        for key in ("allocated_bytes", "reserved_bytes", "free_bytes", "total_bytes"):
            assert key in rec[side], f"{side}.{key} missing"
    assert rec["before"]["reserved_bytes"] == 20 * GIB
    assert rec["after"]["reserved_bytes"] == 2 * GIB


def test_off_cuda_it_records_why_it_measured_nothing():
    class NoCuda:
        class cuda:
            @staticmethod
            def is_available():
                return False
    before = cuda_memory(torch_mod=NoCuda)
    rec = complete_release(before, torch_mod=NoCuda)
    assert rec["before"]["available"] is False
    assert "no CUDA device" in rec["verdict"]
    assert cuda_memory(torch_mod=NoCuda)["available"] is False


# --- the headroom contract --------------------------------------------------

def test_headroom_is_checked_against_the_drivers_free_bytes():
    """A sibling process cannot use the parent's cached blocks, so `reserved`
    falling means nothing unless driver-visible `free` rises with it."""
    plenty = {"available": True, "free_bytes": 30 * GIB, "total_bytes": 44 * GIB,
              "allocated_bytes": 0, "reserved_bytes": 40 * GIB}
    require_headroom(plenty, need_bytes=22 * GIB, what="the trainer")

    starved = {**plenty, "free_bytes": 2 * GIB, "reserved_bytes": 1 * GIB}
    with pytest.raises(DeviceHandoffError, match="refusing to start"):
        require_headroom(starved, need_bytes=22 * GIB, what="the trainer")


def test_the_headroom_check_uses_a_margin():
    snap = {"available": True, "free_bytes": 22 * GIB, "total_bytes": 44 * GIB,
            "allocated_bytes": 0, "reserved_bytes": 0}
    with pytest.raises(DeviceHandoffError):
        require_headroom(snap, need_bytes=22 * GIB, what="x",
                         margin_bytes=2 * GIB)
    require_headroom({**snap, "free_bytes": 25 * GIB}, need_bytes=22 * GIB,
                     what="x", margin_bytes=2 * GIB)


def test_attempt_12s_actual_numbers_would_be_refused():
    """The regression, in the figures the pod reported."""
    attempt12 = {"available": True, "free_bytes": int(2.36 * GIB),
                 "total_bytes": int(44.39 * GIB), "allocated_bytes": 0,
                 "reserved_bytes": 0}
    with pytest.raises(DeviceHandoffError) as exc:
        require_headroom(attempt12, need_bytes=22 * GIB,
                         what="the stage-2 recovery trainer")
    assert "2.36 GiB free" in str(exc.value)


def test_off_cuda_headroom_is_not_asserted():
    """A CPU rehearsal must not fail a GPU contract it cannot evaluate."""
    require_headroom({"available": False}, need_bytes=1 << 40, what="x")


# --- the wiring -------------------------------------------------------------

def test_the_driver_hands_off_after_durability_and_before_stage_2():
    """Order matters: if the release or the contract fails, the five leaves must
    already be off the pod. The other order trades a completed 203-minute search
    for a memory diagnostic."""
    src = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    stage1 = src[src.index("def stage1(self)"):src.index("def probe_config")]

    persist = stage1.index("persist_selected_leaves(")
    snapshot = stage1.index("before = cuda_memory()")
    dropped = stage1.index("del found")
    handoff = stage1.index("complete_release(")
    released = stage1.index("require_released(")
    headroom = stage1.index("require_headroom(")
    done = stage1.index("return self.record(1, True")
    assert persist < snapshot < dropped < handoff < released < headroom < done, (
        "the handoff must sit between durability and stage 1 reporting success, "
        "and the caller must snapshot, then drop, then measure, then gate")
    assert "release_to_subprocess" not in src, (
        "the API that could not release the caller's references is back")
    assert "except DeviceHandoffError" in stage1
    assert "self.record(\n                1, False" in stage1, (
        "a failed handoff must fail stage 1 closed, not warn")


def test_nothing_reads_the_search_object_after_the_release():
    """The release is the point: nothing may survive it holding the search.

    The first version of this block dropped `found` and then read
    `found.summary` four lines later — a NameError after a 203-minute search had
    already succeeded, and one the fast tests could not see because only the
    real orchestration reaches that line.
    """
    import ast

    src = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "stage1")
    body = ast.unparse(fn)
    # Anchored on `del found`, which is now where the object actually dies. It
    # used to be anchored on the release call, back when the release claimed to
    # drop the caller's reference for it and could not.
    after = body[body.index("del found"):]
    reads = [ln for ln in after.splitlines() if "found." in ln]
    assert not reads, f"`found` is read after the release: {reads}"


# --- the attempt-4 regressions ----------------------------------------------
#
# Recovery continuation attempt 4 reported `freed_allocated_bytes: 0`, read its
# own verdict saying 7.55 GiB was genuinely retained, started the trainer anyway
# and lost the probe to a CUDA OOM. Two defects, one per section below.

def test_the_release_helper_accepts_no_objects_it_cannot_release():
    """Defect 1, at the signature.

    `release_to_subprocess(drop=[teacher, evaluator])` copied the sequence into
    a local list and cleared *that*. A callee cannot rebind `teacher` in the
    caller's frame, so nothing was ever released — and because the signature
    promised otherwise, both drivers trusted it.
    """
    import inspect

    from aadistill.autoinit import device_handoff

    assert not hasattr(device_handoff, "release_to_subprocess"), (
        "the API that claimed to drop the caller's references is back")
    params = inspect.signature(complete_release).parameters
    assert "drop" not in params, (
        "complete_release takes objects again; a callee cannot release a "
        "caller's names and must not offer to")
    assert "before" in params, "the caller must supply the BEFORE snapshot"


def test_the_after_measurement_follows_the_callers_del():
    """Defect 1, at the arithmetic.

    The old helper measured `before` AND `after` inside itself, so `after` was
    taken before the caller's `del` ran and a real release could not register.
    Here the caller frees between the two calls, exactly as the drivers do, and
    the record must show it.
    """
    torch = FakeTorch(allocated=8 * GIB, reserved=12 * GIB, free=32 * GIB)
    before = cuda_memory(torch_mod=torch)
    # The caller drops its device-owning objects. Nothing else changes.
    torch.state["allocated"] = 0
    torch.state["reserved"] = 8 * GIB
    rec = complete_release(before, torch_mod=torch)

    assert rec["freed_allocated_bytes"] == 8 * GIB, (
        "a release performed by the caller between the two measurements did "
        "not register — this is attempt 4's freed_allocated_bytes: 0")
    assert rec["live_retention"] is False


def test_a_live_retention_refuses_before_the_trainer_starts():
    """Defect 2: the verdict was recorded and never enforced."""
    torch = FakeTorch(allocated=8 * GIB, reserved=8 * GIB, free=36 * GIB)
    before = cuda_memory(torch_mod=torch)
    rec = complete_release(before, torch_mod=torch)          # caller freed nothing
    assert rec["live_retention"] is True

    with pytest.raises(DeviceHandoffError) as exc:
        require_released(rec, what="the stage-2 recovery trainer")
    assert "still ALLOCATED" in str(exc.value)

    # And it refuses INDEPENDENTLY of free_bytes: 36 GiB free is plenty by the
    # old gate's reckoning, which is precisely how attempt 4 got through.
    require_headroom(rec["after"], need_bytes=22 * GIB, what="x")


def test_a_clean_release_is_not_refused():
    torch = FakeTorch(allocated=0, reserved=2 * GIB, free=42 * GIB)
    before = cuda_memory(torch_mod=torch)
    rec = complete_release(before, torch_mod=torch)
    assert rec["live_retention"] is False
    require_released(rec, what="the stage-2 recovery trainer")   # must not raise


def test_off_cuda_the_retention_check_is_not_asserted():
    class NoCuda:
        class cuda:
            @staticmethod
            def is_available():
                return False
    before = cuda_memory(torch_mod=NoCuda)
    rec = complete_release(before, torch_mod=NoCuda)
    require_released(rec, what="x")            # must not raise on a CPU rehearsal


def test_the_retention_limit_is_the_one_the_verdict_uses():
    """The threshold is named once. A gate reading a different number from the
    verdict that feeds it would refuse runs the record calls healthy."""
    torch = FakeTorch(allocated=LIVE_RETENTION_LIMIT_BYTES + 1,
                      reserved=LIVE_RETENTION_LIMIT_BYTES + 1, free=40 * GIB)
    before = cuda_memory(torch_mod=torch)
    over = complete_release(before, torch_mod=torch)
    assert over["live_retention"] is True
    with pytest.raises(DeviceHandoffError):
        require_released(over, what="x")

    torch = FakeTorch(allocated=LIVE_RETENTION_LIMIT_BYTES,
                      reserved=LIVE_RETENTION_LIMIT_BYTES, free=40 * GIB)
    before = cuda_memory(torch_mod=torch)
    at = complete_release(before, torch_mod=torch)
    assert at["live_retention"] is False
    require_released(at, what="x")


# --- the trainer requirement is measured, not chosen ------------------------

def test_the_trainer_requirement_matches_its_recorded_basis():
    """RECOVERY_TRAINER_BYTES was 22 GiB because somebody rounded up attempt
    12's mid-failure footprint. The trainer's measured peak is 39.79 GiB."""
    import json

    sys.path.insert(0, str(REPO / "scripts/pod"))
    import autoinit_phase_a_driver as drv

    basis = json.loads(
        (REPO / "logs/autoinit_recovery_trainer_memory_basis.json").read_text())
    terms = basis["conversion_to_device_bytes"]["terms_gib"]
    assert drv.RECOVERY_TRAINER_PEAK_ALLOCATED_GIB == terms["peak_allocated"]
    assert drv.RECOVERY_TRAINER_RESERVED_SLACK_GIB == terms["allocator_reserved_slack"]
    assert drv.RECOVERY_TRAINER_NON_TORCH_GIB == terms["non_pytorch_overhead"]
    assert drv.RECOVERY_TRAINER_BYTES == basis["conversion_to_device_bytes"]["need_bytes"]
    assert drv.RECOVERY_TRAINER_BYTES > 39 * GIB, (
        "the requirement is below the trainer's measured peak again")


def test_attempt_4s_free_bytes_would_now_be_refused():
    """The whole point. Attempt 4 saw 36.32 GiB free and started the trainer."""
    sys.path.insert(0, str(REPO / "scripts/pod"))
    import autoinit_phase_a_driver as drv

    attempt_4_after = {"available": True, "allocated_bytes": 8110229504,
                       "reserved_bytes": 8124366848,
                       "free_bytes": 38997983232, "total_bytes": 47665709056}
    with pytest.raises(DeviceHandoffError) as exc:
        require_headroom(attempt_4_after,
                         need_bytes=drv.RECOVERY_TRAINER_BYTES,
                         what="the stage-2 recovery trainer")
    assert "36.32 GiB free" in str(exc.value)


def test_a_released_card_clears_the_new_requirement():
    """And it must not refuse the run it is meant to allow: with the teacher and
    evaluator actually gone, attempt 4's card had room."""
    sys.path.insert(0, str(REPO / "scripts/pod"))
    import autoinit_phase_a_driver as drv

    released = {"available": True, "allocated_bytes": 0, "reserved_bytes": 0,
                # 36.32 GiB free + the 7.55 GiB the driver was holding.
                "free_bytes": 38997983232 + 8110229504,
                "total_bytes": 47665709056}
    require_headroom(released, need_bytes=drv.RECOVERY_TRAINER_BYTES,
                     what="the stage-2 recovery trainer")
