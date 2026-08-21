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
    DeviceHandoffError, cuda_memory, release_to_subprocess, require_headroom,
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
    rec = release_to_subprocess(drop=[object()], torch_mod=torch)

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
    rec = release_to_subprocess(drop=[], torch_mod=torch)

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
    rec = release_to_subprocess(drop=[], torch_mod=torch)

    assert rec["after"]["reserved_bytes"] == 30 * GIB      # nothing returned
    assert rec["after"]["allocated_bytes"] == 64 * 2**20   # and almost nothing live
    assert rec["live_retention"] is False, (
        "a caching allocator holding 30 GiB with 64 MiB live was called a leak")
    assert "allocator reservation" in rec["verdict"]


def test_the_record_carries_both_sides_and_the_deltas():
    torch = FakeTorch(allocated=2 * GIB, reserved=20 * GIB, free=24 * GIB)
    rec = release_to_subprocess(drop=[], torch_mod=torch)
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
    rec = release_to_subprocess(drop=[], torch_mod=NoCuda)
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
    handoff = stage1.index("release_to_subprocess(")
    headroom = stage1.index("require_headroom(")
    done = stage1.index("return self.record(1, True")
    assert persist < handoff < headroom < done, (
        "the handoff must sit between durability and stage 1 reporting success")
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
    after = body[body.index("release_to_subprocess("):]
    reads = [ln for ln in after.splitlines() if "found." in ln]
    assert not reads, f"`found` is read after the release: {reads}"
