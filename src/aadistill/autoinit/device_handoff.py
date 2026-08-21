"""Hand the accelerator from one stage to the next, and prove it happened.

Phase-A attempt 12 died six seconds after Stage 1 succeeded:

    Tried to allocate 3.58 GiB. GPU 0 has 44.39 GiB of which 2.36 GiB is free.
    Process 6820 has 24.05 GiB in use … this process has 17.97 GiB in use.

The driver runs the beam search **in-process** — deliberately, because the rungs
need live `InitializationState` objects — and then spawns `train_stage3.py` as a
subprocess. Two processes, one card, and 44.39 GiB does not fit both.

**What that 24.05 GiB was is a question, not a finding.** `InitializationState`
carries metadata and checkpoint identities, not CUDA models, so the states the
driver keeps should cost almost nothing. The search's teacher, its primed
`StateEvaluator` and the control model it loads are function-local and become
garbage when `run_phase_a_search` returns — but PyTorch's caching allocator does
not hand freed blocks back to the driver, so a process can hold tens of GiB of
**reserved** memory with almost nothing **allocated**. Those are different
diseases with different cures, and calling the second one a leak would send the
next session hunting a bug that is not there.

So this module measures both, on both sides of the release:

* `allocated` — live tensors. If this stays high, something is genuinely held.
* `reserved` — allocator-owned blocks, live or cached. High reserved with low
  allocated is a caching allocator doing its job, and `empty_cache()` fixes it.
* `free` / `total` — what the driver reports, which is what a *subprocess* sees.

The last one is the one that matters for the handoff: a sibling process cannot
use the parent's cached blocks, so `torch.cuda.memory_reserved()` falling is
worth nothing unless `cuda.mem_get_info()` free rises with it.

Nothing here decides policy. `require_headroom` refuses, the caller decides what
to do about it, and the numbers land in the session record either way.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Sequence

__all__ = [
    "DeviceHandoffError",
    "cuda_memory",
    "release_to_subprocess",
    "require_headroom",
]


class DeviceHandoffError(RuntimeError):
    """The accelerator could not be handed to the next stage."""


def cuda_memory(device: int | str = 0, *, torch_mod: Any = None) -> dict:
    """Allocated, reserved and driver-visible free bytes.

    Returns `{"available": False}` off CUDA rather than raising, so the boundary
    is callable from a CPU rehearsal and the record says why it is empty.
    """
    torch = torch_mod
    if torch is None:
        import torch as torch  # noqa: PLC0414

    if not torch.cuda.is_available():
        return {"available": False,
                "note": "no CUDA device; nothing to hand over"}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "available": True,
        # LIVE tensors. If this stays high after a release, something is held.
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        # Allocator-owned blocks, live or cached. High here with low `allocated`
        # is a caching allocator, not a leak.
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        # What the DRIVER reports, and therefore what a sibling process sees.
        # This is the number the handoff actually turns on.
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def release_to_subprocess(
    *,
    drop: Sequence[Any] = (),
    device: int | str = 0,
    torch_mod: Any = None,
    collect: Callable[[], Any] = gc.collect,
) -> dict:
    """Release the stage's device state and measure what that actually freed.

    `drop` is the caller's references. They are cleared here rather than left to
    fall out of scope, because a closure captured by a returned object keeps its
    captures alive for as long as the object lives — which for a driver that
    holds the search result is the rest of the session.

    Returns a record with `before`, `after` and the deltas, and — the point of
    the exercise — a `verdict` that distinguishes a genuine live-allocation hold
    from allocator reservation.
    """
    torch = torch_mod
    if torch is None:
        import torch as torch  # noqa: PLC0414

    before = cuda_memory(device, torch_mod=torch)

    # Clear the caller's references first: dropping the list is not enough if the
    # caller still holds them, which is why they are passed in explicitly.
    refs = list(drop)
    refs.clear()
    del drop
    collected = collect()

    if before.get("available"):
        torch.cuda.empty_cache()
    after = cuda_memory(device, torch_mod=torch)

    record: dict[str, Any] = {
        "schema": "aadistill.autoinit.device_handoff/v1",
        "before": before, "after": after, "gc_collected": collected,
    }
    if not before.get("available"):
        record["verdict"] = "no CUDA device; nothing measured"
        return record

    record["freed_reserved_bytes"] = (before["reserved_bytes"]
                                      - after["reserved_bytes"])
    record["freed_allocated_bytes"] = (before["allocated_bytes"]
                                       - after["allocated_bytes"])
    record["gained_free_bytes"] = after["free_bytes"] - before["free_bytes"]

    still_live = after["allocated_bytes"]
    if still_live > 1 << 30:
        record["verdict"] = (
            f"{still_live / 2**30:.2f} GiB is still ALLOCATED after the release, "
            "so something holds live tensors — this is a genuine retention, not "
            "allocator caching")
        record["live_retention"] = True
    else:
        record["verdict"] = (
            f"{after['allocated_bytes'] / 2**30:.2f} GiB allocated and "
            f"{after['reserved_bytes'] / 2**30:.2f} GiB reserved after the "
            "release; the pre-release figure was allocator reservation, not a "
            "model leak")
        record["live_retention"] = False
    return record


def require_headroom(snapshot: dict, *, need_bytes: int, what: str,
                     margin_bytes: int = 2 * 2**30) -> None:
    """Refuse to start `what` unless the driver reports room for it.

    Checked against `free_bytes` — the driver's number — not against
    `reserved`/`allocated`, because a subprocess cannot use the parent's cached
    blocks however cheap they look from inside the parent.
    """
    if not snapshot.get("available"):
        return
    free = snapshot["free_bytes"]
    if free < need_bytes + margin_bytes:
        raise DeviceHandoffError(
            f"refusing to start {what}: it needs {need_bytes / 2**30:.2f} GiB "
            f"(plus a {margin_bytes / 2**30:.2f} GiB margin) and the driver "
            f"reports {free / 2**30:.2f} GiB free of "
            f"{snapshot['total_bytes'] / 2**30:.2f}. Attempt 12 spent 203.8 min "
            "on a successful search and then lost the probe here, because the "
            "parent still held the card. Failing before the subprocess starts "
            "makes that a stage-1 result plus a diagnosis, not a crash.")
