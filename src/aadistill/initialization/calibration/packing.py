"""Which items share a forward pass, as a named deterministic policy.

`micro_batches` groups consecutive items in the mixture's own order and states
plainly that it never sorts by length. That is the right default and it is not
changed here: two runs of the same mixture must batch identically, and a
grouping that depended on content would drift after an unrelated edit.

But "never sorts by length" is also a cost. Right-padding a group to its
longest member means every shorter member pays for the difference, and on the
frozen 67-item mixture consecutive B4 padding measured 36.73% more
token-positions than B1 computes — enough that a quarter of the kernel
launches still came out 1.46x slower.

So the policy becomes a NAMED CHOICE rather than an assumption. Each policy is
a pure function from item lengths to groups of original indices: no
randomness, no device, no model family, no tokenizer. The indices are what the
caller gets back, because a packing that reorders execution must not be
allowed to reorder anything else.

**A packing policy is identity-bearing for a causal scorer.** Batch
composition is already measured to move causal scores and head maps, so an
operator whose result depends on it must carry the policy name in its hashed
config. This module names the policies; it does not decide who hashes them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aadistill.initialization.calibration.batching import (
    BatchingError, ItemBatch, build_batch,
)

#: The default, and what `micro_batches` has always done.
ORIGINAL_ORDER_V1 = "original_order_v1"
#: Shortest first, ties by original index.
LENGTH_SORTED_V1 = "length_sorted_v1"

PACKING_POLICIES = (ORIGINAL_ORDER_V1, LENGTH_SORTED_V1)


class PackingError(BatchingError):
    """The requested packing cannot be produced."""


@dataclass(frozen=True)
class PackedBatch:
    """One forward's worth of items, and where each row came from.

    `original_indices[r]` is the position of row `r` in the mixture as it was
    frozen. Every consumer that aggregates per-item results MUST use it: under
    a reordering policy the row order is not the mixture order, and a consumer
    that assumes otherwise would attribute one item's value to another.
    """

    batch: ItemBatch
    original_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.original_indices) != self.batch.size:
            raise PackingError(
                f"{len(self.original_indices)} indices for a batch of "
                f"{self.batch.size} rows")


def item_lengths(items: Sequence[Mapping[str, Any]]) -> list[int]:
    """Token count per item, from the prepared `input_ids`."""
    out = []
    for i, item in enumerate(items):
        ids = item.get("input_ids")
        if ids is None:
            raise PackingError(f"item {i} has no input_ids to measure")
        out.append(int(ids.shape[-1]))
    return out


def pack(lengths: Sequence[int], batch_size: int, *,
         packing: str = ORIGINAL_ORDER_V1) -> list[tuple[int, ...]]:
    """Groups of ORIGINAL INDICES, deterministically.

    Pure integer work: no tensors, no device, no model. That is deliberate —
    the whole packing question can then be answered at `$0` from the frozen
    mixture's lengths, which is how the padding cost of a protocol is known
    before a pod exists rather than after.

    `length_sorted_v1` sorts by `(length ascending, original index ascending)`
    and splits contiguously. Ascending rather than descending: with a short
    final partial group, descending leaves the largest remaining item grouped
    with much shorter ones, which is the case that pads worst. The tie-break
    on original index is what makes it a function of the mixture rather than
    of Python's sort stability.
    """
    if batch_size < 1:
        raise PackingError(f"batch_size must be >= 1, got {batch_size}")
    if packing not in PACKING_POLICIES:
        raise PackingError(
            f"unknown packing {packing!r}; known: {list(PACKING_POLICIES)}")
    n = len(lengths)
    if packing == ORIGINAL_ORDER_V1:
        order = list(range(n))
    else:
        order = sorted(range(n), key=lambda i: (int(lengths[i]), i))
    return [tuple(order[s:s + batch_size])
            for s in range(0, n, batch_size)]


def packed_batches(items: Sequence[Mapping[str, Any]], batch_size: int, *,
                   packing: str = ORIGINAL_ORDER_V1, pad_id: int,
                   device: Any = None) -> Iterator[PackedBatch]:
    """`pack` applied to real items, built through the shared `build_batch`.

    At `original_order_v1` this yields exactly what `micro_batches` yields,
    row for row — asserted by a test rather than by inspection, because the
    reference protocol must not become a second implementation of itself.
    """
    items = list(items)
    lengths = item_lengths(items)
    for group in pack(lengths, batch_size, packing=packing):
        yield PackedBatch(
            batch=build_batch([items[i] for i in group], pad_id=pad_id,
                              device=device),
            original_indices=group)


def padding_profile(lengths: Sequence[int], batch_size: int, *,
                    packing: str = ORIGINAL_ORDER_V1) -> dict[str, Any]:
    """What a protocol costs in positions, before anything is executed.

    Derived from the real lengths, so a protocol's padding overhead is a `$0`
    fact. `padded_positions` counts positions no item owns: the group's width
    times its size, less the tokens actually in it.
    """
    groups = pack(lengths, batch_size, packing=packing)
    widths = [max(int(lengths[i]) for i in g) for g in groups]
    valid = sum(int(x) for x in lengths)
    padded = sum(w * len(g) - sum(int(lengths[i]) for i in g)
                 for w, g in zip(widths, groups))
    return {
        "packing": packing, "batch_size": batch_size,
        "n_groups": len(groups),
        "group_sizes": [len(g) for g in groups],
        "group_max_lengths": widths,
        "valid_positions": valid,
        "padded_positions": padded,
        "padding_over_valid": (padded / valid) if valid else 0.0,
        "max_group_width": max(widths) if widths else 0,
        "executed_positions": valid + padded,
    }
