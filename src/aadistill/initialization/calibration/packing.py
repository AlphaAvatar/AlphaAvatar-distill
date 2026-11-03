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

**A forward context is a different thing, and it is science.** The two
policies above keep every item in its own row with its own positions, so an
item is read exactly as it would be alone. A :class:`ForwardContext` packs
several items into each row and reads the row with standard causal attention
and continuous positions (maintainer decision 2026-11-03: random shuffle ->
sequential packing -> truncation, `aadistill.data.sequence_packing`), so a
later item is conditioned on the earlier ones. That changes what every
activation statistic, KL and state metric measures, so a forward context is
carried as a scientific term (hashed operator config, measurement protocol id)
and never as an execution knob. Under it, ``micro_batch_size`` is the number
of PACKED SEQUENCES per forward and the grouping above does not apply.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aadistill.data.sequence_packing import (
    Placement, SequencePackingSpec, plan_report,
)
from aadistill.initialization.calibration.batching import (
    BatchingError, ItemBatch, build_batch, build_packed_batch,
)

#: The default, and what `micro_batches` has always done.
ORIGINAL_ORDER_V1 = "original_order_v1"
#: Shortest first, ties by original index.
LENGTH_SORTED_V1 = "length_sorted_v1"

PACKING_POLICIES = (ORIGINAL_ORDER_V1, LENGTH_SORTED_V1)


class PackingError(BatchingError):
    """The requested packing cannot be produced."""


@dataclass(frozen=True)
class ForwardContext:
    """How calibration and state-evaluation items are composed into forwards.

    A scientific term: the sequence length and shuffle seed are the
    experiment's (P3), the rules are `aadistill.data.sequence_packing`'s. The
    plan is a pure function of the items' stable ids and lengths, so one profile
    (or suite) gets ONE packing for every candidate a search evaluates, and the
    reference and the candidate are always read in the same context.
    """

    sequence_length: int
    shuffle_seed: int

    def __post_init__(self) -> None:
        self.spec  # validates both values

    @property
    def spec(self) -> SequencePackingSpec:
        return SequencePackingSpec(sequence_length=self.sequence_length,
                                   shuffle_seed=self.shuffle_seed)

    def as_dict(self) -> dict[str, Any]:
        return self.spec.as_dict()

    @property
    def context_hash(self) -> str:
        return self.spec.spec_hash

    def plan(self, items: Sequence[Mapping[str, Any]]) -> list[list[Placement]]:
        """The packed sequences, with every placement's ``index`` pointing into
        ``items`` (the caller's order), not into the shuffled order."""
        ids = []
        for i, item in enumerate(items):
            item_id = item.get("item_id") if isinstance(item, Mapping) else getattr(item, "item_id", None)
            if item_id is None:
                raise PackingError(
                    f"item {i} has no item_id; a forward context shuffles by "
                    "stable id and an item without one cannot be placed "
                    "reproducibly")
            ids.append(str(item_id))
        order, plan = self.spec.plan(ids, item_lengths(items))
        return [[Placement(index=order[p.index], tokens=p.tokens, final=p.final)
                 for p in seq] for seq in plan]

    def report(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The plan's token accounting (`sequence_packing.plan_report`)."""
        lengths = item_lengths(items)
        plan = self.plan(items)
        return plan_report(lengths, plan, self.sequence_length)

    def packing_identity(self, items: Sequence[Mapping[str, Any]]) -> str:
        """sha256 of the spec and the plan over these items: which tokens of
        which item sit where. Two profiles under one context differ here."""
        import hashlib
        import json
        plan = [[[str(_item_id(items[p.index])), p.tokens, p.final] for p in seq]
                for seq in self.plan(items)]
        return hashlib.sha256(json.dumps({"spec": self.as_dict(), "plan": plan},
                                         sort_keys=True).encode()).hexdigest()


def _item_id(item: Any) -> Any:
    return item.get("item_id") if isinstance(item, Mapping) else getattr(item, "item_id", None)


def forward_context_from(raw: Mapping[str, Any] | None) -> ForwardContext | None:
    """A :class:`ForwardContext` from its ``as_dict()`` form, or ``None``.

    Refuses a dict naming any rule other than the one this code implements, so a
    record cannot claim a packing it was not read under.
    """
    if raw is None:
        return None
    fc = ForwardContext(sequence_length=int(raw["sequence_length"]),
                        shuffle_seed=int(raw["shuffle_seed"]))
    if dict(raw) != fc.as_dict():
        raise PackingError(
            f"forward context {dict(raw)!r} is not the one this code implements "
            f"({fc.as_dict()!r})")
    return fc


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
        if self.batch.segments is not None:
            #: Packed: one index PER SEGMENT, in segment order.
            if tuple(self.original_indices) != tuple(s.index for s in self.batch.segments):
                raise PackingError(
                    "a packed batch's original_indices must be its segments' "
                    "indices, in segment order")
            return
        if len(self.original_indices) != self.batch.size:
            raise PackingError(
                f"{len(self.original_indices)} indices for a batch of "
                f"{self.batch.size} rows")


def item_lengths(items: Sequence[Mapping[str, Any]]) -> list[int]:
    """Token count per item, from the prepared `input_ids`."""
    out = []
    for i, item in enumerate(items):
        ids = (item.get("input_ids") if isinstance(item, Mapping)
               else getattr(item, "input_ids", None))
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
                   device: Any = None,
                   forward_context: ForwardContext | None = None,
                   ) -> Iterator[PackedBatch]:
    """`pack` applied to real items, built through the shared `build_batch`.

    At `original_order_v1` this yields exactly what `micro_batches` yields,
    row for row — asserted by a test rather than by inspection, because the
    reference protocol must not become a second implementation of itself.

    Under a ``forward_context`` it yields packed batches instead:
    ``batch_size`` consecutive packed sequences of the context's plan per
    batch, ``original_indices`` one per segment. ``packing`` must then be the
    default, because the sequences already fix which items share a forward.
    """
    items = list(items)
    if forward_context is not None:
        if packing != ORIGINAL_ORDER_V1:
            raise PackingError(
                f"packing={packing!r} under a forward context: the packed "
                "sequences fix which items share a forward, so no second "
                "grouping policy may apply")
        if batch_size < 1:
            raise PackingError(f"batch_size must be >= 1, got {batch_size}")
        plan = forward_context.plan(items)
        for start in range(0, len(plan), batch_size):
            batch = build_packed_batch(plan[start:start + batch_size], items,
                                       pad_id=pad_id, device=device)
            yield PackedBatch(batch=batch,
                              original_indices=tuple(s.index for s in batch.segments))
        return
    lengths = item_lengths(items)
    for group in pack(lengths, batch_size, packing=packing):
        yield PackedBatch(
            batch=build_batch([items[i] for i in group], pad_id=pad_id,
                              device=device),
            original_indices=group)


def forward_context_profile(items: Sequence[Mapping[str, Any]], batch_size: int,
                            forward_context: ForwardContext) -> dict[str, Any]:
    """`padding_profile`'s figures for a forward context, at `$0` from lengths:
    the plan's token accounting plus the widest batch the forwards will hold."""
    report = forward_context.report(items)
    plan = forward_context.plan(items)
    rows = [sum(p.tokens for p in seq) for seq in plan]
    groups = [rows[s:s + batch_size] for s in range(0, len(rows), batch_size)]
    widths = [max(g) for g in groups]
    executed = sum(w * len(g) for w, g in zip(widths, groups))
    return {**report, "forward_context": forward_context.as_dict(),
            "batch_size": int(batch_size), "n_groups": len(groups),
            "group_sizes": [len(g) for g in groups],
            "group_max_lengths": widths,
            "valid_positions": report["placed_tokens"],
            "padded_positions": executed - report["placed_tokens"],
            "max_group_width": max(widths) if widths else 0,
            "executed_positions": executed}


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
