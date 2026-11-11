"""Random shuffle -> sequential packing -> truncation, as one named protocol.

The protocol (maintainer decision 2026-11-03) used by Stage-1 search forwards and
Stage-3 offline training alike:

1. examples (or turns) are ordered by a stable id, then shuffled with a frozen
   seed (:func:`shuffled_order`);
2. they are packed IN THAT ORDER into fixed-capacity sequences by the
   maintainer's reference algorithm, :func:`packing_conversation`, with
   ``truncate=True``: no best fit, no length sorting or bucketing, no skipping
   ahead to find a shorter example for the remaining room;
3. each packed sequence is read with standard causal attention and continuous
   position ids: a token sees every earlier token of its sequence, other
   examples included. No block-diagonal mask, no per-example position reset.

**What the reference algorithm returns is example INDICES, not token slices.**
When an example is longer than the room left in the current sequence its index
is appended there and NOT advanced, so the same index opens the next sequence
too. :func:`sequential_pack` states the token materialization this module uses,
and it is the one the reference algorithm's own capacity accounting describes:

* every placement is a PREFIX of its example: ``tokens = min(length, room)``,
  where ``room`` is the capacity left when the example is placed;
* a placement that does not advance the index is a *fragment*: the example's
  first ``room`` tokens fill the end of the sequence, and the example is read
  again FROM ITS FIRST TOKEN in the next sequence (the reference charges the
  full length there). A fragment's tokens are therefore all repeated tokens;
* the placement that advances the index is the example's *final* placement and
  holds ``min(length, capacity)`` tokens; anything beyond the capacity is never
  placed (truncated tokens);
* an example has at most one fragment, and a fragment is always the last
  placement of its sequence, so it is context for no other example.

Padding therefore appears only after the last example of the last sequence.
Whether a fragment's supervised positions enter a loss or a score is decided by
the consumer and stated in its own protocol; this module only says which tokens
are where (:func:`plan_report` counts both kinds).

Nothing here knows a model, a tokenizer, a stage, a sequence length or a seed:
they are arguments (AGENTS.md P3).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "PACKING_ID", "MATERIALIZATION_ID", "SHUFFLE_ID", "ATTENTION_ID",
    "PackingError", "Placement", "SequencePackingSpec", "packing_conversation",
    "sequential_pack", "shuffled_order", "plan_report", "plan_indices",
]

#: The packing rule: the reference algorithm, truncate=True.
PACKING_ID = "sequential_truncate_v1"
#: The token materialization: every placement a prefix; a fragment restarts.
MATERIALIZATION_ID = "prefix_restart_v1"
#: The shuffle: a sort by sha256(seed, stable id).
SHUFFLE_ID = "sha256_keyed_v1"
#: How a packed sequence is read: standard causal attention, positions 0..T-1.
ATTENTION_ID = "causal_continuous_positions_v1"


class PackingError(ValueError):
    """A packing that cannot be produced as declared."""


def packing_conversation(numbers: list[int], capacity: int, truncate: bool = True) -> list[list[int]]:

    r"""An efficient packing algorithm with binary search for the conversation dataset."""

    # The maintainer's reference algorithm (2026-11-03), verbatim. `sequential_pack`
    # reproduces its index lists exactly and checks that it does on every call.
    knapsacks = []

    data_idx_index = 0

    while data_idx_index < len(numbers):

        current_knapsack = []

        remaining_capacity = capacity

        while remaining_capacity > 0 and data_idx_index < len(numbers):

            data_length = numbers[data_idx_index]

            if truncate:

                if data_length >= capacity:

                    if remaining_capacity == capacity:

                        current_knapsack.append(data_idx_index)

                        data_idx_index += 1

                        remaining_capacity -= data_length

                    else:

                        current_knapsack.append(data_idx_index)

                        remaining_capacity -= data_length

                elif data_length > remaining_capacity:

                    current_knapsack.append(data_idx_index)

                    remaining_capacity -= data_length

                else:

                    current_knapsack.append(data_idx_index)

                    data_idx_index += 1

                    remaining_capacity -= data_length

            else:

                if data_length > capacity:

                    # No truncate -> skip the data if it is too long

                    data_idx_index += 1

                elif data_length <= remaining_capacity:

                    current_knapsack.append(data_idx_index)

                    data_idx_index += 1

                    remaining_capacity -= data_length

                else:

                    # No truncate -> break when cannot fit more

                    break

        knapsacks.append(current_knapsack)

    return knapsacks


@dataclass(frozen=True)
class Placement:
    """One example's tokens in one packed sequence.

    ``index`` is the example's position in the list that was packed (the
    shuffled order), ``tokens`` how many of its FIRST tokens are placed, and
    ``final`` whether this is the placement that advances past it (``False`` is
    a fragment, read again from token 0 in the next sequence).
    """

    index: int
    tokens: int
    final: bool


def sequential_pack(lengths: Sequence[int], capacity: int) -> list[list[Placement]]:
    """The reference algorithm (``truncate=True``) with its token materialization.

    Same state machine as :func:`packing_conversation`; it also records how many
    tokens each placement holds (module docstring). The index lists are checked
    against the reference on every call, so this can never become a different
    packing than the one the maintainer specified.
    """
    capacity = int(capacity)
    if capacity < 2:
        raise PackingError(f"capacity must be at least 2, got {capacity}")
    numbers = [int(n) for n in lengths]
    bad = [i for i, n in enumerate(numbers) if n < 1]
    if bad:
        raise PackingError(f"examples {bad[:4]} have no tokens")
    sequences: list[list[Placement]] = []
    index = 0
    while index < len(numbers):
        current: list[Placement] = []
        room = capacity
        while room > 0 and index < len(numbers):
            n = numbers[index]
            final = n <= room or room == capacity
            current.append(Placement(index=index, tokens=min(n, room), final=final))
            room -= n
            if final:
                index += 1
        sequences.append(current)
    if plan_indices(sequences) != packing_conversation(numbers, capacity, True):
        raise PackingError("sequential_pack diverged from the reference packing_conversation")
    return sequences


def plan_indices(plan: Sequence[Sequence[Placement]]) -> list[list[int]]:
    """The plan as the reference algorithm's index lists."""
    return [[p.index for p in seq] for seq in plan]


def shuffled_order(stable_ids: Sequence[str], seed: int, *, namespace: str = "") -> list[int]:
    """Positions of ``stable_ids`` in seeded random order.

    The key of an id is ``sha256(f"{SHUFFLE_ID}|{namespace}|{seed}|{id}")``; ties
    (none in practice) break by the id. A pure function of the ids and the seed:
    independent of the input order, the Python version and any RNG state, and an
    id keeps its relative order with every other id when examples are added or
    removed. Duplicate ids are refused: a stable id that names two examples is
    not stable.
    """
    ids = [str(s) for s in stable_ids]
    if len(set(ids)) != len(ids):
        dup = sorted({s for s in ids if ids.count(s) > 1})[:4]
        raise PackingError(f"duplicate stable ids {dup}")

    def key(i: int) -> tuple[str, str]:
        raw = f"{SHUFFLE_ID}|{namespace}|{int(seed)}|{ids[i]}".encode()
        return hashlib.sha256(raw).hexdigest(), ids[i]

    return sorted(range(len(ids)), key=key)


@dataclass(frozen=True)
class SequencePackingSpec:
    """The packing protocol's scientific identity, with its instance values.

    ``sequence_length`` and ``shuffle_seed`` come from the experiment's config;
    the rest names the rules above. Hashed into every identity a packed result
    carries, because the packing decides what context each token is read in.
    """

    sequence_length: int
    shuffle_seed: int
    namespace: str = ""

    def __post_init__(self) -> None:
        for name in ("sequence_length", "shuffle_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise PackingError(f"{name} must be an int, not {type(value).__name__}")
        if self.sequence_length < 2:
            raise PackingError(f"sequence_length must be at least 2, got {self.sequence_length}")

    def as_dict(self) -> dict[str, Any]:
        out = {"packing": PACKING_ID, "truncate": True, "materialization": MATERIALIZATION_ID,
               "shuffle": SHUFFLE_ID, "attention": ATTENTION_ID,
               "sequence_length": int(self.sequence_length), "shuffle_seed": int(self.shuffle_seed)}
        if self.namespace:
            out["namespace"] = self.namespace
        return out

    @property
    def spec_hash(self) -> str:
        import json
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    def plan(self, stable_ids: Sequence[str], lengths: Sequence[int]) -> tuple[list[int], list[list[Placement]]]:
        """``(order, plan)``: the shuffled order of the examples, and the packing of
        that order. A placement's ``index`` points into ``order``, so example
        ``order[p.index]`` of the caller's list is the one placed."""
        if len(stable_ids) != len(lengths):
            raise PackingError(f"{len(stable_ids)} ids for {len(lengths)} lengths")
        order = shuffled_order(stable_ids, self.shuffle_seed, namespace=self.namespace)
        return order, sequential_pack([int(lengths[i]) for i in order], self.sequence_length)


def plan_report(lengths: Sequence[int], plan: Sequence[Sequence[Placement]], capacity: int,
                supervised: Sequence[Sequence[bool]] | None = None) -> dict[str, Any]:
    """Token accounting of a plan, by the materialization rule.

    ``lengths`` are in the packed order (the ``index`` space of the plan).
    ``supervised``, when given, is each example's per-token supervision mask in
    the same order; the supervised counts are then split the same way.

    * ``unique_tokens``: tokens of the examples, each counted once;
    * ``placed_tokens`` = non-padding tokens of the sequences;
    * ``repeated_tokens``: tokens placed more than once (every fragment);
    * ``truncated_tokens``: tokens never placed (beyond the capacity);
    * ``padding_tokens``: sequence slots no example fills (the last sequence only).
    """
    capacity = int(capacity)
    lengths = [int(n) for n in lengths]
    placed = sum(p.tokens for seq in plan for p in seq)
    fragments = [p for seq in plan for p in seq if not p.final]
    finals = [p for seq in plan for p in seq if p.final]
    if sorted(p.index for p in finals) != list(range(len(lengths))):
        raise PackingError("every example must have exactly one final placement")
    repeated = sum(p.tokens for p in fragments)
    truncated = sum(max(0, n - capacity) for n in lengths)
    rows = [sum(p.tokens for p in seq) for seq in plan]
    padding = sum(capacity - r for r in rows)
    out = {
        "packing": PACKING_ID, "materialization": MATERIALIZATION_ID, "sequence_length": capacity,
        "examples": len(lengths), "sequences": len(plan),
        "unique_tokens": sum(lengths), "placed_tokens": placed,
        "repeated_tokens": repeated, "truncated_tokens": truncated,
        "padding_tokens": padding,
        "padding_share": round(padding / (capacity * len(plan)), 6) if plan else 0.0,
        "fragments": len(fragments), "examples_truncated": sum(n > capacity for n in lengths),
        "sequences_with_padding": sum(r < capacity for r in rows),
        "examples_per_sequence_mean": round(sum(len(s) for s in plan) / len(plan), 4) if plan else 0.0,
    }
    if out["unique_tokens"] - out["truncated_tokens"] + out["repeated_tokens"] != placed:
        raise PackingError("token accounting does not close: unique - truncated + repeated != placed")
    if supervised is not None:
        masks = [list(m) for m in supervised]
        if [len(m) for m in masks] != lengths:
            raise PackingError("supervision masks do not match the example lengths")
        unique_sup = sum(sum(m) for m in masks)
        placed_sup = sum(sum(masks[p.index][:p.tokens]) for seq in plan for p in seq)
        repeated_sup = sum(sum(masks[p.index][:p.tokens]) for p in fragments)
        truncated_sup = sum(sum(m[capacity:]) for m in masks)
        out.update(unique_supervised_tokens=unique_sup, placed_supervised_tokens=placed_sup,
                   repeated_supervised_tokens=repeated_sup, truncated_supervised_tokens=truncated_sup)
        if unique_sup - truncated_sup + repeated_sup != placed_sup:
            raise PackingError("supervised accounting does not close")
    return out
