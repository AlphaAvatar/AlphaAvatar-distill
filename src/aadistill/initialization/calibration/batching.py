"""Micro-batching for calibration-item forwards.

An operator that measures something per calibration item used to issue one model
forward per item. That is correct and it is slow: the frozen
``calib.domain_balanced@v1`` mixture is 67 items, so a single
``depth.causal_kl_greedy_v1`` candidate subset costs 67 forwards, and an
expansion costs 260 of those. The forward is the whole cost, and a 67-item
mixture at sequence lengths around a thousand tokens does not fill an
accelerator one item at a time.

This module is the one place that turns a sequence of items into padded batches
and gives back enough information to take each item's own results out again.

**What batching must not change.** Every operator here measures *per item* and
then aggregates — DEPTH takes a per-item KL, then a subtype mean, then a
domain-balanced score; the statistics collectors sum over each item's real
tokens. Padding is not data. So a batch carries an explicit ``attention_mask``
and the true ``lengths``, and every consumer reduces over valid positions only.
A padded position must never reach a KL, a second moment, a token count or a
domain mean. The estimand is unchanged; only the order in which the accelerator
performs the arithmetic moves.

**Right padding, deliberately.** Real tokens then occupy positions ``0..L-1``,
exactly the positions they occupy when the item is run alone, so no explicit
``position_ids`` are needed and none are invented. Under a causal mask a real
token at position ``i < L`` can only attend to positions ``<= i``, which are all
real, and the ``attention_mask`` zeroes the pads for good measure. Left padding
would shift every real position and silently change the RoPE phase.

This module is deliberately ignorant of *why* a caller chose a batch size: it
takes the number, and the separation between scientific and execution
configuration is enforced one level up.

**The pad id is numerically inert, and is still chosen rather than invented.**
Because of the two properties above, no real position's output depends on what
sits in the pad slots — measured exactly, at ``0.0``, by
``test_the_pad_id_is_numerically_inert``. It must nonetheless be a token the
embedding can look up, so ``resolve_pad_id`` takes it from the model's own
configuration, range-checks it, and falls back to a documented constant only
when the config declares nothing. A *declared* id that does not fit the
vocabulary raises instead of falling through, because that is a real
tokenizer/weights disagreement and not something to paper over.

For the same reason the ``attention_mask`` this module builds is **defensive
rather than load-bearing** under right padding: causality alone already keeps
real tokens away from pads. It is passed anyway — it costs nothing, it is
required the moment anything pads on the other side, and a reader should not
have to re-derive the argument to trust the call.

Nothing here knows about a model family, an operator, a stage or a geometry. It
takes items that already carry ``input_ids`` of shape ``[1, T]`` — the form
:mod:`aadistill.initialization.calibration.items` guarantees — and returns
tensors.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

#: Batch size is NOT resolved here and NOT read from a config mapping. It is an
#: execution property carried by `OperatorContext.execution`, so that it cannot
#: reach `OperatorStep.config_hash` and therefore cannot fork a scientific state
#: id. See `aadistill.initialization.execution`, which owns the default.


class BatchingError(RuntimeError):
    """A calibration batch cannot be built.

    Every refusal below raises this rather than a bare `ValueError`, so a caller
    can tell "the batcher rejected these items" apart from an arithmetic failure
    inside a model. It briefly went missing while the configuration machinery
    was removed from this module, which turned each of those refusals into a
    `NameError` — caught by the core-boundary scan, which reads unresolvable
    global names, and pinned below by a test that asserts the type.
    """


#: The documented last resort. Valid for every non-empty vocabulary, and safe
#: for the reason the module docstring gives and
#: ``test_the_pad_id_is_numerically_inert`` demonstrates: under right padding
#: with an ``attention_mask``, no real position's output depends on what sits in
#: a pad slot. It is a *stated* strategy with a test behind it, which is the
#: opposite of inventing a token and hoping.
FALLBACK_PAD_ID = 0


def resolve_pad_id(model: Any) -> int:
    """A token id safe to place in padded positions.

    Preference order: ``pad_token_id``, ``eos_token_id``, ``bos_token_id``, then
    :data:`FALLBACK_PAD_ID`. Whatever is chosen is range-checked against the
    vocabulary, because an out-of-range id is an embedding lookup failure — the
    one way a pad token can actually break a forward.

    A declared-but-out-of-range id **raises** rather than falling through. A
    config whose ``eos_token_id`` does not fit its own ``vocab_size`` is
    describing a different tokenizer than the weights were built for, and
    quietly substituting 0 would hide that from every operator downstream.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise BatchingError(
            "cannot resolve a pad token id: the model has no `config`")
    vocab = getattr(config, "vocab_size", None)
    for field in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(config, field, None)
        if isinstance(value, (list, tuple)):        # some configs carry several
            value = value[0] if value else None
        if value is None or isinstance(value, bool):
            continue
        token = int(value)
        if vocab is not None and not 0 <= token < int(vocab):
            raise BatchingError(
                f"config.{field} is {token}, outside the vocabulary of "
                f"{int(vocab)}; refusing to pad with an id the embedding "
                "cannot look up, and refusing to silently substitute another")
        return token
    if vocab is not None and int(vocab) <= FALLBACK_PAD_ID:
        raise BatchingError(
            f"vocab_size is {int(vocab)}, so even {FALLBACK_PAD_ID} is not a "
            "valid token id; pass `pad_id=` explicitly")
    return FALLBACK_PAD_ID


@dataclass(frozen=True)
class ItemBatch:
    """Padded tokens for several items, and the means to undo the padding.

    ``items`` holds the original mappings in row order, so a consumer recovers
    every per-item fact it had before — ``item_id``, ``domain``, ``subtype`` —
    without this module knowing that any of them exist.
    """

    input_ids: torch.Tensor          #: [B, T_max], right-padded
    attention_mask: torch.Tensor     #: [B, T_max], 1 real / 0 pad
    lengths: tuple[int, ...]         #: true token count per row
    items: tuple[Mapping[str, Any], ...]
    pad_id: int

    @property
    def size(self) -> int:
        return len(self.lengths)

    @property
    def is_padded(self) -> bool:
        """Whether any row actually carries padding."""
        return len(set(self.lengths)) > 1

    def token_mask(self) -> torch.Tensor:
        """``[B, T_max]`` bool over real token positions."""
        return self.attention_mask.bool()

    def prediction_mask(self) -> torch.Tensor:
        """``[B, T_max - 1]`` bool over the positions an item predicts.

        An item of ``L`` tokens predicts ``L - 1`` positions, the same count the
        unbatched path gets from ``logits[0, :-1]``.
        """
        mask = torch.zeros(self.size, self.input_ids.shape[1] - 1,
                           dtype=torch.bool, device=self.input_ids.device)
        for row, length in enumerate(self.lengths):
            mask[row, :length - 1] = True
        return mask

    def split_predictions(self, batched: torch.Tensor) -> list[torch.Tensor]:
        """``[B, T_max - 1, ...]`` -> one ``[L_i - 1, ...]`` tensor per item.

        This is the inverse of the padding, and it is what keeps a batched
        measurement a *per-item* measurement: each returned tensor is exactly
        what that item's own forward would have produced at
        ``logits[0, :-1]``, so the caller's existing per-item reduction runs
        unchanged over unchanged position counts.
        """
        if batched.shape[0] != self.size:
            raise BatchingError(
                f"expected a leading batch dimension of {self.size}, got "
                f"{tuple(batched.shape)}")
        width = self.input_ids.shape[1] - 1
        if batched.shape[1] < width:
            raise BatchingError(
                f"expected at least {width} prediction positions per row, got "
                f"{tuple(batched.shape)}")
        return [batched[row, :length - 1] for row, length in enumerate(self.lengths)]

    def valid_tokens(self, batched: torch.Tensor) -> torch.Tensor:
        """``[B, T_max, D]`` -> ``[sum(lengths), D]``, padding dropped.

        For accumulators that sum over tokens and do not care which item a token
        came from. The row order is preserved, so the result is exactly the
        concatenation of the per-item token blocks.
        """
        if batched.shape[:2] != self.input_ids.shape[:2]:
            raise BatchingError(
                f"expected leading dimensions {tuple(self.input_ids.shape[:2])}, "
                f"got {tuple(batched.shape[:2])}")
        return batched[self.token_mask()]

    @property
    def n_valid_tokens(self) -> int:
        return int(sum(self.lengths))


def build_batch(items: Sequence[Mapping[str, Any]], *, pad_id: int,
                device: Any = None) -> ItemBatch:
    """One :class:`ItemBatch` from items already carrying ``[1, T]`` ids."""
    if not items:
        raise BatchingError("refusing to build an empty batch")
    rows, lengths = [], []
    for index, item in enumerate(items):
        ids = item.get("input_ids") if isinstance(item, Mapping) else None
        if not isinstance(ids, torch.Tensor):
            raise BatchingError(
                f"item {index}: 'input_ids' is {type(ids).__name__}, not a "
                "torch.Tensor; items reach an operator through "
                "`prepare_calibration_items`, which guarantees one")
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise BatchingError(
                f"item {index}: 'input_ids' has shape {tuple(ids.shape)}, not "
                "[1, T]")
        rows.append(ids[0])
        lengths.append(int(ids.shape[1]))

    width = max(lengths)
    #: Built on the target device directly. A host tensor moved afterwards is an
    #: extra copy of [B, T_max] per batch, and on CUDA it is also a
    #: synchronisation the forward does not need.
    target = device if device is not None else rows[0].device
    input_ids = torch.full((len(rows), width), int(pad_id),
                           dtype=torch.long, device=target)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long,
                                 device=target)
    for row, (ids, length) in enumerate(zip(rows, lengths)):
        input_ids[row, :length] = ids.to(target)
        attention_mask[row, :length] = 1
    return ItemBatch(input_ids=input_ids, attention_mask=attention_mask,
                     lengths=tuple(lengths), items=tuple(items),
                     pad_id=int(pad_id))


def micro_batches(items: Sequence[Mapping[str, Any]], batch_size: int, *,
                  pad_id: int, device: Any = None) -> Iterator[ItemBatch]:
    """Consecutive groups of ``batch_size`` items, in the mixture's own order.

    **Order is preserved and items are never sorted by length.** Grouping items
    of similar length would pad less, and it would also make which items share a
    forward depend on the mixture's contents, so two runs of the same mixture
    could batch differently after an unrelated edit. The mixture's order is the
    one a reader can check against the frozen record — the same argument
    ``_ReferenceLogits`` already makes for its admission order.
    """
    if batch_size < 1:
        raise BatchingError(f"batch_size must be >= 1, got {batch_size}")
    items = list(items)
    for start in range(0, len(items), batch_size):
        yield build_batch(items[start:start + batch_size], pad_id=pad_id,
                          device=device)
