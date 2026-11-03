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
class Segment:
    """One item's tokens inside one row of a PACKED batch.

    The row holds the item's first ``length`` tokens at ``start .. start+length-1``
    (every placement is a prefix: `aadistill.data.sequence_packing`), and its
    ``length - 1`` prediction positions are ``start .. start+length-2``. The
    position after them predicts the NEXT segment's first token and belongs to no
    item. ``index`` is the item's position in the list that was packed, and
    ``final`` is ``False`` for a fragment.
    """

    row: int
    start: int
    length: int
    index: int
    final: bool = True

    @property
    def prediction_slice(self) -> slice:
        return slice(self.start, self.start + self.length - 1)


@dataclass(frozen=True)
class ItemBatch:
    """Padded tokens for several items, and the means to undo the padding.

    ``items`` holds the original mappings in row order, so a consumer recovers
    every per-item fact it had before — ``item_id``, ``domain``, ``subtype`` —
    without this module knowing that any of them exist.

    **A packed batch** (``segments`` set) holds several items per row, read with
    standard causal attention and continuous positions, so a later item is
    conditioned on the earlier ones in its row. ``items`` is then one mapping
    PER SEGMENT, ``lengths`` the non-padding token count per row, and per-item
    results come from the segment views (:meth:`segment_predictions`). The
    per-row views that assume one item per row (:meth:`split_predictions`)
    refuse a packed batch rather than attribute a row to its first item.
    """

    input_ids: torch.Tensor          #: [B, T_max], right-padded
    attention_mask: torch.Tensor     #: [B, T_max], 1 real / 0 pad
    lengths: tuple[int, ...]         #: true token count per row
    items: tuple[Mapping[str, Any], ...]
    pad_id: int
    segments: tuple[Segment, ...] | None = None

    @property
    def size(self) -> int:
        return len(self.lengths)

    @property
    def packed(self) -> bool:
        return self.segments is not None

    def _require_unpacked(self, what: str) -> None:
        if self.segments is not None:
            raise BatchingError(
                f"{what} assumes one item per row and this batch is packed "
                f"({len(self.segments)} segments in {self.size} rows); use the "
                "segment views")

    def segment_predictions(self, batched: torch.Tensor) -> list[torch.Tensor]:
        """``[B, T_max - 1, ...]`` -> one ``[L_s - 1, ...]`` tensor per segment,
        in segment order. Each is the segment's own prediction slice."""
        if self.segments is None:
            return self.split_predictions(batched)
        if batched.shape[0] != self.size:
            raise BatchingError(
                f"expected a leading batch dimension of {self.size}, got "
                f"{tuple(batched.shape)}")
        return [batched[s.row, s.prediction_slice] for s in self.segments]

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
        if self.segments is not None:
            #: A segment's last token predicts the next segment's first token,
            #: which belongs to no item: that position stays False.
            for s in self.segments:
                mask[s.row, s.prediction_slice] = True
            return mask
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
        self._require_unpacked("split_predictions")
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


def build_packed_batch(sequences: Sequence[Sequence[Any]],
                       items: Sequence[Mapping[str, Any]], *, pad_id: int,
                       device: Any = None) -> ItemBatch:
    """One packed :class:`ItemBatch`: each row is one packed sequence.

    ``sequences[r]`` lists row ``r``'s placements in order, each with
    ``index`` (into ``items``), ``tokens`` (how many of the item's FIRST tokens)
    and ``final`` -- `aadistill.data.sequence_packing.Placement`. Items are laid
    end to end with no separator; rows are right-padded to the widest. No
    position ids are built: the model's default ``0 .. T-1`` are the continuous
    positions this layout is read with.
    """
    if not sequences:
        raise BatchingError("refusing to build an empty packed batch")
    rows: list[list[torch.Tensor]] = []
    segments: list[Segment] = []
    seg_items: list[Mapping[str, Any]] = []
    lengths: list[int] = []
    for r, seq in enumerate(sequences):
        at, parts = 0, []
        for p in seq:
            item = items[int(p.index)]
            ids = item.get("input_ids") if isinstance(item, Mapping) else None
            if not isinstance(ids, torch.Tensor) or ids.dim() != 2 or ids.shape[0] != 1:
                raise BatchingError(
                    f"item {p.index}: a packed batch needs 'input_ids' of shape "
                    "[1, T]; items reach an operator through "
                    "`prepare_calibration_items`, which guarantees one")
            n = int(p.tokens)
            if not 1 <= n <= int(ids.shape[1]):
                raise BatchingError(
                    f"item {p.index}: placing {n} tokens of a {int(ids.shape[1])}"
                    "-token item")
            parts.append(ids[0, :n])
            segments.append(Segment(row=r, start=at, length=n, index=int(p.index),
                                    final=bool(p.final)))
            seg_items.append(item)
            at += n
        rows.append(parts)
        lengths.append(at)
    width = max(lengths)
    target = device if device is not None else rows[0][0].device
    input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long,
                           device=target)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long,
                                 device=target)
    for r, parts in enumerate(rows):
        input_ids[r, :lengths[r]] = torch.cat(parts).to(target)
        attention_mask[r, :lengths[r]] = 1
    return ItemBatch(input_ids=input_ids, attention_mask=attention_mask,
                     lengths=tuple(lengths), items=tuple(seg_items),
                     pad_id=int(pad_id), segments=tuple(segments))


def active_rows(mask: Any, shape: Any, device: Any = None):
    """A position mask reshaped to ``[B, T]`` on ``device``, or ``None``.

    Both activation-statistics collectors take an optional "which positions may
    this statistic read" mask from a scoring policy, and both must reconcile it
    with the ``[B, T_max]`` shape of the tokens actually being processed. That
    reconciliation is batch knowledge, so it lives here, once — two copies would
    be two chances to disagree about what a mask of the wrong length means.

    ``[T]`` is accepted for the single-sequence path so a caller with one item
    does not have to add a leading dimension. Anything else **raises**: a mask of
    the wrong length would be broadcast by the ``&`` downstream and would
    silently admit or drop positions, which is the one failure a scoring
    restriction must not have.
    """
    if mask is None:
        return None
    rows, width = int(shape[0]), int(shape[1])
    m = mask if mask.dim() == 2 else mask.reshape(1, -1)
    if tuple(m.shape) != (rows, width):
        raise BatchingError(
            f"active mask has shape {tuple(mask.shape)}, which is not the "
            f"({rows}, {width}) of the tokens being processed; a mis-shaped "
            "scoring mask changes which positions are counted instead of failing")
    if m.dtype != torch.bool:
        raise BatchingError(
            f"active mask has dtype {m.dtype}, not torch.bool. The collectors "
            "support the BINARY form of a position policy; continuous weights "
            "need a weighted denominator in StatsSpec, which no experiment has "
            "asked for yet")
    return m if device is None else m.to(device)


def active_weights(weights: Any, shape: Any, device: Any = None):
    """Continuous token weights reshaped to ``[B, T]`` float64, or ``None``.

    The weighted sibling of :func:`active_rows`, with the same refusal of a
    mis-shaped array, plus a refusal of a negative or non-finite weight: a
    weight enters both a numerator and a denominator, and either would turn a
    weighted mean into something else.
    """
    if weights is None:
        return None
    rows, width = int(shape[0]), int(shape[1])
    w = weights if weights.dim() == 2 else weights.reshape(1, -1)
    if tuple(w.shape) != (rows, width):
        raise BatchingError(
            f"token weights have shape {tuple(weights.shape)}, which is not "
            f"the ({rows}, {width}) of the tokens being processed")
    w = w.to(torch.float64)
    if bool((w < 0).any()) or not bool(torch.isfinite(w).all()):
        raise BatchingError("token weights must be finite and non-negative")
    return w if device is None else w.to(device)


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
