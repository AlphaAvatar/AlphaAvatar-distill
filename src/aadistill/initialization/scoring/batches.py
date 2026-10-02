"""A scoring policy, applied to the batches an operator actually runs.

The policy in :mod:`aadistill.initialization.scoring.positions` answers per item:
"which of this item's positions may the objective read, and how much does each
count?". An operator does not run items; it runs *batches* of them, right-padded
to the longest member, in whatever row order its packing policy chose. Something
has to carry the per-item answer into that shape, and it must do so exactly once
— four operators each doing their own index arithmetic is four chances to
attribute one item's mask to another's row.

So this is the one bridge, and it holds two rules that are easy to get wrong:

**Row order is not mixture order.** Under ``length_sorted_v1`` the rows of a
batch are a permutation of the mixture, and ``PackedBatch.original_indices`` is
the only thing that says which. Every lookup here goes through it. A consumer
that assumed row ``r`` was item ``r`` would silently score one item's positions
against another's activations — the exact failure ``PackedBatch`` carries that
field to prevent.

**Padding is zero under both masks, and the order of the ``and`` does not
matter.** A padded position is not a position, so it is excluded whether or not
the policy would have admitted the real position that happens to sit at the same
index in a shorter row.

The two shapes an operator needs are different because the two axes are
different: an activation statistic restricts over ``[B, T_max]`` token positions,
and a next-token objective weights over ``[B, T_max - 1]`` prediction positions.
Both come from the same :class:`ActivePositions` so they cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from aadistill.initialization.scoring.positions import (
    FORM_ALL,
    FORM_SELECT,
    PREDICTION_AXIS,
    TOKEN_AXIS,
    PositionWeights,
    ScoringPositionError,
    ScoringPositionPolicy,
    WEIGHT_DTYPE,
    weights_for_items,
)


class ActivePositions:
    """One policy evaluated against one ordered item list, reusable per batch.

    Built once per operator invocation. Evaluating the policy is cheap but it is
    not free — it normalizes every item's tags — and a candidate search calls an
    operator's inner loop hundreds of times, so the per-item answer is computed
    once and reshaped per batch.
    """

    def __init__(self, items: Sequence[Any], policy: ScoringPositionPolicy) -> None:
        self.policy = policy
        self.n_items = len(items)
        self._token = weights_for_items(items, policy, axis=TOKEN_AXIS)
        self._prediction = weights_for_items(items, policy, axis=PREDICTION_AXIS)

    # --- what kind of restriction is this? ---------------------------------

    @property
    def token_form(self) -> str:
        return _combined_form(self._token)

    @property
    def prediction_form(self) -> str:
        return _combined_form(self._prediction)

    @property
    def all_token_positions_active(self) -> bool:
        """True when nothing is restricted, so a caller can skip the mask entirely.

        Not a micro-optimization: it is what keeps the incumbent policy on the
        collectors' untouched arithmetic rather than on a mask of all ones.
        """
        return self.token_form == FORM_ALL

    @property
    def all_prediction_positions_active(self) -> bool:
        return self.prediction_form == FORM_ALL

    def require_binary_token_weights(self) -> None:
        """Refuse a continuous policy where only a mask can be honoured.

        The activation collectors accumulate ``sum_t x_t`` against an ``int64``
        position count; a continuous ``w_t`` needs ``sum_t w_t x_t`` over
        ``sum_t w_t``, which is a new ``StatsSpec`` quantity and three divisor
        call sites. Refusing by name is the honest boundary — rounding a
        confidence weight to a mask would run a different experiment than the one
        that was preregistered and would report it as the right one.
        """
        if self.token_form not in (FORM_ALL, FORM_SELECT):
            raise ScoringPositionError(
                f"{self.policy.qualified_id} assigns continuous position weights, "
                "and the activation-statistics collectors implement only the "
                "binary form: their divisor is an integer token count. Weighted "
                "activation statistics need a weighted denominator in StatsSpec "
                "(`uncentered_moment`, `ffn_neuron_importance` and "
                "`residual_covariance` all divide by `residual_count`), which no "
                "experiment has yet asked for. Refusing rather than rounding the "
                "weights to a mask")

    # --- the two batch shapes ----------------------------------------------

    def token_mask_for(self, batch: Any,
                       original_indices: Sequence[int]) -> torch.Tensor | None:
        """``[B, T_max]`` bool over admitted token positions, or ``None``.

        ``None`` when the policy admits every real position, which lets the
        collectors stay on the arithmetic every committed artifact was built
        with. Padding is False regardless, because the batch's own
        ``attention_mask`` is combined with this downstream — and it is also
        False here, so neither side depends on the other being right.
        """
        if self.all_token_positions_active:
            return None
        rows, width = _batch_shape(batch)
        self.require_binary_token_weights()
        mask = torch.zeros(rows, width, dtype=torch.bool,
                           device=batch.input_ids.device)
        for row, index in enumerate(_checked_indices(original_indices, rows,
                                                    self.n_items)):
            w = self._token[index]
            length = min(w.n_positions, width)
            mask[row, :length] = w.active_mask()[:length].to(mask.device)
        return mask

    def prediction_weights_for(self, batch: Any,
                               original_indices: Sequence[int],
                               ) -> torch.Tensor | None:
        """``[B, T_max - 1]`` float weights over prediction positions, or ``None``.

        ``None`` under the incumbent policy, which is what
        :func:`~aadistill.initialization.statistics.contribution.forward_kl_mean_batch`
        reads as "take the path the frozen DEPTH decisions were produced by".
        Unlike the token mask this one is **not** restricted to the binary form:
        a weighted next-token objective needs no schema change, because its
        denominator is formed locally per row.
        """
        if self.all_prediction_positions_active:
            return None
        rows, width = _batch_shape(batch)
        out = torch.zeros(rows, width - 1, dtype=WEIGHT_DTYPE,
                          device=batch.input_ids.device)
        for row, index in enumerate(_checked_indices(original_indices, rows,
                                                    self.n_items)):
            w = self._prediction[index]
            length = min(w.n_positions, width - 1)
            out[row, :length] = w.weights[:length].to(out.device)
        return out

    def prediction_weights_for_item(self, index: int) -> torch.Tensor | None:
        """``[T_pred]`` weights for one item, for the unbatched reference path."""
        if self.all_prediction_positions_active:
            return None
        return self._prediction[index].weights

    def token_mask_for_item(self, index: int) -> torch.Tensor | None:
        """``[T]`` bool for one item, for the unbatched reference path."""
        if self.all_token_positions_active:
            return None
        self.require_binary_token_weights()
        return self._token[index].active_mask()

    # --- evidence ----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """What a trace should say about the restriction that was applied.

        Totals rather than per-item vectors: the per-item answer is derivable
        from the frozen mixture and this policy id, and an operator trace that
        inlined 67 boolean vectors would bury the numbers a reader checks.
        """
        return {
            "position_policy": self.policy.qualified_id,
            "position_policy_hash": self.policy.policy_hash,
            "n_items": self.n_items,
            "token_axis": {
                "form": self.token_form,
                "positions": sum(w.n_positions for w in self._token),
                "active": sum(w.n_active for w in self._token),
                "weight_total": sum(w.total for w in self._token),
            },
            "prediction_axis": {
                "form": self.prediction_form,
                "positions": sum(w.n_positions for w in self._prediction),
                "active": sum(w.n_active for w in self._prediction),
                "weight_total": sum(w.total for w in self._prediction),
            },
        }


def _combined_form(weights: Sequence[PositionWeights]) -> str:
    """The weakest form across items — a mixture is as general as its generalest.

    The frozen mixtures make this concrete: under the supervised-target policy
    the untemplated raw-LM items are :data:`FORM_ALL` and the templated ones are
    :data:`FORM_SELECT`, so the mixture as a whole is ``select`` and a caller
    that read the first item's form would have skipped the mask on two thirds of
    the corpus.
    """
    forms = {w.form for w in weights}
    for candidate in ("weighted", FORM_SELECT, FORM_ALL):
        if candidate in forms:
            return candidate
    return FORM_ALL


def _batch_shape(batch: Any) -> tuple[int, int]:
    ids = getattr(batch, "input_ids", None)
    if ids is None or ids.dim() != 2:
        raise ScoringPositionError(
            "expected an ItemBatch whose `input_ids` is [B, T_max]; a position "
            "mask cannot be shaped against anything else")
    return int(ids.shape[0]), int(ids.shape[1])


def _checked_indices(original_indices: Sequence[int], rows: int,
                     n_items: int) -> list[int]:
    """``original_indices``, validated. The one place row order is trusted.

    Checked rather than assumed because this is where a packing permutation
    becomes a scoring decision: a short, long or out-of-range index list would
    attribute one item's supervised positions to another item's activations, and
    the result would still look like a ranking.
    """
    index = [int(i) for i in original_indices]
    if len(index) != rows:
        raise ScoringPositionError(
            f"{len(index)} original indices for a batch of {rows} rows; a "
            "scoring mask built from a mismatched permutation attributes one "
            "item's positions to another")
    bad = sorted(i for i in index if not 0 <= i < n_items)
    if bad:
        raise ScoringPositionError(
            f"original indices {bad[:4]} are outside the {n_items} items this "
            "policy was evaluated against")
    return index


def active_positions(items: Sequence[Any], policy: ScoringPositionPolicy,
                     ) -> ActivePositions | None:
    """:class:`ActivePositions` for ``items``, or ``None`` when there are none.

    ``None`` rather than an empty object so a weight-only operator — handed no
    calibration items at all — does not have to pretend it has a position policy.
    """
    if not items:
        return None
    return ActivePositions(items, policy)


def consecutive_indices(batch: Any) -> tuple[int, ...]:
    """``(0, 1, ..., B-1)`` offset by nothing — for a caller with no permutation.

    ``micro_batches`` yields bare :class:`ItemBatch` objects with no
    ``original_indices``, and its grouping is consecutive in mixture order, so a
    caller that still uses it supplies the offset itself. Provided here so the
    offset arithmetic is written once; a caller that reaches for it should
    usually be using ``packed_batches`` instead.
    """
    rows, _ = _batch_shape(batch)
    return tuple(range(rows))
