"""Pre-vocabulary position selection: project only the rows a reducer scores.

Every divergence this package computes is a weighted sum over prediction
positions, and a position with zero weight -- padding, or a prompt position a
target-aware policy does not score -- contributes exactly zero to it. The
historical path still pays for those positions in full: the output head
projects every slot of the forward to ``[.., V]`` logits, the reducers
normalize every row over the whole vocabulary, and the weights are applied only
when the per-position values are summed. A mask there saves no vocabulary work.

This module moves the selection BEFORE the output head:

1. the backbone runs over the complete sequence, so every selected position
   keeps its full causal context;
2. the rows a reducer would weight are picked from the final hidden states --
   prediction position ``j`` reads token ``j``'s state, the same causal shift as
   ``logits[:, :-1]``;
3. only those rows are projected to the vocabulary and normalized;
4. the per-row values are scattered back into the dense ``[B, T_pred]`` layout,
   so the reducer that forms the per-item means is the SAME code, with the same
   weights and the same row-wise denominators, as on the historical path.

What it saves is the inactive fraction of the output-head GEMM and of every
vocabulary-wide reduction (log-normalizer, Top-K, tail). What it does not save
is anything in the backbone, and what it does not change is the estimand: a
reference Top-K + tail partition still needs the full-vocabulary normalizer of
every selected row, so the per-row cost stays O(V) -- the saving is in the
number of rows, never in the width.

Selected by ``ExecutionConfig.prediction_rows``; ``all_v1`` keeps the historical
path operation for operation. Reference and candidate always select the same
rows from the same batch layout, which is what keeps the two distributions
position-aligned.
"""
from __future__ import annotations

from typing import Any

import torch

from aadistill.initialization.specs.arch import ArchitectureAdapter, Capability


def scored_rows(prediction_mask: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    """``[B, T_pred]`` bool: valid rows the reducer gives non-zero weight.

    A SUPERSET of what any masked reducer weights, by construction: a row is
    dropped only when it is padding or its weight is exactly zero, and both
    contribute exactly zero to every reduction here. The reducer still applies
    its own weights to the rows that survive, so a row this keeps and the
    reducer zeroes costs work, never correctness.
    """
    rows = prediction_mask.bool()
    if weights is None:
        return rows
    if tuple(weights.shape) != tuple(rows.shape):
        raise ValueError(
            f"weights {tuple(weights.shape)} do not describe the prediction "
            f"mask {tuple(rows.shape)}; selecting rows from a mis-shaped weight "
            "array would score one item's positions under another's weights")
    return rows & (weights.to(rows.device) > 0)


@torch.no_grad()
def scored_prediction_logits(adapter: ArchitectureAdapter, model: Any,
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor | None,
                             rows: torch.Tensor) -> torch.Tensor:
    """``[N, V]`` logits for the selected prediction rows, in row-major order.

    ``input_ids`` is ``[B, T]``; ``rows`` is ``[B, T - 1]`` over PREDICTION
    positions. Row-major order means row 0's selected positions in increasing
    position order, then row 1's, and so on -- the order boolean indexing
    produces, and the order :func:`split_rows` and :func:`scatter_rows` undo.
    """
    adapter.requires(Capability.OUTPUT_HEAD)
    if input_ids.dim() != 2:
        raise ValueError(f"expected [B, T] input ids, got {tuple(input_ids.shape)}")
    b, t = int(input_ids.shape[0]), int(input_ids.shape[1])
    if tuple(rows.shape) != (b, t - 1):
        raise ValueError(
            f"rows {tuple(rows.shape)} do not describe the prediction positions "
            f"of a [{b}, {t}] batch, which are [{b}, {t - 1}]")
    hidden = adapter.final_hidden_states(model, input_ids, attention_mask)
    #: The causal shift: prediction position j is read off token j's state,
    #: exactly as `logits[:, :-1]` is on the historical path.
    picked = hidden[:, :-1][rows.to(hidden.device)]
    return adapter.output_logits(model, picked)


def row_counts(rows: torch.Tensor) -> list[int]:
    """Selected positions per batch row, for splitting row-major values."""
    return [int(c) for c in rows.sum(dim=1).tolist()]


def split_rows(values: torch.Tensor, rows: torch.Tensor) -> list[torch.Tensor]:
    """Row-major selected values back into one tensor per batch row."""
    counts = row_counts(rows)
    if sum(counts) != int(values.shape[0]):
        raise ValueError(
            f"{int(values.shape[0])} selected values for {sum(counts)} selected "
            "positions")
    return list(torch.split(values, counts, dim=0))


def scatter_rows(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``[N]`` per-row values into a dense ``[B, T_pred]`` zero-filled array.

    Unselected positions hold zero and are never read: every reducer this feeds
    masks by its own weights before summing, and an unselected position has
    weight zero by the definition of :func:`scored_rows`.
    """
    if values.dim() != 1:
        raise ValueError(f"expected [N] values, got {tuple(values.shape)}")
    rows = rows.to(values.device)
    if int(rows.sum()) != int(values.shape[0]):
        raise ValueError(
            f"{int(values.shape[0])} values for {int(rows.sum())} selected rows")
    dense = torch.zeros(rows.shape, dtype=values.dtype, device=values.device)
    dense[rows] = values
    return dense


__all__ = ["row_counts", "scatter_rows", "scored_prediction_logits",
           "scored_rows", "split_rows"]
