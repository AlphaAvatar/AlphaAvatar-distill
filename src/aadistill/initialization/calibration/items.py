"""The one boundary between a frozen calibration mixture and an operator.

A materialized mixture on disk stores each item's tokens under **`ids`**, a plain
JSON list. That is the form ``mixture_content_sha256`` hashes, so it is the form
the pinned ``d65c1f40…`` content identity is defined over and it must not change.
Every calibrated operator, meanwhile, reads ``item["input_ids"]`` and hands it
straight to ``collector.process(ids.to(device))`` or slices ``[0, 1:]`` off it —
they want a ``[1, T]`` LongTensor.

Something has to convert. Until now nothing in the library did: the conversion
lived in ``scripts/autoinit/phase_a_search.as_operator_items``, so the search had
it and the fixed-path executor did not. A paid session met that gap — a stage
reached ``depth.apply`` and raised ``KeyError: 'input_ids'`` on the
real frozen mixture, after the pod had loaded 398 weight shards.

**Why it kept being invisible.** Every fixed-path test passes
``calibration_items=`` and every toy fixture builds items already in the operator
shape, so no `$0` run ever asked a real profile for its items. The same class of
defect has appeared before, for the same reason.

So this module exists, and there is exactly one of it:

* it is the **only** place that turns raw ``ids`` into ``input_ids``;
* DEPTH, FFN, WIDTH and ATTENTION get no per-operator fallback, because four
  fallbacks are four chances to disagree about what an item is;
* it validates rather than repairs — an item that cannot be prepared raises,
  since an operator calibrated on a silently-truncated or mislabelled mixture
  still produces a ranking, and the ranking still looks like evidence.

It is deliberately **not** in :mod:`aadistill.autoinit.calibration`: the stored
form is what the frozen content hash is defined over, and ``resolve()`` returning
anything but the raw evidence would make that hash a hash of something else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Where a frozen mixture stores its tokens. Not ``input_ids`` — see the module
#: docstring: this key is inside the pinned content identity.
RAW_TOKENS_KEY = "ids"

#: What every calibrated operator reads.
OPERATOR_TOKENS_KEY = "input_ids"

#: Optional raw field. When present it is a stated fact about the item and is
#: checked, because a mixture whose declared prediction count disagrees with its
#: own token count is not a mixture anybody should calibrate on.
PREDICTION_POSITIONS_KEY = "n_prediction_positions"


class CalibrationItemError(RuntimeError):
    """A calibration item cannot be prepared for an operator."""


def _where(profile_id: str, index: int, item: Any) -> str:
    item_id = item.get("item_id") if isinstance(item, Mapping) else None
    return (f"{profile_id or '<items>'}: item {index}"
            + (f" ({item_id!r})" if item_id is not None else ""))


def _tokens_of(tensor) -> list[int]:
    return [int(t) for t in tensor.reshape(-1).tolist()]


def prepare_calibration_items(
    items: Sequence[Mapping[str, Any]],
    *,
    profile_id: str = "",
) -> list[dict[str, Any]]:
    """Raw materialized items in, operator-ready items out.

    Raw metadata and the raw ``ids`` are preserved exactly; the only thing added
    is ``input_ids``, a ``torch.long`` tensor of shape ``[1, N]`` holding those
    same tokens. Input mappings are never mutated.

    An item that already carries ``input_ids`` — every toy fixture, and the
    device canary — is **validated and accepted as it stands**, never
    re-tokenized. When an item carries both, the two must agree token for token:
    a prepared tensor that has drifted from the raw evidence beside it is the one
    thing this boundary exists to make impossible.

    Raises :class:`CalibrationItemError` on anything it cannot vouch for.
    """
    import torch

    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        where = _where(profile_id, index, item)
        if not isinstance(item, Mapping):
            raise CalibrationItemError(
                f"{where}: a calibration item must be a mapping, not "
                f"{type(item).__name__}")

        raw = item.get(RAW_TOKENS_KEY)
        prepared = item.get(OPERATOR_TOKENS_KEY)

        if prepared is None:
            if raw is None:
                raise CalibrationItemError(
                    f"{where}: has neither {RAW_TOKENS_KEY!r} nor "
                    f"{OPERATOR_TOKENS_KEY!r}. A frozen mixture stores its tokens "
                    f"under {RAW_TOKENS_KEY!r}; an operator reads "
                    f"{OPERATOR_TOKENS_KEY!r}. This item can supply neither.")
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise CalibrationItemError(
                    f"{where}: {RAW_TOKENS_KEY!r} is {type(raw).__name__}, not a "
                    "sequence of token ids")
            ids = list(raw)
            if not ids:
                raise CalibrationItemError(f"{where}: {RAW_TOKENS_KEY!r} is empty")
            try:
                tensor = torch.tensor(ids, dtype=torch.long)[None, :]
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CalibrationItemError(
                    f"{where}: {RAW_TOKENS_KEY!r} is not a flat list of integer "
                    f"token ids ({exc})") from exc
        else:
            if not isinstance(prepared, torch.Tensor):
                raise CalibrationItemError(
                    f"{where}: {OPERATOR_TOKENS_KEY!r} is "
                    f"{type(prepared).__name__}, not a torch.Tensor. The operators "
                    "call .to(device) and slice [0, 1:] on it.")
            tensor = prepared

        if tensor.dim() != 2:
            raise CalibrationItemError(
                f"{where}: {OPERATOR_TOKENS_KEY!r} has rank {tensor.dim()}, not 2 "
                "([batch, tokens])")
        if tensor.shape[0] != 1:
            raise CalibrationItemError(
                f"{where}: {OPERATOR_TOKENS_KEY!r} has batch {tensor.shape[0]}, "
                "not 1. Operators consume one item per forward pass.")
        if tensor.dtype != torch.long:
            raise CalibrationItemError(
                f"{where}: {OPERATOR_TOKENS_KEY!r} has dtype {tensor.dtype}, not "
                "torch.long")
        if tensor.shape[1] == 0:
            raise CalibrationItemError(
                f"{where}: {OPERATOR_TOKENS_KEY!r} holds no tokens")

        if raw is not None and prepared is not None:
            if _tokens_of(tensor) != [int(t) for t in raw]:
                raise CalibrationItemError(
                    f"{where}: carries both {RAW_TOKENS_KEY!r} and "
                    f"{OPERATOR_TOKENS_KEY!r} and they are not the same tokens. "
                    "The raw ids are the evidence the content hash is defined "
                    "over; a prepared tensor that disagrees with them is the "
                    "defect this boundary exists to refuse.")

        declared = item.get(PREDICTION_POSITIONS_KEY)
        if declared is not None:
            expected = int(tensor.shape[1]) - 1
            if int(declared) != expected:
                raise CalibrationItemError(
                    f"{where}: declares {PREDICTION_POSITIONS_KEY}="
                    f"{int(declared)} but holds {int(tensor.shape[1])} tokens, "
                    f"which predict {expected} positions")

        out.append({**item, OPERATOR_TOKENS_KEY: tensor})
    return out
