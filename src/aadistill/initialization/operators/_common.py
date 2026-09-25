"""Shared surgery helpers for the v1 operators.

Two invariants live here rather than in each operator:

**An operator never mutates the model it was handed.** It builds a child at the
new spec and copies into it. The parent is the previous search state's
checkpoint; a search that quietly consumed it could not re-expand that node, and
resume would produce a different tree. ``OperatorImplementation.execute``
enforces this by re-reading the parent spec afterwards and refusing an outcome
whose model *is* the parent.

**Every child parameter is accounted for.** ``ChildBuilder`` starts from a freshly
random-initialized model and refuses to hand back a child with an unassigned
parameter. Without that check, an operator that forgot one norm would ship a
checkpoint carrying a random tensor, and the state evaluation would faithfully
measure it and rank it — a silent corruption that reads as a real result. Random
initialization is kept (rather than a skip-init fast path) precisely so this
check has something to fail on.

**What is NOT here.** `head_rows` moved to
``operators/attention/gqa/_common.py`` when the operators were organised by
topology: concatenated-per-head row arithmetic is a GQA fact, and its only
consumers were the two grouped-head operators. What remains below is genuinely
kind-neutral — child construction, parameter-identity copying, and the one
activation-statistics pass that FFN, RESIDUAL_WIDTH and COMPOSITE_STAGE1 share.

Copying is by **parameter identity**, not by name: ``copy_block_except`` takes the
set of child parameters the operator will assign itself and carries the rest
across positionally. A name-based rule would put ``self_attn.q_proj.weight`` into
this file, which is exactly the family knowledge that belongs in an adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from aadistill.initialization.calibration.batching import (
    micro_batches,
    resolve_pad_id,
)

from aadistill.initialization.specs.arch import ArchitectureAdapter, ArchSpec


class SurgeryError(RuntimeError):
    """A child model was built incorrectly."""


class ChildBuilder:
    """A new model at ``new_spec``, with assignment coverage tracking."""

    def __init__(self, adapter: ArchitectureAdapter, parent: Any, new_spec: ArchSpec,
                 *, seed: int, dtype: Any = None) -> None:
        self.adapter = adapter
        self.parent = parent
        self.spec = new_spec
        self.dtype = dtype if dtype is not None else model_dtype(adapter, parent)
        config = adapter.build_config(parent.config, new_spec)
        self.model = adapter.build_model(config, self.dtype, seed)
        self._names = {id(p): n for n, p in self.model.named_parameters()}
        self._assigned: set[int] = set()

    @torch.no_grad()
    def assign(self, param: torch.nn.Parameter, value: torch.Tensor) -> None:
        if id(param) not in self._names:
            raise SurgeryError("assigned a tensor that is not a parameter of the child")
        if tuple(param.shape) != tuple(value.shape):
            raise SurgeryError(
                f"{self._names[id(param)]}: cannot assign {tuple(value.shape)} into "
                f"{tuple(param.shape)}")
        param.copy_(value.to(param.dtype))
        self._assigned.add(id(param))

    def finish(self) -> Any:
        """Tie weights, verify full coverage, return the child."""
        if bool(self.spec.get("tie_word_embeddings", False)):
            self.model.tie_weights()
            self._names = {id(p): n for n, p in self.model.named_parameters()}
        missing = sorted(n for i, n in self._names.items() if i not in self._assigned)
        if missing:
            raise SurgeryError(
                f"{len(missing)} child parameters were never assigned and still hold "
                f"random values: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        self.model.eval()
        return self.model


def model_dtype(adapter: ArchitectureAdapter, model: Any) -> Any:
    return adapter.embedding(model).weight.dtype


def copy_module_except(builder: ChildBuilder, src: Any, dst: Any,
                       skip: set[int] | None = None) -> None:
    """Carry every parameter of ``dst`` across from ``src`` by matching local name.

    ``skip`` holds ``id()`` of the child parameters the caller assigns itself.
    Shapes are checked by ``assign``, so a parameter the operator *should* have
    transformed but forgot to list fails loudly on the shape mismatch rather than
    silently receiving a wrong-sized copy.
    """
    skip = skip or set()
    source = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        if id(param) in skip:
            continue
        if name not in source:
            raise SurgeryError(f"child has parameter {name!r} with no counterpart in the parent")
        builder.assign(param, source[name])


def copy_embeddings_and_final_norm(builder: ChildBuilder, adapter: ArchitectureAdapter,
                                   parent: Any) -> None:
    builder.assign(adapter.embedding(builder.model).weight,
                   adapter.embedding(parent).weight)
    builder.assign(adapter.final_norm(builder.model).weight,
                   adapter.final_norm(parent).weight)


@torch.no_grad()
def collect_activation_stats(adapter: ArchitectureAdapter, model: Any,
                             token_batches, device: str = "cpu", *,
                             batch_size: int = 1,
                             pad_id: int | None = None) -> dict[str, torch.Tensor]:
    """Streaming sufficient statistics for the model **as it is now**.

    The whole reason width and FFN selection are re-run per state rather than
    computed once from the teacher: after a depth or attention operator the
    residual second moments and the FFN activation distribution are no longer the
    teacher's. E8a's central negative result — a full-width proxy mispredicting
    the compressed initializer — is what this re-collection is answering.

    This is the one forward loop FFN, RESIDUAL_WIDTH and COMPOSITE share, so
    micro-batching is implemented here once rather than three times.
    ``batch_size=1`` keeps the original one-item-per-forward path exactly,
    including calling ``process`` rather than ``process_batch``; anything larger
    pads groups of items together and the collector reduces over real tokens
    only.

    ``token_batches`` accepts either bare ``[1, T]`` id tensors (what the
    operators have always passed) or full calibration items; batching needs only
    the ids, and taking both means no call site has to change shape to opt in.
    """
    items = [it if isinstance(it, Mapping) else {"input_ids": it}
             for it in token_batches]
    collector = adapter.stats_collector(model)
    try:
        if batch_size <= 1:
            for item in items:
                collector.process(item["input_ids"].to(device))
        else:
            if not hasattr(collector, "process_batch"):
                raise TypeError(
                    f"{type(collector).__name__} has no `process_batch`, so it "
                    f"cannot honour batch_size={batch_size}. Pass batch_size=1 "
                    "for the per-item reference path, or give the collector a "
                    "batched entry point — do not let it silently pad.")
            resolved_pad = (resolve_pad_id(model) if pad_id is None else int(pad_id))
            for batch in micro_batches(items, batch_size, pad_id=resolved_pad,
                                       device=device):
                collector.process_batch(batch)
    finally:
        collector.close()
    return collector.state()
