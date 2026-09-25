"""What both DEPTH algorithms share.

The kind's two implementations disagree about *which* blocks to keep and agree
completely about what keeping them means — so the block-copy lives here and the
choice lives in each algorithm's own module. There is no ``standard/`` topology
level under DEPTH: removing a decoder block is the same structural operation
whatever the attention or FFN topology inside it is, so a topology directory
would be symmetry with nothing behind it.
"""

from __future__ import annotations

from typing import Any

from aadistill.initialization.operators._common import (
    ChildBuilder,
    copy_embeddings_and_final_norm,
    copy_module_except,
)
from aadistill.initialization.operators.base import OperatorContext

DEPTH_FIELD = "num_hidden_layers"


def _build_child_with_layers(ctx: OperatorContext, kept: list[int]) -> Any:
    """A child holding exactly the parent blocks ``kept``, in order, verbatim."""
    adapter = ctx.adapter
    new_spec = ctx.parent_spec.replace(**{DEPTH_FIELD: len(kept)})
    builder = ChildBuilder(adapter, ctx.model, new_spec, seed=ctx.seed)
    parent_blocks = adapter.blocks(ctx.model)
    child_blocks = adapter.blocks(builder.model)
    for dst, src_idx in zip(child_blocks, kept):
        copy_module_except(builder, parent_blocks[src_idx], dst)
    copy_embeddings_and_final_norm(builder, adapter, ctx.model)
    return builder.finish()



