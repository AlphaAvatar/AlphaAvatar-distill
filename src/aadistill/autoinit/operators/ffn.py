"""FFN: the per-block feed-forward intermediate width.

``ffn.activation_importance_v0`` keeps the top-k intermediate neurons by
``E[|a_j|] * ||down_proj column j||_2`` — how much the neuron fires on the
calibration mixture, weighted by how strongly it can write to the residual
stream. Both halves matter: a loud neuron whose output column is small barely
moves the stream, and a quiet one with a large column can.

Selection is per block against **this** checkpoint's activations. Running this
operator after a width or attention operator therefore selects different neurons
than running it first, which is one of the order effects the search is built to
measure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...init.project import ffn_neuron_importance
from ..arch import ArchitectureAdapter, ArchSpec, Capability
from ..metrics import OperatorLocalMetrics
from ._common import (
    ChildBuilder,
    collect_activation_stats,
    copy_embeddings_and_final_norm,
    copy_module_except,
)
from .base import (
    CalibrationNeed,
    OperatorContext,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
)

FFN_FIELD = "intermediate_size"


class FFNActivationImportanceV0(OperatorImplementation):
    impl_id = "ffn.activation_importance_v0"
    kind = "FFN"
    version = 0
    description = (
        "Per-block top-k intermediate neurons by E[|a|] * ||down column||, "
        "measured on the checkpoint being transformed. Rows of the gate and up "
        "projections and columns of the down projection are selected together, so "
        "the surviving neurons keep their own complete circuit.")
    required_capabilities = frozenset({Capability.DENSE_FFN, Capability.ACTIVATION_STATS})
    modifies = frozenset({FFN_FIELD})
    preserves = frozenset({"hidden_size", "num_hidden_layers", "num_attention_heads",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.ACTIVATION_STATS
    objective = "retained share of total activation importance"
    deterministic = True
    requires_seed = False
    produces = ("kept_neurons",)
    target_validation = "result intermediate_size equals the target exactly"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        n_items = int((config or {}).get("n_calibration_items", 0))
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{FFN_FIELD: target[FFN_FIELD]}),
            forward_passes=0, stats_passes=max(n_items, 1),
            notes="one statistics pass; selection is a per-block top-k")

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        parent = ctx.model
        keep = ctx.target_spec[FFN_FIELD]

        state = collect_activation_stats(
            adapter, parent, (i["input_ids"] for i in ctx.calibration_items), ctx.device)

        new_spec = ctx.parent_spec.replace(**{FFN_FIELD: keep})
        builder = ChildBuilder(adapter, parent, new_spec, seed=ctx.seed)

        retained_shares, kept_per_layer = [], []
        for idx, (src, dst) in enumerate(zip(adapter.blocks(parent),
                                             adapter.blocks(builder.model))):
            src_ffn, dst_ffn = adapter.ffn(src), adapter.ffn(dst)
            importance = ffn_neuron_importance(state, idx, src_ffn.down_proj.weight)
            kept = torch.topk(importance, keep).indices.sort().values
            retained_shares.append(float(importance[kept].sum() / importance.sum()))
            kept_per_layer.append(kept.tolist())

            src_in = adapter.stream_in_projections(src)
            dst_in = adapter.stream_in_projections(dst)
            transformed = set()
            for role in ("gate", "up"):
                target_w = dst_in[role][0].weight
                builder.assign(target_w, src_in[role][0].weight[kept])
                transformed.add(id(target_w))
            down_dst = adapter.stream_out_projections(dst)["ffn_out"]
            down_src = adapter.stream_out_projections(src)["ffn_out"]
            builder.assign(down_dst.weight, down_src.weight[:, kept])
            transformed.add(id(down_dst.weight))
            copy_module_except(builder, src, dst, skip=transformed)

        copy_embeddings_and_final_norm(builder, adapter, parent)
        child = builder.finish()

        mean_share = sum(retained_shares) / len(retained_shares)
        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="parent_state",
                values={
                    "op.ffn.retained_importance_mean": mean_share,
                    "op.ffn.retained_importance_min": min(retained_shares),
                },
                detail={"per_layer_retained_share": retained_shares}),
            trace={"source": "activation_importance_topk",
                   "kept_fraction": keep / ctx.parent_spec[FFN_FIELD]},
            artifacts={"kept_neurons": kept_per_layer},
        )


FFN_ACTIVATION_IMPORTANCE_V0 = register_implementation(FFNActivationImportanceV0())
