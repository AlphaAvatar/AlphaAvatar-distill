"""RESIDUAL_WIDTH: the dimension of the stream every block reads and writes.

``width.global_pca_v0`` is the incumbent recipe's width half, exposed as an
operator. One global orthonormal ``P`` (d_parent x d_child) from the uncentered
second moments of the residual stream, so the skip path needs no change of basis
anywhere; stream readers become ``sqrt(d_p/d_c) * W diag(w_norm) P`` with the
folded norm set to ones; stream writers become ``P^T W``; the final norm is the
least-squares diagonal from ``init/project.py``. The algorithm is unchanged — the
primitives are imported from ``init.project`` and ``init.sandwich`` rather than
restated — but **what it is computed from** is not: the second moments come from
the checkpoint being transformed, not from the original teacher.

That is the whole point of making width an operator. Run last, it sees a stream
that depth, attention and FFN compression have already reshaped; run first, it
sees the teacher's. Those are different projections, and E8a is the evidence that
the difference is not negligible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...init.project import final_norm_weights, stream_projection
from ...init.sandwich import _in_proj as fold_and_project
from ..arch import ArchitectureAdapter, ArchSpec, Capability
from ..metrics import OperatorLocalMetrics
from ..device import model_device, stats_to
from ._common import ChildBuilder, collect_activation_stats
from .base import (
    CalibrationNeed,
    OperatorContext,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
)

WIDTH_FIELD = "hidden_size"


class WidthGlobalPCAV0(OperatorImplementation):
    impl_id = "width.global_pca_v0"
    kind = "RESIDUAL_WIDTH"
    version = 0
    description = (
        "One global orthonormal projection from the trace-normalized average of "
        "the uncentered residual second moments, with the two stream end points "
        "(embedding output, post-final-norm) upweighted 9/8 because they are the "
        "worst-captured and are the interfaces the tied embedding reads.")
    required_capabilities = frozenset({Capability.RESIDUAL_STREAM,
                                       Capability.PRENORM_BLOCKS,
                                       Capability.RMS_NORM,
                                       Capability.ACTIVATION_STATS})
    modifies = frozenset({WIDTH_FIELD})
    preserves = frozenset({"num_hidden_layers", "intermediate_size",
                           "num_attention_heads", "num_key_value_heads", "head_dim",
                           "vocab_size", "tie_word_embeddings"})
    calibration = CalibrationNeed.ACTIVATION_STATS
    objective = "captured energy fraction of the residual second moment"
    deterministic = True
    requires_seed = False
    produces = ("projection_diagnostics",)
    target_validation = "result hidden_size equals the target exactly"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        n_items = int((config or {}).get("n_calibration_items", 0))
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{WIDTH_FIELD: target[WIDTH_FIELD]}),
            forward_passes=0, stats_passes=max(n_items, 1),
            notes=("one statistics pass over the mixture, then an eigendecomposition "
                   f"of a {spec[WIDTH_FIELD]}x{spec[WIDTH_FIELD]} matrix"))

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        parent = ctx.model
        d_p = ctx.parent_spec[WIDTH_FIELD]
        d_c = ctx.target_spec[WIDTH_FIELD]
        n_layers = ctx.parent_spec["num_hidden_layers"]

        # Shared with `ffn.activation_importance_v0` when both expand the same
        # parent under the same profile: one pass, two operators. Never shared
        # across parents — the key includes the parent's artifact digest, and
        # re-collecting per state is the point of the architecture.
        # The cache is host-resident by contract (autoinit.device). `proj` is
        # derived from it and then multiplied against parent weights, so the
        # working copy is moved to the parent's ACTUAL device here — explicitly,
        # once, and freed when this call returns.
        compute = model_device(parent)
        state = stats_to(ctx.cached_stats(lambda: collect_activation_stats(
            adapter, parent, (i["input_ids"] for i in ctx.calibration_items),
            compute)), compute)

        # Same point set and weights as the incumbent recipe: all pre-norm stream
        # states plus the post-final-norm point, ends upweighted 9/8.
        points = list(range(n_layers + 1))
        weights = [9.0] + [1.0] * (n_layers - 1) + [8.0]
        proj, proj_diag = stream_projection(state, d_c, points, weights)
        proj_diag["point_weights"] = weights
        scale = (d_p / d_c) ** 0.5

        new_spec = ctx.parent_spec.replace(**{WIDTH_FIELD: d_c})
        builder = ChildBuilder(adapter, parent, new_spec, seed=ctx.seed)

        for src, dst in zip(adapter.blocks(parent), adapter.blocks(builder.model)):
            src_in = adapter.stream_in_projections(src)
            dst_in = adapter.stream_in_projections(dst)
            for role, (linear, norm) in src_in.items():
                builder.assign(dst_in[role][0].weight,
                               fold_and_project(linear.weight, norm.weight, proj, scale))
            src_out = adapter.stream_out_projections(src)
            dst_out = adapter.stream_out_projections(dst)
            for role, linear in src_out.items():
                builder.assign(dst_out[role].weight,
                               proj.T @ linear.weight.to(torch.float64))
            # The folded norms become identity; RMSNorm is scale-invariant, so the
            # residual error after this operator is directional, not a scale.
            builder.assign(adapter.attn_norm(dst).weight,
                           torch.ones_like(adapter.attn_norm(dst).weight))
            builder.assign(adapter.ffn_norm(dst).weight,
                           torch.ones_like(adapter.ffn_norm(dst).weight))
            s_attn, d_attn = adapter.attention(src), adapter.attention(dst)
            builder.assign(d_attn.q_norm.weight, s_attn.q_norm.weight)
            builder.assign(d_attn.k_norm.weight, s_attn.k_norm.weight)

        builder.assign(adapter.embedding(builder.model).weight,
                       adapter.embedding(parent).weight.to(torch.float64) @ proj)
        w_final = final_norm_weights(state, proj, adapter.final_norm(parent).weight,
                                     post_norm_point=n_layers)
        builder.assign(adapter.final_norm(builder.model).weight, scale * w_final)
        child = builder.finish()

        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="parent_state",
                values={
                    "op.width.energy_captured_frac": float(proj_diag["energy_captured_frac"]),
                    "op.width.orthonormality_error": float(proj_diag["orthonormality_error"]),
                    "op.width.min_kept_eigenvalue": float(proj_diag["min_kept_eigenvalue"]),
                },
                detail={"stats_tokens": int(state["residual_count"][0])}),
            trace={"source": "global_activation_pca", "scale_compensation": scale,
                   "d_parent": d_p, "d_child": d_c},
            artifacts={"projection_diagnostics": {
                k: v for k, v in proj_diag.items() if k != "points"}},
        )


WIDTH_GLOBAL_PCA_V0 = register_implementation(WidthGlobalPCAV0())
