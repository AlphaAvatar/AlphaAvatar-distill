"""ATTENTION: head structure.

``attention.weight_proxy_v0`` keeps, within each GQA group, the query heads with
the largest ``||W_q rows of h||_F * ||W_o columns of h||_F`` — how strongly the
head can form queries from the (norm-folded) stream and write its result back.
Grouping is preserved, so the KV heads and the RoPE basis are untouched and the
result is still a valid GQA model.

This is a **weight** proxy, not an activation one, and it is registered under
that name for exactly that reason. ``init/sandwich.py`` records the gap: activation
based head importance would need attention hooks Stage 0 never cached. A future
``attention.activation_importance_v1`` or ``attention.causal_kl_v1`` registers a
new id beside this one; this id's meaning never changes.

Because the kind is family-generic and the implementation is not, an MLA or
linear-attention family registers its own ATTENTION implementation with its own
required capabilities and the search core is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...init.sandwich import select_q_heads
from ..arch import ArchitectureAdapter, ArchSpec, Capability
from ..metrics import OperatorLocalMetrics
from ._common import (
    ChildBuilder,
    copy_embeddings_and_final_norm,
    copy_module_except,
    head_rows,
)
from .base import (
    CalibrationNeed,
    OperatorContext,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
)

HEADS_FIELD = "num_attention_heads"


class AttentionWeightProxyV0(OperatorImplementation):
    impl_id = "attention.weight_proxy_v0"
    kind = "ATTENTION"
    version = 0
    description = (
        "Per-GQA-group query-head selection by ||W_q rows||_F * ||W_o columns||_F, "
        "with the preceding norm folded into W_q before scoring. KV heads, head_dim "
        "and grouping are preserved.")
    required_capabilities = frozenset({Capability.ATTENTION_GQA, Capability.RMS_NORM})
    modifies = frozenset({HEADS_FIELD})
    preserves = frozenset({"hidden_size", "num_hidden_layers", "intermediate_size",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.NONE
    objective = "retained share of total head weight-proxy score"
    deterministic = True
    requires_seed = False
    produces = ("kept_heads",)
    target_validation = ("result num_attention_heads equals the target exactly and "
                         "stays divisible by the unchanged KV head count")

    def applicable(self, spec: ArchSpec, target: ArchSpec,
                   adapter: ArchitectureAdapter) -> tuple[bool, str]:
        ok, reason = super().applicable(spec, target, adapter)
        if not ok:
            return ok, reason
        n_kv = spec["num_key_value_heads"]
        if target[HEADS_FIELD] % n_kv:
            return False, (f"target {target[HEADS_FIELD]} query heads is not divisible "
                           f"by {n_kv} KV heads")
        if target[HEADS_FIELD] > spec[HEADS_FIELD]:
            return False, "cannot add query heads"
        return True, "ok"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{HEADS_FIELD: target[HEADS_FIELD]}),
            forward_passes=0, stats_passes=0,
            notes="weight-only proxy; no calibration forward pass")

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        parent = ctx.model
        n_q, n_kv, head_dim = adapter.head_groups(ctx.parent_spec)
        keep_q = ctx.target_spec[HEADS_FIELD]

        new_spec = ctx.parent_spec.replace(**{HEADS_FIELD: keep_q})
        builder = ChildBuilder(adapter, parent, new_spec, seed=ctx.seed)

        retained, kept_per_layer = [], []
        for src, dst in zip(adapter.blocks(parent), adapter.blocks(builder.model)):
            s_attn, d_attn = adapter.attention(src), adapter.attention(dst)
            norm_w = adapter.attn_norm(src).weight.to(torch.float64)
            q64 = s_attn.q_proj.weight.to(torch.float64)
            o64 = s_attn.o_proj.weight.to(torch.float64)
            kept = select_q_heads(q64 * norm_w[None, :], o64, n_q, n_kv, keep_q, head_dim)
            rows = head_rows(kept, head_dim,
                             device=s_attn.q_proj.weight.device)

            # A host diagnostic, deliberately: these are reduced to Python
            # floats immediately below and never meet a parameter again.
            scores = torch.tensor([
                float(q64[h * head_dim:(h + 1) * head_dim, :].norm()
                      * o64[:, h * head_dim:(h + 1) * head_dim].norm())
                for h in range(n_q)])
            retained.append(float(scores[kept].sum() / scores.sum()))
            kept_per_layer.append(list(kept))

            transformed = {id(d_attn.q_proj.weight), id(d_attn.o_proj.weight)}
            builder.assign(d_attn.q_proj.weight, s_attn.q_proj.weight[rows])
            builder.assign(d_attn.o_proj.weight, s_attn.o_proj.weight[:, rows])
            copy_module_except(builder, src, dst, skip=transformed)

        copy_embeddings_and_final_norm(builder, adapter, parent)
        child = builder.finish()

        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="parent_state",
                values={
                    "op.attention.retained_score_mean": sum(retained) / len(retained),
                    "op.attention.retained_score_min": min(retained),
                },
                detail={"per_layer_retained_share": retained}),
            trace={"source": "weight_proxy_per_group_topk",
                   "q_heads": [n_q, keep_q], "kv_heads": n_kv},
            artifacts={"kept_heads": kept_per_layer},
        )


ATTENTION_WEIGHT_PROXY_V0 = register_implementation(AttentionWeightProxyV0())
