"""ATTENTION by realized residual-write energy — the Phase-C1 replacement.

`attention.weight_proxy_v0` asks how strongly a head *could* act, from weight
norms alone. This asks how strongly it *does* act on the frozen calibration
distribution:

    a_h(t) = head h's slice of the concatenated attention output
    z_h(t) = W_o,h @ a_h(t)
    score_h = mean_t ||z_h(t)||^2

scored **exactly** — not approximated — by contracting o_proj's per-head Gram
matrix with the streamed per-head second moment (`init.attention_stats`).

The selection topology is deliberately identical to `weight_proxy_v0`: same
per-GQA-group retention, same deterministic tie-break, same weight slicing, same
`modifies`/`preserves` sets. Only the importance signal moves. That is what makes
Phase C1 an isolation test rather than two changes at once.

**Why this lives in its own module.** `operators/attention.py` and
`operators/__init__.py` are both members of `CONTINUATION_SOURCE_FILES_V2`, the
executable source set that Phase B's closed preregistration binds to digest
`a5ce6311789e…`. Adding a class to either file moves that digest, which would
leave a frozen historical document describing code that did not exist when it
ran — and the fix for that is never to regenerate the document. A new module is
not in the declared set, so the Phase-A/B executable identity is untouched. This
is exactly the extension route `operators/__init__` documents: "an implementation
defined elsewhere joins by calling ``register_implementation`` — no edit here,
and none to the search engine."

**Registration is an explicit call, not an import side effect.** The first
version of this module registered at import, and the full suite caught what that
means: `BeamSearch._allowed_impl_ids` falls back to *every registered
implementation* when `SearchConfig.allowed_impls` is None, so merely importing
the module added a calibrated ATTENTION branch to an unrelated search and broke
`test_two_profiles_do_not_duplicate_the_weight_proxy_expansion` (10 expansions
became 12). Keeping the operator out of `V1_IMPLEMENTATIONS` is therefore *not*
sufficient on its own.

So importing this module is inert. A consumer that wants the operator calls
`register()`, and a future beam acquires it by a decision rather than by an
import anywhere in the process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ...init.attention_stats import AttentionHeadStatsCollector, head_write_energy
from ..arch import ArchitectureAdapter, ArchSpec, Capability
from ..device import model_device
from ..metrics import OperatorLocalMetrics
from ..stats import StatsSpec
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

#: What this operator collects. A DISTINCT spec id and version from
#: `stats.DEFAULT_STATS_SPEC`, so its hash — and any statistics cache key derived
#: from it — cannot collide with the residual/FFN statistics that
#: `ffn.activation_importance_v0` and `width.global_pca_v0` share. The quantities
#: are different tensors entirely; a shared key would hand one operator another's
#: state.
ATTENTION_STATS_SPEC = StatsSpec(
    spec_id="attention_head_output_second_moment",
    version=1,
    accumulation_dtype="float64",
    quantities=("attn_head_sqsum", "attn_token_count"),
)


def select_q_heads_by_score(scores: Sequence[float] | torch.Tensor, n_q_heads: int,
                            n_kv_heads: int, keep_q: int) -> list[int]:
    """Per-GQA-group top-k by a precomputed score. Deterministic.

    Same grouping and same retention arithmetic as
    ``init.sandwich.select_q_heads`` — only the score differs — so the two
    ATTENTION implementations share a selection topology and the C1 contrast is
    the importance signal alone.

    Ties break by **ascending head index**, stated rather than inherited from
    sort stability, because a silent tie-break is exactly what makes a replay
    irreproducible.
    """
    if n_q_heads % n_kv_heads or keep_q % n_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads (GQA grouping)")
    per_g_t, per_g_s = n_q_heads // n_kv_heads, keep_q // n_kv_heads
    if per_g_s > per_g_t:
        raise ValueError(f"cannot keep {per_g_s} of {per_g_t} Q heads per group")
    kept: list[int] = []
    for g in range(n_kv_heads):
        group = range(g * per_g_t, (g + 1) * per_g_t)
        top = sorted(group, key=lambda h: (-float(scores[h]), h))[:per_g_s]
        kept.extend(sorted(top))
    return kept


class AttentionActivationImportanceV1(OperatorImplementation):
    impl_id = "attention.activation_importance_v1"
    kind = "ATTENTION"
    version = 1
    description = (
        "Per-GQA-group query-head selection by mean squared residual-write energy "
        "mean_t ||W_o,h a_h(t)||^2, measured on the checkpoint being transformed "
        "over the operator's calibration profile. KV heads, head_dim, GQA grouping "
        "and the RoPE basis are preserved.")
    required_capabilities = frozenset({Capability.ATTENTION_GQA, Capability.RMS_NORM,
                                       Capability.ACTIVATION_STATS})
    modifies = frozenset({HEADS_FIELD})
    preserves = frozenset({"hidden_size", "num_hidden_layers", "intermediate_size",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.ACTIVATION_STATS
    objective = "retained share of realized attention-output write energy"
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
        n_items = int((config or {}).get("n_calibration_items", 0))
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{HEADS_FIELD: target[HEADS_FIELD]}),
            forward_passes=0, stats_passes=max(n_items, 1),
            notes=("one attention-statistics pass over the calibration mixture; "
                   "selection is a per-GQA-group top-k"))

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        parent = ctx.model
        n_q, n_kv, head_dim = adapter.head_groups(ctx.parent_spec)
        keep_q = ctx.target_spec[HEADS_FIELD]

        # Collected directly, NOT through `ctx.cached_stats`: that cache is keyed
        # on the caller's StatsSpec, and these quantities belong to
        # ATTENTION_STATS_SPEC. Sharing the key would hand this operator the
        # residual/FFN state. There is also nothing to share — within a fixed
        # path ATTENTION runs once, and the other ATTENTION implementation reads
        # no calibration at all.
        compute = model_device(parent)
        collector = AttentionHeadStatsCollector(parent, num_heads=n_q,
                                                head_dim=head_dim)
        try:
            for item in ctx.calibration_items:
                collector.process(item["input_ids"].to(compute))
        finally:
            collector.close()
        stats = collector.state()

        new_spec = ctx.parent_spec.replace(**{HEADS_FIELD: keep_q})
        builder = ChildBuilder(adapter, parent, new_spec, seed=ctx.seed)

        retained, kept_per_layer = [], []
        for idx, (src, dst) in enumerate(zip(adapter.blocks(parent),
                                             adapter.blocks(builder.model))):
            s_attn, d_attn = adapter.attention(src), adapter.attention(dst)
            scores = head_write_energy(stats, idx, s_attn.o_proj.weight, n_q, head_dim)
            kept = select_q_heads_by_score(scores, n_q, n_kv, keep_q)
            rows = head_rows(kept, head_dim, device=s_attn.q_proj.weight.device)

            total = float(scores.sum())
            retained.append(float(scores[kept].sum() / total) if total > 0 else 0.0)
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
                    "op.attention.retained_write_energy_mean":
                        sum(retained) / len(retained),
                    "op.attention.retained_write_energy_min": min(retained),
                },
                detail={"per_layer_retained_share": retained}),
            trace={"source": "activation_write_energy_per_group_topk",
                   "score": "mean_t ||W_o,h a_h(t)||^2",
                   "stats_spec": ATTENTION_STATS_SPEC.spec_hash,
                   "calibration_tokens": int(stats["attn_token_count"]),
                   "q_heads": [n_q, keep_q], "kv_heads": n_kv},
            artifacts={"kept_heads": kept_per_layer},
        )


#: The singleton, **unregistered**. Import is inert; see the module docstring.
ATTENTION_ACTIVATION_IMPORTANCE_V1 = AttentionActivationImportanceV1()


def register(*, replace: bool = False) -> OperatorImplementation:
    """Join the global operator registry. Idempotent for an unchanged signature.

    Explicit because the registry is what an unrestricted `BeamSearch` enumerates:
    registering at import would put this operator into every search in the
    process, including ones that never asked for it.
    """
    return register_implementation(ATTENTION_ACTIVATION_IMPORTANCE_V1,
                                   replace=replace)


def unregister() -> None:
    """Leave the registry again — for tests, so registration cannot leak."""
    from .base import unregister_implementation

    unregister_implementation(ATTENTION_ACTIVATION_IMPORTANCE_V1.impl_id)
