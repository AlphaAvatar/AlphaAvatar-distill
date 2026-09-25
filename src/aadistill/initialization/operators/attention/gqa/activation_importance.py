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

**Why this lives in its own module.** An implementation joins the library by
registering, never by being added to an existing operator module: editing one
would move the digest of any declared executable source set that names it, and
a frozen historical document must keep describing the code that actually ran.
The extension route is the one `operators/__init__` documents — "an
implementation defined elsewhere joins by calling ``register_implementation``".
The specific set and digest this avoided are recorded in
`docs/core-provenance.md`.

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

from aadistill.initialization.operators.attention.gqa._statistics import (
    AttentionHeadStatsCollector,
    head_write_energy,
)
from aadistill.initialization.specs.arch import (
    ArchitectureAdapter,
    ArchSpec,
    Capability,
    UnsupportedCapability,
)
from aadistill.initialization.device import model_device, stats_to
from aadistill.initialization.specs.metrics import OperatorLocalMetrics
from aadistill.initialization.statistics.spec import StatsSpec
from aadistill.initialization.operators._common import (
    ChildBuilder,
    copy_embeddings_and_final_norm,
    copy_module_except,
)
from aadistill.initialization.operators.attention.gqa._common import (
    attention_out_projection,
    head_rows,
    query_projection,
    refuse_unless_reducible_within_groups,
    select_q_heads_by_score,
)
from aadistill.initialization.calibration.batching import (
    micro_batches,
    resolve_pad_id,
)
from aadistill.initialization.calibration.profiles import CalibrationNeed
from aadistill.initialization.operators.base import (
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
        n_q, n_kv, _ = adapter.head_groups(spec)
        #: The grouped-head reduction rule, shared with every other algorithm of
        #: this topology rather than restated per operator — two copies of an
        #: applicability rule is two chances for them to disagree about which
        #: geometries an operator may run on.
        return refuse_unless_reducible_within_groups(n_q, n_kv,
                                                     target[HEADS_FIELD])

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
        #: Resolved through the adapter, in block order, and handed to the
        #: collector. The collector does not look for them.
        out_projections = [attention_out_projection(adapter, b)
                           for b in adapter.blocks(parent)]
        collector = AttentionHeadStatsCollector(parent, out_projections,
                                                num_heads=n_q, head_dim=head_dim)
        #: Micro-batched, and `1` is the per-item reference path — it calls
        #: `process` exactly as this loop always did, so the frozen C1 selection
        #: is reproduced by construction rather than by tolerance. Larger sizes
        #: pad groups together and the collector's mask keeps padded positions
        #: out of `M_h` and out of `attn_token_count`.
        batch_size = ctx.execution.micro_batch_size
        try:
            if batch_size <= 1:
                for item in ctx.calibration_items:
                    collector.process(item["input_ids"].to(compute))
            else:
                for batch in micro_batches(ctx.calibration_items, batch_size,
                                           pad_id=resolve_pad_id(parent),
                                           device=compute):
                    collector.process_batch(batch)
        finally:
            collector.close()
        #: THE TRANSFER BOUNDARY, and a defect this project has already paid
        #: for once (`docs/core-provenance.md`).
        #:
        #: `state()` returns a HOST-RESIDENT snapshot on purpose — that is the
        #: evidence/cache form, and it is what gets hashed and kept. Handing it
        #: straight to `head_write_energy` meets
        #: `o_proj.weight` on cuda:0: `RuntimeError: Expected all tensors to be
        #: on the same device`. Nothing about the persistent cache POLICY was
        #: wrong; what was missing was the per-invocation working copy.
        #:
        #: So: snapshot to the host, release the collector's device accumulator,
        #: and only then build ONE working copy on the compute device. In that
        #: order there are never three copies of the (layers, heads, d, d)
        #: float64 tensor alive at once.
        host_stats = collector.state()
        collector.release()
        stats = stats_to(host_stats, compute)

        new_spec = ctx.parent_spec.replace(**{HEADS_FIELD: keep_q})
        builder = ChildBuilder(adapter, parent, new_spec, seed=ctx.seed)

        retained, kept_per_layer = [], []
        for idx, (src, dst) in enumerate(zip(adapter.blocks(parent),
                                             adapter.blocks(builder.model))):
            s_out, d_out = (attention_out_projection(adapter, src),
                            attention_out_projection(adapter, dst))
            s_q, d_q = query_projection(adapter, src), query_projection(adapter, dst)
            scores = head_write_energy(stats, idx, s_out.weight, n_q, head_dim)
            kept = select_q_heads_by_score(scores, n_q, n_kv, keep_q)
            rows = head_rows(kept, head_dim, device=s_q.weight.device)

            total = float(scores.sum())
            retained.append(float(scores[kept].sum() / total) if total > 0 else 0.0)
            kept_per_layer.append(list(kept))

            transformed = {id(d_q.weight), id(d_out.weight)}
            builder.assign(d_q.weight, s_q.weight[rows])
            builder.assign(d_out.weight, s_out.weight[:, rows])
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
                   # Execution evidence, not identity: `OperatorStep.identity()`
                   # does not read `trace`, and the batch size changes no
                   # estimand. Recorded so a run's evidence states how it ran.
                   "micro_batch_size": batch_size,
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
    from aadistill.initialization.operators.base import unregister_implementation

    unregister_implementation(ATTENTION_ACTIVATION_IMPORTANCE_V1.impl_id)
