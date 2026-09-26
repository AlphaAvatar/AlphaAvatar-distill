"""ATTENTION by one-shot causal head ablation — the C3 candidate.

The two shipped ATTENTION operators score a head by a **proxy** for what it
does: `weight_proxy_v0` from weight norms alone, `activation_importance_v1`
from realized residual-write energy. Both answer "how loudly does this head
write?", which is not the question the model is compressed against. This one
asks the causal question directly, the way `depth.causal_kl_greedy_v1` asks it
of a block:

    score_{l,h} = forward KL(parent || parent with head h of layer l ablated)

aggregated exactly as DEPTH aggregates it — the unweighted mean over domains of
the unweighted mean over each domain's sub-types — so a long-tokenizing subtype
cannot dominate. A head whose removal moves the output distribution most is the
head most worth keeping, and per GQA group the top `keep_q // n_kv` survive.

**ONE SHOT, not greedy.** Every head is scored against the intact parent and
nothing is rescored after a removal. DEPTH is greedy because removing a block
changes what the remaining blocks see enough that stale scores mislead; this
operator deliberately does not make that claim about heads, because a greedy
head search is `n_q` times more expensive per layer and no evidence yet says
the interaction term is worth it. The cost difference is the whole reason this
is a separate implementation rather than a mode of a joint search: 897 corpus
passes against the tens of thousands a greedy head search would take.

**It mutates nothing about the model it is handed — not even `use_cache`.**
`depth.causal_kl_greedy_v1` sets `model.config.use_cache = False`, and that
assignment reaches the CHILD's `config.json` and therefore its
`config_sha256`: the checkpoint's identity depends on whether that operator
ran. This one passes `use_cache=False` per forward instead. Same saved memory,
no config mutation, and one fewer way for an identity to depend on execution
history. (DEPTH is not changed here; it is frozen, and a historical identity
that already depends on this is not repaired by moving it now.)

**Ablation is zeroing, and that is exact.** A head's entire contribution to the
residual stream is `W_o[:, h*d:(h+1)*d] @ a_h(t)`, so zeroing that column block
removes the head and changes nothing else — no hooks, no surgery, no rebuilt
module. It was measured bit-for-bit identical to materializing a model with the
head really deleted. The columns are restored in a `finally`, because a scorer
that leaves the model it was handed damaged would corrupt every later step of
the path rather than fail.

**The batch size is CONFIG, not runtime.** Every other operator reads
`ctx.execution.micro_batch_size`, which is deliberately runtime-only and
deliberately not hashed. This one reads `calibration_forward_batch_size` from
its own step config instead, and the difference is not stylistic: padded
batching is MEASURED to move this family of decisions, so for this operator the
batch size is part of what the result means and must therefore be part of what
its identity commits to. A step that does not declare it gets `1`, the
per-item reference path — never the process-wide default, which would make the
result depend on an unhashed global.

**Registration is an explicit call.** Importing this module is inert, for the
reason `activation_importance.py` records: `BeamSearch._allowed_impl_ids` falls
back to every registered implementation when `allowed_impls` is None, so
registering at import would add a branch to searches that never asked for one.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from aadistill.initialization.calibration.batching import (
    micro_batches,
    resolve_pad_id,
)
from aadistill.initialization.calibration.profiles import CalibrationNeed
from aadistill.initialization.device import model_device
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
from aadistill.initialization.operators.base import (
    OperatorContext,
    OperatorError,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
)
from aadistill.initialization.specs.arch import (
    ArchitectureAdapter,
    ArchSpec,
    Capability,
)
from aadistill.initialization.specs.metrics import OperatorLocalMetrics
from aadistill.initialization.statistics.contribution import (
    domain_balanced_score,
    forward_kl_mean,
    forward_kl_mean_batch,
)

HEADS_FIELD = "num_attention_heads"

#: The step-config key that carries the numerical protocol into the identity.
#: Named here, in the operator that reads it, so the pilot's constant and the
#: executor's merge cannot disagree about its spelling.
BATCH_SIZE_CONFIG_KEY = "calibration_forward_batch_size"


def domain_subtype_map(items: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """domain -> its sub-types, both sorted, for `domain_balanced_score`.

    `depth.causal_kl_greedy_v1` derives the same map from the same fields with
    a private copy of these seven lines. Consolidating them means editing a
    module that two declared frozen source sets pin by sha256, which would move
    a historical executable identity for a refactor — so the duplication is
    deliberate and bounded, and a test asserts the two agree on the real frozen
    mixture rather than trusting that they look alike.
    """
    out: dict[str, list[str]] = {}
    for item in items:
        subs = out.setdefault(item["domain"], [])
        if item["subtype"] not in subs:
            subs.append(item["subtype"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def resolve_forward_batch_size(config: Mapping[str, Any] | None) -> int:
    """The declared batch size, defaulting to the per-item reference path.

    Deliberately NOT `ctx.execution.micro_batch_size`. See the module docstring:
    for this operator the batch size changes what the result means, so it is
    read from the step's own hashed config and from nowhere else.
    """
    value = (config or {}).get(BATCH_SIZE_CONFIG_KEY, 1)
    size = int(value)
    if size < 1:
        raise OperatorError(
            f"{BATCH_SIZE_CONFIG_KEY} must be >= 1, got {value!r}")
    return size


@contextmanager
def head_ablated(out_projection: Any, head: int, head_dim: int):
    """Zero one query head's residual-write columns, and always put them back.

    Equivalent to deleting the head: its output reaches the residual stream
    only through these columns. Restored in a `finally` so a failure anywhere
    inside the scoring loop cannot hand a silently damaged model to the next
    operator in the path.
    """
    cols = slice(head * head_dim, (head + 1) * head_dim)
    saved = out_projection.weight[:, cols].detach().clone()
    try:
        with torch.no_grad():
            out_projection.weight[:, cols] = 0
        yield
    finally:
        with torch.no_grad():
            out_projection.weight[:, cols] = saved


@torch.no_grad()
def _forward_block(model, batch, device: str) -> torch.Tensor:
    """One micro-batch's prediction-position logits as a `[B, T_pred, V]` block.

    The mask is passed for the same reason the DEPTH path passes it: under
    right padding causality already keeps a real token from reaching a pad, so
    it is defensive rather than load-bearing, and it becomes load-bearing the
    instant anything pads on the other side.
    """
    ids = batch.input_ids.to(device)
    mask = batch.attention_mask.to(device)
    return model(ids, attention_mask=mask, use_cache=False).logits[:, :-1]


@torch.no_grad()
def _forward_item(model, item, device: str) -> torch.Tensor:
    """One item's prediction-position logits, `[T-1, V]`, left on the device."""
    return model(item["input_ids"].to(device), use_cache=False).logits[0, :-1]


class AttentionCausalKLV1(OperatorImplementation):
    impl_id = "attention.causal_kl_v1"
    kind = "ATTENTION"
    version = 1
    description = (
        "Per-GQA-group query-head selection by one-shot causal ablation: each "
        "head is scored by forward KL(parent || parent with that head's "
        "residual-write zeroed) over real prediction positions, aggregated as "
        "the unweighted mean over domains of the unweighted mean over each "
        "domain's sub-types. No rescoring and no joint search. KV heads, "
        "head_dim, GQA grouping and the RoPE basis are preserved.")
    required_capabilities = frozenset({Capability.ATTENTION_GQA,
                                       Capability.RMS_NORM,
                                       Capability.LOGIT_COMPARABLE})
    modifies = frozenset({HEADS_FIELD})
    preserves = frozenset({"hidden_size", "num_hidden_layers", "intermediate_size",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.FORWARD_LOGITS
    objective = "forward KL(parent || parent with head ablated), domain-balanced"
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
        return refuse_unless_reducible_within_groups(n_q, n_kv,
                                                     target[HEADS_FIELD])

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        """ALGORITHMIC work, identical in both arms. Not physical invocations.

        `forward_passes` is corpus-equivalent **item** forwards: one ablated
        pass per (layer, head) over the whole mixture, plus one intact
        reference pass over the whole mixture. Batching changes how those
        item-forwards are packed into kernel launches; it changes neither the
        arithmetic that must happen nor the tokens it happens over, so a B4
        plan that priced at a quarter of B1 would be asserting the speedup this
        pilot exists to MEASURE. The two arms therefore price identically and
        the comparison is made on observed wall-clock.
        """
        n_q, n_kv, _ = adapter.head_groups(spec)
        keep_q = target[HEADS_FIELD]
        if keep_q >= n_q:
            raise OperatorError(
                f"{self.impl_id}: nothing to remove ({n_q} -> {keep_q})")
        n_layers = spec["num_hidden_layers"]
        ablations = n_layers * n_q
        n_items = int((config or {}).get("n_calibration_items", 0))
        per_pass = max(n_items, 1)
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{HEADS_FIELD: keep_q}),
            # +1 for the intact-parent reference pass over the mixture.
            forward_passes=(ablations + 1) * per_pass,
            stats_passes=0,
            notes=(f"{ablations} single-head ablations ({n_layers} layers x "
                   f"{n_q} heads) plus one reference pass, each over "
                   f"{n_items} calibration items"))

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        model = ctx.model
        items = list(ctx.calibration_items)

        n_q, n_kv, head_dim = adapter.head_groups(ctx.parent_spec)
        keep_q = ctx.target_spec[HEADS_FIELD]
        compute = model_device(model)
        domains = domain_subtype_map(items)
        batch_size = resolve_forward_batch_size(ctx.config)

        #: Built ONCE and reused for every ablation. Two reasons, the same two
        #: the DEPTH search has: the padded id tensors are identical for all
        #: 896 of them, and a grouping that could differ between ablations
        #: would make one head's score depend on something other than which
        #: head was zeroed.
        groups = list(micro_batches(
            items, batch_size,
            pad_id=(resolve_pad_id(model) if batch_size > 1 else 0),
            device=compute))

        blocks = list(adapter.blocks(model))
        out_projections = [attention_out_projection(adapter, b) for b in blocks]

        #: (layer, head) -> subtype -> per-item KLs, appended in group order.
        per_head: list[list[dict[str, list[float]]]] = [
            [{} for _ in range(n_q)] for _ in blocks]

        #: GROUP OUTER, ABLATION INNER — and this ordering is why the operator
        #: is affordable. The reference is the same tensor for every ablation
        #: of a given group, so computing it once per group turns
        #: `2 * ablations` corpus passes into `ablations + 1`. It does not
        #: touch the estimand: for a fixed (layer, head) the per-item values
        #: are still appended in group order, so every mean below is over the
        #: same values in the same order that the ablation-outer nesting would
        #: have produced.
        #: BOUNDED PROGRESS, AND THE DEADLINE CHECKED WHERE THE COST IS.
        #:
        #: 896 ablations over 67 items is tens of minutes to hours. The DEPTH
        #: search learned this the expensive way -- attempt 10 was silent for
        #: 10 h 47 m and nobody could tell a working search from a stalled
        #: one -- so this prints one line per (group, layer), which is 28 lines
        #: per group rather than 896, and checks the wall clock at the same
        #: instant. The two questions ("where is it?" and "has it run too
        #: long?") are asked together because they are asked for the same
        #: reason.
        started = time.monotonic()
        total_units = len(groups) * len(out_projections)
        unit = 0

        physical_invocations = 0
        for g_index, group in enumerate(groups):
            if batch_size > 1:
                reference = _forward_block(model, group, compute)
                mask = group.prediction_mask().to(reference.device)
            else:
                reference = _forward_item(model, group.items[0], compute)
                mask = None
            physical_invocations += 1
            try:
                for layer, out_projection in enumerate(out_projections):
                    for head in range(n_q):
                        with head_ablated(out_projection, head, head_dim):
                            if batch_size > 1:
                                ablated = _forward_block(model, group, compute)
                            else:
                                ablated = _forward_item(model, group.items[0],
                                                        compute)
                        physical_invocations += 1
                        bucket = per_head[layer][head]
                        if batch_size > 1:
                            #: ONE mean per row over that row's OWN valid
                            #: positions, and ONE host transfer for the batch.
                            values = forward_kl_mean_batch(reference, ablated,
                                                           mask, chunk=512)
                            for item, value in zip(group.items, values.tolist()):
                                bucket.setdefault(item["subtype"], []).append(value)
                        else:
                            bucket.setdefault(
                                group.items[0]["subtype"], []).append(
                                    forward_kl_mean(reference, ablated, chunk=512))
                        del ablated
                    unit += 1
                    mins = (time.monotonic() - started) / 60.0
                    rate = physical_invocations / mins if mins > 0 else 0.0
                    print(f"attention.causal_kl_v1: group {g_index + 1}/"
                          f"{len(groups)} layer {layer + 1}/"
                          f"{len(out_projections)} · {unit}/{total_units} units "
                          f"· {physical_invocations} forwards · {mins:.1f} min "
                          f"· {rate:.1f} fwd/min", flush=True)
                    if ctx.deadline is not None:
                        ctx.deadline.check(
                            f"attention.causal_kl_v1 group {g_index + 1}/"
                            f"{len(groups)} layer {layer + 1}/"
                            f"{len(out_projections)} "
                            f"({physical_invocations} forwards done)")
            finally:
                del reference

        #: One domain-balanced score per head, then the same per-group top-k
        #: every operator of this topology uses.
        scores_per_layer: list[list[float]] = []
        for layer in range(len(blocks)):
            layer_scores = []
            for head in range(n_q):
                means = {k: sum(v) / len(v)
                         for k, v in per_head[layer][head].items()}
                primary, _ = domain_balanced_score(means, domains)
                layer_scores.append(primary)
            scores_per_layer.append(layer_scores)

        new_spec = ctx.parent_spec.replace(**{HEADS_FIELD: keep_q})
        builder = ChildBuilder(adapter, model, new_spec, seed=ctx.seed)

        retained, kept_per_layer = [], []
        for idx, (src, dst) in enumerate(zip(adapter.blocks(model),
                                             adapter.blocks(builder.model))):
            s_out, d_out = (attention_out_projection(adapter, src),
                            attention_out_projection(adapter, dst))
            s_q, d_q = query_projection(adapter, src), query_projection(adapter, dst)
            scores = scores_per_layer[idx]
            kept = select_q_heads_by_score(scores, n_q, n_kv, keep_q)
            rows = head_rows(kept, head_dim, device=s_q.weight.device)

            total = float(sum(scores))
            retained.append(sum(scores[h] for h in kept) / total
                            if total > 0 else 0.0)
            kept_per_layer.append(list(kept))

            transformed = {id(d_q.weight), id(d_out.weight)}
            builder.assign(d_q.weight, s_q.weight[rows])
            builder.assign(d_out.weight, s_out.weight[:, rows])
            copy_module_except(builder, src, dst, skip=transformed)

        copy_embeddings_and_final_norm(builder, adapter, model)
        child = builder.finish()

        valid_tokens = sum(int(item["input_ids"].shape[-1]) for item in items)
        padded_positions = sum(
            g.size * int(g.input_ids.shape[1]) - g.n_valid_tokens for g in groups)
        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="parent_state",
                values={
                    "op.attention.retained_causal_kl_mean":
                        sum(retained) / len(retained),
                    "op.attention.retained_causal_kl_min": min(retained),
                },
                detail={"per_layer_retained_share": retained}),
            trace={"source": "causal_head_ablation_per_group_topk",
                   "score": "domain-balanced forward KL(parent || head ablated)",
                   #: EXECUTION EVIDENCE, not identity. `OperatorStep.identity()`
                   #: does not read `trace` — but unlike every other operator's
                   #: batch size, this one IS hashed, through the step config
                   #: that produced it. It is repeated here so a run's evidence
                   #: states how it ran without a reader having to resolve the
                   #: step.
                   "calibration_forward_batch_size": batch_size,
                   "physical_forward_invocations": physical_invocations,
                   "item_forward_equivalents": (len(blocks) * n_q + 1) * len(items),
                   "valid_tokens": valid_tokens,
                   "padded_positions": int(padded_positions),
                   "ablations": len(blocks) * n_q,
                   "seconds": round(time.monotonic() - started, 3),
                   "q_heads": [n_q, keep_q], "kv_heads": n_kv},
            artifacts={"kept_heads": kept_per_layer},
        )


#: The singleton, **unregistered**. Import is inert; see the module docstring.
ATTENTION_CAUSAL_KL_V1 = AttentionCausalKLV1()


def register(*, replace: bool = False) -> OperatorImplementation:
    """Join the global operator registry. Idempotent for an unchanged signature."""
    return register_implementation(ATTENTION_CAUSAL_KL_V1, replace=replace)


def unregister() -> None:
    """Leave the registry again — for tests, so registration cannot leak."""
    from aadistill.initialization.operators.base import unregister_implementation

    unregister_implementation(ATTENTION_CAUSAL_KL_V1.impl_id)
