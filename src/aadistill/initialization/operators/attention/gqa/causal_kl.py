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

import math
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

    STRICT, and deliberately not `int(value)`. This field is IDENTITY-BEARING:
    it is serialized into the step config and therefore into the state id. A
    coercion would let `4`, `4.0`, `"4"` and — via `bool` being an `int` —
    `True` mean the same execution while hashing to four different states, so
    a replay could disagree with its own record about what it ran. The rule is
    `ExecutionConfig`'s, for the same reason: must be an int, must not be a
    bool, must be at least 1.
    """
    value = (config or {}).get(BATCH_SIZE_CONFIG_KEY, 1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperatorError(
            f"{BATCH_SIZE_CONFIG_KEY} must be an int, not "
            f"{type(value).__name__} ({value!r}); this field is hashed into "
            "the step identity and must not be coerced")
    if value < 1:
        raise OperatorError(
            f"{BATCH_SIZE_CONFIG_KEY} must be >= 1, got {value!r}; 1 is the "
            "one-item-per-forward reference path")
    return value


def _refuse_non_finite(values: Any, where: str) -> None:
    """A non-finite causal score is a FAILED SCORER, never a zero.

    60k forwards feed a `sorted()` that materializes a head map. One NaN sorts
    to an arbitrary position and the map that comes out is a fiction nothing
    downstream can distinguish from a measurement. Clamping or substituting
    would be worse: it would produce a plausible map from a broken run.
    """
    if isinstance(values, torch.Tensor):
        if bool(torch.isfinite(values).all()):
            return
        bad = (~torch.isfinite(values)).nonzero().flatten().tolist()
        raise OperatorError(
            f"non-finite causal KL at {where}: rows {bad} of "
            f"{values.tolist()}. The scorer failed; a head map must not be "
            "materialized from it.")
    if not math.isfinite(float(values)):
        raise OperatorError(
            f"non-finite causal KL at {where}: {values!r}. The scorer failed; "
            "a head map must not be materialized from it.")


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


def _cuda_sync(device: Any) -> None:
    """Block until the device is idle, or do nothing off CUDA."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def warm_up(model, items: Sequence[Mapping[str, Any]], device: Any,
            *, batch_size: int = 1, n: int = 2) -> dict[str, Any]:
    """Untimed forwards so the first timed one is not paying for the last.

    Allocator growth, autotuning and kernel selection all happen on the first
    forwards of a given shape, and they would land entirely on whichever arm
    ran first -- which is a difference between the ARMS' ORDER, not between
    B1 and B4. Both arms run the same warm-up before their timer starts.

    It contributes NO causal evidence: it ablates nothing, its logits are
    discarded, and it returns only how much work it did so a record can show
    the two arms were warmed identically.
    """
    if n < 1 or not items:
        return {"warmup_forwards": 0, "warmup_items": 0}
    used = list(items)[:max(batch_size, 1) * n]
    forwards = 0
    with torch.no_grad():
        for batch in micro_batches(used, max(batch_size, 1),
                                   pad_id=(resolve_pad_id(model)
                                           if batch_size > 1 else 0),
                                   device=device):
            if batch_size > 1:
                _forward_block(model, batch, device)
            else:
                _forward_item(model, batch.items[0], device)
            forwards += 1
    _cuda_sync(device)
    return {"warmup_forwards": forwards, "warmup_items": len(used)}


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


def _group_decisions(scores: Sequence[float], n_q: int, n_kv: int,
                     keep_q: int) -> list[dict[str, Any]]:
    """Where each GQA group's cut fell, and by how much.

    A head map records WHAT was chosen. This records how nearly it went the
    other way: the two scores either side of the cut, and the margin between
    them. Without it a selection that turned on the twelfth decimal place is
    indistinguishable from one that was never in doubt -- which is exactly the
    question a B1-vs-B4 head-map delta raises.

    The ranking here reproduces `select_q_heads_by_score`'s ordering rule
    (descending score, ties to the lower head index) rather than inventing a
    second one; the selected set is checked against it below, so the two
    cannot drift.
    """
    per_g_t, per_g_s = n_q // n_kv, keep_q // n_kv
    out: list[dict[str, Any]] = []
    for g in range(n_kv):
        members = list(range(g * per_g_t, (g + 1) * per_g_t))
        ranked = sorted(members, key=lambda h: (-float(scores[h]), h))
        selected, rejected = ranked[:per_g_s], ranked[per_g_s:]
        cut_sel = float(scores[selected[-1]]) if selected else None
        cut_rej = float(scores[rejected[0]]) if rejected else None
        out.append({
            "group": g,
            "member_heads": members,
            "ranked_heads": ranked,
            "selected_heads": sorted(selected),
            "cutoff_selected": cut_sel,
            "cutoff_rejected": cut_rej,
            "cutoff_margin": (None if cut_sel is None or cut_rej is None
                              else cut_sel - cut_rej),
            "scores": [float(scores[h]) for h in members],
        })
    return out


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

        #: THE RAW SCIENTIFIC EVIDENCE, `[layer][head][item]` in the ORIGINAL
        #: item order rather than the order the groups happen to visit. About
        #: 60k floats at the real geometry, which is small enough that
        #: discarding it to keep only the head map would be throwing away the
        #: landscape that produced the decision. Item metadata is stored ONCE,
        #: as parallel columns, rather than repeated per value.
        #: The row is derived from the CONTIGUOUS grouping, not from object
        #: identity: `micro_batches` yields consecutive slices in the mixture's
        #: own order and never sorts by length, so `row0 + j` is exactly the
        #: item's index in `items`. An `id()` map would break silently the day
        #: the batcher copied a mapping.
        per_item: list[list[list[float | None]]] = [
            [[None] * len(items) for _ in range(n_q)] for _ in blocks]

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
        #: TIMED HONESTLY. CUDA kernels are asynchronous, so a timer that
        #: does not synchronize attributes the tail of the scoring loop to
        #: whatever runs next -- and the B1/B4 adoption gate is a ratio of two
        #: wall clocks, so a mis-attributed tail moves the verdict. The
        #: synchronize is once at the start and once at the end, not per
        #: forward, so it costs nothing measurable and perturbs no kernel.
        _cuda_sync(compute)
        started = time.monotonic()
        total_units = len(groups) * len(out_projections)
        unit = 0

        physical_invocations = 0
        row0 = 0
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
                            #: BEFORE the host conversion, so the refusal names
                            #: the rows rather than a list of floats that has
                            #: already lost which device produced them.
                            _refuse_non_finite(
                                values,
                                f"layer {layer} head {head} group {g_index} "
                                f"(B{batch_size}, {len(group.items)} items)")
                            for j, (item, value) in enumerate(
                                    zip(group.items, values.tolist())):
                                bucket.setdefault(item["subtype"], []).append(value)
                                per_item[layer][head][row0 + j] = value
                        else:
                            item = group.items[0]
                            value = forward_kl_mean(reference, ablated, chunk=512)
                            _refuse_non_finite(
                                value,
                                f"layer {layer} head {head} item "
                                f"{item.get('item_id', g_index)!r} (B1)")
                            bucket.setdefault(item["subtype"], []).append(value)
                            per_item[layer][head][row0] = value
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
            row0 += len(group.items)

        _cuda_sync(compute)
        #: THE SCORER's wall clock, which is what the adoption gate compares.
        #: Everything after this point -- aggregation, selection, building the
        #: child, writing it -- is identical work in both arms, so including
        #: it would dilute the ratio with a constant.
        scorer_seconds = time.monotonic() - started

        #: One domain-balanced score per head, then the same per-group top-k
        #: every operator of this topology uses.
        scores_per_layer: list[list[float]] = []
        for layer in range(len(blocks)):
            layer_scores = []
            for head in range(n_q):
                missing = [i for i, v in enumerate(per_item[layer][head])
                           if v is None]
                if missing:
                    raise OperatorError(
                        f"layer {layer} head {head}: {len(missing)} of "
                        f"{len(items)} items produced no causal KL "
                        f"(rows {missing[:8]}); the evidence is incomplete "
                        "and a head map must not be materialized from it")
                means = {k: sum(v) / len(v)
                         for k, v in per_head[layer][head].items()}
                primary, _ = domain_balanced_score(means, domains)
                #: AGAIN, on the aggregate. The per-item guard above cannot
                #: catch a mean that overflows, and this value is the one
                #: `sorted()` actually reads.
                _refuse_non_finite(primary,
                                   f"aggregate score for layer {layer} "
                                   f"head {head}")
                layer_scores.append(primary)
            scores_per_layer.append(layer_scores)

        new_spec = ctx.parent_spec.replace(**{HEADS_FIELD: keep_q})
        builder = ChildBuilder(adapter, model, new_spec, seed=ctx.seed)

        retained, kept_per_layer = [], []
        gqa_decisions: list[list[dict[str, Any]]] = []
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
            gqa_decisions.append(_group_decisions(scores, n_q, n_kv, keep_q))

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
                   #: The scoring loop alone, synchronized at both ends. The
                   #: B1/B4 gate is a ratio of THIS, not of `apply` -- the
                   #: aggregation, selection and child build after it are
                   #: identical in both arms and would dilute the ratio.
                   "scorer_seconds": round(scorer_seconds, 4),
                   "seconds": round(time.monotonic() - started, 3),
                   "q_heads": [n_q, keep_q], "kv_heads": n_kv},
            artifacts={
                "kept_heads": kept_per_layer,
                #: THE LANDSCAPE, not only the decision. Reconstructing why a
                #: head was dropped needs the per-item values, the aggregate
                #: scores and where each GQA group's cut fell -- a head map
                #: alone cannot distinguish a decisive margin from a tie.
                #: Item metadata is stored once as parallel columns; the
                #: values are `[layer][head][item]` in the mixture's order.
                "causal_head_evidence": {
                    "score": "domain-balanced forward KL(parent || head ablated)",
                    "calibration_forward_batch_size": batch_size,
                    "item_ids": [str(i.get("item_id", n))
                                 for n, i in enumerate(items)],
                    "item_domains": [str(i["domain"]) for i in items],
                    "item_subtypes": [str(i["subtype"]) for i in items],
                    "item_prediction_positions": [
                        int(i["input_ids"].shape[-1]) - 1 for i in items],
                    "per_item_kl": per_item,
                    "head_scores": scores_per_layer,
                    "gqa_decisions": gqa_decisions,
                },
            },
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
