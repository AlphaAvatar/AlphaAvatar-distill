"""DEPTH implementations: which blocks survive.

Two algorithms, deliberately kept as separate immutable ids because E8a showed
they disagree almost completely — for 36 -> 28 they share exactly one removed
layer out of eight, and the causal search preserved the full-width teacher
distribution 3.11x better while initializing 2.8 nats worse once composed with
width/FFN/attention compression. Which is right is the open question the search
exists to answer, so both stay registered and neither is a default.

``depth.positional_v0`` wraps ``init.sandwich.depth_span_map``; the map it
produces at 36 -> 28 is the one behind the canonical Stage-1 checkpoint
``86fbba78...``.

``depth.causal_kl_greedy_v1`` wraps ``init.contribution`` — block bypass, forward
KL against the *unbypassed parent*, domain-balanced aggregation, iterative greedy
removal with a stated tie-break. It is re-run against whatever checkpoint it is
handed, which is what makes it conditional rather than a precomputed teacher
decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ...init.contribution import (
    bypassed_blocks,
    distortion,
    domain_balanced_score,
    expected_evaluations,
    greedy_removal,
)
from ...init.sandwich import depth_span_map
from ..arch import ArchitectureAdapter, ArchSpec, Capability
from ..metrics import OperatorLocalMetrics
from ._common import ChildBuilder, copy_embeddings_and_final_norm, copy_module_except
from .base import (
    CalibrationNeed,
    OperatorContext,
    OperatorError,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
)

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


class DepthPositionalV0(OperatorImplementation):
    impl_id = "depth.positional_v0"
    kind = "DEPTH"
    version = 0
    description = (
        "Positional pairwise merge in a middle band: ~1/5 of the surviving 1:1 "
        "layers stay before the band, the rest after, so both the earliest and "
        "the latest blocks map 1:1. The incumbent map behind qwen3_0p6b_init_v0.")
    required_capabilities = frozenset({Capability.BLOCK_LIST})
    modifies = frozenset({DEPTH_FIELD})
    preserves = frozenset({"hidden_size", "intermediate_size", "num_attention_heads",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.NONE
    objective = "none (fixed positional heuristic; no measurement is taken)"
    deterministic = True
    requires_seed = False
    produces = ("depth_map",)
    target_validation = "result num_hidden_layers equals the target exactly"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        depth_span_map(spec[DEPTH_FIELD], target[DEPTH_FIELD])  # raises if infeasible
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{DEPTH_FIELD: target[DEPTH_FIELD]}),
            forward_passes=0, stats_passes=0,
            notes="no calibration; the map is a function of the two layer counts")

    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        spans = depth_span_map(ctx.parent_spec[DEPTH_FIELD],
                               ctx.target_spec[DEPTH_FIELD])
        kept = [s["representative"] for s in spans]
        model = _build_child_with_layers(ctx, kept)
        removed = sorted(set(range(ctx.parent_spec[DEPTH_FIELD])) - set(kept))
        return OperatorOutcome(
            model=model,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="none",
                values={"op.depth.positional.n_removed": float(len(removed))},
                detail={"note": "a positional heuristic takes no measurement, so it "
                                "reports no comparable objective value"}),
            trace={"kept_layers": kept, "removed_layers": removed,
                   "spans": spans, "source": "positional_pairwise_merge"},
        )


class DepthCausalKLGreedyV1(OperatorImplementation):
    impl_id = "depth.causal_kl_greedy_v1"
    kind = "DEPTH"
    version = 1
    description = (
        "Iterative greedy block removal minimizing forward KL(parent || "
        "parent-with-S-bypassed) over real prediction positions, aggregated as the "
        "unweighted mean over domains of the unweighted mean over each domain's "
        "sub-types. Ties break to the lower layer index. E8a's algorithm, re-run "
        "against the checkpoint it is handed rather than always against the teacher.")
    required_capabilities = frozenset({Capability.BLOCK_LIST, Capability.RESIDUAL_STREAM,
                                       Capability.LOGIT_COMPARABLE})
    modifies = frozenset({DEPTH_FIELD})
    preserves = frozenset({"hidden_size", "intermediate_size", "num_attention_heads",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.FORWARD_LOGITS
    objective = "forward KL(parent || parent-with-S-bypassed), domain-balanced"
    deterministic = True
    requires_seed = False
    produces = ("depth_map", "search_rounds")
    target_validation = "result num_hidden_layers equals the target exactly"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        n_remove = spec[DEPTH_FIELD] - target[DEPTH_FIELD]
        if n_remove <= 0:
            raise OperatorError(
                f"{self.impl_id}: nothing to remove ({spec[DEPTH_FIELD]} -> "
                f"{target[DEPTH_FIELD]})")
        evals = expected_evaluations(spec[DEPTH_FIELD], n_remove)
        n_items = int((config or {}).get("n_calibration_items", 0))
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{DEPTH_FIELD: target[DEPTH_FIELD]}),
            # +1 for the intact-parent reference pass over the mixture.
            forward_passes=(evals + 1) * max(n_items, 1),
            stats_passes=0,
            notes=f"{evals} subset evaluations over {n_items} calibration items")

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        model = ctx.model
        items = list(ctx.calibration_items)
        if getattr(model.config, "use_cache", False):
            model.config.use_cache = False

        domains = _domain_map(items)
        targets = [item["input_ids"][0, 1:].cpu() for item in items]
        reference = _ReferenceLogits(model, items, ctx.device)

        def score(skip: frozenset[int]) -> float:
            per_subtype: dict[str, list[float]] = {}
            for item, tgt in zip(items, targets):
                # Order matters: the reference is the UNBYPASSED parent, so when
                # it is being recomputed it must not be taken inside the bypass.
                ref = reference.get(item)
                abl = _forward_logits(model, item, ctx.device, skip)
                sums = distortion(ref, abl, tgt, chunk=512).as_dict()
                per_subtype.setdefault(item["subtype"], []).append(sums["kl"])
                del abl
            means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
            primary, _ = domain_balanced_score(means, domains)
            return primary

        n_remove = ctx.parent_spec[DEPTH_FIELD] - ctx.target_spec[DEPTH_FIELD]
        result = greedy_removal(score, ctx.parent_spec[DEPTH_FIELD], n_remove)
        kept = result["kept"]
        child = _build_child_with_layers(ctx, kept)

        final_score = result["rounds"][-1]["chosen_score"] if result["rounds"] else 0.0
        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="parent_state",
                values={
                    "op.depth.causal_kl.final": float(final_score),
                    "op.depth.causal_kl.evaluations": float(result["evaluations"]),
                },
                detail={"removal_order": result["removal_order"]}),
            trace={"kept_layers": kept, "removed_layers": result["removed"],
                   "removal_order": result["removal_order"],
                   "source": "causal_kl_greedy"},
            # The memory decision is an artifact, NOT part of the trace. The
            # trace describes the deterministic result — `test_operator_is_
            # deterministic` compares two invocations of it field for field —
            # and which path ran depends on how much memory the host happened to
            # have free. The numbers are identical either way, so the decision
            # belongs with the byproducts.
            artifacts={"search_rounds": result["rounds"],
                       "reference_cache": reference.decision()},
        )


def _domain_map(items) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in items:
        subs = out.setdefault(item["domain"], [])
        if item["subtype"] not in subs:
            subs.append(item["subtype"])
    return {k: sorted(v) for k, v in sorted(out.items())}


@torch.no_grad()
def _forward_logits(model, item, device: str, skip=frozenset()):
    """One item's prediction-position logits, optionally with blocks bypassed.

    Left in the model's own dtype. ``distortion`` upcasts to float32 in chunks
    internally, so widening here would only double the bytes held per item
    without changing a single reduced value.
    """
    ids = item["input_ids"].to(device)
    if not skip:
        return model(ids).logits[0, :-1].cpu()
    with bypassed_blocks(model, skip):
        return model(ids).logits[0, :-1].cpu()


class _ReferenceLogits:
    """The unbypassed parent's logits, cached only when the memory fits.

    The reference is identical for every candidate in the whole greedy search,
    so caching it turns ``2 * evals * n_items`` forward passes into
    ``(evals + 1) * n_items`` — the cost structure ``plan()`` reports.

    It had been cached **unconditionally**, in float32, in host RAM. For the
    frozen ``calib.domain_balanced@v1`` mixture that is 59,763 prediction
    positions x 151,936 vocabulary x 4 B = **33.8 GiB**, held for the duration of
    the search, on every invocation of this operator. Nothing had ever executed
    it against the real mixture: Phase-A attempt 5 died earlier, at the
    calibration file, and every zero-cost run used a toy mixture. Its first real
    execution — this rehearsal — was killed by the OOM killer.

    ``scripts/training/search_depth_map.py``, the E8a script whose algorithm this
    operator re-runs, already had the answer and it was dropped in the port: size
    the cache, and if it does not fit, recompute. Recomputing is numerically
    identical — same deterministic no-grad forward, same tensor, and
    ``distortion`` upcasts either way — so the fallback is automatic and loud
    rather than a flag a caller has to remember.
    """

    #: E8a's fraction. The rest holds the ablated logits, the float32 reduction
    #: chunks, and the activations of the next forward pass.
    BUDGET_FRACTION = 0.66

    def __init__(self, model, items, device: str) -> None:
        self.model = model
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}
        positions = sum(int(i["input_ids"].shape[1]) - 1 for i in items)
        itemsize = next(model.parameters()).dtype.itemsize
        self.estimate_bytes = positions * int(model.config.vocab_size) * itemsize
        self.available_bytes, self.headroom_source = _host_available_memory_bytes()
        self.enabled = (self.available_bytes is None
                        or self.estimate_bytes
                        <= self.BUDGET_FRACTION * self.available_bytes)
        if not self.enabled:
            # Loud, like E8a's. A run that quietly doubled its forward passes is
            # a run whose wall-clock estimate no longer holds.
            print(f"depth.causal_kl_greedy_v1: reference cache "
                  f"{self.estimate_bytes / 2**30:.1f} GiB does not fit in "
                  f"{self.BUDGET_FRACTION:.0%} of {self.available_bytes / 2**30:.1f} "
                  f"GiB ({self.headroom_source}) -> recomputing the reference per "
                  "candidate: identical numbers, ~2x the forward passes",
                  flush=True)

    def get(self, item) -> torch.Tensor:
        if not self.enabled:
            return _forward_logits(self.model, item, self.device)
        hit = self._cache.get(item["item_id"])
        if hit is None:
            hit = _forward_logits(self.model, item, self.device)
            self._cache[item["item_id"]] = hit
        return hit

    def decision(self) -> dict[str, Any]:
        return {
            "cached": self.enabled,
            "estimate_bytes": int(self.estimate_bytes),
            "estimate_gib": round(self.estimate_bytes / 2**30, 3),
            "available_bytes": self.available_bytes,
            "budget_fraction": self.BUDGET_FRACTION,
            "headroom_source": self.headroom_source,
            "fallback": None if self.enabled else (
                "recompute per candidate: identical numbers, ~2x forward passes"),
        }


def _host_available_memory_bytes() -> tuple[int | None, str]:
    """Free **host** memory, because that is where the cache lands.

    ``_forward_logits`` returns ``.cpu()``, so the cache is host-resident no
    matter what device the model is on. E8a kept its cache on the accelerator
    and therefore checked ``torch.cuda.mem_get_info``; copying that probe here
    would measure free VRAM against an allocation that never touches it.

    The **cgroup** grant comes first: inside a container ``/proc/meminfo``
    reports the *host's* memory, and this project has already shipped one bug
    from trusting a host-wide number over the cgroup limit (``nproc`` versus the
    CPU quota). The limit that binds is the smaller of the two.
    """
    candidates: list[tuple[int, str]] = []
    try:                                            # cgroup v2
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            candidates.append((max(int(limit) - used, 0), "cgroup.v2"))
    except (OSError, ValueError):
        pass
    try:                                            # cgroup v1
        limit = int(Path(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
        used = int(Path(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
        # An "unlimited" v1 cgroup reports a sentinel near 2**63.
        if limit < 2**62:
            candidates.append((max(limit - used, 0), "cgroup.v1"))
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append((int(line.split()[1]) * 1024, "proc.meminfo"))
                break
    except (OSError, ValueError):
        pass
    if not candidates:
        return None, "unknown"
    return min(candidates)


DEPTH_POSITIONAL_V0 = register_implementation(DepthPositionalV0())
DEPTH_CAUSAL_KL_GREEDY_V1 = register_implementation(DepthCausalKLGreedyV1())
