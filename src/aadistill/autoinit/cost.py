"""Cost and branching model for an AutoInitializer search.

Zero-cost output whose only job is to make the next decision — beam width, Top-N,
survivor count — an arithmetic one rather than a guess. Everything is derived
from an ``ArchSpec`` through the adapter, so the same model prices a 4B -> 596M
pilot and a 30B -> 4.xB study with no edit.

Anchors, and how good each one is:

* **88.83 effective TFLOP/s on an L40S**, *measured*. E8a's depth search ran 260
  subset evaluations over a 67-item / 59,763-position mixture in 1,300 s at full
  4B width. The FLOP accounting below, applied to that exact workload, gives
  1.1548e17 FLOPs, hence 88.83 TFLOP/s and ~24.5% MFU. That is a real end-to-end
  number including the Python loop, not a spec sheet.
* **Other accelerators are scaled by peak bf16 and marked ESTIMATED.** Scaling
  MFU across architectures is an assumption; it is labelled as one.
* **Activation-statistics collection is a range, not a point.** The one
  measurement available is 4,972 s for 949,859 tokens (5.24 ms/token) on this
  CPU-only dev box, which includes the model forward. On a GPU the forward
  collapses but the float64 ``X^T X`` accumulation stays on the CPU, and the
  split has never been measured. So the model reports a lower bound (GPU forward
  only) and an upper bound (the full CPU rate) and says so. Measuring the split
  is a cheap pre-pilot task and is listed as one.

Nothing here is allowed to quietly shrink a plan to fit a budget. ``price_search``
reports what the search costs; comparing that against an authorization is the
caller's job, and reporting a shortfall specifically is the maintainer's rule.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .arch import ArchitectureAdapter, ArchSpec
from .operators.base import OperatorImplementation

BYTES_PER_PARAM_BF16 = 2


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    price_per_hour_usd: float
    effective_tflops: float
    vram_gb: float
    measured: bool
    source: str

    def seconds_for(self, flops: float) -> float:
        return flops / (self.effective_tflops * 1e12)

    def usd_for(self, seconds: float) -> float:
        return seconds / 3600.0 * self.price_per_hour_usd

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "price_per_hour_usd": self.price_per_hour_usd,
                "effective_tflops": self.effective_tflops, "vram_gb": self.vram_gb,
                "measured": self.measured, "source": self.source}


L40S_MEASURED = HardwareProfile(
    name="L40S", price_per_hour_usd=0.99, effective_tflops=88.83, vram_gb=48.0,
    measured=True,
    source=("E8a: 260 subset evaluations over a 67-item / 59,763-position mixture in "
            "1,300 s at 4.02B full width. This module's own accounting puts that "
            "workload at 1.1548e17 FLOPs, hence 88.83 TFLOP/s and ~24.5% of the "
            "362 TFLOPS bf16 dense peak. Asserted round-trip in tests/autoinit."),
)

A100_80GB_ESTIMATED = HardwareProfile(
    name="A100-SXM-80GB", price_per_hour_usd=1.59, effective_tflops=76.56, vram_gb=80.0,
    measured=False,
    source=("ESTIMATED: the L40S's measured 88.83 scaled by the bf16 dense peak ratio "
            "312/362. MFU is assumed to transfer across architectures; it has not "
            "been measured on this workload."),
)

DEVBOX_CPU_MEASURED = HardwareProfile(
    name="dev-box CPU (16 vCPU)", price_per_hour_usd=0.0, effective_tflops=1.59,
    vram_gb=0.0, measured=True,
    source=("Stage 0 regeneration: 949,859 tokens of 4.02B forward + float64 X^T X "
            "accumulation in 4,972 s"),
)

#: End-to-end statistics-collection rate on the CPU-only dev box, model forward
#: included. Upper bound for a GPU run, because the float64 accumulation does not
#: move to the accelerator in the current implementation.
CPU_STATS_SECONDS_PER_TOKEN = 4972.0 / 949_859.0


# --- FLOP accounting --------------------------------------------------------


def forward_flops_per_token(spec: ArchSpec, seq_len: int, *,
                            layers: float | None = None) -> float:
    """Forward FLOPs for one token, at ``seq_len`` context.

    Attention score/value products are included with a 0.5 causal factor; at the
    sequence lengths a calibration mixture uses they are ~9% of the total, which
    is small but not nothing. ``layers`` may be fractional so a greedy depth
    search — whose candidates have progressively fewer blocks — can be priced
    with its true average rather than the full stack.
    """
    d = spec["hidden_size"]
    n_q, n_kv = spec["num_attention_heads"], spec["num_key_value_heads"]
    head_dim, inter = spec["head_dim"], spec["intermediate_size"]
    vocab = spec["vocab_size"]
    n_layers = spec["num_hidden_layers"] if layers is None else layers

    proj = 2.0 * (n_q * head_dim * d + 2 * n_kv * head_dim * d + d * n_q * head_dim)
    attn = 0.5 * 2.0 * (2.0 * seq_len * n_q * head_dim)
    mlp = 2.0 * 3.0 * inter * d
    return n_layers * (proj + attn + mlp) + 2.0 * vocab * d


def checkpoint_bytes(spec: ArchSpec, adapter: ArchitectureAdapter,
                     bytes_per_param: int = BYTES_PER_PARAM_BF16) -> int:
    return adapter.param_count(spec) * bytes_per_param


def activation_stats_bytes(spec: ArchSpec) -> int:
    """float64 residual second moments dominate: (L+1) x d x d x 8 bytes.

    Reported because it scales as ``d^2`` and is a real 30B -> 4.xB risk: at the
    4B teacher it is 1.94 GB, and the same formula at a 6144-wide, 64-layer
    teacher is ~19.6 GB of float64 state per statistics pass.
    """
    d = spec["hidden_size"]
    n_points = spec["num_hidden_layers"] + 1
    moments = n_points * d * d * 8
    ffn = 2 * spec["num_hidden_layers"] * spec["intermediate_size"] * 8
    return moments + ffn + spec["vocab_size"] * 8


def consumes_activation_stats(impl: OperatorImplementation) -> bool:
    """Does this implementation take an activation-statistics pass?

    The one question that decides whether a parent's collection is shared with
    it. `CalibrationNeed.FORWARD_LOGITS` consumes calibration but collects no
    statistics — `depth.causal_kl_greedy_v1` runs its own forwards — so it must
    not be counted as a consumer of the shared pass.
    """
    from .operators.base import CalibrationNeed

    return getattr(impl, "calibration", None) is CalibrationNeed.ACTIVATION_STATS


# --- non-FLOP per-child overhead --------------------------------------------

#: Sustained bytes/second for the materialization path as a whole: writing the
#: shards, reading them back to hash, and reading them again for the canonical
#: reload. Deliberately pessimistic for container storage — a hard bound may not
#: assume a fast disk — and deliberately ONE number rather than three, because
#: the three are not separately measured and inventing a split would dress a
#: guess as a model.
CHECKPOINT_IO_BYTES_PER_SECOND = 400e6

#: Per child, independent of size: process/Python overhead, config and tokenizer
#: writes, `identify_checkpoint` bookkeeping, the round-trip validation forward
#: and its comparison, and the journal append.
PER_CHILD_FIXED_SECONDS = 10.0

#: How many times the checkpoint's bytes cross storage per child: written once by
#: `adapter.save`, read once by `identify_checkpoint` to hash, read once by the
#: canonical reload. The search contract requires all three and none may be
#: skipped, so the factor is structural rather than tunable.
CHECKPOINT_IO_PASSES = 3


def materialization_overhead_seconds(child_bytes: float) -> float:
    """save -> hash/identify -> canonical reload -> validate, for one child.

    **None of this is FLOPs, and the hardware anchor does not cover it.**
    `L40S_MEASURED.effective_tflops` was measured from 260 subset evaluations —
    pure forward compute — so using it to price checkpoint I/O, hashing, Python
    bookkeeping and a round-trip validation would be reading a number off an
    instrument that was never pointed at them.

    Attempt 3 makes the size of the gap concrete: 544.7 min of Stage 1, 388.2 min
    of measured causal DEPTH, and 156.5 min of everything else that the FLOP model
    accounted for only as a few seconds of state evaluation per child.

    Conservative by construction, and cheap: it is one multiply and an add per
    child, on a term the search performs for every state it produces.
    """
    return (CHECKPOINT_IO_PASSES * float(child_bytes) / CHECKPOINT_IO_BYTES_PER_SECOND
            + PER_CHILD_FIXED_SECONDS)


# --- per-operator cost ------------------------------------------------------


@dataclass(frozen=True)
class OperatorCost:
    impl_id: str
    kind: str
    parent_spec_hash: str
    flops: float
    gpu_seconds: float
    stats_seconds_low: float
    stats_seconds_high: float
    child_bytes: int
    peak_resident_bytes: int
    notes: str = ""

    def seconds_low(self) -> float:
        return self.gpu_seconds + self.stats_seconds_low

    def seconds_high(self) -> float:
        return self.gpu_seconds + self.stats_seconds_high

    def as_dict(self) -> dict[str, Any]:
        return {
            "impl_id": self.impl_id, "kind": self.kind,
            "parent_spec_hash": self.parent_spec_hash,
            "flops": self.flops, "gpu_seconds": self.gpu_seconds,
            "stats_seconds_low": self.stats_seconds_low,
            "stats_seconds_high": self.stats_seconds_high,
            "seconds_low": self.seconds_low(), "seconds_high": self.seconds_high(),
            "child_gib": self.child_bytes / 2**30,
            "peak_resident_gib": self.peak_resident_bytes / 2**30,
            "notes": self.notes,
        }


#: The three ways the intact reference can be paid for, cheapest first. They are
#: not interchangeable and pricing the wrong one is what invalidated the
#: 1.91-7.51 h P=2 range: that range assumed CACHED, attempt 3 ran RECOMPUTED, and
#: the search hit its deadline at 9.08 h without finishing.
REFERENCE_MODES = ("cached", "partial", "recomputed")


def greedy_depth_flops(spec: ArchSpec, n_remove: int, tokens: int, seq_len: int,
                       *, reference_mode: str = "cached",
                       cached_fraction: float = 0.0) -> tuple[float, float]:
    """FLOPs and the average surviving-layer count for a full greedy removal.

    Round ``r`` evaluates ``L - r`` candidates, each with ``r + 1`` blocks
    bypassed, so the work is not ``260 x full model`` — it is the exact sum below,
    plus **the intact reference, whose cost depends on how it is held**:

    ``cached``
        one intact pass for the whole expansion. What this function priced
        unconditionally until 2026-08-27, and the reason the P=2 range was wrong.
    ``recomputed``
        one intact pass *per candidate*: `evaluations` of them. This is what
        Phase-B attempt 3 actually ran — 16.9 GiB of reference against a 13.4 GiB
        allowance, all-or-nothing, so none of it was kept.
    ``partial``
        ``cached_fraction`` of the mixture is resident and paid for once; the
        remainder is recomputed per candidate. This is what the bounded cache
        added, and it is a *fraction*, so a hard bound may not assume it.

    The fraction is over prediction positions, which is what the cache admits on,
    so it maps directly onto forward cost.
    """
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"reference_mode must be one of {REFERENCE_MODES}")
    n_layers = spec["num_hidden_layers"]
    total, evaluations = 0.0, 0
    weighted_layers = 0.0
    for r in range(n_remove):
        candidates = n_layers - r
        surviving = n_layers - (r + 1)
        total += candidates * tokens * forward_flops_per_token(spec, seq_len,
                                                               layers=surviving)
        weighted_layers += candidates * surviving
        evaluations += candidates
    intact = tokens * forward_flops_per_token(spec, seq_len)
    if reference_mode == "cached":
        total += intact
    elif reference_mode == "recomputed":
        total += evaluations * intact
    else:
        share = min(max(float(cached_fraction), 0.0), 1.0)
        total += share * intact + (1.0 - share) * evaluations * intact
    return total, (weighted_layers / evaluations if evaluations else 0.0)


def operator_cost(impl: OperatorImplementation, parent_spec: ArchSpec,
                  target_spec: ArchSpec, adapter: ArchitectureAdapter, *,
                  calibration_tokens: int, seq_len: int,
                  hardware: HardwareProfile,
                  depth_reference_mode: str = "cached",
                  depth_cached_fraction: float = 0.0,
                  include_stats: bool = True) -> OperatorCost:
    plan = impl.plan(parent_spec, target_spec, adapter,
                     {"n_calibration_items": 1})
    child_spec = plan.result_spec
    flops = 0.0
    stats_low = stats_high = 0.0
    notes = plan.notes

    if impl.impl_id == "depth.causal_kl_greedy_v1":
        n_remove = parent_spec["num_hidden_layers"] - target_spec["num_hidden_layers"]
        flops, avg_layers = greedy_depth_flops(
            parent_spec, n_remove, calibration_tokens, seq_len,
            reference_mode=depth_reference_mode,
            cached_fraction=depth_cached_fraction)
        notes = (f"{plan.notes}; mean {avg_layers:.2f} surviving blocks per "
                 f"evaluation; intact reference {depth_reference_mode}")
    elif plan.stats_passes:
        # One forward with hooks, then a float64 eigendecomposition or top-k.
        flops = calibration_tokens * forward_flops_per_token(parent_spec, seq_len)
        if include_stats:
            stats_low = hardware.seconds_for(flops)
            stats_high = calibration_tokens * CPU_STATS_SECONDS_PER_TOKEN
            notes = (f"{plan.notes}; statistics pass priced as a range: "
                     "GPU-forward-only lower bound vs the measured CPU end-to-end rate")
        else:
            # The pass is charged ONCE for this (parent, profile) by the caller,
            # because `StatsCache` shares it across every stats-consuming operator
            # at that parent. Charging it here as well would bill the same
            # collection to composite AND ffn AND width.
            notes = f"{plan.notes}; statistics pass charged once per (parent, profile)"
        flops = 0.0  # accounted inside the stats bounds instead of twice

    child_bytes = checkpoint_bytes(child_spec, adapter)
    parent_bytes = checkpoint_bytes(parent_spec, adapter)
    peak = parent_bytes + child_bytes
    if plan.stats_passes:
        peak += activation_stats_bytes(parent_spec)

    return OperatorCost(
        impl_id=impl.impl_id, kind=impl.kind,
        parent_spec_hash=parent_spec.spec_hash, flops=flops,
        gpu_seconds=hardware.seconds_for(flops),
        stats_seconds_low=stats_low, stats_seconds_high=stats_high,
        child_bytes=child_bytes, peak_resident_bytes=peak, notes=notes)


def evaluation_cost(spec: ArchSpec, *, suite_tokens: int, seq_len: int,
                    hardware: HardwareProfile) -> float:
    """One state evaluation: a single forward of the candidate over the suite."""
    return hardware.seconds_for(suite_tokens * forward_flops_per_token(spec, seq_len))


# --- path enumeration and branching ----------------------------------------


@dataclass(frozen=True)
class PathNode:
    level: int
    impl_id: str
    kind: str
    parent_spec: ArchSpec
    child_spec: ArchSpec


def profile_multiplicity(impls: Sequence[OperatorImplementation],
                         n_profiles: int) -> dict[str, int]:
    """How many distinct invocations each implementation actually has.

    An implementation declaring ``CalibrationNeed.NONE`` has **one**, whatever
    ``n_profiles`` is: it consumes no mixture, so a per-profile branch would
    produce byte-identical states. For the v1 library at ``P`` profiles that
    makes the decomposed space

        24 orderings x (1 + P) DEPTH x P WIDTH x P FFN x 1 ATTENTION

    — 48 paths at P=1, 288 at P=2, 864 at P=3 — rather than the ``48 x P^4`` an
    unconditional branch would claim.
    """
    from .calibration import consumes_calibration

    return {i.impl_id: (n_profiles if consumes_calibration(i) else 1) for i in impls}


def enumerate_paths(root_spec: ArchSpec, target_spec: ArchSpec,
                    adapter: ArchitectureAdapter,
                    impls: Sequence[OperatorImplementation]) -> list[list[PathNode]]:
    """Every complete decomposed path: kind orderings x implementation choices.

    Calibration profiles are *not* multiplied in here — they do not change the
    geometry, so they multiply the node count without changing the per-node cost.
    ``branching_estimate`` applies them, per implementation.
    """
    differing = sorted(root_spec.diff(target_spec))
    by_field: dict[str, list[OperatorImplementation]] = {}
    for impl in impls:
        for field_name in impl.modifies:
            if field_name in differing and len(impl.modifies) == 1:
                by_field.setdefault(field_name, []).append(impl)
    if sorted(by_field) != differing:
        missing = sorted(set(differing) - set(by_field))
        raise ValueError(f"no single-field implementation covers {missing}")

    paths: list[list[PathNode]] = []
    for order in itertools.permutations(differing):
        choices = [sorted(by_field[f], key=lambda i: i.impl_id) for f in order]
        for combo in itertools.product(*choices):
            spec = root_spec
            nodes = []
            for level, impl in enumerate(combo):
                child = impl.plan(spec, target_spec, adapter,
                                  {"n_calibration_items": 1}).result_spec
                nodes.append(PathNode(level, impl.impl_id, impl.kind, spec, child))
                spec = child
            if not spec.matches(target_spec):  # pragma: no cover - defensive
                raise ValueError("enumerated path does not reach the target")
            paths.append(nodes)
    return paths


def branching_estimate(root_spec: ArchSpec, target_spec: ArchSpec,
                       adapter: ArchitectureAdapter,
                       impls: Sequence[OperatorImplementation],
                       *, n_profiles: int, beam_width: int, warmup_levels: int = 0,
                       include_composite: Sequence[OperatorImplementation] = ()) -> dict[str, Any]:
    """How many states a beam of this width actually materializes.

    Reported as a range rather than a point because the per-level branching
    depends on *which* kinds the surviving beam members happen to have applied —
    a parent that has already used DEPTH has fewer remaining options than one
    that has not, and which parents survive is a measurement outcome.
    """
    paths = enumerate_paths(root_spec, target_spec, adapter, impls)
    differing = sorted(root_spec.diff(target_spec))
    n_kinds = len(differing)
    multiplicity = profile_multiplicity(impls, n_profiles)

    # Invocations per structural field: implementations for that field, each
    # counted once per profile it actually consumes.
    by_field: dict[str, int] = {}
    impls_per_field: dict[str, int] = {}
    for impl in impls:
        if len(impl.modifies) != 1:
            continue
        f = next(iter(impl.modifies))
        if f in differing:
            by_field[f] = by_field.get(f, 0) + multiplicity[impl.impl_id]
            impls_per_field[f] = impls_per_field.get(f, 0) + 1

    unbeamed = len(list(itertools.permutations(differing)))
    for field_name in differing:
        unbeamed *= by_field[field_name]

    per_level: list[dict[str, Any]] = []
    total_min = total_max = 0
    for level in range(n_kinds):
        remaining = n_kinds - level
        options = sorted(by_field.values())
        # Best case for branching width: the beam kept the parents that still
        # have the most-branching kinds available. Worst case: the opposite.
        max_inv = sum(options[-remaining:])
        min_inv = sum(options[:remaining])
        if level == 0:
            parents_lo = parents_hi = 1
        elif level <= warmup_levels:
            # The preceding level was a warmup: nothing was pruned, so every
            # child it generated is a parent here. This is what delayed pruning
            # costs, and it is the widest point of the whole search.
            parents_lo = per_level[-1]["children_min"]
            parents_hi = per_level[-1]["children_max"]
        else:
            parents_lo = parents_hi = beam_width
        lo, hi = parents_lo * min_inv, parents_hi * max_inv
        per_level.append({"level": level, "parents_min": parents_lo,
                          "parents_max": parents_hi, "parents": parents_hi,
                          "pruned_here": level >= warmup_levels,
                          "children_min": lo, "children_max": hi})
        total_min += lo
        total_max += hi

    # The composite consumes calibration, so it does branch over profiles.
    composite_children = sum(multiplicity.get(i.impl_id, n_profiles)
                             for i in include_composite)
    return {
        "differing_fields": differing,
        "n_kinds": n_kinds,
        "implementations_per_field": dict(sorted(impls_per_field.items())),
        "invocations_per_field": dict(sorted(by_field.items())),
        "profile_multiplicity": dict(sorted(multiplicity.items())),
        "n_calibration_profiles": n_profiles,
        "beam_width": beam_width,
        "warmup_levels": warmup_levels,
        "complete_paths_unbeamed": unbeamed,
        "complete_paths_geometry_only": len(paths),
        "kind_orderings": len(list(itertools.permutations(differing))),
        "per_level": per_level,
        "states_materialized_min": total_min + composite_children,
        "states_materialized_max": total_max + composite_children,
        "composite_leaves": composite_children,
        "leaves_min": per_level[-1]["children_min"] + composite_children,
        "leaves_max": per_level[-1]["children_max"] + composite_children,
    }


# --- the conservative hard bound --------------------------------------------


@dataclass(frozen=True)
class Expansion:
    """One (parent geometry, implementation) the search will perform at a level.

    `multiplicity` is how many times it actually runs — once per profile it
    consumes, once total otherwise — so `depth.positional_v0` and
    `attention.weight_proxy_v0` stay single at P=2.

    `operator_seconds_*` EXCLUDE the activation-statistics pass. Statistics are
    charged separately, per (parent, profile), because that is what the runtime
    cache does; folding them in here would bill one collection to composite AND
    ffn AND width.
    """

    level: int
    parent_spec_hash: str
    impl_id: str
    multiplicity: int
    consumes_stats: bool
    operator_seconds_low: float
    operator_seconds_high: float
    eval_seconds: float
    overhead_seconds: float
    child_bytes: int
    peak_resident_bytes: int

    def child_seconds_low(self) -> float:
        return self.operator_seconds_low + self.eval_seconds + self.overhead_seconds

    def child_seconds_high(self) -> float:
        return self.operator_seconds_high + self.eval_seconds + self.overhead_seconds


def stats_collections(level: int, expansions: Sequence[Expansion],
                      n_profiles: int) -> int:
    """How many activation-statistics passes ONE parent actually pays for.

    This must match `StatsCache`, not idealize it. The cache key is
    `(parent artifact digest, profile hash, stats spec, adapter, numerics)` and
    the search holds one entry per active profile, so a parent with any number of
    stats-consuming operators pays `n_profiles` collections — one shared pass per
    mixture, reused by `composite`, `ffn` and `width` alike.

    **Except at the root, where the runtime cannot share.** `BeamSearch._stats_key`
    returns `None` when `parent.artifact_digest is None`, and the root's identity
    is a published revision rather than an artifact the search computed, so every
    stats-consuming expansion of the root collects its own. Pricing the root as
    though it shared would understate the widest level in the search by exactly
    the operators that make it wide.
    """
    consumers = [e for e in expansions if e.consumes_stats]
    if not consumers:
        return 0
    if level == 0:
        return sum(e.multiplicity for e in consumers)
    return n_profiles


def _parent_seconds(level: int, expansions: Sequence[Expansion], n_profiles: int,
                    stats_low: float, stats_high: float) -> tuple[float, float]:
    """One parent's whole expansion set, statistics charged once per profile."""
    collections = stats_collections(level, expansions, n_profiles)
    low = sum(e.multiplicity * e.child_seconds_low() for e in expansions)
    high = sum(e.multiplicity * e.child_seconds_high() for e in expansions)
    return low + collections * stats_low, high + collections * stats_high


def conservative_hard_seconds(
        by_level: Mapping[int, Mapping[str, list[Expansion]]],
        stats_cost: Mapping[str, tuple[float, float]],
        branching: Mapping[str, Any],
        n_profiles: int) -> tuple[float, list[dict[str, Any]]]:
    """A hard bound a beam of expensive parents cannot exceed.

    The bound this replaces was ``children_max x mean_node_seconds_high``, and an
    average is not a bound. Node cost at one level varies by more than 2x with the
    parent's geometry — a causal-depth search against the full-width teacher
    against the same search on an already-narrowed parent — and some operators
    cost nothing at all, so a beam that happens to retain the expensive parents
    costs more than ``children_max`` averages.

    The construction is structural rather than statistical:

    1. group the level's expansions by **parent geometry**;
    2. cost each parent over its own expansions, each counted once per profile it
       consumes, plus the statistics passes that parent actually pays for under
       the runtime's sharing rule;
    3. the beam holds up to ``parents_max`` states, so take the ``parents_max``
       most expensive parents. If the level has fewer distinct geometries than
       the beam can hold, pad with the most expensive rather than assuming the
       beam is under-filled.
    """
    total = 0.0
    per_level: list[dict[str, Any]] = []
    for entry in branching["per_level"]:
        level = entry["level"]
        parents = by_level.get(level)
        if not parents:
            continue
        totals = []
        for parent_hash, expansions in parents.items():
            lo, hi = stats_cost.get(parent_hash, (0.0, 0.0))
            totals.append(_parent_seconds(level, expansions, n_profiles, lo, hi)[1])
        totals.sort(reverse=True)
        k = int(entry["parents_max"])
        chosen = totals[:k]
        if len(chosen) < k:                       # beam wider than the geometry
            chosen += [totals[0]] * (k - len(chosen))
        level_seconds = sum(chosen)
        total += level_seconds
        per_level.append({
            "level": level,
            "parents_max": k,
            "distinct_parent_geometries": len(totals),
            "most_expensive_parent_seconds": totals[0],
            "least_expensive_parent_seconds": totals[-1],
            # Reported so the bound can be checked against the thing it replaces
            # WITHIN the same priced world. Comparing `seconds_hard` against an
            # average taken from the `high` world compares two different models
            # and cannot detect a hard bound that quietly went back to a mean.
            "mean_parent_seconds": sum(totals) / len(totals),
            "chosen_mean_seconds": level_seconds / k if k else 0.0,
            "padded_with_the_maximum": len(totals) < k,
            "level_seconds": level_seconds,
        })
    return total, per_level


# --- whole-search pricing ---------------------------------------------------


@dataclass
class SearchCostEstimate:
    hardware: HardwareProfile
    branching: Mapping[str, Any]
    seconds_low: float
    seconds_high: float
    usd_low: float
    usd_high: float
    peak_storage_bytes_retained: int
    peak_storage_bytes_working: int
    total_bytes_written: int
    peak_resident_bytes: int
    #: The conservative beam-compatible bound. `seconds_high` is an averaged
    #: projection and is NOT safe to authorize against; this is.
    seconds_hard: float = 0.0
    usd_hard: float = 0.0
    hard_per_level: list[dict[str, Any]] = field(default_factory=list)
    #: Which intact-reference mode each of the three figures was priced under.
    reference_modes: Mapping[str, str] = field(default_factory=dict)
    per_operator: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.as_dict(),
            "branching": dict(self.branching),
            "seconds_low": self.seconds_low, "seconds_high": self.seconds_high,
            "hours_low": self.seconds_low / 3600, "hours_high": self.seconds_high / 3600,
            "usd_low": round(self.usd_low, 4), "usd_high": round(self.usd_high, 4),
            "seconds_hard": self.seconds_hard,
            "hours_hard": self.seconds_hard / 3600,
            "usd_hard": round(self.usd_hard, 4),
            "hard_per_level": self.hard_per_level,
            "reference_modes": dict(self.reference_modes),
            "peak_storage_gib_retained": self.peak_storage_bytes_retained / 2**30,
            "peak_storage_gib_working": self.peak_storage_bytes_working / 2**30,
            "total_gib_written": self.total_bytes_written / 2**30,
            "peak_resident_gib": self.peak_resident_bytes / 2**30,
            "per_operator": self.per_operator,
            "notes": list(self.notes),
        }


def price_search(root_spec: ArchSpec, target_spec: ArchSpec,
                 adapter: ArchitectureAdapter,
                 impls: Sequence[OperatorImplementation], *,
                 calibration_tokens: int, suite_tokens: int, seq_len: int,
                 n_profiles: int, beam_width: int, hardware: HardwareProfile,
                 warmup_levels: int = 0,
                 composite: Sequence[OperatorImplementation] = (),
                 depth_cached_fraction: float = 0.0,
                 hard_reference_mode: str = "recomputed") -> SearchCostEstimate:
    """Price the search from the expansions it will actually perform.

    Three things this gets right that the first version did not, and they do not
    cancel:

    * **Composite is costed.** `composite.stage1_sandwich_v0` reached only the
      branching counts, so its leaves appeared in `states_materialized` and
      `leaves_max` while costing nothing. Each one consumes activation
      statistics, transforms, materializes, reloads, validates and is evaluated.
    * **Statistics are charged once per (parent, profile)**, matching `StatsCache`,
      instead of once per stats-consuming operator. `composite`, `ffn` and
      `width` share one collection per mixture at a parent — except at the root,
      where `_stats_key` returns `None` and the runtime genuinely cannot share.
    * **The non-FLOP path is priced.** save -> hash -> canonical reload ->
      validate is I/O, hashing and Python; the hardware anchor was measured on
      forward compute and does not cover it.
    """
    paths = enumerate_paths(root_spec, target_spec, adapter, impls)
    branching = branching_estimate(root_spec, target_spec, adapter, impls,
                                   n_profiles=n_profiles, beam_width=beam_width,
                                   warmup_levels=warmup_levels,
                                   include_composite=composite)
    by_impl = {i.impl_id: i for i in impls}
    multiplicity = profile_multiplicity(impls, n_profiles)
    composite_multiplicity = profile_multiplicity(composite, n_profiles)

    modes = {"low": "cached", "high": "partial", "hard": hard_reference_mode}

    # One activation-statistics pass per parent GEOMETRY, priced once and reused
    # by every consumer at that parent — which is what the runtime cache does.
    stats_cost: dict[str, tuple[float, float]] = {}

    def stats_for(parent_spec: ArchSpec) -> tuple[float, float]:
        key = parent_spec.spec_hash
        if key not in stats_cost:
            flops = calibration_tokens * forward_flops_per_token(parent_spec, seq_len)
            stats_cost[key] = (hardware.seconds_for(flops),
                               calibration_tokens * CPU_STATS_SECONDS_PER_TOKEN)
        return stats_cost[key]

    # level -> parent_spec_hash -> [Expansion], one entry per distinct
    # (parent geometry, implementation). The same pair recurs across enumerated
    # paths and must not be counted twice.
    worlds: dict[str, dict[int, dict[str, dict[str, Expansion]]]] = {
        k: {} for k in modes}
    child_sizes: dict[int, list[int]] = {}
    parent_specs: dict[str, ArchSpec] = {}

    def record(which: str, level: int, parent_spec: ArchSpec, child_spec: ArchSpec,
               impl: OperatorImplementation, mult: int) -> None:
        mode = modes[which]
        cost = operator_cost(
            impl, parent_spec, target_spec, adapter,
            calibration_tokens=calibration_tokens, seq_len=seq_len,
            hardware=hardware, depth_reference_mode=mode,
            depth_cached_fraction=depth_cached_fraction, include_stats=False)
        entry = Expansion(
            level=level, parent_spec_hash=parent_spec.spec_hash,
            impl_id=impl.impl_id, multiplicity=mult,
            consumes_stats=consumes_activation_stats(impl),
            operator_seconds_low=cost.seconds_low(),
            operator_seconds_high=cost.seconds_high(),
            eval_seconds=evaluation_cost(child_spec, suite_tokens=suite_tokens,
                                         seq_len=seq_len, hardware=hardware),
            overhead_seconds=materialization_overhead_seconds(cost.child_bytes),
            child_bytes=cost.child_bytes,
            peak_resident_bytes=cost.peak_resident_bytes)
        worlds[which].setdefault(level, {}).setdefault(
            parent_spec.spec_hash, {})[impl.impl_id] = entry
        parent_specs[parent_spec.spec_hash] = parent_spec

    for path in paths:
        for node in path:
            for which in modes:
                record(which, node.level, node.parent_spec, node.child_spec,
                       by_impl[node.impl_id], multiplicity[node.impl_id])
            child_sizes.setdefault(node.level, []).append(
                checkpoint_bytes(node.child_spec, adapter))

    # COMPOSITE. It applies only from the uncompressed root, so it is a level-0
    # expansion, and it branches over profiles because it consumes calibration.
    for impl in composite:
        mult = composite_multiplicity.get(impl.impl_id, n_profiles)
        child_spec = impl.plan(root_spec, target_spec, adapter,
                               {"n_calibration_items": 1}).result_spec
        for which in modes:
            record(which, 0, root_spec, child_spec, impl, mult)
        for _ in range(mult):
            child_sizes.setdefault(0, []).append(
                checkpoint_bytes(child_spec, adapter))

    def flatten(which: str) -> dict[int, dict[str, list[Expansion]]]:
        return {level: {ph: list(by_impl_map.values())
                        for ph, by_impl_map in parents.items()}
                for level, parents in worlds[which].items()}

    stats_by_parent = {ph: stats_for(spec) for ph, spec in parent_specs.items()}

    seconds_low = seconds_high = 0.0
    per_operator: list[dict[str, Any]] = []
    low_levels, high_levels = flatten("low"), flatten("high")
    for entry in branching["per_level"]:
        level = entry["level"]
        if level not in low_levels:
            continue
        def level_mean(levels, index):
            totals = []
            for ph, expansions in levels[level].items():
                lo, hi = stats_by_parent[ph]
                totals.append(_parent_seconds(level, expansions, n_profiles, lo, hi)[index])
            return sum(totals) / len(totals), max(totals)
        mean_low, _ = level_mean(low_levels, 0)
        mean_high, max_high = level_mean(high_levels, 1)
        # A parent's whole expansion set, so the multiplier is parents rather
        # than children: `children_max` counts individual expansions and would
        # multiply a per-parent total by the number of expansions in it.
        seconds_low += entry["parents_min"] * mean_low
        seconds_high += entry["parents_max"] * mean_high
        per_operator.append({
            "level": level,
            "parents_min": entry["parents_min"], "parents_max": entry["parents_max"],
            "children_min": entry["children_min"], "children_max": entry["children_max"],
            "mean_parent_seconds_low": mean_low, "mean_parent_seconds_high": mean_high,
            "max_parent_seconds_high": max_high,
            # A RANGE across the level's parents, not the first one's: at level 2
            # a parent with FFN and WIDTH still to apply pays two collections and
            # one with only DEPTH and ATTENTION left pays none, and reporting
            # whichever happened to be enumerated first reads as though the whole
            # level were free.
            "stats_collections_per_parent": {
                "min": min(stats_collections(level, e, n_profiles)
                           for e in low_levels[level].values()),
                "max": max(stats_collections(level, e, n_profiles)
                           for e in low_levels[level].values())},
            "implementations": sorted({e.impl_id
                                       for expansions in low_levels[level].values()
                                       for e in expansions}),
        })

    seconds_hard, hard_per_level = conservative_hard_seconds(
        flatten("hard"), stats_by_parent, branching, n_profiles)

    # The teacher's reference logits over the suite are computed once for the
    # whole search, not per candidate.
    teacher_ref = evaluation_cost(root_spec, suite_tokens=suite_tokens,
                                  seq_len=seq_len, hardware=hardware)
    seconds_low += teacher_ref
    seconds_high += teacher_ref
    seconds_hard += teacher_ref

    # Storage has three distinct numbers and conflating them understates the
    # disk a run actually needs.
    #
    #   peak_working  the worst moment: a level's whole child set is on disk at
    #                 once, because ranking cannot start until every child has
    #                 been measured, and the beam parents are still there because
    #                 they were the things expanded.
    #   total_written I/O volume over the run, which is what a network volume
    #                 charges for and what a slow disk turns into wall clock.
    #   retained_end  what survives once pruned weights are released.
    peak_working = 0
    total_written = 0.0
    for entry in branching["per_level"]:
        level = entry["level"]
        if level not in child_sizes:
            continue
        sizes = child_sizes[level]
        mean_child = sum(sizes) / len(sizes)
        resident = [e.peak_resident_bytes - e.child_bytes
                    for expansions in low_levels[level].values() for e in expansions]
        parents_resident = entry["parents_max"] * max(resident)
        peak_working = max(peak_working,
                           int(parents_resident + entry["children_max"] * mean_child))
        total_written += entry["children_max"] * mean_child

    retained = beam_width * max(max(v) for v in child_sizes.values())
    peak_resident = max(e.peak_resident_bytes
                        for parents in low_levels.values()
                        for expansions in parents.values() for e in expansions)

    notes = [
        f"hardware anchor: {hardware.source}",
        "statistics-pass cost is a range; the GPU/CPU split of the float64 "
        "accumulation has never been measured separately",
        "statistics are charged ONCE per (parent, profile), matching StatsCache — "
        "except at the root, where _stats_key returns None because the root has no "
        "artifact digest, so every stats-consuming expansion of the root collects "
        "its own",
        f"composite leaves are priced: {sum(composite_multiplicity.get(i.impl_id, n_profiles) for i in composite)} "
        "level-0 expansions, each consuming statistics, materializing, reloading, "
        "validating and being evaluated",
        f"non-FLOP overhead per child: {CHECKPOINT_IO_PASSES} x checkpoint bytes at "
        f"{CHECKPOINT_IO_BYTES_PER_SECOND / 1e6:.0f} MB/s plus "
        f"{PER_CHILD_FIXED_SECONDS:.0f} s fixed — the hardware anchor was measured "
        "on forward compute and does not cover save/hash/reload/validate",
        "storage: 'working' is the concurrent peak (a level's whole child set is "
        "on disk before ranking can start), 'written' is total I/O volume, "
        "'retained' is what survives once pruned weights are released",
        f"intact reference: low={modes['low']}, high={modes['high']} at "
        f"{depth_cached_fraction:.0%} resident, hard={modes['hard']}. The hard "
        "figure assumes NO reference caching unless a mechanical guarantee is "
        "passed, because the bounded cache guarantees a fraction and not the whole",
        "seconds_hard is the beam-compatible bound (the parents_max most expensive "
        "parent geometries per level); seconds_high is an average projection and "
        "must not be used as an authorization ceiling",
    ]
    if not hardware.measured:
        notes.append(f"{hardware.name} throughput is ESTIMATED, not measured")

    return SearchCostEstimate(
        hardware=hardware, branching=branching,
        seconds_low=seconds_low, seconds_high=seconds_high,
        usd_low=hardware.usd_for(seconds_low), usd_high=hardware.usd_for(seconds_high),
        seconds_hard=seconds_hard, usd_hard=hardware.usd_for(seconds_hard),
        hard_per_level=hard_per_level,
        reference_modes={**modes, "depth_cached_fraction": str(depth_cached_fraction)},
        peak_storage_bytes_retained=int(retained),
        peak_storage_bytes_working=int(peak_working),
        total_bytes_written=int(total_written),
        peak_resident_bytes=int(peak_resident),
        per_operator=per_operator, notes=notes)
