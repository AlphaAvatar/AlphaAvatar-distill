"""Partial reference caching must be a speed change and nothing else.

Phase-B attempt 3 landed in the gap the all-or-nothing cache leaves. The intact
reference was **16.9 GiB**; the allowance was 66% of 20.3 GiB = **13.4 GiB**. 79%
of it fit, none of it was kept, and every one of 260 candidates per expansion
recomputed the whole reference — twice the forward passes, twelve expansions
over, 388.2 min of a 544.7 min search.

Caching a bounded prefix closes that. But a cache that changes what the operator
*decides* is not an optimization, it is a different experiment run under the same
implementation id. So the load-bearing test here is the three-way one: fully
cached, fully recomputed, and partially cached must agree on the removal order,
the kept layers, every candidate score in every round, and the reported metrics.

The memory fraction is untouched at 0.66. Partial mode spends **less** than the
allowance the all-or-nothing mode refused to spend at all.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.initialization  # noqa: F401,E402
from aadistill.initialization.specs.arch import get_adapter  # noqa: E402
from aadistill.initialization.operators.depth import causal_kl_greedy as depth_module  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorContext,
    get_implementation,
)

ADAPTER = get_adapter("qwen3")
IMPL = "depth.causal_kl_greedy_v1"


def run_with_memory(model, parent_spec, target_spec, items, profile,
                    monkeypatch, *, available_bytes):
    """Run the operator with the REAL admission logic against a stated budget.

    Forcing `enabled` after construction — what the older helper did — no longer
    describes the object: `admitted` is decided in `__init__`, so a flag flipped
    afterwards would leave every item resident while the record claimed the slow
    path. Driving the memory probe instead exercises the code that actually runs.
    """
    monkeypatch.setattr(depth_module, "_host_available_memory_bytes",
                        lambda: (available_bytes, "test"))
    return get_implementation(IMPL).execute(OperatorContext(
        adapter=ADAPTER, model=model, parent_spec=parent_spec,
        target_spec=target_spec, profile=profile, calibration_items=items,
        seed=1234))


def reference_bytes(model, items) -> int:
    positions = sum(int(i["input_ids"].shape[1]) - 1 for i in items)
    return positions * int(model.config.vocab_size) * next(
        model.parameters()).dtype.itemsize


# --- the three modes exist and are selected by the measurement ---------------


def test_the_three_modes_are_reachable_and_named(teacher, calibration_items,
                                                 monkeypatch):
    total = reference_bytes(teacher, calibration_items)
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION

    def build(available):
        monkeypatch.setattr(depth_module, "_host_available_memory_bytes",
                            lambda: (available, "test"))
        return depth_module._ReferenceLogits(teacher, calibration_items, "cpu")

    plenty = build(int(total / frac) * 4)
    assert plenty.mode == "cached" and plenty.enabled is True
    assert plenty.admitted_bytes == total

    nothing = build(1)
    assert nothing.mode == "recomputed" and nothing.enabled is False
    assert nothing.admitted == set() and nothing.admitted_bytes == 0

    # Sized so roughly half the mixture fits, which is attempt 3's situation.
    half = build(int(total * 0.5 / frac))
    assert half.mode == "partial" and half.enabled is False
    assert 0 < len(half.admitted) < len(calibration_items)
    assert half.admitted_bytes <= frac * (total * 0.5 / frac) + 1


def test_partial_admission_never_exceeds_the_unchanged_budget(
        teacher, calibration_items, monkeypatch):
    """0.66 is not raised, and partial mode stays strictly inside it."""
    total = reference_bytes(teacher, calibration_items)
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION
    assert frac == pytest.approx(0.66)
    for share in (0.1, 0.25, 0.5, 0.75, 0.9):
        available = int(total * share / frac)
        monkeypatch.setattr(depth_module, "_host_available_memory_bytes",
                            lambda a=available: (a, "test"))
        cache = depth_module._ReferenceLogits(teacher, calibration_items, "cpu")
        assert cache.admitted_bytes <= frac * available, share


def test_admission_follows_the_mixture_order_not_the_item_sizes(
        teacher, monkeypatch):
    """A size-greedy rule would make the resident set depend on the host.

    The shared fixture cannot see this: every item in it is 24 tokens long, so
    sorting by size is a stable no-op and the assertion passes against a sorted
    implementation. A first mutation pass proved exactly that — swapping the loop
    for `sorted(items, key=size)` killed nothing. These items differ in length so
    the two rules disagree, and the budget is chosen so they disagree *here*.
    """
    lengths = [41, 6, 31, 7]
    items = [{"item_id": f"i{n}", "input_ids": torch.ones(1, ln, dtype=torch.long)}
             for n, ln in enumerate(lengths)]
    per_position = int(teacher.config.vocab_size) * next(
        teacher.parameters()).dtype.itemsize
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION
    # Fits the first two in mixture order (40 + 5 positions), and would fit three
    # different ones if admission sorted by size (5 + 6 + 30 = 41). Rounded UP:
    # `int()` truncation put the allowance a fraction below 45 positions and
    # admitted only the first item.
    budget_positions = 46
    monkeypatch.setattr(depth_module, "_host_available_memory_bytes",
                        lambda: (math.ceil(budget_positions * per_position / frac), "test"))

    cache = depth_module._ReferenceLogits(teacher, items, "cpu")
    assert cache.mode == "partial"
    assert cache.admitted == {"i0", "i1"}, (
        f"admitted {sorted(cache.admitted)}; the rule must be 'the first k items "
        "of the frozen mixture that fit', not the k smallest — a size-greedy "
        "resident set changes with whatever memory the host happened to have")

    order = [i["item_id"] for i in items]
    admitted_in_order = [i for i in order if i in cache.admitted]
    assert admitted_in_order == order[:len(admitted_in_order)]


# --- the load-bearing equivalence -------------------------------------------


def test_all_three_modes_produce_IDENTICAL_operator_decisions(
        teacher, teacher_spec, target_spec, calibration_items, profile,
        monkeypatch):
    """Cached, partial and recomputed must be the same experiment.

    Not `approx`: the reference is the same deterministic no-grad forward and
    `distortion` upcasts to float32 in chunks either way, so the reduced values
    are the same floats whichever path produced the tensor.
    """
    total = reference_bytes(teacher, calibration_items)
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION

    hot = run_with_memory(teacher, teacher_spec, target_spec, calibration_items,
                          profile, monkeypatch, available_bytes=int(total / frac) * 4)
    part = run_with_memory(teacher, teacher_spec, target_spec, calibration_items,
                           profile, monkeypatch,
                           available_bytes=int(total * 0.5 / frac))
    cold = run_with_memory(teacher, teacher_spec, target_spec, calibration_items,
                           profile, monkeypatch, available_bytes=1)

    assert hot.artifacts["reference_cache"]["mode"] == "cached"
    assert part.artifacts["reference_cache"]["mode"] == "partial"
    assert cold.artifacts["reference_cache"]["mode"] == "recomputed"

    for other, name in ((part, "partial"), (cold, "recomputed")):
        assert other.trace["removal_order"] == hot.trace["removal_order"], name
        assert other.trace["kept_layers"] == hot.trace["kept_layers"], name
        assert other.trace["removed_layers"] == hot.trace["removed_layers"], name
        assert other.local_metrics.values == hot.local_metrics.values, name
        assert other.local_metrics.detail == hot.local_metrics.detail, name
        rounds_a = hot.artifacts["search_rounds"]
        rounds_b = other.artifacts["search_rounds"]
        assert len(rounds_a) == len(rounds_b), name
        for ra, rb in zip(rounds_a, rounds_b):
            assert ra["table"] == rb["table"], (
                f"{name} disagrees with the fully cached run on a candidate "
                "score; the cache would silently change the depth map")


def test_the_partial_path_actually_recomputes_and_actually_hits(
        teacher, teacher_spec, target_spec, calibration_items, profile,
        monkeypatch):
    """Guards the guard: equivalence proves nothing if partial ran as cached."""
    total = reference_bytes(teacher, calibration_items)
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION
    part = run_with_memory(teacher, teacher_spec, target_spec, calibration_items,
                           profile, monkeypatch,
                           available_bytes=int(total * 0.5 / frac))
    counters = part.artifacts["reference_counters"]
    assert counters["reference_hits"] > 0, "nothing was reused; this is the cold path"
    assert counters["reference_recomputes"] > 0, "nothing was recomputed; this is the hot path"
    decision = part.artifacts["reference_cache"]
    assert 0 < decision["items_cached"] < decision["items_total"]
    assert decision["items_recomputed_per_candidate"] > 0


def test_partial_mode_does_strictly_less_work_than_recomputing(
        teacher, teacher_spec, target_spec, calibration_items, profile,
        monkeypatch):
    """The point of the change, measured in forwards rather than asserted."""
    total = reference_bytes(teacher, calibration_items)
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION
    part = run_with_memory(teacher, teacher_spec, target_spec, calibration_items,
                           profile, monkeypatch,
                           available_bytes=int(total * 0.5 / frac))
    cold = run_with_memory(teacher, teacher_spec, target_spec, calibration_items,
                           profile, monkeypatch, available_bytes=1)
    # A reference forward happens on a miss, whether or not the result is kept.
    part_forwards = (part.artifacts["reference_counters"]["reference_fills"]
                     + part.artifacts["reference_counters"]["reference_recomputes"])
    cold_forwards = (cold.artifacts["reference_counters"]["reference_fills"]
                     + cold.artifacts["reference_counters"]["reference_recomputes"])
    assert part_forwards < cold_forwards, (part_forwards, cold_forwards)


def test_the_decision_record_keeps_its_original_boolean_meaning(
        teacher, calibration_items, monkeypatch):
    """`cached` is read by existing records: True iff the WHOLE reference fits."""
    total = reference_bytes(teacher, calibration_items)
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION
    monkeypatch.setattr(depth_module, "_host_available_memory_bytes",
                        lambda: (int(total * 0.5 / frac), "test"))
    decision = depth_module._ReferenceLogits(
        teacher, calibration_items, "cpu").decision()
    assert decision["cached"] is False, (
        "a partially cached reference is not a cached one; flipping this would "
        "make every historical reference_cache record ambiguous")
    assert decision["mode"] == "partial"
    assert decision["fallback"]


def test_timing_is_recorded_and_stays_out_of_the_metrics(
        teacher, teacher_spec, target_spec, calibration_items, profile,
        monkeypatch):
    """Telemetry must never reach a value that a state hashes or ranks on."""
    outcome = run_with_memory(teacher, teacher_spec, target_spec,
                              calibration_items, profile, monkeypatch,
                              available_bytes=1)
    timing = outcome.artifacts["timing"]
    assert timing["candidate_subsets"] > 0
    assert timing["ablated_forwards"] > 0
    assert set(timing) >= {"reference_seconds", "ablated_seconds",
                           "distortion_seconds", "item_seconds"}
    # Not in the trace, not in the metrics.
    assert "timing" not in outcome.trace
    for key in outcome.local_metrics.values:
        assert "second" not in key and "timing" not in key


# --- the canonical batch: cache state must not move the scientific forward ---


def _cache_for(teacher, items, share, monkeypatch):
    """A cache forced into `cached` / `partial` / `recomputed` by its budget."""
    probe = depth_module._ReferenceLogits(teacher, items, "cpu")
    frac = depth_module._ReferenceLogits.BUDGET_FRACTION
    monkeypatch.setattr(depth_module, "_available_memory_bytes",
                        lambda device: (int(probe.estimate_bytes * share / frac),
                                        "test"))
    return depth_module._ReferenceLogits(teacher, items, "cpu")


def _ragged_items(n=4, vocab=128, seed=77):
    """Deliberately ragged, so a sub-batch of missing rows would pad differently."""
    torch.manual_seed(seed)
    lengths = (31, 12, 24, 9)[:n]
    return [{"item_id": f"i{k}",
             "input_ids": torch.randint(1, vocab, (1, L)),
             "domain": "general" if k % 2 else "math",
             "subtype": "text" if k % 2 else "arith"}
            for k, L in enumerate(lengths)]


@pytest.mark.parametrize("share,expected_mode",
                         [(10.0, "cached"), (0.5, "partial"), (0.0, "recomputed")])
def test_the_reference_block_has_the_canonical_shape_in_every_cache_mode(
        teacher, monkeypatch, share, expected_mode):
    """The shape the reducer sees must not depend on runtime memory state.

    A partially cached batch used to build a SMALLER `ItemBatch` of only the
    missing rows. The per-item logits agreed to within float tolerance, but the
    batch width — and therefore the kernel shape and the reduction schedule —
    became a function of how much memory the host happened to have free. That
    put the boundary between "cached" and "recomputed" inside a scientific
    measurement.
    """
    from aadistill.initialization.calibration.batching import build_batch, resolve_pad_id

    items = _ragged_items()
    batch = build_batch(items, pad_id=resolve_pad_id(teacher))
    cache = _cache_for(teacher, items, share, monkeypatch)
    assert cache.mode == expected_mode, cache.mode

    block = cache.reference_block(batch)
    width = batch.input_ids.shape[1] - 1
    assert block.shape[0] == len(items)
    assert block.shape[1] == width, (
        f"{expected_mode}: the reference block is {block.shape[1]} wide but the "
        f"canonical batch is {width}; cache state changed the forward's shape")

    #: Warm it and ask again — the shape must still be canonical, which is the
    #: case a partial cache would previously have got wrong.
    again = cache.reference_block(batch)
    assert again.shape == block.shape


def test_every_cache_mode_reduces_to_the_same_per_item_kl(teacher, monkeypatch):
    """Same numbers in all three modes, compared against the scalar oracle.

    This is what the canonical-batch rule buys: the KL an item receives does not
    depend on whether its neighbours happened to be resident.
    """
    from aadistill.initialization.calibration.batching import build_batch, resolve_pad_id
    from aadistill.initialization.statistics.contribution import (
        forward_kl_mean, forward_kl_mean_batch)

    items = _ragged_items()
    batch = build_batch(items, pad_id=resolve_pad_id(teacher))
    #: `bypassed_blocks` refuses a KV cache, because a cache is indexed by
    #: layer_idx and a filtered layer list no longer matches it. The operator
    #: turns this off in `apply`; a test that reaches the bypass directly owes
    #: the same precondition.
    teacher.config.use_cache = False
    skip = frozenset({1})
    ablated = depth_module._forward_logit_block(teacher, batch, "cpu", skip)
    mask = batch.prediction_mask()

    per_mode = {}
    for share, mode in ((10.0, "cached"), (0.5, "partial"), (0.0, "recomputed")):
        cache = _cache_for(teacher, items, share, monkeypatch)
        assert cache.mode == mode
        block = cache.reference_block(batch)
        per_mode[mode] = forward_kl_mean_batch(block, ablated, mask).tolist()

    #: All three agree with each other...
    for mode, values in per_mode.items():
        for other, others in per_mode.items():
            for a, b in zip(values, others):
                assert abs(a - b) < 1e-9, f"{mode} vs {other}: {a} != {b}"
    #: ...and with the scalar oracle, per item.
    reference_rows = depth_module._forward_logits_batch(teacher, batch, "cpu")
    ablated_rows = batch.split_predictions(ablated)
    for index, (r, a) in enumerate(zip(reference_rows, ablated_rows)):
        want = forward_kl_mean(r, a)
        got = per_mode["cached"][index]
        assert abs(got - want) < 1e-6, f"item {index}: {got} vs oracle {want}"


def test_a_partial_boundary_inside_a_micro_batch_still_forwards_the_whole_batch(
        teacher, monkeypatch):
    """The specific case the rule is about: some rows admitted, some not."""
    from aadistill.initialization.calibration.batching import build_batch, resolve_pad_id

    items = _ragged_items()
    batch = build_batch(items, pad_id=resolve_pad_id(teacher))
    cache = _cache_for(teacher, items, 0.5, monkeypatch)
    assert 0 < len(cache.admitted) < len(items), (
        "fixture does not straddle the admission boundary")

    seen = []
    real = depth_module._forward_logit_block

    def watching(model, b, device, skip=frozenset()):
        seen.append(b.size)
        return real(model, b, device, skip)

    monkeypatch.setattr(depth_module, "_forward_logit_block", watching)
    cache.reference_block(batch)      # first pass: nothing resident
    cache.reference_block(batch)      # second: the admitted rows now are
    assert seen == [len(items), len(items)], (
        f"reference forwards ran at batch sizes {seen}; a partially cached "
        "batch must still forward the WHOLE canonical batch, not a sub-batch "
        "of the missing rows")


def test_a_fully_cached_batch_performs_no_forward_at_all(teacher, monkeypatch):
    from aadistill.initialization.calibration.batching import build_batch, resolve_pad_id

    items = _ragged_items()
    batch = build_batch(items, pad_id=resolve_pad_id(teacher))
    cache = _cache_for(teacher, items, 10.0, monkeypatch)
    cache.reference_block(batch)

    calls = []
    real = depth_module._forward_logit_block
    monkeypatch.setattr(depth_module, "_forward_logit_block",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    cache.reference_block(batch)
    assert not calls, "a fully cached batch ran a forward anyway"


def test_a_cached_row_owns_its_storage_and_is_only_its_valid_slice(
        teacher, monkeypatch):
    """Budget correctness: `(T_i - 1) * V`, not the padded batch block."""
    from aadistill.initialization.calibration.batching import build_batch, resolve_pad_id

    items = _ragged_items()
    batch = build_batch(items, pad_id=resolve_pad_id(teacher))
    cache = _cache_for(teacher, items, 10.0, monkeypatch)
    cache.reference_block(batch)
    for index, item in enumerate(items):
        row = cache._cache.get(item["item_id"])
        assert row is not None
        assert row._base is None, (
            f"{item['item_id']} is cached as a VIEW, pinning the padded block")
        assert row.shape[0] == int(item["input_ids"].shape[1]) - 1, (
            f"{item['item_id']} cached {row.shape[0]} positions; its valid "
            "prediction slice is shorter")
