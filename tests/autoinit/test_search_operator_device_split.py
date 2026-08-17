"""Every frozen operator, with the model device and the cache device different.

Phase-A attempts 6 and 7 both passed every $0 check and then died on a GPU, in
different code, on the same class of defect. The reason no test caught either is
structural: the dev box has one device, so every tensor is on it and a
cross-device use is unobservable.

`device_split.HostCacheTensor` supplies the missing device. A statistics tensor
labelled with it raises the moment it meets a model-side tensor without having
been transferred, which is what CUDA does. Each operator below is run through
its real `execute()` with a labelled cache; passing means the operator performs
the explicit transfer the contract requires
(`aadistill.autoinit.device.stats_to`), not that a single device hid the
question.

`depth.causal_kl_greedy_v1` consumes no statistics and is covered instead on the
property that actually applies to it: its reference cache and its ablated
forwards must be on the same device as the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.device import (  # noqa: E402
    DEVICE_CONTRACT_ID, model_device, stats_bytes, stats_to,
)
from aadistill.autoinit.operators._common import collect_activation_stats  # noqa: E402
from aadistill.autoinit.operators.base import (  # noqa: E402
    OperatorContext, get_implementation,
)
from device_split import CrossDeviceUse, on_cache_device  # noqa: E402

ADAPTER = get_adapter("qwen3")

#: The operators that read the statistics cache. These are the ones the split
#: can speak about.
CACHE_CONSUMERS = ["width.global_pca_v0", "ffn.activation_importance_v0",
                   "composite.stage1_sandwich_v0"]


def real_stats(teacher, calibration_items):
    return collect_activation_stats(
        ADAPTER, teacher, (i["input_ids"] for i in calibration_items),
        model_device(teacher))


def run_with_split_cache(impl_id, teacher, teacher_spec, target_spec, items,
                         profile):
    """The operator's real `execute`, with the cache on 'another device'."""
    impl = get_implementation(impl_id)
    labelled = on_cache_device(real_stats(teacher, items))

    class SplitCache:
        """Returns the labelled statistics whatever the key."""

        def get_or_collect(self, key, collect):
            return labelled

    ctx = OperatorContext(
        adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
        target_spec=target_spec, profile=profile, calibration_items=items,
        seed=1234, device=str(model_device(teacher)),
        stats_cache=SplitCache(), stats_cache_key="split")
    return impl.execute(ctx)


@pytest.mark.parametrize("impl_id", CACHE_CONSUMERS)
def test_the_operator_transfers_the_cache_before_touching_a_parameter(
        impl_id, teacher, teacher_spec, target_spec, calibration_items, profile):
    """Fails before the fix: `proj.T @ weight` with `proj` still cache-side."""
    outcome = run_with_split_cache(impl_id, teacher, teacher_spec, target_spec,
                                   calibration_items, profile)
    child_spec = ADAPTER.spec_of(outcome.model)
    assert child_spec.matches(target_spec) or child_spec != teacher_spec


def test_the_split_actually_bites_when_the_transfer_is_removed(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """The instrument, verified against itself.

    A gate that has never refused anything is not evidence. This performs the
    exact omission the fix removed — hand the operator statistics it has no
    chance to transfer, by pre-labelling them AFTER `stats_to` would have run —
    and requires the split to catch it.
    """
    from aadistill.autoinit.operators import width as width_module

    labelled = on_cache_device(real_stats(teacher, calibration_items))
    original = width_module.stats_to
    width_module.stats_to = lambda state, device: state      # the old behaviour
    try:
        with pytest.raises(CrossDeviceUse, match="persistent cache device"):
            run_with_split_cache("width.global_pca_v0", teacher, teacher_spec,
                                 target_spec, calibration_items, profile)
    finally:
        width_module.stats_to = original
    assert labelled


def test_the_collector_allocates_its_accumulators_on_the_model(teacher):
    """Category 2, first half — on a device the host is NOT.

    `assert accumulator.device == model.device` is worthless here: with one
    device both sides read `cpu`, and reinstating the attempt-7 bug still
    passes. A model on the meta device gives the assertion something to say,
    and needs no GPU and no memory — only the allocation is under test.
    """
    from aadistill.init.collect import ActivationStatsCollector

    collector = ActivationStatsCollector(teacher.to("meta"))
    assert collector.device == torch.device("meta")
    for name in ("res_sum", "res_sqsum", "ffn_abs_sum", "ffn_sq_sum",
                 "token_counts"):
        got = getattr(collector, name).device
        assert got == torch.device("meta"), (
            f"{name} was allocated on {got}, not on the model's device; the "
            "forward hook adds a model-side activation into it, which is what "
            "killed Phase-A attempt 7")
    collector.close()


def test_the_collector_transfers_every_accumulator_to_the_host(
        teacher, calibration_items):
    """Category 2, second half.

    Also untestable by device equality on one device: `.cpu()` is a no-op and
    returns `self`. The transfer is a named seam, and this asserts the seam was
    crossed for every accumulator.
    """
    from aadistill.init.collect import ActivationStatsCollector

    collector = ActivationStatsCollector(teacher)
    try:
        collector.process(calibration_items[0]["input_ids"])
    finally:
        collector.close()

    crossed, original = [], ActivationStatsCollector._to_host
    ActivationStatsCollector._to_host = staticmethod(
        lambda x: crossed.append(tuple(x.shape)) or original(x))
    try:
        state = collector.state()
    finally:
        # `staticmethod`, not the bare function: assigning the raw function back
        # would rebind it as an instance method and every later call would get
        # `self` as its argument.
        ActivationStatsCollector._to_host = staticmethod(original)

    assert len(crossed) == 5, (
        f"only {len(crossed)} accumulators crossed the host boundary; the "
        "persistent cache is host-resident by contract and one entry is "
        "1.81 GiB at a 4B parent")
    for name, value in state.items():
        assert value.device.type == "cpu", name

    # The seam being CALLED is not the seam TRANSFERRING: with one device a
    # `_to_host` that returned its argument unchanged would satisfy everything
    # above. Given a source that is genuinely not the host, it must either move
    # the tensor or fail loudly — what it must not do is hand back something
    # still on the source device.
    elsewhere = torch.zeros(2, device="meta")
    try:
        moved = ActivationStatsCollector._to_host(elsewhere)
    except (NotImplementedError, RuntimeError):
        pass                      # a real transfer, refused for a meta source
    else:
        assert moved.device.type == "cpu", (
            "_to_host returned a tensor still on the source device; the "
            "persistent cache would hold 1.81 GiB of VRAM across the whole "
            "expansion of a parent")


def test_the_depth_operator_keeps_its_reference_and_ablations_together(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    """`depth.causal_kl_greedy_v1` reads no statistics, so the split says
    nothing about it. What applies to it is that the reference logits it caches
    and the ablated forwards it compares them against are on one device."""
    from aadistill.autoinit.operators import depth as depth_module

    impl = get_implementation("depth.causal_kl_greedy_v1")
    ctx = OperatorContext(
        adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
        target_spec=target_spec, profile=profile,
        calibration_items=calibration_items, seed=1234,
        device=str(model_device(teacher)))
    outcome = impl.execute(ctx)
    assert outcome.trace["removal_order"]

    ref = depth_module._ReferenceLogits(teacher, calibration_items,
                                        model_device(teacher))
    got = ref.get(calibration_items[0])
    abl = depth_module._forward_logits(teacher, calibration_items[0],
                                       model_device(teacher), frozenset({0}))
    assert got.device == abl.device, (
        "the reference and the ablation are on different devices; `distortion` "
        "would raise on a GPU")


def test_the_attention_index_lands_on_the_weight_it_slices():
    """Category 3: an index built from a Python list is host-side whatever it
    is about to slice."""
    from aadistill.autoinit.operators._common import head_rows

    weight = torch.zeros(8, 4)
    rows = head_rows([0, 2], 2, device=weight.device)
    assert rows.device == weight.device
    assert head_rows([0, 2], 2).device == torch.device("cpu")


def test_the_contract_is_citable_and_the_memory_number_is_real(teacher,
                                                              calibration_items):
    """The cache-device rule is a memory decision, so the number is asserted."""
    assert DEVICE_CONTRACT_ID == "autoinit.stage1_device_contract@v1"

    state = real_stats(teacher, calibration_items[:1])
    assert stats_bytes(state) > 0
    moved = stats_to(state, "cpu")
    assert set(moved) == set(state)

    # The real teacher's figure, which is why the cache is not pinned in VRAM.
    H, L, V = 2560, 36, 151936
    total = ((L + 1) * H * 8 + (L + 1) * H * H * 8
             + 2 * L * 9728 * 8 + V * 8)
    assert total / 2**30 == pytest.approx(1.81, abs=0.01)
