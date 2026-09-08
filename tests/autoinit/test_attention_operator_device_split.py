"""The treatment operator's device boundary, as C1 attempt 9 found it.

Attempt 9 reached stage F on an L40S and died:

    RuntimeError: Expected all tensors to be on the same device, but found at
    least two devices, cuda:0 and cpu!
    init/attention_stats.py:146, head_write_energy
    scores[h] = (gram * m[h]).sum() / n

`AttentionHeadStatsCollector.state()` returns a HOST-RESIDENT snapshot on
purpose — that is the evidence form. `apply()` passed it straight to
`head_write_energy`, where it met `o_proj.weight` on `cuda:0`. Nothing about the
persistent cache POLICY was wrong; the per-invocation working copy was missing.

A second defect sat behind it, unreachable until the first was fixed:
`scores = torch.empty(num_heads, dtype=torch.float64)` defaults to CPU, so the
first `scores[h] = ...` on a GPU is another cross-device store.

Neither is observable on a one-device box by ordinary arithmetic — which is why
`attention.activation_importance_v1` passed every `$0` regression and then cost a
paid pod. So both are modelled here with the instruments this repository already
has: `HostCacheTensor` for the logical split, `RecordFactories` for placement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.specs.arch import get_adapter  # noqa: E402
from aadistill.initialization.device import model_device  # noqa: E402
from aadistill.initialization.operators import attention_activation  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorContext,
    get_implementation,
)
from aadistill.initialization.statistics.attention import head_write_energy  # noqa: E402

from device_split import CrossDeviceUse, on_cache_device  # noqa: E402
from factory_placement import RecordFactories  # noqa: E402

ADAPTER = get_adapter("qwen3")
IMPL = "attention.activation_importance_v1"

def out_projections_of(model):
    """The attention-output projections, resolved the way production does."""
    from aadistill.initialization.operators.attention_activation import attention_out_projection
    return [attention_out_projection(ADAPTER, b) for b in ADAPTER.blocks(model)]



@pytest.fixture(autouse=True)
def registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


@pytest.fixture
def geo():
    from conftest import TEACHER_GEOMETRY
    return dict(TEACHER_GEOMETRY, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8)


@pytest.fixture
def target(geo):
    return dict(geo, num_attention_heads=2)


def _ctx(teacher, geo, target, items, profile):
    from aadistill.initialization.specs.arch import ArchSpec
    return OperatorContext(
        adapter=ADAPTER, model=teacher,
        parent_spec=ArchSpec.of("qwen3", geo),
        target_spec=ArchSpec.of("qwen3", target),
        profile=profile, calibration_items=items, seed=1234,
        device=str(model_device(teacher)))


def _split_the_snapshot(monkeypatch):
    """Label what `state()` returns, exactly as the host snapshot really is.

    The operator collects directly rather than through `ctx.cached_stats`, so
    the split is applied at `state()` — the real boundary — instead of at a
    cache the operator never consults.
    """
    real_state = attention_activation.AttentionHeadStatsCollector.state

    def labelled_state(self):
        return on_cache_device(real_state(self))

    monkeypatch.setattr(attention_activation.AttentionHeadStatsCollector,
                        "state", labelled_state)


# --- A. the real operator transfers before touching a weight ----------------

def test_A_the_operator_transfers_the_snapshot_before_it_meets_a_weight(
        teacher, geo, target, calibration_items, profile, monkeypatch):
    """The real `execute()`, with the snapshot logically on another device."""
    _split_the_snapshot(monkeypatch)
    impl = get_implementation(IMPL)

    outcome = impl.execute(_ctx(teacher, geo, target, calibration_items, profile))

    child = ADAPTER.spec_of(outcome.model)
    assert child["num_attention_heads"] == target["num_attention_heads"]
    assert child["num_key_value_heads"] == geo["num_key_value_heads"]


def test_A2_mutation_removing_stats_to_reproduces_the_L40S_failure(
        teacher, geo, target, calibration_items, profile, monkeypatch):
    """The instrument, verified against itself at the line that actually failed.

    With `stats_to` neutered, the host snapshot reaches `head_write_energy`
    unmoved and meets `o_proj.weight` — the same product, in the same function,
    that ended attempt 9.
    """
    _split_the_snapshot(monkeypatch)
    monkeypatch.setattr(attention_activation, "stats_to",
                        lambda state, device: state)          # the old behaviour
    impl = get_implementation(IMPL)

    with pytest.raises(CrossDeviceUse, match="persistent cache device"):
        impl.execute(_ctx(teacher, geo, target, calibration_items, profile))


def test_A3_the_snapshot_itself_stays_host_resident(teacher, geo, target,
                                                    calibration_items, profile):
    """The repair must not make the evidence form CUDA-resident."""
    seen = {}
    real_state = attention_activation.AttentionHeadStatsCollector.state

    def watch(self):
        out = real_state(self)
        seen["snapshot_device"] = out["attn_head_sqsum"].device
        return out

    attention_activation.AttentionHeadStatsCollector.state = watch
    try:
        get_implementation(IMPL).execute(
            _ctx(teacher, geo, target, calibration_items, profile))
    finally:
        attention_activation.AttentionHeadStatsCollector.state = real_state
    assert seen["snapshot_device"].type == "cpu"


def test_A4_the_device_accumulator_is_released_after_the_snapshot(
        teacher, geo, target, calibration_items, profile):
    """Snapshot, release, then build the working copy — never three at once."""
    order: list[str] = []
    C = attention_activation.AttentionHeadStatsCollector
    real_state, real_release = C.state, C.release

    def s(self):
        order.append("state")
        return real_state(self)

    def r(self):
        order.append("release")
        return real_release(self)

    C.state, C.release = s, r
    try:
        get_implementation(IMPL).execute(
            _ctx(teacher, geo, target, calibration_items, profile))
    finally:
        C.state, C.release = real_state, real_release
    assert order == ["state", "release"]


def test_A5_a_released_collector_refuses_to_report_state(teacher, geo):
    from aadistill.initialization.statistics.attention import AttentionHeadStatsCollector

    c = AttentionHeadStatsCollector(teacher, out_projections_of(teacher),
                                    num_heads=geo["num_attention_heads"],
                                    head_dim=geo["head_dim"])
    c.close()
    c.release()
    with pytest.raises(ValueError, match="released"):
        c.state()


# --- B. the score vector is placed, not defaulted ---------------------------

def test_B_the_score_vector_factory_names_a_device(teacher, geo, target,
                                                   calibration_items, profile):
    """One PLACED score vector per block, from the real operator path.

    Scoped deliberately: `execute()` also builds the child model, and
    `transformers` allocates its parameters with unplaced `torch.empty` before
    `ChildBuilder` assigns into them. Those are not this repair's, and demanding
    every factory in a model construction name a device would be a different and
    much larger claim. What this asserts is that the score vector — the tensor
    that meets `o_proj.weight` — is placed, once per block.
    """
    n_blocks = geo["num_hidden_layers"]
    with RecordFactories() as rec:
        get_implementation(IMPL).execute(
            _ctx(teacher, geo, target, calibration_items, profile))

    placed = [c for c in rec.calls if c.name == "empty" and c.placed]
    assert len(placed) == n_blocks, (
        f"{len(placed)} placed torch.empty calls for {n_blocks} blocks; the "
        "score vector is allocated once per block and must name a device")


def test_B2_head_write_energy_places_the_vector_on_the_tensor_it_meets():
    """Read directly off the function, on a fixture whose device is known."""
    heads, d, hidden, n = 3, 4, 6, 10
    g = torch.Generator().manual_seed(7)
    state = {"attn_head_sqsum": torch.randn(1, heads, d, d, generator=g).double(),
             "attn_token_count": torch.tensor(n)}
    state["attn_head_sqsum"] = (state["attn_head_sqsum"]
                                @ state["attn_head_sqsum"].transpose(-1, -2))
    w = torch.randn(hidden, heads * d, generator=g).double()

    with RecordFactories() as rec:
        head_write_energy(state, 0, w, heads, d)
    empties = [c for c in rec.calls if c.name == "empty"]
    assert empties and all(c.placed for c in empties), empties


def test_B3_mutation_an_unplaced_empty_is_caught_even_though_cpu_math_works():
    """The point of the instrument: on one device the arithmetic still works.

    A test that only checked the returned numbers could never see this, which is
    why the defect reached a paid pod behind the first one.
    """
    import aadistill.initialization.statistics.attention as AS

    src = Path(AS.__file__).read_text()
    assert "device=w.device" in src, "the placement was removed"

    with RecordFactories() as rec:
        torch.empty(3, dtype=torch.float64)          # the mutated form
    assert rec.unplaced(), "RecordFactories cannot see an unplaced empty"


def test_B4_the_co_location_check_fails_closed():
    """`head_write_energy` reports a mismatch; it does not repair one."""
    heads, d, hidden, n = 2, 3, 5, 4
    m = torch.eye(d).double().expand(1, heads, d, d).clone()
    state = on_cache_device({"attn_head_sqsum": m,
                             "attn_token_count": torch.tensor(n)})
    w = torch.randn(hidden, heads * d).double()
    with pytest.raises((ValueError, CrossDeviceUse)):
        head_write_energy(state, 0, w, heads, d)


def test_B4b_the_check_names_both_devices_and_the_owner_of_the_transfer():
    """A fail-closed error is only useful if it says whose job the fix is."""
    import aadistill.initialization.statistics.attention as AS

    src = Path(AS.__file__).read_text()
    block = src.split("def head_write_energy")[1]
    assert "if m.device != w.device:" in block
    assert "stats_to" in block, "the error does not name the caller's remedy"


def test_B5_the_returned_vector_is_host_resident_for_the_selection_path():
    heads, d, hidden, n = 3, 4, 6, 10
    g = torch.Generator().manual_seed(11)
    a = torch.randn(1, heads, d, d, generator=g).double()
    state = {"attn_head_sqsum": a @ a.transpose(-1, -2),
             "attn_token_count": torch.tensor(n)}
    w = torch.randn(hidden, heads * d, generator=g).double()
    scores = head_write_energy(state, 0, w, heads, d)
    assert scores.device.type == "cpu"
    assert scores.dtype is torch.float64 and scores.shape == (heads,)
