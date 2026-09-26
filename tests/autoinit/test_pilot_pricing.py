"""The pilot's pre-run pricing must not predict the result it is budgeting.

The failure this guards is specific and tempting: estimating B4 at a quarter
of B1 because it issues a quarter of the kernel launches. That is not an
estimate, it is the speedup the pilot exists to measure, asserted in advance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.operators.attention.gqa import causal_kl  # noqa: E402
from aadistill.initialization.operators.register import (  # noqa: E402
    register_builtin_operators)

from experiments.phase_c3 import pricing  # noqa: E402

#: `artifacts/` is gitignored, so the frozen mixture is absent on a fresh
#: clone. Skip on the TREE, never on the file: a missing mixture inside a
#: built tree is a real failure.
BUILT = (REPO / "artifacts/stage1").is_dir()
needs_mixture = pytest.mark.skipif(
    not BUILT, reason="artifacts/ is gitignored and not built in this tree")


@pytest.fixture(autouse=True)
def registered():
    register_builtin_operators()
    causal_kl.register(replace=True)
    yield
    causal_kl.unregister()


@pytest.fixture(scope="module")
def doc():
    if not BUILT:
        pytest.skip("artifacts/ is not built in this tree")
    register_builtin_operators()
    causal_kl.register(replace=True)
    try:
        return pricing.derive(REPO, price_per_hour_usd=1.09, tflops=90.0)
    finally:
        causal_kl.unregister()


# --- the property the whole derivation exists to have ---------------------

@needs_mixture
def test_the_two_arms_price_identically(doc):
    b1, b4 = doc["arms"]["B1"], doc["arms"]["B4"]
    assert b1["item_forward_equivalents"] == b4["item_forward_equivalents"]
    assert b1["estimated_gpu_seconds"] == b4["estimated_gpu_seconds"]
    assert doc["derived"]["arms_price_identically"] is True


@needs_mixture
def test_b4_is_not_priced_at_a_quarter(doc):
    """Stated as the specific arithmetic that must not appear."""
    b1, b4 = doc["arms"]["B1"], doc["arms"]["B4"]
    assert b4["estimated_usd"] != pytest.approx(b1["estimated_usd"] / 4)


def test_the_cost_model_cannot_see_the_batch_size():
    """Structural, not promised: `operator_cost` has no config parameter, so
    it is incapable of pricing the two arms differently."""
    import inspect

    from aadistill.runtime.cost import operator_cost

    assert "config" not in inspect.signature(operator_cost).parameters


# --- the five quantities are reported separately --------------------------

@needs_mixture
def test_physical_invocations_fall_but_the_work_does_not(doc):
    b1, b4 = doc["arms"]["B1"], doc["arms"]["B4"]
    assert b4["physical_forward_invocations"] < b1["physical_forward_invocations"]
    assert doc["derived"]["physical_invocation_ratio_b1_over_b4"] > 3.5
    assert b1["valid_tokens"] == b4["valid_tokens"]


@needs_mixture
def test_b1_pads_nothing_and_b4_pads_a_lot(doc):
    b1, b4 = doc["arms"]["B1"], doc["arms"]["B4"]
    assert b1["padded_positions"] == 0
    assert b1["padding_overhead_ratio"] == 0.0
    assert b4["padded_positions"] > 0
    #: The number that decides whether a 1.25x gate is even reachable: B4
    #: computes this fraction MORE token-positions than B1, so the packing
    #: has to win that back before it wins anything.
    assert b4["padding_overhead_ratio"] > 0.2, (
        "the frozen mixture's lengths are ragged; near-zero padding would "
        "mean the grouping is not what the operator will use")
    assert doc["derived"]["b4_extra_positions_vs_b1"] == b4["padded_positions"]


@needs_mixture
def test_it_prices_the_real_frozen_mixture(doc):
    assert doc["mixture"]["n_items"] == 67
    #: The same token total the frozen C1 ATTENTION treatment recorded.
    assert doc["arms"]["B1"]["valid_tokens"] == 59_830


@needs_mixture
def test_the_work_is_one_ablation_per_head_plus_one_reference(doc):
    g = doc["geometry"]
    assert g["ablations"] == g["layers"] * g["parent_q_heads"] == 28 * 32
    expected = (g["ablations"] + 1) * doc["mixture"]["n_items"]
    assert doc["arms"]["B1"]["item_forward_equivalents"] == expected == 60_099


@needs_mixture
def test_peak_logit_residency_is_reported_and_fits_the_card(doc):
    """The launch needs this number, and B4 is where it grows.

    Two blocks are live at once -- the group's reference for the whole group,
    and one ablated block deleted each iteration -- so the bound follows the
    real free() sites rather than a per-step model.
    """
    b1, b4 = doc["arms"]["B1"], doc["arms"]["B4"]
    assert b4["peak_logit_bytes"] == 4 * b1["peak_logit_bytes"]
    assert b1["peak_logit_gib"] > 0
    #: An L40S is 48 GiB and the bf16 parent is ~1.4 GiB; anything close to
    #: the card here means the arm cannot run as specified.
    assert b4["peak_logit_gib"] < 24.0, (
        "B4's logit residency leaves no room for the model and the reducer")


@needs_mixture
def test_the_vram_bound_is_more_than_the_logit_blocks(doc):
    """4.62 GiB is the LOGITS. It is not the pod's VRAM requirement.

    At B4 the reducer's float32 transient is larger than the logit blocks it
    reduces: `forward_kl_mean_batch` upcasts a `[B, chunk, V]` slice and
    holds several such tensors inside one expression. Reporting the logit
    figure as the readiness bound would understate the requirement by more
    than a factor of two.
    """
    b4 = doc["arms"]["B4"]
    assert b4["reducer_transient_gib"] > b4["peak_logit_gib"], (
        "the reducer transient is being understated")
    assert b4["peak_vram_bound_gib"] > 2 * b4["peak_logit_gib"]
    assert b4["peak_vram_bound_gib"] == pytest.approx(
        b4["peak_logit_gib"] + b4["reducer_transient_gib"]
        + b4["model_weights_gib"] + b4["runtime_overhead_gib"], abs=0.01)


@needs_mixture
def test_the_bound_includes_the_weights_and_an_overhead_allowance(doc):
    for arm in doc["arms"].values():
        assert arm["model_weights_gib"] > 1.0, "the parent is 713M parameters"
        assert arm["runtime_overhead_gib"] > 0


@needs_mixture
def test_the_worst_arm_fits_the_only_authorized_card(doc):
    v = doc["vram_bound"]
    assert v["card"] == "L40S" and v["card_vram_gib"] == 48.0
    assert v["worst_arm_gib"] == max(a["peak_vram_bound_gib"]
                                     for a in doc["arms"].values())
    assert v["fits"] is True
    #: Headroom, not a squeeze: this bound is conservative but the pod also
    #: holds a tokenizer, the checkpoint writer's buffers and whatever the
    #: driver does between steps.
    assert v["worst_arm_gib"] < 0.6 * v["card_vram_gib"]


@needs_mixture
def test_a_heavier_dtype_still_has_to_fit():
    """If the parent were loaded in float32 the bound must still be checked,
    not assumed from the bf16 figure."""
    register_builtin_operators()
    causal_kl.register(replace=True)
    try:
        fp32 = pricing.derive(REPO, 1.09, 90.0, weight_bytes=4)
    finally:
        causal_kl.unregister()
    assert fp32["vram_bound"]["worst_arm_gib"] > 0
    assert fp32["arms"]["B4"]["model_weights_gib"] == pytest.approx(
        2 * 1.329, abs=0.05)


def test_the_reducer_block_count_matches_the_reducer_source():
    """The constant is read FROM the reducer's loop, not invented.

    If `forward_kl_mean_batch` grows another live `[B, chunk, V]` float32
    tensor, this bound silently understates. Counting the upcast expressions
    is crude, but it fails loudly when the loop changes, which a written
    constant would not.
    """
    import inspect

    from aadistill.initialization.statistics import contribution

    src = inspect.getsource(contribution.forward_kl_mean_batch)
    body = src[src.index("for a in range("):]
    upcasts = body.count(".float()") + body.count("log_softmax")
    assert upcasts <= pricing.REDUCER_LIVE_FP32_BLOCKS, (
        f"the reducer loop now holds more float32 blocks ({upcasts}) than "
        f"the VRAM bound allows for ({pricing.REDUCER_LIVE_FP32_BLOCKS})")
    assert pricing.REDUCER_CHUNK == 512, "the operator calls it with chunk=512"


@needs_mixture
def test_the_residency_is_derived_from_the_widest_group(doc):
    for arm in doc["arms"].values():
        expected = 2 * (arm["calibration_forward_batch_size"]
                        * arm["max_group_width"] * 151_936 * 2)
        assert arm["peak_logit_bytes"] == expected


# --- the geometry is checked against the frozen record --------------------

@needs_mixture
def test_the_derived_parent_matches_the_frozen_checkpoint():
    """`PARENT_Q_HEADS` is the one number not read out of a record, so the
    parameter count is the thing that catches it being wrong."""
    import json

    from aadistill.initialization.adapters import QWEN3_ADAPTER

    arms = json.loads((REPO / pricing.C1_ARM_IDENTITIES).read_text())
    assert QWEN3_ADAPTER.param_count(pricing.parent_spec(REPO)) == \
        arms["parent"]["num_parameters"]
    assert QWEN3_ADAPTER.param_count(pricing.target_spec(REPO)) == \
        arms["incumbent"]["num_parameters"]


@needs_mixture
def test_a_wrong_head_count_is_refused(monkeypatch):
    """The check above, shown to fire rather than assumed to."""
    monkeypatch.setattr(pricing, "PARENT_Q_HEADS", 24)
    with pytest.raises(SystemExit, match="PARENT_Q_HEADS=24 is wrong"):
        pricing.parent_spec(REPO)


@needs_mixture
def test_the_target_is_read_from_the_committed_path_record():
    import json

    doc_ = json.loads((REPO / pricing.C1_PATH_RECORD).read_text())
    assert pricing.target_spec(REPO).as_dict() == doc_["path"]["target_spec"]


# --- the estimate is labelled as one --------------------------------------

@needs_mixture
def test_the_price_is_not_presented_as_a_quote(doc):
    """`L40S_MEASURED.price_per_hour_usd` is historical profile metadata and
    has already been reported to a maintainer as if it were live. The record
    must say, in itself, that this number is not one."""
    h = doc["hardware_estimate"]
    assert "_not_a_quote" in h
    assert "re-query" in h["_not_a_quote"]


def test_the_module_pins_no_live_price_as_a_constant():
    """A default on a CLI flag is an estimate; a module constant reads as a
    fact and is what a later reader would copy into an authorization."""
    import ast

    tree = ast.parse((REPO / "scripts/experiments/phase_c3/pricing.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                name = getattr(t, "id", "")
                assert "PRICE" not in name.upper() and "USD" not in name.upper(), (
                    f"{name} pins a price in the module; the live quote is "
                    "re-queried at acquisition")
