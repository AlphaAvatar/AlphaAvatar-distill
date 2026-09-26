"""The B1/B4 structural comparison, driven by real operator evidence.

The comparator runs once, on a paid pod, on the only pair of head maps that
matters. So it is exercised here against evidence the REAL operator produced
at toy scale, and against hand-built cases for the shapes a toy run does not
happen to contain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.profiles import NO_CALIBRATION  # noqa: E402
from aadistill.initialization.operators import get_implementation  # noqa: E402
from aadistill.initialization.operators.attention.gqa import causal_kl  # noqa: E402
from aadistill.initialization.operators.base import OperatorContext  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from conftest import build_tiny_model  # noqa: E402
from experiments.phase_c3.compare import (  # noqa: E402
    NARROW_MARGIN, compare_head_maps, speedup, verdict,
)

GEOMETRY = dict(hidden_size=32, num_hidden_layers=3, intermediate_size=48,
                num_attention_heads=8, num_key_value_heads=2, head_dim=8,
                vocab_size=128, tie_word_embeddings=True)
KEEP = 4


@pytest.fixture(autouse=True)
def registered():
    causal_kl.register(replace=True)
    yield
    causal_kl.unregister()


def items(seed=11):
    g = torch.Generator().manual_seed(seed)
    out = []
    for subtype, lens in (("general", (9, 14)), ("alpha", (11, 7)),
                          ("beta", (16, 10))):
        for k, n in enumerate(lens):
            out.append({"item_id": f"{subtype}/{k}",
                        "domain": "general" if subtype == "general" else "task",
                        "subtype": subtype,
                        "input_ids": torch.randint(0, 128, (1, n), generator=g)})
    return out


def evidence(batch_size, model=None, calib=None):
    geo = ArchSpec.of("qwen3", GEOMETRY)
    calib = calib or items()
    ctx = OperatorContext(
        adapter=QWEN3_ADAPTER, model=model or build_tiny_model(GEOMETRY),
        parent_spec=geo, target_spec=geo.replace(num_attention_heads=KEEP),
        profile=NO_CALIBRATION, calibration_items=calib, seed=0, device="cpu",
        config={"n_calibration_items": len(calib),
                "calibration_forward_batch_size": batch_size})
    out = get_implementation("attention.causal_kl_v1").execute(ctx)
    return out.artifacts["causal_head_evidence"]


# --- against evidence the real operator produced --------------------------

def test_identical_arms_compare_as_identical():
    """On CPU fp32 the two arms agree, so the comparator must say so rather
    than finding a difference in its own bookkeeping."""
    b1, b4 = evidence(1), evidence(4)
    c = compare_head_maps(b1, b4)
    assert c["identical"] is True
    assert c["layers"]["changed"] == 0
    assert c["gqa_groups"]["changed"] == 0
    assert c["retained_slots"]["changed"] == 0
    assert c["symmetric_difference_heads"] == 0
    assert c["changed_selections"] == []


def test_every_count_carries_its_denominator():
    """"N layers changed" is the number that overstates."""
    c = compare_head_maps(evidence(1), evidence(4))
    assert c["layers"]["total"] == GEOMETRY["num_hidden_layers"]
    assert c["gqa_groups"]["total"] == (GEOMETRY["num_hidden_layers"]
                                        * GEOMETRY["num_key_value_heads"])
    assert c["retained_slots"]["total"] == (GEOMETRY["num_hidden_layers"] * KEEP)
    assert "caveat" in " ".join(c["layers"]).lower() or any(
        "widely" in str(v) for v in c["layers"].values())


def test_the_drift_distribution_is_reported_with_its_n():
    c = compare_head_maps(evidence(1), evidence(4))
    d = c["score_drift"]
    assert d["n"] == GEOMETRY["num_hidden_layers"] * GEOMETRY["num_attention_heads"]
    for key in ("min", "max", "mean", "median", "p05", "p95", "abs_max", "std"):
        assert key in d
    assert d["abs_max"] < 1e-5, "CPU fp32 arms should barely drift"


def test_rank_correlation_is_reported_or_honestly_absent():
    c = compare_head_maps(evidence(1), evidence(4))
    assert c["rank_correlation"]["overall"] == pytest.approx(1.0, abs=1e-9)
    assert len(c["rank_correlation"]["per_layer"]) == GEOMETRY["num_hidden_layers"]


def test_the_cutoff_margins_of_both_arms_are_reported():
    c = compare_head_maps(evidence(1), evidence(4))
    for arm in ("b1", "b4"):
        m = c["cutoff_margins"][arm]
        assert m["n"] == (GEOMETRY["num_hidden_layers"]
                          * GEOMETRY["num_key_value_heads"])
        assert m["min"] >= 0, "a selected head cannot score below a rejected one"


# --- the shapes a toy agreement does not produce --------------------------

def _fake(sel_per_layer, margins, scores=None):
    layers = len(sel_per_layer)
    return {
        "head_scores": scores or [[1.0, 0.9, 0.8, 0.7] for _ in range(layers)],
        "gqa_decisions": [
            [{"group": 0, "member_heads": [0, 1, 2, 3],
              "ranked_heads": sorted(sel + [h for h in range(4) if h not in sel]),
              "selected_heads": sel, "cutoff_selected": 0.9,
              "cutoff_rejected": 0.9 - margins[i], "cutoff_margin": margins[i],
              "scores": [1.0, 0.9, 0.8, 0.7]}]
            for i, sel in enumerate(sel_per_layer)],
    }


def test_one_changed_slot_does_not_read_as_a_changed_everything():
    """Three layers, one slot different in one of them."""
    a = _fake([[0, 1], [0, 1], [0, 1]], [0.1, 0.1, 0.1])
    b = _fake([[0, 1], [0, 2], [0, 1]], [0.1, 0.1, 0.1])
    c = compare_head_maps(a, b)
    assert c["identical"] is False
    assert (c["layers"]["changed"], c["layers"]["total"]) == (1, 3)
    assert (c["gqa_groups"]["changed"], c["gqa_groups"]["total"]) == (1, 3)
    assert (c["retained_slots"]["changed"], c["retained_slots"]["total"]) == (1, 6)
    assert c["retained_slots"]["fraction"] == pytest.approx(1 / 6)
    assert c["symmetric_difference_heads"] == 2, "one gained, one lost"


def test_a_narrow_crossing_is_distinguished_from_a_wide_one():
    """The finding the integer counts cannot carry."""
    a = _fake([[0, 1], [0, 1]], [1e-12, 0.4])
    b = _fake([[0, 2], [0, 2]], [1e-12, 0.4])
    c = compare_head_maps(a, b)
    assert c["crossings"] == {"narrow": 1, "wide": 1,
                              "narrow_margin_threshold": NARROW_MARGIN,
                              "_threshold_decides_nothing": c["crossings"][
                                  "_threshold_decides_nothing"]}
    kinds = {d["layer"]: d["crossing"] for d in c["changed_selections"]}
    assert kinds == {0: "narrow", 1: "wide"}
    assert c["changed_selections"][0]["smallest_margin"] == pytest.approx(1e-12)


def test_each_changed_selection_names_what_moved():
    a = _fake([[0, 1]], [0.2])
    b = _fake([[0, 3]], [0.2])
    d = compare_head_maps(a, b)["changed_selections"][0]
    assert d["b1_selected"] == [0, 1] and d["b4_selected"] == [0, 3]
    assert d["gained_by_b4"] == [3] and d["lost_by_b4"] == [1]
    assert d["b1_cutoff_margin"] == d["b4_cutoff_margin"] == 0.2


def test_mismatched_shapes_are_refused_not_silently_compared():
    with pytest.raises(ValueError, match="different layer counts"):
        compare_head_maps(_fake([[0, 1]], [0.1]),
                          _fake([[0, 1], [0, 1]], [0.1, 0.1]))
    with pytest.raises(ValueError, match="head_scores"):
        compare_head_maps({"gqa_decisions": []}, _fake([[0, 1]], [0.1]))


def test_rank_correlation_is_none_rather_than_a_confident_nan():
    """Two heads, or a constant vector, cannot support a correlation."""
    a = {"head_scores": [[1.0, 2.0]],
         "gqa_decisions": [[{"selected_heads": [0], "cutoff_margin": 0.1}]]}
    b = {"head_scores": [[1.0, 2.0]],
         "gqa_decisions": [[{"selected_heads": [0], "cutoff_margin": 0.1}]]}
    assert compare_head_maps(a, b)["rank_correlation"]["overall"] is None
    flat = {"head_scores": [[1.0, 1.0, 1.0]],
            "gqa_decisions": [[{"selected_heads": [0], "cutoff_margin": 0.0}]]}
    assert compare_head_maps(flat, flat)["rank_correlation"]["overall"] is None


# --- the gate ------------------------------------------------------------

def test_the_speedup_is_measured_not_inferred():
    s = speedup(100.0, 50.0)
    assert s["speedup"] == 2.0
    assert "measured" in s["_basis"]


def test_a_nonpositive_b4_clock_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        speedup(100.0, 0.0)


@pytest.mark.parametrize("sp,identical,expected", [
    (1.00, True, "B4_NOT_WORTH_ADOPTION_PILOT"),
    (1.24, False, "B4_NOT_WORTH_ADOPTION_PILOT"),
    (1.25, True, "B4_STRUCTURALLY_EQUIVALENT_AND_FASTER"),
    (3.90, True, "B4_STRUCTURALLY_EQUIVALENT_AND_FASTER"),
    (1.25, False, "B4_FASTER_AND_STRUCTURALLY_DIFFERENT_RECOVERY_TRIGGERED"),
])
def test_the_three_predeclared_outcomes_and_no_fourth(sp, identical, expected):
    assert verdict(sp, identical, threshold=1.25) == expected


def test_a_slow_arm_never_reaches_recovery_however_different():
    """The speed gate is first: a NOT_WORTH verdict ends the pilot whether or
    not the head maps differ, so recovery cannot be triggered by difference
    alone."""
    for identical in (True, False):
        assert verdict(1.0, identical,
                       threshold=1.25) == "B4_NOT_WORTH_ADOPTION_PILOT"


def test_the_threshold_is_the_callers_not_a_module_constant():
    """It is predeclared in the pilot record; a second copy here could drift."""
    import ast
    import inspect

    from experiments.phase_c3 import compare

    tree = ast.parse(inspect.getsource(compare.verdict))
    nums = {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    assert 1.25 not in nums, "the gate threshold is hardcoded in the comparator"
    assert "threshold" in inspect.signature(compare.verdict).parameters


def test_the_scope_record_predeclares_the_threshold_and_the_arm_order():
    import json

    scope = json.loads((REPO / "logs/stages/stage-1/phase_c3/pilots/"
                        "batching-adoption/v1/scope.json").read_text())
    assert scope["speed_gate"]["threshold"] == 1.25
    assert scope["speed_gate"]["predeclared"] is True
    assert scope["arm_order"] == ["causal-B1", "causal-B4"], (
        "the arm order must be frozen in the record BEFORE either timing is "
        "read, or it can be chosen after seeing which way it fell")
