"""The batch-invariance diagnostic's verdict is derived, so derive it correctly.

`derive_conclusion` is the only place the diagnostic decides what its numbers
mean. Everything else measures. That makes it exactly the kind of function a
previous session got wrong in prose beside correct arithmetic, so it is tabled
here against hand-built stage outputs.

The first rehearsal of the real script produced the case at the bottom of this
file: every measured comparison bit-identical, and a locus of
`attention_backend_kernel`. The predicate said "not every backend diverges",
which is true both when one kernel is exact and the rest are not, and when
nothing diverges at all.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/validation/batch_invariance_diagnostic.py"


def _module():
    """Load the script as a module WITHOUT running it.

    Imported by path rather than re-implemented: a test that reconstructs the
    derivation is a test of its own copy.
    """
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location(
        "batch_invariance_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def derive():
    return _module().derive_conclusion


def _report(**stages) -> dict:
    """A report whose stages are all clean unless a test says otherwise."""
    base = {
        "repeatability": {"both_shapes_internally_deterministic": True},
        "case_matrix": {"comparisons": {
            "A_vs_B": {"bitwise_identical": True, "max_abs": 0.0},
            "C_vs_D": {"bitwise_identical": True, "max_abs": 0.0},
            "A_vs_E": {"bitwise_identical": True, "max_abs": 0.0},
        }},
        "first_divergence": {"first_tap_that_is_not_bit_identical": None},
        "gemm_isolation": {"a_bare_gemm_is_shape_dependent": False},
        "length_sweep": {"padding_alone_is_inert": True,
                         "batch_size_alone_moves_it": False},
        "backend_matrix": {"backends_diverging": [], "backends_exact": ["sdpa"],
                           "every_backend_diverges": False,
                           "no_backend_diverges": True},
        "ffn_selection": {"selection_is_batch_invariant": True,
                          "n_layers_with_moved_selection": 0},
        "causal_kl": {"max_rel_diff": 0.0, "item_ranking_identical": True},
    }
    base.update(stages)
    return {"stages": base, "model": {"attention": {
        "config._attn_implementation": "sdpa"}}}


def test_nothing_diverges_is_not_an_attribution(derive):
    """The rehearsal's case. Silence must read as silence."""
    out = derive(_report())
    assert out["any_divergence_observed"] is False
    assert out["stages_reporting_a_divergence"] == []
    assert out["divergence_locus"] == "no_divergence_observed"


def test_one_kernel_exact_and_another_not_names_the_kernel(derive):
    out = derive(_report(backend_matrix={
        "backends_diverging": ["eager"], "backends_exact": ["sdpa:FLASH"],
        "every_backend_diverges": False, "no_backend_diverges": False}))
    assert out["divergence_locus"] == "attention_backend_kernel"
    assert out["attention_backends_diverging"] == ["eager"]


def test_every_backend_diverging_points_below_attention(derive):
    out = derive(_report(backend_matrix={
        "backends_diverging": ["eager", "sdpa"], "backends_exact": [],
        "every_backend_diverges": True, "no_backend_diverges": False}))
    assert out["divergence_locus"] == "below_attention_above_gemm"


def test_a_shape_dependent_gemm_outranks_the_backend(derive):
    """If the bare GEMM already moves, nothing above it needs attributing."""
    out = derive(_report(
        gemm_isolation={"a_bare_gemm_is_shape_dependent": True},
        backend_matrix={"backends_diverging": ["eager"],
                        "backends_exact": ["sdpa"],
                        "every_backend_diverges": False,
                        "no_backend_diverges": False}))
    assert out["divergence_locus"] == "gemm_level_shape_dependent_accumulation"


def test_nondeterminism_outranks_everything(derive):
    """A shape that will not repeat itself cannot support a cross-shape claim."""
    out = derive(_report(
        repeatability={"both_shapes_internally_deterministic": False},
        gemm_isolation={"a_bare_gemm_is_shape_dependent": True}))
    assert out["divergence_locus"] == "run_to_run_nondeterminism"


def test_batch_size_alone_with_exact_backends_lands_between(derive):
    """Nothing localized it, but the length sweep saw it. Say so, don't guess."""
    out = derive(_report(length_sweep={"padding_alone_is_inert": True,
                                       "batch_size_alone_moves_it": True}))
    assert out["any_divergence_observed"] is True
    assert out["stages_reporting_a_divergence"] == ["length_sweep"]
    assert out["divergence_locus"] == "above_gemm_below_logits"


def test_a_divergence_seen_only_in_the_case_matrix_still_counts(derive):
    out = derive(_report(case_matrix={"comparisons": {
        "A_vs_E": {"bitwise_identical": False, "max_abs": 4.9e-2}}}))
    assert out["any_divergence_observed"] is True
    assert "case_matrix" in out["stages_reporting_a_divergence"]
    assert out["divergence_locus"] != "no_divergence_observed"


def test_a_missing_stage_does_not_manufacture_a_verdict(derive):
    """`--only` runs a subset. An absent stage is unknown, never clean."""
    out = derive({"stages": {}, "model": {}})
    assert out["each_shape_internally_deterministic"] is None
    assert out["divergence_locus"] == "no_divergence_observed"
    assert out["stages_reporting_a_divergence"] == []


def test_padding_alone_is_named_rather_than_left_undetermined(derive):
    """The rehearsal's second case: only the padding sweep saw anything.

    With the frozen mixture, the focal item is the longest in its group, so the
    ragged batch puts no padding on the focal ROW and case E reads exactly zero.
    The only stage that sees the effect is the sweep, and the ladder used to
    fall through to `undetermined` while the sweep had named the knob.
    """
    out = derive(_report(length_sweep={"padding_alone_is_inert": False,
                                       "batch_size_alone_moves_it": False}))
    assert out["divergence_locus"] == "padded_width_dependent"
    assert out["divergence_driver"] == "padded_width"


def test_driver_and_locus_are_independent_axes(derive):
    """A named component must not erase which knob drives it."""
    out = derive(_report(
        length_sweep={"padding_alone_is_inert": False,
                      "batch_size_alone_moves_it": False},
        backend_matrix={"backends_diverging": ["eager"],
                        "backends_exact": ["sdpa:FLASH_ATTENTION"],
                        "every_backend_diverges": False,
                        "no_backend_diverges": False}))
    assert out["divergence_locus"] == "attention_backend_kernel"
    assert out["divergence_driver"] == "padded_width"


@pytest.mark.parametrize(("pad_inert", "batch_moves", "expected"), [
    (True, False, "neither"),
    (False, False, "padded_width"),
    (True, True, "batch_size"),
    (False, True, "padded_width_and_batch_size"),
])
def test_every_driver_combination(derive, pad_inert, batch_moves, expected):
    out = derive(_report(length_sweep={"padding_alone_is_inert": pad_inert,
                                       "batch_size_alone_moves_it": batch_moves}))
    assert out["divergence_driver"] == expected


def test_driver_is_unknown_when_the_sweep_did_not_run(derive):
    assert derive({"stages": {}, "model": {}})["divergence_driver"] is None


def test_the_decision_fields_come_from_their_stages(derive):
    out = derive(_report(
        ffn_selection={"selection_is_batch_invariant": False,
                       "n_layers_with_moved_selection": 26},
        causal_kl={"max_rel_diff": 7.06e-3, "item_ranking_identical": False}))
    assert out["ffn_selection_is_batch_invariant"] is False
    assert out["ffn_layers_with_moved_selection"] == 26
    assert out["causal_kl_max_rel_diff"] == pytest.approx(7.06e-3)
    assert out["causal_kl_item_ranking_identical"] is False
