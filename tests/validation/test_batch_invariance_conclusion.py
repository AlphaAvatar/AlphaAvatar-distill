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


#: The FULL field set the conclusion reads from each stage.
#:
#: Not a convenient subset. `derive_conclusion` now RAISES when a stage that ran
#: is missing a field it asks for, because it silently returned `None` for a
#: field whose name had drifted from the stage that emits it. A fixture holding
#: fewer fields than the real stages would reinstate exactly that blind spot, so
#: `test_the_fixture_carries_every_field_the_conclusion_reads` parses the
#: production source and fails when this drifts.
CLEAN_STAGES: dict = {
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
    "reduction_control": {"applicable": True,
                          "divergence_removed_by_disabling": False},
    "statistics_decomposition": {"every_batch_size_bit_identical": True},
    "ffn_selection": {"selection_is_batch_invariant": True,
                      "n_layers_with_moved_selection": 0,
                      "selection_is_batch_invariant_at_every_ratio_and_size": True,
                      "headline_micro_batch_size": 4,
                      "headline_keep_ratio": 0.5},
    "causal_kl": {"max_rel_diff": 0.0, "item_ranking_identical": True},
}


def _report(**stages) -> dict:
    """A report whose stages are all clean unless a test says otherwise."""
    base = {name: dict(fields) for name, fields in CLEAN_STAGES.items()}
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
        ffn_selection={**CLEAN_STAGES["ffn_selection"],
                       "selection_is_batch_invariant": False,
                       "n_layers_with_moved_selection": 26},
        causal_kl={"max_rel_diff": 7.06e-3, "item_ranking_identical": False}))
    assert out["ffn_selection_is_batch_invariant"] is False
    assert out["ffn_layers_with_moved_selection"] == 26
    assert out["causal_kl_max_rel_diff"] == pytest.approx(7.06e-3)
    assert out["causal_kl_item_ranking_identical"] is False


def test_the_fixture_carries_every_field_the_conclusion_reads():
    """The fixture must not be a convenient subset of the real stage outputs.

    `derive_conclusion` raises on a missing field, which only helps if the
    fixture is as complete as the stages are. This reads the `got("stage",
    "field")` calls straight out of the production source, so adding a field
    there and forgetting it here fails HERE rather than on a paid pod.
    """
    import ast
    tree = ast.parse(SCRIPT.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "derive_conclusion")
    required: dict[str, set] = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "got" and len(node.args) >= 2
                and all(isinstance(a, ast.Constant) for a in node.args[:2])):
            required.setdefault(node.args[0].value, set()).add(node.args[1].value)
    assert required, "no got() calls were parsed; the shape of this test is wrong"
    missing = {stage: sorted(fields - set(CLEAN_STAGES.get(stage, {})))
               for stage, fields in required.items()
               if fields - set(CLEAN_STAGES.get(stage, {}))}
    assert not missing, (
        f"CLEAN_STAGES is missing fields derive_conclusion reads: {missing}")


def _conclusion_reads() -> dict[str, set]:
    """The (stage, field) pairs `derive_conclusion` asks for, from the source."""
    import ast
    fn = next(n for n in ast.walk(ast.parse(SCRIPT.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "derive_conclusion")
    reads: dict[str, set] = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "got" and len(node.args) >= 2
                and all(isinstance(a, ast.Constant) for a in node.args[:2])):
            reads.setdefault(node.args[0].value, set()).add(node.args[1].value)
    return reads


def test_every_field_the_conclusion_reads_is_a_key_some_stage_writes():
    """Close the loop on the OTHER side of the rename.

    The fixture test proves the fixture is complete; it cannot notice a stage
    renaming a field it emits, because the tests read the fixture. Renaming
    `selection_is_batch_invariant_at_every_ratio_and_size` in the STAGE passed
    every test while the conclusion went on reading the old name — which on a
    real report is the silent `None` this whole strictness exists to prevent.

    So: every field the conclusion reads must appear somewhere in the module as
    a dict key written OUTSIDE `derive_conclusion`. Coarse on purpose — it
    cannot tell which stage writes it — but a rename breaks it, which is the
    failure that matters.
    """
    import ast
    tree = ast.parse(SCRIPT.read_text())
    conclusion = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef)
                      and n.name == "derive_conclusion")
    conclusion_nodes = set(map(id, ast.walk(conclusion)))
    written: set = set()
    for node in ast.walk(tree):
        if id(node) in conclusion_nodes:
            continue
        if isinstance(node, ast.Dict):
            written |= {k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        elif (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
              and isinstance(node.slice.value, str)):
            written.add(node.slice.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in {"update", "setdefault"}):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    written.add(arg.value)

    orphans = sorted({f for fields in _conclusion_reads().values() for f in fields}
                     - written)
    assert not orphans, (
        "derive_conclusion reads fields no stage writes anywhere in the module; "
        f"a rename on the producing side would go silent: {orphans}")


def test_a_stage_that_ran_but_lacks_a_field_raises(derive):
    """The strictness itself, exercised. A complete fixture never triggers it."""
    report = _report()
    del report["stages"]["ffn_selection"]["headline_keep_ratio"]
    with pytest.raises(KeyError, match="headline_keep_ratio"):
        derive(report)


@pytest.mark.parametrize("marker", [
    {"ran": False, "reason": "statistics decomposition did not run"},
    {"error": "RuntimeError: CUDA out of memory"},
])
def test_a_stage_that_declared_itself_absent_does_not_raise(derive, marker):
    """`ran: False` and an error are legitimate silence, not typos."""
    out = derive(_report(ffn_selection=marker))
    assert out["ffn_selection_is_batch_invariant"] is None
    assert out["divergence_locus"] == "no_divergence_observed"


def test_a_statistics_only_run_does_not_report_no_divergence(derive):
    """The rehearsal's third case, and the worst kind: self-contradiction.

    `--only statistics_decomposition,ffn_selection` is a legitimate mode -- the
    full-mixture variant uses it, because batching 67 items into one logit
    block is 41 GiB. Such a run produced `divergence_locus =
    no_divergence_observed` while its own `every_batch_size_bit_identical` was
    `False`. The report disagreed with itself, in the field a reader quotes.
    """
    stages = {k: v for k, v in CLEAN_STAGES.items()
              if k in ("statistics_decomposition", "ffn_selection")}
    stages["statistics_decomposition"] = {"every_batch_size_bit_identical": False}
    out = derive({"stages": stages, "model": {}})
    assert out["any_divergence_observed"] is True
    assert "statistics_decomposition" in out["stages_reporting_a_divergence"]
    assert out["divergence_locus"] == "observed_but_not_localized"


def test_a_moved_selection_alone_counts_as_observed(derive):
    stages = {"ffn_selection": {**CLEAN_STAGES["ffn_selection"],
                                "selection_is_batch_invariant": False,
                                "n_layers_with_moved_selection": 26}}
    out = derive({"stages": stages, "model": {}})
    assert out["stages_reporting_a_divergence"] == ["ffn_selection"]
    assert out["divergence_locus"] == "observed_but_not_localized"


def test_a_localized_run_still_beats_observed_but_not_localized(derive):
    """The weak branch must stay last."""
    out = derive(_report(
        statistics_decomposition={"every_batch_size_bit_identical": False},
        gemm_isolation={"a_bare_gemm_is_shape_dependent": True}))
    assert out["divergence_locus"] == "gemm_level_shape_dependent_accumulation"


def test_clean_statistics_do_not_manufacture_a_divergence(derive):
    out = derive(_report())
    assert out["stages_reporting_a_divergence"] == []
    assert out["divergence_locus"] == "no_divergence_observed"


def test_every_report_names_the_executable_that_produced_it():
    """P4's evidence-code version, which the reports did not carry.

    It became load-bearing when one investigation collected reports from two
    commits: without it a reader compares numbers from different code with no
    way to notice.
    """
    mod = _module()
    ident = mod.executable_identity()
    assert ident["file"] == "batch_invariance_diagnostic.py"
    assert len(ident["sha256"]) == 64
    import hashlib
    assert ident["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert "git_head" in ident and "this_file_is_clean_at_git_head" in ident
