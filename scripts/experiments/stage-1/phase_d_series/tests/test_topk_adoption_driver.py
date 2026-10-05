"""The Top-K adoption driver's contracts, at $0.

One of these is the reason the file exists. `stage_D_state_eval` was defined and
never called, so the driver would have produced items A, B, C and E and silently
omitted D — a run that cost real money and answered four fifths of what it was
authorized for, with nothing in the record saying so.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod", "scripts/autoinit"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

SOURCE = (REPO / "scripts/pod/topk_adoption_driver.py").read_text()
AUTH = REPO / ("logs/stages/stage-1/phase_d1/validations/topk-adoption/v1/"
               "authorization.json")


class TestEveryStageIsActuallyCalled:

    def test_no_stage_is_defined_and_never_called(self):
        """THE REGRESSION. `stage_D_state_eval` was orphaned."""
        defined = set(re.findall(r"^def (stage_[A-Za-z_]+)", SOURCE, re.M))
        assert defined, "no stages found; this test is not reading the driver"
        orphaned = []
        for name in sorted(defined):
            #: Calls only: the definition line itself is excluded by requiring a
            #: `(` that is not preceded by `def `.
            calls = [m for m in re.finditer(rf"(?<!def ){name}\(", SOURCE)]
            if not calls:
                orphaned.append(name)
        assert not orphaned, (
            f"defined and never called: {orphaned}. A paid run would execute "
            "every other stage and silently omit this one's evidence.")

    def test_each_authorized_item_is_named_by_a_stage_or_an_analysis(self):
        """The authorization's `covers` list and the record's keys must line up."""
        covers = json.loads(AUTH.read_text())["covers"]
        assert len(covers) == 5, (
            f"the authorization covers {len(covers)} items; this test expects "
            "the five of item 10 (A-E) and must be updated deliberately")
        #: Item letters the driver's record actually carries somewhere.
        for marker in ("A_reference_top_k_mass", "B_full_vs_k_plus_1",
                       "C_discrete_decisions", "D_state_eval", "C_depth"):
            assert marker in SOURCE, f"{marker} appears nowhere in the driver"


class TestTheProtocolPolicyIsTheOneOwner:

    def test_the_driver_takes_k_from_the_policy_module(self):
        """Not a literal, so K cannot drift between the policy and the run."""
        assert "D_SERIES_SUPPORT" in SOURCE
        assert "from experiments.phase_d_series.scoring_protocol import" in SOURCE
        #: And no bare 200 in the driver's CODE. Line-prefix filtering is not
        #: enough -- a docstring's body lines start with ordinary words -- so this
        #: walks the tokens and skips comments and docstrings properly.
        import ast
        import io
        import tokenize

        tree = ast.parse(SOURCE)
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                text = ast.get_docstring(node, clean=False)
                if text:
                    docs.add(text)
        hits = []
        for tok in tokenize.generate_tokens(io.StringIO(SOURCE).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING:
                try:
                    value = ast.literal_eval(tok.string)
                except Exception:
                    value = None
                if isinstance(value, str) and value in docs:
                    continue
            #: The exact TOKEN, not a substring: `200_000` is a sample cap and
            #: `0.2001` would be a tolerance. What must not appear is K itself.
            if tok.type == tokenize.NUMBER and tok.string.replace("_", "") == "200":
                hits.append((tok.start[0], tok.string))
        assert not hits, f"the driver carries K as a literal: {hits}"

    def test_the_policy_names_k_once(self):
        from experiments.phase_d_series.scoring_protocol import (
            D_SERIES_SUPPORT, D_SERIES_TOP_K, describe,
        )

        assert D_SERIES_TOP_K == 200
        assert D_SERIES_SUPPORT.top_k == D_SERIES_TOP_K
        assert describe()["top_k"] == D_SERIES_TOP_K
        d = describe()
        assert "no sweep" in d["k_is_not_outcome_selected"].lower()
        #: The reference for DEPTH is the PARENT, not the teacher. Getting this
        #: wrong would score a different question entirely.
        assert "intact current parent" in d["reference_for_depth"]
        assert "NOT" in d["reference_for_depth"]
        assert d["reference_for_state_evaluation"] == "the original teacher"

    def test_the_batteries_are_declared_untouched(self):
        from experiments.phase_d_series.scoring_protocol import describe

        note = describe()["_does_not_move_the_batteries"]
        assert "1e3445f1b6769169287f6d091e50086e3cf9b663" in note


class TestTheAnalysisChecksItself:

    def test_it_compares_its_reconstruction_against_the_operator(self):
        """An analysis that cannot be wrong about its own inputs is not evidence."""
        assert "reconstruction_matches_the_operator" in SOURCE
        assert "operator_chose" in SOURCE

    def test_it_flags_a_k_plus_1_kl_that_exceeds_the_full_one(self):
        """Impossible for a coarsening, so it is a defect rather than a finding."""
        assert "n_positive_signed" in SOURCE
        from topk_adoption_driver import compare_scalars

        bad = compare_scalars([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        assert bad["n_positive_signed"] == 3

    def test_it_states_where_the_comparison_stops_being_exact(self):
        from topk_adoption_driver import compare_scalars  # noqa: F401

        assert "comparison_is_exact_through_round" in SOURCE
        assert "counterfactual" in SOURCE

    def test_the_quantiles_are_the_seven_item_10A_asks_for(self):
        from topk_adoption_driver import quantiles

        got = quantiles([float(i) for i in range(100)])
        for key in ("mean", "min", "p1", "p5", "p50", "p95", "p99", "max"):
            assert key in got, key


class TestStagePsInnerLoopRunsAtAToyGeometry:
    """Every shape stage P reads, exercised on CPU.

    Stage P died 5.84 s into a paid subrun on `spec.fields["num_hidden_layers"]`:
    `fields` is a TUPLE of (name, value) pairs. My $0 probe had checked that every
    module and symbol stage P imports EXISTS, which it did -- and said nothing
    about the SHAPE of what one returns. This runs the loop.
    """

    GEOMETRY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    vocab_size=64, tie_word_embeddings=True)

    @pytest.fixture
    def pieces(self):
        torch = pytest.importorskip("torch")
        sys.path.insert(0, str(REPO / "tests"))
        from support.toy import build_tiny_model

        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.specs.arch import get_adapter

        register_builtin_adapters()
        model = build_tiny_model(self.GEOMETRY)
        model.config.use_cache = False
        items = [{"item_id": f"i{n}",
                  "input_ids": torch.randint(1, 64, (1, 9 + n)),
                  "domain": "general", "subtype": "general"} for n in range(3)]
        return torch, model, items, get_adapter("qwen3")

    def test_the_depth_comes_from_a_subscript_not_a_field_lookup(self, pieces):
        """THE REGRESSION."""
        _, model, _, adapter = pieces
        spec = adapter.spec_of(model)
        assert isinstance(spec.fields, tuple), (
            "if `fields` became a mapping this test is obsolete, not passing")
        with pytest.raises(TypeError):
            spec.fields["num_hidden_layers"]          # what stage P used to do
        assert int(spec["num_hidden_layers"]) == self.GEOMETRY["num_hidden_layers"]

    def test_the_whole_inner_loop_runs(self, pieces):
        """Sketch cache -> packed batch -> ablated forward -> Top-K reduction."""
        torch, model, items, adapter = pieces
        from aadistill.initialization.calibration.packing import packed_batches
        from aadistill.initialization.operators._common import resolve_pad_id
        from aadistill.initialization.operators.depth.causal_kl_greedy import (
            _ReferenceSketches, _forward_logit_block,
        )
        from aadistill.initialization.scoring.support import (
            sketch_forward_kl_mean_batch,
        )

        sketches = _ReferenceSketches(model, items, "cpu", top_k=8)
        groups = [(p.batch, p.original_indices) for p in packed_batches(
            items, 3, packing="length_sorted_v1",
            pad_id=resolve_pad_id(model), device="cpu")]
        assert groups
        for group, _ in groups:
            idx, lp, tail = sketches.sketch_block(group)
            mask = group.prediction_mask()
            assert idx.shape[:2] == lp.shape[:2] == tail.shape
            assert tail.shape == mask.shape
            abls = _forward_logit_block(model, group, "cpu", frozenset({0}))
            out = sketch_forward_kl_mean_batch(
                idx, lp, tail, abls, mask, has_tail=sketches.has_tail)
            assert out.shape == (len(group.items),)
            assert out.dtype == torch.float64
            assert torch.isfinite(out).all()

    def test_the_decision_record_carries_what_stage_P_reports(self, pieces):
        _, model, items, _ = pieces
        from aadistill.initialization.operators.depth.causal_kl_greedy import (
            _ReferenceSketches,
        )

        d = _ReferenceSketches(model, items, "cpu", top_k=8).decision()
        for key in ("sketch_bytes", "full_vocab_bytes_avoided", "mode",
                    "top_k", "has_tail"):
            assert key in d, key
        assert d["sketch_bytes"] < d["full_vocab_bytes_avoided"]
