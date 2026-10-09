"""The frozen path's own half of the root-bounds-deeper premise.

D1's DEPTH cost cell is measured at the ROOT parent, and the design claims that
bounds the same operator at any deeper parent. The generic half of that claim — no
registered operator increases a structural field — is a property of the
operator/adapter contract and lives in the core suite
(`tests/initialization/test_operators_only_shrink.py`).

What lives HERE is the part that is about this campaign: which four
implementations the A3-frozen path applies, and what a committed DEPTH search
record's rounds actually look like. Those are experiment state, and asserting them
from the core suite is what AGENTS.md 2.8a forbids — the split was made after the
core suite caught exactly that import.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "tests"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.base import OperatorContext  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from support.toy import build_tiny_model  # noqa: E402

PARENT = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=6,
              num_attention_heads=4, num_key_value_heads=2, head_dim=8,
              vocab_size=64, tie_word_embeddings=True)
TARGET = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
              num_attention_heads=2, num_key_value_heads=1, head_dim=8,
              vocab_size=64, tie_word_embeddings=True)


def _frozen_operators():
    """The operators D1's frozen path may apply, read from the path spec itself."""
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from aadistill.initialization.operators.base import get_implementation
    from shared.calibration import register_builtin_profiles
    from stages.phase_a3 import a3_session as a3
    from stages.phase_c2.search_space import register_c2_operators

    register_builtin_adapters()
    register_builtin_profiles()
    register_c2_operators()
    activation_importance.register()
    spec = a3.path_spec(workdir_device="cpu")
    return [get_implementation(step.impl_id) for step in spec.steps]


def _items(n=4, vocab=64):
    g = torch.Generator().manual_seed(21)
    return [{"item_id": f"i{k}",
             "input_ids": torch.randint(1, vocab, (1, 9 + k), generator=g),
             "domain": "general" if k % 2 else "math",
             "subtype": "general" if k % 2 else "openmath"}
            for k in range(n)]


class TestTheFrozenSetIsTheOneThePathDeclares:

    def test_it_is_those_four_implementations(self):
        impls = _frozen_operators()
        assert len(impls) == 4, [i.impl_id for i in impls]
        assert {i.impl_id for i in impls} == {
            "depth.causal_kl_greedy_v1", "ffn.activation_importance_v0",
            "width.global_pca_v0", "attention.activation_importance_v1"}

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_each_only_shrinks(self, index, tmp_path):
        """The frozen set specifically, on a real toy parent.

        The core suite sweeps the whole registry; this asserts the four that D1's
        pricing premise actually rests on, so a path change that swapped one in
        fails here rather than being covered by a generic sweep.
        """
        from aadistill.initialization.specs.arch import get_adapter

        impl = _frozen_operators()[index]
        adapter = get_adapter("qwen3")
        model = build_tiny_model(PARENT)
        model.config.use_cache = False
        parent = ArchSpec.of("qwen3", PARENT)
        ctx = OperatorContext(
            adapter=adapter, model=model, parent_spec=parent,
            target_spec=ArchSpec.of("qwen3", TARGET), profile=None,
            calibration_items=_items(), seed=0, device="cpu", workdir=tmp_path,
            config={"n_calibration_items": 4}, execution=ExecutionConfig())
        child = adapter.spec_of(impl.apply(ctx).model)
        grew = [f for f in adapter.structural_fields
                if int(child[f]) > int(parent[f])]
        assert not grew, (
            f"{impl.impl_id} INCREASED {grew}, so a deeper parent is not a "
            "smaller model and the root measurement does not bound the deeper "
            "cell")
        assert any(int(child[f]) < int(parent[f])
                   for f in adapter.structural_fields)


class TestTheCommittedDepthRecordsRoundsGrowBySkipSet:
    """Read off a real greedy run's own round records.

    `removed_before` is the skip set the round starts from and `chosen` is the one
    layer it adds, so the sizes must increase by exactly one per round. This is a
    committed record of THIS campaign, which is why it is here.
    """

    RECORD = (REPO / "logs/stages/stage-1/phase_d1/validations/topk-adoption/v1"
                     "/runs/a4/adoption.slim.json")

    def test_each_round_bypasses_one_more_block_than_the_last(self):
        if not self.RECORD.is_file():
            pytest.skip("no committed DEPTH round record to read")
        rounds = json.loads(self.RECORD.read_text())["C_depth"]["rounds"]
        assert rounds, "no rounds recorded"
        sizes = [len(r.get("removed_before") or []) for r in rounds]
        assert sizes == list(range(len(rounds))), sizes

    def test_the_candidate_count_falls_as_the_skip_set_grows(self):
        if not self.RECORD.is_file():
            pytest.skip("no committed DEPTH round record to read")
        rounds = json.loads(self.RECORD.read_text())["C_depth"]["rounds"]
        counts = [int(r["n_candidates"]) for r in rounds]
        assert counts == sorted(counts, reverse=True), counts
        assert counts[0] == max(counts)
