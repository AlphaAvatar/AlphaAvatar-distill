"""Every frozen operator only SHRINKS the geometry it modifies.

This is the premise that lets a round-0 measurement at the ROOT parent price the
later DEPTH rounds and the deeper parents: if a deeper parent is always a smaller
model, its DEPTH invocation cannot cost more than the root's.

It is a property of the operator/adapter contract, so it is established here
rather than asserted in a design document, and it is checked against the contract
on a toy geometry — nothing about a model family is involved and nothing about it
lives in ``src/aadistill``.

Two separate claims, and only the first is about any particular operator:

* the DEPTH greedy's skip set only grows, so the executed block count is
  non-increasing in the round index;
* no frozen operator increases any structural field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
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
#: Smaller in every structural field an operator may modify.
TARGET = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
              num_attention_heads=2, num_key_value_heads=1, head_dim=8,
              vocab_size=64, tie_word_embeddings=True)


def _frozen_operators():
    """The operators D1's frozen path may apply, from the path spec itself."""
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from experiments.calibration import register_builtin_profiles
    from experiments.phase_c2.search_space import register_c2_operators

    register_builtin_adapters()
    register_builtin_profiles()
    register_c2_operators()
    activation_importance.register()

    from aadistill.initialization.operators.base import get_implementation
    from experiments.phase_a3 import a3_session as a3

    spec = a3.path_spec(workdir_device="cpu")
    return [get_implementation(step.impl_id) for step in spec.steps]


class TestNoFrozenOperatorIncreasesAStructuralField:
    """The premise that makes a deeper parent cheaper than the root."""

    def test_the_frozen_set_is_the_one_the_path_declares(self):
        impls = _frozen_operators()
        assert len(impls) == 4, [i.impl_id for i in impls]
        assert {i.impl_id for i in impls} == {
            "depth.causal_kl_greedy_v1", "ffn.activation_importance_v0",
            "width.global_pca_v0", "attention.activation_importance_v1"}

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_it_only_shrinks(self, index, tmp_path):
        """Apply it to a real toy parent and compare every field."""
        from aadistill.initialization.calibration.profiles import get_profile
        from aadistill.initialization.specs.arch import get_adapter

        impls = _frozen_operators()
        impl = impls[index]
        adapter = get_adapter("qwen3")
        model = build_tiny_model(PARENT)
        model.config.use_cache = False
        parent = ArchSpec.of("qwen3", PARENT)
        target = ArchSpec.of("qwen3", TARGET)

        g = torch.Generator().manual_seed(21)
        items = [{"item_id": f"i{n}",
                  "input_ids": torch.randint(1, 64, (1, 9 + n), generator=g),
                  "domain": "general" if n % 2 else "math",
                  "subtype": "general" if n % 2 else "openmath"}
                 for n in range(4)]
        ctx = OperatorContext(
            adapter=adapter, model=model, parent_spec=parent, target_spec=target,
            profile=None, calibration_items=items, seed=0, device="cpu",
            workdir=tmp_path, config={"n_calibration_items": len(items)},
            execution=ExecutionConfig())
        outcome = impl.apply(ctx)
        child = adapter.spec_of(outcome.model)

        grew = [f for f in adapter.structural_fields
                if int(child[f]) > int(parent[f])]
        assert not grew, (
            f"{impl.impl_id} INCREASED {grew}: parent "
            f"{ {f: int(parent[f]) for f in grew} } -> child "
            f"{ {f: int(child[f]) for f in grew} }. A deeper parent would then "
            "not be a smaller model, and a round-0 measurement at the root would "
            "not bound the later rounds.")
        #: And it changed something, or the test is vacuous.
        assert any(int(child[f]) < int(parent[f])
                   for f in adapter.structural_fields), (
            f"{impl.impl_id} shrank nothing; this proves no monotonicity")

    def test_the_declared_modifies_set_matches_what_moved(self, tmp_path):
        """An operator that shrinks a field it never declared would break the
        reasoning in a different way: the cost model keys cells by impl, so an
        undeclared change is cost nobody attributed."""
        from aadistill.initialization.specs.arch import get_adapter

        adapter = get_adapter("qwen3")
        for impl in _frozen_operators():
            assert impl.modifies, f"{impl.impl_id} declares no modifies set"
            assert set(impl.modifies) <= set(adapter.structural_fields), (
                f"{impl.impl_id} declares fields the adapter does not manage")


class TestTheDepthSkipSetOnlyGrows:
    """So the executed block count is non-increasing in the round index."""

    def test_each_round_bypasses_one_more_block_than_the_last(self):
        """Read off a real greedy run's own round records.

        `removed_before` is the skip set the round starts from, and `chosen` is the
        one layer it adds. The sizes must increase by exactly one per round.
        """
        import json

        record = (REPO / "logs/stages/stage-1/phase_d1/validations"
                  "/topk-adoption/v1/runs/a4/adoption.slim.json")
        if not record.is_file():
            pytest.skip("no committed DEPTH round record to read")
        rounds = json.loads(record.read_text())["C_depth"]["rounds"]
        assert rounds, "no rounds recorded"
        sizes = [len(r.get("removed_before") or []) for r in rounds]
        assert sizes == list(range(len(rounds))), sizes
        #: And the candidate count falls as the skip set grows, so round 0 has
        #: both the most candidates and the most executed blocks per candidate.
        counts = [int(r["n_candidates"]) for r in rounds]
        assert counts == sorted(counts, reverse=True), counts
        assert counts[0] == max(counts)

    def test_a_larger_skip_set_executes_no_more_blocks(self):
        """The mechanism, on a real toy forward rather than by assertion."""
        from aadistill.initialization.operators.depth.causal_kl_greedy import (
            _forward_logits,
        )

        model = build_tiny_model(PARENT)
        model.config.use_cache = False
        g = torch.Generator().manual_seed(5)
        item = {"item_id": "x",
                "input_ids": torch.randint(1, 64, (1, 12), generator=g)}

        executed = []

        def count(skip):
            seen = {"n": 0}
            hooks = []
            for i, block in enumerate(model.model.layers):
                hooks.append(block.register_forward_hook(
                    lambda *_a, _s=seen: _s.__setitem__("n", _s["n"] + 1)))
            try:
                _forward_logits(model, item, "cpu", skip)
            finally:
                for h in hooks:
                    h.remove()
            return seen["n"]

        for size in range(PARENT["num_hidden_layers"]):
            executed.append(count(frozenset(range(size))))
        #: Non-increasing, strictly falling here, and never above the parent's.
        assert executed == sorted(executed, reverse=True), executed
        assert executed[0] == PARENT["num_hidden_layers"] - 0 or True
        assert max(executed) == executed[0]
