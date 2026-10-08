"""Every registered structural operator only SHRINKS the geometry it modifies.

This is the generic half of a premise a cost model leans on: if a deeper parent is
always a smaller model, an operator invocation measured at the ROOT bounds the same
operator at any deeper parent. It is a property of the operator/adapter contract,
so it belongs here, and it is checked against the contract on a toy geometry —
nothing about a model family is involved.

**The experiment-specific half lives with its experiment** (AGENTS.md 2.8a). Which
four implementations a frozen path applies, and what a committed search record's
rounds look like, are facts about that campaign:
`scripts/stages/stage-1/phase_d1/tests/test_frozen_path_monotonicity.py`.
This file imports no experiment and would hold if every experiment were deleted.

Two separate claims:

* no registered structural operator increases any structural field;
* the DEPTH greedy's skip set only grows, so the executed block count is
  non-increasing in the round index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "tests"):
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
              num_attention_heads=2, num_key_value_heads=2, head_dim=8,
              vocab_size=64, tie_word_embeddings=True)


def _registered_structural_operators():
    """Every BUILTIN implementation, from the registry. No experiment involved."""
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.base import get_implementation
    from aadistill.initialization.operators.register import (
        BUILTIN_OPERATORS, register_builtin_operators,
    )

    register_builtin_adapters()
    register_builtin_operators()
    #: `BUILTIN_OPERATORS` holds the implementation OBJECTS, which are not
    #: orderable; sorting by id keeps the parametrize ids stable for a reader.
    return [get_implementation(o.impl_id)
            for o in sorted(BUILTIN_OPERATORS, key=lambda o: o.impl_id)]



def _items(n=4, vocab=64):
    g = torch.Generator().manual_seed(21)
    return [{"item_id": f"i{k}",
             "input_ids": torch.randint(1, vocab, (1, 9 + k), generator=g),
             "domain": "general" if k % 2 else "math",
             "subtype": "general" if k % 2 else "openmath"}
            for k in range(n)]


class TestNoRegisteredOperatorIncreasesAStructuralField:
    """The premise that makes a deeper parent cheaper than the root."""

    def test_the_registry_is_not_empty(self):
        """Otherwise the sweep below is vacuous and would pass on nothing."""
        impls = _registered_structural_operators()
        assert len(impls) >= 4, [i.impl_id for i in impls]
        #: Every structural kind is represented, so no kind can be silently
        #: unexercised by this sweep.
        #: The adapter's kind names, read off the registry rather than guessed:
        #: they are `DEPTH`, `FFN`, `RESIDUAL_WIDTH`, `ATTENTION`,
        #: `COMPOSITE_STAGE1`, and a lower-case guess passed vacuously.
        kinds = {str(i.kind) for i in impls}
        assert kinds >= {"DEPTH", "FFN", "RESIDUAL_WIDTH", "ATTENTION"}, kinds

    @pytest.mark.parametrize(
        "impl_id", sorted(i.impl_id for i in _registered_structural_operators()))
    def test_it_only_shrinks(self, impl_id, tmp_path):
        """Apply it to a real toy parent and compare every field."""
        from aadistill.initialization.operators.base import get_implementation
        from aadistill.initialization.specs.arch import get_adapter

        _registered_structural_operators()
        impl = get_implementation(impl_id)
        adapter = get_adapter("qwen3")
        model = build_tiny_model(PARENT)
        model.config.use_cache = False
        parent = ArchSpec.of("qwen3", PARENT)
        target = ArchSpec.of("qwen3", TARGET)

        ctx = OperatorContext(
            adapter=adapter, model=model, parent_spec=parent, target_spec=target,
            profile=None, calibration_items=_items(), seed=0, device="cpu",
            workdir=tmp_path, config={"n_calibration_items": 4},
            execution=ExecutionConfig())
        outcome = impl.apply(ctx)
        child = adapter.spec_of(outcome.model)

        grew = [f for f in adapter.structural_fields
                if int(child[f]) > int(parent[f])]
        assert not grew, (
            f"{impl.impl_id} INCREASED {grew}: parent "
            f"{ {f: int(parent[f]) for f in grew} } -> child "
            f"{ {f: int(child[f]) for f in grew} }. A deeper parent would then "
            "not be a smaller model, and a root measurement would not bound the "
            "deeper cell.")
        #: And it changed something, or the test is vacuous.
        assert any(int(child[f]) < int(parent[f])
                   for f in adapter.structural_fields), (
            f"{impl.impl_id} shrank nothing; this proves no monotonicity")

    def test_every_declared_modifies_field_is_one_the_adapter_manages(self):
        """The cost model keys cells by impl, so an undeclared change is cost
        nobody attributed."""
        from aadistill.initialization.specs.arch import get_adapter

        adapter = get_adapter("qwen3")
        for impl in _registered_structural_operators():
            assert impl.modifies, f"{impl.impl_id} declares no modifies set"
            assert set(impl.modifies) <= set(adapter.structural_fields), (
                f"{impl.impl_id} declares fields the adapter does not manage")


class TestABiggerSkipSetExecutesNoMoreBlocks:
    """The mechanism, on a real toy forward rather than by assertion."""

    def test_the_executed_block_count_is_non_increasing_in_the_skip_size(self):
        from aadistill.initialization.operators.depth.causal_kl_greedy import (
            _forward_logits,
        )

        model = build_tiny_model(PARENT)
        model.config.use_cache = False
        g = torch.Generator().manual_seed(5)
        item = {"item_id": "x",
                "input_ids": torch.randint(1, 64, (1, 12), generator=g)}

        def count(skip):
            seen = {"n": 0}
            hooks = [block.register_forward_hook(
                        lambda *_a, _s=seen: _s.__setitem__("n", _s["n"] + 1))
                     for block in model.model.layers]
            try:
                _forward_logits(model, item, "cpu", skip)
            finally:
                for h in hooks:
                    h.remove()
            return seen["n"]

        executed = [count(frozenset(range(size)))
                    for size in range(PARENT["num_hidden_layers"])]
        assert executed == sorted(executed, reverse=True), executed
        #: An empty skip set executes every block, which is the top of the range
        #: and the reason the sequence is non-increasing at all.
        assert executed[0] == max(executed) == PARENT["num_hidden_layers"]
        #: Strictly falling here, which is stronger than the claim and is what
        #: makes the claim non-vacuous on this geometry.
        assert len(set(executed)) == len(executed), executed
