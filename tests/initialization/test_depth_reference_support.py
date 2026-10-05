"""`depth.causal_kl_greedy_v1` under each distribution support, end to end.

The operator is run for real on a toy model — not a reducer in isolation — because
what has to hold is that the support reaches the scoring loop, forks the reference
state, and leaves the estimand, the per-item weighting and the aggregation alone.

The two facts this pins:

* at ``top_k >= vocab_size`` the Top-K path chooses the SAME layers as the
  full-vocabulary path, because the partition is not coarsened at all;
* the reference state it holds is ``O(T*K)`` rather than ``O(T*V)``, which is the
  reason the protocol was adopted.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.base import OperatorContext  # noqa: E402
from aadistill.initialization.operators.depth import (  # noqa: E402
    causal_kl_greedy,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    FULL_VOCAB_V1, reference_topk_tail,
)
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from support.toy import build_tiny_model  # noqa: E402

#: `tie_word_embeddings` because an untied `lm_head` is a parameter depth surgery
#: does not assign, and the operator refuses rather than shipping random weights.
#: The shared toy geometry ties them for the same reason.
GEOMETRY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=64, tie_word_embeddings=True)


def _items(n=4, vocab=64):
    g = torch.Generator().manual_seed(11)
    out = []
    for i in range(n):
        out.append({
            "item_id": f"item-{i}",
            "input_ids": torch.randint(1, vocab, (1, 9 + i), generator=g),
            "domain": "general" if i % 2 == 0 else "math",
            "subtype": "text" if i % 2 == 0 else "arith",
        })
    return out


def _context(model, items, *, support, batch_size, target_layers):
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter

    register_builtin_adapters()
    parent = ArchSpec.of("qwen3", GEOMETRY)
    target = ArchSpec.of("qwen3", {**GEOMETRY,
                                   "num_hidden_layers": target_layers})
    config = {"n_calibration_items": len(items)}
    #: The config's declaration and the handed object must agree, and under the
    #: full-vocabulary contract the key is ABSENT rather than null.
    if not support.is_full_vocab:
        config["distribution_support"] = support.as_dict()
    return OperatorContext(
        adapter=get_adapter("qwen3"), model=model, parent_spec=parent,
        target_spec=target, profile=None, calibration_items=items, seed=0,
        device="cpu", config=config, distribution_support=support,
        execution=ExecutionConfig(micro_batch_size=batch_size))


def _run(support, batch_size, target_layers=4):
    model = build_tiny_model(GEOMETRY)
    model.config.use_cache = False
    items = _items(vocab=GEOMETRY["vocab_size"])
    op = causal_kl_greedy.DepthCausalKLGreedyV1()
    return op.apply(_context(model, items, support=support,
                             batch_size=batch_size,
                             target_layers=target_layers))


class TestTheSupportReachesTheScoringLoop:

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_at_k_at_least_v_both_supports_choose_the_same_layers(self, batch_size):
        """The coarsening is the identity when the support is the vocabulary."""
        full = _run(FULL_VOCAB_V1, batch_size)
        topk = _run(reference_topk_tail(GEOMETRY["vocab_size"]), batch_size)

        def chosen(outcome):
            rounds = outcome.artifacts["search_rounds"]
            return [r["chosen"] for r in rounds]

        assert chosen(topk) == chosen(full), (
            "a support equal to the vocabulary must not move a decision")

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_the_reference_state_is_o_t_k_not_o_t_v(self, batch_size):
        topk = _run(reference_topk_tail(8), batch_size)
        cache = topk.artifacts["reference_cache"]

        assert cache["distribution_support"] == "reference_topk_tail_v1"
        assert cache["top_k"] == 8
        assert cache["has_tail"] is True
        #: The ratio is the claim: a fraction of what the full-vocabulary cache
        #: would have held. At K=8 of a 64 vocabulary it is not dramatic; the
        #: point is that it is DERIVED and reported rather than asserted.
        assert 0.0 < cache["bytes_ratio_vs_full_vocab"] < 1.0
        assert cache["sketch_bytes"] < cache["full_vocab_bytes_avoided"]
        #: And it fits, so the recompute fallback never arms.
        assert cache["mode"] == "cached"

    def test_a_small_k_may_move_a_decision_and_that_is_recorded(self):
        """A moved decision is EVIDENCE of a different protocol, not a failure.

        Not asserted to move — on a toy geometry it may not — but asserted to be
        legible either way, because the adoption decision rests on being able to
        see it.
        """
        full = _run(FULL_VOCAB_V1, 2)
        coarse = _run(reference_topk_tail(2), 2)
        a = [r["chosen"] for r in full.artifacts["search_rounds"]]
        b = [r["chosen"] for r in coarse.artifacts["search_rounds"]]
        assert len(a) == len(b)
        assert all(isinstance(x, int) for x in b)
        #: Each round records its margin, which is what makes a near-tie visible.
        for r in coarse.artifacts["search_rounds"]:
            assert "chosen" in r and "chosen_score" in r

    def test_the_full_vocab_path_is_untouched(self):
        """Its cache evidence keeps the fields it always had."""
        full = _run(FULL_VOCAB_V1, 2)
        cache = full.artifacts["reference_cache"]
        assert "distribution_support" not in cache
        assert set(cache) >= {"mode", "cached", "items_total", "items_cached"}


class TestTheDeclarationMustMatchWhatRuns:

    def test_a_support_handed_without_being_declared_is_refused(self):
        """The two sides of the contract, checked where declarations meet reality."""
        from aadistill.initialization.operators.base import ContractViolation

        model = build_tiny_model(GEOMETRY)
        model.config.use_cache = False
        items = _items(vocab=GEOMETRY["vocab_size"])
        ctx = _context(model, items, support=reference_topk_tail(8),
                       batch_size=2, target_layers=4)
        #: Strip the declaration, keep the object: the state id would describe a
        #: full-vocabulary KL that was never computed.
        ctx = type(ctx)(**{**{f.name: getattr(ctx, f.name)
                              for f in __import__("dataclasses").fields(ctx)},
                           "config": {"n_calibration_items": len(items)}})
        op = causal_kl_greedy.DepthCausalKLGreedyV1()
        with pytest.raises(ContractViolation, match="distribution support"):
            op.execute(ctx)

    def test_a_declaration_without_the_object_is_refused(self):
        """The other direction: declared Top-K, handed full vocabulary."""
        from aadistill.initialization.operators.base import ContractViolation

        model = build_tiny_model(GEOMETRY)
        model.config.use_cache = False
        items = _items(vocab=GEOMETRY["vocab_size"])
        ctx = _context(model, items, support=FULL_VOCAB_V1, batch_size=2,
                       target_layers=4)
        ctx = type(ctx)(**{**{f.name: getattr(ctx, f.name)
                              for f in __import__("dataclasses").fields(ctx)},
                           "config": {
                               "n_calibration_items": len(items),
                               "distribution_support":
                                   reference_topk_tail(8).as_dict()}})
        op = causal_kl_greedy.DepthCausalKLGreedyV1()
        with pytest.raises(ContractViolation, match="distribution support"):
            op.execute(ctx)


class TestTheScoreObserverIsExecutionOnly:
    """A validation computes a second reduction from the SAME forwards.

    Paying for a duplicate set of 260 model forwards to compare two reducers
    would buy nothing, so the operator lets a caller watch each reduction. What
    must hold is that watching cannot change anything.
    """

    def _with_observer(self, support, seen):
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.specs.arch import get_adapter
        import dataclasses

        register_builtin_adapters()
        model = build_tiny_model(GEOMETRY)
        model.config.use_cache = False
        items = _items(vocab=GEOMETRY["vocab_size"])
        ctx = _context(model, items, support=support, batch_size=2,
                       target_layers=4)
        ctx = dataclasses.replace(ctx, score_observer=lambda **kw: seen.append(kw))
        op = causal_kl_greedy.DepthCausalKLGreedyV1()
        return op.apply(ctx)

    def test_it_sees_the_real_reference_and_candidate_of_every_reduction(self):
        seen = []
        out = self._with_observer(FULL_VOCAB_V1, seen)
        assert seen, "the observer was never called"
        first = seen[0]
        assert set(first) >= {"skip", "group", "indices", "refs", "abls", "mask",
                              "weights", "values", "support"}
        #: Full-vocabulary refs are the `[B, T, V]` logits, which is what lets an
        #: observer build a sketch and reduce a SECOND time without a new forward.
        assert first["refs"].shape == first["abls"].shape
        assert first["refs"].shape[-1] == GEOMETRY["vocab_size"]
        #: One call per (candidate, group), and the values it is handed are the
        #: ones the operator recorded.
        assert len(first["values"]) == len(first["group"].items)

    def test_watching_cannot_change_a_decision(self):
        """The same run with and without an observer chooses the same layers."""
        without = _run(FULL_VOCAB_V1, 2)
        seen = []
        with_obs = self._with_observer(FULL_VOCAB_V1, seen)
        a = [r["chosen"] for r in without.artifacts["search_rounds"]]
        b = [r["chosen"] for r in with_obs.artifacts["search_rounds"]]
        assert a == b
        assert seen

    def test_an_observer_that_returns_something_is_ignored(self):
        """Its return value must not be mistaken for a score."""
        import dataclasses

        from aadistill.initialization.adapters import register_builtin_adapters
        register_builtin_adapters()
        model = build_tiny_model(GEOMETRY)
        model.config.use_cache = False
        items = _items(vocab=GEOMETRY["vocab_size"])
        ctx = dataclasses.replace(
            _context(model, items, support=FULL_VOCAB_V1, batch_size=2,
                     target_layers=4),
            score_observer=lambda **kw: 1e9)
        out = causal_kl_greedy.DepthCausalKLGreedyV1().apply(ctx)
        baseline = _run(FULL_VOCAB_V1, 2)
        assert [r["chosen"] for r in out.artifacts["search_rounds"]] == \
            [r["chosen"] for r in baseline.artifacts["search_rounds"]]

    def test_a_second_reduction_from_the_observed_forwards_agrees_with_core(self):
        """The thing the hook exists for, end to end on a real operator run."""
        from aadistill.initialization.scoring.support import (
            sketch_forward_kl_mean_batch, sketch_reference,
        )

        seen = []
        self._with_observer(FULL_VOCAB_V1, seen)
        call = seen[0]
        refs, abls = call["refs"], call["abls"]
        B, T, V = refs.shape
        flat = refs.reshape(-1, V)
        sk = sketch_reference(flat, torch.zeros(B * T, dtype=torch.long),
                              top_k=V, chunk=256)
        second = sketch_forward_kl_mean_batch(
            sk.support_indices.reshape(B, T, -1),
            sk.support_log_probs.reshape(B, T, -1),
            sk.tail_log_prob.reshape(B, T), abls, call["mask"],
            has_tail=False, weights=call["weights"])
        #: At K >= V the second reduction must reproduce the operator's own.
        assert torch.allclose(second, call["values"], rtol=1e-4), (
            "a second reduction from the same forwards disagrees with the first")
