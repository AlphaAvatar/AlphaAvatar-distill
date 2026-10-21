"""DEPTH prefix reuse is an execution path: the same scores, bit for bit.

`ExecutionConfig.depth_prefix_reuse` lets a round's candidates share the
current model's prefix: a candidate bypassing block ``c`` starts from the input
block ``c`` received in one capture forward, instead of recomputing every block
before it. The claim is EXACT equality, so these tests compare floats with
``==``, never approximately, on a tiny genuine Qwen3 with right-padded batches
of unequal length:

1. every round's candidate table, the chosen layers and the kept layers are
   identical to the per-candidate path, under both position policies and with
   the last surviving block among the candidates (the no-suffix fallback);
2. the child bytes are the same;
3. the knob is execution only: absent from the fingerprint, present in the
   trace, and inert on every path it does not cover.

Whether CUDA kernels give the same bits for the same layer input inside and
outside a full forward is the GPU validation's question, not this file's.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.execution import (  # noqa: E402
    ExecutionConfig, ExecutionError,
)
from aadistill.initialization.operators.base import OperatorContext  # noqa: E402
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.scoring.positions import (  # noqa: E402
    ALL_POSITIONS_V1, SUPERVISED_TARGET_V1, policy_config,
)
from aadistill.initialization.scoring.support import reference_topk_tail  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    bypassed_blocks, captured_block_inputs, greedy_removal, has_suffix,
    resumed_after_block,
)

from support.toy import build_tiny_model  # noqa: E402

GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=64, tie_word_embeddings=True)


@pytest.fixture(scope="module", autouse=True)
def _registered():
    from aadistill.initialization.adapters import register_builtin_adapters
    register_builtin_adapters()
    yield


@pytest.fixture(scope="module")
def model():
    m = build_tiny_model(GEOMETRY)
    m.config.use_cache = False
    return m


def _raw_items(n: int = 7, vocab: int = 64):
    g = torch.Generator().manual_seed(31)
    out = []
    for i in range(n):
        length = 10 + 4 * i
        ids = torch.randint(1, vocab, (1, length), generator=g)
        item = {"item_id": f"item-{i}", "input_ids": ids,
                "domain": ("general", "math", "logic")[i % 3],
                "subtype": ("text", "arith", "rules")[i % 3], "tags": {}}
        if i % 2 == 0:
            item["tags"]["assistant"] = list(range(4, length - 1))
        out.append(item)
    return out


def _execution(reuse: bool, batch_size: int = 3) -> ExecutionConfig:
    return ExecutionConfig(micro_batch_size=batch_size,
                           calibration_batch_packing="length_sorted_v1",
                           prediction_rows="scored_v1", depth_prefix_reuse=reuse)


def _depth(model, policy, execution, remove: int = 3):
    items = prepare_calibration_items(_raw_items(), profile_id="toy")
    support = reference_topk_tail(8)
    parent = ArchSpec.of("qwen3", GEOMETRY)
    target = ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": 6 - remove})
    config = {"n_calibration_items": len(items), **policy_config(policy),
              "distribution_support": support.as_dict()}
    ctx = OperatorContext(
        adapter=get_adapter("qwen3"), model=model, parent_spec=parent,
        target_spec=target, profile=None, calibration_items=items, seed=0,
        device="cpu", config=config, distribution_support=support,
        execution=execution, position_policy=policy)
    return causal_kl_greedy.DepthCausalKLGreedyV1().apply(ctx)


def _tables(outcome):
    return [(r["chosen"], [(e["candidate"], e["score"]) for e in r["table"]])
            for r in outcome.artifacts["search_rounds"]]


class TestTheScoresAreBitIdentical:

    @pytest.mark.parametrize("policy", [ALL_POSITIONS_V1, SUPERVISED_TARGET_V1],
                             ids=["all_v1", "supervised_target_v1"])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_every_round_table_is_equal(self, model, policy, batch_size):
        base = _depth(model, policy, _execution(False, batch_size))
        fast = _depth(model, policy, _execution(True, batch_size))
        assert _tables(fast) == _tables(base)          # exact float equality
        assert fast.trace["kept_layers"] == base.trace["kept_layers"]
        assert fast.trace["removal_order"] == base.trace["removal_order"]
        assert fast.local_metrics.values == base.local_metrics.values
        assert fast.artifacts["timing"]["prefix_capture_forwards"] > 0

    def test_the_last_surviving_block_is_scored_by_the_fallback(self, model):
        """Removing 5 of 6 leaves rounds whose last candidate has no suffix."""
        base = _depth(model, SUPERVISED_TARGET_V1, _execution(False), remove=5)
        fast = _depth(model, SUPERVISED_TARGET_V1, _execution(True), remove=5)
        assert _tables(fast) == _tables(base)

    def test_the_child_bytes_are_equal(self, model):
        base = _depth(model, SUPERVISED_TARGET_V1, _execution(False))
        fast = _depth(model, SUPERVISED_TARGET_V1, _execution(True))
        a, b = base.model.state_dict(), fast.model.state_dict()
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)

    def test_the_parent_is_left_intact(self, model):
        before = {k: v.clone() for k, v in model.state_dict().items()}
        n_blocks = len(get_adapter("qwen3").blocks(model))
        _depth(model, SUPERVISED_TARGET_V1, _execution(True))
        assert len(get_adapter("qwen3").blocks(model)) == n_blocks
        after = model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert not any(b._forward_pre_hooks for b in get_adapter("qwen3").blocks(model))


class TestThePrimitives:

    def test_resumed_forward_equals_the_bypassed_forward(self, model):
        adapter = get_adapter("qwen3")
        ids = torch.randint(1, 64, (2, 11), generator=torch.Generator().manual_seed(5))
        attn = torch.ones_like(ids)
        skipped = {4}
        with captured_block_inputs(model, skipped) as inputs:
            adapter.final_hidden_states(model, ids, attn)
        assert sorted(inputs) == [0, 1, 2, 3, 5]
        for c in (0, 2, 3):
            with bypassed_blocks(model, skipped | {c}):
                want = adapter.final_hidden_states(model, ids, attn)
            with resumed_after_block(model, skipped, c, inputs[c]):
                got = adapter.final_hidden_states(model, ids, attn)
            assert torch.equal(got, want)

    def test_has_suffix(self):
        assert has_suffix(6, {4}, 3) is True
        assert has_suffix(6, {5}, 4) is False
        assert has_suffix(6, set(), 5) is False

    def test_a_round_scorer_must_answer_every_candidate(self):
        with pytest.raises(ValueError, match="2 values for 3 candidates"):
            greedy_removal(lambda s: 0.0, 3, 1, score_round=lambda s, c: [0.0, 1.0])

    def test_a_round_scorer_feeds_the_same_table(self):
        def one(skip):
            return float(sum(skip) * 7 % 5)
        def round_(skipped, candidates):
            return [one(skipped | {c}) for c in candidates]
        a = greedy_removal(one, 6, 3)
        b = greedy_removal(one, 6, 3, score_round=round_)
        assert a == b


class TestItIsExecutionOnly:

    def test_trace_but_not_fingerprint(self):
        off, on = _execution(False), _execution(True)
        assert on.as_fingerprint() == off.as_fingerprint()
        assert on.as_trace() == {**off.as_trace(), "depth_prefix_reuse": True}
        assert "depth_prefix_reuse" not in off.as_trace()

    def test_must_be_a_bool(self):
        with pytest.raises(ExecutionError, match="must be a bool"):
            ExecutionConfig(depth_prefix_reuse=1)

    def test_the_trace_records_it_only_when_it_ran(self, model):
        fast = _depth(model, SUPERVISED_TARGET_V1, _execution(True))
        base = _depth(model, SUPERVISED_TARGET_V1, _execution(False))
        assert fast.trace["depth_prefix_reuse"] is True
        assert "depth_prefix_reuse" not in base.trace
        # the all-rows path is not covered: the knob is inert there
        allrows = ExecutionConfig(micro_batch_size=3,
                                  calibration_batch_packing="length_sorted_v1",
                                  depth_prefix_reuse=True)
        out = _depth(model, SUPERVISED_TARGET_V1, allrows)
        assert "depth_prefix_reuse" not in out.trace
        assert "prefix_capture_forwards" not in out.artifacts["timing"]
