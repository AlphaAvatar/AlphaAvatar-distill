"""Pre-vocabulary row selection is an execution path, not a new measurement.

`ExecutionConfig.prediction_rows="scored_v1"` runs the backbone over the whole
sequence and projects only the rows a reducer weights. The claims, each on a
tiny but genuine Qwen3 model with right-padded batches:

1. the historical identity is untouched: the default adds no trace or
   fingerprint key, and the default fingerprint is the pre-field hash;
2. the selected rows ARE the model's own logits at those positions -- same
   causal shift, same row order;
3. DEPTH chooses the same layers with the same per-candidate scores;
4. the state evaluator reports the same values, counts and denominators, under
   both supports, both policies and both forward groupings, while its
   protocol id records that the row-selected path ran.

What a CPU cannot answer -- whether CUDA's output-head GEMM on N selected rows
reduces bit-identically to the same rows of a B*T GEMM -- is the GPU
validation's question, and these tests deliberately do not pretend to it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.calibration.packing import packed_batches  # noqa: E402
from aadistill.initialization.execution import (  # noqa: E402
    ExecutionConfig, ExecutionError,
)
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorContext, OperatorError,
)
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.scoring.positions import (  # noqa: E402
    ALL_POSITIONS_V1, SUPERVISED_TARGET_V1, policy_config,
)
from aadistill.initialization.scoring.rows import (  # noqa: E402
    scatter_rows, scored_prediction_logits, scored_rows, split_rows,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    FULL_VOCAB_V1, reference_topk_tail,
)
from aadistill.initialization.specs.arch import (  # noqa: E402
    ArchSpec, UnsupportedCapability, get_adapter,
)
from aadistill.initialization.specs.materialization import (  # noqa: E402
    FINGERPRINT_SCHEMA, NumericalEnvironment, numerical_execution_fingerprint,
)
from aadistill.initialization.specs.metrics import (  # noqa: E402
    MeasurementError, StateEvalSuite, SuiteItem,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

from support.toy import build_tiny_model  # noqa: E402

GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=64, tie_word_embeddings=True)
CPU = NumericalEnvironment(device_type="cpu", compute_dtype="float32")


def _execution(batch_size: int, rows: str) -> ExecutionConfig:
    return ExecutionConfig(
        micro_batch_size=batch_size,
        calibration_batch_packing=("length_sorted_v1" if batch_size > 1
                                   else "original_order_v1"),
        prediction_rows=rows)


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


def _raw_items(n: int = 6, vocab: int = 64):
    """Unequal lengths, so batches pad; HALF tagged, like the frozen mixtures."""
    g = torch.Generator().manual_seed(23)
    out = []
    for i in range(n):
        length = 12 + 3 * i
        ids = torch.randint(1, vocab, (1, length), generator=g)
        item = {"item_id": f"item-{i}", "input_ids": ids,
                "domain": "general" if i % 2 == 0 else "math",
                "subtype": "text" if i % 2 == 0 else "arith", "tags": {}}
        if i % 2 == 0:
            item["tags"]["assistant"] = list(range(5, length - 1))
            item["tags"]["final_answer"] = list(range(length - 4, length - 1))
        out.append(item)
    return out


# --- 1. identity -------------------------------------------------------------

class TestTheHistoricalIdentityIsUntouched:

    def test_the_default_adds_no_trace_or_fingerprint_key(self):
        cfg = ExecutionConfig(micro_batch_size=3,
                              calibration_batch_packing="length_sorted_v1")
        assert set(cfg.as_trace()) == {"micro_batch_size",
                                       "calibration_batch_packing"}
        assert set(cfg.as_fingerprint()) == set(cfg.as_trace())
        assert cfg.selects_prediction_rows is False

    def test_the_default_fingerprint_is_the_pre_field_hash(self):
        """Derived independently: the five-field preimage every recorded
        fingerprint was computed from before this field existed."""
        cfg = ExecutionConfig(micro_batch_size=3,
                              calibration_batch_packing="length_sorted_v1")
        expected = sha256_json({
            "schema": FINGERPRINT_SCHEMA, "micro_batch_size": 3,
            "calibration_batch_packing": "length_sorted_v1",
            "device_type": "cpu", "compute_dtype": "float32",
            "accumulation_dtype": "float64"})
        assert numerical_execution_fingerprint(cfg, CPU) == expected

    def test_selecting_rows_forks_the_fingerprint(self):
        base = _execution(3, "all_v1")
        scored = _execution(3, "scored_v1")
        assert scored.as_trace()["prediction_rows"] == "scored_v1"
        assert (numerical_execution_fingerprint(base, CPU)
                != numerical_execution_fingerprint(scored, CPU))

    def test_an_unknown_row_policy_is_refused(self):
        with pytest.raises(ExecutionError, match="prediction_rows"):
            ExecutionConfig(prediction_rows="active_v9")


# --- 2. the rows are the model's own logits ------------------------------------

class TestTheSelectedRowsAreTheModelsLogits:

    def _batch(self):
        items = prepare_calibration_items(_raw_items(), profile_id="toy")
        return next(iter(packed_batches(items, 3, packing="length_sorted_v1",
                                        pad_id=0, device="cpu"))).batch

    def test_same_positions_same_values_same_order(self, model):
        batch = self._batch()
        adapter = get_adapter("qwen3")
        mask = batch.prediction_mask()
        assert not bool(mask.all()), "the batch must pad, or nothing is tested"
        g = torch.Generator().manual_seed(5)
        chosen = torch.rand(mask.shape, generator=g) > 0.4
        rows = scored_rows(mask, chosen.double())
        full = model(batch.input_ids, attention_mask=batch.attention_mask
                     ).logits[:, :-1]
        picked = scored_prediction_logits(adapter, model, batch.input_ids,
                                          batch.attention_mask, rows)
        assert picked.shape == (int(rows.sum()), GEOMETRY["vocab_size"])
        torch.testing.assert_close(picked, full[rows], rtol=1e-5, atol=1e-6)

    def test_padding_is_never_selected(self, model):
        batch = self._batch()
        mask = batch.prediction_mask()
        rows = scored_rows(mask, torch.ones(mask.shape, dtype=torch.float64))
        assert bool((rows <= mask).all()) and bool((rows == mask).all())

    def test_scatter_and_split_undo_row_major_order(self):
        rows = torch.tensor([[True, False, True], [False, True, True]])
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dense = scatter_rows(values, rows)
        assert dense.tolist() == [[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]]
        assert [t.tolist() for t in split_rows(values, rows)] == [[1.0, 2.0],
                                                                  [3.0, 4.0]]

    def test_an_adapter_without_the_capability_is_refused(self, model):
        class NoHead(type(get_adapter("qwen3"))):
            capabilities = frozenset()

        rows = torch.ones(1, 3, dtype=torch.bool)
        with pytest.raises(UnsupportedCapability):
            scored_prediction_logits(NoHead(), model,
                                     torch.ones(1, 4, dtype=torch.long), None,
                                     rows)


# --- 3. DEPTH ----------------------------------------------------------------

def _depth(model, policy, execution, support):
    items = prepare_calibration_items(_raw_items(), profile_id="toy")
    parent = ArchSpec.of("qwen3", GEOMETRY)
    target = ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": 4})
    config = {"n_calibration_items": len(items), **policy_config(policy)}
    if not support.is_full_vocab:
        config["distribution_support"] = support.as_dict()
    ctx = OperatorContext(
        adapter=get_adapter("qwen3"), model=model, parent_spec=parent,
        target_spec=target, profile=None, calibration_items=items, seed=0,
        device="cpu", config=config, distribution_support=support,
        execution=execution, position_policy=policy)
    return causal_kl_greedy.DepthCausalKLGreedyV1().apply(ctx)


def _scores(outcome):
    return [(r["chosen"], [(e["candidate"], e["score"]) for e in r["table"]])
            for r in outcome.artifacts["search_rounds"]]


class TestDepthChoosesTheSameLayers:

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("policy", [ALL_POSITIONS_V1, SUPERVISED_TARGET_V1],
                             ids=["all_v1", "supervised_target_v1"])
    def test_decisions_and_scores_agree(self, model, policy, batch_size):
        support = reference_topk_tail(8)       # K < V: the tail bucket is real
        base = _depth(model, policy, _execution(batch_size, "all_v1"), support)
        rows = _depth(model, policy, _execution(batch_size, "scored_v1"),
                      support)
        assert rows.trace["kept_layers"] == base.trace["kept_layers"]
        for (chose_a, table_a), (chose_b, table_b) in zip(_scores(base),
                                                          _scores(rows)):
            assert chose_a == chose_b
            assert [c for c, _ in table_a] == [c for c, _ in table_b]
            for (_, a), (_, b) in zip(table_a, table_b):
                assert b == pytest.approx(a, rel=1e-6, abs=1e-9)

    def test_only_scored_rows_reach_the_output_head(self, model):
        out = _depth(model, SUPERVISED_TARGET_V1, _execution(3, "scored_v1"),
                     reference_topk_tail(8))
        timing = out.artifacts["timing"]
        assert 0 < timing["scored_rows"] < timing["forward_rows"]
        assert out.trace["prediction_rows"] == "scored_v1"
        assert out.artifacts["reference_cache"]["scored_rows_only"] is True

    def test_the_all_rows_trace_is_unchanged(self, model):
        out = _depth(model, SUPERVISED_TARGET_V1, _execution(3, "all_v1"),
                     reference_topk_tail(8))
        assert "prediction_rows" not in out.trace
        assert "scored_rows_only" not in out.artifacts["reference_cache"]

    def test_the_full_vocabulary_oracle_is_refused(self, model):
        with pytest.raises(OperatorError, match="oracle"):
            _depth(model, SUPERVISED_TARGET_V1, _execution(3, "scored_v1"),
                   FULL_VOCAB_V1)


# --- 4. state evaluation -------------------------------------------------------

def _suite_and_items():
    items = []
    for raw in _raw_items():
        n_pred = int(raw["input_ids"].shape[1]) - 1
        tags = {}
        for name, index in raw["tags"].items():
            mask = torch.zeros(n_pred, dtype=torch.bool)
            mask[index] = True
            tags[name] = mask
        #: A critical-style tag on EVERY item, including untagged ones, so the
        #: count restoration is exercised on positions the policy scores and on
        #: an item where it scores everything.
        tags["eos_like"] = raw["input_ids"][0, 1:] % 7 == 0
        items.append(SuiteItem(item_id=raw["item_id"],
                               input_ids=raw["input_ids"], domain=raw["domain"],
                               subtype=raw["subtype"], tags=tags))
    suite = StateEvalSuite(
        suite_id="test.rows", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "final_answer"), n_items=len(items),
        description="row-selection equivalence", general_domain="general")
    return suite, items


def _evaluate(model, teacher, *, support, execution, policy):
    suite, items = _suite_and_items()
    ev = StateEvaluator(suite, items, device="cpu", position_policy=policy,
                        execution=execution, distribution_support=support,
                        numerics=CPU, suite_content_sha256="toy")
    ev.prime_reference(teacher)
    return ev, ev.evaluate(model, "candidate")


@pytest.fixture(scope="module")
def candidate():
    return build_tiny_model(GEOMETRY, seed=19)


class TestStateEvaluationIsTheSameMeasurement:

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("support", [FULL_VOCAB_V1, reference_topk_tail(8)],
                             ids=["full_vocab", "topk_tail"])
    @pytest.mark.parametrize("policy", [ALL_POSITIONS_V1, SUPERVISED_TARGET_V1],
                             ids=["all_v1", "supervised_target_v1"])
    def test_values_counts_and_denominators_agree(self, model, candidate,
                                                  support, batch_size, policy):
        ev_a, base = _evaluate(candidate, model, support=support, policy=policy,
                               execution=_execution(batch_size, "all_v1"))
        ev_b, rows = _evaluate(candidate, model, support=support, policy=policy,
                               execution=_execution(batch_size, "scored_v1"))
        assert set(rows.values) == set(base.values)
        for key, value in base.values.items():
            #: A DELTA is a difference of two pooled CEs of ~ln(V) each, so the
            #: float32 regrouping of their sums (fewer rows per chunk) shows up
            #: magnified by the cancellation. It is bounded at the scale of the
            #: CEs it subtracts, not at its own; every other value at 1e-6 rel.
            tol = (dict(abs=1e-6) if "delta" in key
                   else dict(rel=1e-6, abs=1e-9))
            assert rows.values[key] == pytest.approx(value, **tol), key
        assert rows.positions == base.positions
        assert rows.detail["scored_weight"] == pytest.approx(
            base.detail["scored_weight"])
        assert rows.detail["tagged"].keys() == base.detail["tagged"].keys()
        for tag, entry in base.detail["tagged"].items():
            assert rows.detail["tagged"][tag]["positions"] == entry["positions"]
            assert rows.detail["tagged"][tag]["weight"] == pytest.approx(
                entry["weight"])
        #: Same estimand, different arithmetic path: the identity says which.
        assert ev_a.measurement_protocol_id != ev_b.measurement_protocol_id

    def test_scored_rows_without_declared_numerics_are_refused(self, model):
        suite, items = _suite_and_items()
        with pytest.raises(MeasurementError, match="NumericalEnvironment"):
            StateEvaluator(suite, items, device="cpu",
                           position_policy=SUPERVISED_TARGET_V1,
                           execution=_execution(3, "scored_v1"))
