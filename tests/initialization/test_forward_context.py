"""The packed causal forward context (maintainer decision 2026-11-03), on a tiny real Qwen3.

Claims:

1. the plan and the packed batch hold each item's FIRST tokens at the recorded
   offsets, in the reference algorithm's order over the seeded shuffle;
2. a packed row is read with standard causal attention and continuous positions:
   the first segment reads exactly as the item alone, a later segment does not
   (it sees the items before it), and nothing builds a block-diagonal mask;
3. the scoring masks land at each segment's offset, and the position predicting
   the next segment's first token belongs to no item;
4. DEPTH's packed scores equal a from-scratch computation over the same
   sequences (per-item means pooling every placement, sub-type means, domain
   balance), and prefix reuse reproduces them;
5. the packed state evaluation equals a from-scratch per-segment reduction, and
   a teacher measured against itself scores zero;
6. the context is SCIENCE: it enters the hashed operator config, the
   measurement protocol id and the search config hash, and a disagreement
   between a declaration and the object is refused.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aadistill.initialization.calibration.items import prepare_calibration_items  # noqa: E402
from aadistill.initialization.calibration.packing import (  # noqa: E402
    ForwardContext, PackingError, forward_context_from, packed_batches,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    ContractViolation, OperatorContext, OperatorError,
)
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.planning.operator_config import (  # noqa: E402
    hashed_operator_config, operator_config_hash,
)
from aadistill.initialization.scoring.batches import active_positions  # noqa: E402
from aadistill.initialization.scoring.positions import (  # noqa: E402
    PREDICTION_AXIS, SUPERVISED_TARGET_V1,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    distortion_on_reference_support, reference_topk_tail, sketch_forward_kl,
    sketch_reference,
)
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.initialization.specs.materialization import NumericalEnvironment  # noqa: E402
from aadistill.initialization.specs.metrics import (  # noqa: E402
    MeasurementError, StateEvalSuite, SuiteItem,
)
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    DistortionSums, bypassed_blocks, domain_balanced_score,
)

from support.toy import build_tiny_model  # noqa: E402

GEOMETRY = dict(hidden_size=32, num_hidden_layers=6, intermediate_size=48,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=64, tie_word_embeddings=True)
CPU = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
CONTEXT = ForwardContext(sequence_length=40, shuffle_seed=9)
SCORED = ExecutionConfig(micro_batch_size=1, prediction_rows="scored_v1")
SUPPORT = reference_topk_tail(8)


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
        length = 9 + 4 * i
        ids = torch.randint(1, vocab, (1, length), generator=g)
        out.append({"item_id": f"item-{i}", "input_ids": ids,
                    "domain": "general" if i % 2 == 0 else "math",
                    "subtype": "text" if i % 2 == 0 else "arith",
                    "tags": {"assistant": list(range(3, length - 1))}})
    return out


def _items():
    return prepare_calibration_items(_raw_items(), profile_id="toy")


# --- 1. plan and batch ----------------------------------------------------------

class TestThePackedBatch:

    def test_each_segment_is_its_items_prefix(self):
        items = _items()
        groups = list(packed_batches(items, 1, pad_id=0, forward_context=CONTEXT))
        assert len(groups) > 1
        fragments = 0
        for packed in groups:
            batch = packed.batch
            assert batch.packed and batch.size == 1
            for seg, item in zip(batch.segments, batch.items):
                assert item is items[seg.index]
                assert batch.input_ids[0, seg.start:seg.start + seg.length].tolist() == \
                    items[seg.index]["input_ids"][0, :seg.length].tolist()
                fragments += not seg.final
            assert packed.original_indices == tuple(s.index for s in batch.segments)
        assert fragments > 0, "the toy plan must exercise a fragment"

    def test_the_plan_is_the_reference_over_the_shuffle(self):
        from aadistill.data.sequence_packing import packing_conversation, shuffled_order
        items = _items()
        order = shuffled_order([i["item_id"] for i in items], CONTEXT.shuffle_seed)
        ref = packing_conversation([int(items[i]["input_ids"].shape[1]) for i in order],
                                   CONTEXT.sequence_length, True)
        assert [[p.index for p in s] for s in CONTEXT.plan(items)] == \
            [[order[i] for i in s] for s in ref]

    def test_a_second_grouping_policy_is_refused(self):
        with pytest.raises(PackingError, match="forward context"):
            list(packed_batches(_items(), 1, packing="length_sorted_v1", pad_id=0,
                                forward_context=CONTEXT))

    def test_per_row_views_refuse_a_packed_batch(self):
        batch = next(iter(packed_batches(_items(), 1, pad_id=0, forward_context=CONTEXT))).batch
        with pytest.raises(Exception, match="packed"):
            batch.split_predictions(torch.zeros(1, batch.input_ids.shape[1] - 1))

    def test_the_context_round_trips_and_refuses_other_rules(self):
        assert forward_context_from(CONTEXT.as_dict()) == CONTEXT
        with pytest.raises(PackingError):
            forward_context_from({**CONTEXT.as_dict(), "packing": "best_fit"})


# --- 2. causal attention, continuous positions -----------------------------------

class TestStandardCausalAttention:

    def test_first_segment_reads_alone_later_ones_do_not(self, model):
        items = _items()
        packed = next(iter(packed_batches(items, 1, pad_id=0, forward_context=CONTEXT)))
        batch = packed.batch
        assert len(batch.segments) >= 2
        logits = model(batch.input_ids, attention_mask=batch.attention_mask).logits[0]
        first, second = batch.segments[0], batch.segments[1]
        alone = model(items[first.index]["input_ids"][:, :first.length]).logits[0]
        torch.testing.assert_close(logits[:first.length], alone, rtol=1e-5, atol=1e-5)
        alone2 = model(items[second.index]["input_ids"][:, :second.length]).logits[0]
        shifted = logits[second.start:second.start + second.length]
        assert not torch.allclose(shifted, alone2, atol=1e-4), \
            "a later segment must see the earlier ones (no block-diagonal mask)"
        # continuous positions: the default 0..T-1 equal explicit ones
        explicit = model(batch.input_ids, position_ids=torch.arange(
            batch.input_ids.shape[1])[None]).logits[0]
        torch.testing.assert_close(logits, explicit, rtol=1e-5, atol=1e-5)


# --- 3. scoring masks ------------------------------------------------------------

class TestScoringMasks:

    def test_masks_land_at_each_segment_offset(self):
        items = _items()
        active = active_positions(items, SUPERVISED_TARGET_V1)
        for packed in packed_batches(items, 1, pad_id=0, forward_context=CONTEXT):
            batch = packed.batch
            pw = active.prediction_weights_for(batch, packed.original_indices)
            pm = batch.prediction_mask()
            tm = active.token_mask_for(batch, packed.original_indices)
            for seg in batch.segments:
                full = SUPERVISED_TARGET_V1.weights(items[seg.index], axis=PREDICTION_AXIS).weights
                assert pw[seg.row, seg.prediction_slice].tolist() == full[:seg.length - 1].tolist()
                end = seg.start + seg.length - 1
                if end < pm.shape[1]:
                    assert not bool(pm[seg.row, end]), "the boundary position belongs to no item"
                    assert float(pw[seg.row, end]) == 0.0
            assert tm is not None and int(tm.sum()) > 0

    def test_mismatched_indices_are_refused(self):
        items = _items()
        active = active_positions(items, SUPERVISED_TARGET_V1)
        packed = next(iter(packed_batches(items, 1, pad_id=0, forward_context=CONTEXT)))
        with pytest.raises(Exception, match="segments"):
            active.prediction_weights_for(packed.batch, tuple(reversed(packed.original_indices)))


# --- 4. DEPTH ----------------------------------------------------------------------

def _depth_ctx(model, items, *, context=CONTEXT, execution=SCORED, support=SUPPORT,
               declare=True):
    parent = ArchSpec.of("qwen3", GEOMETRY)
    target = ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": 4})
    impl = causal_kl_greedy.DepthCausalKLGreedyV1()
    config = hashed_operator_config(implementation=impl, policy=SUPERVISED_TARGET_V1,
                                    support=support, items=items,
                                    forward_context=context if declare else None)
    return impl, OperatorContext(
        adapter=get_adapter("qwen3"), model=model, parent_spec=parent,
        target_spec=target, profile=None, calibration_items=items, seed=0,
        device="cpu", config=config, distribution_support=support,
        execution=execution, position_policy=SUPERVISED_TARGET_V1,
        forward_context=context)


@torch.no_grad()
def _naive_depth_score(model, items, skip):
    """From scratch: full logits of each packed sequence, scored positions per
    segment, Top-8 + tail sketch of the intact parent, per-item pooled means."""
    sums, weights = {}, {}
    for seq in CONTEXT.plan(items):
        ids = torch.cat([items[p.index]["input_ids"][0, :p.tokens] for p in seq])[None]
        ref = model(ids).logits[0, :-1]
        with bypassed_blocks(model, skip):
            cand = model(ids).logits[0, :-1]
        at = 0
        for p in seq:
            w = SUPERVISED_TARGET_V1.weights(items[p.index], axis=PREDICTION_AXIS).weights[:p.tokens - 1]
            pos = torch.arange(at, at + p.tokens - 1)[w > 0]
            if not len(pos):
                at += p.tokens
                continue
            sk = sketch_reference(ref[pos], torch.zeros(len(pos), dtype=torch.long), top_k=8, chunk=512)
            kl = sketch_forward_kl(sk.support_indices, sk.support_log_probs, sk.tail_log_prob,
                                   cand[pos], has_tail=True)
            sums[p.index] = sums.get(p.index, 0.0) + float((kl.double() * w[w > 0].double()).sum())
            weights[p.index] = weights.get(p.index, 0.0) + float(w.sum())
            at += p.tokens
    per_subtype = {}
    for i, item in enumerate(items):
        per_subtype.setdefault(item["subtype"], []).append(sums[i] / weights[i])
    means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
    domains = {}
    for item in items:
        domains.setdefault(item["domain"], set()).add(item["subtype"])
    return domain_balanced_score(means, {d: sorted(s) for d, s in domains.items()})[0]


class TestPackedDepth:

    def test_scores_equal_a_from_scratch_computation(self, model):
        items = _items()
        impl, ctx = _depth_ctx(model, items)
        out = impl.execute(ctx)
        first = out.artifacts["search_rounds"][0]
        for entry in first["table"][:3]:
            want = _naive_depth_score(model, items, frozenset({entry["candidate"]}))
            assert entry["score"] == pytest.approx(want, rel=1e-5, abs=1e-7)
        assert out.trace["forward_context"] == CONTEXT.as_dict()
        assert out.trace["packing"]["fragments"] > 0
        assert out.trace["packing"]["repeated_scored_positions"] > 0
        assert out.artifacts["reference_cache"]["keyed_by"] == "packed_group"

    def test_prefix_reuse_reproduces_the_scores(self, model):
        items = _items()
        impl, ctx = _depth_ctx(model, items)
        base = impl.execute(ctx)
        impl, ctx = _depth_ctx(model, items, execution=ExecutionConfig(
            micro_batch_size=1, prediction_rows="scored_v1", depth_prefix_reuse=True))
        reuse = impl.execute(ctx)
        assert reuse.trace["kept_layers"] == base.trace["kept_layers"]
        for ra, rb in zip(base.artifacts["search_rounds"], reuse.artifacts["search_rounds"]):
            for ea, eb in zip(ra["table"], rb["table"]):
                assert eb["score"] == pytest.approx(ea["score"], rel=1e-6, abs=1e-9)

    def test_two_sequences_per_forward_score_the_same(self, model):
        items = _items()
        impl, ctx = _depth_ctx(model, items)
        one = impl.execute(ctx)
        impl, ctx = _depth_ctx(model, items, execution=ExecutionConfig(
            micro_batch_size=2, prediction_rows="scored_v1"))
        two = impl.execute(ctx)
        assert two.trace["kept_layers"] == one.trace["kept_layers"]

    def test_unsupported_paths_are_refused(self, model):
        items = _items()
        impl, ctx = _depth_ctx(model, items, execution=ExecutionConfig(micro_batch_size=1))
        with pytest.raises(OperatorError, match="scored_v1"):
            impl.execute(ctx)

    def test_a_context_the_config_does_not_declare_is_refused(self, model):
        items = _items()
        impl, ctx = _depth_ctx(model, items, declare=False)
        with pytest.raises(ContractViolation, match="forward context"):
            impl.execute(ctx)

    def test_the_context_forks_the_operator_config_hash(self):
        items = _items()
        impl = causal_kl_greedy.DepthCausalKLGreedyV1()
        a = hashed_operator_config(implementation=impl, policy=SUPERVISED_TARGET_V1,
                                   support=SUPPORT, items=items)
        b = hashed_operator_config(implementation=impl, policy=SUPERVISED_TARGET_V1,
                                   support=SUPPORT, items=items, forward_context=CONTEXT)
        assert "forward_context" not in a and b["forward_context"] == CONTEXT.as_dict()
        assert operator_config_hash(a) != operator_config_hash(b)


# --- 5. state evaluation ------------------------------------------------------------

def _suite():
    items = []
    for raw in _raw_items():
        n_pred = int(raw["input_ids"].shape[1]) - 1
        assistant = torch.zeros(n_pred, dtype=torch.bool)
        assistant[raw["tags"]["assistant"]] = True
        tags = {"assistant": assistant, "eos_like": raw["input_ids"][0, 1:] % 5 == 0}
        items.append(SuiteItem(item_id=raw["item_id"], input_ids=raw["input_ids"],
                               domain=raw["domain"], subtype=raw["subtype"], tags=tags))
    suite = StateEvalSuite(
        suite_id="test.packed", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like",), n_items=len(items),
        description="packed state evaluation", general_domain="general")
    return suite, items


def _evaluator(context=CONTEXT, **kw):
    suite, items = _suite()
    return StateEvaluator(suite, items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
                          execution=SCORED, distribution_support=SUPPORT, numerics=CPU,
                          suite_content_sha256="toy", forward_context=context, **kw)


class TestPackedStateEvaluation:

    @torch.no_grad()
    def test_per_subtype_kl_equals_a_from_scratch_reduction(self, model):
        candidate = build_tiny_model(GEOMETRY, seed=19)
        ev = _evaluator()
        ev.prime_reference(model)
        got = ev.evaluate(candidate, "candidate")
        suite, items = _suite()
        per_subtype = {}
        for seq in CONTEXT.plan([{"item_id": i.item_id, "input_ids": i.input_ids} for i in items]):
            ids = torch.cat([items[p.index].input_ids[0, :p.tokens] for p in seq])[None]
            ref, cand = model(ids).logits[0, :-1], candidate(ids).logits[0, :-1]
            at = 0
            for p in seq:
                item = items[p.index]
                n = p.tokens - 1
                sel = item.tags["assistant"][:n]
                pos = torch.arange(at, at + n)[sel]
                if not len(pos):
                    at += p.tokens
                    continue
                targets = item.input_ids[0, 1:p.tokens][sel]
                sk = sketch_reference(ref[pos], targets, top_k=8, chunk=512)
                sums = distortion_on_reference_support(
                    sk, cand[pos], targets, tags={k: v[:n][sel] for k, v in item.tags.items()},
                    weights=torch.ones(int(sel.sum()), dtype=torch.float64), chunk=512)
                per_subtype.setdefault(item.subtype, DistortionSums()).merge(sums)
                at += p.tokens
        for sub, want in per_subtype.items():
            assert got.detail["per_subtype_kl"][sub] == pytest.approx(
                want.as_dict()["kl"], rel=1e-5, abs=1e-8)
        assert got.detail["forward_context"] == CONTEXT.as_dict()

    def test_the_teacher_against_itself_scores_zero_and_caches_its_sketches(self, model):
        ev = _evaluator()
        ev.prime_reference(model)
        first = ev.evaluate(model, "teacher")
        again = ev.evaluate(model, "teacher")
        assert first.values["state.teacher_kl.equal_domain_mean"] == pytest.approx(0.0, abs=1e-6)
        assert again.values == first.values and again.detail == first.detail
        assert ev.packed_teacher_forwards_skipped > 0

    def test_the_context_moves_the_protocol_id(self):
        assert _evaluator().measurement_protocol_id != _evaluator(context=None).measurement_protocol_id
        other = ForwardContext(sequence_length=40, shuffle_seed=10)
        assert _evaluator().measurement_protocol_id != _evaluator(context=other).measurement_protocol_id

    def test_unsupported_paths_are_refused(self):
        suite, items = _suite()
        with pytest.raises(MeasurementError, match="scored_v1"):
            StateEvaluator(suite, items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
                           execution=ExecutionConfig(micro_batch_size=1),
                           distribution_support=SUPPORT, numerics=CPU, forward_context=CONTEXT)


# --- 6. the search binds the context -------------------------------------------------

def test_the_search_config_hash_carries_the_context():
    from dataclasses import replace

    from aadistill.initialization.planning.search import SearchConfig
    import support.toy as toy
    suite, _ = _suite()
    cfg = SearchConfig(run_id="r", target_spec=ArchSpec.of("qwen3", toy.TARGET_GEOMETRY),
                       schedule=__import__("aadistill.initialization.planning.ranking",
                                           fromlist=["SCHEDULE_V1"]).SCHEDULE_V1,
                       seed=0, workdir=".", profiles=(toy.make_profile("a"),),
                       policy=__import__("aadistill.initialization.planning.ranking",
                                         fromlist=["PARETO_V1"]).PARETO_V1, suite=suite)
    assert "forward_context" not in cfg.as_dict()
    assert replace(cfg, forward_context=CONTEXT).config_hash != cfg.config_hash


class _Measurer:
    """The evaluator as a search measurer, exposing what the search checks."""

    def __init__(self, ev):
        self.ev = ev
        self.forward_context = ev.forward_context
        self.distribution_support = ev.distribution_support
        self.measurement_protocol_id = ev.measurement_protocol_id

    def __call__(self, model, digest):
        return self.ev.evaluate(model, digest)


class TestASearchUnderTheContext:
    """A real `BeamSearch` over DEPTH and FFN with the packed context end to end."""

    def _search(self, tmp_path, *, evaluator_context=CONTEXT):
        from support.toy import TEACHER_GEOMETRY, make_items, make_profile
        from aadistill.initialization.operators.register import register_builtin_operators
        from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1
        from aadistill.initialization.planning.search import BeamSearch, SearchConfig
        from aadistill.initialization.specs.arch import adapter_for_config
        from aadistill.initialization.specs.metrics import ReferenceStrategy
        register_builtin_operators()
        teacher = build_tiny_model(TEACHER_GEOMETRY)
        teacher.config.use_cache = False
        calib = prepare_calibration_items(make_items(n_per_subtype=3), profile_id="test.balanced@v1")
        suite_items = tuple(SuiteItem(item_id=i["item_id"], domain=i["domain"], subtype=i["subtype"],
                                      input_ids=i["input_ids"], tags=i["tags"])
                            for i in prepare_calibration_items(make_items(seed=303), profile_id="s"))
        suite = StateEvalSuite(suite_id="test.packed_search", version=1, domains=("general", "math"),
                               subtypes={"general": ("text",), "math": ("arith",)},
                               critical_tags=("eos_like", "answer_like"), n_items=len(suite_items),
                               general_domain="general")
        ev = StateEvaluator(suite, suite_items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
                            execution=SCORED, distribution_support=SUPPORT, numerics=CPU,
                            suite_content_sha256="toy", reference_strategy=ReferenceStrategy.RECOMPUTE,
                            forward_context=evaluator_context)
        ev.prime_reference(teacher)
        target = {**TEACHER_GEOMETRY, "num_hidden_layers": 4, "intermediate_size": 24}
        config = SearchConfig(
            run_id="packed", target_spec=ArchSpec.of("qwen3", target), schedule=SCHEDULE_V1, seed=7,
            workdir=tmp_path, profiles=(make_profile("balanced"),), policy=PARETO_V1, suite=suite,
            position_policy=SUPERVISED_TARGET_V1, distribution_support=SUPPORT,
            measurement_protocol_id=ev.measurement_protocol_id, forward_context=CONTEXT,
            allowed_impls=("depth.causal_kl_greedy_v1", "ffn.activation_importance_v0"),
            device="cpu")
        return BeamSearch(adapter=adapter_for_config(teacher.config), config=config,
                          root_teacher_id="toy/teacher", root_teacher_sha256="de" * 32,
                          root_loader=lambda: teacher, calibration_loader=lambda profile: calib,
                          measurer=_Measurer(ev), execution=SCORED, numerics=CPU)

    def test_it_runs_and_every_state_carries_the_context(self, tmp_path):
        result = self._search(tmp_path).run()
        assert result.complete_leaves
        evaluated = [s for s in result.states.values() if s.evaluation is not None]
        assert evaluated and all(
            s.evaluation.detail.get("forward_context") == CONTEXT.as_dict() for s in evaluated)
        stepped = [s for s in result.states.values() if s.steps]
        assert stepped and all(s.steps[-1].config_hash for s in stepped)

    def test_a_measurer_reading_another_context_is_refused(self, tmp_path):
        from aadistill.initialization.planning.search import SearchError
        with pytest.raises(SearchError, match="forward context"):
            self._search(tmp_path, evaluator_context=None)

    def test_a_replacement_run_resumes_every_state_from_the_journal(self, tmp_path):
        first = self._search(tmp_path).run()
        again = self._search(tmp_path).run()
        expanded = [s for s in first.states.values() if s.parent_id is not None]
        assert expanded and len(again.resumed) == len(expanded)
        assert sorted(again.states) == sorted(first.states)
