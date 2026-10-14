"""The reference-confidence objective: frozen-reference weights on a root KL.

`FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL` weights each scored position's KL by
the ROOT reference's own full-vocabulary confidence ``c = 1 - H / log|V|``,
measures the KL against that root for every consumer (DEPTH included, where
the historical reference is the parent), and -- when declared -- weights the
structural operators' activation statistics by the same confidence.

Each claim that would be easy to get backwards is pinned against an
independent computation: the entropy and its direction, which model's
distribution the weight comes from, the fixed ``1/T_x`` denominator, the
root-vs-parent reference, the token alignment of the statistics weights and
the weighted denominators.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from support.toy import (  # noqa: E402
    TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile,
)
from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.calibration.batching import (  # noqa: E402
    BatchingError, micro_batches,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators._common import (  # noqa: E402
    SurgeryError, collect_activation_stats, statistics_positions,
)
from aadistill.initialization.operators.attention.gqa._statistics import (  # noqa: E402
    ATTN_TOKEN_WEIGHT, AttentionHeadStatsCollector,
)
from aadistill.initialization.operators.base import (  # noqa: E402
    ContractViolation, OperatorContext, get_implementation,
)
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.planning.operator_config import (  # noqa: E402
    hashed_operator_config, root_reference_needed, statistics_weighted_by_root,
)
from aadistill.initialization.planning.ranking import (  # noqa: E402
    SCHEDULE_V1, STATE_ID_TIEBREAK, VALIDITY_GUARDRAIL, BeamRankingPolicy,
    Objective,
)
from aadistill.initialization.planning.search import (  # noqa: E402
    BeamSearch, SearchConfig, SearchError,
)
from aadistill.initialization.scoring.batches import ActivePositions  # noqa: E402
from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL, FORM_UNCERTAINTY_WEIGHTED_KL,
    KL_REFERENCE_ROOT, ObjectiveError, ScoringObjective,
    confidence_from_vocabulary_entropy, full_vocabulary_entropy,
    position_terms, student_topk_terms,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    SUPERVISED_TARGET_V1, policy_config,
)
from aadistill.initialization.scoring.reference_store import (  # noqa: E402
    ReferenceStoreError, build_root_reference_store, root_reference_identity,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    FULL_VOCAB_V1, reference_topk_tail, sketch_forward_kl, sketch_reference,
)
from aadistill.initialization.specs.arch import (  # noqa: E402
    ArchSpec, adapter_for_config, get_adapter,
)
from aadistill.initialization.specs.materialization import (  # noqa: E402
    NumericalEnvironment,
)
from aadistill.initialization.specs.metrics import (  # noqa: E402
    StateEvalSuite, SuiteItem,
)
from aadistill.initialization.statistics.collect import (  # noqa: E402
    RESIDUAL_WEIGHT, ActivationStatsCollector, residual_covariance,
    statistics_denominator,
)
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    bypassed_blocks, domain_balanced_score,
)
from aadistill.initialization.transforms.project import (  # noqa: E402
    ffn_neuron_importance, uncentered_moment,
)

K = 8
FULL = ScoringObjective("objective.toy_reference_confidence", 1,
                        FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
                        kl_reference=KL_REFERENCE_ROOT,
                        weights_activation_statistics=True)
NARROW = ScoringObjective("objective.toy_reference_confidence_narrow", 1,
                          FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
                          kl_reference=KL_REFERENCE_ROOT)
STUDENT = ScoringObjective("objective.toy_uncertainty", 1,
                           FORM_UNCERTAINTY_WEIGHTED_KL, entropy_top_k=K)
CPU = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
GEOMETRY = dict(TEACHER_GEOMETRY)
ROOT_ID = {"teacher_id": "toy/root", "teacher_sha256": "ab" * 32}


def _brute_entropy(row: np.ndarray) -> float:
    x = row.astype(np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


@pytest.fixture(scope="module", autouse=True)
def _registered():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )
    register_builtin_adapters()
    register_builtin_operators()
    yield


@pytest.fixture(scope="module")
def root():
    m = build_tiny_model(GEOMETRY)
    m.config.use_cache = False
    return m


@pytest.fixture(scope="module")
def parent():
    """A model that is NOT the root, standing in for a compressed parent."""
    m = build_tiny_model(GEOMETRY, seed=19)
    m.config.use_cache = False
    return m


def _items(seq_lens=(20, 14, 17, 11)):
    raw = make_items(n_per_subtype=2, seq_len=max(seq_lens))
    for item, n in zip(raw, seq_lens):
        item["input_ids"] = item["input_ids"][:, :n]
        m = n - 1
        item["tags"] = {k: v[:m] for k, v in item["tags"].items()}
        item["tags"]["assistant"] = torch.arange(m) >= 4
    return prepare_calibration_items(raw, profile_id="toy@v1")


def _store(model, items, *, bsz=1, support=None):
    support = support or reference_topk_tail(K)
    return build_root_reference_store(
        model, items, support=support,
        identity=root_reference_identity(root_identity=ROOT_ID,
                                         profile_hash="p" * 64,
                                         support=support),
        micro_batch_size=bsz,
        packing="length_sorted_v1" if bsz > 1 else "original_order_v1",
        device="cpu")


# --- 1. the entropy: full vocabulary, the reference's, pointing the right way --

class TestFullVocabularyEntropy:

    def test_matches_an_independent_computation(self):
        g = torch.Generator().manual_seed(5)
        logits = torch.randn(13, 300, generator=g) * 4
        h = full_vocabulary_entropy(logits)
        for r in range(13):
            assert float(h[r]) == pytest.approx(
                _brute_entropy(logits[r].numpy()), abs=2e-5)

    @pytest.mark.parametrize("chunk", [1, 3, 512])
    def test_chunking_changes_nothing(self, chunk):
        logits = torch.randn(9, 50, generator=torch.Generator().manual_seed(1))
        assert torch.equal(full_vocabulary_entropy(logits, chunk=chunk),
                           full_vocabulary_entropy(logits, chunk=512))

    def test_bfloat16_is_promoted_to_float32(self):
        logits = torch.randn(4, 64).bfloat16()
        h = full_vocabulary_entropy(logits)
        assert h.dtype == torch.float32
        assert torch.allclose(h, full_vocabulary_entropy(logits.float()))

    def test_boundaries(self):
        v = 64
        uniform = torch.zeros(1, v)
        peaked = torch.full((1, v), -1e4)
        peaked[0, 3] = 0.0
        h = full_vocabulary_entropy(torch.cat([uniform, peaked]))
        assert float(h[0]) == pytest.approx(math.log(v), abs=1e-5)
        assert float(h[1]) == pytest.approx(0.0, abs=1e-6)
        c = confidence_from_vocabulary_entropy(h, v)
        assert float(c[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(c[1]) == pytest.approx(1.0, abs=1e-6)

    def test_sharper_means_lower_entropy_and_higher_confidence(self):
        base = torch.randn(1, 80, generator=torch.Generator().manual_seed(2))
        h = full_vocabulary_entropy(torch.cat([base, 2 * base, 4 * base]))
        assert float(h[0]) > float(h[1]) > float(h[2])
        c = confidence_from_vocabulary_entropy(h, 80)
        assert float(c[0]) < float(c[1]) < float(c[2])
        assert torch.allclose(c, 1 - h / math.log(80))

    def test_it_is_not_the_topk_conditional_entropy(self):
        logits = torch.randn(6, 100, generator=torch.Generator().manual_seed(4))
        full = full_vocabulary_entropy(logits)
        topk, _ = student_topk_terms(logits, torch.logsumexp(logits, -1),
                                     top_k=K)
        assert not torch.allclose(full, topk.float(), atol=1e-3)

    def test_a_vocabulary_without_scale_is_refused(self):
        with pytest.raises(ObjectiveError, match="no entropy scale"):
            confidence_from_vocabulary_entropy(torch.zeros(1), 1)


# --- 2. the form: identity, integrand, denominator ---------------------------

class TestTheForm:

    def test_it_reads_no_student_quantity_and_names_its_reference(self):
        d = FULL.as_dict()
        assert d["kl_reference"] == KL_REFERENCE_ROOT
        assert d["confidence_source"].startswith("reference_full_vocabulary")
        assert "entropy_top_k" not in d and "reward_scale" not in d
        assert FULL.requires_root_reference and FULL.uses_reference_confidence
        assert not FULL.uses_student_confidence

    def test_the_scope_is_part_of_the_identity(self):
        assert FULL.objective_hash != NARROW.objective_hash
        assert FULL.as_dict()["statistics_weighting"] != \
            NARROW.as_dict()["statistics_weighting"]

    def test_student_forms_keep_their_historical_shape(self):
        assert set(STUDENT.as_dict()) == {
            "schema", "objective_id", "version", "form", "entropy_definition",
            "entropy_top_k", "confidence_mapping", "denominator",
            "aggregation", "direction"}

    @pytest.mark.parametrize("bad", [
        dict(entropy_top_k=8), dict(reward_scale=1.0),
        dict(kl_reference=None), dict(kl_reference="parent")])
    def test_the_reference_form_refuses_foreign_fields(self, bad):
        kw = {"kl_reference": KL_REFERENCE_ROOT, **bad}
        with pytest.raises(ObjectiveError):
            ScoringObjective("x", 1, FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL, **kw)

    @pytest.mark.parametrize("bad", [dict(kl_reference=KL_REFERENCE_ROOT),
                                     dict(weights_activation_statistics=True)])
    def test_student_forms_refuse_the_reference_fields(self, bad):
        with pytest.raises(ObjectiveError):
            ScoringObjective("x", 1, FORM_UNCERTAINTY_WEIGHTED_KL,
                             entropy_top_k=K, **bad)

    def test_the_integrand_is_confidence_times_kl(self):
        kl = torch.tensor([0.0, 1.0, 2.0, float("inf"), float("inf")])
        c = torch.tensor([0.5, 0.25, 1.0, 0.3, 0.0])
        g, w = FULL.integrand(kl, c)
        assert torch.equal(w, c)
        assert g[:3].tolist() == [0.0, 0.25, 2.0]
        assert math.isinf(float(g[3])) and math.isinf(float(g[4]))

    def test_the_candidate_cannot_move_the_weight(self):
        kl = torch.rand(7)
        ent = torch.rand(7)
        conf = confidence_from_vocabulary_entropy(ent, 50)
        a = position_terms(FULL, kl, torch.randn(7, 50), torch.zeros(7), None,
                           reference_entropy=ent, reference_confidence=conf)
        b = position_terms(FULL, kl, torch.randn(7, 50) * 9, torch.zeros(7),
                           None, reference_entropy=ent,
                           reference_confidence=conf)
        assert torch.equal(a["component"], b["component"])
        assert torch.equal(a["objective"], b["objective"])

    def test_a_missing_or_misaligned_weight_is_refused(self):
        with pytest.raises(ObjectiveError, match="none was supplied"):
            position_terms(FULL, torch.rand(4), torch.randn(4, 9),
                           torch.zeros(4), None)
        with pytest.raises(ObjectiveError, match="aligned"):
            position_terms(FULL, torch.rand(4), torch.randn(4, 9),
                           torch.zeros(4), None,
                           reference_entropy=torch.rand(3),
                           reference_confidence=torch.rand(3))

    def test_one_rule_says_who_needs_the_root(self):
        depth = get_implementation("depth.causal_kl_greedy_v1")
        ffn = get_implementation("ffn.activation_importance_v0")
        for impl in (depth, ffn):
            assert root_reference_needed(impl, FULL)
        assert root_reference_needed(depth, NARROW)
        assert not root_reference_needed(ffn, NARROW)
        assert not root_reference_needed(depth, STUDENT)
        assert statistics_weighted_by_root(ffn, FULL)
        assert not statistics_weighted_by_root(ffn, NARROW)
        assert not statistics_weighted_by_root(depth, FULL)

    def test_only_full_scope_puts_the_objective_in_a_statistics_config(self):
        items = _items()
        common = dict(policy=SUPERVISED_TARGET_V1,
                      support=reference_topk_tail(K), items=items)
        ffn = get_implementation("ffn.activation_importance_v0")
        full = hashed_operator_config(implementation=ffn, objective=FULL,
                                      **common)
        narrow = hashed_operator_config(implementation=ffn, objective=NARROW,
                                        **common)
        none = hashed_operator_config(implementation=ffn, objective=None,
                                      **common)
        assert full["scoring_objective"] == FULL.as_dict()
        assert narrow == none


# --- 3. the root reference store -----------------------------------------------

class TestTheRootReferenceStore:

    def test_each_entry_is_the_roots_own_distribution(self, root):
        items = _items()
        store = _store(root, items)
        for item in items:
            with torch.no_grad():
                logits = root(item["input_ids"]).logits[0, :-1].float()
            h = torch.tensor([_brute_entropy(r.numpy()) for r in logits],
                             dtype=torch.float64)
            assert torch.allclose(store.per_position(item, "entropy").double(),
                                  h, atol=2e-5)
            assert torch.allclose(
                store.per_position(item, "confidence").double(),
                1 - h / math.log(GEOMETRY["vocab_size"]), atol=2e-6)
            ref = sketch_reference(logits, item["input_ids"][0, 1:], top_k=K)
            sk = store.sketch_item(item)
            assert torch.equal(sk.support_indices, ref.support_indices)
            assert torch.allclose(sk.support_log_probs, ref.support_log_probs)

    def test_batch_one_and_three_store_the_same_values(self, root):
        items = _items()
        one, three = _store(root, items), _store(root, items, bsz=3)
        assert one.identity_hash == three.identity_hash
        for item in items:
            for name in ("entropy", "confidence", "target_log_prob"):
                assert torch.allclose(one.per_position(item, name),
                                      three.per_position(item, name),
                                      atol=1e-5), name

    def test_identity_is_a_declaration_not_the_bytes(self, root, parent):
        items = _items()
        a, b = _store(root, items), _store(parent, items)
        assert a.identity_hash == b.identity_hash
        assert a.content_digest() != b.content_digest()
        other = root_reference_identity(root_identity={"teacher_id": "x"},
                                        profile_hash="p" * 64,
                                        support=reference_topk_tail(K))
        assert other != a.identity

    def test_it_refuses_items_it_was_not_built_from(self, root):
        items = _items()
        store = _store(root, items[:2])
        with pytest.raises(ReferenceStoreError, match="another"):
            store.require_items(items)
        altered = [dict(items[0], input_ids=items[0]["input_ids"].flip(-1))]
        with pytest.raises(ReferenceStoreError, match="content"):
            store.require_items(altered)
        with pytest.raises(ReferenceStoreError, match="consumer"):
            store.require_identity(profile_hash="q" * 64)

    def test_the_full_vocabulary_support_is_not_a_store(self, root):
        with pytest.raises(ReferenceStoreError, match="Top-K"):
            _store(root, _items(), support=FULL_VOCAB_V1)

    def test_token_weights_carry_the_prediction_they_feed(self, root):
        items = _items()
        store = _store(root, items)
        w = store.token_confidence(items[0])
        n = int(items[0]["input_ids"].shape[1])
        assert w.shape == (n,) and float(w[-1]) == 0.0
        assert torch.allclose(w[:-1], store.per_position(items[0],
                                                         "confidence").double())

    def test_the_report_states_weight_sum_mean_and_ess(self, root):
        store = _store(root, _items())
        rep = store.confidence_report()
        c = torch.cat([e.confidence.double() for e in store.entries.values()])
        assert rep["weight_sum"] == pytest.approx(float(c.sum()))
        assert rep["mean_weight"] == pytest.approx(float(c.mean()))
        assert rep["effective_sample_size"] == pytest.approx(
            float(c.sum() ** 2 / (c * c).sum()))


# --- 4. DEPTH: the root is the reference, and the root's confidence weighs -----

def _depth_ctx(model, store, *, objective=FULL, batch_size=1, rows="all_v1"):
    items = _items()
    support = reference_topk_tail(K)
    config = {"n_calibration_items": len(items),
              **policy_config(SUPERVISED_TARGET_V1),
              "distribution_support": support.as_dict()}
    if objective is not None:
        config["scoring_objective"] = objective.as_dict()
    return OperatorContext(
        adapter=get_adapter("qwen3"), model=model,
        parent_spec=ArchSpec.of("qwen3", GEOMETRY),
        target_spec=ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": 4}),
        profile=None, calibration_items=items, seed=0, device="cpu",
        config=config, distribution_support=support,
        execution=ExecutionConfig(
            micro_batch_size=batch_size,
            calibration_batch_packing=("length_sorted_v1" if batch_size > 1
                                       else "original_order_v1"),
            prediction_rows=rows),
        position_policy=SUPERVISED_TARGET_V1, scoring_objective=objective,
        root_reference=store), items


def _depth(model, store, **kw):
    ctx, items = _depth_ctx(model, store, **kw)
    return causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx), items


def _independent(reference, model, items, skip):
    """The objective for one subset from first principles: the REFERENCE
    model's sketch and full-vocabulary confidence, the bypassed model's
    logits, the fixed 1/T_x, then the sub-type and domain rule."""
    per_subtype: dict[str, list[float]] = {}
    for item in items:
        ids = item["input_ids"]
        with torch.no_grad():
            ref = reference(ids).logits[0, :-1].float()
            with bypassed_blocks(model, set(skip)):
                cand = model(ids).logits[0, :-1].float()
        sk = sketch_reference(ref, ids[0, 1:], top_k=K)
        kl = sketch_forward_kl(sk.support_indices, sk.support_log_probs,
                               sk.tail_log_prob, cand, has_tail=True)
        h = torch.tensor([_brute_entropy(r.numpy()) for r in ref])
        c = 1 - h / math.log(GEOMETRY["vocab_size"])
        scored = SUPERVISED_TARGET_V1.weights(
            item, axis="prediction").weights > 0
        value = float((c[scored] * kl[scored].double()).sum()) / int(scored.sum())
        per_subtype.setdefault(item["subtype"], []).append(value)
    domains: dict[str, set] = {}
    for item in items:
        domains.setdefault(item["domain"], set()).add(item["subtype"])
    return domain_balanced_score(
        {k: sum(v) / len(v) for k, v in per_subtype.items()},
        {d: sorted(s) for d, s in domains.items()})[0]


class TestDepthMeasuresAgainstTheRoot:

    def test_every_score_is_the_independent_root_objective(self, root, parent):
        store = _store(root, _items())
        out, items = _depth(parent, store)
        for entry in out.artifacts["search_rounds"][0]["table"][:3]:
            want = _independent(root, parent, items, {entry["candidate"]})
            assert entry["score"] == pytest.approx(want, rel=1e-4, abs=1e-6)

    def test_at_the_root_the_root_kl_is_d1s_parent_kl(self, root):
        """The free check: where the parent IS the root, the unweighted root
        KL of every round-0 candidate equals D1's KL-only DEPTH score."""
        store = _store(root, _items())
        d1, _ = _depth(root, None, objective=None)
        d2, _ = _depth(root, store)
        d1_round0 = {e["candidate"]: e["score"]
                     for e in d1.artifacts["search_rounds"][0]["table"]}
        cands = d2.artifacts["objective_candidates"]
        for layer, score in d1_round0.items():
            rec = cands[str(layer)]
            assert rec["raw_kl"] == pytest.approx(score, rel=1e-5, abs=1e-7)
            assert rec["parent_kl"] == pytest.approx(score, rel=1e-5, abs=1e-7)

    def test_below_the_root_the_two_references_differ(self, root, parent):
        store = _store(root, _items())
        d1, _ = _depth(parent, None, objective=None)
        d2, _ = _depth(parent, store)
        d1_round0 = {e["candidate"]: e["score"]
                     for e in d1.artifacts["search_rounds"][0]["table"]}
        cands = d2.artifacts["objective_candidates"]
        differ = 0
        for layer, score in d1_round0.items():
            rec = cands[str(layer)]
            #: The diagnostic IS D1's parent-reference KL ...
            assert rec["parent_kl"] == pytest.approx(score, rel=1e-5, abs=1e-7)
            #: ... and the decision's raw KL is against the root instead.
            differ += abs(rec["raw_kl"] - score) > 1e-4
        assert differ == len(d1_round0)
        assert d2.local_metrics.reference == "root_reference"

    def test_the_parent_kl_is_recorded_in_round_zero_only(self, root, parent):
        store = _store(root, _items())
        out, _ = _depth(parent, store)
        with_parent = {k for k, v in out.artifacts["objective_candidates"].items()
                       if "parent_kl" in v}
        round0 = {str(e["candidate"])
                  for e in out.artifacts["search_rounds"][0]["table"]}
        assert with_parent == round0
        assert out.artifacts["parent_kl_diagnostic"] == "round 0 candidates only"

    def test_the_weight_is_the_same_for_every_candidate(self, root, parent):
        store = _store(root, _items())
        out, _ = _depth(parent, store)
        confs = {round(v["reference_confidence"], 12)
                 for v in out.artifacts["objective_candidates"].values()}
        assert len(confs) == 1

    def test_records_carry_every_component(self, root, parent):
        store = _store(root, _items())
        out, _ = _depth(parent, store)
        rec = next(iter(out.artifacts["objective_candidates"].values()))
        for name in ("objective", "raw_kl", "reference_entropy",
                     "reference_confidence", "component",
                     "reference_tail_mass", "confidence_histogram"):
            assert name in rec
        assert "student_entropy" not in rec

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_scored_rows_score_the_objective_identically(self, root, parent,
                                                         batch_size):
        store = _store(root, _items(), bsz=batch_size)
        base, _ = _depth(parent, store, batch_size=batch_size)
        rows, _ = _depth(parent, store, batch_size=batch_size,
                         rows="scored_v1")
        assert rows.trace["kept_layers"] == base.trace["kept_layers"]
        for ra, rb in zip(base.artifacts["search_rounds"],
                          rows.artifacts["search_rounds"]):
            for ea, eb in zip(ra["table"], rb["table"]):
                assert eb["score"] == pytest.approx(ea["score"], rel=1e-5,
                                                    abs=1e-8)

    def test_batch_one_and_three_agree(self, root, parent):
        one, _ = _depth(parent, _store(root, _items()), batch_size=1)
        three, _ = _depth(parent, _store(root, _items(), bsz=3), batch_size=3)
        for ra, rb in zip(one.artifacts["search_rounds"],
                          three.artifacts["search_rounds"]):
            for ea, eb in zip(ra["table"], rb["table"]):
                assert eb["score"] == pytest.approx(ea["score"], rel=1e-4,
                                                    abs=1e-6)

    def test_no_store_is_a_contract_violation(self, parent):
        ctx, _ = _depth_ctx(parent, None)
        with pytest.raises(ContractViolation, match="ROOT"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)

    def test_a_store_of_another_mixture_is_refused(self, root, parent):
        store = _store(root, _items()[:2])
        ctx, _ = _depth_ctx(parent, store)
        with pytest.raises(ReferenceStoreError):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)


# --- 5. the state evaluator ------------------------------------------------------

def _suite():
    items = []
    for k, raw in enumerate(make_items(seed=202)):
        keep = 10 + 4 * k
        ids = raw["input_ids"][:, :keep]
        n = keep - 1
        tags = {name: mask[:n] for name, mask in raw["tags"].items()}
        tags["eos_like"] = tags["eos_like"] | (torch.arange(n) == n - 1)
        tags["assistant"] = torch.arange(n) >= 4
        items.append(SuiteItem(item_id=raw["item_id"], input_ids=ids,
                               domain=raw["domain"], subtype=raw["subtype"],
                               tags=tags))
    suite = StateEvalSuite(
        suite_id="test.refconf", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(items),
        general_domain="general")
    return suite, items


def _evaluate(teacher, candidate, *, objective=FULL, rows="all_v1", batch=1):
    suite, items = _suite()
    ev = StateEvaluator(
        suite, items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
        execution=ExecutionConfig(
            micro_batch_size=batch,
            calibration_batch_packing=("length_sorted_v1" if batch > 1
                                       else "original_order_v1"),
            prediction_rows=rows),
        distribution_support=reference_topk_tail(K), numerics=CPU,
        suite_content_sha256="toy", scoring_objective=objective)
    ev.prime_reference(teacher)
    return ev, ev.evaluate(candidate, "candidate")


class TestStateEvaluation:

    def test_each_item_is_the_independent_weighted_mean(self, root, parent):
        _, ev = _evaluate(root, parent)
        _, items = _suite()
        per_item = ev.detail["objective"]["per_item"]
        for item in items:
            with torch.no_grad():
                ref = root(item.input_ids).logits[0, :-1].float()
                cand = parent(item.input_ids).logits[0, :-1].float()
            sk = sketch_reference(ref, item.input_ids[0, 1:], top_k=K)
            kl = sketch_forward_kl(sk.support_indices, sk.support_log_probs,
                                   sk.tail_log_prob, cand, has_tail=True)
            h = torch.tensor([_brute_entropy(r.numpy()) for r in ref])
            c = 1 - h / math.log(GEOMETRY["vocab_size"])
            scored = item.tags["assistant"]
            want = float((c[scored] * kl[scored].double()).sum()) / int(
                scored.sum())
            assert per_item[item.item_id]["objective"] == pytest.approx(
                want, rel=1e-4, abs=1e-6)
            assert per_item[item.item_id]["scored_positions"] == int(
                scored.sum())

    def test_raw_kl_reproduces_the_kl_metrics(self, root, parent):
        _, ev = _evaluate(root, parent)
        v = ev.values
        assert v["state.objective.raw_kl.equal_domain_mean"] == pytest.approx(
            v["state.teacher_kl.equal_domain_mean"], rel=1e-6)
        assert v["state.objective.raw_kl.critical_token"] == pytest.approx(
            v["state.critical_token_kl"], rel=1e-6)

    def test_the_weight_is_the_teachers_not_the_candidates(self, root, parent):
        _, a = _evaluate(root, parent)
        _, b = _evaluate(root, build_tiny_model(GEOMETRY, seed=23))
        assert a.values["state.objective.reference_confidence.equal_domain_mean"] \
            == pytest.approx(
                b.values["state.objective.reference_confidence.equal_domain_mean"],
                abs=1e-9)
        assert a.values["state.objective.raw_kl.equal_domain_mean"] != \
            pytest.approx(b.values["state.objective.raw_kl.equal_domain_mean"],
                          abs=1e-6)

    @pytest.mark.parametrize("batch", [1, 3])
    def test_scored_rows_measure_the_same_objective(self, root, parent, batch):
        _, base = _evaluate(root, parent, batch=batch)
        _, rows = _evaluate(root, parent, batch=batch, rows="scored_v1")
        for key, value in base.values.items():
            if key.startswith("state.objective."):
                assert rows.values[key] == pytest.approx(value, rel=1e-5,
                                                         abs=1e-8), key

    def test_batch_one_and_three_agree(self, root, parent):
        _, one = _evaluate(root, parent, batch=1)
        _, three = _evaluate(root, parent, batch=3)
        for key, value in one.values.items():
            if key.startswith("state.objective."):
                assert three.values[key] == pytest.approx(value, rel=1e-4,
                                                          abs=1e-7), key

    def test_histograms_count_every_scored_position(self, root, parent):
        _, ev = _evaluate(root, parent)
        joint = ev.detail["objective"]["joint_histograms"]
        total = ev.detail["objective"]["scored_positions"]["total"]
        assert sum(map(sum, joint["confidence_x_kl"])) == total
        assert sum(ev.detail["objective"]["confidence_histogram"]) == total


# --- 6. weighted activation statistics -----------------------------------------

def _collect(model, items, *, weights=None, batch=1):
    if batch <= 1:
        col = ActivationStatsCollector(model)
        try:
            for item, w in zip(items, weights or [None] * len(items)):
                kw = {} if w is None else {"token_weights": w}
                col.process(item["input_ids"], **kw)
        finally:
            col.close()
        return col.state()
    col = ActivationStatsCollector(model)
    try:
        for batch_ in micro_batches(items, batch, pad_id=0):
            kw = {}
            if weights is not None:
                wt = torch.zeros(batch_.input_ids.shape, dtype=torch.float64)
                for r, it in enumerate(batch_.items):
                    idx = next(i for i, x in enumerate(items) if x is it)
                    wt[r, :weights[idx].numel()] = weights[idx]
                kw = {"token_weights": wt}
            col.process_batch(batch_, **kw)
    finally:
        col.close()
    return col.state()


def _hidden(model, ids):
    with torch.no_grad():
        return [h[0].double() for h in model(ids, output_hidden_states=True)
                .hidden_states]


class TestWeightedStatistics:

    def test_every_accumulator_is_the_weighted_sum(self, root):
        items = _items()
        g = torch.Generator().manual_seed(9)
        weights = [torch.rand(int(i["input_ids"].shape[1]), generator=g,
                              dtype=torch.float64) for i in items]
        weights[0][2] = 0.0
        state = _collect(root, items, weights=weights)
        want_sum = torch.zeros_like(state["residual_sum"])
        want_sq = torch.zeros_like(state["residual_sqsum"])
        for item, w in zip(items, weights):
            for p, h in enumerate(_hidden(root, item["input_ids"])):
                want_sum[p] += (w[:, None] * h).sum(0)
                want_sq[p] += (h * w[:, None]).T @ h
        assert torch.allclose(state["residual_sum"], want_sum, rtol=1e-9,
                              atol=1e-9)
        assert torch.allclose(state["residual_sqsum"], want_sq, rtol=1e-9,
                              atol=1e-8)
        total = float(sum(float(w.sum()) for w in weights))
        assert float(state[RESIDUAL_WEIGHT][0]) == pytest.approx(total)
        assert statistics_denominator(state) == pytest.approx(total)
        #: The integer count and the histogram count positions WITH weight.
        assert int(state["residual_count"][0]) == sum(
            int((w > 0).sum()) for w in weights)
        assert int(state["token_counts"].sum()) == int(state["residual_count"][0])

    def test_the_ffn_magnitudes_are_weighted(self, root):
        items = _items()[:1]
        w = torch.linspace(0.1, 1.0, int(items[0]["input_ids"].shape[1]),
                           dtype=torch.float64)
        captured = []
        adapter = adapter_for_config(root.config)
        proj = adapter.stream_out_projections(adapter.blocks(root)[1])["ffn_out"]
        hook = proj.register_forward_pre_hook(
            lambda _m, a: captured.append(a[0].detach()[0].double()))
        try:
            with torch.no_grad():
                root(items[0]["input_ids"])
        finally:
            hook.remove()
        a = captured[0]
        state = _collect(root, items, weights=[w])
        assert torch.allclose(state["ffn_abs_sum"][1], w @ a.abs(), atol=1e-9)
        assert torch.allclose(state["ffn_sq_sum"][1], w @ (a * a), atol=1e-9)

    def test_padding_and_batching_do_not_move_a_weighted_statistic(self, root):
        items = _items()
        weights = [torch.rand(int(i["input_ids"].shape[1]),
                              generator=torch.Generator().manual_seed(k),
                              dtype=torch.float64)
                   for k, i in enumerate(items)]
        one = _collect(root, items, weights=weights, batch=1)
        three = _collect(root, items, weights=weights, batch=3)
        for key in ("residual_sum", "residual_sqsum", "ffn_abs_sum",
                    "ffn_sq_sum", RESIDUAL_WEIGHT):
            assert torch.allclose(one[key], three[key], rtol=1e-6,
                                  atol=1e-6), key
        assert int(one["residual_count"][0]) == int(three["residual_count"][0])

    def test_unit_weights_reproduce_the_unweighted_moments(self, root):
        items = _items()
        plain = _collect(root, items)
        ones = _collect(root, items, weights=[
            torch.ones(int(i["input_ids"].shape[1]), dtype=torch.float64)
            for i in items])
        assert RESIDUAL_WEIGHT not in plain
        for key in ("residual_sum", "residual_sqsum", "ffn_abs_sum",
                    "token_counts", "residual_count"):
            assert torch.allclose(plain[key].double(), ones[key].double(),
                                  rtol=1e-9, atol=1e-8), key
        assert statistics_denominator(ones) == pytest.approx(
            statistics_denominator(plain))
        assert torch.allclose(uncentered_moment(ones, 1),
                              uncentered_moment(plain, 1), rtol=1e-9)
        mean_w, cov_w = residual_covariance(ones, 1)
        mean_p, cov_p = residual_covariance(plain, 1)
        assert torch.allclose(mean_w, mean_p) and torch.allclose(cov_w, cov_p)

    def test_the_denominators_divide_by_the_weight_sum(self):
        state = {"residual_count": torch.tensor([4]),
                 RESIDUAL_WEIGHT: torch.tensor([2.0], dtype=torch.float64),
                 "residual_sqsum": torch.ones(2, 3, 3, dtype=torch.float64),
                 "ffn_abs_sum": torch.ones(1, 5, dtype=torch.float64)}
        assert torch.allclose(uncentered_moment(state, 0),
                              torch.full((3, 3), 0.5, dtype=torch.float64))
        imp = ffn_neuron_importance(state, 0, torch.ones(3, 5))
        assert torch.allclose(imp, torch.full((5,), 0.5 * math.sqrt(3),
                                              dtype=torch.float64))

    def test_the_attention_second_moment_is_weighted(self, root):
        items = _items()[:2]
        adapter = adapter_for_config(root.config)
        from aadistill.initialization.operators.attention.gqa._common import (
            attention_out_projection,
        )
        outs = [attention_out_projection(adapter, b)
                for b in adapter.blocks(root)]
        weights = [torch.rand(int(i["input_ids"].shape[1]),
                              generator=torch.Generator().manual_seed(3),
                              dtype=torch.float64) for i in items]
        captured: list[torch.Tensor] = []
        hook = outs[0].register_forward_pre_hook(
            lambda _m, a: captured.append(a[0].detach()[0].double()))
        col = AttentionHeadStatsCollector(
            root, outs, num_heads=GEOMETRY["num_attention_heads"],
            head_dim=GEOMETRY["head_dim"])
        try:
            for item, w in zip(items, weights):
                col.process(item["input_ids"], token_weights=w)
        finally:
            col.close()
            hook.remove()
        state = col.state()
        want = torch.zeros_like(state["attn_head_sqsum"][0])
        for a, w in zip(captured, weights):
            a = a.reshape(-1, GEOMETRY["num_attention_heads"],
                          GEOMETRY["head_dim"])
            want += torch.einsum("t,thi,thj->hij", w, a, a)
        assert torch.allclose(state["attn_head_sqsum"][0], want, atol=1e-9)
        assert float(state[ATTN_TOKEN_WEIGHT]) == pytest.approx(
            float(sum(w.sum() for w in weights)))

    def test_mixing_and_malformed_weights_are_refused(self, root):
        items = _items()
        col = ActivationStatsCollector(root)
        try:
            n = int(items[0]["input_ids"].shape[1])
            col.process(items[0]["input_ids"],
                        token_weights=torch.ones(n, dtype=torch.float64))
            with pytest.raises(ValueError, match="count and a weight sum"):
                col.process(items[1]["input_ids"])
            with pytest.raises(ValueError, match="not both"):
                col.process(items[0]["input_ids"],
                            active_mask=torch.ones(n, dtype=torch.bool),
                            token_weights=torch.ones(n, dtype=torch.float64))
            with pytest.raises(BatchingError, match="non-negative"):
                col.process(items[0]["input_ids"],
                            token_weights=-torch.ones(n, dtype=torch.float64))
        finally:
            col.close()

    def test_statistics_positions_align_the_confidence_to_its_token(self, root):
        items = _items()
        store = _store(root, items)
        ctx = OperatorContext(
            adapter=get_adapter("qwen3"), model=root,
            parent_spec=ArchSpec.of("qwen3", GEOMETRY),
            target_spec=ArchSpec.of("qwen3", GEOMETRY), profile=None,
            calibration_items=items, seed=0, device="cpu", config={},
            position_policy=SUPERVISED_TARGET_V1, scoring_objective=FULL,
            root_reference=store)
        ffn = get_implementation("ffn.activation_importance_v0")
        active = statistics_positions(ctx, ffn)
        assert active.token_weighted
        for k, item in enumerate(items):
            policy_w = SUPERVISED_TARGET_V1.weights(item, axis="token").weights
            assert torch.allclose(active.token_weights_for_item(k),
                                  policy_w * store.token_confidence(item))
        assert active.report()["token_weights_multiplied_by"].endswith(
            store.identity_hash)
        #: Narrow scope and the student forms leave the policy alone.
        for objective in (NARROW, STUDENT):
            ctx.scoring_objective = objective
            assert not statistics_positions(ctx, ffn).token_weighted
        ctx.scoring_objective, ctx.root_reference = FULL, None
        with pytest.raises(SurgeryError, match="root reference"):
            statistics_positions(ctx, ffn)

    @pytest.mark.parametrize("batch", [1, 3])
    def test_the_operator_pass_equals_a_direct_weighted_pass(self, root, batch):
        items = _items()
        store = _store(root, items)
        active = ActivePositions(
            items, SUPERVISED_TARGET_V1,
            token_multipliers=[store.token_confidence(i) for i in items],
            multiplier_source="test")
        got = collect_activation_stats(
            get_adapter("qwen3"), root, items, "cpu", batch_size=batch,
            packing=("length_sorted_v1" if batch > 1 else "original_order_v1"),
            active=active)
        direct = _collect(root, items, weights=[
            active.token_weights_for_item(k) for k in range(len(items))])
        for key in ("residual_sqsum", "ffn_abs_sum", RESIDUAL_WEIGHT):
            assert torch.allclose(got[key], direct[key], rtol=1e-6,
                                  atol=1e-6), key


# --- 7. the search and the replay ------------------------------------------------

OBJECTIVE_PARETO = BeamRankingPolicy(
    policy_id="beam.toy_refconf_pareto", version=1,
    description="toy ranking on the objective metrics",
    objectives=(
        Objective("state.objective.equal_domain_mean", "minimize", "mean"),
        Objective("state.objective.worst_domain", "minimize", "worst"),
        Objective("state.objective.critical_token", "minimize", "critical"),
    ),
    tie_break=("state.objective.equal_domain_mean",
               "state.objective.worst_domain",
               "state.objective.critical_token", STATE_ID_TIEBREAK),
    guardrails=(VALIDITY_GUARDRAIL,),
    epsilon={"state.objective.equal_domain_mean": 1e-4,
             "state.objective.worst_domain": 1e-4,
             "state.objective.critical_token": 1e-4},
    diversity_key="parent_path")


class _Measurer:
    def __init__(self, ev):
        self.ev = ev
        self.scoring_objective = ev.scoring_objective
        self.distribution_support = ev.distribution_support

    def __call__(self, model, digest):
        return self.ev.evaluate(model, digest)


def _search(root, *, objective=FULL, support=None, allowed=(
        "depth.causal_kl_greedy_v1", "ffn.activation_importance_v0"),
        measurer=None):
    suite, items = _suite()
    calib = _items()
    support = support or reference_topk_tail(K)
    if measurer is None:
        ev = StateEvaluator(suite, items, device="cpu",
                            position_policy=SUPERVISED_TARGET_V1,
                            distribution_support=support,
                            scoring_objective=objective)
        ev.prime_reference(root)
        measurer = _Measurer(ev)
    config = SearchConfig(
        run_id="refconf", target_spec=ArchSpec.of(
            "qwen3", {**GEOMETRY, "num_hidden_layers": 4,
                      "intermediate_size": 24}),
        schedule=SCHEDULE_V1, seed=7, workdir=Path(tempfile.mkdtemp()),
        profiles=(make_profile("balanced"),), policy=OBJECTIVE_PARETO,
        suite=suite, position_policy=SUPERVISED_TARGET_V1,
        distribution_support=support, scoring_objective=objective,
        allowed_impls=allowed, max_depth=2, device="cpu")
    return BeamSearch(
        adapter=adapter_for_config(root.config), config=config,
        root_teacher_id="toy/teacher", root_teacher_sha256="de" * 32,
        root_loader=lambda: root, calibration_loader=lambda profile: calib,
        measurer=measurer, execution=ExecutionConfig(), numerics=CPU)


class TestTheSearch:

    def test_a_root_objective_needs_a_topk_support(self, root):
        #: A bare callable measurer, so the measurer-agreement checks (which
        #: would also refuse) are skipped and THIS refusal is what is tested.
        with pytest.raises(SearchError, match="Top-K support"):
            _search(root, support=FULL_VOCAB_V1,
                    measurer=lambda model, digest: None)

    def test_one_store_per_profile_serves_every_expansion(self, root):
        search = _search(root)
        result = search.run()
        assert result.states
        built = [r for r in search.telemetry._events
                 if r["event"] == "root_reference_built"]
        assert len(built) == 1
        evidence = search.root_reference_evidence()
        (only,) = evidence.values()
        assert only["identity"]["root"] == {"teacher_id": "toy/teacher",
                                            "teacher_sha256": "de" * 32}
        assert only["confidence"]["positions"] > 0
        depth = [r for r in search.telemetry._events
                 if r.get("impl_id") == "depth.causal_kl_greedy_v1"]
        assert depth and all(r["reference_cache"]["mode"]
                             == "root_reference_store" for r in depth)
        assert all("parent_reference_cache" in r for r in depth)

    def test_weighted_and_unweighted_passes_never_share_a_cache_key(self, root):
        from types import SimpleNamespace

        search = _search(root)
        #: The key reads only the parent's digest.
        parent = SimpleNamespace(artifact_digest="f" * 64)
        profile = make_profile("balanced")
        ffn = get_implementation("ffn.activation_importance_v0")
        width = get_implementation("width.global_pca_v0")
        depth = get_implementation("depth.causal_kl_greedy_v1")
        weighted = search._stats_key(parent, profile, ffn)
        assert weighted == search._stats_key(parent, profile, width)
        assert weighted != search._stats_key(parent, profile, depth)
        assert weighted != search._stats_key(parent, profile)
        narrow = _search(root, objective=NARROW)
        assert narrow._stats_key(parent, profile, ffn) == \
            narrow._stats_key(parent, profile)


class TestTheFixedPathReplay:

    PROFILE_ID = "test.balanced@v1"

    @pytest.fixture
    def registered_profile(self):
        from aadistill.initialization.calibration.profiles import (
            register_profile, unregister_profile,
        )
        p = make_profile("balanced")
        register_profile(p, replace=True)
        yield p
        unregister_profile(p.profile_id)

    def _spec(self):
        from aadistill.initialization.planning.fixed_path import (
            FixedPathSpec, FixedPathStep,
        )
        return FixedPathSpec(
            path_id="test.refconf", family="qwen3",
            target_spec=ArchSpec.of("qwen3", {**GEOMETRY,
                                              "num_hidden_layers": 4,
                                              "intermediate_size": 24}),
            steps=(FixedPathStep("depth.causal_kl_greedy_v1", self.PROFILE_ID),
                   FixedPathStep("ffn.activation_importance_v0",
                                 self.PROFILE_ID)),
            root_repo_id="test/teacher", root_revision="deadbeef",
            position_policy=SUPERVISED_TARGET_V1,
            distribution_support=reference_topk_tail(K),
            scoring_objective=FULL)

    def test_the_stores_are_built_before_the_root_is_released(
            self, registered_profile, tmp_path):
        from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
        from aadistill.initialization.planning.fixed_path import (
            materialize_fixed_path,
        )
        out = materialize_fixed_path(
            self._spec(), adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(GEOMETRY), workdir=tmp_path,
            calibration_items={self.PROFILE_ID: make_items()})
        assert [r.impl_id for r in out] == ["depth.causal_kl_greedy_v1",
                                            "ffn.activation_importance_v0"]
        #: The FFN step, which runs on a compressed parent after the root is
        #: gone, still weighted its statistics by the ROOT's confidence.
        weighting = out[1].trace["scoring_positions"][
            "token_weights_multiplied_by"]
        assert weighting.startswith("root_reference_confidence:")

    def test_a_replay_without_the_root_needs_the_stores(
            self, registered_profile, tmp_path):
        from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER
        from aadistill.initialization.planning.fixed_path import (
            FixedPathError, _run_steps,
        )
        spec = self._spec()
        model = build_tiny_model({**GEOMETRY, "num_hidden_layers": 4})
        with pytest.raises(FixedPathError, match="does not hold the root"):
            _run_steps(spec, adapter=QWEN3_ADAPTER, model=model, indices=[1],
                       parent_spec=QWEN3_ADAPTER.spec_of(model),
                       parent_digest="f" * 64, workdir=tmp_path,
                       repo_root=".", stats_cache=None,
                       calibration_items={self.PROFILE_ID: make_items()},
                       on_step=None, deadline=None)
