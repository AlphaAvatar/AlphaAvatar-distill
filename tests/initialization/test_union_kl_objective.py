"""The union-KL objective: reference Top-K confidence x union Top-K KL.

`FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL` on the
`reference_student_union_topk_tail_v1` support. Pinned here against
independent float64 computations and against the refusals that keep it from
being confused with its neighbours:

* the union KL and its bounds ``KL_{S_T+tail} <= KL_U <= KL_full`` against a
  brute-force small-vocabulary oracle, with dedup, both tails, ``+inf`` and
  deterministic ties;
* the ``K+1``-category reference confidence, its category count, and its
  independence from any candidate;
* exact retrieval of the reference's probability of candidate-only entries
  through the output head, never a tail approximation;
* every consumer -- DEPTH, the state evaluator, the beam's metrics and the
  weighted activation statistics -- reads the same objective and the same
  weight, and none falls back to the unweighted KL;
* the store's identity, the refusals of a store / support / objective that do
  not belong together, and the untouched historical identities.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from support.toy import (  # noqa: E402
    TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile,
)
from aadistill.initialization.calibration.items import (  # noqa: E402
    prepare_calibration_items,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators._common import (  # noqa: E402
    statistics_positions,
)
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorContext, OperatorError, get_implementation,
)
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.planning.ranking import (  # noqa: E402
    SCHEDULE_V1, STATE_ID_TIEBREAK, VALIDITY_GUARDRAIL, BeamRankingPolicy,
    Objective,
)
from aadistill.initialization.planning.search import (  # noqa: E402
    BeamSearch, SearchConfig, SearchError,
)
from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_KL_MINUS_CONFIDENCE_REWARD, FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
    FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL, KL_REFERENCE_ROOT,
    UNION_COMPONENTS, ObjectiveError, ScoringObjective, position_terms,
    reference_topk_confidence, require_objective_support,
)
from aadistill.initialization.scoring.output_head import (  # noqa: E402
    LinearOutputHead, OutputHeadError,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    SUPERVISED_TARGET_V1, policy_config,
)
from aadistill.initialization.scoring.reference_store import (  # noqa: E402
    SCHEMA, UNION_SCHEMA, ReferenceStoreError, build_root_reference_store,
    root_reference_identity,
)
from aadistill.initialization.scoring.support import (  # noqa: E402
    TOPK_TIE_BREAK, TeacherRetrieval, deterministic_topk,
    reference_student_union_topk_tail, reference_topk_tail,
    sketch_forward_kl_terms, sketch_reference, sketch_reference_deterministic,
    topk_tail_categories, topk_tail_entropy, union_forward_kl_terms,
)
from aadistill.initialization.specs.arch import (  # noqa: E402
    ArchSpec, Capability, adapter_for_config, get_adapter,
)
from aadistill.initialization.specs.materialization import (  # noqa: E402
    NumericalEnvironment,
)
from aadistill.initialization.specs.metrics import (  # noqa: E402
    StateEvalSuite, SuiteItem,
)
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    bypassed_blocks, domain_balanced_score,
)

K = 8
UNION = reference_student_union_topk_tail(K)
V3 = ScoringObjective("objective.toy_union", 1,
                      FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL,
                      entropy_top_k=K, kl_reference=KL_REFERENCE_ROOT,
                      weights_activation_statistics=True)
V2 = ScoringObjective("objective.toy_reference_confidence", 1,
                      FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
                      kl_reference=KL_REFERENCE_ROOT,
                      weights_activation_statistics=True)
D3_LIKE = ScoringObjective("objective.toy_reward", 2,
                           FORM_KL_MINUS_CONFIDENCE_REWARD, entropy_top_k=K,
                           reward_scale=7.34)
CPU = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
GEOMETRY = dict(TEACHER_GEOMETRY)
ROOT_ID = {"teacher_id": "toy/root", "teacher_sha256": "ab" * 32}


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
    m = build_tiny_model(GEOMETRY, seed=19)
    m.config.use_cache = False
    return m


# --- the float64 oracle ---------------------------------------------------------

def _log_softmax64(x):
    return torch.log_softmax(x.double(), dim=-1)


def _partition_kl(p_log, q_log, members):
    """Forward KL of one row on `members` (singletons) + one bucket for the
    rest, in float64 by brute force. +inf where the bucket needs it."""
    v = int(p_log.shape[-1])
    mask = torch.zeros(v, dtype=torch.bool)
    mask[sorted(members)] = True
    p, q = p_log.exp(), q_log.exp()
    on = p[mask] > 0
    kl = (p[mask][on] * (p_log[mask][on] - q_log[mask][on])).sum().item()
    pr, qr = p[~mask].sum().item(), q[~mask].sum().item()
    if pr > 0:
        kl += math.inf if qr == 0 else pr * (math.log(pr) - math.log(qr))
    return kl


def _topk_set(row, k):
    """The deterministic Top-k: value descending, index ascending."""
    order = sorted(range(int(row.shape[0])), key=lambda i: (-float(row[i]), i))
    return set(order[:k])


def _head_and_logits(seed, v=60, h=10, t=30, k_scale=2.0):
    g = torch.Generator().manual_seed(seed)
    head = LinearOutputHead(weight=torch.randn(v, h, generator=g))
    hidden = torch.randn(t, h, generator=g) * k_scale
    return head, hidden, head.logits(hidden)


def _union_terms(head, hidden, ref_logits, cand, k=K, **kw):
    sk, log_z = sketch_reference_deterministic(
        ref_logits, torch.zeros(int(ref_logits.shape[0]), dtype=torch.long),
        top_k=k)
    terms = union_forward_kl_terms(
        sk.support_indices, sk.support_log_probs, sk.tail_log_prob, cand,
        TeacherRetrieval(hidden, log_z, head), top_k=k,
        has_tail=k < int(ref_logits.shape[-1]), **kw)
    return sk, log_z, terms


# --- 1. the union KL against the oracle ----------------------------------------

class TestUnionKLOracle:

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("noise", [0.3, 2.0, 6.0])
    def test_union_kl_and_its_lower_bound_match_the_oracle(self, seed, noise):
        head, hidden, ref = _head_and_logits(seed)
        g = torch.Generator().manual_seed(100 + seed)
        cand = ref + noise * torch.randn(ref.shape, generator=g)
        _, _, terms = _union_terms(head, hidden, ref, cand, reverse=True)
        p, q = _log_softmax64(ref), _log_softmax64(cand)
        for t in range(int(ref.shape[0])):
            s_t, s_c = _topk_set(ref[t], K), _topk_set(cand[t], K)
            u = s_t | s_c
            assert float(terms.kl[t]) == pytest.approx(
                _partition_kl(p[t], q[t], u), rel=2e-5, abs=2e-6)
            assert float(terms.kl_reference_topk[t]) == pytest.approx(
                _partition_kl(p[t], q[t], s_t), rel=2e-5, abs=2e-6)
            assert float(terms.reverse_kl[t]) == pytest.approx(
                _partition_kl(q[t], p[t], u), rel=2e-5, abs=2e-6)
            assert int(terms.union_size[t]) == len(u)
            assert int(terms.overlap[t]) == len(s_t & s_c)
            out = sorted(set(range(int(ref.shape[1]))) - u)
            assert float(terms.reference_union_tail_log[t].exp()) == \
                pytest.approx(float(p[t][out].exp().sum()), rel=1e-5,
                              abs=1e-9)
            assert float(terms.candidate_union_tail_log[t].exp()) == \
                pytest.approx(float(q[t][out].exp().sum()), rel=1e-5,
                              abs=1e-9)

    @pytest.mark.parametrize("seed", range(6))
    def test_the_bounds_hold_on_every_position(self, seed):
        head, hidden, ref = _head_and_logits(seed, v=200, t=40, k_scale=3.0)
        g = torch.Generator().manual_seed(seed)
        cand = ref.flip(-1) * 0.5 + torch.randn(ref.shape, generator=g)
        _, _, terms = _union_terms(head, hidden, ref, cand)
        full = (_log_softmax64(ref).exp()
                * (_log_softmax64(ref) - _log_softmax64(cand))).sum(-1)
        slack = 1e-5 * (1 + full.abs())
        assert bool((terms.kl_reference_topk.double()
                     <= terms.kl.double() + slack).all())
        assert bool((terms.kl.double() <= full + slack).all())

    def test_k_at_or_above_the_vocabulary_is_the_full_kl(self):
        head, hidden, ref = _head_and_logits(4, v=12)
        cand = ref + torch.randn(ref.shape)
        _, _, terms = _union_terms(head, hidden, ref, cand, k=12)
        full = (_log_softmax64(ref).exp()
                * (_log_softmax64(ref) - _log_softmax64(cand))).sum(-1)
        #: float32's own floor on a 12-term sum, and exactly the historical
        #: reducer's: at K >= V the two supports are the same partition.
        assert torch.allclose(terms.kl.double(), full, rtol=1e-5, atol=5e-6)
        sk = sketch_reference(ref, torch.zeros(int(ref.shape[0]),
                                               dtype=torch.long), top_k=12)
        hist, _ = sketch_forward_kl_terms(sk.support_indices,
                                          sk.support_log_probs,
                                          sk.tail_log_prob, cand,
                                          has_tail=False)
        assert torch.allclose(terms.kl, hist, rtol=0, atol=1e-6)
        assert bool((terms.union_size == 12).all())

    def test_a_candidate_inside_the_reference_topk_is_the_k_plus_1_kl(self):
        """Dedup: S_C ⊆ S_T leaves U = S_T, so the two KLs coincide."""
        head, hidden, ref = _head_and_logits(5)
        cand = ref * 1.5          # same order, so the same Top-K
        _, _, terms = _union_terms(head, hidden, ref, cand)
        assert bool((terms.union_size == K).all())
        assert bool((terms.overlap == K).all())
        assert torch.allclose(terms.kl, terms.kl_reference_topk, atol=1e-6)

    def test_a_disjoint_candidate_doubles_the_union(self):
        head, hidden, ref = _head_and_logits(6)
        cand = -ref               # the reference's bottom is its top
        _, _, terms = _union_terms(head, hidden, ref, cand)
        assert bool((terms.union_size == 2 * K).all())
        assert bool((terms.overlap == 0).all())

    def test_zero_candidate_mass_on_reference_mass_is_infinite(self):
        head, hidden, ref = _head_and_logits(7)
        cand = ref.clone()
        top = ref[0].argmax()
        cand[0, top] = float("-inf")
        _, _, terms = _union_terms(head, hidden, ref, cand)
        assert math.isinf(float(terms.kl[0])) and float(terms.kl[0]) > 0
        assert bool(torch.isfinite(terms.kl[1:]).all())
        g, _ = V3.integrand(terms.kl, torch.ones_like(terms.kl))
        assert math.isinf(float(g[0]))

    def test_zero_reference_mass_contributes_nothing(self):
        head, hidden, ref = _head_and_logits(8)
        ref = ref.clone()
        ref[:, -5:] = float("-inf")          # the reference forbids 5 entries
        cand = ref.clone()
        cand[:, -5:] = 30.0                  # the candidate loves them
        sk, log_z = sketch_reference_deterministic(
            ref, torch.zeros(int(ref.shape[0]), dtype=torch.long), top_k=K)

        class _Fixed:
            def token_logits(self, h, ids):
                return ref[torch.arange(int(ids.shape[0]))[:, None],
                           ids.to(torch.long)]
        terms = union_forward_kl_terms(
            sk.support_indices, sk.support_log_probs, sk.tail_log_prob, cand,
            TeacherRetrieval(hidden, log_z, _Fixed()), top_k=K, has_tail=True)
        p, q = _log_softmax64(ref), _log_softmax64(cand)
        for t in range(int(ref.shape[0])):
            want = _partition_kl(p[t], q[t],
                                 _topk_set(ref[t], K) | _topk_set(cand[t], K))
            assert math.isfinite(float(terms.kl[t]))
            assert float(terms.kl[t]) == pytest.approx(want, rel=1e-4,
                                                       abs=1e-5)

    def test_nan_logits_never_become_a_finite_score(self):
        head, hidden, ref = _head_and_logits(9)
        cand = ref.clone()
        cand[2, 3] = float("nan")
        _, _, terms = _union_terms(head, hidden, ref, cand)
        assert math.isnan(float(terms.kl[2]))
        assert bool(torch.isfinite(terms.kl[:2]).all())

    @pytest.mark.parametrize("chunk", [1, 7, 512])
    def test_chunking_changes_nothing(self, chunk):
        head, hidden, ref = _head_and_logits(10)
        cand = ref + torch.randn(ref.shape,
                                 generator=torch.Generator().manual_seed(1))
        _, _, a = _union_terms(head, hidden, ref, cand, chunk=chunk)
        _, _, b = _union_terms(head, hidden, ref, cand, chunk=512)
        assert torch.allclose(a.kl, b.kl, rtol=0, atol=1e-6)

    def test_leading_dimensions_are_generic(self):
        head, hidden, ref = _head_and_logits(11, t=12)
        cand = ref + torch.randn(ref.shape,
                                 generator=torch.Generator().manual_seed(2))
        sk, log_z = sketch_reference_deterministic(
            ref, torch.zeros(12, dtype=torch.long), top_k=K)
        flat = union_forward_kl_terms(
            sk.support_indices, sk.support_log_probs, sk.tail_log_prob, cand,
            TeacherRetrieval(hidden, log_z, head), top_k=K, has_tail=True)
        blocked = union_forward_kl_terms(
            sk.support_indices.view(3, 4, K), sk.support_log_probs.view(3, 4, K),
            sk.tail_log_prob.view(3, 4), cand.view(3, 4, -1),
            TeacherRetrieval(hidden, log_z, head), top_k=K, has_tail=True)
        assert blocked.kl.shape == (3, 4)
        assert torch.allclose(blocked.kl.reshape(-1), flat.kl, atol=1e-7)

    def test_float32_against_float64_at_a_realistic_k(self):
        """V=4096, K=200: the float32 reduction against the float64 oracle on
        identical logits, with Top-K masses within 1e-10 of one.

        The float32 floor is ABSOLUTE: every log probability carries the
        rounding of its row's log-normalizer (about one ulp of |log Z|, ~4e-6
        at these logits), and that shift does not cancel between two models.
        It is the historical reducer's floor too, which the second assertion
        pins: the union path adds no error of its own.
        """
        g = torch.Generator().manual_seed(12)
        head = LinearOutputHead(weight=torch.randn(4096, 16, generator=g))
        hidden = torch.randn(24, 16, generator=g) * 4
        ref = head.logits(hidden)
        cand = ref + torch.randn(ref.shape, generator=g) * 0.7
        sk, _, terms = _union_terms(head, hidden, ref, cand, k=200)
        p, q = _log_softmax64(ref), _log_softmax64(cand)
        floor = 4 * torch.finfo(torch.float32).eps * float(
            torch.logsumexp(ref, -1).abs().max())
        for t in range(24):
            want = _partition_kl(p[t], q[t], _topk_set(ref[t], 200)
                                 | _topk_set(cand[t], 200))
            assert abs(float(terms.kl[t]) - want) <= floor + 1e-5 * want
        hist, _ = sketch_forward_kl_terms(sk.support_indices,
                                          sk.support_log_probs,
                                          sk.tail_log_prob, cand,
                                          has_tail=True)
        assert torch.allclose(terms.kl_reference_topk, hist, rtol=1e-6,
                              atol=1e-7)


# --- 2. deterministic Top-K -----------------------------------------------------

class TestDeterministicTopK:

    def test_boundary_ties_admit_the_lowest_indices(self):
        x = torch.tensor([[1.0, 3.0, 3.0, 3.0, 2.0, 3.0, 0.0]])
        assert deterministic_topk(x, 2)[1].tolist() == [[1, 2]]
        assert deterministic_topk(x, 4)[1].tolist() == [[1, 2, 3, 5]]
        assert deterministic_topk(x, 5)[1].tolist() == [[1, 2, 3, 5, 4]]

    def test_bfloat16_ties_are_resolved_the_same_way(self):
        g = torch.Generator().manual_seed(3)
        x = (torch.randn(64, 500, generator=g) * 2).bfloat16()
        values, idx = deterministic_topk(x, 50)
        for r in range(64):
            assert idx[r].tolist() == sorted(
                range(500), key=lambda i: (-float(x[r, i]), i))[:50]
        assert values.dtype == torch.bfloat16

    def test_it_is_invariant_to_how_the_rows_are_blocked(self):
        x = torch.randint(0, 6, (40, 90)).float()       # many ties
        whole = deterministic_topk(x, 13)[1]
        parts = torch.cat([deterministic_topk(x[i:i + 7], 13)[1]
                           for i in range(0, 40, 7)])
        assert torch.equal(whole, parts)

    def test_nan_rows_select_deterministically_and_keep_their_nan(self):
        x = torch.tensor([[0.0, float("nan"), 2.0, 1.0]])
        values, idx = deterministic_topk(x, 2)
        assert idx.tolist() == [[1, 2]]
        assert math.isnan(float(values[0, 0]))

    def test_the_historical_sketch_still_uses_torch_topk(self):
        """`reference_topk_tail_v1` is D1's and D3's: its arithmetic must not
        move. Without ties the two selections agree exactly."""
        logits = torch.randn(20, 64, generator=torch.Generator().manual_seed(4))
        tg = torch.zeros(20, dtype=torch.long)
        old = sketch_reference(logits, tg, top_k=K)
        new, _ = sketch_reference_deterministic(logits, tg, top_k=K)
        assert torch.equal(old.support_indices, new.support_indices)
        assert torch.equal(old.tail_log_prob, new.tail_log_prob)


# --- 3. the reference's K+1 confidence ------------------------------------------

class TestTopKTailConfidence:

    def test_it_is_the_entropy_of_absolute_probabilities(self):
        logits = torch.randn(9, 40, generator=torch.Generator().manual_seed(5))
        sk, _ = sketch_reference_deterministic(
            logits, torch.zeros(9, dtype=torch.long), top_k=K)
        h, c = reference_topk_confidence(sk.support_log_probs,
                                         sk.tail_log_prob, top_k=K,
                                         vocab_size=40)
        p = _log_softmax64(logits).exp()
        for r in range(9):
            top = sorted(_topk_set(logits[r], K))
            cats = [float(p[r, i]) for i in top]
            cats.append(1.0 - sum(cats))        # absolute, not renormalized
            want = -sum(x * math.log(x) for x in cats if x > 0)
            assert float(h[r]) == pytest.approx(want, abs=1e-5)
            assert float(c[r]) == pytest.approx(1 - want / math.log(K + 1),
                                                abs=1e-6)

    def test_categories_follow_the_support(self):
        assert topk_tail_categories(200, 151936) == 201
        assert topk_tail_categories(8, 8) == 8
        assert topk_tail_categories(9, 8) == 8
        uniform = torch.zeros(1, 8)
        sk, _ = sketch_reference_deterministic(
            uniform, torch.zeros(1, dtype=torch.long), top_k=8)
        h, c = reference_topk_confidence(sk.support_log_probs,
                                         sk.tail_log_prob, top_k=8,
                                         vocab_size=8)
        assert float(h[0]) == pytest.approx(math.log(8), abs=1e-6)
        assert float(c[0]) == pytest.approx(0.0, abs=1e-6)

    def test_zero_log_zero_is_zero(self):
        lp = torch.tensor([[0.0, float("-inf"), float("-inf")]])
        h = topk_tail_entropy(lp, torch.tensor([float("-inf")]), has_tail=True)
        assert float(h[0]) == 0.0

    def test_the_candidate_cannot_move_it(self, root, parent):
        items = _items()
        a = _store(root, items)
        for item in items:
            before = a.per_position(item, "confidence").clone()
            #: Whatever a candidate does, the store holds the root's value.
            _depth(parent, a)
            assert torch.equal(a.per_position(item, "confidence"), before)


# --- 4. retrieval through the output head ---------------------------------------

class TestRetrieval:

    def test_selected_entries_equal_the_full_head(self):
        g = torch.Generator().manual_seed(6)
        for kw in ({}, {"bias": torch.randn(50, generator=g)},
                   {"logit_scale": 0.5}, {"final_softcap": 3.0}):
            head = LinearOutputHead(weight=torch.randn(50, 7, generator=g),
                                    **kw)
            hidden = torch.randn(11, 7, generator=g)
            ids = torch.randint(0, 50, (11, 6), generator=g)
            full = head.logits(hidden)
            assert torch.allclose(head.token_logits(hidden, ids),
                                  full.gather(1, ids), atol=1e-5)

    def test_student_only_entries_get_the_teachers_exact_log_prob(self):
        head, hidden, ref = _head_and_logits(13)
        sk, log_z = sketch_reference_deterministic(
            ref, torch.zeros(int(ref.shape[0]), dtype=torch.long), top_k=K)
        r = TeacherRetrieval(hidden, log_z, head)
        ids = torch.argsort(ref, dim=-1)[:, :K]        # the reference's BOTTOM
        want = torch.log_softmax(ref, -1).gather(1, ids)
        assert torch.allclose(r.log_probs(ids), want, atol=1e-5)
        #: and on the reference's own Top-K, the sketch's own values
        assert torch.allclose(r.log_probs(sk.support_indices),
                              sk.support_log_probs, atol=1e-5)

    def test_the_qwen3_head_reproduces_the_models_logits(self, root):
        adapter = adapter_for_config(root.config)
        adapter.requires(Capability.OUTPUT_HEAD_LINEAR)
        head = adapter.linear_output_head(root)
        ids = _items()[0]["input_ids"]
        with torch.no_grad():
            hidden = adapter.final_hidden_states(root, ids)[0]
            want = root(ids).logits[0]
        assert torch.allclose(head.logits(hidden), want.float(), atol=1e-5)
        assert head.identity()["tied_embeddings"] is True

    def test_a_foreign_hidden_state_or_token_is_refused(self):
        head = LinearOutputHead(weight=torch.randn(20, 4))
        with pytest.raises(OutputHeadError, match="different models"):
            head.logits(torch.randn(3, 5))
        with pytest.raises(OutputHeadError, match="outside"):
            head.token_logits(torch.randn(2, 4), torch.tensor([[0, 20]] * 2))


# --- 5. the objective form and its pairing --------------------------------------

class TestTheForm:

    def test_its_identity_names_both_redefinitions(self):
        d = V3.as_dict()
        assert d["form"] == FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL
        assert d["confidence_top_k"] == K
        assert d["kl_support"] == UNION.support_id
        assert d["confidence_source"].startswith("reference_topk_plus_tail")
        assert d["kl_reference"] == KL_REFERENCE_ROOT
        assert V3.uses_union_support and V3.requires_root_reference
        assert V3.objective_hash != V2.objective_hash
        assert V3.components() == UNION_COMPONENTS

    @pytest.mark.parametrize("bad", [dict(entropy_top_k=None),
                                     dict(entropy_top_k=0),
                                     dict(reward_scale=1.0),
                                     dict(kl_reference=None)])
    def test_it_refuses_incomplete_or_foreign_fields(self, bad):
        kw = {"entropy_top_k": K, "kl_reference": KL_REFERENCE_ROOT, **bad}
        with pytest.raises(ObjectiveError):
            ScoringObjective("x", 1,
                             FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL,
                             **kw)

    def test_the_union_support_and_form_travel_together(self):
        require_objective_support(V3, UNION)
        require_objective_support(V2, reference_topk_tail(K))
        require_objective_support(D3_LIKE, reference_topk_tail(K))
        require_objective_support(None, reference_topk_tail(K))
        for objective, support in ((V3, reference_topk_tail(K)),
                                   (V2, UNION), (D3_LIKE, UNION), (None, UNION),
                                   (V3, reference_student_union_topk_tail(K + 1))):
            with pytest.raises(ObjectiveError):
                require_objective_support(objective, support)

    def test_no_fallback_to_the_unweighted_kl(self):
        kl = torch.rand(5) + 0.1
        conf = torch.tensor([0.1, 0.5, 0.9, 1.0, 0.0])
        union = {k: torch.zeros(5) for k in
                 ("kl_reference_topk", "union_size", "union_overlap",
                  "reference_union_tail_mass", "candidate_union_tail_mass")}
        terms = position_terms(V3, kl, torch.randn(5, 9), torch.zeros(5), None,
                               reference_entropy=torch.rand(5),
                               reference_confidence=conf, union=union)
        assert torch.allclose(terms["objective"], conf * kl)
        assert not torch.allclose(terms["objective"], terms["raw_kl"])
        with pytest.raises(ObjectiveError, match="union reduction"):
            position_terms(V3, kl, torch.randn(5, 9), torch.zeros(5), None,
                           reference_entropy=torch.rand(5),
                           reference_confidence=conf)
        with pytest.raises(ObjectiveError, match="none was supplied"):
            position_terms(V3, kl, torch.randn(5, 9), torch.zeros(5), None,
                           union=union)

    def test_the_support_identity_carries_the_tie_break(self):
        assert UNION.as_dict() == {"support_id": UNION.support_id, "top_k": K,
                                   "topk_tie_break": TOPK_TIE_BREAK}
        #: D1's / D3's support serializes exactly as before.
        assert reference_topk_tail(200).as_dict() == {
            "support_id": "reference_topk_tail_v1", "top_k": 200}


# --- 6. the store ------------------------------------------------------------------

def _items(seq_lens=(20, 14, 17, 11)):
    raw = make_items(n_per_subtype=2, seq_len=max(seq_lens))
    for item, n in zip(raw, seq_lens):
        item["input_ids"] = item["input_ids"][:, :n]
        m = n - 1
        item["tags"] = {k: v[:m] for k, v in item["tags"].items()}
        item["tags"]["assistant"] = torch.arange(m) >= 4
    return prepare_calibration_items(raw, profile_id="toy@v1")


def _identity(objective=V3, support=UNION, profile_hash="p" * 64):
    return root_reference_identity(root_identity=ROOT_ID,
                                   profile_hash=profile_hash, support=support,
                                   objective=objective)


def _store(model, items, *, bsz=1):
    return build_root_reference_store(
        model, items, support=UNION, identity=_identity(),
        micro_batch_size=bsz,
        packing="length_sorted_v1" if bsz > 1 else "original_order_v1",
        device="cpu")


class TestTheStore:

    def test_each_entry_is_the_roots_own_head_distribution(self, root):
        items = _items()
        store = _store(root, items)
        assert store.retrieves and store.identity["schema"] == UNION_SCHEMA
        adapter = adapter_for_config(root.config)
        head = adapter.linear_output_head(root)
        for item in items:
            with torch.no_grad():
                hid = adapter.final_hidden_states(root, item["input_ids"])[0,
                                                                          :-1]
            logits = head.logits(hid)
            sk, log_z = sketch_reference_deterministic(
                logits, item["input_ids"][0, 1:], top_k=K)
            got = store.sketch_item(item)
            assert torch.equal(got.support_indices, sk.support_indices)
            assert torch.allclose(got.support_log_probs, sk.support_log_probs)
            h, c = reference_topk_confidence(sk.support_log_probs,
                                             sk.tail_log_prob, top_k=K,
                                             vocab_size=GEOMETRY["vocab_size"])
            assert torch.allclose(store.per_position(item, "confidence"), c)
            r = store.retrieval_item(item)
            ids = torch.randint(0, GEOMETRY["vocab_size"], (int(hid.shape[0]), 5))
            assert torch.allclose(r.log_probs(ids),
                                  torch.log_softmax(logits, -1).gather(1, ids),
                                  atol=1e-5)

    def test_batch_one_and_three_store_the_same_values(self, root):
        items = _items()
        one, three = _store(root, items), _store(root, items, bsz=3)
        #: The EXECUTION is part of a union store's identity ...
        assert one.identity_hash != three.identity_hash
        assert one.identity["build"]["micro_batch_size"] == 1
        #: ... and in float32 on CPU the values agree (the bf16 drift between
        #: them is a GPU validation diagnostic, never a gate).
        for item in items:
            for name in ("entropy", "confidence", "target_log_prob"):
                assert torch.allclose(one.per_position(item, name),
                                      three.per_position(item, name),
                                      atol=1e-5), name

    def test_the_identity_binds_every_declared_input(self, root):
        base = _identity()
        assert base["support"] == UNION.as_dict()
        for other in (_identity(profile_hash="q" * 64),
                      _identity(support=reference_student_union_topk_tail(9),
                                objective=ScoringObjective(
                                    "objective.toy_union", 1,
                                    FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL,
                                    entropy_top_k=9,
                                    kl_reference=KL_REFERENCE_ROOT))):
            assert other != base
        store = _store(root, _items())
        #: RESUME: a store rebuilt from the same root, mixture and execution
        #: (a resumed search rebuilds it) declares the same identity, so the
        #: statistics cache keys it enters still match.
        assert _store(root, _items()).identity_hash == store.identity_hash
        build = store.identity["build"]
        assert build["output_head"]["logit_definition"].startswith("float32")
        assert build["adapter"]["capabilities"] == [
            Capability.OUTPUT_HEAD, Capability.OUTPUT_HEAD_LINEAR]
        assert build["device_type"] == "cpu" and build["model_dtype"] == "float32"

    def test_the_historical_identity_is_unchanged(self):
        support = reference_topk_tail(K)
        for objective in (None, V2):
            assert root_reference_identity(
                root_identity=ROOT_ID, profile_hash="p" * 64, support=support,
                objective=objective) == {
                    "schema": SCHEMA, "root": ROOT_ID,
                    "profile_hash": "p" * 64, "support": support.as_dict(),
                    "entropy_definition":
                        "reference_full_vocabulary_shannon_entropy/v1",
                    "confidence_mapping":
                        "one_minus_entropy_over_log_vocabulary/v1",
                    "rows": "all_real_prediction_positions/v1"}

    def test_mismatched_declarations_are_refused(self, root):
        items = _items()
        with pytest.raises(ReferenceStoreError, match="union identity"):
            build_root_reference_store(
                root, items, support=UNION,
                identity=_identity(objective=None, support=UNION),
                micro_batch_size=1, packing="original_order_v1", device="cpu")
        with pytest.raises(ReferenceStoreError, match="non-union support"):
            build_root_reference_store(
                root, items, support=reference_topk_tail(K),
                identity=_identity(), micro_batch_size=1,
                packing="original_order_v1", device="cpu")
        with pytest.raises(ReferenceStoreError, match="union support"):
            root_reference_identity(root_identity=ROOT_ID,
                                    profile_hash="p" * 64,
                                    support=reference_topk_tail(K),
                                    objective=V3)

    def test_a_store_without_retrieval_cannot_serve_retrieval(self, root):
        items = _items()
        support = reference_topk_tail(K)
        v2 = build_root_reference_store(
            root, items, support=support,
            identity=root_reference_identity(root_identity=ROOT_ID,
                                             profile_hash="p" * 64,
                                             support=support),
            micro_batch_size=1, packing="original_order_v1", device="cpu")
        assert not v2.retrieves
        with pytest.raises(ReferenceStoreError, match="no retrieval"):
            v2.retrieval_item(items[0])

    def test_a_changed_head_is_refused(self, root):
        items = _items()
        store = _store(root, items)
        store.require_items(items)
        store.head_checksum += 1.0
        with pytest.raises(ReferenceStoreError, match="no longer holds"):
            store.require_items(items)


# --- 7. DEPTH ------------------------------------------------------------------------

def _depth_ctx(model, store, *, objective=V3, support=UNION, batch_size=1,
               rows="all_v1"):
    items = _items()
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


def _independent(reference, model, items, skip, *, weighted=True):
    """One subset's DEPTH objective from first principles: the reference's
    float32 head distribution and K+1 confidence, the bypassed model's logits,
    the float64 union-KL oracle, the fixed 1/T_x, then sub-type and domain."""
    adapter = adapter_for_config(reference.config)
    head = adapter.linear_output_head(reference)
    per_subtype: dict[str, list[float]] = {}
    for item in items:
        ids = item["input_ids"]
        with torch.no_grad():
            ref = head.logits(adapter.final_hidden_states(reference, ids)[0, :-1])
            with bypassed_blocks(model, set(skip)):
                cand = model(ids).logits[0, :-1].float()
        p, q = _log_softmax64(ref), _log_softmax64(cand)
        sk, _ = sketch_reference_deterministic(ref, ids[0, 1:], top_k=K)
        _, c = reference_topk_confidence(sk.support_log_probs,
                                         sk.tail_log_prob, top_k=K,
                                         vocab_size=GEOMETRY["vocab_size"])
        scored = SUPERVISED_TARGET_V1.weights(item,
                                              axis="prediction").weights > 0
        total = 0.0
        for t in torch.nonzero(scored).flatten().tolist():
            kl = _partition_kl(p[t], q[t], _topk_set(ref[t], K)
                               | _topk_set(cand[t], K))
            total += (float(c[t]) if weighted else 1.0) * kl
        per_subtype.setdefault(item["subtype"], []).append(
            total / int(scored.sum()))
    domains: dict[str, set] = {}
    for item in items:
        domains.setdefault(item["domain"], set()).add(item["subtype"])
    return domain_balanced_score(
        {k: sum(v) / len(v) for k, v in per_subtype.items()},
        {d: sorted(s) for d, s in domains.items()})[0]


class TestDepth:

    def test_every_score_is_the_independent_union_objective(self, root, parent):
        store = _store(root, _items())
        out, items = _depth(parent, store)
        cands = out.artifacts["objective_candidates"]
        for entry in out.artifacts["search_rounds"][0]["table"][:3]:
            want = _independent(root, parent, items, {entry["candidate"]})
            assert entry["score"] == pytest.approx(want, rel=1e-4, abs=1e-6)
            raw = _independent(root, parent, items, {entry["candidate"]},
                               weighted=False)
            rec = cands[str(entry["candidate"])]
            assert rec["raw_kl"] == pytest.approx(raw, rel=1e-4, abs=1e-6)
            #: the lower bound and the union's evidence are recorded beside it
            assert rec["kl_reference_topk"] <= rec["raw_kl"] + 1e-6
            assert K <= rec["union_size"] <= 2 * K
            assert rec["objective"] < rec["raw_kl"]

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_scored_rows_score_identically(self, root, parent, batch_size):
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

    def test_the_parent_kl_stays_a_round_zero_diagnostic(self, root, parent):
        out, _ = _depth(parent, _store(root, _items()))
        with_parent = {k for k, v in out.artifacts["objective_candidates"].items()
                       if "parent_kl" in v}
        assert with_parent == {str(e["candidate"]) for e in
                               out.artifacts["search_rounds"][0]["table"]}
        assert out.local_metrics.reference == "root_reference"

    def test_changing_the_weight_changes_the_decision_scores(self, root, parent):
        items = _items()
        store = _store(root, items)
        base, _ = _depth(parent, store)
        for e in store.entries.values():
            e.confidence = torch.linspace(0, 1, int(e.confidence.shape[0]))
        moved, _ = _depth(parent, store)
        a = base.artifacts["search_rounds"][0]["table"]
        b = moved.artifacts["search_rounds"][0]["table"]
        assert any(abs(x["score"] - y["score"]) > 1e-6 for x, y in zip(a, b))
        #: ... and only the weight moved: the raw union KL is unchanged.
        ca, cb = (base.artifacts["objective_candidates"],
                  moved.artifacts["objective_candidates"])
        for key in ca:
            if key in cb:
                assert cb[key]["raw_kl"] == pytest.approx(ca[key]["raw_kl"],
                                                          rel=1e-6)

    def test_refusals(self, root, parent):
        items = _items()
        #: no store
        ctx, _ = _depth_ctx(parent, None)
        with pytest.raises(Exception, match="ROOT"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)
        #: a store without retrieval (the historical support's)
        support = reference_topk_tail(K)
        v2 = build_root_reference_store(
            root, items, support=support,
            identity=root_reference_identity(root_identity=ROOT_ID,
                                             profile_hash="p" * 64,
                                             support=support),
            micro_batch_size=1, packing="original_order_v1", device="cpu")
        ctx, _ = _depth_ctx(parent, v2)
        with pytest.raises(ReferenceStoreError, match="does not match"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)
        #: the union support without the union form
        ctx, _ = _depth_ctx(parent, _store(root, items), objective=V2)
        with pytest.raises(Exception, match="union"):
            causal_kl_greedy.DepthCausalKLGreedyV1().apply(ctx)
        #: a store built for another mixture
        ctx, _ = _depth_ctx(parent, _store(root, items[:2]))
        with pytest.raises(ReferenceStoreError):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)

    def test_d3s_path_is_untouched(self, root, parent):
        """The student-confidence reward on the reference Top-K support keeps
        the PARENT reference and the K+1 KL: no store, no union."""
        support = reference_topk_tail(K)
        out, _ = _depth(parent, None, objective=D3_LIKE, support=support)
        assert out.local_metrics.reference == "parent_state"
        rec = next(iter(out.artifacts["objective_candidates"].values()))
        assert "union_size" not in rec and "student_entropy" in rec


# --- 8. the state evaluator and the beam ---------------------------------------------

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
        suite_id="test.union", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(items),
        general_domain="general")
    return suite, items


def _evaluator(teacher, *, objective=V3, support=UNION, rows="all_v1",
               batch=1):
    suite, items = _suite()
    ev = StateEvaluator(
        suite, items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
        execution=ExecutionConfig(
            micro_batch_size=batch,
            calibration_batch_packing=("length_sorted_v1" if batch > 1
                                       else "original_order_v1"),
            prediction_rows=rows),
        distribution_support=support, numerics=CPU,
        suite_content_sha256="toy", scoring_objective=objective)
    ev.prime_reference(teacher)
    return ev


class TestStateEvaluation:

    def test_each_item_is_the_independent_union_objective(self, root, parent):
        ev = _evaluator(root).evaluate(parent, "candidate")
        adapter = adapter_for_config(root.config)
        head = adapter.linear_output_head(root)
        _, items = _suite()
        per_item = ev.detail["objective"]["per_item"]
        for item in items:
            with torch.no_grad():
                ref = head.logits(adapter.final_hidden_states(
                    root, item.input_ids)[0, :-1])
                cand = parent(item.input_ids).logits[0, :-1].float()
            p, q = _log_softmax64(ref), _log_softmax64(cand)
            sk, _ = sketch_reference_deterministic(ref, item.input_ids[0, 1:],
                                                   top_k=K)
            _, c = reference_topk_confidence(sk.support_log_probs,
                                             sk.tail_log_prob, top_k=K,
                                             vocab_size=GEOMETRY["vocab_size"])
            scored = torch.nonzero(item.tags["assistant"]).flatten().tolist()
            want = sum(float(c[t]) * _partition_kl(
                p[t], q[t], _topk_set(ref[t], K) | _topk_set(cand[t], K))
                for t in scored) / len(scored)
            assert per_item[item.item_id]["objective"] == pytest.approx(
                want, rel=1e-4, abs=1e-6)

    def test_raw_kl_is_the_union_kl_metric(self, root, parent):
        """One source: the objective accumulator's raw KL reproduces the
        state's own KL metrics, which are now the union KL."""
        v = _evaluator(root).evaluate(parent, "candidate").values
        assert v["state.objective.raw_kl.equal_domain_mean"] == pytest.approx(
            v["state.teacher_kl.equal_domain_mean"], rel=1e-6)
        assert v["state.objective.raw_kl.critical_token"] == pytest.approx(
            v["state.critical_token_kl"], rel=1e-6)
        assert v["state.objective.kl_reference_topk.equal_domain_mean"] <= \
            v["state.objective.raw_kl.equal_domain_mean"] + 1e-6
        for key in ("state.objective.equal_domain_mean",
                    "state.objective.worst_domain",
                    "state.objective.critical_token"):
            assert key in v

    def test_the_teacher_is_forwarded_once(self, root, parent):
        ev = _evaluator(root)
        calls = []
        hook = root.register_forward_hook(lambda *a: calls.append(1))
        try:
            ev.evaluate(parent, "a")
            ev.evaluate(parent, "b")
        finally:
            hook.remove()
        assert calls == []

    @pytest.mark.parametrize("batch", [1, 3])
    def test_scored_rows_measure_the_same_objective(self, root, parent, batch):
        base = _evaluator(root, batch=batch).evaluate(parent, "c")
        rows = _evaluator(root, batch=batch, rows="scored_v1").evaluate(parent,
                                                                        "c")
        for key, value in base.values.items():
            if key.startswith("state.objective."):
                assert rows.values[key] == pytest.approx(value, rel=1e-5,
                                                         abs=1e-8), key

    def test_the_protocol_id_moves_with_the_redefinition(self, root):
        support = reference_topk_tail(K)
        v2 = StateEvaluator(*_suite(), device="cpu",
                            position_policy=SUPERVISED_TARGET_V1,
                            distribution_support=support,
                            scoring_objective=V2)
        v3 = StateEvaluator(*_suite(), device="cpu",
                            position_policy=SUPERVISED_TARGET_V1,
                            distribution_support=UNION, scoring_objective=V3)
        assert v2.measurement_protocol_id != v3.measurement_protocol_id

    def test_a_mismatched_pair_is_refused(self):
        from aadistill.initialization.specs.metrics import MeasurementError
        with pytest.raises(MeasurementError):
            StateEvaluator(*_suite(), device="cpu", distribution_support=UNION,
                           scoring_objective=V2)
        with pytest.raises(MeasurementError):
            StateEvaluator(*_suite(), device="cpu",
                           distribution_support=reference_topk_tail(K),
                           scoring_objective=V3)


OBJECTIVE_PARETO = BeamRankingPolicy(
    policy_id="beam.toy_union_pareto", version=1,
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


def _search(root, *, objective=V3, support=UNION, measurer=None):
    suite, items = _suite()
    calib = _items()
    if measurer is None:
        ev = StateEvaluator(suite, items, device="cpu",
                            position_policy=SUPERVISED_TARGET_V1,
                            distribution_support=support,
                            scoring_objective=objective)
        ev.prime_reference(root)
        measurer = _Measurer(ev)
    config = SearchConfig(
        run_id="union", target_spec=ArchSpec.of(
            "qwen3", {**GEOMETRY, "num_hidden_layers": 4,
                      "intermediate_size": 24}),
        schedule=SCHEDULE_V1, seed=7, workdir=Path(tempfile.mkdtemp()),
        profiles=(make_profile("balanced"),), policy=OBJECTIVE_PARETO,
        suite=suite, position_policy=SUPERVISED_TARGET_V1,
        distribution_support=support, scoring_objective=objective,
        allowed_impls=("depth.causal_kl_greedy_v1",
                       "ffn.activation_importance_v0"),
        max_depth=2, device="cpu")
    return BeamSearch(
        adapter=adapter_for_config(root.config), config=config,
        root_teacher_id="toy/teacher", root_teacher_sha256="de" * 32,
        root_loader=lambda: root, calibration_loader=lambda profile: calib,
        measurer=measurer, execution=ExecutionConfig(), numerics=CPU)


class TestSearchAndStatistics:

    def test_the_search_ranks_on_the_union_objective(self, root):
        search = _search(root)
        result = search.run()
        assert result.states
        (evidence,) = search.root_reference_evidence().values()
        assert evidence["identity"]["schema"] == UNION_SCHEMA
        assert evidence["decision"]["retrieval_bytes"] > 0
        measured = [s for s in result.states.values()
                    if s.evaluation is not None]
        assert measured
        for state in measured:
            v = state.evaluation.values
            assert "state.objective.union_size.equal_domain_mean" in v
            assert v["state.objective.equal_domain_mean"] <= \
                v["state.objective.raw_kl.equal_domain_mean"] + 1e-9

    def test_a_mismatched_pair_is_refused_at_construction(self, root):
        with pytest.raises(SearchError):
            _search(root, objective=V2,
                    measurer=lambda model, digest: None)

    def test_the_weight_identity_enters_the_statistics_cache_key(self, root):
        from types import SimpleNamespace

        parent = SimpleNamespace(artifact_digest="f" * 64)
        profile = make_profile("balanced")
        ffn = get_implementation("ffn.activation_importance_v0")
        v3 = _search(root)
        v2 = _search(root, objective=V2, support=reference_topk_tail(K))
        assert v3._stats_key(parent, profile, ffn) != \
            v2._stats_key(parent, profile, ffn)
        assert v3._stats_key(parent, profile, ffn) != \
            v3._stats_key(parent, profile)

    def test_weighted_statistics_read_the_k_plus_1_confidence(self, root):
        items = _items()
        store = _store(root, items)
        ctx = OperatorContext(
            adapter=get_adapter("qwen3"), model=root,
            parent_spec=ArchSpec.of("qwen3", GEOMETRY),
            target_spec=ArchSpec.of("qwen3", GEOMETRY), profile=None,
            calibration_items=items, seed=0, device="cpu", config={},
            position_policy=SUPERVISED_TARGET_V1, scoring_objective=V3,
            root_reference=store)
        for impl_id in ("ffn.activation_importance_v0", "width.global_pca_v0"):
            active = statistics_positions(ctx, get_implementation(impl_id))
            assert active.token_weighted
            for k, item in enumerate(items):
                policy_w = SUPERVISED_TARGET_V1.weights(item,
                                                        axis="token").weights
                conf = store.per_position(item, "confidence").double()
                want = policy_w * torch.cat([conf, torch.zeros(1,
                                             dtype=torch.float64)])
                assert torch.allclose(active.token_weights_for_item(k), want)
            assert active.report()["token_weights_multiplied_by"].endswith(
                store.identity_hash)

    def test_changing_the_weight_changes_the_statistics_weights(self, root):
        items = _items()
        store = _store(root, items)
        ctx = OperatorContext(
            adapter=get_adapter("qwen3"), model=root,
            parent_spec=ArchSpec.of("qwen3", GEOMETRY),
            target_spec=ArchSpec.of("qwen3", GEOMETRY), profile=None,
            calibration_items=items, seed=0, device="cpu", config={},
            position_policy=SUPERVISED_TARGET_V1, scoring_objective=V3,
            root_reference=store)
        ffn = get_implementation("ffn.activation_importance_v0")
        before = statistics_positions(ctx, ffn).token_weights_for_item(0).clone()
        store.entries[str(items[0]["item_id"])].confidence *= 0.5
        after = statistics_positions(ctx, ffn).token_weights_for_item(0)
        assert not torch.allclose(before, after)
