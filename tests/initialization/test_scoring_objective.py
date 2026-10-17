"""Student-confidence scoring objectives: definition, direction, identity, use.

`aadistill.initialization.scoring.objective` adds two per-position objectives
to the KL every D-series decision already minimizes, both reading the COMPARED
(student) model's own Top-K conditional entropy:

* uncertainty-weighted KL:   g = (1 - c) * KL
* KL minus a confidence reward:  g = KL - lambda * c

averaged per item over a FIXED count of scored positions. Each claim below has
a test, and the ones that would be easy to get backwards -- the sign of each
form, the denominator, which model's logits the confidence comes from -- are
pinned against an independent computation rather than against the module.
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
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.base import (  # noqa: E402
    ContractViolation, OperatorContext, OperatorError, get_implementation,
)
from aadistill.initialization.operators.depth import causal_kl_greedy  # noqa: E402
from aadistill.initialization.planning.metrics import StateEvaluator  # noqa: E402
from aadistill.initialization.planning.operator_config import (  # noqa: E402
    hashed_operator_config,
)
from aadistill.initialization.planning.ranking import (  # noqa: E402
    PARETO_V1, SCHEDULE_V1, STATE_ID_TIEBREAK, VALIDITY_GUARDRAIL,
    BeamRankingPolicy, Objective,
)
from aadistill.initialization.planning.search import (  # noqa: E402
    BeamSearch, SearchConfig, SearchError,
)
from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_KL_MINUS_CONFIDENCE_REWARD, FORM_UNCERTAINTY_WEIGHTED_KL,
    ObjectiveError, ScoringObjective, confidence_from_entropy,
    fixed_denominator_item_means, student_topk_terms,
)
from aadistill.initialization.scoring.positions import (  # noqa: E402
    SUPERVISED_TARGET_V1, policy_config,
)
from aadistill.initialization.scoring.protocol_identity import (  # noqa: E402
    ReductionSemantics,
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
    MeasurementError, StateEvalSuite, SuiteItem,
)
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    bypassed_blocks, domain_balanced_score,
)

K = 8
WEIGHTED = ScoringObjective("objective.toy_uncertainty_weighted_kl", 1,
                            FORM_UNCERTAINTY_WEIGHTED_KL, entropy_top_k=K)
REWARD = ScoringObjective("objective.toy_kl_minus_confidence", 1,
                          FORM_KL_MINUS_CONFIDENCE_REWARD, entropy_top_k=K,
                          reward_scale=0.25)
CPU = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
LOG_Z_UNUSED = None


def _terms(logits, k=K):
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return student_topk_terms(logits, log_z, top_k=k)


def _brute_entropy(row: np.ndarray, k: int) -> float:
    """Independent: sort, keep the K largest, renormalize, entropy."""
    top = np.sort(row.astype(np.float64))[::-1][:k]
    p = np.exp(top - top.max())
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


# --- 1. the entropy is the student's own Top-K conditional entropy ---------

class TestStudentTopKEntropy:

    def test_matches_an_independent_computation(self):
        g = torch.Generator().manual_seed(3)
        logits = torch.randn(17, 50, generator=g) * 3
        entropy, _ = _terms(logits)
        for r in range(17):
            assert float(entropy[r]) == pytest.approx(
                _brute_entropy(logits[r].numpy(), K), abs=1e-5)

    def test_mass_outside_the_topk_does_not_enter(self):
        """Changing an entry outside the student's own Top-K moves nothing."""
        logits = torch.linspace(5.0, -5.0, 40)[None]
        before, _ = _terms(logits)
        moved = logits.clone()
        moved[0, -1] = 0.0          # still far below the 8th-largest entry
        after, _ = _terms(moved)
        assert float(after[0]) == pytest.approx(float(before[0]), abs=1e-7)

    def test_it_is_not_the_full_vocabulary_entropy(self):
        logits = torch.zeros(1, 40)        # uniform over 40 entries
        entropy, mass = _terms(logits)
        assert float(entropy[0]) == pytest.approx(math.log(K), abs=1e-6)
        assert float(entropy[0]) < math.log(40) - 1.0
        assert float(mass[0]) == pytest.approx(K / 40, rel=1e-6)

    def test_uniform_topk_is_zero_confidence_and_one_hot_is_full(self):
        uniform = torch.zeros(1, 40)
        sharp = torch.full((1, 40), -60.0)
        sharp[0, 7] = 60.0
        h_u, _ = _terms(uniform)
        h_s, _ = _terms(sharp)
        #: Unclamped bounds first, so the clamp below cannot hide a defect.
        assert -1e-6 <= float(h_s[0]) and float(h_u[0]) <= math.log(K) + 1e-5
        assert float(confidence_from_entropy(h_u, K)[0]) == pytest.approx(
            0.0, abs=1e-5)
        assert float(confidence_from_entropy(h_s, K)[0]) == pytest.approx(
            1.0, abs=1e-5)

    def test_topk_mass_is_the_full_softmax_mass_of_the_topk(self):
        g = torch.Generator().manual_seed(5)
        logits = torch.randn(4, 30, generator=g)
        _, mass = _terms(logits)
        p = torch.softmax(logits.double(), dim=-1)
        expected = torch.topk(p, K, dim=-1).values.sum(dim=-1)
        assert torch.allclose(mass.double(), expected, rtol=1e-5)

    def test_bfloat16_selects_and_scores_its_own_values_exactly(self):
        g = torch.Generator().manual_seed(9)
        logits = (torch.randn(6, 64, generator=g) * 4).to(torch.bfloat16)
        h_bf, _ = _terms(logits)
        h_fp, _ = _terms(logits.float())
        assert torch.equal(h_bf, h_fp)

    def test_k_larger_than_the_vocabulary_is_refused(self):
        with pytest.raises(ObjectiveError, match="exceeds the vocabulary"):
            _terms(torch.zeros(1, 4), k=8)

    def test_k_below_two_is_refused(self):
        with pytest.raises(ObjectiveError, match="at least 2"):
            ScoringObjective("x", 1, FORM_UNCERTAINTY_WEIGHTED_KL,
                             entropy_top_k=1)


# --- 2. direction of each form ----------------------------------------------

class TestEachFormPointsTheRightWay:

    KL = torch.tensor([0.4, 0.4, 0.4])

    def test_weighted_form_is_one_minus_confidence(self):
        conf = torch.tensor([0.0, 0.5, 1.0])
        g, weight = WEIGHTED.integrand(self.KL, conf)
        assert torch.allclose(weight, torch.tensor([1.0, 0.5, 0.0]))
        assert torch.allclose(g, torch.tensor([0.4, 0.2, 0.0]))

    def test_more_confidence_lowers_the_weighted_penalty(self):
        g_low, _ = WEIGHTED.integrand(self.KL, torch.full((3,), 0.2))
        g_high, _ = WEIGHTED.integrand(self.KL, torch.full((3,), 0.8))
        assert bool((g_high < g_low).all())

    def test_more_confidence_lowers_the_reward_objective(self):
        g_low, _ = REWARD.integrand(self.KL, torch.full((3,), 0.2))
        g_high, reward = REWARD.integrand(self.KL, torch.full((3,), 0.8))
        assert bool((g_high < g_low).all())
        assert torch.allclose(reward, torch.full((3,), 0.25 * 0.8))
        assert torch.allclose(g_high, self.KL - 0.2)

    def test_the_reward_is_not_a_weight(self):
        """Confidence lowers the reward form by lambda*c whatever the KL is."""
        conf = torch.full((2,), 0.6)
        g, _ = REWARD.integrand(torch.tensor([0.0, 5.0]), conf)
        assert torch.allclose(g, torch.tensor([-0.15, 4.85]))

    def test_infinite_kl_is_never_rescued_by_confidence(self):
        inf = torch.tensor([math.inf])
        g_w, _ = WEIGHTED.integrand(inf, torch.tensor([1.0]))
        g_r, _ = REWARD.integrand(inf, torch.tensor([1.0]))
        assert math.isinf(float(g_w)) and float(g_w) > 0
        assert math.isinf(float(g_r)) and float(g_r) > 0

    @pytest.mark.parametrize("lam", [None, 0.0, -1.0, math.inf, math.nan])
    def test_the_reward_form_needs_a_positive_finite_lambda(self, lam):
        with pytest.raises(ObjectiveError, match="reward_scale"):
            ScoringObjective("x", 1, FORM_KL_MINUS_CONFIDENCE_REWARD,
                             entropy_top_k=K, reward_scale=lam)

    def test_the_weighted_form_refuses_a_lambda(self):
        with pytest.raises(ObjectiveError, match="no reward term"):
            ScoringObjective("x", 1, FORM_UNCERTAINTY_WEIGHTED_KL,
                             entropy_top_k=K, reward_scale=0.5)


# --- 3. the fixed denominator -------------------------------------------------

class TestTheDenominatorIsTheScoredCount:

    def test_uniform_half_confidence_halves_the_mean(self):
        """Fixed 1/T_x. A weight-sum denominator would return the KL mean."""
        kl = torch.tensor([[0.2, 0.4, 0.6, 9.0]])
        mask = torch.tensor([[True, True, True, False]])
        g, w = WEIGHTED.integrand(kl, torch.full((1, 4), 0.5))
        means, counts = fixed_denominator_item_means(g, mask)
        assert int(counts[0]) == 3
        assert float(means[0]) == pytest.approx(0.5 * 0.4)
        dynamic = float((g * mask).sum() / (w * mask).sum())
        assert dynamic == pytest.approx(0.4)
        assert float(means[0]) != pytest.approx(dynamic)

    def test_a_uniform_rise_in_confidence_is_not_cancelled(self):
        kl = torch.tensor([[0.3, 0.5]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        low, _ = fixed_denominator_item_means(
            WEIGHTED.integrand(kl, torch.full((1, 2), 0.1))[0], mask)
        high, _ = fixed_denominator_item_means(
            WEIGHTED.integrand(kl, torch.full((1, 2), 0.9))[0], mask)
        assert float(high[0]) < float(low[0])

    def test_unscored_positions_are_never_read(self):
        values = torch.tensor([[1.0, math.nan, math.inf, 3.0]])
        mask = torch.tensor([[True, False, False, True]])
        means, counts = fixed_denominator_item_means(values, mask)
        assert float(means[0]) == pytest.approx(2.0)
        assert int(counts[0]) == 2

    def test_a_policy_weight_of_zero_removes_a_position(self):
        values = torch.tensor([[1.0, 5.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        weights = torch.tensor([[1.0, 0.0, 1.0]])
        means, counts = fixed_denominator_item_means(values, mask, weights)
        assert float(means[0]) == pytest.approx(2.0)
        assert int(counts[0]) == 2

    def test_continuous_position_weights_are_refused(self):
        with pytest.raises(ObjectiveError, match="binary"):
            fixed_denominator_item_means(
                torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[0.5, 1.0]]))

    def test_an_item_with_no_scored_position_is_refused(self):
        with pytest.raises(ObjectiveError, match="T_x is zero"):
            fixed_denominator_item_means(torch.ones(1, 2),
                                         torch.zeros(1, 2, dtype=torch.bool))


# --- 4. identity --------------------------------------------------------------

class TestIdentity:

    def test_the_two_forms_are_two_identities(self):
        assert WEIGHTED.objective_hash != REWARD.objective_hash

    def test_lambda_and_k_are_in_the_identity(self):
        other_lam = ScoringObjective(REWARD.objective_id, 1, REWARD.form,
                                     entropy_top_k=K, reward_scale=0.5)
        other_k = ScoringObjective(WEIGHTED.objective_id, 1, WEIGHTED.form,
                                   entropy_top_k=16)
        assert other_lam.objective_hash != REWARD.objective_hash
        assert other_k.objective_hash != WEIGHTED.objective_hash

    def test_every_definitional_choice_is_named(self):
        d = WEIGHTED.as_dict()
        for key in ("entropy_definition", "entropy_top_k", "confidence_mapping",
                    "denominator", "aggregation", "direction", "form"):
            assert key in d
        assert d["direction"] == "minimize"
        assert "reward_scale" not in d
        assert REWARD.as_dict()["reward_scale"] == 0.25

    def test_no_objective_leaves_the_reduction_identity_unchanged(self):
        support = reference_topk_tail(K)
        before = ReductionSemantics(chunk=512, reference_strategy="recompute",
                                    distribution_support=support)
        with_none = ReductionSemantics(chunk=512, reference_strategy="recompute",
                                       distribution_support=support,
                                       scoring_objective=None)
        with_obj = ReductionSemantics(chunk=512, reference_strategy="recompute",
                                      distribution_support=support,
                                      scoring_objective=WEIGHTED)
        assert with_none.as_dict() == before.as_dict()
        assert "scoring_objective" not in before.as_dict()
        assert with_obj.as_dict()["scoring_objective"] == WEIGHTED.as_dict()

    def test_only_a_consuming_operator_hashes_the_objective(self):
        _register()
        depth = get_implementation("depth.causal_kl_greedy_v1")
        ffn = get_implementation("ffn.activation_importance_v0")
        items = prepare_calibration_items(make_items(), profile_id="t@v1")
        common = dict(policy=SUPERVISED_TARGET_V1,
                      support=reference_topk_tail(K), items=items)
        assert "scoring_objective" not in hashed_operator_config(
            implementation=depth, **common)
        assert hashed_operator_config(
            implementation=depth, objective=WEIGHTED,
            **common)["scoring_objective"] == WEIGHTED.as_dict()
        assert "scoring_objective" not in hashed_operator_config(
            implementation=ffn, objective=WEIGHTED, **common)


def _register():
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )
    register_builtin_adapters()
    register_builtin_operators()


@pytest.fixture(scope="module", autouse=True)
def _registered():
    _register()
    yield


# --- 5. DEPTH: the confidence is the candidate's -------------------------------

GEOMETRY = dict(TEACHER_GEOMETRY)


@pytest.fixture(scope="module")
def model():
    m = build_tiny_model(GEOMETRY)
    m.config.use_cache = False
    return m


def _items():
    raw = make_items(n_per_subtype=2, seq_len=20)
    for item in raw:
        n = int(item["input_ids"].shape[1]) - 1
        item["tags"] = {**item["tags"],
                        "assistant": torch.arange(n) >= 6}
    return prepare_calibration_items(raw, profile_id="toy@v1")


def _depth_ctx(model, *, objective, batch_size=1, rows="all_v1",
               support=None, declare=True):
    support = support or reference_topk_tail(K)
    items = _items()
    config = {"n_calibration_items": len(items),
              **policy_config(SUPERVISED_TARGET_V1)}
    if not support.is_full_vocab:
        config["distribution_support"] = support.as_dict()
    if objective is not None and declare:
        config["scoring_objective"] = objective.as_dict()
    execution = ExecutionConfig(
        micro_batch_size=batch_size,
        calibration_batch_packing=("length_sorted_v1" if batch_size > 1
                                   else "original_order_v1"),
        prediction_rows=rows)
    return OperatorContext(
        adapter=get_adapter("qwen3"), model=model,
        parent_spec=ArchSpec.of("qwen3", GEOMETRY),
        target_spec=ArchSpec.of("qwen3", {**GEOMETRY, "num_hidden_layers": 4}),
        profile=None, calibration_items=items, seed=0, device="cpu",
        config=config, distribution_support=support, execution=execution,
        position_policy=SUPERVISED_TARGET_V1, scoring_objective=objective), items


def _depth(model, **kw):
    ctx, items = _depth_ctx(model, **kw)
    return causal_kl_greedy.DepthCausalKLGreedyV1().apply(ctx), items


def _independent_subset_score(model, items, skip, objective):
    """The objective for one subset, from first principles.

    Parent logits for the reference sketch, the BYPASSED model's logits for the
    KL and -- the point -- for the student's entropy, the brute-force Top-K
    entropy, the fixed denominator, then the sub-type and domain rule.
    """
    per_subtype: dict[str, list[float]] = {}
    for item in items:
        ids = item["input_ids"]
        with torch.no_grad():
            parent = model(ids).logits[0, :-1].float()
            with bypassed_blocks(model, set(skip)):
                cand = model(ids).logits[0, :-1].float()
        sk = sketch_reference(parent, torch.zeros(parent.shape[0],
                                                  dtype=torch.long), top_k=K)
        kl = sketch_forward_kl(sk.support_indices, sk.support_log_probs,
                               sk.tail_log_prob, cand, has_tail=True)
        weights = SUPERVISED_TARGET_V1.weights(item, axis="prediction").weights
        scored = weights > 0
        values = []
        for t in range(int(cand.shape[0])):
            if not bool(scored[t]):
                continue
            h = _brute_entropy(cand[t].numpy(), K)
            c = 1.0 - h / math.log(K)
            if objective.form == FORM_UNCERTAINTY_WEIGHTED_KL:
                values.append((1.0 - c) * float(kl[t]))
            else:
                values.append(float(kl[t]) - objective.reward_scale * c)
        per_subtype.setdefault(item["subtype"], []).append(
            sum(values) / len(values))
    domains = {}
    for item in items:
        domains.setdefault(item["domain"], set()).add(item["subtype"])
    means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
    return domain_balanced_score(means, {d: sorted(s) for d, s in
                                         domains.items()})[0]


class TestDepthUnderAnObjective:

    @pytest.mark.parametrize("objective", [WEIGHTED, REWARD],
                             ids=["weighted", "reward"])
    def test_every_score_is_the_independent_objective(self, model, objective):
        out, items = _depth(model, objective=objective)
        first = out.artifacts["search_rounds"][0]
        for entry in first["table"][:3]:
            expected = _independent_subset_score(model, items,
                                                 {entry["candidate"]}, objective)
            assert entry["score"] == pytest.approx(expected, rel=1e-4,
                                                   abs=1e-6)

    def test_the_confidence_is_the_candidates_not_the_parents(self, model):
        out, items = _depth(model, objective=WEIGHTED)
        record = out.artifacts["objective_candidates"]["0"]
        with torch.no_grad():
            ids = items[0]["input_ids"]
            parent = model(ids).logits[0, :-1]
            with bypassed_blocks(model, {0}):
                cand = model(ids).logits[0, :-1]
        h_parent = np.mean([_brute_entropy(r.numpy(), K) for r in parent])
        h_cand = np.mean([_brute_entropy(r.numpy(), K) for r in cand])
        assert abs(h_parent - h_cand) > 1e-4, "the toy cannot tell them apart"
        #: The recorded entropy follows the candidate's change in confidence.
        sub_rec = {}
        for subset in ("0", "5"):
            sub_rec[subset] = out.artifacts["objective_candidates"][subset][
                "student_entropy"]
        assert sub_rec["0"] != pytest.approx(sub_rec["5"], abs=1e-6)
        assert 0.0 <= record["student_confidence"] <= 1.0

    def test_records_carry_every_component(self, model):
        out, _ = _depth(model, objective=REWARD)
        record = next(iter(out.artifacts["objective_candidates"].values()))
        for name in ("objective", "raw_kl", "student_entropy",
                     "student_confidence", "component", "student_topk_mass",
                     "reference_tail_mass", "confidence_histogram"):
            assert name in record
        assert out.trace["scoring_objective"] == REWARD.as_dict()
        assert "op.depth.objective.final" in out.local_metrics.values
        assert "objective_seconds" in out.artifacts["timing"]

    def test_the_kl_key_keeps_meaning_kl(self, model):
        out, _ = _depth(model, objective=REWARD)
        removed = out.trace["removed_layers"]
        final = out.artifacts["objective_candidates"][
            ",".join(str(i) for i in sorted(removed))]
        assert out.local_metrics.values["op.depth.causal_kl.final"] == \
            pytest.approx(final["raw_kl"])

    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_the_row_path_scores_the_objective_identically(self, model,
                                                           batch_size):
        base, _ = _depth(model, objective=WEIGHTED, batch_size=batch_size)
        rows, _ = _depth(model, objective=WEIGHTED, batch_size=batch_size,
                         rows="scored_v1")
        assert rows.trace["kept_layers"] == base.trace["kept_layers"]
        for ra, rb in zip(base.artifacts["search_rounds"],
                          rows.artifacts["search_rounds"]):
            assert ra["chosen"] == rb["chosen"]
            for ea, eb in zip(ra["table"], rb["table"]):
                assert eb["score"] == pytest.approx(ea["score"], rel=1e-6,
                                                    abs=1e-9)

    def test_batched_and_per_item_paths_agree(self, model):
        one, _ = _depth(model, objective=REWARD, batch_size=1)
        three, _ = _depth(model, objective=REWARD, batch_size=3)
        for ra, rb in zip(one.artifacts["search_rounds"],
                          three.artifacts["search_rounds"]):
            for ea, eb in zip(ra["table"], rb["table"]):
                assert eb["score"] == pytest.approx(ea["score"], rel=1e-5,
                                                    abs=1e-7)

    def test_the_full_vocabulary_oracle_refuses_an_objective(self, model):
        with pytest.raises(OperatorError, match="oracle"):
            _depth(model, objective=WEIGHTED, support=FULL_VOCAB_V1)

    def test_an_undeclared_objective_is_a_contract_violation(self, model):
        ctx, _ = _depth_ctx(model, objective=WEIGHTED, declare=False)
        with pytest.raises(ContractViolation, match="scoring objective"):
            causal_kl_greedy.DepthCausalKLGreedyV1().execute(ctx)


# --- 6. the state evaluator ----------------------------------------------------

def _suite():
    """UNEQUAL lengths, so item-equal and T_x-weighted aggregation differ."""
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
        suite_id="test.objective", version=1, domains=("general", "math"),
        subtypes={"general": ("text",), "math": ("arith",)},
        critical_tags=("eos_like", "answer_like"), n_items=len(items),
        general_domain="general")
    return suite, items


def _evaluate(teacher, candidate, *, objective, rows="all_v1", batch=1,
              support=None):
    suite, items = _suite()
    ev = StateEvaluator(
        suite, items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
        execution=ExecutionConfig(
            micro_batch_size=batch,
            calibration_batch_packing=("length_sorted_v1" if batch > 1
                                       else "original_order_v1"),
            prediction_rows=rows),
        distribution_support=support or reference_topk_tail(K), numerics=CPU,
        suite_content_sha256="toy", scoring_objective=objective)
    ev.prime_reference(teacher)
    return ev, ev.evaluate(candidate, "candidate")


@pytest.fixture(scope="module")
def candidate():
    return build_tiny_model(GEOMETRY, seed=19)


class TestStateEvaluationUnderAnObjective:

    def test_the_kl_metrics_are_untouched_by_the_objective(self, model,
                                                           candidate):
        ev_a, base = _evaluate(model, candidate, objective=None)
        ev_b, with_obj = _evaluate(model, candidate, objective=WEIGHTED)
        for key, value in base.values.items():
            assert with_obj.values[key] == value, key
        added = set(with_obj.values) - set(base.values)
        assert added and all(k.startswith("state.objective.") for k in added)
        assert ev_a.measurement_protocol_id != ev_b.measurement_protocol_id
        assert with_obj.detail["reduction"]["scoring_objective"] == \
            WEIGHTED.as_dict()
        assert "scoring_objective" not in base.detail["reduction"]

    def test_items_are_weighted_by_their_scored_positions(self, model,
                                                          candidate):
        """The KL-only path's weighting: T_x-weighted item means per sub-type."""
        _, ev = _evaluate(model, candidate, objective=REWARD)
        suite, items = _suite()
        per_item = ev.detail["objective"]["per_item"]
        num: dict[str, float] = {}
        den: dict[str, int] = {}
        for item in items:
            rec = per_item[item.item_id]
            num[item.subtype] = num.get(item.subtype, 0.0) + (
                rec["objective"] * rec["scored_positions"])
            den[item.subtype] = den.get(item.subtype, 0) + rec["scored_positions"]
        assert len({r["scored_positions"] for r in per_item.values()}) > 1
        means = {k: num[k] / den[k] for k in num}
        primary, per_domain = domain_balanced_score(
            means, {d: list(suite.subtypes[d]) for d in suite.domains})
        assert ev.values["state.objective.equal_domain_mean"] == \
            pytest.approx(primary)
        assert ev.values["state.objective.worst_domain"] == \
            pytest.approx(max(per_domain.values()))
        for item in items:
            rec = per_item[item.item_id]
            assert rec["objective"] == pytest.approx(
                rec["raw_kl"] - 0.25 * rec["student_confidence"], abs=1e-6)

    def test_raw_kl_under_the_objective_reproduces_the_kl_metrics(
            self, model, candidate):
        """THE KL-ONLY CONTROL. Aggregating the raw KL the way the objective
        is aggregated gives the KL-only metrics of the same evaluation, so a
        ranking difference cannot come from how items were weighted."""
        _, ev = _evaluate(model, candidate, objective=WEIGHTED)
        v = ev.values
        assert v["state.objective.raw_kl.equal_domain_mean"] == pytest.approx(
            v["state.teacher_kl.equal_domain_mean"], rel=1e-6)
        assert v["state.objective.raw_kl.worst_domain"] == pytest.approx(
            v["state.teacher_kl.worst_domain"], rel=1e-6)
        for domain, value in ev.detail["objective"]["per_domain"][
                "raw_kl"].items():
            assert value == pytest.approx(v[f"state.teacher_kl.{domain}"],
                                          rel=1e-6)
        assert v["state.objective.raw_kl.critical_token"] == pytest.approx(
            v["state.critical_token_kl"], rel=1e-6)

    def test_item_equal_weighting_would_not_reproduce_them(self, model,
                                                           candidate):
        """The control is not vacuous: on this suite the two weightings differ."""
        _, ev = _evaluate(model, candidate, objective=WEIGHTED)
        suite, items = _suite()
        per_item = ev.detail["objective"]["per_item"]
        by_subtype: dict[str, list[float]] = {}
        for item in items:
            by_subtype.setdefault(item.subtype, []).append(
                per_item[item.item_id]["raw_kl"])
        item_equal, _ = domain_balanced_score(
            {k: sum(x) / len(x) for k, x in by_subtype.items()},
            {d: list(suite.subtypes[d]) for d in suite.domains})
        assert item_equal != pytest.approx(
            ev.values["state.teacher_kl.equal_domain_mean"], rel=1e-6)

    def test_joint_histograms_count_every_scored_position(self, model,
                                                         candidate):
        _, ev = _evaluate(model, candidate, objective=WEIGHTED)
        joint = ev.detail["objective"]["joint_histograms"]
        total = ev.detail["objective"]["scored_positions"]["total"]
        assert sum(map(sum, joint["confidence_x_kl"])) == total
        assert sum(map(sum, joint["confidence_x_topk_mass"])) == total

    def test_the_row_path_measures_the_same_objective(self, model, candidate):
        _, base = _evaluate(model, candidate, objective=WEIGHTED, batch=3)
        _, rows = _evaluate(model, candidate, objective=WEIGHTED, batch=3,
                            rows="scored_v1")
        for key, value in base.values.items():
            if key.startswith("state.objective."):
                assert rows.values[key] == pytest.approx(value, rel=1e-6,
                                                         abs=1e-9), key

    def test_the_full_vocabulary_oracle_refuses_an_objective(self, model):
        with pytest.raises(MeasurementError, match="oracle"):
            _evaluate(model, model, objective=WEIGHTED, support=FULL_VOCAB_V1)


# --- 7. one objective for the operators, the measurer and the ranking ----------

OBJECTIVE_PARETO = BeamRankingPolicy(
    policy_id="beam.toy_objective_pareto", version=1,
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


def _search(model, *, objective, policy, evaluator_objective="same"):
    suite, items = _suite()
    calib = _items()
    ev = StateEvaluator(
        suite, items, device="cpu", position_policy=SUPERVISED_TARGET_V1,
        distribution_support=reference_topk_tail(K),
        scoring_objective=(objective if evaluator_objective == "same"
                           else evaluator_objective))
    ev.prime_reference(model)
    config = SearchConfig(
        run_id="objective", target_spec=ArchSpec.of(
            "qwen3", {**GEOMETRY, "num_hidden_layers": 4}),
        schedule=SCHEDULE_V1, seed=7, workdir=Path(tempfile.mkdtemp()),
        profiles=(make_profile("balanced"),), policy=policy, suite=suite,
        position_policy=SUPERVISED_TARGET_V1,
        distribution_support=reference_topk_tail(K),
        scoring_objective=objective,
        allowed_impls=("depth.causal_kl_greedy_v1",), max_depth=1,
        device="cpu")
    return BeamSearch(
        adapter=adapter_for_config(model.config), config=config,
        root_teacher_id="toy/teacher", root_teacher_sha256="de" * 32,
        root_loader=lambda: model, calibration_loader=lambda profile: calib,
        measurer=_Measurer(ev), execution=ExecutionConfig(),
        numerics=CPU), config


class _Measurer:
    """A callable measurer that still exposes what it measures under."""

    def __init__(self, ev):
        self.ev = ev
        self.scoring_objective = ev.scoring_objective
        self.distribution_support = ev.distribution_support

    def __call__(self, model, digest):
        return self.ev.evaluate(model, digest)


class TestOneObjectiveEverywhere:

    def test_an_objective_moves_the_config_hash_and_none_does_not(self, model):
        _, with_obj = _search(model, objective=WEIGHTED, policy=OBJECTIVE_PARETO)
        base = dict(with_obj.__dict__)
        base.pop("scoring_objective")
        plain = SearchConfig(**{**base, "policy": PARETO_V1})
        assert "scoring_objective" not in plain.as_dict()
        assert with_obj.as_dict()["scoring_objective"] == WEIGHTED.as_dict()

    def test_objective_metrics_without_an_objective_are_refused(self, model):
        with pytest.raises(SearchError, match="declares no scoring objective"):
            _search(model, objective=None, policy=OBJECTIVE_PARETO)

    def test_an_objective_ranked_on_the_kl_is_refused(self, model):
        with pytest.raises(SearchError, match="must rank on"):
            _search(model, objective=WEIGHTED, policy=PARETO_V1)

    def test_a_measurer_under_another_objective_is_refused(self, model):
        with pytest.raises(SearchError, match="measurer"):
            _search(model, objective=WEIGHTED, policy=OBJECTIVE_PARETO,
                    evaluator_objective=REWARD)

    def test_the_search_runs_and_ranks_on_the_objective(self, model):
        search, _ = _search(model, objective=WEIGHTED, policy=OBJECTIVE_PARETO)
        result = search.run()
        measured = [s for s in result.states.values() if s.evaluation]
        assert measured
        for state in measured:
            assert "state.objective.equal_domain_mean" in state.evaluation.values
