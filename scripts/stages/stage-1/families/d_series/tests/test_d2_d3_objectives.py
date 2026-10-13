"""D2's and D3's objectives as registered: identities, ranking and lambda, at $0.

The forms themselves are core and tested there
(`tests/initialization/test_scoring_objective.py`). These tests pin the
instances this campaign uses and the rule that will set D3's lambda: what is
frozen now, what is refused, and that nothing about lambda is a free choice.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[6]

from stages.d_series import calibrate_d3_lambda as CAL  # noqa: E402
from stages.d_series import objectives as O  # noqa: E402

from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_KL_MINUS_CONFIDENCE_REWARD, FORM_UNCERTAINTY_WEIGHTED_KL,
)


class TestTheTwoObjectives:

    def test_d2_is_the_uncertainty_weighted_form_at_k_200(self):
        d = O.D2_OBJECTIVE.as_dict()
        assert d["form"] == FORM_UNCERTAINTY_WEIGHTED_KL
        assert d["entropy_top_k"] == 200
        assert d["denominator"] == "scored_position_count/v1"
        assert "reward_scale" not in d

    def test_d3_is_the_reward_form_and_distinct_from_d2(self):
        d3 = O.d3_objective(0.5)
        assert d3.form == FORM_KL_MINUS_CONFIDENCE_REWARD
        assert d3.objective_hash != O.D2_OBJECTIVE.objective_hash
        assert d3.objective_id != O.D2_OBJECTIVE.objective_id

    def test_d3_has_no_objective_until_lambda_is_frozen(self, tmp_path):
        with pytest.raises(RuntimeError, match="has not been measured"):
            O.frozen_d3_objective(tmp_path)

    def test_a_lambda_measured_under_another_rule_is_refused(self, tmp_path):
        path = tmp_path / O.LAMBDA_RECORD
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"rule_hash": "other", "reward_scale": 1.0}))
        with pytest.raises(RuntimeError, match="different rule"):
            O.frozen_d3_objective(tmp_path)

    def test_one_ranking_policy_ranks_only_the_objective(self):
        keys = [o.key for o in O.OBJECTIVE_PARETO_V1.objectives]
        assert keys and all(k.startswith("state.objective.") for k in keys)
        assert all(o.direction == "minimize"
                   for o in O.OBJECTIVE_PARETO_V1.objectives)
        assert set(O.OBJECTIVE_PARETO_V1.epsilon.values()) == {1e-4}


class TestTheLambdaRule:

    def test_the_rule_hash_is_stable_and_bound(self):
        a, b = O.lambda_rule(), O.lambda_rule()
        assert a["rule_hash"] == b["rule_hash"]
        assert a["formula"] == "lambda = IQR(item KL) / IQR(item confidence)"

    def test_it_is_the_ratio_of_inter_quartile_ranges(self):
        kl = [0.1 * i for i in range(1, 41)]
        conf = [0.5 + 0.01 * i for i in range(1, 41)]
        got = O.lambda_from_reference(kl, conf)
        assert got["reward_scale"] == pytest.approx(10.0, rel=1e-9)
        assert got["n_items"] == 40

    def test_it_rounds_to_three_significant_figures(self):
        kl = [0.123456 * i for i in range(1, 41)]
        conf = [0.01 * i for i in range(1, 41)]
        got = O.lambda_from_reference(kl, conf)
        assert got["reward_scale"] == 12.3
        assert got["unrounded"] == pytest.approx(12.3456, rel=1e-6)

    def test_a_heavy_tail_does_not_set_the_scale(self):
        kl = [0.1 * i for i in range(1, 41)]
        conf = [0.5 + 0.01 * i for i in range(1, 41)]
        heavy = list(kl)
        heavy[-1] = 1000.0
        assert O.lambda_from_reference(heavy, conf)["reward_scale"] == \
            O.lambda_from_reference(kl, conf)["reward_scale"]

    @pytest.mark.parametrize("kl,conf,match", [
        ([0.1] * 10, [0.5] * 10, "at least 20"),
        ([0.1 * i for i in range(30)], [0.5] * 30, "same confidence"),
        ([math.inf] + [0.1] * 29, [0.01 * i for i in range(30)], "not finite"),
    ])
    def test_refusals(self, kl, conf, match):
        with pytest.raises(ValueError, match=match):
            O.lambda_from_reference(kl, conf)

    def test_the_record_carries_the_rule_and_reports_sensitivity_apart(self):
        per_set = {
            "a": [{"item_id": f"a{i}", "kl": 0.1 * i, "confidence": 0.3 + 0.01 * i}
                  for i in range(1, 31)],
            "b": [{"item_id": f"b{i}", "kl": 0.2 * i, "confidence": 0.4 + 0.01 * i}
                  for i in range(1, 11)],
        }
        record = CAL.calibrate(per_set, identities={"student": "toy"})
        assert record["rule_hash"] == O.lambda_rule()["rule_hash"]
        assert "refused" in record["per_set_sensitivity_not_used"]["b"]
        assert record["reward_scale"] > 0


class TestTheInputsAreComputedOnTheStudent:

    def test_item_terms_on_a_tiny_pair(self):
        from support.toy import TEACHER_GEOMETRY, build_tiny_model, make_items
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.calibration.items import (
            prepare_calibration_items,
        )
        from aadistill.initialization.scoring.positions import (
            SUPERVISED_TARGET_V1,
        )

        register_builtin_adapters()
        teacher = build_tiny_model(TEACHER_GEOMETRY)
        student = build_tiny_model(TEACHER_GEOMETRY, seed=19)
        raw = make_items()
        for item in raw:
            n = int(item["input_ids"].shape[1]) - 1
            item["tags"] = {**item["tags"], "assistant": torch.arange(n) >= 5}
        items = prepare_calibration_items(raw, profile_id="toy@v1")
        rows = CAL.item_terms(teacher, student, items,
                              policy=SUPERVISED_TARGET_V1, support_top_k=8,
                              entropy_top_k=8, device="cpu")
        assert len(rows) == len(items)
        for r in rows:
            assert math.isfinite(r["kl"]) and r["kl"] >= 0
            assert 0.0 <= r["confidence"] <= 1.0
            assert r["scored_positions"] == int(
                items[0]["input_ids"].shape[1]) - 1 - 5
        #: The SAME model as student gives zero KL: the reference is the teacher.
        same = CAL.item_terms(teacher, teacher, items[:1],
                              policy=SUPERVISED_TARGET_V1, support_top_k=8,
                              entropy_top_k=8, device="cpu")
        assert same[0]["kl"] == pytest.approx(0.0, abs=1e-6)


def test_the_committed_contract_is_current():
    path = REPO / O.CONTRACT
    committed = json.loads(path.read_text())
    assert committed == json.loads(json.dumps(O.contract(REPO)))


class TestTheContractStatesWhyAndWhereTheClaimsStop:

    def test_the_four_quantities_are_kept_apart(self):
        record = O.contract(REPO)
        assert set(record["measurement_distinctions"]) == {
            "student_entropy_on_teacher_native_prefixes",
            "student_entropy_on_student_rollout_prefixes",
            "student_teacher_distribution_fit",
            "autonomous_rollout_correctness"}
        assert "teacher-native" in record["calibration_this_round"].lower() \
            or "TEACHER-NATIVE" in record["calibration_this_round"]

    def test_the_boundary_refuses_claims_the_evidence_cannot_carry(self):
        boundary = O.contract(REPO)["claim_boundary"]
        joined = " ".join(boundary["may_not_claim"])
        assert "lower D2/D3 search objective" in joined
        assert "B or q2" in joined
        assert set(boundary["not_the_goal"]) == {
            "maximizing student confidence", "minimizing KL to the teacher"}

    def test_mechanism_analyses_are_limited_to_measured_candidates(self):
        diag = O.contract(REPO)["behavioural_diagnostics"]
        for analysis in diag["analyses"]:
            if "rollout_rescoring" in analysis["instrument"]:
                assert analysis["arms"] == "new D2/D3 candidates only"
        assert "NOT available" in diag["historical_units"]

    def test_the_objective_identities_did_not_move_with_the_text(self):
        record = O.contract(REPO)
        assert record["d2"]["objective_hash"] == O.D2_OBJECTIVE.objective_hash
        assert record["revision"] == 2
