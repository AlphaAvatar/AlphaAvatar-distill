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

from stages.d_series import reference_calibration as CAL  # noqa: E402
from stages.d_series import objectives as O  # noqa: E402

from aadistill.initialization.scoring.objective import (  # noqa: E402
    FORM_KL_MINUS_CONFIDENCE_REWARD, FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
    FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL,
    FORM_UNCERTAINTY_WEIGHTED_KL, KL_REFERENCE_ROOT,
)


def _stage_l_premises() -> dict:
    """What stage L recorded it ran, from its booked, hash-verified evidence."""
    from stages.d_series import d2_budget as BUD
    ev = BUD.calibration_evidence(REPO)
    return json.loads((REPO / ev["evidence_session"]).read_text())["L_premises"]


class TestTheTwoObjectives:

    def test_d2_is_the_union_kl_form_at_full_scope(self):
        from stages.d_series import scoring_protocol as SP
        d = O.D2_OBJECTIVE.as_dict()
        assert O.D2_OBJECTIVE is O.D2_V3_OBJECTIVE
        assert d["form"] == FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL
        assert d["kl_reference"] == KL_REFERENCE_ROOT
        assert d["confidence_source"] == \
            "reference_topk_plus_tail_absolute_shannon_entropy/v1"
        assert d["confidence_top_k"] == SP.D_SERIES_TOP_K == 200
        assert d["kl_support"] == "reference_student_union_topk_tail_v1"
        assert O.D2_SUPPORT.as_dict()["top_k"] == d["confidence_top_k"]
        assert d["statistics_weighting"] != "none"
        assert d["denominator"] == "scored_position_count/v1"
        assert "reward_scale" not in d and "entropy_top_k" not in d
        assert O.D2_V3_NARROW_SCOPE_COMPARISON.objective_hash != \
            O.D2_OBJECTIVE.objective_hash

    def test_version_2_is_kept_exactly_as_stage_v_validated_it(self):
        v2 = O.D2_V2_OBJECTIVE.as_dict()
        assert v2["form"] == FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL
        assert O.D2_V2_OBJECTIVE.objective_hash.startswith("079031e3")
        assert O.D2_V2_OBJECTIVE.objective_hash != \
            O.D2_OBJECTIVE.objective_hash
        sup = O.D2_V2_SUPERSESSION
        assert sup["status"] == ("SUPERSEDED BEFORE FORMAL SEARCH -- STAGE V "
                                 "ENGINEERING VALIDATION COMPLETED")
        assert sup["executed"]["d2_search"] is False
        assert sup["contract_hash"] == O.REVISION_7_CONTRACT_HASH
        assert sup["preregistration_hash"] == O.PREREGISTRATION_V2_HASH
        plan = json.loads((REPO / "logs/stages/stage-1/families/d_series/"
                           "validations/d2-gpu-validation/v1/plan.json"
                           ).read_text())
        assert plan["objective_hash"] == O.D2_V2_OBJECTIVE.objective_hash
        assert plan["contract_hash"] == O.REVISION_7_CONTRACT_HASH

    def test_d1_and_d3_keep_the_historical_support(self):
        from stages.d_series import scoring_protocol as SP
        assert SP.D_SERIES_SUPPORT.as_dict() == {
            "support_id": "reference_topk_tail_v1", "top_k": 200}
        assert SP.D2_V3_SUPPORT.support_id != SP.D_SERIES_SUPPORT.support_id
        d3 = O.frozen_d3_objective(REPO)
        assert O.ranking_identity(d3, SP.D_SERIES_SUPPORT)[
            "ranking_identity_hash"] != O.ranking_identity(
            O.D2_OBJECTIVE, O.D2_SUPPORT)["ranking_identity_hash"]

    def test_the_superseded_d2_is_kept_exactly_as_stage_l_ran_it(self):
        old = O.D2_STUDENT_UNCERTAINTY_SUPERSEDED.as_dict()
        assert old["form"] == FORM_UNCERTAINTY_WEIGHTED_KL
        assert old["entropy_top_k"] == 200
        assert O.D2_STUDENT_UNCERTAINTY_SUPERSEDED.objective_hash == \
            _stage_l_premises()["d2_objective_hash"]
        assert O.D2_STUDENT_UNCERTAINTY_SUPERSEDED.objective_hash.startswith(
            "063bafd4")
        assert O.D2_SUPERSESSION["status"] == "SUPERSEDED -- NEVER EXECUTED"
        assert O.D2_OBJECTIVE.objective_hash != \
            O.D2_STUDENT_UNCERTAINTY_SUPERSEDED.objective_hash

    def test_d3_is_unchanged_by_the_redesign(self):
        d3 = O.frozen_d3_objective(REPO)
        assert d3.objective_hash.startswith("7b2545389874")
        assert d3.reward_scale == 7.34

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

    def test_the_record_carries_both_rules_and_reports_sensitivity_apart(self):
        rows = [{"set": "a", "item_id": f"a{i}", "content": f"c{i}",
                 "kl": 0.1 * i, "confidence": 0.3 + 0.01 * i,
                 "scored_positions": 5} for i in range(1, 31)]
        draws = ([{"set": "a", "item_id": f"a{i}", "content": f"c{i}"}
                  for i in range(1, 31)]
                 + [{"set": "b", "item_id": f"b{i}", "content": f"c{i}"}
                    for i in (1, 1, 2)])
        positions = {"kl": torch.linspace(0, 1, 101),
                     "confidence": torch.linspace(0, 1, 101),
                     "topk_mass": torch.linspace(0, 1, 101)}
        record = CAL.calibrate(rows, draws, positions,
                               identities={"student": "toy"})
        assert record["rule_hash"] == O.lambda_rule()["rule_hash"]
        assert record["threshold_rule_hash"] == \
            O.diagnostic_threshold_rule()["rule_hash"]
        assert record["independence"] == {"draws": 33, "unique_items": 30,
                                          "scored_positions": 101}
        assert "refused" in record["lambda_sensitivity_not_used"]["set:b"]
        assert record["lambda_sensitivity_not_used"]["with_duplicate_draws"][
            "n_items"] == 33
        t = record["diagnostic_thresholds"]
        assert t["high_kl"] == pytest.approx(0.9)
        assert t["low_topk_mass"] == pytest.approx(0.1)


class TestUniqueItems:

    def test_repeated_draws_count_once(self):
        a = {"item_id": "x", "input_ids": torch.tensor([[1, 2, 3]])}
        b = {"item_id": "y", "input_ids": torch.tensor([[4, 5]])}
        again = {"item_id": "x-dup", "input_ids": torch.tensor([[1, 2, 3]])}
        unique, draws, report = CAL.unique_items({"s1": [a, b],
                                                  "s2": [again, again]})
        assert report["draws"] == 4 and report["unique"] == 2
        assert report["per_set"]["s2"] == {"draws": 2, "unique_within": 1,
                                           "new_unique": 0}
        assert [n for n, _ in unique] == ["s1", "s1"]


class TestOfflineFlags:

    JOINT = {"confidence_edges": [i / 10 for i in range(11)],
             "kl_edges": [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0],
             "topk_mass_edges": [i / 10 for i in range(11)],
             "confidence_x_kl": [[1] * 10 for _ in range(10)],
             "confidence_x_topk_mass": [[1] * 10 for _ in range(10)]}

    def test_thresholds_on_the_grid_count_exactly(self):
        f = O.flag_counts(self.JOINT, {"high_confidence": 0.9, "high_kl": 1.0,
                                       "low_topk_mass": 0.2})
        #: confidence bin 9 x KL bins >= 1.0 (edges index 6 -> bins 7..9).
        assert f["high_confidence_high_kl"] == 3
        assert f["high_confidence_low_topk_mass"] == 2
        assert f["high_raw_kl"] == 30 and f["positions"] == 100

    def test_off_grid_thresholds_never_overcount(self):
        exact = O.flag_counts(self.JOINT, {"high_confidence": 0.9,
                                           "high_kl": 1.0, "low_topk_mass": 0.2})
        off = O.flag_counts(self.JOINT, {"high_confidence": 0.85,
                                         "high_kl": 0.5, "low_topk_mass": 0.25})
        assert off["high_confidence_high_kl"] <= exact[
            "high_confidence_high_kl"] + 10
        snapped = O.flag_counts(self.JOINT, {"high_confidence": 0.85,
                                             "high_kl": 0.5,
                                             "low_topk_mass": 0.25})
        #: 0.85 -> 0.9, 0.5 -> 1.0, 0.25 -> 0.2: identical to the grid case.
        assert snapped == exact


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
        rows, positions = CAL.item_terms(
            teacher, student, [("toy", i) for i in items],
            policy=SUPERVISED_TARGET_V1, support_top_k=8, entropy_top_k=8,
            device="cpu")
        assert len(rows) == len(items)
        assert int(positions["kl"].numel()) == sum(r["scored_positions"]
                                                   for r in rows)
        for r in rows:
            assert math.isfinite(r["kl"]) and r["kl"] >= 0
            assert 0.0 <= r["confidence"] <= 1.0
            assert r["scored_positions"] == int(
                items[0]["input_ids"].shape[1]) - 1 - 5
        #: The SAME model as student gives zero KL: the reference is the teacher.
        same, _ = CAL.item_terms(teacher, teacher, [("toy", items[0])],
                                 policy=SUPERVISED_TARGET_V1, support_top_k=8,
                                 entropy_top_k=8, device="cpu")
        assert same[0]["kl"] == pytest.approx(0.0, abs=1e-6)

    def test_rescoring_timing_projects_a_measured_rate(self):
        from support.toy import TEACHER_GEOMETRY, build_tiny_model, make_items
        from aadistill.initialization.calibration.items import (
            prepare_calibration_items,
        )
        teacher = build_tiny_model(TEACHER_GEOMETRY)
        student = build_tiny_model(TEACHER_GEOMETRY, seed=19)
        items = prepare_calibration_items(make_items(), profile_id="toy@v1")
        t = CAL.time_rescoring(teacher, student, [("toy", i) for i in items],
                               n=2, support_top_k=8, entropy_top_k=8)
        assert t["sequences"] == 2 and t["tokens"] == 48
        assert t["projected_minutes_per_probe"] >= 0


def test_the_committed_contract_is_current():
    path = REPO / O.CONTRACT
    committed = json.loads(path.read_text())
    assert committed == json.loads(json.dumps(O.contract(REPO)))


class TestTheContractStatesWhyAndWhereTheClaimsStop:

    def test_the_five_quantities_are_kept_apart(self):
        """Revision 7 adds the TEACHER's entropy -- D2's -- beside the
        student's -- D3's -- so neither can be read as the other's."""
        record = O.contract(REPO)
        assert set(record["measurement_distinctions"]) == {
            "student_entropy_on_teacher_native_prefixes",
            "teacher_entropy_on_teacher_native_prefixes",
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

    def test_the_record_carries_the_objectives_it_declares(self):
        record = O.contract(REPO)
        assert record["d2"]["objective_hash"] == O.D2_OBJECTIVE.objective_hash
        assert record["d2"]["version"] == 3
        assert record["revision"] == 8
        assert record["revision_history"][-1]["supersedes_contract_hash"] == \
            O.REVISION_7_CONTRACT_HASH
        assert record["revision_history"][-2]["supersedes_contract_hash"] == \
            O.REVISION_6_CONTRACT_HASH
        assert record["revision_history"][-1]["kind"] == \
            "Evidence-Informed Pre-Search Scientific Revision"
        assert record["d2_v2_superseded"]["objective_hash"] == \
            O.D2_V2_OBJECTIVE.objective_hash
        assert record["d2_v2_superseded"]["executed"]["d2_search"] is False
        assert record["d2"]["ranking_identity"] == O.ranking_identity(
            O.D2_OBJECTIVE, O.D2_SUPPORT)
        assert record["d2"]["objective"]["aggregation"] == \
            "kl_path_weighting_per_consumer/v2"
        assert record["d2_superseded"]["objective_hash"] == \
            O.D2_STUDENT_UNCERTAINTY_SUPERSEDED.objective_hash
        assert record["d2_superseded"]["executed"]["d2_search"] is False

    def test_d2_declares_three_changes_and_a_frozen_full_scope(self):
        d2 = O.contract(REPO)["d2"]
        changes = d2["declared_changes"]["changes"]
        assert len(changes) == 3
        assert "PARENT" in changes[0]["change"] and "ROOT" in changes[0]["change"]
        assert "UNITED" in changes[1]["change"]
        assert "confidence" in changes[2]["change"]
        assert d2["scope"]["frozen_scope"] == "full"
        assert len(d2["scope"]["acts_directly_on"]) == 4
        assert d2["scope"]["narrow_scope_comparison"]["objective_hash"] == \
            O.D2_V3_NARROW_SCOPE_COMPARISON.objective_hash
        joined = " ".join(d2["claim_boundary"]["may_not_claim"])
        assert "one variable" in joined
        assert "full-vocabulary KL" in joined

    def test_d2_states_its_support_and_confidence_definitions(self):
        d2 = O.contract(REPO)["d2"]
        sup, conf = d2["distribution_support"], d2["teacher_confidence"]
        assert sup["support"] == O.D2_SUPPORT.as_dict()
        assert "D1 and D3 keep" in sup["applies_to"]
        assert "KL_{S_T + tail} <= KL_U <= KL_full" in sup["bounds"]
        assert "ABSOLUTE" in conf["definition"]
        assert "201" in conf["categories"]
        assert any("exp(-H)" in n for n in conf["not"])
        assert "NOT this objective" in conf["theoretical_note_only"]

    def test_d3s_scope_names_three_places_and_excludes_three_operators(self):
        scope = O.contract(REPO)["d3"]["scope"]
        assert scope["applies_to"] == "D3 only"
        assert len(scope["acts_directly_on"]) == 3
        assert "FFN" in scope["does_not_act_on"][0]
        assert "INTERMEDIATE" in scope["intermediate_vs_final_student"]

    def test_d2_and_d3_never_share_a_confidence_signal(self):
        """Revision 7: nothing shared says the student's confidence is D2's,
        nothing D3's says the teacher's is, and the old shared keys are gone."""
        c = O.contract(REPO)
        for gone in ("objective_scope", "where_each_objective_applies",
                     "diagnostics_recorded", "d2_degeneracy_watch",
                     "scientific_motivation"):
            assert gone not in c, gone
        assert c["d3"]["motivation"]["applies_to"] == "D3 only"
        assert "D2 reads no student confidence" in \
            c["d3"]["motivation"]["role_of_confidence"]
        md = c["measurement_distinctions"]
        assert "D2 does NOT use it" in \
            md["student_entropy_on_teacher_native_prefixes"]
        assert "D2" in md["teacher_entropy_on_teacher_native_prefixes"]
        d2_text = json.dumps(c["d2"]).lower()
        assert "student's own top" not in d2_text
        assert "full scope" in json.dumps(
            c["d2"]["where_it_applies"]).lower()
        assert "not operator-local" in json.dumps(
            c["d3"]["where_it_applies"]).lower()
        shared = " ".join(c["claim_boundary"]["may_not_claim"])
        assert "one variable" not in shared and "all four operators" not in shared
        assert "one variable" in " ".join(c["d2"]["claim_boundary"]["may_not_claim"])
        assert "all four operators" in " ".join(
            c["d3"]["claim_boundary"]["may_not_claim"])

    def test_lambda_and_thresholds_precede_d2(self):
        assert "BEFORE D2" in O.lambda_rule()["when"]
        never = O.diagnostic_threshold_rule()["never"]
        assert any("ranking" in n for n in never)


class TestTheD2Budget:

    def test_it_is_derived_and_current(self):
        from stages.d_series import d2_budget as BUD
        record = json.loads((REPO / BUD.RECORD).read_text())
        assert record == json.loads(json.dumps(BUD.build()))

    def test_the_shared_protocol_field_trains_no_b_or_q2(self):
        from stages.d_series import d2_budget as BUD
        b = BUD.build(k_candidates=4)
        assert b["steps"]["d2_screening"]["n_probes"] == 8
        #: The four arms are STAGED from the relay (persisted search
        #: products), not rebuilt on the pod; the confirmation stages one.
        assert b["steps"]["d2_screening"]["n_arms_staged_from_relay"] == 4
        assert b["steps"]["d2_confirmation"]["n_arms_staged_from_relay"] == 1
        staging = b["steps"]["d2_screening"]["arm_staging"]
        assert 0 < staging["expected_minutes_per_arm"] < \
            staging["hard_minutes_per_arm"] < 22.0
        assert b["steps"]["d2_confirmation"]["n_probes"] == 3
        for k in ("d2_screening", "d2_confirmation"):
            assert b["steps"][k]["hard"]["all_in_usd"] <= b["per_session_cap_usd"]
        #: The search is reported, not assumed: with version 3's PROVISIONAL
        #: union terms its planning bound may exceed the cap, and the flags
        #: must say so rather than the cap being raised or K cut.
        search = b["steps"]["d2_search"]
        assert b["every_session_under_cap"] == (
            search["hard"]["all_in_usd"] <= b["per_session_cap_usd"])
        assert search["hard_fits_per_session_cap"] == (
            search["hard"]["all_in_usd"] <= b["per_session_cap_usd"])
        modes = search["by_prediction_rows"]
        assert modes["scored_v1"]["hard"]["all_in_usd"] <= \
            modes["all_v1"]["hard"]["all_in_usd"]
        assert search["hard"] == modes[search["prediction_rows_in_force"]][
            "hard"]

    def test_the_unmeasured_diagnostic_is_not_in_any_ceiling(self):
        from stages.d_series import d2_budget as BUD
        b = BUD.build()
        hard = sum(b["steps"][k]["hard"]["all_in_usd"]
                   for k in ("reference_calibration", "d2_search",
                             "d2_screening", "d2_confirmation"))
        assert b["totals"]["hard_ceiling_usd"] == pytest.approx(
            hard + b["steps"]["r0"]["hard"]["all_in_usd"])
        rescoring = b["steps"]["d2_screening"]["rescoring"]
        if rescoring["in_ceilings"]:
            assert rescoring["source"].startswith("measured")
            assert b["steps"]["d2_screening"][
                "rescoring_provisional_not_a_bound"]["all_in_usd"] == 0.0
        else:
            assert "only after" in b["totals"]["rescoring"]



class TestTheContractCannotContradictItsAggregation:
    """Revision 3 shipped with a stale sentence saying the state objective
    averaged items equally. The contract is generated, so the check is on the
    generator's output: no description may claim item-equal state weighting,
    and the stated aggregation must be the one the objectives carry."""

    STALE = ("per item first", "averages per item", "item-equal",
             "items equally in the state", "per-item rule")

    def _walk(self, obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from self._walk(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from self._walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            yield path, obj

    def test_no_state_evaluation_text_claims_item_equal_weighting(self):
        record = O.contract(REPO)
        offenders = [
            (path, text) for path, text in self._walk(record)
            if "state" in path.lower() and "revision_history" not in path
            and "aggregation_control/issue" not in path
            and any(s in text.lower() for s in self.STALE)]
        assert offenders == []

    def test_the_stated_aggregation_is_the_objectives(self):
        record = O.contract(REPO)
        stated = record["aggregation_difference_from_d1"][
            "matches_objective_identity_field"]
        assert stated == O.D2_OBJECTIVE.as_dict()["aggregation"]
        assert stated == O.d3_objective(1.0).as_dict()["aggregation"]


class TestTheProgrammeScenarios:
    """Arithmetic with INJECTED limits: the live ledger moves, the rules don't."""

    LIMITS = {"project_remaining_usd": 122.4554, "project_cap_usd": 590.0,
              "formal_remaining_usd": 127.9624,
              "engineering_remaining_usd": 6.651, "per_session_cap_usd": 30.0}

    def _scen(self, limits=None):
        from stages.d_series import d2_budget as BUD
        return BUD.programme_scenarios(limits or self.LIMITS)

    def test_reserves_are_counted_once_per_session(self, monkeypatch):
        from stages.d_series import d2_budget as BUD
        b = BUD.build()["steps"]
        #: A COMPLETE D2 search is inside the live limits through its closeout,
        #: so the D2 block plans only the sessions still ahead of it.
        for searched, sessions in (
                (None, ("d2_search", "d2_screening", "d2_confirmation")),
                ({"run_id": "d2_search_x"}, ("d2_screening", "d2_confirmation"))):
            monkeypatch.setattr(BUD, "completed_d2_search",
                                lambda *a, _s=searched, **k: _s)
            s = self._scen()
            block = s["search_pricing"][s["search_pricing_in_force"]]["d2_block"]
            assert block["sessions"] == sorted(sessions)
            assert block["hard_with_reserves"] == round(sum(
                b[k]["hard"]["all_in_usd"] + BUD.TEARDOWN_RESERVE_USD
                for k in sessions), 2)
            assert s["d2_search_completed"] == searched
        #: R0's and the calibration's reserves are INSIDE their grants, and a
        #: CLOSED campaign holds nothing: its recorded cost is already inside
        #: the live limits, so subtracting its grant would count it twice.
        camps = s["engineering"]["campaigns"]
        assert s["engineering"]["hard_with_reserves"] == round(sum(
            0.0 if c["closed"] else c["grant_usd"] - c["booked_usd"]
            for c in camps.values()), 2)
        for c in camps.values():
            if c["closed"]:
                assert c["outstanding_usd"] == 0.0

    def test_a_completion_sentence_does_not_release_the_search(self, tmp_path):
        """The reservation is released by the completion CONTRACT only.

        Its clauses are tested clause by clause in test_search_completion.py;
        this pins that the budget uses it: a closeout that merely SAYS
        "COMPLETE FORMAL SEARCH" releases nothing, and is reported as a
        contradiction rather than silently ignored.
        """
        import shutil

        from stages.d_series import d2_budget as BUD
        prereg = REPO / "logs/stages/stage-1/phase_d2/plans/d2_preregistration_v3.json"
        (tmp_path / prereg.relative_to(REPO)).parent.mkdir(parents=True)
        shutil.copy(prereg, tmp_path / prereg.relative_to(REPO))
        spec = BUD.d2_search_completion_spec(REPO)

        def closeout(run_id: str, classification: str) -> None:
            path = (tmp_path / "logs/stages/stage-1/phase_d2/runs" / run_id
                    / "closeout/outcome.json")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "run_id": run_id, "classification": classification,
                "money": {"all_in_usd": 5.5, "authorized_hard_usd": 23.7}}))

        assert BUD.completed_d2_search(tmp_path, spec) is None
        closeout("d2_search_a", "FAILED FORMAL SEARCH: stage C")
        assert BUD.completed_d2_search(tmp_path, spec) is None
        closeout("d2_search_b", "COMPLETE FORMAL SEARCH: 92/92 expansions")
        assert BUD.completed_d2_search(tmp_path, spec) is None
        status = BUD.d2_search_completion(tmp_path, spec)
        assert [c.split()[0] for c in status["contradictions"]] == [
            "d2_search_b"]

    def test_the_reservation_rule_is_reported_not_assumed(self):
        for pricing in self._scen()["search_pricing"].values():
            rule = pricing["reservation_rule"]
            assert rule["satisfied"] == (
                rule["left_after_d2_at_hard_usd"]
                >= rule["needed_for_d3_and_fresh_at_hard_usd"])
        rich = self._scen({**self.LIMITS, "project_remaining_usd": 500.0})
        for pricing in rich["search_pricing"].values():
            assert pricing["reservation_rule"]["satisfied"] is True
            assert pricing["reservation_rule"]["shortfall_usd"] == 0.0

    def test_every_scenario_states_whether_it_fits(self):
        s = self._scen()
        for pricing in s["search_pricing"].values():
            for scen in pricing["scenarios"].values():
                assert scen["fits_project_at_hard"] == (
                    scen["project_remaining_after_hard_usd"] >= 0)
            assert pricing["search_fits_per_session_cap"] == (
                pricing["search_hard_usd"] <= self.LIMITS["per_session_cap_usd"])
        assert s["engineering_fits_allowance"] is True

    def test_hard_ceilings_and_unverified_estimates_stay_apart(self):
        for pricing in self._scen()["search_pricing"].values():
            for scen in pricing["scenarios"].values():
                assert scen["project_remaining_after_hard_usd"] == round(
                    self.LIMITS["project_remaining_usd"]
                    - scen["hard_with_reserves_usd"], 2)
                assert scen["conservative_planning_usd"] == round(
                    scen["hard_with_reserves_usd"]
                    + scen["unverified_diagnostic_estimate_usd"], 2)

    def test_the_price_in_force_follows_the_verified_evidence(self):
        from stages.d_series import d2_budget as BUD
        s = self._scen()
        validated = BUD.calibration_evidence()["cache_fix_validated"]
        assert s["search_pricing_in_force"] == (
            "cache_fix_validated" if validated else "records_max")
        assert (s["search_pricing"]["cache_fix_validated"]["search_hard_usd"]
                < s["search_pricing"]["records_max"]["search_hard_usd"])


class TestTheSearchPriceComesFromD1sRecords:
    """D2's search priced from D1's own telemetry, verified, by D1's machinery."""

    def test_the_telemetry_is_d1s_by_its_recorded_hash(self, tmp_path, monkeypatch):
        from stages.d_series import d2_budget as BUD
        rows = BUD.d1_search_telemetry()
        assert len(rows) == 92
        fake = tmp_path / BUD.D1_TELEMETRY
        fake.parent.mkdir(parents=True)
        fake.write_bytes((REPO / BUD.D1_TELEMETRY).read_bytes() + b"\n")
        manifest = tmp_path / BUD.D1_SEARCH_RUN / "evidence/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text((REPO / BUD.D1_SEARCH_RUN /
                             "evidence/manifest.json").read_text())
        monkeypatch.setattr(BUD, "REPO_ROOT", tmp_path)
        with pytest.raises(ValueError, match="not D1's telemetry"):
            BUD.d1_search_telemetry()

    def test_the_basis_separates_what_the_objective_touches(self):
        from stages.d_series import d2_budget as BUD
        basis = BUD.telemetry_basis(BUD.d1_search_telemetry(), 551.99)
        assert basis["expansions"] == 92
        assert basis["depth"]["expansions"] == 16
        assert basis["depth"]["recomputed"] == 15
        assert basis["depth"]["candidates_per_expansion"] == [260]
        assert basis["cells_total_minutes"] < basis["session_minutes"]
        depth = basis["cells_minutes"][BUD.DEPTH_IMPL]
        assert depth["root_max"] > depth["deeper_max"] > 30

    @staticmethod
    def _evidence_repo(tmp_path, *, cache_fix=True, objective_hash=None,
                       tamper=False):
        """A tmp tree holding a booked stage-L session, as a closeout books it."""
        import hashlib
        from stages.d_series import d2_budget as BUD
        from stages.d_series import incumbent as INC
        session = {
            "status": "COMPLETE",
            "L_premises": {
                "d2_objective_hash": (
                    objective_hash
                    or O.D2_STUDENT_UNCERTAINTY_SUPERSEDED.objective_hash),
                "lambda_rule_hash": O.lambda_rule()["rule_hash"],
                "threshold_rule_hash": O.diagnostic_threshold_rule()["rule_hash"]},
            "L_reference_calibration": {
                "B": {"identity": {"artifact_digest":
                                   INC.standing_incumbent()["artifact_digest"]}},
                "calibration": {"reward_scale": 1.0},
                "timing_depth_one_round": {
                    "overhead_seconds_per_candidate": 0.2,
                    "same_cache_mode": True, "cache_fix_validated": cache_fix},
                "timing_state_eval": {"overhead_seconds_per_evaluation": 0.5}}}
        ev = tmp_path / "logs/session/adoption.json"
        ev.parent.mkdir(parents=True)
        body = json.dumps(session).encode()
        ev.write_bytes(body)
        camp = tmp_path / BUD.CALIBRATION_CAMPAIGN
        camp.parent.mkdir(parents=True)
        camp.write_text(json.dumps({"subruns": [{
            "subrun_id": "c1", "cost_usd": 0.5,
            "evidence_session": "logs/session/adoption.json",
            "evidence_session_sha256": hashlib.sha256(
                body + (b" " if tamper else b"")).hexdigest()}]}))
        return tmp_path

    def test_no_booked_session_keeps_the_records_price(self, tmp_path):
        from stages.d_series import d2_budget as BUD
        ev = BUD.calibration_evidence(tmp_path)
        assert ev["usable"] is False and ev["cache_fix_validated"] is False

    def test_verified_evidence_switches_price_and_overhead(self, tmp_path):
        from stages.d_series import d2_budget as BUD
        ev = BUD.calibration_evidence(self._evidence_repo(tmp_path))
        assert ev["usable"] and ev["cache_fix_validated"]
        basis = BUD.telemetry_basis(BUD.d1_search_telemetry(), 551.99)
        ov = BUD.objective_overhead(ev, basis)
        assert ov["source"].startswith("measured once")
        #: One timing is a measurement, not a bound: the bound has a margin.
        assert ov["depth_seconds_per_candidate"] == {
            "expected": 0.2, "bound": round(0.2 * BUD.MEASURED_OVERHEAD_BOUND_FACTOR, 4)}

    @pytest.mark.parametrize("bad", ["tamper", "objective", "no_fix"])
    def test_evidence_that_does_not_verify_changes_nothing(self, tmp_path, bad):
        from stages.d_series import d2_budget as BUD
        root = self._evidence_repo(
            tmp_path, tamper=bad == "tamper",
            objective_hash="0" * 64 if bad == "objective" else None,
            cache_fix=bad != "no_fix")
        ev = BUD.calibration_evidence(root)
        assert ev["cache_fix_validated"] is False
        if bad != "no_fix":
            assert ev["usable"] is False
            basis = BUD.telemetry_basis(BUD.d1_search_telemetry(), 551.99)
            assert BUD.objective_overhead(ev, basis)["source"].startswith(
                "provisional")

    def test_the_committed_record_carries_the_evidence_it_priced_on(self):
        from stages.d_series import d2_budget as BUD
        record = json.loads((REPO / BUD.RECORD).read_text())
        search = record["steps"]["d2_search"]
        assert search["pricing_in_force"] == (
            "cache_fix_validated"
            if search["calibration_evidence"]["cache_fix_validated"]
            else "records_max")
        assert search["cache_fix"]["validated_on_gpu"] == \
            search["calibration_evidence"]["cache_fix_validated"]
        want = (search["hard_if_cache_fix_validated"]
                if search["pricing_in_force"] == "cache_fix_validated"
                else search["hard_records_max"])
        assert search["hard"] == want

    def test_only_depth_and_state_evaluation_carry_the_overhead(self, tmp_path):
        """D3's pricing: the student-confidence overhead stage L measured."""
        from stages.d_series import d2_budget as BUD
        ev = BUD.calibration_evidence(self._evidence_repo(tmp_path))
        ev["depth"]["overhead_seconds_per_candidate"] = 0.0
        ev["state_eval"]["overhead_seconds_per_evaluation"] = 0.0
        with_zero = BUD.priced_from_records(1.09, 400, ev,
                                            overhead=BUD.objective_overhead)
        provisional = BUD.priced_from_records(1.09, 400,
                                              overhead=BUD.objective_overhead)
        assert with_zero["expected_added_minutes"] == 0.0
        assert provisional["priced"]["hard_ceiling_minutes"] > \
            with_zero["priced"]["hard_ceiling_minutes"]

    def test_the_redesigned_d2_is_never_priced_from_stage_ls_objective(
            self, tmp_path):
        """Stage L timed the SUPERSEDED objective. Even with its evidence
        verified, the redesigned D2's added work stays provisional; only
        stage L's measured round-0 candidate count is read."""
        from stages.d_series import d2_budget as BUD
        ev = BUD.calibration_evidence(self._evidence_repo(tmp_path))
        ev["depth"]["overhead_seconds_per_candidate"] = 1000.0
        ev["round0_candidates_measured"] = 36
        basis = BUD.telemetry_basis(BUD.d1_search_telemetry(), 551.99)
        #: validation=False: the PROVISIONAL model (what the registered plan
        #: uses), independent of whether an adopted Stage V now exists.
        ov = BUD.teacher_confidence_overhead(ev, basis, validation=False)
        assert ov["source"].startswith("provisional")
        assert ov["depth_seconds_per_candidate"]["bound"] < 1000.0
        assert ov["depth_seconds_per_expansion"]["round0_candidates"] == 36
        assert ov["one_time_minutes"]["profiles"] == 2
        unknown = BUD.teacher_confidence_overhead(None, basis,
                                                  validation=False)
        assert unknown["depth_seconds_per_expansion"]["round0_candidates"] \
            == 260
        d2 = BUD.priced_from_records(
            1.09, 400, ev, overhead=lambda e, b: BUD.teacher_confidence_overhead(
                e, b, validation=False))
        assert d2["priced"]["one_time_minutes"] == \
            ov["one_time_minutes"]["bound"]

    def test_the_adopted_stage_v_prices_the_redesigned_d2(self):
        """Once Stage V is booked and ADOPTED, the D2 price reads its
        measurements -- and only then."""
        from stages.d_series import d2_budget as BUD
        adopted = BUD.adopted_validation()
        if adopted is None:
            pytest.skip("no adopted Stage V in this tree")
        record = json.loads((REPO / BUD.RECORD).read_text())
        ov = record["steps"]["d2_search"]["objective_overhead"]
        w = BUD.adopted_v3_validation()
        if w is None:
            #: Version 2's terms are Stage V's measurements; version 3's
            #: union terms stay PROVISIONAL until its own GPU validation.
            assert ov["source"].startswith("version 2's terms: measured once")
            assert adopted["evidence_session"] in ov["source"]
            assert ov["union_terms"]["status"] == "PROVISIONAL -- not measured"
        else:
            #: Once stage W is ADOPTED its measurements price version 3 whole.
            assert ov["source"].startswith("measured once")
            assert w["evidence_session"] in ov["source"]
            assert ov["union_terms"]["status"] == "MEASURED (stage W, adopted)"
        #: Either way the rows a candidate reduces over are Stage V's.
        geo = ov["union_terms"]["geometry"]
        assert adopted["evidence_session"] in geo["depth_rows_per_candidate"][
            "source"]
        if w is None:
            added = ov["union_terms"]["added"]
            for key in ("depth_seconds_per_candidate",
                        "state_eval_seconds_per_evaluation",
                        "one_time_minutes"):
                assert ov[key]["bound"] >= added[key]["bound"] > 0
        else:
            #: Measured: the bound is the measurement times the margin.
            f = BUD.MEASURED_OVERHEAD_BOUND_FACTOR
            for key in ("depth_seconds_per_candidate",
                        "state_eval_seconds_per_evaluation"):
                assert ov[key]["bound"] == pytest.approx(
                    ov[key]["expected"] * f, abs=1e-3)

    def test_d3_is_priced_with_its_own_search_in_the_programme(self):
        from stages.d_series import d2_budget as BUD
        record = json.loads((REPO / BUD.RECORD).read_text())
        d3 = record["steps"]["d2_search"]["d3_search_for_programme_view"]
        assert d3["objective_overhead"]["source"].startswith("measured once")
        s = BUD.programme_scenarios(TestTheProgrammeScenarios.LIMITS)
        pricing = s["search_pricing"][s["search_pricing_in_force"]]
        assert pricing["d3_block"]["search_hard_usd"] == \
            d3["hard"]["all_in_usd"]
