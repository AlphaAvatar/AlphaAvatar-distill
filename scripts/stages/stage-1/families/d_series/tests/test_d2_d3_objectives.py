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

    def test_the_record_carries_the_objectives_it_declares(self):
        record = O.contract(REPO)
        assert record["d2"]["objective_hash"] == O.D2_OBJECTIVE.objective_hash
        assert record["revision"] == 4
        assert record["d2"]["objective"]["aggregation"] == \
            "kl_path_weighting_per_consumer/v2"

    def test_the_scope_names_three_places_and_excludes_three_operators(self):
        scope = O.contract(REPO)["objective_scope"]
        assert len(scope["acts_directly_on"]) == 3
        assert "FFN" in scope["does_not_act_on"][0]
        assert "INTERMEDIATE" in scope["intermediate_vs_final_student"]

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
        assert b["steps"]["d2_screening"]["n_arms_materialized"] == 4
        assert b["steps"]["d2_confirmation"]["n_probes"] == 3
        for k in ("d2_screening", "d2_confirmation"):
            assert b["steps"][k]["hard"]["all_in_usd"] <= b["per_session_cap_usd"]
        #: The search is reported, not assumed: under D1's recompute-mode
        #: maxima it may exceed the cap, and the flag must say so.
        search = b["steps"]["d2_search"]
        assert b["every_session_under_cap"] == (
            search["hard"]["all_in_usd"] <= b["per_session_cap_usd"])
        assert search["hard_if_cache_fix_validated"]["all_in_usd"] <= \
            b["per_session_cap_usd"]

    def test_the_unmeasured_diagnostic_is_not_in_any_ceiling(self):
        from stages.d_series import d2_budget as BUD
        b = BUD.build()
        hard = sum(b["steps"][k]["hard"]["all_in_usd"]
                   for k in ("reference_calibration", "d2_search",
                             "d2_screening", "d2_confirmation"))
        assert b["totals"]["hard_ceiling_usd"] == pytest.approx(
            hard + b["steps"]["r0"]["hard"]["all_in_usd"])
        assert "only after" in b["totals"]["rescoring_excluded"]



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

    def test_reserves_are_counted_once_per_session(self):
        from stages.d_series import d2_budget as BUD
        s = self._scen()
        b = BUD.build()["steps"]
        expected = round(sum(b[k]["hard"]["all_in_usd"] + BUD.TEARDOWN_RESERVE_USD
                             for k in ("d2_search", "d2_screening",
                                       "d2_confirmation")), 2)
        assert s["search_pricing"]["records_max"]["d2_block"][
            "hard_with_reserves"] == expected
        #: R0's and the calibration's reserves are INSIDE their grants.
        assert s["engineering"]["hard_with_reserves"] == \
            round(b["r0"]["grant_proposed_usd"]
                  + b["reference_calibration"]["grant_proposed_usd"], 2)

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

    def test_the_price_in_force_is_the_unvalidated_one(self):
        s = self._scen()
        assert s["search_pricing_in_force"] == "records_max"
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

    def test_a_measured_overhead_replaces_the_provisional(self, tmp_path):
        from stages.d_series import d2_budget as BUD
        session = tmp_path / "adoption.json"
        session.write_text(json.dumps({"L_reference_calibration": {
            "timing_depth_one_round": {"overhead_seconds_per_candidate": 0.2,
                                       "same_cache_mode": True},
            "timing_state_eval": {"overhead_seconds_per_evaluation": 0.5}}}))
        basis = BUD.telemetry_basis(BUD.d1_search_telemetry(), 551.99)
        measured = BUD.objective_overhead(session, basis)
        assert measured["source"].startswith("measured")
        assert measured["depth_seconds_per_candidate"] == {"expected": 0.2,
                                                           "bound": 0.2}
        provisional = BUD.objective_overhead(None, basis)
        assert provisional["source"].startswith("provisional")
        #: A pair that did not complete on one cache mode measures nothing.
        session.write_text(json.dumps({"L_reference_calibration": {
            "timing_depth_one_round": {"overhead_seconds_per_candidate": 0.2,
                                       "same_cache_mode": False},
            "timing_state_eval": {"skipped": "deadline"}}}))
        fallback = BUD.objective_overhead(session, basis)
        assert fallback["source"].startswith("provisional")
        assert "measurement_unusable" in fallback

    def test_only_depth_and_state_evaluation_carry_the_overhead(self, tmp_path):
        from stages.d_series import d2_budget as BUD
        zero = tmp_path / "zero.json"
        zero.write_text(json.dumps({"L_reference_calibration": {
            "timing_depth_one_round": {"overhead_seconds_per_candidate": 0.0,
                                       "same_cache_mode": True},
            "timing_state_eval": {"overhead_seconds_per_evaluation": 0.0}}}))
        with_zero = BUD.priced_from_records(1.09, 400, zero)
        provisional = BUD.priced_from_records(1.09, 400)
        assert with_zero["expected_added_minutes"] == 0.0
        assert provisional["priced"]["hard_ceiling_minutes"] > \
            with_zero["priced"]["hard_ceiling_minutes"]
