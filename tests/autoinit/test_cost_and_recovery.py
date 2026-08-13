"""The cost model's anchors, the corrected branch arithmetic, and recovery gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.artifact import CheckpointIdentity, ShardRecord  # noqa: E402
from aadistill.autoinit.cost import (  # noqa: E402
    A100_80GB_ESTIMATED,
    CPU_STATS_SECONDS_PER_TOKEN,
    L40S_MEASURED,
    activation_stats_bytes,
    branching_estimate,
    checkpoint_bytes,
    enumerate_paths,
    forward_flops_per_token,
    greedy_depth_flops,
    price_search,
    profile_multiplicity,
)
from aadistill.autoinit.metrics import StateEvaluation  # noqa: E402
from aadistill.autoinit.operators import V1_IMPLEMENTATIONS  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    CATASTROPHIC_V1,
    E1_KD_HEAVY_0860K,
    PREFLIGHT_PLAN_V1,
    SEED_SA,
    SEED_SB,
    SEED_SC,
    EquivalenceRule,
    RecoveryAdmissionError,
    SuccessiveHalvingPlan,
    admit_leaves,
    assert_preregistered,
    probe_configs,
)
from aadistill.autoinit.state import child_state, make_root_state  # noqa: E402
from aadistill.autoinit.state import OperatorStep  # noqa: E402
from test_frozen_records import TARGET_596M, TEACHER_36  # noqa: E402

ADAPTER = get_adapter("qwen3")
DECOMPOSED = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) == 1]
COMPOSITE = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) > 1]


# --- cost anchors -----------------------------------------------------------


def test_the_flop_accounting_reproduces_e8as_measured_wall_clock():
    """The anchor and the arithmetic must stay consistent with each other."""
    flops, avg_layers = greedy_depth_flops(TEACHER_36, n_remove=8, tokens=59_763,
                                           seq_len=892)
    assert avg_layers == pytest.approx(31.66, abs=0.01)
    assert L40S_MEASURED.seconds_for(flops) == pytest.approx(1300.0, rel=0.005)
    assert L40S_MEASURED.measured and not A100_80GB_ESTIMATED.measured


def test_checkpoint_and_statistics_sizes_are_the_ones_on_disk():
    assert checkpoint_bytes(TARGET_596M, ADAPTER) == 596_049_920 * 2
    stats = activation_stats_bytes(TEACHER_36)
    assert abs(stats - 1_947_442_680) / 1_947_442_680 < 0.01


def test_a_depth_only_intermediate_is_the_size_e8b_reported():
    depth_only = TEACHER_36.replace(num_hidden_layers=28)
    assert ADAPTER.param_count(depth_only) == pytest.approx(3.21e9, rel=0.01)
    assert checkpoint_bytes(depth_only, ADAPTER) / 2**30 == pytest.approx(5.98, rel=0.02)


def test_the_cpu_statistics_rate_comes_from_the_stage0_regeneration():
    assert CPU_STATS_SECONDS_PER_TOKEN == pytest.approx(4972.0 / 949_859.0)
    assert 59_763 * CPU_STATS_SECONDS_PER_TOKEN == pytest.approx(313, rel=0.02)


# --- the corrected branch arithmetic ----------------------------------------


def test_no_calibration_operators_do_not_branch_over_profiles():
    """A fixed heuristic cannot be changed by a mixture, so it has one invocation."""
    for n_profiles in (1, 2, 3):
        multiplicity = profile_multiplicity(V1_IMPLEMENTATIONS, n_profiles)
        assert multiplicity["depth.positional_v0"] == 1
        assert multiplicity["attention.weight_proxy_v0"] == 1
        assert multiplicity["depth.causal_kl_greedy_v1"] == n_profiles
        assert multiplicity["width.global_pca_v0"] == n_profiles
        assert multiplicity["ffn.activation_importance_v0"] == n_profiles
        assert multiplicity["composite.stage1_sandwich_v0"] == n_profiles


@pytest.mark.parametrize("n_profiles,expected", [(1, 48), (2, 288), (3, 864)])
def test_the_decomposed_search_space_is_24_times_1_plus_p_times_p_squared(
        n_profiles, expected):
    """24 orderings x (1+P) DEPTH x P WIDTH x P FFN x 1 ATTENTION.

    Not ``48 x P^4``: branching ATTENTION and positional DEPTH over profiles
    would manufacture byte-identical states and inflate the count by a factor
    that means nothing.
    """
    estimate = branching_estimate(TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED,
                                  n_profiles=n_profiles, beam_width=6)
    assert estimate["complete_paths_unbeamed"] == expected
    assert estimate["kind_orderings"] == 24
    assert estimate["invocations_per_field"] == {
        "num_hidden_layers": 1 + n_profiles,
        "hidden_size": n_profiles,
        "intermediate_size": n_profiles,
        "num_attention_heads": 1,
    }


def test_a_warmup_level_widens_the_next_level_and_is_recorded():
    """Delayed pruning has a cost, and the model must show it rather than hide it."""
    pruning = branching_estimate(TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED,
                                 n_profiles=1, beam_width=6, warmup_levels=0)
    delayed = branching_estimate(TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED,
                                 n_profiles=1, beam_width=6, warmup_levels=1)
    assert delayed["per_level"][0]["pruned_here"] is False
    assert pruning["per_level"][0]["pruned_here"] is True
    # Level 0 has 5 children at one profile; with warmup all five are parents.
    assert delayed["per_level"][0]["children_max"] == 5
    assert delayed["per_level"][1]["parents_max"] == 5
    assert delayed["warmup_levels"] == 1


def test_pricing_produces_a_range_and_names_its_assumptions():
    estimate = price_search(
        TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED,
        calibration_tokens=59_763, suite_tokens=59_763, seq_len=892,
        n_profiles=1, beam_width=6, warmup_levels=1, hardware=L40S_MEASURED,
        composite=COMPOSITE)
    assert estimate.seconds_low < estimate.seconds_high
    assert (estimate.peak_storage_bytes_retained
            < estimate.peak_storage_bytes_working
            < estimate.total_bytes_written)
    assert any("range" in n for n in estimate.notes)
    assert json.dumps(estimate.as_dict())


def test_an_unreachable_target_is_refused_by_the_cost_model():
    wrong_vocab = TARGET_596M.replace(vocab_size=32_000)
    with pytest.raises(ValueError, match="no single-field implementation"):
        enumerate_paths(TEACHER_36, wrong_vocab, ADAPTER, DECOMPOSED)


# --- recovery gates ---------------------------------------------------------


def plan(**overrides):
    kwargs = dict(
        plan_id="autoinit.v1.pilot", recipe=E1_KD_HEAVY_0860K,
        searched_leaves=5, survivors=2, feasibility_min=0.50,
        equivalence=EquivalenceRule(n_pooled=340).materialize(
            p_pool=0.1867, p_sa=0.1867, p_sb=0.1867),
        capability_schema=None,
        survivor_rule=("feasible by usable_rollout_rate >= 0.50, then top 2 searched "
                       "leaves by correct_overall; the control advances regardless"),
        winner_rule="top 1 by mean correct_overall over sa and sb among feasible",
        battery_asset_id="recovery.search_battery")
    kwargs.update(overrides)
    return SuccessiveHalvingPlan(**kwargs)


def test_the_plan_counts_six_probes_then_three():
    p = plan()
    assert p.rung1_probes == 6      # 5 searched + 1 canonical control
    assert p.rung2_probes == 3      # 2 survivors + the control
    assert p.probe_count == 9
    assert p.seeds == (SEED_SA, SEED_SB)
    assert p.tie_break_seed is not None


def test_the_plan_requires_two_seeds_and_stated_rules():
    with pytest.raises(ValueError, match="seed cannot rank"):
        plan(seeds=(SEED_SA,))
    with pytest.raises(ValueError, match="must be stated before the run"):
        plan(survivor_rule="")
    with pytest.raises(ValueError, match="fewer than"):
        plan(searched_leaves=3, survivors=3)


def test_the_feasibility_constraint_and_the_objective_must_be_different_metrics():
    """No weighted usable+correct score, structurally."""
    with pytest.raises(ValueError, match="blind to correctness"):
        plan(primary_metric="usable_rollout_rate")
    p = plan()
    assert p.feasibility_metric == "usable_rollout_rate"
    assert p.primary_metric == "correct_overall"
    assert p.secondary_metric == "correct_given_usable"
    assert "No weighted combination" in p.as_dict()["selection"]["rule"]


def test_the_equivalence_interval_has_exactly_one_definition():
    """Formula frozen now; value materialized from the control, then immutable."""
    pending = EquivalenceRule(n_pooled=340)
    assert pending.value is None
    assert pending.as_dict()["status"] == "PENDING_CONTROL_CHARACTERIZATION"
    with pytest.raises(RecoveryAdmissionError, match="not materialized"):
        pending.require_value()

    frozen = pending.materialize(p_pool=0.1867, p_sa=0.1867, p_sb=0.1867)
    assert frozen.value == pytest.approx(2 * (0.1867 * 0.8133 / 340) ** 0.5)
    assert frozen.require_value() == frozen.value
    with pytest.raises(ValueError, match="already materialized"):
        frozen.materialize(p_pool=0.25, p_sa=0.25, p_sb=0.25)


def test_the_equivalence_interval_is_seed_aware():
    """340 prompts from one checkpoint say nothing about training-seed variation."""
    rule = EquivalenceRule(n_pooled=340)
    # Seeds agree: the binomial term governs.
    tight = rule.components(0.19, 0.19, 0.19)
    assert tight["seed_se_proxy"] == 0.0
    assert tight["value"] == pytest.approx(2 * tight["binomial_se"])
    # Seeds disagree by 0.10 -- the seed term dominates and widens the interval.
    wide = rule.components(0.19, 0.14, 0.24)
    assert wide["seed_se_proxy"] == pytest.approx(0.05)
    assert wide["value"] == pytest.approx(0.10)
    assert wide["value"] > tight["value"] * 2
    # It is a floor, so it can only widen -- never narrow.
    assert wide["value"] >= 2 * wide["binomial_se"]
    materialized = rule.materialize(p_pool=0.19, p_sa=0.14, p_sb=0.24)
    assert materialized.as_dict()["dominant_term"] == "seed_range"


def test_the_feasibility_floor_is_seed_aware_and_has_an_absolute_guard():
    from aadistill.autoinit.recovery import FeasibilityRule

    rule = FeasibilityRule(n_pooled=380)
    tight = rule.components(0.73, 0.73, 0.73)
    wide = rule.components(0.73, 0.66, 0.80)
    assert wide["value"] < tight["value"], "seed spread must lower the floor"
    # The absolute floor binds when the control is itself weak.
    weak = rule.components(0.32, 0.20, 0.44)
    assert weak["value"] == pytest.approx(0.30)
    with pytest.raises(RecoveryAdmissionError, match="not materialized"):
        rule.require_value()


def test_selection_refuses_to_run_before_the_control_is_characterized():
    """No prior fallback: a fallback would be the second definition again."""
    unmaterialized = plan(equivalence=EquivalenceRule(n_pooled=340))
    rows = [{"state_id": "a", "usable_rollout_rate": 0.8, "correct_overall": 0.2},
            {"state_id": "b", "usable_rollout_rate": 0.8, "correct_overall": 0.1}]
    with pytest.raises(RecoveryAdmissionError, match="not materialized"):
        unmaterialized.select_final_winner(rows)


# --- fail-closed capability schema ------------------------------------------


def caps(**overrides):
    from aadistill.autoinit.recovery import CAPABILITY_SCHEMA_V1

    base = {c: {"usable_rollout_rate": 0.6, "n": 30, "usable": 18}
            for c in CAPABILITY_SCHEMA_V1.expected}
    base.update(overrides)
    return base


def test_a_missing_capability_breakdown_raises_rather_than_passing():
    from aadistill.autoinit.recovery import CAPABILITY_SCHEMA_V1, CapabilitySchemaError

    p = plan(capability_schema=CAPABILITY_SCHEMA_V1)
    rows = [{"state_id": "c", "usable_rollout_rate": 0.8, "correct_overall": 0.2}]
    with pytest.raises(CapabilitySchemaError, match="no per_capability breakdown"):
        p.select_rung1_survivors(rows)


@pytest.mark.parametrize("mutation,pattern", [
    ({"tool": None}, "capability set"),
    ({"tool": {"n": 30, "usable": 18}}, "missing 'usable_rollout_rate'"),
    ({"tool": {"usable_rollout_rate": float("nan"), "n": 30, "usable": 18}},
     "NaN and Inf"),
    ({"tool": {"usable_rollout_rate": float("inf"), "n": 30, "usable": 18}},
     "NaN and Inf"),
    ({"tool": {"usable_rollout_rate": 1.5, "n": 30, "usable": 18}}, "outside"),
    ({"tool": {"usable_rollout_rate": 0.6, "usable": 18}}, "missing count 'n'"),
    ({"tool": {"usable_rollout_rate": 0.6, "n": 30, "usable": 40}}, "exceeds"),
    ({"extra_capability": {"usable_rollout_rate": 0.6, "n": 30, "usable": 18}},
     "capability set"),
])
def test_malformed_capability_metrics_fail_closed(mutation, pattern):
    from aadistill.autoinit.recovery import CAPABILITY_SCHEMA_V1, CapabilitySchemaError

    breakdown = caps()
    for key, value in mutation.items():
        if value is None:
            breakdown.pop(key)
        else:
            breakdown[key] = value
    p = plan(capability_schema=CAPABILITY_SCHEMA_V1)
    rows = [{"state_id": "c", "usable_rollout_rate": 0.8, "correct_overall": 0.2,
             "per_capability": breakdown}]
    with pytest.raises(CapabilitySchemaError, match=pattern):
        p.select_rung1_survivors(rows)


def test_no_defaults_are_invented_for_missing_capability_values():
    """'missing -> 1.0' or 'missing -> 0.0' would turn a data bug into a pass."""
    from aadistill.autoinit.recovery import CAPABILITY_SCHEMA_V1

    source = (REPO / "src/aadistill/autoinit/recovery.py").read_text()
    body = source[source.index("    def validate(self, result"):
                  source.index("    def validate_all(")]
    # `.get(` is legitimate for reading the optional label and the breakdown
    # itself; what must not appear is a defaulted *metric* read.
    assert ".get(metric" not in body and ".get(count" not in body, (
        "the validator defaults a metric read; a default is exactly what it "
        "exists to prevent")
    assert 'entry[metric]' in body and 'entry[count]' in body
    assert CAPABILITY_SCHEMA_V1.expected == (
        "gsm8k", "math_verified", "multihop", "rag", "knowledge", "tool")


# --- catastrophic per-capability gate ---------------------------------------


def test_the_catastrophic_capability_rule_is_enforced_at_both_rungs():
    """A candidate that collapses on one capability is excluded mechanically."""
    from aadistill.autoinit.recovery import CapabilitySchema

    p = plan(capability_schema=CapabilitySchema(expected=("tool", "gsm8k"),
                                                required_counts=()))
    control = {"state_id": "control", "is_control": True,
               "usable_rollout_rate": 0.75, "correct_overall": 0.19,
               "per_capability": {"tool": {"usable_rollout_rate": 0.62},
                                  "gsm8k": {"usable_rollout_rate": 0.80}}}
    collapsed = {"state_id": "collapsed", "usable_rollout_rate": 0.71,
                 "correct_overall": 0.30,
                 "per_capability": {"tool": {"usable_rollout_rate": 0.04},
                                    "gsm8k": {"usable_rollout_rate": 0.90}}}
    healthy = {"state_id": "healthy", "usable_rollout_rate": 0.70,
               "correct_overall": 0.29,
               "per_capability": {"tool": {"usable_rollout_rate": 0.55},
                                  "gsm8k": {"usable_rollout_rate": 0.78}}}
    rows = [control, collapsed, healthy]

    for out in (p.select_rung1_survivors(rows), p.select_final_winner(rows)):
        excluded = out["excluded_by_catastrophic_capability"]
        assert [e["state_id"] for e in excluded] == ["collapsed"], out["rung"]
        violation = excluded[0]["violations"][0]
        assert violation["capability"] == "tool"
        assert violation["candidate_value"] == pytest.approx(0.04)
        assert violation["control_value"] == pytest.approx(0.62)
        assert "catastrophic collapse on tool" in excluded[0]["reason"]
        # ... and it does not win despite leading the primary metric.
        assert "collapsed" not in [r["state_id"] for r in out["ranked"]]
        assert out["capability_schema_enforced"] is True

    final = p.select_final_winner(rows)
    assert final["decision_status"] == "resolved"
    assert final["winner"] == "healthy"


def test_without_a_capability_contract_the_rule_is_disabled_not_silently_passing():
    """A rule that cannot see a capability must not report a verdict on it."""
    p = plan(capability_schema=None)
    collapsed = {"state_id": "collapsed", "usable_rollout_rate": 0.71,
                 "correct_overall": 0.30,
                 "per_capability": {"tool": {"usable_rollout_rate": 0.04}}}
    control = {"state_id": "control", "is_control": True,
               "usable_rollout_rate": 0.75, "correct_overall": 0.19,
               "per_capability": {"tool": {"usable_rollout_rate": 0.62}}}
    out = p.select_rung1_survivors([control, collapsed])
    assert out["capability_schema_enforced"] is False
    assert out["excluded_by_catastrophic_capability"] == []


def test_the_catastrophic_rule_needs_the_control_to_fire():
    """No control row means no reference; the report says so rather than passing."""
    from aadistill.autoinit.recovery import CapabilitySchema

    p = plan(capability_schema=CapabilitySchema(expected=("tool",),
                                                required_counts=()))
    collapsed = {"state_id": "collapsed", "usable_rollout_rate": 0.71,
                 "correct_overall": 0.30,
                 "per_capability": {"tool": {"usable_rollout_rate": 0.04}}}
    out = p.select_rung1_survivors([collapsed])
    assert out["control_present"] is False
    assert out["excluded_by_catastrophic_capability"] == []


def test_the_rule_does_not_fire_when_the_control_is_also_weak():
    """A capability the incumbent cannot do either is not the candidate's failure."""
    from aadistill.autoinit.recovery import CapabilitySchema

    p = plan(capability_schema=CapabilitySchema(expected=("tool",),
                                                required_counts=()))
    control = {"state_id": "control", "is_control": True,
               "usable_rollout_rate": 0.75, "correct_overall": 0.19,
               "per_capability": {"tool": {"usable_rollout_rate": 0.20}}}
    candidate = {"state_id": "c", "usable_rollout_rate": 0.71, "correct_overall": 0.30,
                 "per_capability": {"tool": {"usable_rollout_rate": 0.04}}}
    out = p.select_final_winner([control, candidate])
    assert out["excluded_by_catastrophic_capability"] == []
    assert out["winner"] == "c"


def test_the_catastrophic_rule_is_part_of_the_plan_hash():
    from aadistill.autoinit.recovery import CatastrophicCapabilityRule

    base = plan()
    looser = plan(catastrophic=CatastrophicCapabilityRule(candidate_max=0.01))
    assert base.plan_hash != looser.plan_hash
    assert base.as_dict()["catastrophic_capability_rule"]["enforced_at"] == [
        "rung1", "final"]


# --- tie semantics ----------------------------------------------------------


def test_a_tie_after_two_seeds_is_pending_not_a_winner():
    p = plan(equivalence=EquivalenceRule(n_pooled=340).materialize(
        p_pool=0.20, p_sa=0.20, p_sb=0.20))
    rows = [{"state_id": "control", "is_control": True, "usable_rollout_rate": 0.73,
             "correct_overall": 0.200, "seeds": [SEED_SA, SEED_SB]},
            {"state_id": "leaf", "usable_rollout_rate": 0.80,
             "correct_overall": 0.205, "seeds": [SEED_SA, SEED_SB]}]
    out = p.select_final_winner(rows)
    assert out["decision_status"] == "tie_pending"
    assert out["winner"] is None
    assert out["provisional_leader"] == "leaf"
    assert out["needs_tie_break_seed"] is True
    assert set(out["tie_break_candidates"]) == {"control", "leaf"}


def test_a_tie_that_survives_the_third_seed_is_unresolved_not_broken():
    """No fourth seed, and no state-id tie-break to manufacture a winner."""
    p = plan(equivalence=EquivalenceRule(n_pooled=340).materialize(
        p_pool=0.20, p_sa=0.20, p_sb=0.20))
    rows = [{"state_id": "control", "is_control": True, "usable_rollout_rate": 0.73,
             "correct_overall": 0.200, "seeds": [SEED_SA, SEED_SB, SEED_SC]},
            {"state_id": "leaf", "usable_rollout_rate": 0.80,
             "correct_overall": 0.205, "seeds": [SEED_SA, SEED_SB, SEED_SC]}]
    out = p.select_final_winner(rows)
    assert out["decision_status"] == "unresolved_equivalence"
    assert out["winner"] is None
    assert out["needs_tie_break_seed"] is False
    assert out["tie_break_candidates"] == []
    assert "did not resolve a unique behavioural winner" in out["interpretation"]


def test_a_clear_lead_resolves(teacher_spec, target_spec):
    p = plan(equivalence=EquivalenceRule(n_pooled=340).materialize(
        p_pool=0.20, p_sa=0.20, p_sb=0.20))
    rows = [{"state_id": "control", "is_control": True, "usable_rollout_rate": 0.73,
             "correct_overall": 0.10, "seeds": [SEED_SA, SEED_SB]},
            {"state_id": "leaf", "usable_rollout_rate": 0.80,
             "correct_overall": 0.30, "seeds": [SEED_SA, SEED_SB]}]
    out = p.select_final_winner(rows)
    assert out["decision_status"] == "resolved"
    assert out["winner"] == "leaf"
    assert out["needs_tie_break_seed"] is False


# --- correct => usable ------------------------------------------------------


def test_correctness_is_defined_as_correct_in_a_usable_rollout():
    from aadistill.autoinit.recovery import score_recovery_row

    answered_then_looped = score_recovery_row(usable=False, scorer_correct=True)
    assert answered_then_looped["correct"] is False
    assert answered_then_looped["correct_but_unusable"] is True, (
        "the gap between 'the scorer found an answer' and 'we counted it' must be "
        "visible, not absorbed")
    assert score_recovery_row(usable=True, scorer_correct=True)["correct"] is True
    # A behaviour-only row can never be correct.
    assert score_recovery_row(usable=True, scorer_correct=True,
                              scorable=False)["correct"] is False


def test_the_scoring_contract_names_the_offending_prompt():
    from aadistill.autoinit.recovery import ScoringContractError, validate_scored_rows

    good = [{"id": "a", "usable": True, "correct": True, "scorable": True}]
    assert validate_scored_rows(good)["correct"] == 1
    with pytest.raises(ScoringContractError, match="correct => usable"):
        validate_scored_rows([{"id": "bad-1", "set": "gsm8k", "usable": False,
                               "correct": True}])
    with pytest.raises(ScoringContractError, match="behaviour-only"):
        validate_scored_rows([{"id": "bad-2", "set": "code", "usable": True,
                               "correct": True, "scorable": False}])


def test_rung1_gates_on_stability_then_ranks_on_correctness():
    """A stable-but-wrong candidate must not outrank a correct one.

    ``usable_rollout`` is blind to correctness by construction — a terse
    contentless reply scores perfectly — so it gates and never scores.
    """
    p = plan(survivors=2)
    results = [
        {"state_id": "stable_wrong", "usable_rollout_rate": 0.95, "correct_overall": 0.05},
        {"state_id": "usable_right", "usable_rollout_rate": 0.60, "correct_overall": 0.25},
        {"state_id": "unstable", "usable_rollout_rate": 0.10, "correct_overall": 0.99},
    ]
    out = p.select_rung1_survivors(results)
    assert out["selected_searched"] == ["usable_right", "stable_wrong"]
    # The high-correctness candidate that cannot hold a rollout is excluded by the
    # constraint, not silently averaged into a combined score.
    assert [e["state_id"] for e in out["excluded_by_feasibility"]] == ["unstable"]
    assert "below the preregistered feasibility floor" in \
        out["excluded_by_feasibility"][0]["reason"]


def test_the_control_advances_from_rung1_without_consuming_a_slot():
    p = plan(survivors=1)
    results = [
        {"state_id": "control", "is_control": True,
         "usable_rollout_rate": 0.10, "correct_overall": 0.99},
        {"state_id": "leaf_a", "usable_rollout_rate": 0.80, "correct_overall": 0.20},
        {"state_id": "leaf_b", "usable_rollout_rate": 0.80, "correct_overall": 0.10},
    ]
    out = p.select_rung1_survivors(results)
    assert not out["excluded_by_feasibility"], "the control must never be gated out"
    # It does not take a survivor slot even though it leads the primary metric.
    assert out["selected_searched"] == ["leaf_a"]
    assert out["auto_advanced_control"] == ["control"]
    assert out["advancing"] == ["control", "leaf_a"]


# --- the asymmetry fix ------------------------------------------------------


def test_the_canonical_control_can_win_the_final_comparison():
    """'AutoInitializer v1 did not improve on the incumbent' must be reachable.

    A final selector that excluded the control could confirm an improvement and
    could never refute one, which is not an experiment.
    """
    p = plan()
    pooled = [
        {"state_id": "control", "is_control": True, "seeds": [SEED_SA, SEED_SB],
         "usable_rollout_rate": 0.73, "correct_overall": 0.2600},
        {"state_id": "leaf_a", "seeds": [SEED_SA, SEED_SB],
         "usable_rollout_rate": 0.80, "correct_overall": 0.1500},
        {"state_id": "leaf_b", "seeds": [SEED_SA, SEED_SB],
         "usable_rollout_rate": 0.78, "correct_overall": 0.1600},
    ]
    out = p.select_final_winner(pooled)
    assert out["decision_status"] == "resolved"
    assert out["winner"] == "control"
    assert out["winner_is_control"] is True

    # And a searched leaf wins when it actually leads.
    pooled[1]["correct_overall"] = 0.3600
    better = p.select_final_winner(pooled)
    assert better["winner"] == "leaf_a"
    assert better["winner_is_control"] is False


def test_the_third_seed_is_offered_to_a_tied_control_too():
    p = plan(equivalence=EquivalenceRule(n_pooled=340).materialize(
        p_pool=0.20, p_sa=0.20, p_sb=0.20))
    close = [
        {"state_id": "control", "is_control": True, "seeds": [SEED_SA, SEED_SB],
         "usable_rollout_rate": 0.73, "correct_overall": 0.19},
        {"state_id": "leaf_a", "seeds": [SEED_SA, SEED_SB],
         "usable_rollout_rate": 0.80, "correct_overall": 0.20},
    ]
    out = p.select_final_winner(close)
    assert set(out["tied_within_equivalence"]) == {"control", "leaf_a"}
    assert out["needs_tie_break_seed"] is True
    assert "control" in out["tie_break_candidates"]
    assert out["decision_status"] == "tie_pending" and out["winner"] is None

    separated = [
        {"state_id": "control", "is_control": True, "seeds": [SEED_SA, SEED_SB],
         "usable_rollout_rate": 0.73, "correct_overall": 0.10},
        {"state_id": "leaf_a", "seeds": [SEED_SA, SEED_SB],
         "usable_rollout_rate": 0.80, "correct_overall": 0.30},
    ]
    assert p.select_final_winner(separated)["needs_tie_break_seed"] is False


# --- seed aggregation -------------------------------------------------------


def test_pooled_counts_are_not_averaged_rates():
    """The distortion the frozen definition exists to prevent."""
    from aadistill.autoinit.recovery import POOLED_COUNTS_V1

    per_seed = [
        {"seed": SEED_SA, "n": 100, "usable": 30, "correct": 12},
        {"seed": SEED_SB, "n": 100, "usable": 90, "correct": 18},
    ]
    pooled = POOLED_COUNTS_V1.pool(per_seed)
    assert pooled["n"] == 200 and pooled["usable"] == 120 and pooled["correct"] == 30
    assert pooled["correct_overall"] == pytest.approx(0.15)
    assert pooled["usable_rollout_rate"] == pytest.approx(0.60)
    assert pooled["correct_given_usable"] == pytest.approx(30 / 120)
    # Averaging the per-seed conditional rates would report 0.30 — the seed with
    # 30 usable rollouts weighted equally with the one that had 90.
    averaged = (12 / 30 + 18 / 90) / 2
    assert averaged == pytest.approx(0.30)
    assert pooled["correct_given_usable"] != pytest.approx(averaged)


def test_pooling_extends_to_a_third_seed_and_refuses_bad_input():
    from aadistill.autoinit.recovery import SEED_SC, POOLED_COUNTS_V1

    three = POOLED_COUNTS_V1.pool([
        {"seed": SEED_SA, "n": 50, "usable": 25, "correct": 10},
        {"seed": SEED_SB, "n": 50, "usable": 25, "correct": 10},
        {"seed": SEED_SC, "n": 50, "usable": 50, "correct": 10},
    ])
    assert three["seeds"] == sorted([SEED_SA, SEED_SB, SEED_SC])
    assert three["correct_given_usable"] == pytest.approx(30 / 100)

    with pytest.raises(ValueError, match="duplicate seeds"):
        POOLED_COUNTS_V1.pool([{"seed": 1, "n": 1, "usable": 1, "correct": 1}] * 2)
    with pytest.raises(ValueError, match="not an integer count"):
        POOLED_COUNTS_V1.pool([{"seed": 1, "n": 10, "usable": 0.8, "correct": 0.2}])
    with pytest.raises(ValueError, match="cannot have been scored correct"):
        POOLED_COUNTS_V1.pool([{"seed": 1, "n": 10, "usable": 2, "correct": 5}])


def test_a_candidate_with_no_usable_rollouts_has_no_conditional_accuracy():
    from aadistill.autoinit.recovery import POOLED_COUNTS_V1

    pooled = POOLED_COUNTS_V1.pool([{"seed": SEED_SA, "n": 50, "usable": 0,
                                     "correct": 0}])
    assert pooled["correct_given_usable"] is None, (
        "0.0 would make an unmeasured quantity look measured")


def test_the_aggregation_is_part_of_the_frozen_plan_identity():
    from aadistill.autoinit.recovery import SeedAggregation

    base = plan()
    other = plan(aggregation=SeedAggregation(aggregation_id="averaged_rates",
                                             version=1, description="wrong"))
    assert base.plan_hash != other.plan_hash
    assert base.as_dict()["seed_aggregation"]["aggregation_id"] == "pooled_counts"


def test_thresholds_cannot_move_after_freezing(tmp_path):
    frozen_path = tmp_path / "plan.json"
    original = plan()
    original.freeze(frozen_path)
    assert assert_preregistered(original, frozen_path)["plan_hash"] == original.plan_hash
    with pytest.raises(RecoveryAdmissionError, match="after freezing"):
        assert_preregistered(plan(feasibility_min=0.40), frozen_path)
    with pytest.raises(RecoveryAdmissionError, match="after freezing"):
        assert_preregistered(
            plan(equivalence=EquivalenceRule(n_pooled=340).materialize(
                p_pool=0.25, p_sa=0.25, p_sb=0.25)),
            frozen_path)
    with pytest.raises(RecoveryAdmissionError, match="no frozen plan"):
        assert_preregistered(original, tmp_path / "missing.json")


def _artifact(name):
    return CheckpointIdentity(
        path=f"/tmp/{name}",
        shards=(ShardRecord("model.safetensors", f"sha-{name}", 10),),
        config_sha256="cfg", arch_signature="arch", num_parameters=1)


def leaf(teacher_spec, target_spec, name, complete=True, control=False):
    parent = make_root_state(root_teacher_id="t", root_teacher_sha256="r",
                             spec=teacher_spec, target_spec=target_spec,
                             num_parameters=1, seed=1)
    spec = target_spec if complete else target_spec.replace(
        hidden_size=target_spec["hidden_size"] * 2)
    state = child_state(parent, OperatorStep(
        index=0, kind="DEPTH", impl_id="depth.positional_v0", impl_signature_hash="s",
        profile_id=f"{name}@v1", profile_hash=name, config_hash="c", seed=1,
        result_spec_hash="x"), spec, 1, 1)
    art = _artifact(name)
    state.mark_materialized(art)
    state.mark_validated()
    state.attach_evaluation(StateEvaluation(
        artifact_digest=art.artifact_digest, suite_id="s@v1", suite_hash="h",
        reference="root_teacher", positions=1,
        values={"state.teacher_kl.equal_domain_mean": 0.1,
                "state.teacher_kl.worst_domain": 0.1,
                "state.critical_token_kl": 0.1}))
    if control:
        state.provenance = "retained_canonical"
    return state


def test_admission_refuses_intermediates_and_short_candidate_sets(teacher_spec,
                                                                  target_spec):
    searched = [leaf(teacher_spec, target_spec, f"g{i}") for i in range(5)]
    control = leaf(teacher_spec, target_spec, "canonical", control=True)
    assert len(admit_leaves([*searched, control], plan())) == 6

    with_intermediate = [*searched[:4],
                         leaf(teacher_spec, target_spec, "bad", complete=False),
                         control]
    with pytest.raises(Exception, match="intermediate search state"):
        admit_leaves(with_intermediate, plan())

    with pytest.raises(RecoveryAdmissionError, match="report the shortfall"):
        admit_leaves([*searched[:3], control], plan())


def test_admission_refuses_a_run_with_no_canonical_control(teacher_spec, target_spec):
    """A re-executed composite is not the retained incumbent."""
    searched = [leaf(teacher_spec, target_spec, f"g{i}") for i in range(5)]
    with pytest.raises(RecoveryAdmissionError, match="canonical control"):
        admit_leaves(searched, plan())


def test_probe_descriptors_are_identical_except_for_the_initialization(teacher_spec,
                                                                       target_spec):
    candidates = [leaf(teacher_spec, target_spec, f"g{i}") for i in range(5)]
    candidates.append(leaf(teacher_spec, target_spec, "canonical", control=True))
    configs = probe_configs(candidates, plan(), rung=1)
    assert len(configs) == 6
    varying = {k for c in configs for k in c
               if len({json.dumps(x[k], sort_keys=True) for x in configs}) > 1}
    assert varying == {"probe_id", "state_id", "path", "student_checkpoint",
                       "student_artifact_digest", "student_single_shard_sha256",
                       "is_control"}
    assert {c["seed"] for c in configs} == {SEED_SA}
    assert {c["recipe"] for c in configs} == {"e1_p1_kd_heavy@0.86M"}
    assert sum(c["is_control"] for c in configs) == 1

    rung2 = probe_configs(candidates[:3], plan(), rung=2)
    assert {c["seed"] for c in rung2} == {SEED_SB}


# --- recovery identity: protocol vs probe -----------------------------------


def _protocol(**overrides):
    from aadistill.autoinit.recovery import RecoveryProtocolFingerprint

    base = dict(
        pack="ladder_uniform_probe", pack_blocks_sha256="6f324cb0", rung=860_000,
        train_blocks=682, train_supervised_tokens=864_750, block_len=8192,
        packing="ladder", val_blocks=16, block_ordering="ladder order",
        ce_weight=0.25, kd_weight=1.0, kd_temperature=1.0, kd_scope="all",
        kd_chunk=512, optimizer="AdamW", lr=5e-5, weight_decay=0.01,
        betas=(0.9, 0.95), eps=1e-8, grad_clip=1.0, total_steps=1023,
        warmup_steps=51, min_lr_frac=0.1, lr_schedule="cosine",
        blocks_per_step=2, micro_blocks=1, dtype="float32", autocast_bf16=True,
        gradient_checkpointing=True, trainable_patterns=("a",),
        trainable_params=440_467_456, teacher_id="Qwen/Qwen3-4B-Thinking-2507",
        teacher_revision="768f209d", teacher_dtype="bfloat16", teacher_attn="sdpa",
        tokenizer_sha256="7781771a", resume_semantics="v2",
        trainer_source_digest="abc123", trainer_source_set_version=1,
        runtime_digest="rt-1")
    base.update(overrides)
    return RecoveryProtocolFingerprint(**base)


def test_the_protocol_identity_excludes_the_treatment_and_the_replicate():
    """Initialization and seed are variables, not part of what must be identical."""
    protocol = _protocol()
    fields = set(protocol.identity())
    for excluded in ("student_init_path", "student_init_sha256", "seed",
                     "initialization_artifact_digest"):
        assert excluded not in fields, (
            f"{excluded} is in the protocol identity; every comparable pair of arms "
            "would then be 'mismatched' by construction")
    assert protocol.as_dict()["excluded_by_design"] == [
        "student initialization artifact", "seed"]


def test_a_matched_pair_is_same_protocol_same_seed_different_init():
    from aadistill.autoinit.recovery import RecoveryProbeIdentity

    protocol = _protocol()
    control = RecoveryProbeIdentity(protocol=protocol,
                                    initialization_artifact_digest="canonical",
                                    seed=SEED_SA, label="control-sa")
    searched = RecoveryProbeIdentity(protocol=protocol,
                                     initialization_artifact_digest="searched-A",
                                     seed=SEED_SA, label="searched-A-sa")
    verdict = control.matched_against(searched)
    assert verdict["is_single_variable_comparison"]
    assert verdict["protocol_identical"] and verdict["same_seed"]
    assert verdict["initializations_differ"]
    assert control.probe_id != searched.probe_id


@pytest.mark.parametrize("mutation,reason", [
    ({"seed": SEED_SB}, "seeds differ"),
    ({"init": "canonical"}, "initializations are identical"),
    ({"protocol": "different"}, "protocol differs"),
])
def test_a_pair_that_is_not_single_variable_is_refused(mutation, reason):
    from aadistill.autoinit.recovery import RecoveryProbeIdentity

    protocol = _protocol()
    control = RecoveryProbeIdentity(protocol=protocol,
                                    initialization_artifact_digest="canonical",
                                    seed=SEED_SA, label="control-sa")
    other = RecoveryProbeIdentity(
        protocol=_protocol(lr=1e-4) if mutation.get("protocol") else protocol,
        initialization_artifact_digest=mutation.get("init", "searched-A"),
        seed=mutation.get("seed", SEED_SA), label="other")
    verdict = control.matched_against(other)
    assert not verdict["is_single_variable_comparison"]
    assert reason in verdict["verdict"]


# --- trainer source digest --------------------------------------------------


def test_a_docs_only_change_does_not_move_the_trainer_digest():
    """Whole-repo HEAD must not be the material identity."""
    from aadistill.autoinit.recovery import (
        TRAINER_SOURCE_FILES_V1, trainer_source_digest)

    first = trainer_source_digest(REPO)
    second = trainer_source_digest(REPO)
    assert first["digest"] == second["digest"]
    assert len(first["files"]) == len(TRAINER_SOURCE_FILES_V1)
    # The declared set is the trainer, not the repository.
    covered = {e["path"] for e in first["files"]}
    assert "src/aadistill/training/train.py" in covered
    assert "src/aadistill/data/ladder.py" in covered
    assert not any(p.startswith("logs/") or p.startswith("docs/") for p in covered), (
        "documentation is in the trainer digest; a docs commit would invalidate a "
        "control")


def test_the_scoring_contract_covers_the_composition_not_one_scorer_file():
    """v1 pinned `capability.py` alone, which is how the defect hid."""
    from aadistill.autoinit.recovery import (
        RECOVERY_SCORING_FILES_V2, recovery_scoring_contract)

    contract = recovery_scoring_contract(REPO)
    assert contract["contract"] == "recovery_search_scoring@v2"
    assert len(contract["digest"]) == 64
    covered = {e["path"] for e in contract["files"]}
    for required in ("scripts/autoinit/score_recovery_search.py",
                     "src/aadistill/evaluation/usable_rollout.py",
                     "src/aadistill/evaluation/strict_answer.py",
                     "src/aadistill/evaluation/behavior.py",
                     "src/aadistill/evaluation/capability.py",
                     # the rule relating two numbers is part of the metric
                     "src/aadistill/autoinit/recovery.py"):
        assert required in covered, required
    assert covered == set(RECOVERY_SCORING_FILES_V2)
    assert contract["supersedes"]["contract"] == "recovery_search_scoring@v1"
    # Same failure mode as the trainer digest: never a smaller contract.
    with pytest.raises(RecoveryAdmissionError, match="is missing"):
        recovery_scoring_contract(
            REPO, files=("src/aadistill/evaluation/capability.py",
                         "src/aadistill/evaluation/does_not_exist.py"))
    # A change anywhere in the set moves the digest.
    subset = recovery_scoring_contract(
        REPO, files=tuple(f for f in RECOVERY_SCORING_FILES_V2
                          if not f.endswith("recovery.py")))
    assert subset["digest"] != contract["digest"]


def test_the_preregistration_binds_the_scoring_contract_and_supersession():
    path = REPO / "logs/autoinit_phase_a_preregistration.json"
    if not path.is_file():
        pytest.skip("preregistration not present")
    from aadistill.autoinit.recovery import recovery_scoring_contract

    prereg = json.loads(path.read_text())
    contract = prereg["recovery_scoring_contract"]
    assert contract["digest"] == recovery_scoring_contract(REPO)["digest"], (
        "the preregistration is bound to a scoring contract that no longer "
        "matches the code; re-emit it")
    sup = contract["supersession"]
    assert sup["superseded"] == "recovery_search_scoring@v1"
    assert "before any paid measurement" in sup["statement"]
    assert "NOT an adaptive response" in sup["classification"]
    assert sup["prompt_content_unchanged"] is True
    assert sup["evidence"]["tool_usable_rate"] == {"before": 0.0, "after": 1.0}
    assert contract["validation"]["all_checks_pass"] is True
    gate = contract["tool_usable_gate"]
    assert "tool_args_schema_ok" not in gate["definition"]
    assert "tool_call_exact_match" not in gate["definition"]
    assert gate["worked_examples"][
        "well-formed declared call, wrong arguments"] == "usable, incorrect"


def test_a_missing_trainer_source_file_raises_rather_than_shrinking_the_digest():
    from aadistill.autoinit.recovery import trainer_source_digest

    with pytest.raises(RecoveryAdmissionError, match="is missing"):
        trainer_source_digest(REPO, files=("src/aadistill/training/train.py",
                                           "src/aadistill/does_not_exist.py"))


# --- runtime environment ----------------------------------------------------


def test_the_runtime_fingerprint_needs_more_than_a_torch_version():
    from aadistill.autoinit.recovery import RuntimeEnvironmentFingerprint

    rt = RuntimeEnvironmentFingerprint.observe(image_digest="sha256:deadbeef")
    payload = rt.as_dict()
    for field in ("image_digest", "python_version", "torch_version",
                  "transformers_version", "cuda_runtime", "attention_backend"):
        assert field in payload
    assert len(rt.digest) == 64
    # An unpinned runtime cannot back a permanent control.
    with pytest.raises(RecoveryAdmissionError, match="image digest"):
        RuntimeEnvironmentFingerprint.observe().require_pinned()
    # A different image is a different runtime.
    other = RuntimeEnvironmentFingerprint.observe(image_digest="sha256:0000")
    assert other.digest != rt.digest


# --- staged preflight -------------------------------------------------------


def test_permanent_controls_cannot_be_trained_before_the_gates_pass():
    """The ordering exists so a failed gate does not cost $2.80 of dead controls."""
    from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1

    plan_ = PREFLIGHT_PLAN_V1
    assert [s.stage for s in plan_.stages] == [0, 1, 2, 3]
    assert [s.blocking for s in plan_.stages] == [True, True, True, False]

    with pytest.raises(RecoveryAdmissionError, match="no recorded result"):
        plan_.advance_to(2, {0: {"passed": True}})
    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        plan_.advance_to(2, {0: {"passed": True},
                             1: {"passed": False, "reason": "peak memory 46 GiB"}})
    plan_.advance_to(2, {0: {"passed": True}, 1: {"passed": True}})
    # Characterization additionally needs the controls to exist.
    with pytest.raises(RecoveryAdmissionError, match="stage 2"):
        plan_.advance_to(3, {0: {"passed": True}, 1: {"passed": True},
                             2: {"passed": False, "reason": "fingerprint mismatch"}})


def test_the_preflight_plan_is_hashable_and_orders_money_last():
    from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1

    assert len(PREFLIGHT_PLAN_V1.plan_hash) == 64
    names = [s.name for s in PREFLIGHT_PLAN_V1.stages]
    assert names.index("runtime attestation") < names.index("cheap machine gates")
    assert names.index("cheap machine gates") < names.index(
        "permanent canonical controls")
    assert "epsilon" in " ".join(PREFLIGHT_PLAN_V1.stages[1].stop_conditions)


# --- legacy control status --------------------------------------------------


def test_the_historical_controls_are_not_labelled_as_valid_controls():
    audit_path = REPO / "logs/autoinit_recovery_fingerprint_audit.json"
    if not audit_path.is_file():
        pytest.skip("fingerprint audit not present")
    audit = json.loads(audit_path.read_text())
    assert audit["historical_controls_are_recipe_matched"] is False
    for name, entry in audit["comparisons"].items():
        assert entry["recipe_matched_control"] is False, name
        assert entry["passes_legacy_lineage_subset"] is True, name
        # The misreadable field is gone.
        assert "matches_intended_control_protocol" not in entry, name
        assert entry["comparison"]["unverifiable_fields"], (
            "the mismatch report must stay attached to the status")
    # The intended Phase-A pair is NOT claimed matched while the runtime is
    # unknown, and IS matched once the environment fields carry real values.
    intended = audit["intended_phase_a_comparison"]
    pending = intended["pending_materialization"]["check"]
    assert pending["is_single_variable_comparison"] is False, (
        "the preregistration claims a MATCHED comparison while runtime_digest is "
        "still null on both sides")
    assert "runtime_digest" in pending["unmaterialized_fields"]
    assert intended["after_stage_0_attestation"]["check"][
        "is_single_variable_comparison"]


def test_the_availability_report_does_not_claim_a_matched_control():
    """Lineage verification is a strict subset of protocol matching."""
    path = REPO / "logs/autoinit_control_availability.json"
    if not path.is_file():
        pytest.skip("availability report not present")
    report = json.loads(path.read_text())
    assert report["any_recipe_matched_control"] is False
    for name, entry in report["controls"].items():
        assert "matches_intended_control_protocol" not in entry, name
        assert entry["recipe_matched_control"] is False, name
        assert entry["artifact_available"] is True, name
        assert entry["hash_verified"] is True, name
        assert entry["passes_legacy_lineage_subset"] is True, name
    blob = json.dumps(report).lower()
    assert "no recovery retraining is needed" not in blob, (
        "stale prose still says the historical checkpoints make retraining "
        "unnecessary")


# --- materialization: unknown is not identical ------------------------------


def test_unknown_on_both_sides_is_never_reported_as_matched():
    """`None == None` is True in Python and false as a scientific claim."""
    unknown_a = _protocol(runtime_digest=None)
    unknown_b = _protocol(runtime_digest=None)
    comparison = unknown_a.compare(unknown_b)
    assert "runtime_digest" not in comparison["matched_fields"]
    assert comparison["unmaterialized_fields"] == ["runtime_digest"]
    assert comparison["both_materialized"] is False
    assert not comparison["protocol_identical"]
    assert any(u["field"] == "runtime_digest"
               for u in comparison["unverifiable_fields"])
    # Verified identical digests, by contrast, do match.
    known = _protocol(runtime_digest="rt-1").compare(_protocol(runtime_digest="rt-1"))
    assert "runtime_digest" in known["matched_fields"]
    assert known["both_materialized"] and known["protocol_identical"]


@pytest.mark.parametrize("missing", [
    "trainer_source_digest", "trainer_source_set_version", "runtime_digest"])
def test_an_unmaterialized_protocol_cannot_produce_a_matched_pair(missing):
    from aadistill.autoinit.recovery import RecoveryProbeIdentity

    protocol = _protocol(**{missing: None})
    assert protocol.is_materialized is False
    assert missing in protocol.unmaterialized_fields()
    with pytest.raises(RecoveryAdmissionError, match="not materialized"):
        protocol.require_materialized()

    control = RecoveryProbeIdentity(protocol=protocol,
                                    initialization_artifact_digest="canonical",
                                    seed=SEED_SA, label="control-sa")
    searched = RecoveryProbeIdentity(protocol=protocol,
                                     initialization_artifact_digest="searched-A",
                                     seed=SEED_SA, label="searched-A-sa")
    verdict = control.matched_against(searched)
    # Everything else about the pair is right; materialization alone blocks it.
    assert verdict["same_seed"] and verdict["initializations_differ"]
    assert verdict["protocols_materialized"] is False
    assert verdict["is_single_variable_comparison"] is False
    assert "NOT ELIGIBLE FOR MATCHED" in verdict["verdict"]
    assert missing in verdict["verdict"]


def test_materialization_fills_only_what_was_unknown_and_refuses_drift():
    from aadistill.autoinit.recovery import RuntimeEnvironmentFingerprint

    runtime = RuntimeEnvironmentFingerprint.observe(image_digest="sha256:aa")
    trainer = {"digest": "trainer-1", "set_version": 1}
    blank = _protocol(runtime_digest=None, trainer_source_digest=None,
                      trainer_source_set_version=None)
    attested = blank.materialized(runtime=runtime, trainer_source=trainer)
    assert attested.is_materialized
    assert attested.runtime_digest == runtime.digest
    assert attested.trainer_source_digest == "trainer-1"
    attested.require_materialized()

    # Idempotent under the same attestation.
    assert attested.materialized(runtime=runtime,
                                 trainer_source=trainer).fingerprint == \
        attested.fingerprint
    # A contradicting attestation is protocol drift, not an overwrite.
    with pytest.raises(RecoveryAdmissionError, match="contradicts the preregistered"):
        attested.materialized(runtime=runtime,
                              trainer_source={"digest": "trainer-2", "set_version": 1})
    # An unpinned runtime cannot materialize anything.
    with pytest.raises(RecoveryAdmissionError, match="image digest"):
        blank.materialized(runtime=RuntimeEnvironmentFingerprint.observe(),
                           trainer_source=trainer)


def test_stage_2_compares_against_the_attested_hash_not_the_preregistered_one():
    from aadistill.autoinit.recovery import (
        RecoveryProbeIdentity, RuntimeEnvironmentFingerprint)

    preregistered = _protocol(runtime_digest=None)
    attested = preregistered.materialized(
        runtime=RuntimeEnvironmentFingerprint.observe(image_digest="sha256:aa"),
        trainer_source={"digest": "abc123", "set_version": 1})

    ran_attested = RecoveryProbeIdentity(
        protocol=attested, initialization_artifact_digest="canonical",
        seed=SEED_SA, label="control-sa")
    ran_attested.require_attested(attested.fingerprint)

    # A control trained under an unpinned runtime is refused even though its
    # non-environment fields are identical to the preregistered protocol.
    ran_unpinned = RecoveryProbeIdentity(
        protocol=preregistered, initialization_artifact_digest="canonical",
        seed=SEED_SA, label="control-sa-unpinned")
    with pytest.raises(RecoveryAdmissionError, match="not materialized"):
        ran_unpinned.require_attested(attested.fingerprint)

    # A control that ran a different protocol is refused.
    other = attested.materialized  # noqa: F841  (keep the attested one in scope)
    drifted = RecoveryProbeIdentity(
        protocol=_protocol(lr=1e-4, runtime_digest=attested.runtime_digest),
        initialization_artifact_digest="canonical", seed=SEED_SA, label="drifted")
    with pytest.raises(RecoveryAdmissionError, match="attested Phase-A protocol"):
        drifted.require_attested(attested.fingerprint)


def test_stage_0_produces_the_frozen_protocol_artifact_before_controls():
    """The handshake is in the plan, not only in prose."""
    from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1

    stage0 = PREFLIGHT_PLAN_V1.stages[0]
    produced = " ".join(stage0.produces)
    assert "materialized RecoveryProtocolFingerprint" in produced
    assert "autoinit_phase_a_protocol_attested.json" in produced
    stage2 = PREFLIGHT_PLAN_V1.stages[2]
    conditions = " ".join(stage2.stop_conditions)
    assert "ATTESTED" in conditions and "not materialized" in conditions


def test_every_preflight_stop_condition_is_a_whole_string():
    """A missing trailing comma serializes as one entry per character."""
    from aadistill.autoinit.recovery import PreflightStage

    stage3 = PREFLIGHT_PLAN_V1.stages[3].as_dict()
    assert len(stage3["stop_conditions"]) == 1
    assert stage3["stop_conditions"] == [
        "capability schema validation fails -> scoring defect, STOP"]
    for stage in PREFLIGHT_PLAN_V1.as_dict()["stages"]:
        for group in ("produces", "stop_conditions"):
            for item in stage[group]:
                assert len(item) > 1, (stage["name"], group, item)
    # And the shape is refused at construction, so it cannot recur.
    with pytest.raises(TypeError, match="trailing comma"):
        PreflightStage(stage=9, name="x", purpose="y", produces=("a",),
                       blocking=False, stop_conditions="one condition")
