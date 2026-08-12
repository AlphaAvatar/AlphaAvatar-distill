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
    E1_KD_HEAVY_0860K,
    SEED_SA,
    SEED_SB,
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
        equivalence_interval=0.02,
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


def test_selection_gates_on_stability_then_ranks_on_correctness():
    """A stable-but-wrong candidate must not outrank a correct one.

    ``usable_rollout`` is blind to correctness by construction — a terse
    contentless reply scores perfectly — so it gates and never scores.
    """
    p = plan()
    results = [
        {"state_id": "stable_wrong", "usable_rollout_rate": 0.95, "correct_overall": 0.05},
        {"state_id": "usable_right", "usable_rollout_rate": 0.60, "correct_overall": 0.25},
        {"state_id": "unstable", "usable_rollout_rate": 0.10, "correct_overall": 0.99},
    ]
    out = p.select(results, k=2)
    assert out["selected"] == ["usable_right", "stable_wrong"]
    # The high-correctness candidate that cannot hold a rollout is excluded by the
    # constraint, not silently averaged into a combined score.
    assert [e["state_id"] for e in out["excluded_by_feasibility"]] == ["unstable"]
    assert "below the preregistered feasibility floor" in \
        out["excluded_by_feasibility"][0]["reason"]


def test_the_control_survives_the_feasibility_gate_and_advances():
    p = plan()
    results = [
        {"state_id": "control", "is_control": True,
         "usable_rollout_rate": 0.10, "correct_overall": 0.18},
        {"state_id": "leaf", "usable_rollout_rate": 0.80, "correct_overall": 0.20},
    ]
    out = p.select(results, k=1)
    assert not out["excluded_by_feasibility"], "the control must never be gated out"
    assert "control" in [r["state_id"] for r in out["ranked"]]
    # ... and it does not consume a searched-candidate slot.
    assert out["selected"] == ["leaf"]


def test_candidates_inside_the_equivalence_interval_ask_for_a_third_seed():
    p = plan(equivalence_interval=0.03)
    close = [
        {"state_id": "a", "usable_rollout_rate": 0.8, "correct_overall": 0.20},
        {"state_id": "b", "usable_rollout_rate": 0.8, "correct_overall": 0.19},
    ]
    out = p.select(close, k=2)
    assert set(out["tied_within_equivalence"]) == {"a", "b"}
    assert out["needs_tie_break_seed"] is True

    separated = [
        {"state_id": "a", "usable_rollout_rate": 0.8, "correct_overall": 0.30},
        {"state_id": "b", "usable_rollout_rate": 0.8, "correct_overall": 0.10},
    ]
    assert p.select(separated, k=2)["needs_tie_break_seed"] is False


def test_thresholds_cannot_move_after_freezing(tmp_path):
    frozen_path = tmp_path / "plan.json"
    original = plan()
    original.freeze(frozen_path)
    assert assert_preregistered(original, frozen_path)["plan_hash"] == original.plan_hash
    with pytest.raises(RecoveryAdmissionError, match="after freezing"):
        assert_preregistered(plan(feasibility_min=0.40), frozen_path)
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
