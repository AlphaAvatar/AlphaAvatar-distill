"""The cost model's anchors, and the recovery admission gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
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
)
from aadistill.autoinit.operators import V1_IMPLEMENTATIONS  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    E1_KD_HEAVY_0860K,
    RecoveryAdmissionError,
    SuccessiveHalvingPlan,
    admit_leaves,
    assert_preregistered,
    probe_configs,
)
from aadistill.autoinit.state import child_state, make_root_state  # noqa: E402
from test_frozen_records import TARGET_596M, TEACHER_36  # noqa: E402

ADAPTER = get_adapter("qwen3")
DECOMPOSED = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) == 1]
COMPOSITE = [i for i in V1_IMPLEMENTATIONS if len(i.modifies) > 1]


# --- cost anchors -----------------------------------------------------------


def test_the_flop_accounting_reproduces_e8as_measured_wall_clock():
    """The anchor and the arithmetic must stay consistent with each other.

    88.83 TFLOP/s was derived *from* this accounting applied to E8a's 1,300 s
    search, so this is a regression test on the derivation: change the FLOP
    formula without re-deriving the constant and the round-trip breaks.
    """
    flops, avg_layers = greedy_depth_flops(TEACHER_36, n_remove=8, tokens=59_763,
                                           seq_len=892)
    assert avg_layers == pytest.approx(31.66, abs=0.01)
    assert L40S_MEASURED.seconds_for(flops) == pytest.approx(1300.0, rel=0.005)
    assert L40S_MEASURED.measured and not A100_80GB_ESTIMATED.measured


def test_checkpoint_and_statistics_sizes_are_the_ones_on_disk():
    # The Stage 1 init is 596,049,920 params; bf16 gives ~1.11 GiB.
    assert checkpoint_bytes(TARGET_596M, ADAPTER) == 596_049_920 * 2
    # The Stage 0 activation cache on disk is 1,947,442,680 bytes; the float64
    # second moments dominate and the formula must land on that scale.
    stats = activation_stats_bytes(TEACHER_36)
    assert 1.9e9 < stats < 2.0e9
    assert abs(stats - 1_947_442_680) / 1_947_442_680 < 0.01


def test_a_depth_only_intermediate_is_the_size_e8b_reported():
    """E8b's depth-only arm was ~3.2B; the model must price intermediates right."""
    depth_only = TEACHER_36.replace(num_hidden_layers=28)
    assert ADAPTER.param_count(depth_only) == pytest.approx(3.21e9, rel=0.01)
    assert checkpoint_bytes(depth_only, ADAPTER) / 2**30 == pytest.approx(5.98, rel=0.02)


def test_the_cpu_statistics_rate_comes_from_the_stage0_regeneration():
    assert CPU_STATS_SECONDS_PER_TOKEN == pytest.approx(4972.0 / 949_859.0)
    # 59,763 calibration positions is minutes, not seconds — which is why the
    # statistics pass is priced as a range rather than folded into the GPU time.
    assert 59_763 * CPU_STATS_SECONDS_PER_TOKEN == pytest.approx(313, rel=0.02)


# --- branching --------------------------------------------------------------


def test_the_pilot_search_space_is_the_one_the_geometry_implies():
    paths = enumerate_paths(TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED)
    # Four differing structural fields -> 24 orderings; DEPTH has two
    # implementations and the others one each -> 48 geometric paths.
    assert len(paths) == 48
    assert len({tuple(n.kind for n in p) for p in paths}) == 24
    for path in paths:
        assert len(path) == 4
        assert path[-1].child_spec.matches(TARGET_596M)


def test_branching_scales_with_beam_width_and_profiles():
    one = branching_estimate(TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED,
                             n_profiles=1, beam_width=4, include_composite=COMPOSITE)
    three = branching_estimate(TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED,
                               n_profiles=3, beam_width=4, include_composite=COMPOSITE)
    assert one["n_kinds"] == 4
    assert one["kind_orderings"] == 24
    assert one["implementations_per_field"]["num_hidden_layers"] == 2
    assert three["complete_paths_unbeamed"] == 48 * 3 ** 4
    assert three["states_materialized_min"] > one["states_materialized_min"]
    # Level 0 always has exactly one parent, whatever the beam width.
    assert one["per_level"][0]["parents"] == 1
    assert one["per_level"][1]["parents"] == 4


def test_pricing_produces_a_range_and_names_its_assumptions():
    estimate = price_search(
        TEACHER_36, TARGET_596M, ADAPTER, DECOMPOSED,
        calibration_tokens=59_763, suite_tokens=59_763, seq_len=892,
        n_profiles=1, beam_width=4, hardware=L40S_MEASURED, composite=COMPOSITE)
    assert estimate.seconds_low < estimate.seconds_high
    assert estimate.usd_low < estimate.usd_high
    # The three storage numbers are distinct and ordered: what survives pruning
    # < what is on disk at the worst moment < total bytes ever written.
    assert (estimate.peak_storage_bytes_retained
            < estimate.peak_storage_bytes_working
            < estimate.total_bytes_written)
    assert any("range" in n for n in estimate.notes)
    payload = estimate.as_dict()
    assert payload["hardware"]["measured"] is True
    assert json.dumps(payload)


def test_an_unreachable_target_is_refused_by_the_cost_model():
    wrong_vocab = TARGET_596M.replace(vocab_size=32_000)
    with pytest.raises(ValueError, match="no single-field implementation"):
        enumerate_paths(TEACHER_36, wrong_vocab, ADAPTER, DECOMPOSED)


# --- recovery gates ---------------------------------------------------------


def plan(**overrides):
    kwargs = dict(
        plan_id="autoinit.v1.pilot", recipe=E1_KD_HEAVY_0860K, top_n=4, survivors=2,
        survivor_rule="top 2 by usable_rollout_rate on seed sa, ties to lower state_id",
        winner_rule="top 1 by mean usable_rollout_rate over sa and sb",
        battery_asset_id="recovery.search_battery")
    kwargs.update(overrides)
    return SuccessiveHalvingPlan(**kwargs)


def test_the_plan_requires_two_seeds_and_stated_rules():
    with pytest.raises(ValueError, match="seed cannot rank"):
        plan(seeds=(20260726,))
    with pytest.raises(ValueError, match="must be stated before the run"):
        plan(survivor_rule="")
    with pytest.raises(ValueError, match="fewer than"):
        plan(top_n=3, survivors=3)


def test_the_probe_count_is_top_n_plus_survivors():
    assert plan(top_n=6, survivors=3).probe_count == 9
    assert plan().recipe.tokens == 860_000
    assert plan().recipe.ce_weight == 0.25 and plan().recipe.kd_weight == 1.0


def test_thresholds_cannot_move_after_freezing(tmp_path):
    frozen_path = tmp_path / "plan.json"
    original = plan()
    original.freeze(frozen_path)
    assert assert_preregistered(original, frozen_path)["plan_hash"] == original.plan_hash
    with pytest.raises(RecoveryAdmissionError, match="after freezing"):
        assert_preregistered(plan(survivors=3), frozen_path)
    with pytest.raises(RecoveryAdmissionError, match="no frozen plan"):
        assert_preregistered(original, tmp_path / "missing.json")


def leaf(teacher_spec, target_spec, name, complete=True):
    from aadistill.autoinit.metrics import StateEvaluation
    from aadistill.autoinit.state import OperatorStep

    parent = make_root_state(root_teacher_id="t", root_teacher_sha256="r",
                             spec=teacher_spec, target_spec=target_spec,
                             num_parameters=1, seed=1)
    spec = target_spec if complete else target_spec.replace(
        hidden_size=target_spec["hidden_size"] * 2)
    state = child_state(parent, OperatorStep(
        index=0, kind="DEPTH", impl_id="depth.positional_v0", impl_signature_hash="s",
        profile_id=f"{name}@v1", profile_hash=name, config_hash="c", seed=1,
        result_spec_hash="x"), spec, 1, 1)
    state.mark_materialized(f"/tmp/{name}", f"sha-{name}", "cfg")
    state.mark_validated()
    state.attach_evaluation(StateEvaluation(
        checkpoint_sha256=f"sha-{name}", suite_id="s@v1", suite_hash="h",
        reference="root_teacher", positions=1,
        values={"state.teacher_kl.equal_domain_mean": 0.1,
                "state.critical_token_kl": 0.1, "state.nll.general": 1.0}))
    return state


def test_admission_refuses_intermediates_and_short_candidate_sets(teacher_spec,
                                                                  target_spec):
    good = [leaf(teacher_spec, target_spec, f"g{i}") for i in range(4)]
    assert len(admit_leaves(good, plan())) == 4

    with_intermediate = [*good[:3], leaf(teacher_spec, target_spec, "bad",
                                         complete=False)]
    with pytest.raises(Exception, match="intermediate search state"):
        admit_leaves(with_intermediate, plan())

    with pytest.raises(RecoveryAdmissionError, match="report the shortfall"):
        admit_leaves(good[:2], plan())


def test_probe_descriptors_are_identical_except_for_the_initialization(teacher_spec,
                                                                       target_spec):
    leaves = [leaf(teacher_spec, target_spec, f"g{i}") for i in range(4)]
    configs = probe_configs(leaves, plan())
    assert len(configs) == 4
    varying = {k for c in configs for k in c
               if len({json.dumps(x[k], sort_keys=True) for x in configs}) > 1}
    assert varying == {"probe_id", "state_id", "path", "student_checkpoint",
                       "student_sha256"}
    assert {c["seed"] for c in configs} == {20260726}
    assert {c["recipe"] for c in configs} == {"e1_p1_kd_heavy@0.86M"}
