"""The corrected cost model, checked against work that has actually happened.

Phase-B attempt 3 is the first execution of this search at scale, so it is no
longer a matter of taste which reference mode the model prices. The run is a
**replay constraint**: a model that cannot reproduce work already observed cannot
be used to authorize more of it.

Observed, from `logs/autoinit_phase_b_attempt3.json` and the driver log:

* 12 `depth.causal_kl_greedy_v1` invocations completed;
* 388.2 min total, mean 32.3, range 28.5-38.4;
* the intact reference ran in `recomputed` mode — 16.9 GiB against a 13.4 GiB
  allowance, all-or-nothing, so none of it was kept;
* Stage 1 reached 544.7 min (9.08 h) and had **not** finished.

The old model priced the intact reference as a single pass — the `cached` mode —
and produced a 1.91-7.51 h range for the P=2 search. That ceiling is below the
9.08 h the run spent without completing, which is what "invalidated by hardware"
means concretely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

import price_phase_b as pb  # noqa: E402
from aadistill.autoinit.cost import (  # noqa: E402
    L40S_MEASURED,
    REFERENCE_MODES,
    conservative_hard_seconds,
    enumerate_paths,
    evaluation_cost,
    greedy_depth_flops,
    operator_cost,
    price_search,
    profile_multiplicity,
)
from aadistill.autoinit.ranking import SCHEDULE_V1  # noqa: E402

# --- what attempt 3 actually did --------------------------------------------

OBSERVED_INVOCATIONS = 12
OBSERVED_TOTAL_MIN = 388.2
OBSERVED_MEAN_MIN = 32.3
OBSERVED_MIN_MIN = 28.5
OBSERVED_MAX_MIN = 38.4
OBSERVED_STAGE1_MIN = 544.7
#: 16.9 GiB of reference against 66% of 20.3 GiB.
OBSERVED_CACHED_FRACTION = 13.4 / 16.9


def depth_costs_by_geometry(mode: str,
                            cached_fraction: float = 0.0) -> dict[tuple[int, str], float]:
    """Minutes for one causal-depth expansion at each reachable parent geometry."""
    by_impl = {i.impl_id: i for i in pb.DECOMPOSED}
    out: dict[tuple[int, str], float] = {}
    for path in enumerate_paths(pb.TEACHER, pb.TARGET, pb.ADAPTER, pb.DECOMPOSED):
        for node in path:
            if node.impl_id != "depth.causal_kl_greedy_v1":
                continue
            key = (node.level, node.parent_spec.spec_hash[:8])
            if key in out:
                continue
            cost = operator_cost(
                by_impl[node.impl_id], node.parent_spec, pb.TARGET, pb.ADAPTER,
                calibration_tokens=pb.CALIBRATION_TOKENS,
                seq_len=pb.CALIBRATION_SEQ_LEN, hardware=L40S_MEASURED,
                depth_reference_mode=mode, depth_cached_fraction=cached_fraction)
            out[key] = cost.seconds_high() / 60.0
    return out


def estimate(n_profiles: int, cached_fraction: float = 0.0):
    return price_search(
        pb.TEACHER, pb.TARGET, pb.ADAPTER, pb.DECOMPOSED,
        calibration_tokens=pb.CALIBRATION_TOKENS,
        suite_tokens=pb.CALIBRATION_TOKENS,
        seq_len=pb.CALIBRATION_SEQ_LEN, n_profiles=n_profiles,
        beam_width=SCHEDULE_V1.width, warmup_levels=SCHEDULE_V1.warmup_levels,
        hardware=L40S_MEASURED, composite=pb.COMPOSITE,
        depth_cached_fraction=cached_fraction)


# --- the modes are distinct and ordered -------------------------------------


def test_the_three_reference_modes_are_priced_differently_and_in_order():
    spec, n_remove = pb.TEACHER, 8
    tokens, seq = pb.CALIBRATION_TOKENS, pb.CALIBRATION_SEQ_LEN
    cached, _ = greedy_depth_flops(spec, n_remove, tokens, seq,
                                   reference_mode="cached")
    partial, _ = greedy_depth_flops(spec, n_remove, tokens, seq,
                                    reference_mode="partial",
                                    cached_fraction=OBSERVED_CACHED_FRACTION)
    recomputed, _ = greedy_depth_flops(spec, n_remove, tokens, seq,
                                       reference_mode="recomputed")
    assert cached < partial < recomputed
    assert set(REFERENCE_MODES) == {"cached", "partial", "recomputed"}
    # A fully-admitted partial IS the cached mode, and an empty one IS recompute.
    assert greedy_depth_flops(spec, n_remove, tokens, seq, reference_mode="partial",
                              cached_fraction=1.0)[0] == pytest.approx(cached)
    assert greedy_depth_flops(spec, n_remove, tokens, seq, reference_mode="partial",
                              cached_fraction=0.0)[0] == pytest.approx(recomputed)


def test_an_unknown_reference_mode_is_refused():
    with pytest.raises(ValueError, match="reference_mode"):
        greedy_depth_flops(pb.TEACHER, 8, 100, 10, reference_mode="guessed")


# --- the replay constraint --------------------------------------------------


def test_the_recomputed_mode_does_not_underpredict_the_observed_invocations():
    """The constraint the reviewer set: do not underpredict work already seen."""
    costs = depth_costs_by_geometry("recomputed")
    assert max(costs.values()) >= OBSERVED_MAX_MIN, (
        f"the model's most expensive causal-depth expansion is "
        f"{max(costs.values()):.1f} min, below the {OBSERVED_MAX_MIN} min actually "
        "observed; a bound that cannot reproduce the past cannot bound the future")
    # And the observed band sits inside the modelled band rather than beyond it.
    assert min(costs.values()) <= OBSERVED_MIN_MIN


def test_pricing_the_CACHED_mode_would_underpredict_every_observed_invocation():
    """Why the old range was wrong, stated as a measurement rather than a claim."""
    cached = depth_costs_by_geometry("cached")
    assert max(cached.values()) < OBSERVED_MIN_MIN, (
        f"the cached mode's most expensive expansion is {max(cached.values()):.1f} "
        f"min and the CHEAPEST invocation actually observed was "
        f"{OBSERVED_MIN_MIN} min. The old 1.91-7.51 h range was built on this mode.")


def test_the_hard_bound_exceeds_the_time_attempt_3_spent_without_finishing():
    est = estimate(2, OBSERVED_CACHED_FRACTION)
    assert est.seconds_hard / 60.0 > OBSERVED_STAGE1_MIN, (
        f"hard bound {est.seconds_hard / 60:.1f} min does not even cover the "
        f"{OBSERVED_STAGE1_MIN} min attempt 3 spent WITHOUT completing")


def test_the_old_ceiling_was_below_what_the_run_actually_spent():
    """The invalidation, pinned so it cannot quietly come back."""
    old_ceiling_hours = 7.51
    assert old_ceiling_hours * 60 < OBSERVED_STAGE1_MIN


# --- the hard bound is a bound, not an average ------------------------------


def test_the_hard_bound_is_above_the_averaged_projection():
    est = estimate(2, OBSERVED_CACHED_FRACTION)
    assert est.seconds_hard > est.seconds_high > est.seconds_low > 0


def test_children_max_times_the_mean_can_underprice_an_expensive_beam():
    """The specific defect: an average is not a bound.

    Node cost at one level varies by more than 2x with the parent's geometry, so
    a beam that retains the expensive parents costs more than `children_max`
    averages — and that product was the authorization ceiling.
    """
    by_impl = {i.impl_id: i for i in pb.DECOMPOSED}
    multiplicity = profile_multiplicity(pb.DECOMPOSED, 2)
    level_costs: dict[int, list] = {}
    for path in enumerate_paths(pb.TEACHER, pb.TARGET, pb.ADAPTER, pb.DECOMPOSED):
        for node in path:
            cost = operator_cost(
                by_impl[node.impl_id], node.parent_spec, pb.TARGET, pb.ADAPTER,
                calibration_tokens=pb.CALIBRATION_TOKENS,
                seq_len=pb.CALIBRATION_SEQ_LEN, hardware=L40S_MEASURED,
                depth_reference_mode="recomputed")
            level_costs.setdefault(node.level, []).append(cost)

    est = estimate(2, OBSERVED_CACHED_FRACTION)
    hard, per_level = conservative_hard_seconds(
        level_costs, est.branching, multiplicity)

    # The construction this replaces, rebuilt here so the comparison is explicit
    # rather than asserted: `children_max` expansions each priced at the level's
    # MEAN node cost.
    old_bound = 0.0
    for entry in est.branching["per_level"]:
        costs = level_costs.get(entry["level"])
        if not costs:
            continue
        old_bound += entry["children_max"] * (
            sum(c.seconds_high() for c in costs) / len(costs))

    assert hard > old_bound, (
        f"the conservative bound ({hard / 3600:.2f} h) is not above the "
        f"children_max x mean construction ({old_bound / 3600:.2f} h); the change "
        "would not have raised the ceiling it was meant to raise")
    assert per_level
    assert all(e["most_expensive_parent_seconds"]
               >= e["least_expensive_parent_seconds"] for e in per_level)
    # Some operators genuinely cost ~0 (a positional heuristic takes no
    # measurement), which is exactly why a per-level MEAN understates a beam that
    # retains causal-depth parents.
    cheapest = min(c.seconds_high() for costs in level_costs.values() for c in costs)
    assert cheapest == 0.0


def test_the_hard_bound_uses_the_beam_width_not_the_child_count():
    est = estimate(2, OBSERVED_CACHED_FRACTION)
    by_level = {e["level"]: e for e in est.branching["per_level"]}
    for entry in est.hard_per_level:
        assert entry["parents_max"] == by_level[entry["level"]]["parents_max"]


def test_no_calibration_operators_are_counted_ONCE_in_the_hard_bound():
    """P=2 must not duplicate `depth.positional_v0` or `attention.weight_proxy_v0`.

    They consume no mixture, so a per-profile branch would price byte-identical
    states twice and inflate the ceiling with work that never happens.
    """
    mult = profile_multiplicity(pb.DECOMPOSED, 2)
    assert mult["depth.positional_v0"] == 1
    assert mult["attention.weight_proxy_v0"] == 1
    assert mult["depth.causal_kl_greedy_v1"] == 2
    assert mult["ffn.activation_importance_v0"] == 2
    assert mult["width.global_pca_v0"] == 2


def test_the_estimate_records_which_mode_each_figure_used():
    est = estimate(2, OBSERVED_CACHED_FRACTION)
    modes = est.reference_modes
    assert modes["low"] == "cached"
    assert modes["high"] == "partial"
    assert modes["hard"] == "recomputed", (
        "the hard figure must not assume the reference fits; the bounded cache "
        "guarantees a fraction, not the whole")
    body = est.as_dict()
    assert body["hours_hard"] > body["hours_high"] > body["hours_low"]
