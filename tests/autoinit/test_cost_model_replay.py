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


def test_an_average_over_parents_is_below_the_bound_over_the_worst_parents():
    """The specific defect the hard bound replaces: an average is not a bound.

    Node cost at one level varies by more than 2x with the parent's geometry, and
    some operators cost nothing at all (a positional heuristic takes no
    measurement), so a beam that retains the expensive parents costs more than
    `parents_max` averages. Rebuilt here from the estimate's own per-level report
    so the comparison is explicit rather than asserted.
    """
    est = estimate(2, OBSERVED_CACHED_FRACTION)
    averaged = sum(e["parents_max"] * e["mean_parent_seconds_high"]
                   for e in est.per_operator)
    worst = sum(e["parents_max"] * e["max_parent_seconds_high"]
                for e in est.per_operator)
    assert averaged < worst, (
        "the level means and maxima coincide, so this fixture no longer exercises "
        "the spread the bound exists to handle")
    assert est.seconds_hard > averaged
    # And at least one level really does have that spread.
    assert any(e["max_parent_seconds_high"] > 1.5 * e["mean_parent_seconds_high"]
               for e in est.per_operator)

    # WITHIN the hard world, not across two of them. A first mutation pass
    # replaced the per-parent maxima with the level mean and this test still
    # passed, because `averaged` above is built from the `high` world while
    # `seconds_hard` is built from the `hard` one — two different models, so the
    # comparison could never detect the substitution.
    spread_levels = [e for e in est.hard_per_level
                     if e["most_expensive_parent_seconds"] > e["least_expensive_parent_seconds"]]
    assert spread_levels, "no level has a parent-cost spread to bound"
    for entry in spread_levels:
        assert entry["chosen_mean_seconds"] >= entry["mean_parent_seconds"] - 1e-9, entry
    # Strictly greater exactly where the beam cannot hold every geometry — and
    # NOT where it can, because a beam that holds all of them legitimately costs
    # their sum. Level 2 has six distinct geometries and a width of six.
    selective = [e for e in spread_levels
                 if e["distinct_parent_geometries"] != e["parents_max"]]
    assert selective, "no level is narrower than its geometry; nothing to select"
    for entry in selective:
        assert entry["chosen_mean_seconds"] > entry["mean_parent_seconds"], (
            f"level {entry['level']}: the bound averages "
            f"{entry['chosen_mean_seconds']:.1f} s per parent against a level mean "
            f"of {entry['mean_parent_seconds']:.1f} s — it is selecting the mean, "
            "not the most expensive admissible beam")


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


# --- composite is priced, and statistics are charged once -------------------
#
# These two corrections were found together and must be tested together: the
# first version omitted composite entirely while double-counting FFN/WIDTH
# statistics, and the two errors partly cancelled in the total. A test that
# checked only the total would have passed against both defects.


def test_composite_leaves_are_priced_not_merely_counted():
    """`composite.stage1_sandwich_v0` reached the state counts and nothing else.

    Each P=2 composite leaf consumes activation statistics, transforms,
    materializes, reloads, validates and is evaluated. Pricing it at zero while
    reporting it in `leaves_max` is a bill that does not mention an item.
    """
    with_composite = estimate(2, OBSERVED_CACHED_FRACTION)
    without = price_search(
        pb.TEACHER, pb.TARGET, pb.ADAPTER, pb.DECOMPOSED,
        calibration_tokens=pb.CALIBRATION_TOKENS,
        suite_tokens=pb.CALIBRATION_TOKENS, seq_len=pb.CALIBRATION_SEQ_LEN,
        n_profiles=2, beam_width=SCHEDULE_V1.width,
        warmup_levels=SCHEDULE_V1.warmup_levels, hardware=L40S_MEASURED,
        composite=(), depth_cached_fraction=OBSERVED_CACHED_FRACTION)

    assert with_composite.branching["composite_leaves"] == 2
    assert without.branching["composite_leaves"] == 0
    for field in ("seconds_low", "seconds_high", "seconds_hard"):
        assert getattr(with_composite, field) > getattr(without, field), (
            f"{field} is identical with and without composite, so composite is "
            "still counted in the branching and priced at nothing")

    level0 = [e for e in with_composite.per_operator if e["level"] == 0][0]
    assert "composite.stage1_sandwich_v0" in level0["implementations"]


def test_statistics_are_charged_once_per_parent_and_profile():
    """The pricing must match `StatsCache`, not idealize it or ignore it.

    At a non-root parent, `composite`, `ffn` and `width` share one collection per
    mixture. Charging each operator its own pass would bill the same collection
    three times; charging one pass for the whole parent would ignore the second
    profile.
    """
    from aadistill.autoinit.cost import Expansion, stats_collections

    def expansion(impl_id, consumes, mult=2):
        return Expansion(level=1, parent_spec_hash="p", impl_id=impl_id,
                         multiplicity=mult, consumes_stats=consumes,
                         operator_seconds_low=0.0, operator_seconds_high=0.0,
                         eval_seconds=0.0, overhead_seconds=0.0,
                         child_bytes=0, peak_resident_bytes=0)

    consumers = [expansion("composite.stage1_sandwich_v0", True),
                 expansion("ffn.activation_importance_v0", True),
                 expansion("width.global_pca_v0", True)]
    # Three consumers, two profiles, ONE collection per profile.
    assert stats_collections(1, consumers, n_profiles=2) == 2
    assert stats_collections(1, consumers, n_profiles=1) == 1
    # A parent with no stats consumer pays nothing.
    assert stats_collections(1, [expansion("depth.positional_v0", False, 1)],
                             n_profiles=2) == 0


def test_the_ROOT_cannot_share_and_is_priced_accordingly():
    """`_stats_key` returns None for the root, so the runtime collects per operator.

    The root has no artifact digest — its identity is a published revision, not
    something the search computed — so sharing has nothing to key on. Level 0 is
    the widest level in the search, and pricing it as though it shared would
    understate it by exactly the operators that make it wide.
    """
    import inspect

    from aadistill.autoinit import search as search_module
    from aadistill.autoinit.cost import Expansion, stats_collections

    source = inspect.getsource(search_module.BeamSearch._stats_key)
    assert "if parent.artifact_digest is None:" in source
    assert "return None" in source

    def expansion(impl_id):
        return Expansion(level=0, parent_spec_hash="root", impl_id=impl_id,
                         multiplicity=2, consumes_stats=True,
                         operator_seconds_low=0.0, operator_seconds_high=0.0,
                         eval_seconds=0.0, overhead_seconds=0.0,
                         child_bytes=0, peak_resident_bytes=0)

    consumers = [expansion("composite.stage1_sandwich_v0"),
                 expansion("ffn.activation_importance_v0"),
                 expansion("width.global_pca_v0")]
    # Six, not two: three consumers x two profiles, none of them shared.
    assert stats_collections(0, consumers, n_profiles=2) == 6

    est = estimate(2, OBSERVED_CACHED_FRACTION)
    level0 = [e for e in est.per_operator if e["level"] == 0][0]
    assert level0["stats_collections_per_parent"] == {"min": 6, "max": 6}
    level1 = [e for e in est.per_operator if e["level"] == 1][0]
    assert level1["stats_collections_per_parent"] == {"min": 2, "max": 2}


def test_causal_depth_is_not_counted_as_a_statistics_consumer():
    """It consumes calibration but collects no statistics; it runs its own forwards."""
    from aadistill.autoinit.cost import consumes_activation_stats
    from aadistill.autoinit.operators.base import get_implementation

    assert not consumes_activation_stats(get_implementation("depth.causal_kl_greedy_v1"))
    assert consumes_activation_stats(get_implementation("composite.stage1_sandwich_v0"))
    assert consumes_activation_stats(get_implementation("ffn.activation_importance_v0"))
    assert consumes_activation_stats(get_implementation("width.global_pca_v0"))


# --- the non-FLOP path ------------------------------------------------------


def test_materialization_overhead_is_explicit_and_scales_with_the_checkpoint():
    """The hardware anchor was measured on forward compute and does not cover I/O."""
    from aadistill.autoinit.cost import (
        CHECKPOINT_IO_PASSES, PER_CHILD_FIXED_SECONDS,
        materialization_overhead_seconds,
    )

    assert materialization_overhead_seconds(0) == pytest.approx(PER_CHILD_FIXED_SECONDS)
    big = materialization_overhead_seconds(8 * 2**30)
    small = materialization_overhead_seconds(1 * 2**30)
    assert big > small > PER_CHILD_FIXED_SECONDS
    # save + hash + reload: three passes over the bytes, structurally.
    assert CHECKPOINT_IO_PASSES == 3


def test_the_overhead_actually_reaches_the_priced_total():
    """A component nothing multiplies is a comment, not a model."""
    import aadistill.autoinit.cost as cost_module

    baseline = estimate(2, OBSERVED_CACHED_FRACTION).seconds_hard
    original = cost_module.PER_CHILD_FIXED_SECONDS
    try:
        cost_module.PER_CHILD_FIXED_SECONDS = original + 60.0
        inflated = estimate(2, OBSERVED_CACHED_FRACTION).seconds_hard
    finally:
        cost_module.PER_CHILD_FIXED_SECONDS = original
    assert inflated > baseline, (
        "raising the per-child fixed overhead by a minute changed nothing; the "
        "overhead component is not reaching the total")


def test_the_replay_covers_the_levels_attempt_3_actually_entered():
    """Attempt 3's 12 causal-depth invocations reconstruct exactly.

    Level 0 expands the root and offers causal DEPTH twice (once per profile).
    With `warmup_levels=1` nothing is pruned there, so all eight decomposed
    level-0 children become level-1 parents; five of them have not yet applied
    DEPTH, giving 5 x 2 = 10 more. 2 + 10 = 12, and the deadline fired at a
    level-1 FFN expansion — consistent with being partway through that level.

    So the hard model must cover levels 0 and 1 in at least the 544.7 min the run
    spent without finishing them.
    """
    est = estimate(2, OBSERVED_CACHED_FRACTION)
    by_level = {e["level"]: e for e in est.hard_per_level}
    covered = (by_level[0]["level_seconds"] + by_level[1]["level_seconds"]) / 60.0
    assert covered >= OBSERVED_STAGE1_MIN, (
        f"levels 0+1 are bounded at {covered:.1f} min but attempt 3 spent "
        f"{OBSERVED_STAGE1_MIN} min inside them without finishing")
    # Level 0 alone must cover the two root causal-depth invocations, which are
    # the two most expensive observed: 38.4 + 37.7 = 76.1 min.
    assert by_level[0]["level_seconds"] / 60.0 >= 76.1
