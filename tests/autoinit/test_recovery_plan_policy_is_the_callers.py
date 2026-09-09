"""A recovery plan states its policy. It cannot inherit one.

`SuccessiveHalvingPlan` used to reach for module constants as **dataclass
defaults** -- `SEED_SA`, `CAPABILITY_SCHEMA_V1`, `CATASTROPHIC_V1`,
`n_pooled=300`. `SuccessiveHalvingPlan(...)` with no policy arguments was
therefore silently *this* study, and a second study would have inherited its
seeds, capability names, collapse thresholds and pooled count without anyone
saying so and without anything reporting it.

The defaults are gone and every policy field is `kw_only` and required. Proving
that needs **two** policies, not one: a single caller cannot distinguish "the
plan used what I passed" from "the plan used the default, which happens to be
what I passed". So this module runs the same mechanism under the current Phase-A
study and under a deliberately unrelated second study, and requires the outputs
to differ everywhere the policy differs.

Study B is not a plausible variant of study A. Different seeds, different
capability names, a different collapse threshold, a different pooled count, a
different aggregation, different metric names -- because a near-miss would still
pass if some field quietly fell back.
"""
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError  # noqa: F401  (documents intent)
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.planning.recovery import (  # noqa: E402
    CapabilitySchema,
    CatastrophicCapabilityRule,
    EquivalenceRule,
    ScorableAwareSeedAggregation,
    SeedAggregation,
    SuccessiveHalvingPlan,
)

from experiments.recovery_policy import plan_policy  # noqa: E402
from test_cost_and_recovery import E1_KD_HEAVY_0860K  # noqa: E402

#: A second study's policy. Deliberately unlike the first in every field.
STUDY_B = dict(
    seeds=(11, 22, 33),
    tie_break_seed=44,
    reported_components=("answered", "terminated"),
    equivalence=EquivalenceRule(n_pooled=97),
    catastrophic=CatastrophicCapabilityRule(metric="acc", control_min=0.42),
    capability_schema=CapabilitySchema(expected=("alpha", "beta")),
    aggregation=SeedAggregation(),
    feasibility_metric="b_feasible",
    primary_metric="b_primary",
    secondary_metric="b_secondary",
)

#: `__post_init__` requires both rules to be stated before the run -- "a rule
#: written after the table is a rule chosen on the outcome". They are not
#: policy under test here; they are the plan's own precondition.
COMMON = dict(plan_id="policy.probe", recipe=E1_KD_HEAVY_0860K,
              searched_leaves=5, survivors=2,
              survivor_rule="top 2 searched leaves; the control advances",
              winner_rule="top 1 among feasible")


def plan(policy: dict) -> SuccessiveHalvingPlan:
    return SuccessiveHalvingPlan(**COMMON, **policy)


@pytest.fixture
def study_a():
    return plan_policy(capability_schema=CapabilitySchema(expected=("x", "y")))


# --- the mechanism carries no study of its own ------------------------------

class TestTwoCallersGetTwoPlans:
    def test_the_seeds_are_the_callers(self, study_a):
        a, b = plan(study_a), plan(STUDY_B)
        assert a.seeds != b.seeds
        assert b.seeds == (11, 22, 33)
        assert b.tie_break_seed == 44

    def test_the_capability_names_are_the_callers(self, study_a):
        a, b = plan(study_a), plan(STUDY_B)
        assert a.capability_schema.expected == ("x", "y")
        assert b.capability_schema.expected == ("alpha", "beta")

    def test_the_collapse_threshold_is_the_callers(self):
        b = plan(STUDY_B)
        assert b.catastrophic.control_min == 0.42
        assert b.catastrophic.metric == "acc"

    def test_the_pooled_count_is_the_callers(self, study_a):
        a, b = plan(study_a), plan(STUDY_B)
        assert b.equivalence.n_pooled == 97
        assert a.equivalence.n_pooled != 97

    def test_the_metric_names_are_the_callers(self):
        b = plan(STUDY_B)
        assert (b.feasibility_metric, b.primary_metric, b.secondary_metric) == \
            ("b_feasible", "b_primary", "b_secondary")

    def test_the_aggregation_is_the_callers(self, study_a):
        a, b = plan(study_a), plan(STUDY_B)
        assert isinstance(b.aggregation, SeedAggregation)
        assert isinstance(a.aggregation, ScorableAwareSeedAggregation)

    def test_the_two_plans_hash_differently(self, study_a):
        """The policy is IN the identity. Two studies that hashed alike would
        make the plan hash useless for telling them apart."""
        assert plan(study_a).plan_hash != plan(STUDY_B).plan_hash

    def test_the_same_policy_hashes_stably(self, study_a):
        assert plan(study_a).plan_hash == plan(study_a).plan_hash


# --- and it refuses to invent one -------------------------------------------

class TestAbsentPolicyIsRefused:
    """The whole point. Silence must not resolve to the current experiment."""

    @pytest.mark.parametrize("missing", [
        "seeds", "tie_break_seed", "reported_components", "equivalence",
        "catastrophic", "capability_schema", "aggregation",
    ])
    def test_omitting_a_policy_field_raises(self, study_a, missing):
        policy = {k: v for k, v in study_a.items() if k != missing}
        with pytest.raises(TypeError, match=missing):
            plan(policy)

    def test_a_plan_with_no_policy_at_all_raises(self):
        with pytest.raises(TypeError):
            SuccessiveHalvingPlan(**COMMON)

    def test_the_core_module_no_longer_defines_the_study(self):
        """The constants that used to be the defaults are not importable from
        the core -- so nothing can reach them by accident either."""
        import aadistill.initialization.planning.recovery as core

        for name in ("SEED_SA", "SEED_SB", "SEED_SC", "CATASTROPHIC_V1",
                     "CAPABILITY_SCHEMA_V1", "POOLED_COUNTS_V1",
                     "POOLED_COUNTS_V2", "PREFLIGHT_PLAN_V1",
                     "TRAINER_SOURCE_FILES_V1", "RECOVERY_SCORING_FILES_V2",
                     "RECOVERY_SCORING_FILES_V3"):
            assert not hasattr(core, name), (
                f"{name} is back in the core; a dataclass default can reach it "
                "again and a plan built with no arguments becomes this study")


# --- the relocation moved the values, not their meaning ---------------------

def test_the_studys_policy_is_unchanged_by_the_move():
    """Study A's identities must be exactly what they were as core constants.

    The point of the extraction was ownership, not revision -- if a value had
    drifted, the Phase-A plan hash would move and its frozen record would stop
    matching.
    """
    from experiments.recovery_policy import (
        CAPABILITY_SCHEMA_V1, CATASTROPHIC_V1, PREFLIGHT_PLAN_V1, SEED_SA,
        SEED_SB, SEED_SC)

    assert (SEED_SA, SEED_SB, SEED_SC) == (20260726, 20260801, 20260813)
    assert CATASTROPHIC_V1.metric == "usable_rollout_rate"
    assert CATASTROPHIC_V1.control_min == 0.4
    assert CAPABILITY_SCHEMA_V1.expected == (
        "gsm8k", "math_verified", "multihop", "rag", "knowledge", "tool")
    assert PREFLIGHT_PLAN_V1.plan_hash.startswith("83218ddd283c961f")
