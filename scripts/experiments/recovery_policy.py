"""The recovery-search policy this project's Phase-A study runs under.

The application layer. `aadistill.initialization.planning.recovery` holds the
mechanisms -- what a `SuccessiveHalvingPlan` is, how seeds are pooled, how an
equivalence interval is computed, what a capability schema checks -- and now
carries none of the instances below.

They used to be module constants in the core, reached as **dataclass defaults**.
That is the specific problem this fixes: `SuccessiveHalvingPlan(...)` with no
policy arguments silently became the current experiment, so a second study would
have inherited this one's seeds, capability names, collapse thresholds and
pooled count without saying so, and nothing would have reported it.

Every value here is transcribed from the previous in-code definitions without
edit, so each identity is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.planning.recovery import (  # noqa: E402
    CapabilitySchema, CatastrophicCapabilityRule, CorrectnessRule,
    EquivalenceRule, PreflightPlan, PreflightStage,
    ScorableAwareSeedAggregation, SeedAggregation)

POLICY_CONFIG = REPO / "configs/experiments/phase_a/recovery_policy.json"
PREFLIGHT_PLAN_CONFIG = REPO / "configs/experiments/phase_a/preflight_plan.json"

_POLICY = json.loads(POLICY_CONFIG.read_text())
_PLAN = json.loads(PREFLIGHT_PLAN_CONFIG.read_text())

#: The two search seeds, and the tie-break seed used ONLY for candidates that
#: finish inside the preregistered equivalence interval after two seeds.
SEED_SA: int = _POLICY["seeds"]["sa"]
SEED_SB: int = _POLICY["seeds"]["sb"]
SEED_SC: int = _POLICY["seeds"]["tie_break"]
SEEDS: tuple[int, ...] = (SEED_SA, SEED_SB)

#: Pooled counts, not averaged rates. Both aggregation instances; v2 is
#: scorable-aware and is what a current plan should use.
POOLED_COUNTS_V1 = SeedAggregation()
POOLED_COUNTS_V2 = ScorableAwareSeedAggregation()

#: The per-capability collapse gate, enforced at both rungs.
CATASTROPHIC_V1 = CatastrophicCapabilityRule(
    **{k: v for k, v in _POLICY["catastrophic"].items()
       if k not in ("rule_id", "version")})

#: What every result must carry for that gate to see anything.
CAPABILITY_SCHEMA_V1 = CapabilitySchema(
    expected=tuple(_POLICY["capability_schema"]["expected"]))

#: The behaviour equivalence interval. A rule, not a constant: the formula is
#: frozen and the numeric value comes from the control characterization.
EQUIVALENCE_V1 = EquivalenceRule(n_pooled=_POLICY["equivalence"]["n_pooled"])

REPORTED_COMPONENTS: tuple[str, ...] = tuple(_POLICY["reported_components"])
#: Study choices that used to be generic dataclass defaults.
INCLUDE_CANONICAL_CONTROL: bool = _POLICY["include_canonical_control"]
FEASIBILITY_MIN: float = _POLICY["feasibility_min"]
FEASIBILITY_METRIC: str = _POLICY["feasibility_metric"]
PRIMARY_METRIC: str = _POLICY["primary_metric"]
SECONDARY_METRIC: str = _POLICY["secondary_metric"]

#: Everything a `SuccessiveHalvingPlan` needs that used to arrive as a default.
#: Spread into the constructor by callers, so a plan states its policy.
PLAN_POLICY = dict(
    seeds=SEEDS,
    tie_break_seed=SEED_SC,
    reported_components=REPORTED_COMPONENTS,
    equivalence=EQUIVALENCE_V1,
    catastrophic=CATASTROPHIC_V1,
    capability_schema=CAPABILITY_SCHEMA_V1,
    aggregation=POOLED_COUNTS_V2,
    feasibility_metric=FEASIBILITY_METRIC,
    feasibility_min=FEASIBILITY_MIN,
    primary_metric=PRIMARY_METRIC,
    secondary_metric=SECONDARY_METRIC,
    include_canonical_control=INCLUDE_CANONICAL_CONTROL,
)

#: The Phase-A preflight's stage topology.
PREFLIGHT_PLAN_V1 = PreflightPlan(
    plan_id=_PLAN["plan_id"],
    version=_PLAN["version"],
    stages=tuple(
        PreflightStage(stage=s["stage"], name=s["name"], purpose=s["purpose"],
                       produces=tuple(s["produces"]), blocking=s["blocking"],
                       stop_conditions=tuple(s.get("stop_conditions", ())))
        for s in _PLAN["stages"]),
)

def plan_policy(**overrides) -> dict:
    """This study's plan policy, with a caller's explicit values winning.

    `SuccessiveHalvingPlan` requires every policy field by keyword, so a plan
    states what it runs under. This is the study's answer; a different study
    passes its own, and a plan that wants one field different says so here
    rather than relying on a default that used to live in the core.
    """
    return {**PLAN_POLICY, **overrides}


__all__ = ["plan_policy", "POLICY_CONFIG", "PREFLIGHT_PLAN_CONFIG", "SEED_SA", "SEED_SB",
           "SEED_SC", "SEEDS", "POOLED_COUNTS_V1", "POOLED_COUNTS_V2",
           "CATASTROPHIC_V1", "CAPABILITY_SCHEMA_V1", "EQUIVALENCE_V1",
           "REPORTED_COMPONENTS", "FEASIBILITY_METRIC", "FEASIBILITY_MIN",
           "INCLUDE_CANONICAL_CONTROL", "PRIMARY_METRIC",
           "SECONDARY_METRIC", "PLAN_POLICY", "PREFLIGHT_PLAN_V1",
           "CORRECT_IN_USABLE_ROLLOUT"]


#: **The scientific rule, unchanged.** `correct` means "correct in a usable
#: rollout": a checkpoint that emits the right answer and then loops forever
#: cannot produce trajectories for Stage 5, and counting it as correct would
#: let the primary metric reward the exact failure that dominates this project
#: -- roughly 31% of rollouts hitting the context limit. `correct_given_usable`
#: then means what it says, and `correct_overall` means "answered correctly, in
#: a rollout we could actually use".
#:
#: The rejected alternative is recorded: scoring correctness independently of
#: usability would make `correct_overall` a measure of latent capability rather
#: than of deployable behaviour, and would break `correct <= usable` in the
#: aggregate.
#:
#: This lived inside `score_recovery_row`, which made a reusable planning
#: module assert what "this battery" defines. The arithmetic below is
#: transcribed exactly; no scored row changes.
CORRECT_IN_USABLE_ROLLOUT = CorrectnessRule(
    rule_id="correct_in_usable_rollout@v1",
    decide=lambda scorable, usable, scorer_correct: (
        scorable and usable and scorer_correct),
    note="this battery defines correct as 'correct in a usable rollout'",
    correct_implies_usable=True,
)
