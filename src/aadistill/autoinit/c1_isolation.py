"""The Phase-C1 two-arm isolation contract, and its decision rule.

Phase A/B selected among candidates with successive halving. C1 asks a different
question — does one operator change one fixed path — and the Phase-A/B
instrument is the wrong shape for it. `SuccessiveHalvingPlan.__post_init__`
*requires* `searched_leaves >= 2` and `1 <= survivors < searched_leaves`, so a
two-arm design would be forced to eliminate one arm at rung 1 on single-seed
evidence. That is the exact mechanism that removed `cca699c93f34` from Phase B.

So this is a separate type, not a subclass. Nothing in Phase A/B is touched:
`SuccessiveHalvingPlan`, `EquivalenceRule` and `FeasibilityRule` remain in force
for the phases that froze them.

The design goal is that illegal states are **unrepresentable** rather than
merely rejected. There is no `survivors` field to set to zero, no `rungs` list
to leave empty and no `tie_break_seed` to set to `None` — those concepts do not
exist here, so no future edit can quietly reintroduce elimination.

Everything numeric is frozen by `logs/phase_c0_preregistration.json`; this module
is the executable form of that document, and `assert_preregistered` refuses a
plan whose hash does not match the frozen record.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json

SCHEMA = "aadistill.autoinit.c1_isolation_plan/v1"

#: The Phase-A/B recovery seeds. `fe9683e6a9c7` was selected under these, so
#: reusing them for confirmation would leave a winner's-curse channel.
HISTORICAL_SEEDS = (20260726, 20260801, 20260813)

#: The frozen Phase-C0 preregistration digest (commit be2ab08). Every derived
#: quantity in C1 is domain-separated from it, so nothing can be chosen after a
#: candidate exists.
C0_PREREGISTRATION_SHA256 = (
    "fb2eeea531f9f0d11f84b77cd47dff30697122de90a072a7a80c3a7535e89280")


class C1PlanError(RuntimeError):
    """The plan does not describe a legal C1 isolation experiment."""


# --- seeds ------------------------------------------------------------------

def derive_recovery_seeds(base: str = C0_PREREGISTRATION_SHA256, *, count: int = 3,
                          exclude: Sequence[int] = HISTORICAL_SEEDS) -> list[int]:
    """Three fresh seeds, derived from an identity frozen before any candidate.

        H_i    = SHA256(base + ":phase-c1:recovery-seed:" + decimal(i))
        seed_i = uint32_be(H_i[0:4]) mod 2**31

    starting at ``i = 0`` and advancing deterministically past any collision
    with a historical seed or an earlier draw.

    The base is the pushed C0 preregistration digest, so the values are fixed by
    a document that predates the replacement ATTENTION implementation and every
    C1 result. There is no discretion left to exercise, which is the point:
    human-chosen seeds cannot be shown to be independent of anything.
    """
    if count < 1:
        raise ValueError("count must be positive")
    seeds: list[int] = []
    i = 0
    while len(seeds) < count:
        h = hashlib.sha256(f"{base}:phase-c1:recovery-seed:{i}".encode()).digest()
        seed = int.from_bytes(h[:4], "big") % (2 ** 31)
        if seed not in exclude and seed not in seeds:
            seeds.append(seed)
        i += 1
        if i > 10_000:                       # unreachable; refuses to spin
            raise C1PlanError("seed derivation failed to converge")
    return seeds


# --- the plan ---------------------------------------------------------------

@dataclass(frozen=True)
class C1Arm:
    """One arm. The ATTENTION implementation is the only thing that may differ."""

    arm_id: str
    role: str                       # "incumbent" | "treatment"
    attention_impl_id: str
    attention_profile_id: str

    def __post_init__(self) -> None:
        if self.role not in ("incumbent", "treatment"):
            raise C1PlanError(
                f"{self.arm_id}: role must be 'incumbent' or 'treatment', "
                f"got {self.role!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "role": self.role,
                "attention_impl_id": self.attention_impl_id,
                "attention_profile_id": self.attention_profile_id}


@dataclass(frozen=True)
class C1IsolationPlan:
    """Exactly two arms, exactly three fresh seeds, no elimination of any kind."""

    plan_id: str
    arms: tuple[C1Arm, C1Arm]
    seeds: tuple[int, int, int]
    battery_asset_id: str
    battery_content_sha256: str
    #: Frozen by C0. Present so the plan hash covers them; not caller options.
    sesoi: float = 0.010
    design_alternative: float = 0.015
    alpha: float = 0.05
    primary_metric: str = "correct_overall"
    secondary_metric: str = "usable_rollout_rate"
    seed_robustness_min_positive: int = 2
    usable_pooled_min_delta: float = -0.05
    usable_per_seed_min_delta: float = -0.10
    catastrophic_candidate_max: float = 0.10
    catastrophic_control_min: float = 0.40
    catastrophic_rule_id: str = "per_capability_collapse"
    catastrophic_control_operand: str = "incumbent"
    catastrophic_candidate_operand: str = "treatment"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.arms) != 2:
            raise C1PlanError(
                f"C1 is a two-arm isolation experiment; got {len(self.arms)} arms")
        roles = sorted(a.role for a in self.arms)
        if roles != ["incumbent", "treatment"]:
            raise C1PlanError(
                f"C1 needs exactly one incumbent and one treatment arm, got {roles}")
        if self.arms[0].arm_id == self.arms[1].arm_id:
            raise C1PlanError("the two arms must have distinct ids")
        if self.arms[0].attention_impl_id == self.arms[1].attention_impl_id:
            raise C1PlanError(
                "both arms name the same ATTENTION implementation; then nothing "
                "is being isolated")

        if len(self.seeds) != 3:
            raise C1PlanError(
                f"C1 registers exactly 3 confirmation seeds, got {len(self.seeds)}")
        if len(set(self.seeds)) != 3:
            raise C1PlanError(f"the 3 seeds must be distinct, got {self.seeds}")
        reused = sorted(set(self.seeds) & set(HISTORICAL_SEEDS))
        if reused:
            raise C1PlanError(
                f"seeds {reused} are Phase-A/B selection seeds. fe9683e6a9c7 was "
                "selected under them, so reusing them for confirmation reinstates "
                "the winner's-curse channel C0 removed. C1 seeds must be fresh.")
        for s in self.seeds:
            if not isinstance(s, int) or isinstance(s, bool) or s < 0:
                raise C1PlanError(f"seed {s!r} is not a non-negative integer")

        if not 0 < self.sesoi <= self.design_alternative:
            raise C1PlanError(
                f"design alternative {self.design_alternative} must be at least the "
                f"SESOI {self.sesoi}")
        if not 0 < self.alpha < 0.5:
            raise C1PlanError(f"alpha {self.alpha} is not a tail probability")
        if self.primary_metric == self.secondary_metric:
            raise C1PlanError(
                "the primary and secondary metrics must differ; usable_rollout is "
                "blind to correctness by construction, which is why it vetoes "
                "rather than ranks")
        if not 1 <= self.seed_robustness_min_positive <= len(self.seeds):
            raise C1PlanError("the seed-robustness threshold is out of range")
        if self.catastrophic_control_operand == self.catastrophic_candidate_operand:
            raise C1PlanError("the catastrophic veto operands must be different arms")

    # There is deliberately no `survivors`, `rungs`, `tie_break_seed` or
    # `equivalence` field. Elimination is not a configurable option here; it is
    # absent from the type.

    @property
    def incumbent(self) -> C1Arm:
        return next(a for a in self.arms if a.role == "incumbent")

    @property
    def treatment(self) -> C1Arm:
        return next(a for a in self.arms if a.role == "treatment")

    @property
    def probe_count(self) -> int:
        """Both arms run every seed. No arm is ever dropped part-way."""
        return len(self.arms) * len(self.seeds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "plan_id": self.plan_id,
            "arms": [a.as_dict() for a in self.arms],
            "seeds": list(self.seeds),
            "probe_count": self.probe_count,
            "battery_asset_id": self.battery_asset_id,
            "battery_content_sha256": self.battery_content_sha256,
            "sesoi": self.sesoi,
            "design_alternative": self.design_alternative,
            "alpha": self.alpha,
            "primary_metric": self.primary_metric,
            "secondary_metric": self.secondary_metric,
            "seed_robustness_min_positive": self.seed_robustness_min_positive,
            "usable_pooled_min_delta": self.usable_pooled_min_delta,
            "usable_per_seed_min_delta": self.usable_per_seed_min_delta,
            "catastrophic": {
                "rule_id": self.catastrophic_rule_id,
                "candidate_max": self.catastrophic_candidate_max,
                "control_min": self.catastrophic_control_min,
                "control_operand": self.catastrophic_control_operand,
                "candidate_operand": self.catastrophic_candidate_operand,
                "statement": (
                    f"candidate usable_rollout_rate < {self.catastrophic_candidate_max} "
                    f"AND control usable_rollout_rate > {self.catastrophic_control_min} "
                    "-> catastrophic failure, replacement vetoed"),
                "asymmetry": ("deliberate: this can veto the treatment, never the "
                              "incumbent. It is never positive ranking evidence."),
            },
            "structure": {
                "successive_halving": False, "elimination_rung": False,
                "tie_break_rung": False, "search_ranking": False,
                "both_arms_run_every_seed": True,
            },
            "historical_seeds_rejected": list(HISTORICAL_SEEDS),
            "notes": dict(self.notes),
        }

    @property
    def plan_hash(self) -> str:
        return sha256_json(self.as_dict())

    def freeze(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**self.as_dict(), "plan_hash": self.plan_hash},
                                indent=1) + "\n")
        return p


def assert_preregistered(plan: C1IsolationPlan, path: str | Path) -> dict[str, Any]:
    """Refuse a plan whose hash does not match the frozen record."""
    record = json.loads(Path(path).read_text())
    if record.get("plan_hash") != plan.plan_hash:
        raise C1PlanError(
            f"plan hash {plan.plan_hash} does not match the preregistered "
            f"{record.get('plan_hash')}: the plan changed after it was frozen")
    return record


# --- the paired estimand and its bootstrap ----------------------------------

#: Bound here rather than left to a caller, because C0 froze the rule but not
#: these. They must be fixed before any C1 datum exists, and they were chosen on
#: reproducibility grounds alone: a named generator with an explicit algorithm,
#: a seed derived from the same frozen C0 digest by a distinct domain string, an
#: iteration count large enough that the 5% quantile index is exact, and a
#: stated index convention so a reported bound is recomputable to the integer.
BOOTSTRAP_ALGORITHM = "python.random.Random(seed).randrange, Mersenne Twister"
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_QUANTILE_CONVENTION = (
    "sorted ascending; lower bound = stats[floor(alpha * B)], upper bound = "
    "stats[min(B - 1, floor((1 - alpha) * B))]; two-sided uses alpha/2 each side")
BOOTSTRAP_STRATUM_CONVENTION = (
    "resample n_h prompts with replacement WITHIN each stratum, independently "
    "across strata; the overall statistic is the size-weighted mean of the "
    "stratum means, which equals the unstratified mean when strata sizes are "
    "held at their observed values")


def bootstrap_seed(base: str = C0_PREREGISTRATION_SHA256) -> int:
    """Domain-separated from the recovery seeds so the two cannot coincide."""
    h = hashlib.sha256(f"{base}:phase-c1:bootstrap".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31)


def paired_differences(incumbent: Mapping[int, Mapping[str, bool]],
                       treatment: Mapping[int, Mapping[str, bool]],
                       ) -> dict[str, float]:
    """`d_j = mean_s [ correct_treatment(j,s) - correct_incumbent(j,s) ]`.

    Both arguments map seed -> {prompt_id: correct}. Every seed must carry
    exactly the same prompt set in both arms, because a prompt missing from one
    arm is an unpaired observation and the estimand has no meaning for it.
    """
    seeds = sorted(incumbent)
    if seeds != sorted(treatment):
        raise C1PlanError(
            f"arms cover different seeds: {seeds} vs {sorted(treatment)}")
    if not seeds:
        raise C1PlanError("no seeds")
    ids = set(incumbent[seeds[0]])
    for s in seeds:
        for arm, name in ((incumbent, "incumbent"), (treatment, "treatment")):
            if set(arm[s]) != ids:
                raise C1PlanError(
                    f"seed {s} {name}: prompt set differs from the reference seed; "
                    "the design is complete by construction and an incomplete "
                    "one must not be silently averaged")
    return {j: sum(bool(treatment[s][j]) - bool(incumbent[s][j]) for s in seeds)
               / len(seeds)
            for j in sorted(ids)}


def stratified_cluster_bootstrap(d: Mapping[str, float],
                                 strata: Mapping[str, str], *,
                                 iterations: int = BOOTSTRAP_ITERATIONS,
                                 seed: int | None = None,
                                 alpha: float = 0.05) -> dict[str, Any]:
    """Resample PROMPTS within strata, keeping each prompt's arm x seed vector.

    Seeds are **not** resampled. The resulting interval is prompt-distribution
    uncertainty conditional on the three preregistered recovery-seed checkpoint
    pairs, and must never be reported as a confidence interval over a population
    of hypothetical future seeds.
    """
    if not d:
        raise C1PlanError("no prompts")
    missing = sorted(set(d) - set(strata))
    if missing:
        raise C1PlanError(f"{len(missing)} prompts have no stratum: {missing[:5]}")
    rng = random.Random(bootstrap_seed() if seed is None else seed)
    groups: dict[str, list[float]] = {}
    for j, v in d.items():
        groups.setdefault(strata[j], []).append(float(v))
    for k in groups:
        groups[k].sort()                      # stable order -> reproducible draws
    n = sum(len(v) for v in groups.values())
    point = sum(sum(v) for v in groups.values()) / n

    stats = []
    for _ in range(iterations):
        total = 0.0
        for values in groups.values():
            m = len(values)
            total += sum(values[rng.randrange(m)] for _ in range(m))
        stats.append(total / n)
    stats.sort()
    lo_i = int(math.floor(alpha * iterations))
    hi_i = min(iterations - 1, int(math.floor((1 - alpha) * iterations)))
    two_lo = int(math.floor(alpha / 2 * iterations))
    two_hi = min(iterations - 1, int(math.floor((1 - alpha / 2) * iterations)))
    return {
        "delta": point,
        "lcb_one_sided": stats[lo_i],
        "ucb_one_sided": stats[hi_i],
        "ci_two_sided_low": stats[two_lo],
        "ci_two_sided_high": stats[two_hi],
        "level": 1 - alpha,
        "iterations": iterations,
        "seed": bootstrap_seed() if seed is None else seed,
        "algorithm": BOOTSTRAP_ALGORITHM,
        "quantile_convention": BOOTSTRAP_QUANTILE_CONVENTION,
        "stratum_convention": BOOTSTRAP_STRATUM_CONVENTION,
        "n_prompts": n,
        "n_strata": len(groups),
        "resamples": "prompts within strata; seeds are FIXED BLOCKS, not resampled",
        "claim_boundary": ("prompt-distribution uncertainty CONDITIONAL ON the three "
                           "preregistered fresh recovery-seed checkpoint pairs; NOT a "
                           "CI over hypothetical future recovery seeds"),
    }


def decide(plan: C1IsolationPlan, *, boot: Mapping[str, Any],
           per_seed_delta: Sequence[float],
           usable_pooled_delta: float,
           usable_per_seed_delta: Sequence[float],
           catastrophic_violations: Sequence[Mapping[str, Any]] = (),
           ) -> dict[str, Any]:
    """The frozen three-way rule. No forced winner.

    GO    : one-sided 95% LCB > 0 AND delta >= SESOI AND >=2/3 seed deltas > 0
            AND every behavioural guardrail passes
    NO-GO : one-sided 95% UCB < SESOI OR a behavioural veto fires
    INCONCLUSIVE : otherwise
    """
    if len(per_seed_delta) != len(plan.seeds):
        raise C1PlanError(
            f"expected {len(plan.seeds)} seed-specific deltas, got "
            f"{len(per_seed_delta)}")
    if len(usable_per_seed_delta) != len(plan.seeds):
        raise C1PlanError("usable deltas must cover every seed")

    vetoes: list[str] = []
    if usable_pooled_delta <= plan.usable_pooled_min_delta:
        vetoes.append(
            f"pooled usable_rollout delta {usable_pooled_delta:+.4f} is not above "
            f"{plan.usable_pooled_min_delta}")
    for s, u in zip(plan.seeds, usable_per_seed_delta):
        if u <= plan.usable_per_seed_min_delta:
            vetoes.append(
                f"seed {s} usable_rollout delta {u:+.4f} is not above "
                f"{plan.usable_per_seed_min_delta}")
    for v in catastrophic_violations:
        vetoes.append(f"catastrophic capability collapse: {v.get('reason', v)}")

    n_positive = sum(1 for x in per_seed_delta if x > 0)
    robust = n_positive >= plan.seed_robustness_min_positive
    lcb_clears = boot["lcb_one_sided"] > 0
    point_clears = boot["delta"] >= plan.sesoi
    ucb_excludes = boot["ucb_one_sided"] < plan.sesoi

    if lcb_clears and point_clears and robust and not vetoes:
        verdict = "GO"
    elif ucb_excludes or vetoes:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "delta": boot["delta"],
        "lcb_one_sided": boot["lcb_one_sided"],
        "ucb_one_sided": boot["ucb_one_sided"],
        "ci_two_sided": [boot["ci_two_sided_low"], boot["ci_two_sided_high"]],
        "sesoi": plan.sesoi,
        "criteria": {
            "lcb_above_zero": lcb_clears,
            "point_at_or_above_sesoi": point_clears,
            "seed_robustness": {"n_positive": n_positive,
                                "required": plan.seed_robustness_min_positive,
                                "passed": robust},
            "guardrails_passed": not vetoes,
            "ucb_below_sesoi": ucb_excludes,
        },
        "per_seed_delta": list(per_seed_delta),
        "usable": {"pooled_delta": usable_pooled_delta,
                   "per_seed_delta": list(usable_per_seed_delta)},
        "vetoes": vetoes,
        "claim_boundary": boot["claim_boundary"],
        "no_forced_winner": True,
    }
