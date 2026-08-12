"""Recovery Top-N orchestration — configuration and gates only.

**No paid recovery runs from this module.** It defines the successive-halving
schedule, the admission gate and the preregistration record; executing the probes
is a separate, separately authorized step. That split is the point: the schedule
and its thresholds have to be frozen *before* the run that they judge, and the
only way to make that checkable is to have an artifact that exists first.

The schedule:

    N searched leaves + the retained canonical control
      ->  rung 1: identical 0.86M recovery on seed sa, all of them
      ->  search-battery evaluation
      ->  rung 2: the control (unconditionally) + the best S searched leaves, seed sb
      ->  Top-1 from the two-seed result
      ->  optional seed sc, for tied candidates only
      ->  full recovery (separately authorized)

Rules the rest of the project already paid to learn:

* **Selection is on autonomous behaviour, not state NLL** (E7: a −5.22 nat NLL
  swing moved behaviour by +0.0000; the best-NLL checkpoint of its trajectory
  produced zero protocol-valid generations).
* **Selection is a constraint followed by an objective, never a weighted sum.**
  ``usable_rollout`` is a *feasibility* gate — it is blind to correctness by
  construction, so a terse contentless reply scores perfectly on it and a
  weighted ``usable + correct`` score would let stability buy its way past a
  capability failure. Feasible candidates are then ranked on ``correct_overall``,
  with ``correct_given_usable`` reported as a diagnostic that explains *why* a
  candidate ranks where it does without moving it.
* **Two seeds minimum, because one is unreadable.** The behaviour metric moves
  0.1290 on seed alone.
* **The control advances to rung 2 regardless of its rung-1 result.** A baseline
  eliminated at rung 1 gives a two-seed comparison with nothing to compare
  against, and its variance is exactly what makes the searched candidates'
  differences readable.
* **Thresholds are preregistered, never chosen after seeing the table**
  (AGENTS.md 4.5). ``SuccessiveHalvingPlan.freeze`` writes the record and
  ``assert_preregistered`` refuses a selection whose plan hash does not match.

``N`` is deliberately unset here. The cost model informs it; the decision is the
maintainer's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json
from .state import InitializationState

SCHEMA = "aadistill.autoinit.recovery_plan/v1"


class RecoveryAdmissionError(RuntimeError):
    """Something that is not a complete target-size leaf was offered to a probe."""


@dataclass(frozen=True)
class RecoveryRecipe:
    """The frozen KD-heavy recovery, reused so initialization stays the variable."""

    recipe_id: str
    ce_weight: float
    kd_weight: float
    temperature: float
    kd_scope: str
    tokens: int
    pack: str
    pack_sha256: str | None = None
    block_len: int = 8192
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id, "ce_weight": self.ce_weight,
            "kd_weight": self.kd_weight, "temperature": self.temperature,
            "kd_scope": self.kd_scope, "tokens": self.tokens, "pack": self.pack,
            "pack_sha256": self.pack_sha256, "block_len": self.block_len,
            "description": self.description,
        }


#: E1/P1 at the 0.86M probe rung. Frozen: AutoInitializer v1 changes the
#: initialization, not the recovery objective (handoff 5.1 item 18).
E1_KD_HEAVY_0860K = RecoveryRecipe(
    recipe_id="e1_p1_kd_heavy@0.86M",
    ce_weight=0.25, kd_weight=1.0, temperature=1.0, kd_scope="all",
    tokens=860_000, pack="artifacts/stage3/ladder_uniform_probe",
    pack_sha256="6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c",
    description=("The recovery recipe every E1 arm used, at the rung the frozen "
                 "battery was sampled from. Held fixed so a difference between two "
                 "probes is a difference between two initializations."),
)

SEED_SA = 20260726
SEED_SB = 20260801
#: Tie-break seed. Used only for candidates that finish inside the preregistered
#: equivalence interval after two seeds — never as a third look at everything.
SEED_SC = 20260813


@dataclass(frozen=True)
class SuccessiveHalvingPlan:
    """A preregistered probe schedule. Freeze it, then run it."""

    plan_id: str
    recipe: RecoveryRecipe
    #: Searched leaves admitted at rung 1. The canonical control is additional.
    searched_leaves: int
    #: Searched leaves advancing to rung 2. The control advances unconditionally.
    survivors: int
    seeds: tuple[int, ...] = (SEED_SA, SEED_SB)
    tie_break_seed: int | None = SEED_SC
    include_canonical_control: bool = True
    #: Feasibility constraint. Blind to correctness by construction, so it gates
    #: rather than scores.
    feasibility_metric: str = "usable_rollout_rate"
    feasibility_min: float = 0.0
    #: The capability objective, applied among feasible candidates only.
    primary_metric: str = "correct_overall"
    #: Reported to explain a ranking; never changes one.
    secondary_metric: str = "correct_given_usable"
    reported_components: tuple[str, ...] = (
        "non_empty", "natural_termination", "no_severe_repetition",
        "no_context_limit", "protocol_valid")
    #: Two candidates within this much of each other on the primary metric are
    #: not distinguished by two seeds and go to the tie-break seed.
    equivalence_interval: float = 0.0
    survivor_rule: str = ""
    winner_rule: str = ""
    battery_asset_id: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.searched_leaves < 2:
            raise ValueError("successive halving needs at least 2 searched leaves")
        if not 1 <= self.survivors < self.searched_leaves:
            raise ValueError(
                f"survivors ({self.survivors}) must be at least 1 and fewer than "
                f"searched_leaves ({self.searched_leaves}); otherwise rung 1 "
                "selects nothing")
        if len(self.seeds) < 2:
            raise ValueError(
                "one seed cannot rank close candidates: the behaviour metric's "
                "seed-only spread is 0.1290")
        if self.feasibility_metric == self.primary_metric:
            raise ValueError(
                "the feasibility constraint and the capability objective must be "
                "different metrics; usable_rollout is blind to correctness by "
                "construction, which is exactly why it gates rather than ranks")
        if self.equivalence_interval < 0:
            raise ValueError("equivalence_interval must be >= 0")
        if not self.survivor_rule or not self.winner_rule:
            raise ValueError(
                "survivor_rule and winner_rule must be stated before the run; a rule "
                "written after the table is a rule chosen on the outcome")

    @property
    def rung1_probes(self) -> int:
        return self.searched_leaves + (1 if self.include_canonical_control else 0)

    @property
    def rung2_probes(self) -> int:
        return self.survivors + (1 if self.include_canonical_control else 0)

    @property
    def probe_count(self) -> int:
        """Probes excluding any tie-break rung, which is conditional."""
        return self.rung1_probes + self.rung2_probes

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "plan_id": self.plan_id,
            "recipe": self.recipe.as_dict(),
            "searched_leaves": self.searched_leaves,
            "survivors": self.survivors, "seeds": list(self.seeds),
            "tie_break_seed": self.tie_break_seed,
            "include_canonical_control": self.include_canonical_control,
            "rung1_probes": self.rung1_probes, "rung2_probes": self.rung2_probes,
            "probe_count": self.probe_count,
            "selection": {
                "feasibility_metric": self.feasibility_metric,
                "feasibility_min": self.feasibility_min,
                "primary_metric": self.primary_metric,
                "secondary_metric": self.secondary_metric,
                "equivalence_interval": self.equivalence_interval,
                "rule": ("feasibility constraint, then primary objective among "
                         "feasible candidates; the secondary metric is reported and "
                         "never reorders. No weighted combination."),
            },
            "reported_components": list(self.reported_components),
            "survivor_rule": self.survivor_rule, "winner_rule": self.winner_rule,
            "battery_asset_id": self.battery_asset_id,
            "notes": dict(self.notes),
        }

    def select(self, results: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
        """Constraint, then objective. Returns the ranking and every exclusion.

        ``results`` are per-candidate dicts carrying at least the feasibility and
        primary metrics plus ``state_id`` and ``is_control``. The control is never
        excluded by the feasibility gate — a baseline that fails the gate is a
        finding about the gate or the baseline, and dropping it would leave the
        searched candidates with nothing to be compared against.
        """
        feasible, excluded = [], []
        for row in results:
            value = float(row.get(self.feasibility_metric, 0.0))
            if row.get("is_control"):
                feasible.append(row)
                continue
            if value < self.feasibility_min:
                excluded.append({**dict(row), "reason": (
                    f"{self.feasibility_metric}={value:.4f} below the preregistered "
                    f"feasibility floor {self.feasibility_min:.4f}")})
            else:
                feasible.append(row)

        ranked = sorted(
            feasible,
            key=lambda r: (-float(r.get(self.primary_metric, 0.0)), str(r["state_id"])))
        chosen = [r for r in ranked if not r.get("is_control")][:k]
        best = float(chosen[0][self.primary_metric]) if chosen else None
        tied = [r["state_id"] for r in chosen
                if best is not None
                and abs(float(r[self.primary_metric]) - best) <= self.equivalence_interval]
        return {
            "ranked": ranked, "selected": [r["state_id"] for r in chosen],
            "excluded_by_feasibility": excluded,
            "tied_within_equivalence": tied,
            "needs_tie_break_seed": len(tied) > 1 and self.tie_break_seed is not None,
        }

    @property
    def plan_hash(self) -> str:
        return sha256_json(self.as_dict())

    def freeze(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            **self.as_dict(),
            "plan_hash": self.plan_hash,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n")
        return p


def assert_preregistered(plan: SuccessiveHalvingPlan, path: str | Path) -> dict[str, Any]:
    """Refuse a selection whose plan is not the frozen one."""
    p = Path(path)
    if not p.is_file():
        raise RecoveryAdmissionError(
            f"no frozen plan at {p}; a Top-N selection needs its rules registered first")
    frozen = json.loads(p.read_text())
    if frozen.get("plan_hash") != plan.plan_hash:
        raise RecoveryAdmissionError(
            f"the plan in {p} hashes to {frozen.get('plan_hash', '')[:12]} but the "
            f"plan being executed hashes to {plan.plan_hash[:12]}; a threshold moved "
            "after freezing")
    return frozen


def admit_leaves(candidates: Sequence[InitializationState],
                 plan: SuccessiveHalvingPlan) -> list[InitializationState]:
    """The gate. Only complete, measured, target-size leaves get through.

    Refusing intermediates is not a formality. A depth-only intermediate is
    *larger* than the target and will usually look better on teacher KL than any
    fully compressed leaf — E8b measured exactly that reversal, DC beating DP by
    0.89–1.18 nats at full width while FC lost to FP by 0.90–2.82 nats once
    compressed. An unguarded Top-N would fill up with states that cannot be
    deployed and whose ranking does not transfer.
    """
    searched = [s for s in candidates if s.provenance != "retained_canonical"]
    for state in candidates:
        state.require_recovery_admissible()
    if len(searched) < plan.searched_leaves:
        raise RecoveryAdmissionError(
            f"plan asks for {plan.searched_leaves} searched leaves but only "
            f"{len(searched)} admissible ones exist; report the shortfall rather "
            "than shrinking N")
    if plan.include_canonical_control and len(searched) == len(candidates):
        raise RecoveryAdmissionError(
            "the plan includes the canonical control but none was supplied; a "
            "comparison whose baseline is a re-executed recipe rather than the "
            "retained checkpoint is not the comparison the plan describes")
    return list(candidates)


def probe_configs(selected: Sequence[InitializationState],
                  plan: SuccessiveHalvingPlan, *, rung: int = 1) -> list[dict[str, Any]]:
    """Probe descriptors: identical recovery, one per candidate, one seed.

    Descriptors only — nothing here launches anything.
    """
    if rung < 1 or rung > len(plan.seeds):
        raise RecoveryAdmissionError(
            f"rung {rung} has no seed in {list(plan.seeds)}")
    seed = plan.seeds[rung - 1]
    suffix = f"s{'abc'[rung - 1]}"
    return [
        {
            "probe_id": f"{plan.plan_id}.rung{rung}.{state.state_id[:12]}.{suffix}",
            "rung": rung,
            "state_id": state.state_id,
            "path": state.path_label,
            "is_control": state.provenance == "retained_canonical",
            "student_checkpoint": state.checkpoint_path,
            "student_artifact_digest": state.artifact_digest,
            "student_single_shard_sha256": state.checkpoint_sha256,
            "recipe": plan.recipe.recipe_id,
            "seed": seed,
            "feasibility_metric": plan.feasibility_metric,
            "primary_metric": plan.primary_metric,
            "_purpose": ("identical recovery across candidates; the only intended "
                         "difference between probes is the initialization"),
        }
        for state in selected
    ]
