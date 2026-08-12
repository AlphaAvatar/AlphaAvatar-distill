"""Recovery Top-N orchestration — configuration and gates only.

**No paid recovery runs from this module.** It defines the successive-halving
schedule, the admission gate and the preregistration record; executing the probes
is a separate, separately authorized step. That split is the point: the schedule
and its thresholds have to be frozen *before* the run that they judge, and the
only way to make that checkable is to have an artifact that exists first.

The schedule:

    complete target-size leaves  ->  Top-N, identical 0.86M recovery, seed sa
      ->  search-battery evaluation  ->  survivors (preregistered rule)
      ->  seed sb  ->  Top-1  ->  full recovery

Three rules the rest of the project already paid to learn:

* **Selection here is on autonomous behaviour, not state NLL** (E7: a −5.22 nat
  NLL swing moved behaviour by +0.0000; the best-NLL checkpoint of its trajectory
  produced zero protocol-valid generations).
* **Two seeds, because one is unreadable.** The behaviour metric moves 0.1290 on
  seed alone, so a single-seed ranking of close candidates is noise. Hence
  survivors are re-run on ``sb`` rather than the Top-1 being taken from ``sa``.
* **Thresholds are preregistered, never chosen after seeing the table**
  (AGENTS.md 4.5). ``SuccessiveHalvingPlan.freeze`` writes the record and
  ``assert_preregistered`` refuses a selection whose plan hash does not match.

``N`` is deliberately unset here. The cost model decides it, and the decision is
the maintainer's.
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


@dataclass(frozen=True)
class SuccessiveHalvingPlan:
    """A preregistered probe schedule. Freeze it, then run it."""

    plan_id: str
    recipe: RecoveryRecipe
    top_n: int
    survivors: int
    seeds: tuple[int, ...] = (SEED_SA, SEED_SB)
    selection_metric: str = "usable_rollout_rate"
    reported_components: tuple[str, ...] = (
        "non_empty", "natural_termination", "no_severe_repetition",
        "no_context_limit", "protocol_valid")
    secondary_metrics: tuple[str, ...] = ("correct_overall", "correct_given_usable")
    survivor_rule: str = ""
    winner_rule: str = ""
    battery_asset_id: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_n < 2:
            raise ValueError("successive halving needs at least 2 candidates")
        if not 1 <= self.survivors < self.top_n:
            raise ValueError(
                f"survivors ({self.survivors}) must be at least 1 and fewer than "
                f"top_n ({self.top_n}); otherwise the first rung selects nothing")
        if len(self.seeds) < 2:
            raise ValueError(
                "one seed cannot rank close candidates: the behaviour metric's "
                "seed-only spread is 0.1290")
        if not self.survivor_rule or not self.winner_rule:
            raise ValueError(
                "survivor_rule and winner_rule must be stated before the run; a rule "
                "written after the table is a rule chosen on the outcome")

    @property
    def probe_count(self) -> int:
        """Total recovery probes: every leaf on sa, then survivors on sb."""
        return self.top_n + self.survivors

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "plan_id": self.plan_id,
            "recipe": self.recipe.as_dict(), "top_n": self.top_n,
            "survivors": self.survivors, "seeds": list(self.seeds),
            "probe_count": self.probe_count,
            "selection_metric": self.selection_metric,
            "reported_components": list(self.reported_components),
            "secondary_metrics": list(self.secondary_metrics),
            "survivor_rule": self.survivor_rule, "winner_rule": self.winner_rule,
            "battery_asset_id": self.battery_asset_id,
            "notes": dict(self.notes),
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
    for state in candidates:
        state.require_recovery_admissible()
    if len(candidates) < plan.top_n:
        raise RecoveryAdmissionError(
            f"plan asks for Top-{plan.top_n} but only {len(candidates)} admissible "
            "leaves exist; report the shortfall rather than shrinking N")
    return list(candidates)


def probe_configs(selected: Sequence[InitializationState],
                  plan: SuccessiveHalvingPlan) -> list[dict[str, Any]]:
    """Rung-1 probe descriptors: identical recovery, one per leaf, seed sa.

    Descriptors only — nothing here launches anything.
    """
    sa = plan.seeds[0]
    return [
        {
            "probe_id": f"{plan.plan_id}.rung1.{state.state_id[:12]}.sa",
            "rung": 1,
            "state_id": state.state_id,
            "path": state.path_label,
            "student_checkpoint": state.checkpoint_path,
            "student_sha256": state.checkpoint_sha256,
            "recipe": plan.recipe.recipe_id,
            "seed": sa,
            "selection_metric": plan.selection_metric,
            "_purpose": ("identical recovery across leaves; the only intended "
                         "difference between probes is the initialization"),
        }
        for state in selected
    ]
