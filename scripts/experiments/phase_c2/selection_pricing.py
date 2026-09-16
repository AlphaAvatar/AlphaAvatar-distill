"""What a bounded behavioural recovery SELECTION stage costs, derived.

    PYTHONPATH=src:scripts python -m experiments.phase_c2.selection_pricing

Zero cost. It trains nothing, launches nothing and needs no GPU.

The full joint re-search produces a Top-K candidate set on a **cheap** metric.
Cheap-metric order is not behavioural order — E7 measured a `-5.22` nat NLL swing
that moved behaviour by `+0.0000` — so choosing the C2 incumbent requires
training each admitted candidate under the frozen `0.86M` recovery recipe and
scoring it on the frozen battery. This module prices exactly that, from probes
this project has actually run.

**Every probe is trained fresh.** `logs/shared/analyses/autoinit_historical_probe_reuse.json`
records `reuse_verified: false` — all eleven historical probes fail
`scoring_contract_matches_live` — so there is no admissible reuse to net off, and
pricing as though there were would underfund the stage. That record is read here
rather than remembered, and the price rises if it ever changes.

**The probe cost is measured, not modelled.** C1 attempt 18 trained six probes
under this recipe and evaluated six on a frozen battery, and its committed stage
record carries the wall-clock and the spend at each boundary. Per-probe minutes
are recovered from that record by division, which is honest about being an
average over six and is stated as such.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


class SelectionPricingError(RuntimeError):
    """The selection stage cannot be priced from the committed record."""


#: The measured source: the most recent formal session that trained probes under
#: the frozen recipe AND evaluated them on a frozen battery, so both halves come
#: from one run on one card.
MEASURED_SOURCE = (
    "logs/stages/stage-1/phase_c1/runs/attempt18/evidence/c1_evidence.json")

#: The reuse ruling this price depends on. Read, never assumed.
REUSE_RULING = "logs/shared/analyses/autoinit_historical_probe_reuse.json"

#: Stage ids in the measured source, and what each one bounds.
TRAIN_STAGE = "G"
EVAL_STAGE = "H"
PRE_TRAIN_STAGE = "F"

#: Non-probe session time, in minutes. Same provenance rule as every other
#: session estimate in this project: the figures the launchers plan with, not the
#: lucky warm-image observations. Setup has been observed at 6, 8.5 and over 150
#: minutes on the same script and card.
SESSION_PHASE_MINUTES: tuple[tuple[str, float], ...] = (
    ("setup_and_asset_staging", 45.0),
    ("bundle_transfer", 6.0),
    ("teacher_fetch_and_verify", 8.0),
    ("machine_gates", 22.0),
    ("selection_commit_and_artifact_manifest", 8.0),
    ("artifact_synchronization", 6.0),
)

CONTINGENCY_FRACTION = 0.10
ARTIFACT_RECOVERY_RESERVE_MINUTES = 30.0


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class ProbeCost:
    """Per-probe minutes and dollars, recovered from a measured session."""

    train_minutes: float
    eval_minutes: float
    train_usd: float
    eval_usd: float
    n_probes: int
    price_per_hour: float
    source: str

    @property
    def minutes(self) -> float:
        return round(self.train_minutes + self.eval_minutes, 4)

    @property
    def usd(self) -> float:
        return round(self.train_usd + self.eval_usd, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_minutes": round(self.train_minutes, 3),
            "eval_minutes": round(self.eval_minutes, 3),
            "total_minutes": self.minutes,
            "train_usd": round(self.train_usd, 4),
            "eval_usd": round(self.eval_usd, 4),
            "total_usd": self.usd,
            "n_probes_measured": self.n_probes,
            "price_per_hour_observed": self.price_per_hour,
            "source": self.source,
            "_it_is_a_mean": (
                f"recovered by dividing the stage's wall-clock and spend by the "
                f"{self.n_probes} probes it processed. It is a per-probe MEAN "
                "over that session, not a maximum, and a selection stage with "
                "more probes will vary around it."),
        }


def measured_probe_cost(repo_root: str | Path = REPO_ROOT) -> ProbeCost:
    """Per-probe train and evaluate cost, from the committed stage record."""
    path = Path(repo_root) / MEASURED_SOURCE
    if not path.is_file():
        raise SelectionPricingError(
            f"{MEASURED_SOURCE} is missing, so there is no measured probe cost "
            "to price from. Refusing to model one.")
    stages = json.loads(path.read_text())["stages"]
    for key in (PRE_TRAIN_STAGE, TRAIN_STAGE, EVAL_STAGE):
        if key not in stages:
            raise SelectionPricingError(
                f"the measured source has no stage {key!r}; the probe cost "
                "cannot be bounded without both of its endpoints")
    pre, train, evaluate = (stages[PRE_TRAIN_STAGE], stages[TRAIN_STAGE],
                            stages[EVAL_STAGE])
    n_trained = int(train["probes_trained"])
    n_evaluated = int(evaluate["probes_evaluated"])
    if n_trained != n_evaluated or n_trained <= 0:
        raise SelectionPricingError(
            f"{n_trained} probes trained and {n_evaluated} evaluated; a "
            "per-probe cost from mismatched counts would be wrong for both")

    train_minutes = (_utc(train["finished_utc"])
                     - _utc(pre["finished_utc"])).total_seconds() / 60
    eval_minutes = (_utc(evaluate["finished_utc"])
                    - _utc(train["finished_utc"])).total_seconds() / 60
    train_usd = float(train["spend_usd"]) - float(pre["spend_usd"])
    eval_usd = float(evaluate["spend_usd"]) - float(train["spend_usd"])
    #: The rate the measured session actually billed at, recovered from its own
    #: spend and wall-clock rather than assumed from a quote.
    rate = (train_usd + eval_usd) / ((train_minutes + eval_minutes) / 60)
    return ProbeCost(
        train_minutes=train_minutes / n_trained,
        eval_minutes=eval_minutes / n_evaluated,
        train_usd=train_usd / n_trained,
        eval_usd=eval_usd / n_evaluated,
        n_probes=n_trained, price_per_hour=round(rate, 4),
        source=f"{MEASURED_SOURCE}: stages {PRE_TRAIN_STAGE}/{TRAIN_STAGE}/{EVAL_STAGE}")


def reuse_is_admissible(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Whether any historical probe may be cited instead of retrained."""
    path = Path(repo_root) / REUSE_RULING
    if not path.is_file():
        return {"admissible": False, "n_admitted": 0,
                "why": f"{REUSE_RULING} is absent; no reuse may be assumed"}
    doc = json.loads(path.read_text())
    admitted = list(doc.get("admitted_reusable_probes") or ())
    return {
        "admissible": bool(doc.get("reuse_verified")) and bool(admitted),
        "n_admitted": len(admitted),
        "n_examined": doc.get("n_probes"),
        "record": REUSE_RULING,
        "why": ("every probe is trained fresh: the ruling records "
                f"reuse_verified={doc.get('reuse_verified')} with "
                f"{len(admitted)} admitted of {doc.get('n_probes')} examined"),
    }


@dataclass(frozen=True)
class ProbeSchedule:
    """How many probes each rung owes, before any result is visible.

    Registered prospectively: a candidate set or a rung plan that can grow once
    results are visible is not a preregistered plan.
    """

    top_k: int
    anchors: tuple[str, ...]
    sa_probes: int
    sb_probes: int
    sc_probes_worst_case: int

    @property
    def minimum(self) -> int:
        return self.sa_probes + self.sb_probes

    @property
    def worst_case(self) -> int:
        return self.minimum + self.sc_probes_worst_case

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k, "anchors": list(self.anchors),
            "sa_probes": self.sa_probes, "sb_probes": self.sb_probes,
            "sc_probes_worst_case": self.sc_probes_worst_case,
            "minimum_probes": self.minimum,
            "worst_case_probes": self.worst_case,
        }


def price(*, schedule: ProbeSchedule, price_per_hour: float,
          authorized_usd: float, repo_root: str | Path = REPO_ROOT,
          probes: int | None = None):
    """A `BudgetPlan` for one selection session. Priced, which is not funded.

    The expected path carries the rungs that always run; the conditional `sc`
    rung is a named `soft_stop_reserve`, which is what that field is for — a
    bounded risk that is not on the expected path.

    `plan_session` RAISES when the plan does not fit `authorized_usd`; that
    refusal is how a stage that has outgrown its budget says so.
    """
    from aadistill.infrastructure.budget import (
        MEASURED_STEP_SECONDS, Phase, StepTime, plan_session)

    cost = measured_probe_cost(repo_root)
    n = schedule.minimum if probes is None else probes
    phases = dict(SESSION_PHASE_MINUTES)
    conditional = round(
        schedule.sc_probes_worst_case * cost.minutes, 2)
    return plan_session(
        price_per_hour=price_per_hour, authorized_usd=authorized_usd,
        #: Probes are priced as measured PHASES, not as `arms x steps`: the
        #: measured per-probe minutes already include the whole train-and-score
        #: cycle, and re-deriving it from a step time would replace a
        #: measurement with a model.
        arms=0, steps_per_arm=0,
        step_time=StepTime(
            seconds=MEASURED_STEP_SECONDS,
            source=("unused: probes are priced from measured per-probe "
                    "minutes, so arms=0 and the step term is zero")),
        setup_minutes=phases["setup_and_asset_staging"],
        transfer_minutes=phases["bundle_transfer"],
        other_phases=(
            *(Phase(name, m) for name, m in SESSION_PHASE_MINUTES
              if name not in ("setup_and_asset_staging", "bundle_transfer")),
            Phase("recovery_probes", round(n * cost.train_minutes, 2)),
            Phase("battery_evaluation", round(n * cost.eval_minutes, 2)),
        ),
        contingency_fraction=CONTINGENCY_FRACTION,
        soft_stop_reserves=(Phase("conditional_sc_rung", conditional),),
        artifact_recovery_reserve_minutes=ARTIFACT_RECOVERY_RESERVE_MINUTES)


def report(*, schedule: ProbeSchedule, price_per_hour: float,
           repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Everything a maintainer needs to decide whether to fund the stage."""
    cost = measured_probe_cost(repo_root)
    plan = price(schedule=schedule, price_per_hour=price_per_hour,
                 authorized_usd=10_000.0, repo_root=repo_root)
    return {
        "probe_cost": cost.as_dict(),
        "reuse": reuse_is_admissible(repo_root),
        "schedule": schedule.as_dict(),
        "price_per_hour": price_per_hour,
        "expected_minutes": round(plan.expected_minutes, 2),
        "expected_usd": round(plan.expected_minutes / 60 * price_per_hour, 4),
        "hard_ceiling_minutes": round(plan.hard_terminate_minutes, 2),
        "hard_ceiling_usd": round(
            plan.hard_terminate_minutes / 60 * price_per_hour, 4),
        "_ceiling_includes": (
            "the conditional sc rung as a soft-stop reserve, a "
            f"{CONTINGENCY_FRACTION:.0%} contingency and a "
            f"{ARTIFACT_RECOVERY_RESERVE_MINUTES:.0f}-minute artifact recovery "
            "reserve"),
        "authorizes": "nothing",
    }


def main() -> int:
    from experiments.phase_c2.search_space import PRICE_PER_HOUR_LAST_QUOTED

    #: Illustrative only — the real schedule is fixed by the protocol document,
    #: which is what a launch would read.
    schedule = ProbeSchedule(
        top_k=5, anchors=("canonical_control", "frozen_c1_treatment_b"),
        sa_probes=7, sb_probes=4, sc_probes_worst_case=4)
    doc = report(schedule=schedule,
                 price_per_hour=PRICE_PER_HOUR_LAST_QUOTED)
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
