"""What the bounded C2 behavioural selection stage costs, derived and BOUNDED.

    PYTHONPATH=src:scripts python -m experiments.phase_c2.selection_pricing

Zero cost. It trains nothing, launches nothing and needs no GPU.

The full joint re-search produces a Top-K candidate set on a **cheap** metric.
Cheap-metric order is not behavioural order — E7 measured a `-5.22` nat NLL swing
that moved behaviour by `+0.0000` — so naming the C2 incumbent requires training
each admitted candidate under the frozen `0.86M` recovery recipe and scoring it
on the frozen **Phase-C/C1** battery under C1's statistical discipline.

**Every probe is trained fresh.** `logs/shared/analyses/autoinit_historical_probe_reuse.json`
records `reuse_verified: false` — all eleven historical probes fail
`scoring_contract_matches_live` — so there is no admissible reuse to net off.

**A mean is not a bound, and an earlier version of this module scaled one into a
field labelled a hard ceiling.** The repair is to use the per-probe timings that
exist: C1 attempt 18's driver emitted a `PROBE_TRAINED` and a `PROBE_SCORED`
marker per probe, so six observed training durations and six observed scoring
durations are recoverable from the committed log. The expected path uses their
MEANS; the ceiling is built on their MAXIMA plus a named reserve, in exactly the
structure the search pricing uses.

Two asymmetries in that evidence are worth stating, because they decide how much
reserve is honest:

* **training is effectively deterministic** — a fixed 860,000-token budget at a
  fixed block length produced a `1.003x` spread across six probes, so its
  observed maximum is a sound bound;
* **scoring is the variable half** — `1.241x` spread across the same six, because
  generation length depends on the checkpoint, and under P18 unrestricted
  generation it is bounded only by the effective context. Six probes from ONE
  experiment's two arms cannot bound what a different initialization will do, so
  the ceiling carries an explicit `generation_length_risk` reserve instead of
  pretending they can.
"""
from __future__ import annotations

import json
import re
import statistics
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
#: the frozen recipe AND scored them on the frozen Phase-C battery, so both
#: halves come from one run on one card — and the same workload this stage runs.
MEASURED_SOURCE = (
    "logs/stages/stage-1/phase_c1/runs/attempt18/evidence/c1_evidence.json")

#: The PER-PROBE marker stream from that same session. This is what makes a
#: bound possible rather than a mean: the stage record alone gives only two
#: aggregate boundaries.
MARKER_SOURCE = (
    "logs/stages/stage-1/phase_c1/runs/attempt18/evidence/driver_run.log")

#: The reuse ruling this price depends on. Read, never assumed.
REUSE_RULING = "logs/shared/analyses/autoinit_historical_probe_reuse.json"

TRAIN_STAGE = "G"
EVAL_STAGE = "H"
PRE_TRAIN_STAGE = "F"
TRAIN_MARKER = "PROBE_TRAINED"
SCORE_MARKER = "PROBE_SCORED"

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

#: How much MORE than its observed maximum one probe's scoring is allowed to
#: take, as a multiple of that maximum. `1.0` funds a doubling.
#:
#: This is a stated JUDGEMENT, not a measurement, and it is isolated here so it
#: can be argued with. Its basis: scoring varied `1.241x` across six probes drawn
#: from two arms of one experiment, and C2's probes are different initializations
#: whose generation length under P18 is bounded only by the effective context.
#: The generator does stop on semantic degeneration and `usable_rollout` screens
#: repetition, so the realistic tail is context-limit hits on some prompts rather
#: than unbounded decoding — a doubling covers that with room, and the reserve is
#: a soft stop rather than expected spend.
GENERATION_LENGTH_RISK_MULTIPLE = 1.0


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _markers(log: str, kind: str) -> list[tuple[datetime, str]]:
    return [(_utc(m.group(1)), m.group(2)) for m in
            re.finditer(rf"^(\S+Z) MARKER:{kind}:(\S+)$", log, re.M)]


def _consecutive_durations(marks: list[tuple[datetime, str]],
                           start: datetime) -> list[float]:
    """Minutes between consecutive completion markers, from `start`.

    The probes run sequentially, so a completion marker closes one probe and
    opens the next. This is per-probe wall clock including everything that probe
    paid for, which is the quantity that has to be bounded.
    """
    out: list[float] = []
    previous = start
    for stamp, _probe in marks:
        out.append((stamp - previous).total_seconds() / 60)
        previous = stamp
    return out


@dataclass(frozen=True)
class ProbeCost:
    """Per-probe minutes, with the MEAN and the observed MAXIMUM kept apart."""

    train_minutes_mean: float
    train_minutes_max: float
    eval_minutes_mean: float
    eval_minutes_max: float
    n_probes: int
    price_per_hour: float
    source: str

    @property
    def expected_minutes(self) -> float:
        return round(self.train_minutes_mean + self.eval_minutes_mean, 4)

    @property
    def bounding_minutes(self) -> float:
        """The observed per-probe worst case. The basis for the ceiling."""
        return round(self.train_minutes_max + self.eval_minutes_max, 4)

    @property
    def generation_length_reserve_minutes(self) -> float:
        return round(self.eval_minutes_max * GENERATION_LENGTH_RISK_MULTIPLE, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_probes_observed": self.n_probes,
            "train_minutes": {"mean": round(self.train_minutes_mean, 3),
                              "max": round(self.train_minutes_max, 3),
                              "spread": round(
                                  self.train_minutes_max
                                  / self.train_minutes_mean, 4)},
            "eval_minutes": {"mean": round(self.eval_minutes_mean, 3),
                             "max": round(self.eval_minutes_max, 3),
                             "spread": round(
                                 self.eval_minutes_max
                                 / self.eval_minutes_mean, 4)},
            "expected_minutes_per_probe": self.expected_minutes,
            "bounding_minutes_per_probe": self.bounding_minutes,
            "generation_length_reserve_minutes_per_probe":
                self.generation_length_reserve_minutes,
            "price_per_hour_observed": self.price_per_hour,
            "source": self.source,
            "_the_bound_is_not_the_mean": (
                "the expected path uses the MEANS and the ceiling is built on "
                "the observed MAXIMA plus a named generation-length reserve. An "
                "earlier version of this module scaled the mean into a field "
                "labelled a hard ceiling, which is not a bound."),
            "_why_training_max_is_a_sound_bound": (
                "a fixed 860,000-token budget at a fixed block length; six "
                "probes spread 1.003x, so the workload is effectively "
                "deterministic"),
            "_why_scoring_max_is_not": (
                "generation length depends on the checkpoint and under P18 is "
                "bounded only by the effective context. Six probes from one "
                f"experiment's two arms spread 1.241x; the "
                f"{GENERATION_LENGTH_RISK_MULTIPLE:.0%} reserve funds a "
                "doubling of the observed maximum for unseen initializations."),
        }


def measured_probe_cost(repo_root: str | Path = REPO_ROOT) -> ProbeCost:
    """Per-probe train and score cost, from the committed per-probe markers."""
    root = Path(repo_root)
    evidence_path, marker_path = root / MEASURED_SOURCE, root / MARKER_SOURCE
    for path, rel in ((evidence_path, MEASURED_SOURCE),
                      (marker_path, MARKER_SOURCE)):
        if not path.is_file():
            raise SelectionPricingError(
                f"{rel} is missing, so there is no measured probe cost to price "
                "from. Refusing to model one.")
    stages = json.loads(evidence_path.read_text())["stages"]
    for key in (PRE_TRAIN_STAGE, TRAIN_STAGE, EVAL_STAGE):
        if key not in stages:
            raise SelectionPricingError(
                f"the measured source has no stage {key!r}; the probe cost "
                "cannot be bounded without its endpoints")
    pre, train, evaluate = (stages[PRE_TRAIN_STAGE], stages[TRAIN_STAGE],
                            stages[EVAL_STAGE])

    log = marker_path.read_text()
    trained = _markers(log, TRAIN_MARKER)
    scored = _markers(log, SCORE_MARKER)
    n_trained = int(train["probes_trained"])
    n_scored = int(evaluate["probes_evaluated"])
    if not (len(trained) == n_trained and len(scored) == n_scored
            and n_trained == n_scored and n_trained > 0):
        raise SelectionPricingError(
            f"marker stream disagrees with the stage record: "
            f"{len(trained)}/{len(scored)} markers against "
            f"{n_trained}/{n_scored} recorded probes. A per-probe bound from a "
            "partial marker stream would understate the maximum.")

    train_minutes = _consecutive_durations(
        trained, _utc(pre["finished_utc"]))
    eval_minutes = _consecutive_durations(scored, trained[-1][0])

    #: Dollars come from the stage boundaries; the RATE is recovered from that
    #: session's own spend and wall clock rather than assumed from a quote.
    train_usd = float(train["spend_usd"]) - float(pre["spend_usd"])
    eval_usd = float(evaluate["spend_usd"]) - float(train["spend_usd"])
    total_minutes = sum(train_minutes) + sum(eval_minutes)
    rate = (train_usd + eval_usd) / (total_minutes / 60)

    return ProbeCost(
        train_minutes_mean=statistics.mean(train_minutes),
        train_minutes_max=max(train_minutes),
        eval_minutes_mean=statistics.mean(eval_minutes),
        eval_minutes_max=max(eval_minutes),
        n_probes=n_trained, price_per_hour=round(rate, 4),
        source=(f"{MEASURED_SOURCE} stages "
                f"{PRE_TRAIN_STAGE}/{TRAIN_STAGE}/{EVAL_STAGE}, with per-probe "
                f"{TRAIN_MARKER}/{SCORE_MARKER} markers from {MARKER_SOURCE}"))


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
    """The bounded two-stage C2 schedule, fixed before any candidate exists.

    Phase-C discipline is a PAIRED comparison against an incumbent, decided by a
    prompt-cluster bootstrap over a fixed seed block. C1 had two arms and needed
    no selection; C2 has `top_k` candidates, so the multiplicity has to be
    handled prospectively.

    **Screening is disjoint in BOTH dimensions.** A separate seed is not enough:
    C0's inferential unit is the prompt and it measured substantial same-prompt
    cross-seed dependence (ICC `0.25 +/- 0.095`; `P(correct | correct on another
    seed) = 0.257` against a `0.022` marginal). So screening runs on its own
    seed AND its own prompts, and the confirmation evidence is genuinely held
    out.

    **Confirmation is the selected candidate against B, and nothing else.** The
    original canonical control is NOT carried forward: for C2 the scientifically
    relevant comparator is the current incumbent. If a candidate beats the old
    control but loses to B it must not promote, and if it beats B the old control
    adds no promotion information — so a third recovery arm would buy nothing and
    cost three probes. Guardrails use the incumbent-relative semantics C1 already
    used, with B as the comparator arm.

    There is NO conditional rung: C0's terminal states are GO / NO-GO /
    INCONCLUSIVE with no forced winner and no fourth seed, so the probe count is
    exact.
    """

    top_k: int
    screening_seeds: int
    confirmation_seeds: int
    advanced_candidates: int
    #: The anchor in each rung. B in both; nothing else in either.
    screening_anchors: tuple[str, ...]
    confirmation_anchors: tuple[str, ...]
    #: Which battery each rung scores on. Different assets, by design.
    screening_battery: str
    confirmation_battery: str

    @property
    def screening_probes(self) -> int:
        return (self.top_k + len(self.screening_anchors)) * self.screening_seeds

    @property
    def confirmation_probes(self) -> int:
        return ((self.advanced_candidates + len(self.confirmation_anchors))
                * self.confirmation_seeds)

    @property
    def total_probes(self) -> int:
        return self.screening_probes + self.confirmation_probes

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "screening": {
                "seeds": self.screening_seeds,
                "arms": self.top_k + len(self.screening_anchors),
                "anchors": list(self.screening_anchors),
                "battery": self.screening_battery,
                "probes": self.screening_probes,
                "decides": (f"which {self.advanced_candidates} candidate(s) "
                            "advance. Ranking only — no veto is evaluated here "
                            "and no promotion can be claimed from it."),
                "ranking_rule": (
                    "maximize the paired single-seed "
                    "delta correct_overall(candidate - B) on the screening "
                    "battery. B is the anchor, never an advancing candidate. "
                    "usable_rollout is NOT positive ranking credit."),
                "tie_break": (
                    "the already-frozen full-search ordering, then the "
                    "deterministic state id. Both are fixed before any "
                    "behavioural datum exists."),
            },
            "confirmation": {
                "seeds": self.confirmation_seeds,
                "arms": self.advanced_candidates + len(
                    self.confirmation_anchors),
                "anchors": list(self.confirmation_anchors),
                "battery": self.confirmation_battery,
                "probes": self.confirmation_probes,
                "decides": ("the C2 incumbent, under the frozen Phase-C "
                            "decision rule. This is the only evidence that may "
                            "name one."),
            },
            "advanced_candidates": self.advanced_candidates,
            "total_probes": self.total_probes,
            "conditional_rung": None,
            "_no_conditional_rung": (
                "C0's terminal states are GO / NO-GO / INCONCLUSIVE with no "
                "forced winner and no fourth seed, so there is no tie-break "
                "rung to reserve for and the probe count is exact."),
            "_disjoint_in_both_dimensions": (
                "screening and confirmation use disjoint preregistered SEEDS "
                "and disjoint preregistered PROMPTS. Seeds alone would be "
                "insufficient: C0's inferential unit is the prompt and "
                "same-prompt cross-seed dependence is substantial, so selecting "
                "and confirming on the same prompts would let the selection "
                "leak through that dependence."),
            "_no_canonical_control": (
                "the original control is not a C2 arm. The comparator is the "
                "current incumbent B; a candidate that beats the old control "
                "but loses to B must not promote, and one that beats B gains "
                "nothing from the old control. Guardrails are "
                "incumbent-relative, as C1's were."),
        }


def eval_scale(schedule: ProbeSchedule,
               repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """How the two batteries' sizes scale the measured per-probe scoring cost.

    The measured scoring minutes come from probes scored on
    `c1_confirmation_v1`. A screening probe is scored on a DIFFERENT asset, so
    its cost is scaled by scorable-prompt count rather than assumed equal. When
    the two are the same size the factor is exactly 1 and the arithmetic is a
    no-op — which is the case today, because the screening battery deliberately
    preserves C1's mixture and size.
    """
    root = Path(repo_root)
    sizes = {}
    for role, rel in (("screening", schedule.screening_battery),
                      ("confirmation", schedule.confirmation_battery)):
        path = root / rel / "manifest.json"
        if not path.is_file():
            raise SelectionPricingError(
                f"{rel}/manifest.json is missing, so the {role} rung's "
                "evaluation cost cannot be scaled to the battery it scores on.")
        sizes[role] = int(json.loads(path.read_text())["n_scorable_prompts"])
    factor = sizes["screening"] / sizes["confirmation"]
    return {"scorable_prompts": sizes, "screening_eval_factor": round(factor, 6),
            "_basis": ("measured scoring minutes were observed on the "
                       "confirmation battery; the screening rung is scaled by "
                       "scorable-prompt count"),
            "_is_a_no_op_today": factor == 1.0}


def price(*, schedule: ProbeSchedule, price_per_hour: float,
          authorized_usd: float, repo_root: str | Path = REPO_ROOT):
    """A `BudgetPlan` for one selection session. Priced, which is not funded.

    The expected path carries the probes at their MEAN observed minutes. Two
    named soft-stop reserves lift the ceiling: `probe_duration_risk` to the
    observed per-probe maximum, and `generation_length_risk` for the fact that
    six probes from one experiment cannot bound an unseen checkpoint's decoding.

    `plan_session` RAISES when the plan does not fit `authorized_usd`; that
    refusal is how a stage that has outgrown its budget says so.
    """
    from aadistill.infrastructure.budget import (
        MEASURED_STEP_SECONDS, Phase, StepTime, plan_session)

    cost = measured_probe_cost(repo_root)
    scale = eval_scale(schedule, repo_root)
    n = schedule.total_probes
    #: Training is battery-independent; only scoring scales with the asset.
    train_mean = n * cost.train_minutes_mean
    eval_mean = (schedule.screening_probes * cost.eval_minutes_mean
                 * scale["screening_eval_factor"]
                 + schedule.confirmation_probes * cost.eval_minutes_mean)
    eval_max = (schedule.screening_probes * cost.eval_minutes_max
                * scale["screening_eval_factor"]
                + schedule.confirmation_probes * cost.eval_minutes_max)
    train_max = n * cost.train_minutes_max
    phases = dict(SESSION_PHASE_MINUTES)
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
            Phase("recovery_probes", round(train_mean, 2)),
            Phase("battery_evaluation", round(eval_mean, 2)),
        ),
        contingency_fraction=CONTINGENCY_FRACTION,
        soft_stop_reserves=(
            Phase("probe_duration_risk",
                  round((train_max + eval_max) - (train_mean + eval_mean), 2)),
            Phase("generation_length_risk",
                  round(eval_max * GENERATION_LENGTH_RISK_MULTIPLE, 2)),
        ),
        artifact_recovery_reserve_minutes=ARTIFACT_RECOVERY_RESERVE_MINUTES)


def report(*, schedule: ProbeSchedule, price_per_hour: float,
           repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Everything a maintainer needs to decide whether to fund the stage."""
    cost = measured_probe_cost(repo_root)
    plan = price(schedule=schedule, price_per_hour=price_per_hour,
                 authorized_usd=10_000.0, repo_root=repo_root)
    scale = eval_scale(schedule, repo_root)
    n = schedule.total_probes
    return {
        "probe_cost": cost.as_dict(),
        "battery_scaling": scale,
        "reuse": reuse_is_admissible(repo_root),
        "schedule": schedule.as_dict(),
        "price_per_hour": price_per_hour,
        "n_probes": n,
        "expected_minutes": round(plan.expected_minutes, 2),
        "expected_usd": round(plan.expected_minutes / 60 * price_per_hour, 4),
        "hard_ceiling_minutes": round(plan.hard_terminate_minutes, 2),
        "hard_ceiling_usd": round(
            plan.hard_terminate_minutes / 60 * price_per_hour, 4),
        "bounding_basis": {
            "probe_minutes_expected": round(
                n * cost.train_minutes_mean
                + (schedule.screening_probes * scale["screening_eval_factor"]
                   + schedule.confirmation_probes) * cost.eval_minutes_mean, 2),
            "probe_minutes_observed_max": round(
                n * cost.train_minutes_max
                + (schedule.screening_probes * scale["screening_eval_factor"]
                   + schedule.confirmation_probes) * cost.eval_minutes_max, 2),
            "generation_length_reserve_minutes": round(
                (schedule.screening_probes * scale["screening_eval_factor"]
                 + schedule.confirmation_probes) * cost.eval_minutes_max
                * GENERATION_LENGTH_RISK_MULTIPLE, 2),
            "_rule": (
                "expected path on per-probe MEANS; ceiling lifted to the "
                "observed per-probe MAXIMUM by the probe_duration_risk reserve, "
                "then again by generation_length_risk for unseen checkpoints. "
                "No mean is presented as a maximum."),
        },
        "_ceiling_includes": (
            f"both named reserves, a {CONTINGENCY_FRACTION:.0%} contingency and "
            f"a {ARTIFACT_RECOVERY_RESERVE_MINUTES:.0f}-minute artifact recovery "
            "reserve"),
        "authorizes": "nothing",
    }


def main() -> int:
    from experiments.phase_c2.search_space import PRICE_PER_HOUR_LAST_QUOTED

    #: Illustrative only — the real schedule is fixed by the protocol document,
    #: which is what a launch would read.
    schedule = ProbeSchedule(
        top_k=5, screening_seeds=1, confirmation_seeds=3,
        advanced_candidates=1,
        screening_anchors=("frozen_c1_treatment_b",),
        confirmation_anchors=("frozen_c1_treatment_b",),
        screening_battery="artifacts/stage3/c2_screening_v1",
        confirmation_battery="artifacts/stage3/c1_confirmation_v1")
    print(json.dumps(report(schedule=schedule,
                            price_per_hour=PRICE_PER_HOUR_LAST_QUOTED),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
