"""Four separate cost thresholds for a paid GPU session.

E6b billed $7.68 against a $7.12 authorization. The proximate cause was a step
time priced from E4's comparable arms (3.625 s/step) against the 4.15 s/step the
run actually sustained — a 14% miss, ~$0.81 of unbudgeted time. The structural
cause was that the session had **one** number: the authorization was also the
kill point, so by the time the deadline mattered there was no time left in which
to bundle, transfer and verify artifacts, and no earlier point at which the
session was supposed to stop starting new work.

This module replaces that single number with four, which must be kept distinct
because they answer different questions:

``expected``
    What the session should cost if nothing goes wrong. The number quoted in an
    authorization request.
``soft_stop``
    The point past which no *new* phase may start. Work already running
    continues; nothing new begins. This is the threshold a driver consults.
``artifact_recovery_reserve``
    Time deliberately held back **after** the soft stop so that bundling,
    hashing, transfer and verification have somewhere to happen. E6b had no such
    reserve, which is why a late run and a lost event stream arrived together.
``hard_terminate``
    The absolute provider-side kill point, enforced by
    :mod:`aadistill.infrastructure.watchdog` against the provider API. It is
    ``soft_stop + reserve`` and it must land **at or under** the authorization.

Budget enforcement reserves teardown time rather than setting the kill point at
the nominal dollar cap (decision record 2026-08-09).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


class BudgetError(ValueError):
    """A session plan that cannot be executed within its authorization."""


# The step time E6b actually sustained: L40S, Stage 3 ladder arms, 2916 steps,
# bf16, teacher-in-memory KD. Both arms, mean of the per-10-step console
# timings. Any Stage 3 L40S estimate starts here.
#
# 3.625 s/step — the figure derived from E4's comparable arms and used to price
# E6b — is superseded. It is recorded in `SUPERSEDED_STEP_SECONDS` so that a
# plan reaching for it is rejected by name rather than by an anonymous bound.
MEASURED_STEP_SECONDS = 4.15
SUPERSEDED_STEP_SECONDS = 3.625


@dataclass(frozen=True)
class StepTime:
    """A step-time estimate together with where it came from.

    The `source` is not decoration. E6b's overrun is legible only because the
    provenance of 3.625 s/step was recorded; a bare float would have made the
    miss untraceable. A plan refuses an estimate that does not carry one.
    """

    seconds: float
    source: str
    measured: bool = True

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise BudgetError(f"step seconds must be positive, got {self.seconds}")
        if not self.source.strip():
            raise BudgetError(
                "a step-time estimate must record its source; an unsourced "
                "estimate is how E6b came to be priced from the wrong run")


@dataclass(frozen=True)
class Phase:
    """One named contribution to the expected session length."""

    name: str
    minutes: float


@dataclass(frozen=True)
class BudgetPlan:
    """The four thresholds, in minutes from session start and in dollars.

    Elapsed time is measured from the **first pod create of the session**, not
    from the current pod: a cold-host redraw replaces the pod but not the meter.
    """

    price_per_hour: float
    authorized_usd: float
    expected_minutes: float
    soft_stop_minutes: float
    artifact_recovery_reserve_minutes: float
    hard_terminate_minutes: float
    step_time: StepTime
    breakdown: tuple[Phase, ...] = field(default_factory=tuple)
    #: Named reserves folded into the soft stop after the contingency multiplier.
    #: Kept as named entries rather than one number so a session record says what
    #: each one is for and a reader can check it against its derivation.
    soft_stop_reserves: tuple[Phase, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def usd_at(self, minutes: float) -> float:
        return minutes / 60.0 * self.price_per_hour

    @property
    def expected_usd(self) -> float:
        return self.usd_at(self.expected_minutes)

    @property
    def soft_stop_usd(self) -> float:
        return self.usd_at(self.soft_stop_minutes)

    @property
    def artifact_recovery_reserve_usd(self) -> float:
        return self.usd_at(self.artifact_recovery_reserve_minutes)

    @property
    def hard_terminate_usd(self) -> float:
        return self.usd_at(self.hard_terminate_minutes)

    def phase_at(self, elapsed_minutes: float) -> str:
        """Which regime the session is in, for a driver deciding what to do next.

        ``nominal``            — start whatever is next.
        ``over_expected``      — behind schedule; finish, start nothing large.
        ``artifact_recovery``  — past the soft stop: stop starting work, collect.
        ``terminate``          — past the hard limit: the watchdog kills the pod.
        """
        if elapsed_minutes >= self.hard_terminate_minutes:
            return "terminate"
        if elapsed_minutes >= self.soft_stop_minutes:
            return "artifact_recovery"
        if elapsed_minutes >= self.expected_minutes:
            return "over_expected"
        return "nominal"

    def may_start(self, elapsed_minutes: float, phase_minutes: float) -> bool:
        """May a phase of `phase_minutes` start now and still leave the reserve?

        This is the gate E6b did not have. The driver re-priced before each arm
        against the *authorization*, so an arm that would finish just under the
        cap was allowed to start — leaving nothing for teardown.
        """
        return elapsed_minutes + phase_minutes <= self.soft_stop_minutes

    def as_dict(self) -> dict:
        d = asdict(self)
        d["breakdown"] = [asdict(p) for p in self.breakdown]
        d["soft_stop_reserves"] = [asdict(r) for r in self.soft_stop_reserves]
        d["soft_stop_reserve_minutes"] = sum(r.minutes
                                             for r in self.soft_stop_reserves)
        d["step_time"] = asdict(self.step_time)
        d["expected_usd"] = round(self.expected_usd, 4)
        d["soft_stop_usd"] = round(self.soft_stop_usd, 4)
        d["artifact_recovery_reserve_usd"] = round(
            self.artifact_recovery_reserve_usd, 4)
        d["hard_terminate_usd"] = round(self.hard_terminate_usd, 4)
        return d


def plan_session(
    *,
    price_per_hour: float,
    authorized_usd: float,
    arms: int,
    steps_per_arm: int,
    step_time: StepTime,
    setup_minutes: float,
    eval_minutes_per_arm: float = 0.0,
    transfer_minutes: float = 0.0,
    other_phases: tuple[Phase, ...] = (),
    contingency_fraction: float = 0.10,
    soft_stop_reserves: tuple[Phase, ...] = (),
    artifact_recovery_reserve_minutes: float = 30.0,
    step_time_floor: float = MEASURED_STEP_SECONDS,
    below_floor_reason: str = "",
) -> BudgetPlan:
    """Price a session and place its four thresholds.

    `step_time_floor` defaults to the E6b measurement because that is the only
    Stage 3 L40S number this project has *observed*. A different workload may
    legitimately be faster, but it must say so: passing a `step_time` under the
    floor requires `below_floor_reason`, which lands in the plan's notes and
    therefore in the session record. This is what makes "no future launch may be
    estimated from E4's 3.6 s/step alone" enforceable rather than aspirational.

    Raises `BudgetError` when the hard terminate point exceeds the
    authorization. It never silently shrinks the experiment to fit: the caller
    must either obtain a larger authorization or choose a smaller run, and both
    are the maintainer's decision, not this function's.
    """
    if arms < 0 or steps_per_arm < 0:
        raise BudgetError("arms and steps_per_arm must be non-negative")
    if price_per_hour <= 0:
        raise BudgetError("price_per_hour must be positive")
    if artifact_recovery_reserve_minutes <= 0:
        raise BudgetError(
            "the artifact-recovery reserve must be positive: a session with no "
            "reserve is E6b, which reached its deadline with the event stream "
            "still on the pod")
    if contingency_fraction < 0:
        raise BudgetError("contingency_fraction must be non-negative")

    notes: list[str] = []
    if step_time.seconds < step_time_floor:
        if not below_floor_reason.strip():
            raise BudgetError(
                f"step time {step_time.seconds} s/step is below the measured "
                f"floor of {step_time_floor} s/step ({step_time.source!r}). "
                "E6b was priced at 3.625 s/step from E4 and sustained 4.15; "
                "pass below_floor_reason to record why this workload is "
                "genuinely faster.")
        notes.append(
            f"below-floor step time accepted: {step_time.seconds} s/step vs "
            f"floor {step_time_floor} — {below_floor_reason}")
    if abs(step_time.seconds - SUPERSEDED_STEP_SECONDS) < 1e-9:
        notes.append(
            "step time equals the superseded E4-derived 3.625 s/step; this is "
            "the figure that under-priced E6b by 14%")

    train_minutes = arms * steps_per_arm * step_time.seconds / 60.0
    eval_minutes = arms * eval_minutes_per_arm
    phases: list[Phase] = [
        Phase("setup", setup_minutes),
        Phase("train", train_minutes),
        Phase("evaluate", eval_minutes),
        Phase("transfer", transfer_minutes),
        *other_phases,
    ]
    expected = sum(p.minutes for p in phases)
    # Reserves for identified, bounded risks that are NOT part of the expected
    # path, added AFTER the contingency multiplier and BEFORE the soft stop.
    #
    # The placement is the point. A reserve added as a phase would inflate the
    # expected figure and be multiplied by the contingency, and one added after
    # the soft stop would not protect the work at all: the driver's `afford()`
    # refuses to START anything that would cross the soft stop, so a risk that
    # materializes EARLY in a session — Phase A's reference-cache fallback is
    # consumed entirely inside stage 1 — would be paid for out of the later
    # stages' budget and silently truncate them. A reserve that only moves the
    # watchdog's kill time protects the pod, not the experiment.
    reserve_minutes = sum(r.minutes for r in soft_stop_reserves)
    if any(r.minutes < 0 for r in soft_stop_reserves):
        raise BudgetError("a soft-stop reserve cannot be negative")
    soft_stop = expected * (1.0 + contingency_fraction) + reserve_minutes
    hard = soft_stop + artifact_recovery_reserve_minutes

    plan = BudgetPlan(
        price_per_hour=price_per_hour,
        authorized_usd=authorized_usd,
        expected_minutes=expected,
        soft_stop_minutes=soft_stop,
        artifact_recovery_reserve_minutes=artifact_recovery_reserve_minutes,
        hard_terminate_minutes=hard,
        step_time=step_time,
        breakdown=tuple(phases),
        soft_stop_reserves=tuple(soft_stop_reserves),
        notes=tuple(notes),
    )
    if plan.hard_terminate_usd > authorized_usd:
        shortfall = plan.hard_terminate_usd - authorized_usd
        raise BudgetError(
            f"the plan terminates at ${plan.hard_terminate_usd:.2f} "
            f"({hard:.0f} min at ${price_per_hour:.3f}/h) but only "
            f"${authorized_usd:.2f} is authorized — a shortfall of "
            f"${shortfall:.2f}. Expected ${plan.expected_usd:.2f}, soft stop "
            f"${plan.soft_stop_usd:.2f}, recovery reserve "
            f"${plan.artifact_recovery_reserve_usd:.2f}. Ask for "
            f"${plan.hard_terminate_usd:.2f} or reduce the run; do not shrink "
            "it silently to fit.")
    return plan
