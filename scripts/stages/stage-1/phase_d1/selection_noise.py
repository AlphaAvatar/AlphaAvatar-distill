"""What screening does to the number it reports, and how often it picks right.

Two different quantities, kept apart here because conflating them produced a
wrong validity argument in this module's first version.

**1. The winner's curse ON THE SCREENING ESTIMATE.** A screening rung that
advances the best of `K` candidates reports a maximum over `K` noisy estimates,
and the maximum of noisy estimates is biased upward even when every candidate is
exactly as good as the anchor. So the *screening* number overstates the advancing
candidate by a derivable amount. That is a fact about the screening estimate and
nothing else.

**2. Discrimination.** How often the rung advances a candidate that really is
better. This is what a screening design is FOR, and it is unrelated to (1)
except that both move with `K`.

**What this module does NOT claim, stated because it did.** An earlier version
filtered designs on `selection_bias < SESOI` and described that as the condition
for the confirmation rung to be interpretable. **It is not.** When confirmation
runs on genuinely fresh, disjoint prompts and fresh seeds, the confirmation
estimate is unbiased under the global null regardless of how inflated the
screening estimate was — the upward bias lives in the screening draw, and a fresh
independent draw does not inherit it. Screening selects WHICH candidate is
confirmed; it does not bias the confirmation measurement of that candidate. The
figures below are therefore planning and sensitivity analysis, never an
admissibility rule, and nothing in the D1 design is chosen by comparing them to
the SESOI.

**The quantity that actually governs whether D1 can find something.** It is

    P(a behaviourally good candidate is in the Top-K at all)
      x  P(screening advances it | it is in the Top-K)

The second factor is what `advance_probability` models. **The first factor is
UNKNOWN**: it depends on how well the search's own state-eval ranking predicts
behaviour, and C2 is the only evidence this project has on that — C2's beam
ranking did not identify a behaviourally better candidate, which is a reason not
to assume the ranking is a strong behavioural predictor. No number in this module
estimates the first factor, and the product must not be quoted as though the
first were 1.

**The measured input, and how little of it there is.** A3 ran three paired seeds
on the frozen confirmation battery and reported per-seed `correct_overall` deltas
of `+0.012941`, `-0.011765` and `+0.007059` against a pooled `+0.002745` — a
sample standard deviation of `0.012910` from **n = 3**. A standard deviation from
three observations is a weak estimate: its own 95% sampling interval spans a
factor of about twelve (see `SEED_SD_INTERVAL`). C3 independently measured a
seed-level spread of `±0.022353` on arms whose pooled differences were within
`0.001961` of zero, which is the same magnitude read as a range rather than a
deviation. Every derived probability is therefore reported across that interval,
and no single figure from this module is a precise power claim.

Nothing here is specific to D1: it is the arithmetic of screening `K` arms on `m`
seeds at a measured noise level. It lives under `experiments/` rather than in
`src/aadistill` because the measured constant is an instance fact about this
project's battery and recovery recipe, not a property of the framework.
"""
from __future__ import annotations

import math
from functools import lru_cache

#: Sample standard deviation of the paired per-seed `correct_overall` delta, from
#: A3's three frozen C3 seeds on `c1_confirmation_v1`. Derived from the three
#: published deltas rather than quoted, by `_a3_seed_sd` below, so a reader can
#: check it against the closeout.
A3_SEED_DELTAS: tuple[float, ...] = (0.012941, -0.011765, 0.007059)

#: C3's independently measured seed-level spread, as a RANGE. Kept as a
#: cross-check, not as an input: a range and a standard deviation are different
#: statistics and averaging them would be meaningless.
C3_SEED_LEVEL_SPREAD_RANGE = 0.022353

#: C0's decision boundary. Used below ONLY as the effect size at which
#: discrimination is evaluated — "how often does screening advance a candidate
#: that is better by the SESOI". It is NOT a threshold any bias figure is
#: compared against; see the module docstring.
SESOI = 0.010

#: Chi-square quantiles at 2 degrees of freedom (n = 3), for the sampling
#: interval of a standard deviation estimated from three observations. Standard
#: table values, named so the interval below is derived rather than asserted.
_CHI2_DF2_LOWER_025 = 0.0506356
_CHI2_DF2_UPPER_975 = 7.377759

#: `E[max of k standard normals]`, by numerical integration rather than a table,
#: so a design that wants an unlisted `k` gets a real number instead of an
#: interpolation. Exact for k=1 (0) and k=2 (1/sqrt(pi) = 0.5642).
_INTEGRATION_STEPS = 200_000
_INTEGRATION_LIMIT = 12.0


def _a3_seed_sd() -> float:
    """Sample standard deviation of A3's three per-seed deltas."""
    n = len(A3_SEED_DELTAS)
    mean = sum(A3_SEED_DELTAS) / n
    return math.sqrt(sum((d - mean) ** 2 for d in A3_SEED_DELTAS) / (n - 1))


#: The measured input, derived at import so it cannot drift from the deltas.
SEED_SD = _a3_seed_sd()


def seed_sd_interval(deltas: tuple[float, ...] = A3_SEED_DELTAS) -> tuple[float, float]:
    """95% sampling interval for the per-seed standard deviation.

    From the chi-square distribution of `(n-1)s^2/sigma^2`. At `n = 3` the
    interval spans roughly `0.52 s` to `6.3 s` — a factor of twelve — which is
    the honest width of what three observations say about a spread. Every
    probability this module reports is evaluated across it, so a reader cannot
    mistake a point estimate for a measurement.
    """
    n = len(deltas)
    if n < 2:
        raise ValueError("a standard deviation needs at least two observations")
    mean = sum(deltas) / n
    ss = sum((d - mean) ** 2 for d in deltas)
    return (math.sqrt(ss / _CHI2_DF2_UPPER_975) if n == 3
            else math.sqrt(ss / _CHI2_DF2_UPPER_975),
            math.sqrt(ss / _CHI2_DF2_LOWER_025))


#: The interval, derived at import beside the point estimate it qualifies.
SEED_SD_INTERVAL = seed_sd_interval()


@lru_cache(maxsize=None)
def expected_max_of_standard_normals(k: int) -> float:
    """`E[max(Z_1..Z_k)]` for standard normal `Z`.

    By the survival-function identity

        E[max] = int_0^inf (1 - F(x)^k) dx  -  int_0^inf F(-x)^k dx

    which is numerically well behaved for the small `k` a screening rung uses and
    is checked against the two closed forms — `0` at `k=1` and `1/sqrt(pi)` at
    `k=2` — by :func:`self_check`. The check exists because the first version of
    this function wrote the negative term as `1 - (1 - F(-x))^k`, which is
    algebraically the positive term again: the two cancelled and every figure
    came out exactly zero, which read as "no design has a selection problem".
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return 0.0
    step = _INTEGRATION_LIMIT / _INTEGRATION_STEPS
    total = 0.0
    root_two = math.sqrt(2.0)
    for i in range(_INTEGRATION_STEPS):
        x = (i + 0.5) * step
        upper = 0.5 * (1.0 + math.erf(x / root_two))        # F(x)
        lower = 0.5 * (1.0 + math.erf(-x / root_two))       # F(-x)
        total += ((1.0 - upper ** k) - lower ** k) * step
    return total


#: The two values the integration is checked against. `k=2` is `1/sqrt(pi)`.
CLOSED_FORMS: dict[int, float] = {1: 0.0, 2: 1.0 / math.sqrt(math.pi)}


def self_check(tolerance: float = 1e-6) -> dict[int, float]:
    """Verify the integration against its closed forms, or raise.

    Called by `report()` so a broken quadrature cannot produce a design table.
    A figure of exactly zero everywhere is indistinguishable from "screening is
    free", and that is the one wrong answer this function must not return
    quietly.
    """
    out = {}
    for k, expected in CLOSED_FORMS.items():
        got = expected_max_of_standard_normals(k)
        if abs(got - expected) > tolerance:
            raise ValueError(
                f"E[max of {k} standard normals] integrated to {got!r} but the "
                f"closed form is {expected!r}; the quadrature is wrong and every "
                "figure derived from it would be too")
        out[k] = got
    return out


def screening_estimate_inflation(top_k: int, screening_seeds: int, *,
                                 seed_sd: float = SEED_SD) -> float:
    """How much the SCREENING estimate overstates the advancing candidate.

    Under the global null — every candidate exactly as good as the anchor — the
    advancing candidate's *screening* delta has expectation
    `E[max of K] * seed_sd / sqrt(m)` rather than zero. This is the winner's
    curse, and its scope is exactly the screening number: it says the screening
    delta must not be read as an effect estimate, and it says nothing about the
    confirmation rung, which measures the advanced candidate on fresh disjoint
    prompts and seeds and is unbiased under that null.

    Two consequences, both of which the design observes:

    * the screening delta is never reported as D1's effect estimate, and never
      promotes anything by itself;
    * this figure is not compared to the SESOI, and no design is admitted or
      rejected by it.

    `screening_seeds == 0` means no screening rung at all: the search's own cheap
    metric nominates and there is no behavioural maximum, so the inflation is
    zero. That is not an argument for skipping screening — it moves the risk from
    an inflated screening number to an unfiltered nomination — it is the honest
    zero.

    The independence assumption is optimistic: candidates from one search share
    prompts and a parent, so their errors are positively correlated, which
    REDUCES the spread of the maximum. The figure is an upper bound.
    """
    if screening_seeds < 1:
        return 0.0
    sd = seed_sd / math.sqrt(screening_seeds)
    return round(expected_max_of_standard_normals(top_k) * sd, 6)


#: Draws for the discrimination estimate. Deterministic: the generator is seeded
#: from a fixed constant so the table is a property of the arithmetic rather than
#: of when it ran. 200k is enough for three decimal places and runs in seconds.
_DISCRIMINATION_DRAWS = 200_000
_DISCRIMINATION_SEED = 20261003


#: MEMOIZED. Both of these are pure functions of their arguments — the
#: integration is deterministic and the simulation is seeded from a constant —
#: so a cache changes no value. It changes the cost: the sensitivity analysis
#: evaluates each design at three spreads, and `report()` tabulates eighteen
#: designs, which took 170 s of 200k-draw simulations per call before this.
@lru_cache(maxsize=None)
def advance_probability(top_k: int, screening_seeds: int, *,
                        effect: float = SESOI,
                        seed_sd: float = SEED_SD) -> float:
    """P(screening advances the better candidate | it is among the `top_k`).

    **The conditioning is the whole point, and it is not a formality.** This is
    the second factor of

        P(a good candidate is in the Top-K)  x  P(advance | in the Top-K)

    and only the second. The first depends on how well the search's state-eval
    ranking predicts behaviour, which this project has not measured and which
    C2's result gives reason to doubt. A design cannot claim the product from
    this number alone.

    The model is deliberately the simplest one that captures the trade: one
    candidate is better than the anchor by `effect`, the other `top_k - 1` are
    exactly as good as it, and each candidate's screening estimate is its true
    delta plus zero-mean noise at `seed_sd / sqrt(screening_seeds)`. A `top_k` of
    1 advances its only candidate by construction — which is a tautology, not a
    perfect screen.
    """
    if screening_seeds < 1 or top_k < 1:
        return 1.0 if top_k == 1 else 0.0
    if top_k == 1:
        return 1.0
    import random

    sd = seed_sd / math.sqrt(screening_seeds)
    rng = random.Random(_DISCRIMINATION_SEED + top_k * 100 + screening_seeds)
    wins = 0
    for _ in range(_DISCRIMINATION_DRAWS):
        best = effect + rng.gauss(0.0, sd)
        if all(best > rng.gauss(0.0, sd) for _ in range(top_k - 1)):
            wins += 1
    return round(wins / _DISCRIMINATION_DRAWS, 4)


def advance_probability_sensitivity(top_k: int, screening_seeds: int, *,
                                    effect: float = SESOI) -> dict[str, float]:
    """`advance_probability` across the standard deviation's own interval.

    The point estimate rests on three observations. Reporting it alone would
    give a number like `0.7808` the appearance of a power calculation; reporting
    it beside the values implied by the low and high ends of the sampling
    interval shows what it actually is — a planning figure whose uncertainty is
    larger than the differences between candidate designs.
    """
    low, high = SEED_SD_INTERVAL
    return {
        "sd_low": round(low, 6),
        "sd_point": round(SEED_SD, 6),
        "sd_high": round(high, 6),
        "p_at_sd_low": advance_probability(top_k, screening_seeds, effect=effect,
                                           seed_sd=low),
        "p_at_sd_point": advance_probability(top_k, screening_seeds,
                                             effect=effect, seed_sd=SEED_SD),
        "p_at_sd_high": advance_probability(top_k, screening_seeds, effect=effect,
                                            seed_sd=high),
    }


def pipeline_detection_probability(top_k: int, screening_seeds: int, *,
                                   p_in_top_k: float,
                                   effect: float = SESOI) -> float:
    """The product, which requires the caller to SUPPLY the unknown factor.

    There is no default for `p_in_top_k`, deliberately: the only way to quote a
    pipeline probability is to state what you assumed about the search ranking's
    behavioural predictiveness, and this project has not measured it. Callers
    report a range (see `report`), never one number.
    """
    if not 0.0 <= p_in_top_k <= 1.0:
        raise ValueError(f"p_in_top_k must be a probability, got {p_in_top_k}")
    return round(p_in_top_k * advance_probability(
        top_k, screening_seeds, effect=effect), 4)


#: The assumed values of the unknown factor that the report tabulates. A range
#: chosen to span "the ranking is uninformative about behaviour" through "the
#: ranking always puts a good candidate in the Top-K"; the true value is unknown
#: and C2 argues against the top of the range.
P_IN_TOP_K_SENSITIVITY: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

#: WHAT MAY AND MAY NOT BE CLAIMED FROM THIS MODULE. Carried into the design
#: record so the boundary travels with the numbers rather than living in a
#: docstring a reader may not open.
CLAIM_BOUNDARY = (
    "The inflation figures describe the winner's curse on the SCREENING "
    "estimate only. They are not a validity condition for the confirmation "
    "rung: with fresh disjoint prompts and fresh seeds the confirmation "
    "estimate is unbiased under the global null whatever the screening "
    "inflation was, and no design here is admitted or rejected by comparing "
    "inflation to the SESOI. advance_probability is conditional on the good "
    "candidate being in the Top-K; the unconditional probability also needs "
    "P(good candidate in Top-K), which is UNKNOWN and which C2's negative "
    "result argues is not close to 1. The per-seed standard deviation comes "
    "from three A3 deltas, so every probability is planning and sensitivity "
    "analysis, not a power calculation."
)


def report() -> dict:
    """Both quantities for every design, with the interval and the boundary."""
    checked = self_check()
    rows = []
    for top_k in (1, 2, 3, 4, 5, 6):
        for seeds in (1, 2, 3):
            rows.append({
                "top_k": top_k, "screening_seeds": seeds,
                "screening_estimate_inflation": screening_estimate_inflation(
                    top_k, seeds),
                "advance_probability": advance_probability(top_k, seeds),
                "advance_probability_sensitivity":
                    advance_probability_sensitivity(top_k, seeds),
                "screening_probes": (top_k + 1) * seeds,
            })
    chosen = {"top_k": 2, "screening_seeds": 2}
    return {
        "schema": "aadistill.autoinit.screening_selection_noise/v2",
        "seed_sd": round(SEED_SD, 6),
        "seed_sd_n": len(A3_SEED_DELTAS),
        "seed_sd_interval_95": [round(SEED_SD_INTERVAL[0], 6),
                                round(SEED_SD_INTERVAL[1], 6)],
        "seed_sd_source": ("A3's three paired per-seed correct_overall deltas on "
                           "c1_confirmation_v1, logs/stages/stage-1/phase_a3/"
                           "analyses/a3_closeout.md"),
        "seed_sd_cross_check_range": C3_SEED_LEVEL_SPREAD_RANGE,
        "sesoi": SESOI,
        "sesoi_role": ("the effect size at which discrimination is evaluated; "
                       "NOT a threshold any inflation figure is compared to"),
        "independence_assumption": (
            "candidates from one search share prompts and a parent, so their "
            "errors are positively correlated and the true spread of the "
            "maximum is SMALLER. The inflation figures are upper bounds."),
        "quadrature_self_check": {str(k): v for k, v in checked.items()},
        "unknown_factor": {
            "name": "P(a behaviourally good candidate is in the Top-K)",
            "status": "UNMEASURED",
            "why": ("it depends on how well the search's state-eval ranking "
                    "predicts behaviour. C2's beam ranking did not identify a "
                    "behaviourally better candidate, which is a reason not to "
                    "assume the ranking is a strong behavioural predictor."),
            "pipeline_probability_at_assumed_values": {
                str(p): pipeline_detection_probability(p_in_top_k=p, **chosen)
                for p in P_IN_TOP_K_SENSITIVITY},
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "rows": rows,
    }


def main() -> int:
    import json

    doc = report()
    print(f"per-seed sd of the paired delta: {doc['seed_sd']:.6f} "
          f"from n={doc['seed_sd_n']}")
    print(f"  95% sampling interval        : {doc['seed_sd_interval_95'][0]:.6f} "
          f"– {doc['seed_sd_interval_95'][1]:.6f}  (a factor of "
          f"{doc['seed_sd_interval_95'][1] / doc['seed_sd_interval_95'][0]:.1f})")
    print(f"SESOI: {doc['sesoi']} — {doc['sesoi_role']}\n")
    print(f"  {'K':>3} {'seeds':>6} {'probes':>7} {'inflation':>10} "
          f"{'P(adv|inK)':>11} {'P at sd lo/hi':>16}")
    for row in doc["rows"]:
        s = row["advance_probability_sensitivity"]
        print(f"  {row['top_k']:>3} {row['screening_seeds']:>6} "
              f"{row['screening_probes']:>7} "
              f"{row['screening_estimate_inflation']:>10.6f} "
              f"{row['advance_probability']:>11.4f} "
              f"{s['p_at_sd_low']:>7.4f}/{s['p_at_sd_high']:<8.4f}")
    print("\n  inflation   = winner's curse on the SCREENING estimate under the")
    print("                null. NOT compared to the SESOI, and NOT a validity")
    print("                condition for the fresh confirmation rung.")
    print("  P(adv|inK)  = P(the candidate better by the SESOI is advanced |")
    print("                it is among the K). The unconditional probability")
    print("                also needs P(good candidate in Top-K) — UNKNOWN.")
    print()
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
