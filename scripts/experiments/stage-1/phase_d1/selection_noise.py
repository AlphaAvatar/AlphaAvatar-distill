"""What a screening rung adds to the candidate it advances, when none is better.

C2's behavioural stage screened five candidates against the incumbent on ONE
seed and advanced the best. That is a maximum over five noisy estimates, and the
maximum of noisy estimates is biased upward even when every candidate is exactly
as good as the anchor. If that bias is comparable to the SESOI, the confirmation
rung is testing an inflated nomination and the design cannot tell selection noise
from the effect it exists to measure.

So the bias is derived here rather than assumed small, from the project's own
measured per-seed spread, and it is what makes the D1 screening design a
*derivation* instead of a copy of C2's numbers.

**The measured input.** A3 ran three paired seeds on the frozen confirmation
battery and reported per-seed `correct_overall` deltas of `+0.012941`,
`-0.011765` and `+0.007059` against a pooled `+0.002745` — a sample standard
deviation of `0.012910`. C3 independently measured a seed-level spread of
`±0.022353` on arms whose pooled differences were within `0.001961` of zero,
which is the same magnitude read as a range rather than a deviation. Both are
measurements of the same thing: how much a paired single-seed delta moves when
nothing has changed but the seed.

**The arithmetic, and its assumption.** For `k` independent zero-mean normal
draws with standard deviation `s`, the expected maximum is `s * E_k` where `E_k`
is the expected value of the maximum of `k` standard normals. Averaging `m`
screening seeds divides `s` by `sqrt(m)`. The independence assumption is
optimistic — candidates from one search share prompts and a parent, so their
errors are positively correlated, which REDUCES the spread of the maximum. So
this figure is an upper bound on the bias, which is the direction a design
decision wants to be wrong in.

Nothing here is specific to D1: it is the arithmetic of screening `k` arms on `m`
seeds at a measured noise level. It lives under `experiments/` rather than in
`src/aadistill` because the measured constant is an instance fact about this
project's battery and recovery recipe, not a property of the framework.
"""
from __future__ import annotations

import math

#: Sample standard deviation of the paired per-seed `correct_overall` delta, from
#: A3's three frozen C3 seeds on `c1_confirmation_v1`. Derived from the three
#: published deltas rather than quoted, by `_a3_seed_sd` below, so a reader can
#: check it against the closeout.
A3_SEED_DELTAS: tuple[float, ...] = (0.012941, -0.011765, 0.007059)

#: C3's independently measured seed-level spread, as a RANGE. Kept as a
#: cross-check, not as an input: a range and a standard deviation are different
#: statistics and averaging them would be meaningless.
C3_SEED_LEVEL_SPREAD_RANGE = 0.022353

#: C0's decision boundary. Not a power target — the figure a design's selection
#: bias has to sit under for the confirmation rung to be interpretable.
SESOI = 0.010

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


def expected_max_of_standard_normals(k: int) -> float:
    """`E[max(Z_1..Z_k)]` for standard normal `Z`.

    By the survival-function identity

        E[max] = int_0^inf (1 - F(x)^k) dx  -  int_0^inf F(-x)^k dx

    which is numerically well behaved for the small `k` a screening rung uses and
    is checked against the two closed forms — `0` at `k=1` and `1/sqrt(pi)` at
    `k=2` — by :func:`self_check`. The check exists because the first version of
    this function wrote the negative term as `1 - (1 - F(-x))^k`, which is
    algebraically the positive term again: the two cancelled and every bias came
    out exactly zero, which read as "no design has a selection problem".
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
    A bias of exactly zero everywhere is indistinguishable from "screening is
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
                "selection-bias figure derived from it would be too")
        out[k] = got
    return out


def expected_max_bias(top_k: int, screening_seeds: int, *,
                      seed_sd: float = SEED_SD) -> float:
    """The advancing candidate's expected apparent delta under the null.

    `screening_seeds == 0` means no screening rung: the search's own cheap metric
    nominates and there is no behavioural maximum to be biased by, so the bias is
    zero. That is not an argument for skipping screening — it moves the risk from
    an inflated nomination to an unfiltered one — it is the honest zero.
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


def advance_probability(top_k: int, screening_seeds: int, *,
                        effect: float = SESOI,
                        seed_sd: float = SEED_SD) -> float:
    """P(screening advances the one genuinely better candidate).

    **Bias and discrimination are different properties, and a design has to
    satisfy both.** The bias above bounds how much the advancing candidate's
    apparent delta is inflated when nothing is better; this bounds how often the
    rung finds the thing it is looking for when something IS. They pull opposite
    ways in `top_k`: a smaller field is less inflated and also less likely to
    contain — or to correctly rank — the better candidate.

    The model is deliberately the simplest one that captures the trade: one
    candidate is better than the anchor by `effect`, the other `top_k - 1` are
    exactly as good as it, and each candidate's screening estimate is its true
    delta plus zero-mean noise at `seed_sd / sqrt(screening_seeds)`. A `top_k` of
    1 advances its only candidate by construction.
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


def report() -> dict:
    """Every design's bias and discrimination beside the SESOI, for the record."""
    checked = self_check()
    rows = []
    for top_k in (1, 2, 3, 4, 5, 6):
        for seeds in (1, 2, 3):
            bias = expected_max_bias(top_k, seeds)
            rows.append({"top_k": top_k, "screening_seeds": seeds,
                         "selection_bias": bias,
                         "bias_over_sesoi": round(bias / SESOI, 4),
                         "under_sesoi": bias < SESOI,
                         "advance_probability": advance_probability(
                             top_k, seeds),
                         "screening_probes": (top_k + 1) * seeds})
    return {
        "schema": "aadistill.autoinit.screening_selection_noise/v1",
        "seed_sd": round(SEED_SD, 6),
        "seed_sd_source": ("A3's three paired per-seed correct_overall deltas on "
                           "c1_confirmation_v1, logs/stages/stage-1/phase_a3/"
                           "analyses/a3_closeout.md"),
        "seed_sd_cross_check_range": C3_SEED_LEVEL_SPREAD_RANGE,
        "sesoi": SESOI,
        "independence_assumption": (
            "candidates from one search share prompts and a parent, so their "
            "errors are positively correlated and the true spread of the "
            "maximum is SMALLER. These figures are upper bounds."),
        "quadrature_self_check": {str(k): v for k, v in checked.items()},
        "rows": rows,
    }


def main() -> int:
    import json

    doc = report()
    print(f"measured per-seed sd of the paired delta: {doc['seed_sd']:.6f}")
    print(f"SESOI: {doc['sesoi']}\n")
    print(f"  {'K':>3} {'seeds':>6} {'probes':>7} {'bias':>9} {'/SESOI':>8} "
          f"{'P(advance)':>11}  verdict")
    for row in doc["rows"]:
        print(f"  {row['top_k']:>3} {row['screening_seeds']:>6} "
              f"{row['screening_probes']:>7} "
              f"{row['selection_bias']:>9.6f} {row['bias_over_sesoi']:>8.3f} "
              f"{row['advance_probability']:>11.4f}  "
              f"{'ok' if row['under_sesoi'] else 'EXCEEDS THE SESOI'}")
    print("\n  bias        = apparent delta of the advancing candidate under the"
          " null")
    print("  P(advance)  = P(the one candidate better by the SESOI is the one"
          " advanced)")
    print("  The two pull opposite ways in K. A design needs bias < SESOI AND a")
    print("  P(advance) worth the probes it spends.")
    print()
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
