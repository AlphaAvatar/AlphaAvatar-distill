"""The Phase-D1 target-aware search space, derived, and what the chain costs.

    PYTHONPATH=src:scripts python -m experiments.phase_d1.search_space

Zero cost. It loads no model, reads no checkpoint and needs no GPU.

**What D1 varies, and it is not the operator set.** C1, C2 and C3 varied which
ATTENTION *algorithm* ran. D1 varies the SCORING SEMANTICS the whole search uses:
every calibration-derived operator objective and the global state metric read
positions through one hash-bound
:class:`~aadistill.initialization.scoring.positions.ScoringPositionPolicy`
instead of over every token. The four implementations are **frozen at the current
best per structural kind**, so the only experimental variable is the policy.

That makes the space much smaller than C2's full joint re-search — one
implementation per kind rather than a library — and the cost arithmetic is the
same, so it is imported rather than copied: `experiments.search_cost_model` owns
branching and pricing, and `experiments.phase_c2.search_space` owns the measured
per-expansion cost table and the non-search session shape.

**The cost table is a PROVISIONAL planning ceiling, not a measurement of D1.**
Every cell was measured with an UNBATCHED state evaluation and a
one-item-per-forward statistics pass. D1 runs both batched, and batching does
**not** reliably reduce the time per expansion: A3 measured
`attention.activation_importance_v1`'s scorer 8.2–10.0% SLOWER at batch 3 than
at batch 1, on three separate pods with the sign never flipping, at +29.6% peak
VRAM. Length-sorted packing won `1.1884x` on causal-KL, which has 60,099
forwards each carrying a fixed ablation setup cost to amortize; an operator
whose per-forward fixed cost is near zero gets padded positions and wider
tensors instead. So batching moves different operators in different directions,
and the net effect on a D1 expansion — which runs all four — is **unmeasured**.

The figures here are therefore a planning ceiling to be REFRESHED by the owed
short GPU qualification (real CUDA/bf16 execution, the real state-eval memory
peak, the target-aware batched path's correctness, and the actual timing of a
representative expansion). They are not a finalized authorization price, and no
cell is adjusted on a predicted speed-up: a ceiling derived from a prediction is
a prediction.

**What this module does NOT do.** It does not authorize anything, it does not
choose the behavioural design, and it does not pretend the chain is fundable.
The behavioural rungs are priced from C2's and C3's measured per-probe minutes;
the design that decides how many probes there are lives in the D1 plan, and the
evidence capacity that decides whether those probes can be measured at all lives
in `d1_evidence_capacity`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
#: `phase_a_frozen` owns the target geometry and the teacher identity, and it
#: lives beside the Phase-A driver rather than under `experiments/`. Added the
#: same way `experiments.phase_c2.full_search_space` reaches it.
if str(REPO_ROOT / "scripts/autoinit") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from aadistill.initialization.planning.ranking import SCHEDULE_V1  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from experiments.search_cost_model import (  # noqa: E402
    SearchSpace, bound as _bound, decomposition, trajectory as _trajectory,
    walk_leaves,
)
#: The measured cost table, the session shape and the teacher geometry. One
#: owner: a second copy is how two phases come to price the same expansion
#: differently.
from experiments.phase_c2.full_search_space import (  # noqa: E402
    cost_model,
)
from experiments.phase_c2.search_space import (  # noqa: E402
    PRICE_PER_HOUR_LAST_QUOTED, SESSION_PHASE_MINUTES, TEACHER_GEOMETRY,
    register_c2_operators,
)

FAMILY = "qwen3"


class D1SpaceError(RuntimeError):
    """The D1 space cannot be derived or priced as configured."""


#: THE FROZEN CURRENT BEST PER STRUCTURAL KIND, by maintainer instruction. One
#: implementation per kind, so the operator library is not an experimental
#: variable in D1 and the scoring semantics are the only one.
#:
#: `ffn.activation_importance_v0` stays the FFN operator: the maintainer
#: explicitly deferred `ffn.residual_write_energy_v1` and any new formal FFN
#: experiment, so the incumbent FFN is carried forward rather than reopened.
FROZEN_IMPLEMENTATIONS: dict[str, str] = {
    "DEPTH": "depth.causal_kl_greedy_v1",
    "FFN": "ffn.activation_importance_v0",
    "RESIDUAL_WIDTH": "width.global_pca_v0",
    "ATTENTION": "attention.activation_importance_v1",
}

#: Why each alternative is out. Recorded per id, because "the operator zoo is
#: closed" is a decision and not a fact about the registry.
EXCLUSIONS: dict[str, str] = {
    "attention.weight_proxy_v0":
        "C1 is the isolation experiment between it and "
        "attention.activation_importance_v1 and returned GO. Promotion is what "
        "an isolation verdict is for.",
    "attention.causal_kl_v1":
        "formal C3 measured it and returned NO_GO on the primary "
        "operator-isolation contrast; it does not promote.",
    "depth.positional_v0":
        "a fixed positional heuristic that consumes no calibration, so no "
        "position policy can change its output. Admitting it would put a "
        "branch in D1 that the experimental variable cannot reach.",
    "composite.stage1_sandwich_v0":
        "reaches the target in ONE step, so a leaf containing it has no "
        "per-kind operator decisions for a scoring policy to change. D1 varies "
        "how the four structural decisions are scored; a path that makes none "
        "of them separately is outside the question.",
}

#: Both materialized mixtures. Every frozen implementation consumes calibration,
#: so each of the four steps branches over both — which is the whole reason the
#: space is larger than 4! orderings.
PROFILE_IDS: tuple[str, ...] = ("calib.domain_balanced@v1",
                                "calib.reasoning_heavy@v2")

#: Measured per-probe recovery minutes, from the two sessions that ran the frozen
#: `E1_KD_HEAVY_0860K` probe end to end. MAX across both, not mean: a ceiling
#: built on a mean is not a ceiling.
#:
#: C2 full-search pricing observed 6 probes (train mean 61.778, eval mean 25.225,
#: eval max 27.633); C3's nine-probe pricing decomposed 556.2 train + 237.6 eval
#: minutes over 9, which is 61.80 and 26.40. The two agree to within a minute,
#: which is why one bounding figure serves both.
PROBE_MINUTES: dict[str, float] = {
    "train_max": 61.883,
    "eval_max": 27.633,
}

#: Stage overheads a behavioural session pays once, from C3's own component
#: table: setup, gates, the parent replay that proves the incumbent rebuilds,
#: and the decide/bootstrap/closeout tail.
BEHAVIOURAL_SESSION_MINUTES: tuple[tuple[str, float], ...] = (
    ("setup", 45.0),
    ("machine_gates", 22.0),
    ("parent_replay_and_incumbent_rebuild", 22.0),
    ("decide_bootstrap_closeout", 15.0),
)

#: Hard-ceiling multiplier on the train+eval block, C3's own figure.
TRAIN_EVAL_OVERRUN_FACTOR = 1.3

#: RunPod bills container disk separately from the GPU, and it is not rounding
#: error on a 30-hour session. Same rate C2's pricing record used.
DISK_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 720.0


# --- the space --------------------------------------------------------------


def _target_spec() -> ArchSpec:
    from phase_a_frozen import TARGET_GEOMETRY

    return ArchSpec.of(FAMILY, TARGET_GEOMETRY)


def d1_space() -> SearchSpace:
    """One implementation per kind, order free, both mixtures free per step."""
    from aadistill.initialization.operators.base import registered_implementations

    registered = set(registered_implementations())
    allowed = tuple(sorted(FROZEN_IMPLEMENTATIONS.values()))
    missing = [i for i in allowed if i not in registered]
    if missing:
        raise D1SpaceError(
            f"{missing} are not registered, so the frozen D1 operator set could "
            "not be searched. `attention.activation_importance_v1` is "
            "registered by its own module rather than as a shipped default — "
            "call `register_c2_operators()` first.")
    return SearchSpace(
        allowed_impls=allowed, profile_ids=PROFILE_IDS,
        #: None = every calibration-consuming operator branches over every active
        #: mixture. Calibration ASSIGNMENT stays a free variable: the directive
        #: permits the search to explore ordering, valid composition and
        #: legitimate calibration-profile assignments, and freezing it would
        #: inherit C2's assignment, which was chosen under the full-sequence
        #: scoring D1 is testing.
        impl_profiles=None,
        teacher=ArchSpec.of(FAMILY, TEACHER_GEOMETRY),
        target=_target_spec(), family=FAMILY)


def coverage(*, beam_width: int | None = None,
             statistic: str = "max") -> dict[str, Any]:
    """How much of the space the beam actually visits, level by level.

    **Stated because a reviewer should not have to derive it.** The space has
    384 reachable leaves and the beam reaches about a dozen: pruning is the
    point of a beam, and the project's standing position is that the goal is
    never exhaustive enumeration but that every admissible alternative *competes
    inside one search*. What makes that true here is the warm-up level — every
    one of the root's children survives level 0, so no structural-kind /
    mixture hypothesis is eliminated before it has been measured once, and the
    pruning that follows is between paths rather than between hypotheses.

    Read from the shared `trajectory`, which is the same walk the price is
    derived from, so the coverage and the cost cannot disagree about the shape
    of the beam.
    """
    walk = _trajectory(
        d1_space(), cost_model(), prefer_costly=True, statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=SCHEDULE_V1.warmup_levels)
    levels = [{k: v for k, v in level.items() if k != "minutes"}
              for level in walk.get("levels", [])]
    leaves_visited = levels[-1]["generated"] if levels else 0
    total = decomposition(d1_space())["total_leaves"]
    return {
        "levels": levels,
        "states_produced": walk["expansions"],
        "complete_leaves_visited": leaves_visited,
        "reachable_leaves": total,
        "fraction_of_leaves_visited": round(leaves_visited / total, 5),
        "root_children_all_survive_level_0": True,
        "_why_that_is_the_part_that_matters": (
            "the warm-up level keeps every root child, so each of the 4 kinds x "
            "2 mixtures is measured once before anything is pruned. A narrower "
            "beam would explore fewer PATHS; it would not eliminate a "
            "hypothesis unmeasured."),
        "_this_is_not_a_defence_of_the_width": (
            "beam width 6 is the standing declared schedule and is held fixed "
            "across D1/D2/D3 so the scoring semantics are the only variable. "
            "Whether 6 is the right breadth for a 384-leaf space is a separate "
            "question, and changing it here would be a new breadth decision "
            "taken for no measured reason."),
    }


def size_report() -> dict[str, Any]:
    """The derived space, and what each exclusion costs in leaves."""
    from dataclasses import replace

    space = d1_space()
    full = decomposition(space)
    with_all = decomposition(replace(space, allowed_impls=tuple(sorted(
        set(space.allowed_impls) | set(EXCLUSIONS)))))
    return {
        "d1_frozen_set": full,
        "if_no_operator_were_excluded": {
            "total_leaves": with_all["total_leaves"],
            "decomposed_leaves": with_all["decomposed_leaves"],
        },
        "frozen_implementations": dict(sorted(FROZEN_IMPLEMENTATIONS.items())),
        "exclusions": dict(sorted(EXCLUSIONS.items())),
        "profiles": list(PROFILE_IDS),
        "beam": {"width": SCHEDULE_V1.width,
                 "warmup_levels": SCHEDULE_V1.warmup_levels,
                 "schedule_id": SCHEDULE_V1.schedule_id,
                 "_why_unchanged": (
                     "the standing declared schedule. D1/D2/D3 must share a "
                     "beam width for the scoring semantics to be the only "
                     "variable, and adopting a different one here would be a "
                     "new breadth decision taken for no measured reason.")},
    }


def search_bound(*, beam_width: int | None = None, statistic: str = "max"):
    return _bound(
        d1_space(), cost_model(),
        statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=SCHEDULE_V1.warmup_levels)


def search_cost(*, price_per_hour: float = PRICE_PER_HOUR_LAST_QUOTED,
                beam_width: int | None = None, statistic: str = "max",
                container_disk_gb: int = 400) -> dict[str, Any]:
    """Minutes and dollars for the SEARCH session alone.

    The session overheads are C2's measured ones, imported rather than
    re-estimated: the same launcher, the same image, the same staging.
    """
    window = search_bound(beam_width=beam_width)
    #: THE EXPECTED PATH is the costly-operator-early trajectory, which is the
    #: same nomination `price()` uses — not the structural minimum. The cheapest
    #: beam is not the smallest one: deferring the expensive operator is cheap
    #: now and generates more children later, so the window's `min_minutes`
    #: describes a different beam than its `min_expansions`.
    expected = _trajectory(
        d1_space(), cost_model(), prefer_costly=True, statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=SCHEDULE_V1.warmup_levels)
    overhead = sum(minutes for _, minutes in SESSION_PHASE_MINUTES)
    expected_minutes = round(expected["minutes"] + overhead, 2)
    hard_minutes = round(window.max_minutes + overhead, 2)
    gpu = round(hard_minutes / 60.0 * price_per_hour, 4)
    disk = round(hard_minutes / 60.0 * container_disk_gb
                 * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH, 4)
    return {
        "beam_width": SCHEDULE_V1.width if beam_width is None else beam_width,
        "warmup_levels": SCHEDULE_V1.warmup_levels,
        "expansions_min": window.min_expansions,
        "expansions_max": window.max_expansions,
        "structural_minutes_min": round(window.min_minutes, 2),
        "structural_minutes_max": round(window.max_minutes, 2),
        "expected_trajectory_minutes": round(expected["minutes"], 2),
        "expected_trajectory_expansions": expected.get("expansions"),
        "session_overhead_minutes": overhead,
        "expected_minutes": expected_minutes,
        "hard_ceiling_minutes": hard_minutes,
        "price_per_hour": price_per_hour,
        "gpu_usd": gpu,
        "container_disk_gb": container_disk_gb,
        "container_disk_usd": disk,
        "hard_ceiling_usd": round(gpu + disk, 4),
        "unmeasured_inputs": list(window.unmeasured),
        "_cost_basis": cost_model().source,
    }


def behavioural_cost(*, n_probes: int,
                     price_per_hour: float = PRICE_PER_HOUR_LAST_QUOTED,
                     container_disk_gb: int = 120) -> dict[str, Any]:
    """Minutes and dollars for a behavioural session of ``n_probes`` probes.

    `n_probes` is an ARGUMENT, not a constant: how many probes there are is the
    design's decision and it is made in the plan against the evidence, not here.
    """
    if n_probes < 1:
        raise D1SpaceError(f"a behavioural session needs probes, got {n_probes}")
    per_probe = PROBE_MINUTES["train_max"] + PROBE_MINUTES["eval_max"]
    probe_minutes = round(per_probe * n_probes, 2)
    overhead = sum(minutes for _, minutes in BEHAVIOURAL_SESSION_MINUTES)
    expected_minutes = round(probe_minutes + overhead, 2)
    hard_minutes = round(probe_minutes * TRAIN_EVAL_OVERRUN_FACTOR + overhead, 2)
    gpu = round(hard_minutes / 60.0 * price_per_hour, 4)
    disk = round(hard_minutes / 60.0 * container_disk_gb
                 * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH, 4)
    return {
        "n_probes": n_probes,
        "minutes_per_probe_bounding": round(per_probe, 3),
        "probe_minutes": probe_minutes,
        "session_overhead_minutes": overhead,
        "expected_minutes": expected_minutes,
        "hard_ceiling_minutes": hard_minutes,
        "overrun_factor": TRAIN_EVAL_OVERRUN_FACTOR,
        "price_per_hour": price_per_hour,
        "gpu_usd": gpu,
        "container_disk_gb": container_disk_gb,
        "container_disk_usd": disk,
        "hard_ceiling_usd": round(gpu + disk, 4),
    }


def chain_cost(*, screening_probes: int, confirmation_probes: int,
               price_per_hour: float = PRICE_PER_HOUR_LAST_QUOTED,
               ) -> dict[str, Any]:
    """The complete D1 chain: search, then screening, then confirmation.

    THREE SESSIONS, priced separately and summed, because that is how they are
    authorized and because a search that commits its candidate set stops — the
    behavioural rungs cannot be bound until the set it produced exists.
    """
    search = search_cost(price_per_hour=price_per_hour)
    screening = behavioural_cost(n_probes=screening_probes,
                                 price_per_hour=price_per_hour)
    confirmation = behavioural_cost(n_probes=confirmation_probes,
                                    price_per_hour=price_per_hour)
    sessions = {"search": search, "screening": screening,
                "confirmation": confirmation}
    return {
        "sessions": sessions,
        "price_per_hour": price_per_hour,
        "expected_usd": round(sum(
            s["expected_minutes"] / 60.0 * price_per_hour
            + s["container_disk_usd"] * s["expected_minutes"]
            / s["hard_ceiling_minutes"]
            for s in sessions.values()), 4),
        "hard_ceiling_usd": round(
            sum(s["hard_ceiling_usd"] for s in sessions.values()), 4),
        "max_session_hard_ceiling_usd": round(
            max(s["hard_ceiling_usd"] for s in sessions.values()), 4),
        "total_probes": screening_probes + confirmation_probes,
    }


def designs(*, price_per_hour: float = PRICE_PER_HOUR_LAST_QUOTED
            ) -> list[dict[str, Any]]:
    """The behavioural designs worth considering, each priced and each with its
    selection-noise properties.

    The point of the table is that the probe count, the screening estimate's
    winner's curse and the discrimination all move together, so the trade is
    visible rather than inherited.

    **No column here is an admissibility rule.** ``screening_estimate_inflation``
    describes the SCREENING estimate under the null; it is not compared to the
    SESOI, and a design is not admitted or rejected by it — a fresh disjoint
    confirmation rung is unbiased under that null whatever the screening
    inflation was. ``advance_probability`` is conditional on a good candidate
    being inside the screened field. See
    ``selection_noise.CLAIM_BOUNDARY``.
    """
    from experiments.phase_d1.selection_noise import (
        advance_probability,
        advance_probability_sensitivity,
        screening_estimate_inflation,
    )

    out = []
    #: The grid, not a shortlist: the designs that bracket the trade, including
    #: the one C2 actually ran (K=5, one screening seed) so its properties are in
    #: the table beside the alternatives rather than described in prose.
    for top_k, screen_seeds in ((1, 0), (2, 1), (2, 2), (3, 2), (3, 3),
                                (4, 2), (5, 1), (5, 3)):
        #: `top_k + 1` arms per screening seed: the candidates plus the incumbent
        #: anchor, which is never an advancing candidate and is re-measured on
        #: the screening battery so the ranking is paired.
        screening_probes = (top_k + 1) * screen_seeds if screen_seeds else 0
        confirmation_probes = 2 * 3            # one candidate + B, three seeds
        cost = chain_cost(screening_probes=max(screening_probes, 0),
                          confirmation_probes=confirmation_probes,
                          price_per_hour=price_per_hour) \
            if screening_probes else None
        if cost is None:
            search = search_cost(price_per_hour=price_per_hour)
            confirmation = behavioural_cost(n_probes=confirmation_probes,
                                            price_per_hour=price_per_hour)
            cost = {"sessions": {"search": search,
                                 "confirmation": confirmation},
                    "hard_ceiling_usd": round(search["hard_ceiling_usd"]
                                              + confirmation["hard_ceiling_usd"], 4),
                    "total_probes": confirmation_probes}
        out.append({
            "top_k": top_k,
            "screening_seeds": screen_seeds,
            "screening_probes": screening_probes,
            "confirmation_probes": confirmation_probes,
            "total_probes": cost["total_probes"],
            "screening_estimate_inflation": screening_estimate_inflation(
                top_k, screen_seeds),
            "advance_probability": advance_probability(top_k, screen_seeds),
            "advance_probability_sensitivity": advance_probability_sensitivity(
                top_k, screen_seeds),
            "hard_ceiling_usd": cost["hard_ceiling_usd"],
            "fresh_batteries_required": (2 if screen_seeds else 1),
            "_c2_design": top_k == 5 and screen_seeds == 1,
        })
    return out


def report() -> dict[str, Any]:
    register_c2_operators()
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    activation_importance.register()
    return {
        "schema": "aadistill.autoinit.phase_d1_space/v1",
        "size": size_report(),
        "search": search_cost(),
        "designs": designs(),
        "_authorizes": "nothing",
    }


def main() -> int:
    doc = report()
    size, search = doc["size"], doc["search"]
    full = size["d1_frozen_set"]

    print("Phase-D1 target-aware search — derived space\n")
    print(f"  differing fields        : {', '.join(full['required_fields'])}")
    for kind, impl_id in size["frozen_implementations"].items():
        print(f"  FROZEN {kind:16} {impl_id}")
    for impl_id, why in size["exclusions"].items():
        print(f"  EXCLUDED {impl_id}\n     {why[:92]}…")
    print(f"\n  profiles                : {', '.join(size['profiles'])}")
    print(f"  beam                    : width {size['beam']['width']}, "
          f"warmup {size['beam']['warmup_levels']}")
    print(f"  reachable leaves        : {full['total_leaves']}")
    print(f"  if nothing were excluded: "
          f"{size['if_no_operator_were_excluded']['total_leaves']}")
    print(f"\n  expansions              : {search['expansions_min']}–"
          f"{search['expansions_max']}")
    print(f"  search hard ceiling     : {search['hard_ceiling_minutes']:.1f} min "
          f"= ${search['hard_ceiling_usd']:.4f} at "
          f"${search['price_per_hour']}/h")
    if search["unmeasured_inputs"]:
        print(f"  UNMEASURED inputs       : {search['unmeasured_inputs']}")

    print("\n  behavioural designs, priced:\n")
    print(f"  {'K':>3} {'seed':>5} {'probes':>7} {'inflate':>8} {'P(adv)':>7} "
          f"{'lo/hi':>15} {'batt':>5} {'chain $':>9}")
    for row in doc["designs"]:
        flag = "  <- C2's design" if row["_c2_design"] else ""
        sens = row["advance_probability_sensitivity"]
        print(f"  {row['top_k']:>3} {row['screening_seeds']:>5} "
              f"{row['total_probes']:>7} "
              f"{row['screening_estimate_inflation']:>8.5f} "
              f"{row['advance_probability']:>7.4f} "
              f"{sens['p_at_sd_low']:>6.3f}/{sens['p_at_sd_high']:<8.3f}"
              f"{row['fresh_batteries_required']:>5} "
              f"{row['hard_ceiling_usd']:>9.4f}{flag}")
    print("\n  inflate = the winner's curse on the SCREENING estimate under the")
    print("            null. NOT compared to the SESOI and NOT an admissibility")
    print("            rule: a fresh disjoint confirmation rung is unbiased")
    print("            under that null whatever the screening inflation was.")
    print("  P(adv)  = P(the candidate better by the SESOI is advanced | it is")
    print("            among the K). lo/hi span the per-seed sd's own 95%")
    print("            sampling interval, which comes from THREE observations.")
    print("  batt    = fresh disjoint batteries the design consumes.")
    print("\n  AUTHORIZES NOTHING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
