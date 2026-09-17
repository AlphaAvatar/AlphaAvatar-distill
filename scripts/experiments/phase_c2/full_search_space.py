"""The Phase-C2 FULL JOINT re-search space, derived, and what it costs.

    PYTHONPATH=src:scripts python -m experiments.phase_c2.full_search_space

Zero cost. It loads no model, reads no checkpoint and needs no GPU.

**What this is not.** It is not Search-1. Search-1 is DONE and FROZEN: it held
DEPTH, FFN and RESIDUAL_WIDTH at the Phase-B incumbent's mixtures and varied
only ATTENTION's, and its evidence (`runs/attempt4`, `runs/attempt8`) is
preserved exactly as it was measured. `experiments.phase_c2.search_space` still
owns that space and is unchanged. This module is the **successor** experiment,
and the two share the branching-and-cost arithmetic in
`experiments.search_cost_model` rather than a copy of it.

**What changed scientifically.** C1 isolated ATTENTION on a fixed path and its
frozen Stage-I rule returned `GO`, so `attention.activation_importance_v1` is
the **accepted** ATTENTION operator. That operator declares
`CalibrationNeed.ACTIVATION_STATS`, where the one it replaced
(`attention.weight_proxy_v0`) declares `NONE` — so ATTENTION now **consumes
calibration**, and an operator that consumes calibration branches over every
active mixture instead of being offered once. That single change is why the
space is larger than the one Phase B searched, and it is derived below rather
than asserted.

**Why nothing is held fixed.** Search-1 answered a restricted question and
answered it usefully: after promoting the new ATTENTION operator, changing order
and composition alone moved the cheap metrics enough to put four candidates in a
better Pareto front than the frozen C1 treatment baseline, two of them
dominating it on all three ranked objectives. That is evidence that the search
procedure finds real structure here — and it is also the reason the incumbent
DEPTH / FFN / WIDTH calibration assignments can no longer be assumed optimal:
they were chosen by a search in which ATTENTION could not consume calibration at
all. So this space exposes implementations, applicable profiles and order
jointly, and lets calibration choices affect pruning.

**The cost table is DERIVED, not transcribed.** It is recomputed at import from
both committed telemetry files — Phase-B attempt 5 and C2 attempt 4 — taking the
per-cell maximum across both. Two consequences matter:

* `attention.activation_importance_v1` is **no longer an unmeasured input.**
  Search-1 priced it at `1.5x width.global_pca_v0` because it had never run
  inside a search; C2 attempt 4 then ran it 14 times, and the measured root cost
  is *below* the proxy. The margin was conservative in the safe direction and is
  now retired.
* `depth.causal_kl_greedy_v1` deeper cost went **up**: attempt 4 observed
  36.07 min against Phase B's 31.10. Taking the max across both sources is the
  only honest ceiling, and it is the single figure the price turns on.
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aadistill.initialization.planning.ranking import SCHEDULE_V1  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from experiments.search_cost_model import (  # noqa: E402
    CostModel, SearchSpace, bound as _bound, decomposition,
    price as _price, trajectory as _trajectory, walk_leaves,
)
#: The non-search session shape — setup, transfer, teacher fetch, gates, commit,
#: sync — is identical to Search-1's: same launcher, same image, same staging.
#: One owner, and it is the module that derived those figures from the preflight
#: launcher rather than from a lucky warm-image observation.
from experiments.phase_c2.search_space import (  # noqa: E402
    PRICE_PER_HOUR_LAST_QUOTED, SESSION_PHASE_MINUTES, TEACHER_GEOMETRY,
    register_c2_operators,
)


class FullSearchSpaceError(RuntimeError):
    """The full joint space cannot be derived or priced as configured."""


FAMILY = "qwen3"

#: The measured-optimization record the cost table is refreshed by, when
#: it exists. Absent, the pooled pre-optimization figures stand -- which
#: is the safe direction: they over-state rather than under-state.
MEASURED_OPTIMIZATION = ("logs/stages/stage-1/phase_c2/plans/"
                        "phase_c2_measured_optimization.json")

# --- the two committed telemetry sources ------------------------------------

#: Every search this project has run on this image and card. The cost table is
#: the per-cell MAXIMUM across both: a ceiling that ignores the more expensive of
#: two observations is not a ceiling.
TELEMETRY_SOURCES: tuple[tuple[str, str, str | None], ...] = (
    ("phase_b_attempt5",
     "logs/stages/stage-1/phase_b/runs/attempt5/search_telemetry.jsonl",
     "logs/stages/stage-1/phase_b/runs/attempt5/search_result.json"),
    ("phase_c2_attempt4",
     "logs/stages/stage-1/phase_c2/runs/attempt4/evidence/telemetry.jsonl",
     None),
)

#: The phases that make up one expansion, end to end. Same definition Search-1's
#: table uses, so the two are comparable and can be maximised cell by cell.
EXPANSION_PHASES = ("materialize_seconds", "identify_seconds",
                    "canonical_reload_seconds", "validation_seconds",
                    "state_evaluation_seconds")


def _phase_seconds(row: dict, key: str) -> float:
    value = row.get(key) or {}
    return (float(value.get("seconds", 0.0)) if isinstance(value, dict)
            else float(value or 0))


def _observations(repo_root: Path) -> dict[tuple[str, str], list[float]]:
    """`(impl_id, root|deeper) -> minutes`, pooled over every committed run.

    Whether a parent is the ROOT is resolved from the run's own level record
    when it has one, and otherwise from the telemetry's parent/child structure:
    a parent id that never appears as a state id is a root. Both answers are
    derived from the record rather than assumed from ordering.
    """
    pooled: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for name, telemetry, result in TELEMETRY_SOURCES:
        path = repo_root / telemetry
        if not path.is_file():
            raise FullSearchSpaceError(
                f"{name}: {telemetry} is missing, so the cost table would be "
                "built over a smaller set of observations than the repository "
                "actually holds. Refusing to price from a partial history.")
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        if result is not None:
            levels = json.loads((repo_root / result).read_text())["levels"]
            level_of = {sid: L["level"] for L in levels for sid in L["generated"]}
            def is_root(row, _level_of=level_of):
                return _level_of.get(row["state_id"]) == 0
        else:
            state_ids = {r["state_id"] for r in rows}
            roots = {r["parent_id"] for r in rows
                     if r["parent_id"] not in state_ids}
            def is_root(row, _roots=roots):
                return row["parent_id"] in _roots
        for row in rows:
            minutes = (row["operator_seconds"] + row["parent_load_seconds"]
                       + sum(_phase_seconds(row, k)
                             for k in EXPANSION_PHASES)) / 60
            pooled[(row["impl_id"], "root" if is_root(row) else "deeper")
                   ].append(minutes)
    return dict(pooled)


def derive_cost_table(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The per-expansion minutes table, recomputed from committed telemetry.

    Returns the table plus its observation counts, so a reader can see how much
    evidence each cell rests on. A cell backed by one observation is still the
    best available answer, but it should not look like one backed by thirty-five.
    """
    pooled = _observations(Path(repo_root))
    table: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for (impl_id, where), values in pooled.items():
        table.setdefault(impl_id, {})[f"{where}_max"] = round(max(values), 4)
        table.setdefault(impl_id, {})[f"{where}_mean"] = round(
            statistics.mean(values), 4)
        counts.setdefault(impl_id, {})[where] = len(values)
    return {"minutes": table, "observations": counts,
            "sources": [name for name, _t, _r in TELEMETRY_SOURCES],
            "_rule": ("per-cell maximum and mean over every expansion recorded "
                      "by any committed search on this image and card")}


#: A `COMPOSITE_STAGE1` leaf is one step, so it has no `deeper` observation and
#: never will: it reaches the target from the root. Naming the fallback here,
#: rather than letting the cost model raise, keeps a single-step operator
#: admissible without inventing a number — its deeper cell is its root cell.
SINGLE_STEP_DEEPER_FALLBACK = ("composite.stage1_sandwich_v0",)


def cost_model(repo_root: str | Path = REPO_ROOT) -> CostModel:
    derived = derive_cost_table(repo_root)
    table = {impl: dict(row) for impl, row in derived["minutes"].items()}
    for impl in SINGLE_STEP_DEEPER_FALLBACK:
        row = table.get(impl)
        if row is None:
            continue
        for statistic in ("max", "mean"):
            row.setdefault(f"deeper_{statistic}", row[f"root_{statistic}"])
    #: THE MEASURED REFRESH, applied if it exists.
    #:
    #: The pooled table above is per-expansion minutes from committed searches
    #: on the PRE-optimization executable. The 2026-09-18 performance round
    #: made the state-eval reduction device-resident (76.0x measured on a real
    #: L40S) and gave DEPTH a forward-KL-only path (1.10x), so those cells now
    #: over-state what an expansion costs.
    #:
    #: Applied as a named ADJUSTMENT rather than folded in: the record states
    #: every input, the arithmetic is one formula, and the pre-refresh figures
    #: stay visible in it. A ratio applied to a whole cell would have been
    #: wrong -- the saving is a component of one phase, capped at that phase.
    refresh_path = Path(repo_root) / MEASURED_OPTIMIZATION
    refreshed: dict[str, float] = {}
    if refresh_path.is_file():
        record = json.loads(refresh_path.read_text())
        for impl, cell in record["cells"].items():
            if impl not in table:
                continue
            was = cell["observed_total_minutes"]
            now = cell["refreshed_total_minutes"]
            if was <= 0:
                continue
            factor = now / was
            #: Scale BOTH statistics by the same measured factor. The max is
            #: what the ceiling rests on and the mean is what the expected path
            #: uses, and the optimization applies to both.
            for statistic in ("max", "mean"):
                for where in ("root", "deeper"):
                    key = f"{where}_{statistic}"
                    if key in table[impl]:
                        table[impl][key] = round(table[impl][key] * factor, 4)
            refreshed[impl] = round(factor, 4)

    source = ("derived at import from " + ", ".join(derived["sources"])
              + "; per-cell max across both. No proxied rows: every "
                "implementation in this space has run inside a real search.")
    if refreshed:
        source += (" REFRESHED by " + MEASURED_OPTIMIZATION + ": measured "
                   "component speedups from the 2026-09-18 performance round, "
                   "applied per cell as " + json.dumps(refreshed) + ". The "
                   "reference-cache recompute waste is deliberately NOT "
                   "claimed -- it was instrumented, not fixed.")
    return CostModel(minutes=table, proxies={}, source=source)


# --- the space ---------------------------------------------------------------

#: Both materialized mixtures, for every operator that consumes calibration.
#: `calib.reasoning_heavy@v1` and `calib.stage0_current@v1` are declared but
#: unbuilt and `resolve()` refuses them, so they cannot be search branches.
PROFILE_IDS = ("calib.domain_balanced@v1", "calib.reasoning_heavy@v2")

#: The promoted ATTENTION operator. C1's frozen Stage-I rule returned `GO` on
#: `weight_proxy_v0 -> activation_importance_v1`, which is what "accepted
#: operator library" means here.
PROMOTED_ATTENTION = "attention.activation_importance_v1"

#: WHY each registry entry is excluded, if it is. An exclusion is a scientific
#: claim and it is recorded as one; "all admissible alternatives compete" is the
#: default and anything held out has to earn it.
EXCLUSIONS: dict[str, str] = {
    "attention.weight_proxy_v0": (
        "CLOSED BY C1, not held out for cost. C1 was the isolation experiment "
        "between exactly this operator and attention.activation_importance_v1 "
        "on a fixed path, three paired seeds, the frozen 0.86M recipe and the "
        "frozen battery, and its preregistered Stage-I rule returned GO. "
        "Re-entering the loser as a branch would spend roughly half again as "
        "much search budget re-deciding a question that has a completed "
        "verdict, and would make this search's result depend on it. Promotion "
        "is what an isolation verdict is FOR. If the promotion is ever "
        "withdrawn, this exclusion is withdrawn with it."),
}

#: Nothing else is excluded. In particular `depth.positional_v0` and
#: `composite.stage1_sandwich_v0` are IN: both are cheap, both are applicable,
#: neither has ever competed against a calibration-consuming ATTENTION, and
#: excluding a cheap alternative on cost grounds would be exactly the kind of
#: unjustified narrowing this space exists to remove.


def full_joint_space(repo_root: str | Path = REPO_ROOT) -> SearchSpace:
    """The joint space: every applicable implementation, every applicable
    profile, order free, minus the recorded exclusions."""
    from aadistill.initialization.operators.base import registered_implementations

    allowed = tuple(sorted(
        i for i in registered_implementations() if i not in EXCLUSIONS))
    if PROMOTED_ATTENTION not in allowed:
        raise FullSearchSpaceError(
            f"{PROMOTED_ATTENTION} is not registered, so the promoted ATTENTION "
            "operator could not be searched. Call register_c2_operators() "
            "first: it is deliberately not a shipped default.")
    return SearchSpace(
        allowed_impls=allowed, profile_ids=PROFILE_IDS,
        #: None = every operator that consumes calibration branches over every
        #: active mixture. This is the whole point of the re-search: Search-1
        #: pinned three of the four and that is what is being reopened.
        impl_profiles=None,
        teacher=ArchSpec.of(FAMILY, TEACHER_GEOMETRY),
        target=_target_spec(), family=FAMILY)


def competing_attention_space(repo_root: str | Path = REPO_ROOT) -> SearchSpace:
    """The same space with the C1-losing ATTENTION operator re-admitted.

    Not the proposed experiment. Derived so the cost of NOT taking the
    exclusion is a number a reviewer can see, rather than a claim in prose.
    """
    from dataclasses import replace

    space = full_joint_space(repo_root)
    return replace(space, allowed_impls=tuple(sorted(
        set(space.allowed_impls) | set(EXCLUSIONS))))


def _target_spec() -> ArchSpec:
    from phase_a_frozen import TARGET_GEOMETRY

    return ArchSpec.of(FAMILY, TARGET_GEOMETRY)


def phase_b_reference_space(repo_root: str | Path = REPO_ROOT) -> SearchSpace:
    """Phase B's space, for the size comparison the maintainer asked for.

    The shipped registry with no promoted ATTENTION and no restriction: exactly
    what Phase B searched. Its leaf count is the baseline the new space's
    growth is measured against.
    """
    from dataclasses import replace

    from aadistill.initialization.operators.register import BUILTIN_OPERATORS

    space = full_joint_space(repo_root)
    return replace(space, allowed_impls=tuple(
        sorted(i.impl_id for i in BUILTIN_OPERATORS)))


# --- questions this module answers ------------------------------------------


def bound(space: SearchSpace, *, beam_width: int | None = None,
          warmup_levels: int | None = None, statistic: str = "max",
          repo_root: str | Path = REPO_ROOT):
    return _bound(
        space, cost_model(repo_root), statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=(SCHEDULE_V1.warmup_levels if warmup_levels is None
                       else warmup_levels))


def trajectory(space: SearchSpace, *, prefer_costly: bool = True,
               beam_width: int | None = None, warmup_levels: int | None = None,
               statistic: str = "max", repo_root: str | Path = REPO_ROOT):
    return _trajectory(
        space, cost_model(repo_root), prefer_costly=prefer_costly,
        statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=(SCHEDULE_V1.warmup_levels if warmup_levels is None
                       else warmup_levels))


def price(space: SearchSpace, *, price_per_hour: float, authorized_usd: float,
          beam_width: int | None = None, statistic: str = "max",
          repo_root: str | Path = REPO_ROOT):
    """A `BudgetPlan` for the full-search session. Priced, which is not funded.

    RAISES `BudgetError` when the plan does not fit `authorized_usd`. That
    refusal is how a search that has outgrown the remaining budget says so,
    instead of being quietly shrunk to fit — and at the standing beam width it
    does refuse against the project's remaining headroom. See the pricing
    record and the decision it asks for.
    """
    return _price(
        space, cost_model(repo_root), price_per_hour=price_per_hour,
        authorized_usd=authorized_usd, session_phases=SESSION_PHASE_MINUTES,
        setup_phase="setup_and_asset_staging", transfer_phase="bundle_transfer",
        search_phase_name="beam_search_costly_early", statistic=statistic,
        beam_width=SCHEDULE_V1.width if beam_width is None else beam_width,
        warmup_levels=SCHEDULE_V1.warmup_levels)


def size_report(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The derived space size, and how it grew relative to Phase B.

    No number here is written down anywhere: every one is enumerated from the
    registry through the same two functions the search itself calls.
    """
    full = decomposition(full_joint_space(repo_root))
    reference = decomposition(phase_b_reference_space(repo_root))
    competing = decomposition(competing_attention_space(repo_root))
    return {
        "full_joint": full,
        "phase_b_reference": reference,
        "if_the_exclusion_were_not_taken": competing,
        "growth_vs_phase_b": {
            "total_leaves": f"{reference['total_leaves']} -> "
                            f"{full['total_leaves']}",
            "decomposed_leaves": f"{reference['decomposed_leaves']} -> "
                                 f"{full['decomposed_leaves']}",
            "_why": ("the promoted ATTENTION operator consumes calibration "
                     "where the one it replaced did not, so ATTENTION branches "
                     "over every active mixture instead of being offered once. "
                     "Same kinds, same order freedom, one more branching "
                     "factor."),
        },
        "cost_of_not_excluding": {
            "total_leaves": competing["total_leaves"],
            "reason_excluded": EXCLUSIONS["attention.weight_proxy_v0"],
        },
    }


def main() -> int:
    register_c2_operators()
    space = full_joint_space()
    report = size_report()
    cost = cost_model()
    full = report["full_joint"]

    print("Phase-C2 FULL JOINT re-search — derived space\n")
    print(f"  family                  : {space.family}")
    print(f"  differing fields        : {', '.join(full['required_fields'])}")
    print(f"  allowed implementations : {len(space.allowed_impls)}")
    for impl_id in space.allowed_impls:
        print(f"     {impl_id}")
    for impl_id, why in sorted(EXCLUSIONS.items()):
        print(f"  EXCLUDED {impl_id}\n     {why[:96]}…")
    print(f"\n  root options by kind")
    for kind, options in full["root_options_by_kind"].items():
        print(f"     {kind:18} {len(options)}")
    print(f"\n  reachable leaves        : {full['total_leaves']}")
    for n, count in full["leaves_by_operator_count"].items():
        print(f"     {n}-operator leaves   : {count}")
    print(f"  Phase-B reference       : "
          f"{report['phase_b_reference']['total_leaves']} leaves")
    print(f"  if exclusion not taken  : "
          f"{report['if_the_exclusion_were_not_taken']['total_leaves']} leaves")

    print(f"\n  cost model: {cost.source}")
    print(f"     {'impl':36} {'root_max':>9} {'deeper_max':>11}")
    for impl_id in sorted(cost.minutes):
        row = cost.minutes[impl_id]
        print(f"     {impl_id:36} {row.get('root_max', float('nan')):9.2f} "
              f"{row.get('deeper_max', float('nan')):11.2f}")
    print(f"     unmeasured inputs    : {cost.unmeasured or 'none'}")

    print(f"\n  beam-width menu at ${PRICE_PER_HOUR_LAST_QUOTED}/h "
          "(max statistic)")
    print(f"     {'width':>5} {'expansions':>12} {'costly-early h':>15} "
          f"{'structural max h':>17}")
    for width in (SCHEDULE_V1.width, 4, 3, 2):
        b = bound(space, beam_width=width)
        early = trajectory(space, beam_width=width)
        print(f"     {width:5} {b.min_expansions:5}..{b.max_expansions:<5} "
              f"{early['hours']:15.2f} {b.max_minutes / 60:17.2f}")
    print("\n  Priced is not funded, and a bound is not a plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
