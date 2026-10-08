"""What a beam search over initialization paths costs, for ANY search instance.

This is the branching-and-cost model that priced Phase-C2 Search-1, with every
experiment-specific number taken out of it. Nothing here names an operator, a
calibration profile, a model family, a GPU, a price, a session phase or a
repository path: an instance supplies those and this module does the arithmetic.

It was extracted from `experiments.phase_c2.search_space` when a **second**
consumer appeared — the C2 full joint re-search — and not before. The machinery
is identical; what moved is ownership. `phase_c2.search_space` remains the
Search-1 instance and re-exports these names, so its own constants, its
`replay_phase_b` back-test and every existing caller are unchanged.

Three questions, deliberately separate:

* :func:`decomposition` — how large is the space? A count of reachable ordered
  complete paths, derived from the registry rather than asserted.
* :func:`trajectory` — what does ONE nominated beam cost, level by level? The
  question with evidence behind it, when a previous run tells you how the
  ranking behaved.
* :func:`bound` — what is the exact extremum over EVERY beam the policy could
  return? A claim about arithmetic, not about where the Pareto front falls.

`children_max x mean node cost` is what authorized Phase-B attempt 3 at a
1.91-7.51 h projection; it ran 9.08 h and did not finish. That is why
:func:`bound` enumerates beam compositions instead of multiplying averages.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

from aadistill.initialization.operators.base import (  # noqa: E402
    applicable_implementations, get_implementation,
)
from aadistill.initialization.planning.search import expansion_profiles  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402


class CostModelError(RuntimeError):
    """A cost was asked for that this model cannot honestly supply."""


# --- the cost of one expansion ---------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Minutes per EXPANSION, end to end, keyed by implementation.

    "End to end" means operator + parent load + materialize + identify +
    canonical reload + validate + state_eval — everything an expansion pays for.
    An instance derives the table from its own committed telemetry; this type
    only serves it and refuses what it does not have.

    `root` and deeper parents are separated because the teacher-width parent is
    the expensive one, and `max`/`mean` because **a ceiling built on `mean` is
    not a ceiling**.

    A `proxy` lets an instance price an implementation that has never run, from
    one that has, times a stated factor. Those implementations are named in
    :attr:`unmeasured` and carried through :class:`Bound` so a reader sees the
    one input that is a margin rather than a measurement — instead of having it
    blended into a single number.
    """

    minutes: Mapping[str, Mapping[str, float]]
    proxies: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    source: str = ""

    @property
    def unmeasured(self) -> tuple[str, ...]:
        return tuple(sorted(self.proxies))

    def key(self, *, root: bool, statistic: str) -> str:
        return ("root_" if root else "deeper_") + statistic

    def minutes_for(self, impl_id: str, *, root: bool,
                    statistic: str = "max") -> float:
        """Minutes for one expansion of `impl_id` on a root / deeper parent."""
        cell = self.key(root=root, statistic=statistic)
        proxy = self.proxies.get(impl_id)
        if proxy is not None:
            proxy_impl, factor = proxy
            return round(self._cell(proxy_impl, cell) * factor, 4)
        return self._cell(impl_id, cell)

    def _cell(self, impl_id: str, cell: str) -> float:
        row = self.minutes.get(impl_id)
        if row is None:
            raise CostModelError(
                f"no measured cost for {impl_id!r}. A search that can expand an "
                "implementation the cost model has never seen cannot be priced; "
                "measure it, declare a proxy for it, or exclude it from the "
                f"space. Known: {sorted(self.minutes)}")
        if cell not in row:
            raise CostModelError(
                f"{impl_id!r} has no {cell!r} observation. Pricing a cell that "
                "was never observed would invent the number the ceiling rests "
                f"on. Known cells: {sorted(row)}")
        return float(row[cell])

    def most_expensive(self, impl_ids: Sequence[str], *,
                       statistic: str = "max") -> str | None:
        """The costliest implementation among `impl_ids`, by root cost.

        Used to nominate a trajectory without naming an operator: the beam that
        defers the expensive operator is the cheap one now and the expensive one
        later, so which implementation that is has to be derived rather than
        written down. In Phase B and C2 it resolves to the DEPTH operator, which
        is what those instances used to hardcode.
        """
        priced = [i for i in impl_ids if i in self.minutes or i in self.proxies]
        if not priced:
            return None
        return max(priced, key=lambda i: self.minutes_for(
            i, root=True, statistic=statistic))


# --- the space --------------------------------------------------------------


class _NamedProfile:
    """Only `qualified_id` matters to `expansion_profiles`.

    A real `CalibrationProfile` would have to be registered and, for a mixture
    with an items file, would want that file present. Counting branches needs
    neither.
    """

    def __init__(self, qualified_id: str) -> None:
        self.qualified_id = qualified_id


@lru_cache(maxsize=None)
def named_profiles(ids: tuple[str, ...]) -> tuple[_NamedProfile, ...]:
    return tuple(_NamedProfile(i) for i in ids)


@dataclass(frozen=True)
class SearchSpace:
    """One search's reachable classes, with the real registry behind them.

    A state's expansion cost and branching depend only on WHICH implementations
    it has applied, never on which calibration profile fed them: every operator
    sets its field to the target's value, so the geometry is a function of the
    applied set. Two states differing only in one operator's mixture are
    therefore one class with multiplicity two — and multiplicity is kept,
    because each of them is expanded and paid for separately.
    """

    allowed_impls: tuple[str, ...]
    profile_ids: tuple[str, ...]
    impl_profiles: Mapping[str, tuple[str, ...]] | None
    teacher: ArchSpec
    target: ArchSpec
    family: str

    @property
    def adapter(self):
        return get_adapter(self.family)

    @property
    def required_fields(self) -> tuple[str, ...]:
        """The architecture fields that still differ from the target."""
        return tuple(sorted(self.teacher.diff(self.target)))

    def spec_of(self, applied: frozenset[str]) -> ArchSpec:
        """The geometry after applying `applied`, in any order."""
        spec = self.teacher
        for impl_id in sorted(applied):
            spec = get_implementation(impl_id).plan(
                spec, self.target, self.adapter, {}).result_spec
        return spec

    def options(self, applied: frozenset[str]):
        """`(implementation, n_profiles)` for a parent in this class.

        Both halves come from the code the search runs: which implementations
        apply is `applicable_implementations`, and how many states each
        generates is `expansion_profiles` — the same function
        `BeamSearch._candidate_expansions` calls, which is the point. Counting
        it independently predicted 12 children of the Phase-B root where the
        run generated 10, because a `CalibrationNeed.NONE` operator is offered
        once however many mixtures are active.
        """
        spec = self.spec_of(applied)
        kinds = tuple(sorted({get_implementation(i).kind for i in applied}))
        found = applicable_implementations(
            self.adapter, spec, self.target,
            exclude_kinds=kinds, allow_impls=sorted(self.allowed_impls))
        out = []
        for impl, _ in sorted(found, key=lambda pair: pair[0].impl_id):
            profiles = expansion_profiles(
                impl, named_profiles(self.profile_ids),
                dict(self.impl_profiles) if self.impl_profiles else None)
            out.append((impl, tuple(profiles)))
        return out

    def branching(self, applied: frozenset[str]) -> list[tuple[str, int]]:
        """`(impl_id, how many states it generates)`."""
        return [(impl.impl_id, len(profiles))
                for impl, profiles in self.options(applied)]

    def is_complete(self, applied: frozenset[str]) -> bool:
        return not self.spec_of(applied).diff(self.target)


# --- how big is it ----------------------------------------------------------


def walk_leaves(space: SearchSpace) -> Iterator[tuple[tuple[str, ...],
                                                      tuple[str, ...]]]:
    """Every reachable ordered complete path, as `(impl_ids, labels)`.

    Enumeration, not estimation: the recursion asks the registry what applies
    at each parent, exactly as the search will. This is what makes a stated
    space size checkable — a theoretical product like
    `kinds! x impls x profiles` silently assumes every kind is required and
    every operator addresses exactly one field, and neither holds once a
    composite operator can reach the target in a single step.
    """
    def walk(applied: frozenset[str], spec: ArchSpec,
             steps: tuple[str, ...], labels: tuple[str, ...]):
        if not spec.diff(space.target):
            yield steps, labels
            return
        kinds = tuple(sorted({get_implementation(i).kind for i in applied}))
        found = applicable_implementations(
            space.adapter, spec, space.target,
            exclude_kinds=kinds, allow_impls=sorted(space.allowed_impls))
        for impl, _ in sorted(found, key=lambda pair: pair[0].impl_id):
            profiles = expansion_profiles(
                impl, named_profiles(space.profile_ids),
                dict(space.impl_profiles) if space.impl_profiles else None)
            nxt = impl.plan(spec, space.target, space.adapter, {}).result_spec
            for profile in profiles:
                pid = getattr(profile, "qualified_id", str(profile))
                yield from walk(applied | {impl.impl_id}, nxt,
                                steps + (impl.impl_id,),
                                labels + (f"{impl.kind}({pid})",))
    yield from walk(frozenset(), space.teacher, (), ())


def decomposition(space: SearchSpace) -> dict[str, Any]:
    """The space's size, derived, and split by how many operators a leaf uses.

    The split matters and a single total hides it: a composite operator that
    reaches the target in one step contributes leaves that are not part of the
    decomposed subspace at all, so "the space is N" and "the decomposed space
    is N" are different claims about different sets.
    """
    by_length: dict[int, int] = {}
    per_kind_options: dict[str, list[str]] = {}
    for steps, _labels in walk_leaves(space):
        by_length[len(steps)] = by_length.get(len(steps), 0) + 1
    for impl, profiles in space.options(frozenset()):
        per_kind_options.setdefault(impl.kind, []).extend(
            f"{impl.impl_id}@{getattr(p, 'qualified_id', p)}" for p in profiles)
    total = sum(by_length.values())
    longest = max(by_length) if by_length else 0
    return {
        "total_leaves": total,
        "leaves_by_operator_count": {str(k): v for k, v in sorted(by_length.items())},
        "decomposed_leaves": by_length.get(longest, 0),
        "decomposed_operator_count": longest,
        "required_fields": list(space.required_fields),
        "root_options_by_kind": {k: sorted(v)
                                 for k, v in sorted(per_kind_options.items())},
        "allowed_impls": list(space.allowed_impls),
        "profile_ids": list(space.profile_ids),
        "_derivation": (
            "enumerated by walking the real registry through "
            "applicable_implementations and expansion_profiles at every parent, "
            "the same two functions the search itself calls. Not a product "
            "formula: a composite operator reaching the target in one step "
            "contributes leaves outside the decomposed subspace, so the total "
            "and the decomposed count are different numbers and both are given."),
    }


# --- what does it cost ------------------------------------------------------


@dataclass
class Bound:
    min_minutes: float
    max_minutes: float
    min_expansions: int
    max_expansions: int
    statistic: str
    unmeasured: tuple[str, ...]
    beam_width: int
    warmup_levels: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "beam_width": self.beam_width,
            "warmup_levels": self.warmup_levels,
            "min_minutes": round(self.min_minutes, 2),
            "max_minutes": round(self.max_minutes, 2),
            "min_hours": round(self.min_minutes / 60, 3),
            "max_hours": round(self.max_minutes / 60, 3),
            "min_expansions": self.min_expansions,
            "max_expansions": self.max_expansions,
            "_extrema_are_separate": (
                "min/max minutes and min/max expansions are each the extremum "
                "of their OWN quantity over beams. The cheapest beam is not the "
                "smallest: deferring the expensive operator is cheap now and "
                "generates more children later, so one number cannot describe "
                "both."),
            "unmeasured_inputs": list(self.unmeasured),
        }


def level_children(space: SearchSpace, beam: tuple[tuple[frozenset[str], int], ...],
                   cost: CostModel, *, statistic: str = "max"):
    """One level of expansion: `(minutes, partial classes -> count, n)`.

    The whole branching model, and the only place it lives. :func:`bound`
    optimises over it and an instance's replay back-test feeds it a real run's
    beams.
    """
    minutes = 0.0
    n = 0
    partial: dict[frozenset[str], int] = {}
    for applied, count in beam:
        root = not applied
        for impl_id, branches in space.branching(applied):
            minutes += count * branches * cost.minutes_for(
                impl_id, root=root, statistic=statistic)
            n += count * branches
            child = applied | {impl_id}
            if not space.is_complete(child):
                partial[child] = partial.get(child, 0) + count * branches
    return minutes, partial, n


def compositions(pool: Mapping[frozenset[str], int], keep: int | None):
    """Every sub-multiset of `pool` a beam of `keep` could be."""
    classes = sorted(pool, key=lambda c: tuple(sorted(c)))
    total = sum(pool.values())
    take = total if keep is None else min(keep, total)
    if take == total:
        yield tuple((c, pool[c]) for c in classes)
        return
    for counts in itertools.product(*(range(pool[c] + 1) for c in classes)):
        if sum(counts) == take:
            yield tuple((c, k) for c, k in zip(classes, counts) if k)


def bound(space: SearchSpace, cost: CostModel, *, beam_width: int,
          warmup_levels: int, statistic: str = "max",
          max_depth: int | None = None) -> Bound:
    """Exact extrema over every beam the ranking policy could return.

    Not a projection. The recursion enumerates every admissible beam
    composition, so `max_minutes` is attained by some ranking and no ranking can
    exceed it. What it deliberately does NOT model is which beam the Pareto
    policy actually returns: that depends on measurements this function has not
    taken, and guessing it is what turned Phase B's bound into an underestimate.

    Minutes and expansions are optimised SEPARATELY, because they disagree.
    """
    depth = max_depth if max_depth is not None else len(
        space.teacher.diff(space.target))

    def extremum(weigh, pick):
        @lru_cache(maxsize=None)
        def walk(beam, level):
            if not beam or level >= depth:
                return 0.0
            minutes, partial, n = level_children(space, beam, cost,
                                                 statistic=statistic)
            here = weigh(minutes, n)
            if not partial:
                return here
            keep = None if level < warmup_levels else beam_width
            return here + pick(walk(comp, level + 1) for comp
                               in compositions(partial, keep))
        return walk(((frozenset(), 1),), 0)

    return Bound(
        min_minutes=extremum(lambda m, n: m, min),
        max_minutes=extremum(lambda m, n: m, max),
        min_expansions=int(extremum(lambda m, n: float(n), min)),
        max_expansions=int(extremum(lambda m, n: float(n), max)),
        statistic=statistic, beam_width=beam_width,
        warmup_levels=warmup_levels,
        unmeasured=tuple(i for i in cost.unmeasured
                         if i in space.allowed_impls))


def trajectory(space: SearchSpace, cost: CostModel, *, prefer_costly: bool,
               beam_width: int, warmup_levels: int, statistic: str = "max",
               costly_impl: str | None = None) -> dict[str, Any]:
    """Cost ONE nominated beam trajectory, level by level.

    Not a bound — :func:`bound` is the bound. This answers a different and also
    necessary question: what does the search cost if the ranking behaves the way
    it was last observed to behave? At Phase-B level 1 all six retained states
    contained the DEPTH operator, so `prefer_costly=True` is the trajectory with
    evidence behind it and `prefer_costly=False` is the adversary the ceiling
    has to survive.

    The gap between them is the whole cost risk, and it is a BEAM-COMPOSITION
    risk rather than a per-node one: the expensive operator costs what it costs
    wherever it runs, so what the price turns on is how many beam members still
    owe it.

    **Which operator is "the expensive one" is DERIVED** from the cost model
    rather than named, so this function carries no operator identity and a
    different family's search nominates its own trajectory correctly.
    """
    costly = costly_impl or cost.most_expensive(space.allowed_impls,
                                                statistic=statistic)
    depth = len(space.teacher.diff(space.target))

    beam: tuple[tuple[frozenset[str], int], ...] = ((frozenset(), 1),)
    minutes = 0.0
    expansions = 0
    rows = []
    for level in range(depth):
        if not beam:
            break
        m, partial, n = level_children(space, beam, cost, statistic=statistic)
        minutes += m
        expansions += n
        rows.append({"level": level, "parents": sum(c for _, c in beam),
                     "generated": n, "minutes": round(m, 1)})
        if not partial:
            break
        ordered = sorted(partial.items(), key=lambda kv: (
            (costly not in kv[0]) if prefer_costly
            else (costly in kv[0]), tuple(sorted(kv[0]))))
        keep: list[tuple[frozenset[str], int]] = []
        room = sum(partial.values()) if level < warmup_levels else beam_width
        for cls, count in ordered:
            if room <= 0:
                break
            take = min(count, room)
            keep.append((cls, take))
            room -= take
        beam = tuple(keep)
    return {"minutes": round(minutes, 2), "hours": round(minutes / 60, 3),
            "expansions": expansions, "levels": rows,
            "prefer_costly": prefer_costly, "costly_impl": costly,
            "statistic": statistic}


def price(space: SearchSpace, cost: CostModel, *, price_per_hour: float,
          authorized_usd: float, session_phases: Sequence[tuple[str, float]],
          beam_width: int, warmup_levels: int, statistic: str = "max",
          setup_phase: str, transfer_phase: str,
          search_phase_name: str = "beam_search",
          contingency_fraction: float = 0.10,
          artifact_recovery_reserve_minutes: float = 30.0):
    """A `BudgetPlan` for a search-only session. Priced, which is not funded.

    The expected path carries the costly-operator-early search; the difference
    up to the structural worst case is a named `soft_stop_reserve`, which is
    exactly what that field is for — an identified, bounded risk that is not on
    the expected path, added after the contingency multiplier so it protects the
    work rather than merely moving the watchdog's kill time.

    `plan_session` RAISES when the plan does not fit `authorized_usd`. That
    refusal is the point: it is how a search that has outgrown its budget says
    so, instead of being quietly shrunk to fit.
    """
    from aadistill.infrastructure.budget import (
        MEASURED_STEP_SECONDS, Phase, StepTime, plan_session)

    expected = trajectory(space, cost, prefer_costly=True,
                          beam_width=beam_width, warmup_levels=warmup_levels,
                          statistic=statistic)
    limit = bound(space, cost, beam_width=beam_width,
                  warmup_levels=warmup_levels, statistic=statistic)
    phases = dict(session_phases)
    return plan_session(
        price_per_hour=price_per_hour, authorized_usd=authorized_usd,
        #: No training. The search is a phase, not an arm; `step_time` is
        #: required by the plan type and multiplies zero arms.
        arms=0, steps_per_arm=0,
        step_time=StepTime(
            seconds=MEASURED_STEP_SECONDS,
            source=("unused: a search session trains nothing, so arms=0 and "
                    "the step term is zero. The measured floor is passed so "
                    "the below-floor guard cannot be satisfied by accident")),
        setup_minutes=phases[setup_phase],
        transfer_minutes=phases[transfer_phase],
        other_phases=(
            *(Phase(name, m) for name, m in session_phases
              if name not in (setup_phase, transfer_phase)),
            Phase(search_phase_name, expected["minutes"]),
        ),
        contingency_fraction=contingency_fraction,
        soft_stop_reserves=(
            Phase("beam_composition_risk",
                  round(limit.max_minutes - expected["minutes"], 2)),),
        artifact_recovery_reserve_minutes=artifact_recovery_reserve_minutes)


__all__ = [
    "Bound", "CostModel", "CostModelError", "SearchSpace", "bound",
    "compositions", "decomposition", "level_children", "named_profiles",
    "price", "trajectory", "walk_leaves",
]
