"""Beam selection: a hashable ranking policy plus a hashable width schedule.

Ranking is **not** minimum NLL, and that is an empirical finding rather than a
preference. E7 moved held-out FineWeb NLL by −5.22 nats and autonomous behaviour
by exactly +0.0000, and the checkpoint with the best held-out NLL of its
trajectory produced *zero* protocol-valid generations. NLL is therefore recorded
as a diagnostic and is **not** a v1 objective: it may not be the reason a path
dies.

Three mechanisms keep hypotheses alive that a naive beam would kill.

**Delayed pruning.** ``BeamSchedule`` retains *every* child of the root before any
quality pruning happens. Level 0 offers one child per applicable operator, and
those are the distinct structural hypotheses the search exists to compare;
discarding one on a single step-0 measurement is exactly the mistake E8a
documented, where a proxy that looked 3.11x better reversed after composition.

**ε-dominance.** A state is only eliminated when another is *meaningfully* better:
no worse than ε on any objective and better by more than ε on at least one.
Plain floating-point Pareto lets a 1e-9 edge on one axis kill a genuinely
different hypothesis. With ε = 0 this reduces exactly to strict Pareto.

**Lineage diversity.** Within a front, selection rotates over distinct parent
paths before falling through to the deterministic state-id tie-break, so a beam
of six cannot fill up with six variants of one hypothesis while another is
extinguished.

Two separations are enforced as before: objectives must be ``state.`` metrics
(``metrics.require_state_metric``), and both the policy and the schedule hash, so
a changed rule cannot be passed off afterwards as the one that ran.

``PARETO_V1`` and ``SCHEDULE_V1`` are the v1 defaults. The exact paid-pilot
selection rule is frozen in the preregistration before launch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .metrics import require_state_metric
from .state import InitializationState, StateValidity

STATE_ID_TIEBREAK = "state_id"


class RankingError(RuntimeError):
    """A policy could not rank the states it was given."""


@dataclass(frozen=True)
class Objective:
    key: str
    direction: str = "minimize"
    description: str = ""

    def __post_init__(self) -> None:
        require_state_metric(self.key)
        if self.direction not in ("minimize", "maximize"):
            raise RankingError(f"{self.key}: direction must be minimize or maximize")

    def better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "minimize" else a > b

    def sort_value(self, v: float) -> float:
        return v if self.direction == "minimize" else -v

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "direction": self.direction,
                "description": self.description}


@dataclass(frozen=True)
class Guardrail:
    """A hard filter applied before Pareto sorting.

    Guardrails exist for *structural* disqualifiers — a state that failed
    validation, produced non-finite logits, or blew past a size bound. Using one
    as a quality threshold on a scalar metric would reintroduce exactly the
    single-objective pruning the Pareto front is here to avoid, so the default
    policy ships with one guardrail and it is about validity.
    """

    name: str
    description: str
    predicate: Callable[[InitializationState], bool]

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


def _is_measured(state: InitializationState) -> bool:
    return state.validity is StateValidity.MEASURED and state.evaluation is not None


@dataclass(frozen=True)
class BeamSchedule:
    """How many states survive at each level. Hashable, and part of the record.

    Separate from the ranking policy because they answer different questions:
    the policy says which states are better, the schedule says how many the
    search is willing to carry. Changing either changes the search, so both hash.
    """

    schedule_id: str
    version: int
    description: str
    #: Levels expanded with **no** quality pruning at all. `1` retains every
    #: child of the root.
    warmup_levels: int
    #: Beam width once warmup ends.
    width: int

    def __post_init__(self) -> None:
        if self.warmup_levels < 0:
            raise RankingError(f"{self.schedule_id}: warmup_levels must be >= 0")
        if self.width < 1:
            raise RankingError(f"{self.schedule_id}: width must be >= 1")

    @property
    def qualified_id(self) -> str:
        return f"{self.schedule_id}@v{self.version}"

    def width_at(self, level: int) -> int | None:
        """Beam width for a level; ``None`` means retain everything."""
        return None if level < self.warmup_levels else self.width

    @property
    def schedule_hash(self) -> str:
        return hashlib.sha256(json.dumps({
            "schedule_id": self.schedule_id, "version": self.version,
            "warmup_levels": self.warmup_levels, "width": self.width,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {"schedule_id": self.schedule_id, "version": self.version,
                "description": self.description,
                "warmup_levels": self.warmup_levels, "width": self.width,
                "schedule_hash": self.schedule_hash}


@dataclass(frozen=True)
class BeamRankingPolicy:
    policy_id: str
    version: int
    description: str
    objectives: tuple[Objective, ...]
    tie_break: tuple[str, ...]
    guardrails: tuple[Guardrail, ...] = ()
    #: Per-objective practical-equivalence tolerance. A difference no larger than
    #: this is not a reason to eliminate anything.
    epsilon: Mapping[str, float] = field(default_factory=dict)
    #: How lineage is defined for diversity. ``parent_path`` groups states by the
    #: operator path that produced their parent, which at level 1 is exactly the
    #: first-step hypothesis.
    diversity_key: str = "parent_path"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objectives:
            raise RankingError(f"{self.policy_id}: declares no objectives")
        if len(self.objectives) == 1:
            # Not a style rule: a single-objective beam is the failure mode E7
            # documented. If one is genuinely wanted, it needs its own policy id
            # and an explicit override in metadata.
            if not self.metadata.get("single_objective_acknowledged"):
                raise RankingError(
                    f"{self.policy_id}: a single-objective beam reduces to "
                    "selecting on one scalar, which E7 showed can prune every "
                    "state worth keeping. Declare "
                    "metadata['single_objective_acknowledged'] to insist.")
        if not self.tie_break or self.tie_break[-1] != STATE_ID_TIEBREAK:
            raise RankingError(
                f"{self.policy_id}: tie_break must end in {STATE_ID_TIEBREAK!r} so the "
                "ordering is total and the beam is reproducible")
        for key in self.tie_break[:-1]:
            require_state_metric(key)
        declared = {o.key for o in self.objectives}
        stray = sorted(set(self.epsilon) - declared)
        if stray:
            raise RankingError(
                f"{self.policy_id}: epsilon declared for non-objectives {stray}; a "
                "tolerance on a metric nothing is ranked by has no effect and reads "
                "as though it does")
        if any(v < 0 for v in self.epsilon.values()):
            raise RankingError(f"{self.policy_id}: negative epsilon")
        if self.diversity_key not in ("parent_path", "first_step", "none"):
            raise RankingError(
                f"{self.policy_id}: unknown diversity_key {self.diversity_key!r}")

    @property
    def qualified_id(self) -> str:
        return f"{self.policy_id}@v{self.version}"

    def required_metrics(self) -> tuple[str, ...]:
        keys = [o.key for o in self.objectives]
        keys += [k for k in self.tie_break if k != STATE_ID_TIEBREAK]
        return tuple(dict.fromkeys(keys))

    @property
    def policy_hash(self) -> str:
        return hashlib.sha256(json.dumps({
            "policy_id": self.policy_id,
            "version": self.version,
            "objectives": [o.as_dict() for o in self.objectives],
            "tie_break": list(self.tie_break),
            "guardrails": [g.as_dict() for g in self.guardrails],
            "epsilon": dict(sorted(self.epsilon.items())),
            "diversity_key": self.diversity_key,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id, "version": self.version,
            "description": self.description,
            "objectives": [o.as_dict() for o in self.objectives],
            "tie_break": list(self.tie_break),
            "guardrails": [g.as_dict() for g in self.guardrails],
            "epsilon": dict(sorted(self.epsilon.items())),
            "diversity_key": self.diversity_key,
            "required_metrics": list(self.required_metrics()),
            "policy_hash": self.policy_hash,
            "metadata": dict(self.metadata),
        }

    # --- ranking -----------------------------------------------------------

    def _vector(self, state: InitializationState) -> tuple[float, ...]:
        values = state.evaluation.values
        return tuple(float(values[o.key]) for o in self.objectives)

    def _dominates(self, a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        """``a`` ε-dominates ``b``: nowhere meaningfully worse, somewhere meaningfully better.

        Both halves use the tolerance. Without it, a state that is 1e-9 better on
        one axis and identical elsewhere eliminates a structurally different
        hypothesis, which is a floating-point accident rather than a finding. With
        ε = 0 this is exactly strict Pareto dominance.
        """
        eps = [float(self.epsilon.get(o.key, 0.0)) for o in self.objectives]
        no_worse = all(
            not o.better(bv, av) or abs(av - bv) <= e
            for o, av, bv, e in zip(self.objectives, a, b, eps))
        strictly = any(
            o.better(av, bv) and abs(av - bv) > e
            for o, av, bv, e in zip(self.objectives, a, b, eps))
        return no_worse and strictly

    def _tie_key(self, state: InitializationState) -> tuple:
        out: list[Any] = []
        for key in self.tie_break:
            if key == STATE_ID_TIEBREAK:
                out.append(state.state_id)
            else:
                out.append(float(state.evaluation.values[key]))
        return tuple(out)

    def lineage(self, state: InitializationState) -> str:
        """The group a state belongs to for diversity purposes."""
        if self.diversity_key == "none":
            return state.state_id
        if self.diversity_key == "first_step":
            return state.impl_ids[0] if state.impl_ids else "root"
        return "|".join(state.impl_ids[:-1]) or "root"   # parent_path

    def _select_with_diversity(self, ordered: Sequence[InitializationState],
                               room: int) -> list[InitializationState]:
        """Take ``room`` states, giving every lineage a slot before any gets two.

        ``ordered`` is the whole eligible set sorted by (front, tie-break), so
        the selection is still quality-first *within* a lineage. The rotation
        runs **across fronts**, not only inside one, because the failure this
        exists to prevent is precisely a lineage whose states all dominate
        another's: front-order selection alone would take the first lineage's
        second-best before the second lineage's best and extinguish a distinct
        structural hypothesis on a step-0 measurement. E8a is the standing
        warning that a step-0 ordering can reverse after composition.
        """
        if room >= len(ordered):
            return list(ordered)
        if self.diversity_key == "none":
            return list(ordered[:room])
        groups: dict[str, list[InitializationState]] = {}
        for state in ordered:
            groups.setdefault(self.lineage(state), []).append(state)
        # Lineages are visited in the order of their own best member, so a
        # stronger hypothesis still gets served first on every rotation.
        order = sorted(groups, key=lambda k: ordered.index(groups[k][0]))
        picked: list[InitializationState] = []
        while len(picked) < room:
            progressed = False
            for key in order:
                if groups[key]:
                    picked.append(groups[key].pop(0))
                    progressed = True
                    if len(picked) == room:
                        break
            if not progressed:   # pragma: no cover - room <= len(ordered) guarantees progress
                break
        return picked

    def rank(self, states: Sequence[InitializationState],
             beam_width: int | None) -> "RankingResult":
        """Rank states; ``beam_width=None`` retains every eligible state.

        ``None`` is how the schedule expresses a warmup level. It is a distinct
        value rather than a very large width so the manifest can say "this level
        pruned nothing by design" instead of "this level happened not to prune".
        """
        if beam_width is not None and beam_width < 1:
            raise RankingError("beam width must be at least 1")
        required = self.required_metrics()
        eligible, rejected = [], []
        for state in states:
            if not _is_measured(state):
                rejected.append((state, f"not measured ({state.validity.value})"))
                continue
            try:
                state.ready_for_ranking(required)
            except Exception as exc:
                rejected.append((state, f"unrankable: {exc}"))
                continue
            failed = [g.name for g in self.guardrails if not g.predicate(state)]
            if failed:
                rejected.append((state, f"guardrail {failed}"))
                continue
            eligible.append(state)

        vectors = {s.state_id: self._vector(s) for s in eligible}
        remaining = list(eligible)
        fronts: list[list[InitializationState]] = []
        while remaining:
            front = [
                s for s in remaining
                if not any(self._dominates(vectors[o.state_id], vectors[s.state_id])
                           for o in remaining if o is not s)
            ]
            if not front:  # pragma: no cover - only reachable with a broken comparator
                raise RankingError(
                    f"{self.qualified_id}: dominance produced an empty front over "
                    f"{len(remaining)} states; the comparator is not a strict order")
            fronts.append(sorted(front, key=self._tie_key))
            ids = {id(s) for s in front}
            remaining = [s for s in remaining if id(s) not in ids]

        ordered = [state for front in fronts for state in front]
        room = len(ordered) if beam_width is None else beam_width
        selected = self._select_with_diversity(ordered, room)
        chosen_ids = {s.state_id for s in selected}

        decisions: list[dict[str, Any]] = []
        for f_index, front in enumerate(fronts):
            for position, state in enumerate(front):
                keep = state.state_id in chosen_ids
                if keep and beam_width is None:
                    reason = "kept: warmup level, no quality pruning by design"
                elif keep:
                    reason = (f"kept: front {f_index}, lineage "
                              f"{self.lineage(state)!r}, beam {beam_width}")
                elif f_index:
                    reason = (f"pruned: epsilon-dominated (front {f_index}) and the "
                              f"beam of {beam_width} was already full")
                else:
                    reason = (f"pruned: non-dominated but ranked {position + 1} in "
                              f"front 0 under the tie-break and lineage rotation, "
                              f"beam {beam_width}")
                decisions.append({
                    "state_id": state.state_id,
                    "path": state.path_label,
                    "front": f_index,
                    "position_in_front": position,
                    "lineage": self.lineage(state),
                    "selected": keep,
                    "objectives": {o.key: float(state.evaluation.values[o.key])
                                   for o in self.objectives},
                    "diagnostics": {k: float(v) for k, v in
                                    state.evaluation.values.items()
                                    if k.startswith("state.nll.")},
                    "reason": reason,
                })
        for state, why in rejected:
            decisions.append({"state_id": state.state_id, "path": state.path_label,
                              "front": None, "position_in_front": None,
                              "lineage": None, "selected": False, "objectives": {},
                              "diagnostics": {}, "reason": f"pruned: {why}"})

        return RankingResult(
            policy_id=self.qualified_id, policy_hash=self.policy_hash,
            beam_width=beam_width, selected=tuple(selected),
            fronts=tuple(tuple(s.state_id for s in f) for f in fronts),
            decisions=tuple(decisions),
            lineages_kept=tuple(sorted({self.lineage(s) for s in selected})),
        )


@dataclass(frozen=True)
class RankingResult:
    policy_id: str
    policy_hash: str
    beam_width: int | None
    selected: tuple[InitializationState, ...]
    fronts: tuple[tuple[str, ...], ...]
    decisions: tuple[dict[str, Any], ...]
    lineages_kept: tuple[str, ...] = ()

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return tuple(s.state_id for s in self.selected)

    @property
    def pruned(self) -> tuple[dict[str, Any], ...]:
        return tuple(d for d in self.decisions if not d["selected"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id, "policy_hash": self.policy_hash,
            "beam_width": self.beam_width, "selected": list(self.selected_ids),
            "fronts": [list(f) for f in self.fronts],
            "lineages_kept": list(self.lineages_kept),
            "decisions": list(self.decisions),
        }


VALIDITY_GUARDRAIL = Guardrail(
    name="structurally_valid",
    description="the state reached MEASURED through materialize -> reload -> hash -> validate",
    predicate=_is_measured,
)

#: v1 default. Frozen per run, not per session: a paid search records this hash
#: in its preregistration and the manifest asserts it did not move.
#:
#: **NLL is not an objective.** E7 is the reason, and it is a direct measurement
#: rather than a worry: a −5.22 nat swing in held-out NLL moved autonomous
#: behaviour by exactly +0.0000, and the best-NLL checkpoint of its trajectory
#: produced zero protocol-valid generations. NLL is recorded per domain and shown
#: beside every ranking decision as a diagnostic; it may not kill a path.
#:
#: The three objectives are original-teacher fidelity seen three ways: the
#: equal-domain mean (central tendency), the worst domain (so a path cannot buy
#: an average by sacrificing tool or code fidelity outright), and critical-token
#: fidelity (the tokens that decide whether a rollout terminates are a vanishing
#: fraction of positions and a token-mean washes them out).
#:
#: Using all five per-domain KLs as separate objectives was considered and
#: rejected for v1: with five domains plus critical tokens over ~30 states almost
#: nothing is dominated, so the tie-break rather than the dominance rule would be
#: doing the selecting — which should be a decision, not a side effect.
PARETO_V1 = BeamRankingPolicy(
    policy_id="beam.pareto_multi_objective",
    version=2,
    description=(
        "epsilon-Pareto over original-teacher fidelity: equal-domain mean KL, "
        "worst-domain KL and critical-token KL. NLL is diagnostic and never "
        "prunes. Lineage diversity precedes the deterministic state-id tie-break."),
    objectives=(
        Objective("state.teacher_kl.equal_domain_mean", "minimize",
                  "equal-domain mean forward KL from the original teacher"),
        Objective("state.teacher_kl.worst_domain", "minimize",
                  "the domain this state preserves least well"),
        Objective("state.critical_token_kl", "minimize",
                  "unweighted mean KL over declared critical-token classes"),
    ),
    tie_break=(
        "state.teacher_kl.equal_domain_mean",
        "state.teacher_kl.worst_domain",
        "state.critical_token_kl",
        STATE_ID_TIEBREAK,
    ),
    guardrails=(VALIDITY_GUARDRAIL,),
    # Practical equivalence. 1e-4 nats of KL is far below anything this project
    # has ever been able to act on, and treating it as decisive would let
    # arithmetic noise extinguish a structurally distinct path.
    epsilon={
        "state.teacher_kl.equal_domain_mean": 1e-4,
        "state.teacher_kl.worst_domain": 1e-4,
        "state.critical_token_kl": 1e-4,
    },
    diversity_key="parent_path",
    metadata={"nll_status": "diagnostic only; recorded per domain, never an objective"},
)

#: v1 schedule: retain every child of the root, then a beam of 6.
#:
#: Level 0 offers one child per applicable operator — five for the decomposed
#: library at one calibration profile, plus the composite. Those are the distinct
#: structural hypotheses the search exists to compare, and pruning one of them on
#: a single step-0 measurement is the mistake E8a documented: a proxy that looked
#: 3.11x better reversed once composition happened. Width 6 afterwards is wider
#: than the 5 hypotheses it carries forward, so the first pruning level can keep
#: every surviving lineage and still admit a second variant of one of them.
SCHEDULE_V1 = BeamSchedule(
    schedule_id="beam.delayed_prune",
    version=1,
    description=("no quality pruning at level 0; beam width 6 from level 1 onward"),
    warmup_levels=1,
    width=6,
)

_POLICIES: dict[str, BeamRankingPolicy] = {PARETO_V1.qualified_id: PARETO_V1}


def register_policy(policy: BeamRankingPolicy, *, replace: bool = False) -> BeamRankingPolicy:
    existing = _POLICIES.get(policy.qualified_id)
    if existing is not None and not replace:
        if existing.policy_hash != policy.policy_hash:
            raise RankingError(
                f"{policy.qualified_id} is already registered with hash "
                f"{existing.policy_hash[:12]}; a changed policy needs a new version")
        return existing
    _POLICIES[policy.qualified_id] = policy
    return policy


def get_policy(qualified_id: str) -> BeamRankingPolicy:
    if qualified_id not in _POLICIES:
        raise KeyError(f"no ranking policy {qualified_id!r}; registered: {sorted(_POLICIES)}")
    return _POLICIES[qualified_id]


def registered_policies() -> list[str]:
    return sorted(_POLICIES)
