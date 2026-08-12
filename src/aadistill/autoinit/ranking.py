"""``BeamRankingPolicy`` — a separately versioned, hashable selection rule.

Ranking is **not** minimum NLL, and that is an empirical finding rather than a
preference. E7 moved held-out FineWeb NLL by −5.22 nats and autonomous behaviour
by exactly +0.0000, and the checkpoint with the best held-out NLL of its
trajectory produced *zero* protocol-valid generations. A single-objective beam on
NLL would have confidently pruned the states worth keeping.

So v1 is Pareto-first: a state survives if nothing in the round dominates it
across every declared objective. Only when a front is larger than the beam does a
tie-break run, and the tie-break is a configured ordered list ending in the state
id — a total order, so the same round always yields the same beam.

Three separations are enforced here:

* objectives must be ``state.`` metrics. An operator-local objective cannot rank
  the beam (``metrics.require_state_metric``), because E8a is a worked example of
  an operator-local win that reversed once composed.
* the policy is data, not code in an operator. Operators never see it.
* the policy hashes. A run records the hash; changing an objective changes the
  hash and therefore cannot be passed off as the same policy afterwards.

``PARETO_V1`` is a **default, not a decision**. The exact paid-pilot policy is
frozen before launch and recorded in the preregistration.
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
class BeamRankingPolicy:
    policy_id: str
    version: int
    description: str
    objectives: tuple[Objective, ...]
    tie_break: tuple[str, ...]
    guardrails: tuple[Guardrail, ...] = ()
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
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id, "version": self.version,
            "description": self.description,
            "objectives": [o.as_dict() for o in self.objectives],
            "tie_break": list(self.tie_break),
            "guardrails": [g.as_dict() for g in self.guardrails],
            "required_metrics": list(self.required_metrics()),
            "policy_hash": self.policy_hash,
            "metadata": dict(self.metadata),
        }

    # --- ranking -----------------------------------------------------------

    def _vector(self, state: InitializationState) -> tuple[float, ...]:
        values = state.evaluation.values
        return tuple(float(values[o.key]) for o in self.objectives)

    def _dominates(self, a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        """``a`` dominates ``b``: no worse on every objective, strictly better on one."""
        no_worse = all(not o.better(bv, av)
                       for o, av, bv in zip(self.objectives, a, b))
        strictly = any(o.better(av, bv) for o, av, bv in zip(self.objectives, a, b))
        return no_worse and strictly

    def _tie_key(self, state: InitializationState) -> tuple:
        out: list[Any] = []
        for key in self.tie_break:
            if key == STATE_ID_TIEBREAK:
                out.append(state.state_id)
            else:
                out.append(float(state.evaluation.values[key]))
        return tuple(out)

    def rank(self, states: Sequence[InitializationState], beam_width: int) -> "RankingResult":
        if beam_width < 1:
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

        selected: list[InitializationState] = []
        decisions: list[dict[str, Any]] = []
        for f_index, front in enumerate(fronts):
            for position, state in enumerate(front):
                room = beam_width - len(selected)
                keep = room > 0
                if keep:
                    selected.append(state)
                decisions.append({
                    "state_id": state.state_id,
                    "path": state.path_label,
                    "front": f_index,
                    "position_in_front": position,
                    "selected": keep,
                    "objectives": {o.key: float(state.evaluation.values[o.key])
                                   for o in self.objectives},
                    "reason": ("kept: Pareto front %d" % f_index) if keep else
                              (f"pruned: dominated (front {f_index}) and the beam of "
                               f"{beam_width} was already full"
                               if f_index else
                               f"pruned: non-dominated but ranked {position + 1} in "
                               f"front 0 under the tie-break, beam {beam_width}"),
                })
        for state, why in rejected:
            decisions.append({"state_id": state.state_id, "path": state.path_label,
                              "front": None, "position_in_front": None,
                              "selected": False, "objectives": {},
                              "reason": f"pruned: {why}"})

        return RankingResult(
            policy_id=self.qualified_id, policy_hash=self.policy_hash,
            beam_width=beam_width, selected=tuple(selected),
            fronts=tuple(tuple(s.state_id for s in f) for f in fronts),
            decisions=tuple(decisions),
        )


@dataclass(frozen=True)
class RankingResult:
    policy_id: str
    policy_hash: str
    beam_width: int
    selected: tuple[InitializationState, ...]
    fronts: tuple[tuple[str, ...], ...]
    decisions: tuple[dict[str, Any], ...]

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
            "decisions": list(self.decisions),
        }


VALIDITY_GUARDRAIL = Guardrail(
    name="structurally_valid",
    description="the state reached MEASURED through materialize -> reload -> hash -> validate",
    predicate=_is_measured,
)

#: v1 default. Frozen per run, not per session: a paid search records this hash
#: in its preregistration and the manifest asserts it did not move.
PARETO_V1 = BeamRankingPolicy(
    policy_id="beam.pareto_multi_objective",
    version=1,
    description=(
        "Non-dominated sorting over original-teacher domain-balanced KL, "
        "critical-token KL and general NLL, with a total tie-break. NLL "
        "participates but cannot prune alone: a state that is worse on NLL and "
        "better on reasoning/domain fidelity is non-dominated and survives."),
    objectives=(
        Objective("state.teacher_kl.equal_domain_mean", "minimize",
                  "equal-domain mean forward KL from the original teacher"),
        Objective("state.critical_token_kl", "minimize",
                  "unweighted mean KL over declared critical-token classes"),
        Objective("state.nll.general", "minimize",
                  "general-language NLL, a guardrail axis rather than the target"),
    ),
    tie_break=(
        "state.teacher_kl.equal_domain_mean",
        "state.critical_token_kl",
        "state.nll.general",
        STATE_ID_TIEBREAK,
    ),
    guardrails=(VALIDITY_GUARDRAIL,),
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
