"""The quality contract for a teacher-generated training target.

A teacher candidate is accepted as a training target only for what can be
shown about it, and every candidate -- accepted or not -- keeps the full record
of what was shown. The categories are kept apart because they answer different
questions and none implies another:

* ``naturally_terminated`` -- the generation ended on its own end token;
* ``length_limited`` -- it reached the generation allowance (right-censored,
  never trimmed into a target);
* ``degenerated`` -- a degeneration detector stopped or flagged it;
* ``protocol_valid`` -- the teacher's own output protocol parses (reasoning
  delimiters closed, a non-empty final answer, no stray markers);
* ``answer`` -- ``correct`` / ``incorrect`` / ``unverifiable``, from a
  task-specific verifier the caller supplies, recorded with its ``scope``:
  what the check actually covers ("final answer equals the generator's gold";
  "the declared tests passed: not full semantic correctness"; "the
  environment's goal was reached on replay");
* ``steps`` -- verification of the INTERMEDIATE steps, kept apart from the
  answer: ``verified`` (a check covered every step), ``partial`` (a check
  covered some steps, or showed that the steps were made without showing that
  each is valid), ``contradicted`` (a step conflicts with the task), or
  ``not_assessed``. Only a check that reads the steps against the task sets
  it. A correct final answer, an answer restated at the end of the reasoning,
  or a legal action sequence does not establish that the reasoning is right;
  where no step checker exists the state is ``not_assessed`` and is reported
  so;
* ``observations`` -- any other named check, in its own vocabulary, recorded
  and counted but a verdict only where the group's policy says so: textual
  consistency between the reasoning and the final answer, environment action
  legality, and the like;
* ``accepted_for_training`` -- the conjunction the caller's policy requires.

Length is recorded, never judged: no rule here accepts, rejects or selects a
candidate by how long it is (AGENTS.md P10, P17). When several candidates are
accepted, the target is the lowest candidate index (sampling order), which is
independent of length and of the verifier's internals.

The verifiers, the degeneration detector and the protocol check are injected;
nothing here knows a task family, a dataset or a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

CONTRACT_ID = "aadistill.data.target_quality/v2"

ANSWER_STATES = ("correct", "incorrect", "unverifiable")
STEP_STATES = ("verified", "partial", "contradicted", "not_assessed")


@dataclass(frozen=True)
class Check:
    """One verifier's verdict: a state from its vocabulary, a reason, and the
    scope of what the verifier actually covered."""

    state: str
    reason: str = ""
    scope: str = ""


@dataclass(frozen=True)
class AcceptancePolicy:
    """What a data group requires of a target. Declared per group by the caller.

    ``require_answer`` -- the answer states that may be accepted; a primary
    reasoning group admits only ``correct``. ``admit_unverifiable`` is a
    separate, explicit switch so an unverifiable group can never pass by
    omission. ``require_steps`` -- step states a group demands (empty: none),
    e.g. a family whose acceptance is gated on its reference proof.
    ``reject_steps_contradicted`` drops a candidate whose step check found a
    contradiction even when its answer is right. ``reject_observations`` --
    observation name -> the states that reject (e.g. an illegal action).
    """

    require_answer: tuple[str, ...] = ("correct",)
    admit_unverifiable: bool = False
    require_steps: tuple[str, ...] = ()
    reject_steps_contradicted: bool = True
    reject_observations: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        bad = set(self.require_answer) - set(ANSWER_STATES)
        if bad:
            raise ValueError(f"unknown answer states {sorted(bad)}")
        if "unverifiable" in self.require_answer and not self.admit_unverifiable:
            raise ValueError("'unverifiable' requires admit_unverifiable=True")
        if set(self.require_steps) - set(STEP_STATES):
            raise ValueError(f"unknown step states {sorted(set(self.require_steps) - set(STEP_STATES))}")


@dataclass
class CandidateRecord:
    """Everything shown about one candidate, and the verdict."""

    unit_id: str
    candidate_index: int
    naturally_terminated: bool
    length_limited: bool
    degenerated: bool
    protocol_valid: bool
    answer: str
    answer_scope: str
    steps: str
    observations: dict[str, str] = field(default_factory=dict)
    generated_tokens: int | None = None
    reasons: list[str] = field(default_factory=list)
    accepted_for_training: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def assess(unit_id: str, candidate_index: int, *, stop_reason: str,
           protocol_reason: str | None, answer: Check, steps: Check | None = None,
           observations: Mapping[str, Check] | None = None,
           policy: AcceptancePolicy, end_reasons: Mapping[str, Sequence[str]],
           generated_tokens: int | None = None) -> CandidateRecord:
    """Classify one candidate and apply the group's policy.

    ``stop_reason`` is the generation record's own label; ``end_reasons`` maps
    the engine's labels to the three ending categories (keys ``natural``,
    ``length``, ``degeneration``), so this module carries no engine
    vocabulary. An unmapped stop reason is an error, not a silent pass.
    """
    if answer.state not in ANSWER_STATES:
        raise ValueError(f"unknown answer state {answer.state!r}")
    steps = steps or Check("not_assessed")
    if steps.state not in STEP_STATES:
        raise ValueError(f"unknown step state {steps.state!r}")
    observations = dict(observations or {})
    known = {r for v in end_reasons.values() for r in v}
    if stop_reason not in known:
        raise ValueError(f"stop reason {stop_reason!r} is not mapped in end_reasons")
    rec = CandidateRecord(
        unit_id=unit_id, candidate_index=int(candidate_index),
        naturally_terminated=stop_reason in end_reasons.get("natural", ()),
        length_limited=stop_reason in end_reasons.get("length", ()),
        degenerated=stop_reason in end_reasons.get("degeneration", ()),
        protocol_valid=protocol_reason is None,
        answer=answer.state, answer_scope=answer.scope, steps=steps.state,
        observations={k: v.state for k, v in sorted(observations.items())},
        generated_tokens=generated_tokens)
    if not rec.naturally_terminated:
        rec.reasons.append(f"ending:{stop_reason}")
    if protocol_reason is not None:
        rec.reasons.append(f"protocol:{protocol_reason}")
    if answer.state not in policy.require_answer:
        rec.reasons.append(f"answer:{answer.state}" + (f":{answer.reason}" if answer.reason else ""))
    if steps.state == "contradicted" and policy.reject_steps_contradicted:
        rec.reasons.append("steps:contradicted" + (f":{steps.reason}" if steps.reason else ""))
    elif policy.require_steps and steps.state not in policy.require_steps:
        rec.reasons.append(f"steps:{steps.state}" + (f":{steps.reason}" if steps.reason else ""))
    for name, rejecting in sorted(policy.reject_observations.items()):
        if name not in observations:
            rec.reasons.append(f"{name}:missing")
        elif observations[name].state in rejecting:
            o = observations[name]
            rec.reasons.append(f"{name}:{o.state}" + (f":{o.reason}" if o.reason else ""))
    rec.accepted_for_training = not rec.reasons
    return rec


def select_target(records: Sequence[CandidateRecord]) -> CandidateRecord | None:
    """The accepted candidate with the lowest index, or None. Never by length."""
    accepted = [r for r in records if r.accepted_for_training]
    return min(accepted, key=lambda r: r.candidate_index) if accepted else None


def summarize(records: Iterable[CandidateRecord],
              group_of: Callable[[CandidateRecord], str] | None = None) -> dict:
    """Candidate- and unit-level acceptance, with every category's count.

    A unit counts as accepted when at least one of its candidates is.
    ``group_of`` splits the summary (by data group, axis, generator, ...).
    """
    groups: dict[str, list[CandidateRecord]] = defaultdict(list)
    for r in records:
        groups[group_of(r) if group_of else "all"].append(r)
    out = {}
    for g, rows in sorted(groups.items()):
        units: dict[str, bool] = {}
        reasons: Counter = Counter()
        for r in rows:
            units[r.unit_id] = units.get(r.unit_id, False) or r.accepted_for_training
            reasons.update(r.reasons)
        n = len(rows)
        out[g] = {
            "candidates": n,
            "units": len(units),
            "naturally_terminated": sum(r.naturally_terminated for r in rows),
            "length_limited": sum(r.length_limited for r in rows),
            "degenerated": sum(r.degenerated for r in rows),
            "protocol_valid": sum(r.protocol_valid for r in rows),
            "answer": dict(Counter(r.answer for r in rows)),
            "answer_scope": dict(Counter(r.answer_scope for r in rows if r.answer_scope)),
            "steps": dict(Counter(r.steps for r in rows)),
            "observations": {name: dict(Counter(r.observations[name] for r in rows if name in r.observations))
                             for name in sorted({k for r in rows for k in r.observations})},
            "accepted_candidates": sum(r.accepted_for_training for r in rows),
            "accepted_units": sum(units.values()),
            "candidate_accept_rate": round(sum(r.accepted_for_training for r in rows) / n, 6) if n else None,
            "unit_accept_rate": round(sum(units.values()) / len(units), 6) if units else None,
            "rejection_reasons": dict(sorted(reasons.items())),
        }
    return out
