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
  task-specific verifier the caller supplies;
* ``trace`` -- the PROCESS, kept apart from the answer: ``verified`` (a
  task-specific check covered the whole process, e.g. every action of an
  episode legal on environment replay), ``partial`` (a check covered part of it,
  or a sampled audit found no contradiction), ``contradicted`` (the process
  conflicts with the task or with its own answer), or ``not_assessed``. A
  correct final answer does not establish that every step is right; where a
  task offers no checkable intermediate the trace is ``not_assessed`` and is
  reported so -- never ``verified``;
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

CONTRACT_ID = "aadistill.data.target_quality/v1"

ANSWER_STATES = ("correct", "incorrect", "unverifiable")
TRACE_STATES = ("verified", "partial", "contradicted", "not_assessed")


@dataclass(frozen=True)
class Check:
    """One verifier's verdict: a state from its vocabulary and a reason."""

    state: str
    reason: str = ""


@dataclass(frozen=True)
class AcceptancePolicy:
    """What a data group requires of a target. Declared per group by the caller.

    ``require_answer`` -- the answer states that may be accepted; a primary
    reasoning group admits only ``correct``. ``admit_unverifiable`` is a
    separate, explicit switch so an unverifiable group can never pass by
    omission. ``reject_trace_contradicted`` drops a candidate whose process
    check found a contradiction even when its answer is right.
    """

    require_answer: tuple[str, ...] = ("correct",)
    admit_unverifiable: bool = False
    reject_trace_contradicted: bool = True

    def __post_init__(self):
        bad = set(self.require_answer) - set(ANSWER_STATES)
        if bad:
            raise ValueError(f"unknown answer states {sorted(bad)}")
        if "unverifiable" in self.require_answer and not self.admit_unverifiable:
            raise ValueError("'unverifiable' requires admit_unverifiable=True")


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
    trace: str
    generated_tokens: int | None = None
    reasons: list[str] = field(default_factory=list)
    accepted_for_training: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def assess(unit_id: str, candidate_index: int, *, stop_reason: str,
           protocol_reason: str | None, answer: Check, trace: Check | None,
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
    trace = trace or Check("not_assessed")
    if trace.state not in TRACE_STATES:
        raise ValueError(f"unknown trace state {trace.state!r}")
    known = {r for v in end_reasons.values() for r in v}
    if stop_reason not in known:
        raise ValueError(f"stop reason {stop_reason!r} is not mapped in end_reasons")
    rec = CandidateRecord(
        unit_id=unit_id, candidate_index=int(candidate_index),
        naturally_terminated=stop_reason in end_reasons.get("natural", ()),
        length_limited=stop_reason in end_reasons.get("length", ()),
        degenerated=stop_reason in end_reasons.get("degeneration", ()),
        protocol_valid=protocol_reason is None,
        answer=answer.state, trace=trace.state, generated_tokens=generated_tokens)
    if not rec.naturally_terminated:
        rec.reasons.append(f"ending:{stop_reason}")
    if protocol_reason is not None:
        rec.reasons.append(f"protocol:{protocol_reason}")
    if answer.state not in policy.require_answer:
        rec.reasons.append(f"answer:{answer.state}" + (f":{answer.reason}" if answer.reason else ""))
    if trace.state == "contradicted" and policy.reject_trace_contradicted:
        rec.reasons.append("trace:contradicted" + (f":{trace.reason}" if trace.reason else ""))
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
            "trace": dict(Counter(r.trace for r in rows)),
            "accepted_candidates": sum(r.accepted_for_training for r in rows),
            "accepted_units": sum(units.values()),
            "candidate_accept_rate": round(sum(r.accepted_for_training for r in rows) / n, 6) if n else None,
            "unit_accept_rate": round(sum(units.values()) / len(units), 6) if units else None,
            "rejection_reasons": dict(sorted(reasons.items())),
        }
    return out
