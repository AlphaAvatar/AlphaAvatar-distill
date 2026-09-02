"""The six C1 probe results, and the decision inputs built from them.

The decision rule in `c1_isolation` reads paired per-prompt outcomes. Getting
from six scored probes to those vectors is where a two-arm design quietly goes
wrong, so it is a typed step here rather than a dict assembled at the call site.

The shape that was rejected is worth naming: a mapping whose keys are sometimes
`(arm, seed)` tuples and sometimes strings like `"usable_pooled_delta"`. It type-
checks, it reads fine, and one typo silently drops an arm from the pairing.

Three properties are enforced rather than assumed.

**The design is complete.** Six probes, two arms x three seeds, each carrying
exactly 950 rows of which exactly 850 are scorable, and the scorable prompt id
set identical across all six. A prompt missing from one arm is an unpaired
observation and the paired estimand has no meaning for it.

**Correctness and behaviour keep their own denominators.** `correct_overall` is
over the 850 scorable prompts; `usable_rollout_rate` is over all 950, because
behaviour is measurable where correctness has no oracle. They are never combined.

**The strata are the frozen six.** C0 fixes the primary inference as a stratified
prompt-cluster bootstrap over the six scorable sets; `code` is behaviour-only and
is not a stratum of the correctness estimand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..infrastructure.manifest import sha256_json

SCHEMA = "aadistill.autoinit.c1_probe_results/v1"

#: The frozen strata of the primary estimand (C0: `primary_inference.strata`).
PRIMARY_STRATA: tuple[str, ...] = (
    "gsm8k", "math_verified", "multihop", "rag", "knowledge", "tool")
N_PROMPTS = 950
N_SCORABLE = 850
ARMS = ("incumbent", "treatment")


class C1ResultsError(RuntimeError):
    """The six results do not form a complete paired design."""


@dataclass(frozen=True)
class C1ProbeRecord:
    """One (arm, seed) probe: what it was built from, and what it measured."""

    probe_id: str
    arm: str
    seed: int
    initialization_artifact_digest: str
    trained_run: Mapping[str, Any]
    result_path: str
    result_sha256: str
    per_sample_path: str
    per_sample_sha256: str
    generations: Mapping[str, Mapping[str, Any]]
    counts: Mapping[str, int]
    rates: Mapping[str, float | None]
    per_capability: Mapping[str, Mapping[str, Any]]
    scoring_contract: Mapping[str, Any]
    battery: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise C1ResultsError(f"{self.probe_id}: unknown arm {self.arm!r}")
        for key in ("n", "usable", "correct", "n_scorable", "usable_scorable"):
            if key not in self.counts:
                raise C1ResultsError(f"{self.probe_id}: missing count {key!r}")
        if self.counts["n"] != N_PROMPTS:
            raise C1ResultsError(
                f"{self.probe_id}: {self.counts['n']} rows, not {N_PROMPTS}")
        if self.counts["n_scorable"] != N_SCORABLE:
            raise C1ResultsError(
                f"{self.probe_id}: {self.counts['n_scorable']} scorable rows, not "
                f"{N_SCORABLE}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id, "arm": self.arm, "seed": self.seed,
            "initialization_artifact_digest": self.initialization_artifact_digest,
            "trained_run": dict(self.trained_run),
            "result_path": self.result_path, "result_sha256": self.result_sha256,
            "per_sample_path": self.per_sample_path,
            "per_sample_sha256": self.per_sample_sha256,
            "generations": {k: dict(v) for k, v in self.generations.items()},
            "counts": dict(self.counts), "rates": dict(self.rates),
            "per_capability": {k: dict(v) for k, v in self.per_capability.items()},
            "scoring_contract": dict(self.scoring_contract),
            "battery": dict(self.battery),
        }


@dataclass(frozen=True)
class C1DecisionInputs:
    """Exactly what `decide()` consumes. Counts and booleans, never rounded rates."""

    correct: Mapping[str, Mapping[int, Mapping[str, bool]]]
    strata: Mapping[str, str]
    usable_pooled_delta: float
    usable_per_seed_delta: tuple[float, ...]
    catastrophic_violations: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any] = field(default_factory=dict)

    def arm(self, role: str) -> dict[int, dict[str, bool]]:
        return {s: dict(v) for s, v in self.correct[role].items()}


def _rows_by_probe(per_sample: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
                   seeds: Sequence[int]) -> None:
    expected = {(a, s) for a in ARMS for s in seeds}
    got = set(per_sample)
    if got != expected:
        raise C1ResultsError(
            f"the design is 2 arms x {len(seeds)} seeds = {len(expected)} probes; "
            f"got {sorted(got)}")


def decision_inputs(
    per_sample: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]], *,
    seeds: Sequence[int],
    catastrophic_candidate_max: float = 0.10,
    catastrophic_control_min: float = 0.40,
    candidate_operand: str = "treatment",
    control_operand: str = "incumbent",
) -> C1DecisionInputs:
    """Align six probes into the paired vectors the frozen rule reads.

    Refuses a duplicate id, a missing id, a prompt set that differs between
    probes, a scorable count that is not 850 or a total that is not 950. Each of
    those would still produce a number.
    """
    _rows_by_probe(per_sample, seeds)

    indexed: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    strata: dict[str, str] = {}
    for key, rows in per_sample.items():
        by_id: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            pid = row["id"]
            if pid in by_id:
                raise C1ResultsError(f"{key}: duplicate prompt id {pid!r}")
            by_id[pid] = row
        if len(by_id) != N_PROMPTS:
            raise C1ResultsError(
                f"{key}: {len(by_id)} prompts, not {N_PROMPTS}")
        scorable = {pid for pid, r in by_id.items() if r["scorable"]}
        if len(scorable) != N_SCORABLE:
            raise C1ResultsError(
                f"{key}: {len(scorable)} scorable prompts, not {N_SCORABLE}")
        for pid in scorable:
            s = by_id[pid]["set"]
            if s not in PRIMARY_STRATA:
                raise C1ResultsError(
                    f"{key}: scorable prompt {pid!r} is in set {s!r}, which is not "
                    f"one of the frozen strata {list(PRIMARY_STRATA)}")
            if strata.setdefault(pid, s) != s:
                raise C1ResultsError(
                    f"prompt {pid!r} is in stratum {strata[pid]!r} in one probe "
                    f"and {s!r} in another")
        indexed[key] = by_id

    reference = {pid for pid, r in indexed[(ARMS[0], seeds[0])].items()
                 if r["scorable"]}
    for key, by_id in indexed.items():
        got = {pid for pid, r in by_id.items() if r["scorable"]}
        if got != reference:
            missing, extra = sorted(reference - got)[:3], sorted(got - reference)[:3]
            raise C1ResultsError(
                f"{key}: scorable prompt set differs from the reference probe "
                f"(missing e.g. {missing}, extra e.g. {extra}); the design is "
                "complete by construction and an incomplete one must not be "
                "silently averaged")

    correct = {arm: {s: {pid: bool(indexed[(arm, s)][pid]["correct"])
                         for pid in reference}
                     for s in seeds} for arm in ARMS}

    usable = {arm: {s: sum(1 for r in indexed[(arm, s)].values() if r["usable"])
                    for s in seeds} for arm in ARMS}
    per_seed = tuple((usable[candidate_operand][s] - usable[control_operand][s])
                     / N_PROMPTS for s in seeds)
    pooled = (sum(usable[candidate_operand].values())
              - sum(usable[control_operand].values())) / (N_PROMPTS * len(seeds))

    violations = []
    for capability in PRIMARY_STRATA:
        rate = {}
        for arm in ARMS:
            n = sum(1 for s in seeds for r in indexed[(arm, s)].values()
                    if r["set"] == capability)
            u = sum(1 for s in seeds for r in indexed[(arm, s)].values()
                    if r["set"] == capability and r["usable"])
            rate[arm] = u / n if n else None
        cand, ctl = rate[candidate_operand], rate[control_operand]
        if cand is not None and ctl is not None and (
                cand < catastrophic_candidate_max and ctl > catastrophic_control_min):
            violations.append({
                "capability": capability,
                "candidate_usable_rollout_rate": round(cand, 4),
                "control_usable_rollout_rate": round(ctl, 4),
                "reason": (f"{capability}: {candidate_operand} usable "
                           f"{cand:.4f} < {catastrophic_candidate_max} while "
                           f"{control_operand} usable {ctl:.4f} > "
                           f"{catastrophic_control_min}")})

    return C1DecisionInputs(
        correct=correct, strata=strata, usable_pooled_delta=pooled,
        usable_per_seed_delta=per_seed,
        catastrophic_violations=tuple(violations),
        audit={
            "n_prompts": N_PROMPTS, "n_scorable": N_SCORABLE,
            "seeds": list(seeds), "arms": list(ARMS),
            "strata_sizes": {s: sum(1 for v in strata.values() if v == s)
                             for s in PRIMARY_STRATA},
            "usable_counts": {a: dict(usable[a]) for a in ARMS},
            "usable_denominator": (
                f"{N_PROMPTS} prompts per probe; behaviour is measurable where "
                "correctness has no oracle"),
            "correct_denominator": (
                f"{N_SCORABLE} scorable prompts; code is behaviour-only"),
            "catastrophic_operands": {"candidate": candidate_operand,
                                      "control": control_operand},
        })


def build_probe_results(records: Sequence[C1ProbeRecord], *, plan_hash: str,
                        seeds: Sequence[int], inputs: C1DecisionInputs,
                        ) -> dict[str, Any]:
    """The `c1_probe_results.json` product: all six, bound to what produced them."""
    if len(records) != len(ARMS) * len(seeds):
        raise C1ResultsError(
            f"{len(records)} probe records, expected {len(ARMS) * len(seeds)}")
    keys = {(r.arm, r.seed) for r in records}
    if keys != {(a, s) for a in ARMS for s in seeds}:
        raise C1ResultsError(f"probe records do not cover the design: {sorted(keys)}")

    contracts = {r.scoring_contract.get("digest") for r in records}
    batteries = {r.battery.get("content_sha256") for r in records}
    if len(contracts) != 1 or len(batteries) != 1:
        raise C1ResultsError(
            "the six probes were not all scored under one contract and one "
            f"battery: contracts={contracts}, batteries={batteries}")

    doc = {
        "schema": SCHEMA,
        "plan_hash": plan_hash,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "n_probes": len(records),
        "scoring_contract": dict(records[0].scoring_contract),
        "battery": dict(records[0].battery),
        "probes": [r.as_dict() for r in sorted(records, key=lambda r: (r.arm, r.seed))],
        "decision_inputs_audit": dict(inputs.audit),
        "usable_pooled_delta": inputs.usable_pooled_delta,
        "usable_per_seed_delta": list(inputs.usable_per_seed_delta),
        "catastrophic_violations": [dict(v) for v in inputs.catastrophic_violations],
        "reads": ("counts and per-sample booleans; the decision layer never "
                  "consumes a rounded rate as primary data"),
    }
    doc["results_sha256"] = sha256_json(doc)
    return doc
