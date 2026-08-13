"""Recovery Top-N orchestration — configuration and gates only.

**No paid recovery runs from this module.** It defines the successive-halving
schedule, the admission gate and the preregistration record; executing the probes
is a separate, separately authorized step. That split is the point: the schedule
and its thresholds have to be frozen *before* the run that they judge, and the
only way to make that checkable is to have an artifact that exists first.

The schedule:

    N searched leaves + the retained canonical control
      ->  rung 1: identical 0.86M recovery on seed sa, all of them
      ->  search-battery evaluation
      ->  rung 2: the control (unconditionally) + the best S searched leaves, seed sb
      ->  Top-1 from the two-seed result
      ->  optional seed sc, for tied candidates only
      ->  full recovery (separately authorized)

Rules the rest of the project already paid to learn:

* **Selection is on autonomous behaviour, not state NLL** (E7: a −5.22 nat NLL
  swing moved behaviour by +0.0000; the best-NLL checkpoint of its trajectory
  produced zero protocol-valid generations).
* **Selection is a constraint followed by an objective, never a weighted sum.**
  ``usable_rollout`` is a *feasibility* gate — it is blind to correctness by
  construction, so a terse contentless reply scores perfectly on it and a
  weighted ``usable + correct`` score would let stability buy its way past a
  capability failure. Feasible candidates are then ranked on ``correct_overall``,
  with ``correct_given_usable`` reported as a diagnostic that explains *why* a
  candidate ranks where it does without moving it.
* **Two seeds minimum, because one is unreadable.** The behaviour metric moves
  0.1290 on seed alone.
* **The control advances to rung 2 regardless of its rung-1 result.** A baseline
  eliminated at rung 1 gives a two-seed comparison with nothing to compare
  against, and its variance is exactly what makes the searched candidates'
  differences readable.
* **Thresholds are preregistered, never chosen after seeing the table**
  (AGENTS.md 4.5). ``SuccessiveHalvingPlan.freeze`` writes the record and
  ``assert_preregistered`` refuses a selection whose plan hash does not match.

``N`` is deliberately unset here. The cost model informs it; the decision is the
maintainer's.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from ..infrastructure.manifest import sha256_file, sha256_json
from .state import InitializationState

SCHEMA = "aadistill.autoinit.recovery_plan/v1"


class RecoveryAdmissionError(RuntimeError):
    """Something that is not a complete target-size leaf was offered to a probe."""


@dataclass(frozen=True)
class RecoveryRecipe:
    """The frozen KD-heavy recovery, reused so initialization stays the variable."""

    recipe_id: str
    ce_weight: float
    kd_weight: float
    temperature: float
    kd_scope: str
    tokens: int
    pack: str
    pack_sha256: str | None = None
    block_len: int = 8192
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id, "ce_weight": self.ce_weight,
            "kd_weight": self.kd_weight, "temperature": self.temperature,
            "kd_scope": self.kd_scope, "tokens": self.tokens, "pack": self.pack,
            "pack_sha256": self.pack_sha256, "block_len": self.block_len,
            "description": self.description,
        }


#: E1/P1 at the 0.86M probe rung. Frozen: AutoInitializer v1 changes the
#: initialization, not the recovery objective (handoff 5.1 item 18).
E1_KD_HEAVY_0860K = RecoveryRecipe(
    recipe_id="e1_p1_kd_heavy@0.86M",
    ce_weight=0.25, kd_weight=1.0, temperature=1.0, kd_scope="all",
    tokens=860_000, pack="artifacts/stage3/ladder_uniform_probe",
    pack_sha256="6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c",
    description=("The recovery recipe every E1 arm used, at the rung the frozen "
                 "battery was sampled from. Held fixed so a difference between two "
                 "probes is a difference between two initializations."),
)

@dataclass(frozen=True)
class SeedAggregation:
    """How per-seed probe results become one number. Executable, versioned, hashed.

    **Pooled counts, not averaged rates.** For a conditional rate the two are not
    the same thing: ``correct_given_usable`` averaged over seeds weights a seed
    with 30 usable rollouts equally with one that had 120, so a probe that
    happened to terminate rarely gets its conditional accuracy amplified. Pooling
    the counts is the estimator that answers "of the rollouts this checkpoint
    actually completed, what fraction were right?"

        correct_overall      = (correct_sa + correct_sb) / (n_sa + n_sb)
        usable_rollout_rate  = (usable_sa + usable_sb) / (n_sa + n_sb)
        correct_given_usable = (correct_sa + correct_sb) / (usable_sa + usable_sb)

    The same rule extends to a third seed: pool across **all completed seeds**
    for the tied finalists, never a rate of rates.

    This definition participates in the plan hash, so a run cannot silently
    change how its seeds were combined.
    """

    aggregation_id: str = "pooled_counts"
    version: int = 1
    description: str = ("pooled numerator/denominator across all completed seeds; "
                        "conditional rates are pooled, never averaged")

    def as_dict(self) -> dict[str, Any]:
        return {"aggregation_id": self.aggregation_id, "version": self.version,
                "description": self.description,
                "formulas": {
                    "correct_overall": "sum(correct_s) / sum(n_s)",
                    "usable_rollout_rate": "sum(usable_s) / sum(n_s)",
                    "correct_given_usable": "sum(correct_s) / sum(usable_s)",
                }}

    def pool(self, per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Combine one candidate's per-seed counts.

        Each entry needs ``n``, ``usable`` and ``correct`` counts, and its
        ``seed``. Rates are *derived* here and never read from the input, so a
        caller cannot pass a pre-averaged rate and have it silently accepted.
        """
        if not per_seed:
            raise ValueError("no seed results to pool")
        seeds = [int(r["seed"]) for r in per_seed]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds in {seeds}; each seed counts once")

        def count(row, key) -> int:
            value = row[key]
            # Refuse a float outright rather than truncating it. `int(0.8) == 0`
            # would silently turn a caller's rate into a count of zero, which is
            # the exact confusion this whole definition exists to prevent.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"seed {row.get('seed')}: {key}={value!r} is not an integer "
                    "count. These are counts, not rates — pass the numerator and "
                    "denominator, not a per-seed rate.")
            if value < 0:
                raise ValueError(f"seed {row.get('seed')}: {key} is negative")
            return value

        n = sum(count(r, "n") for r in per_seed)
        usable = sum(count(r, "usable") for r in per_seed)
        correct = sum(count(r, "correct") for r in per_seed)
        if n <= 0:
            raise ValueError("pooled n is zero")
        if usable > n or correct > n:
            raise ValueError(
                f"pooled usable={usable} / correct={correct} exceed n={n}; these are "
                "counts, not rates")
        if correct > usable:
            raise ValueError(
                f"pooled correct={correct} exceeds usable={usable}; a rollout that is "
                "not usable cannot have been scored correct")
        return {
            "seeds": sorted(seeds),
            "n": n, "usable": usable, "correct": correct,
            "usable_rollout_rate": usable / n,
            "correct_overall": correct / n,
            # Undefined rather than 0.0 when nothing was usable: a checkpoint that
            # never produced a valid rollout has no conditional accuracy, and
            # reporting 0.0 would make it look measured.
            "correct_given_usable": (correct / usable) if usable else None,
            "aggregation": f"{self.aggregation_id}@v{self.version}",
        }


POOLED_COUNTS_V1 = SeedAggregation()


@dataclass(frozen=True)
class EquivalenceRule:
    """The behaviour equivalence interval. **One definition, and it is seed-aware.**

    Prompt-level binomial uncertainty alone understates what this project has
    measured. The behaviour metric moves **0.1290 on training seed alone**, and
    340 prompts drawn from a *single* recovered checkpoint say nothing about that:
    they estimate one checkpoint's rate precisely and the *recipe's* rate not at
    all. An interval built only from the binomial term would call two
    initializations different when a reseed of either would have crossed the gap.

    So the frozen rule takes the larger of the two uncertainties:

        binomial_se    = sqrt(p_pool * (1 - p_pool) / n_pooled)
        seed_se_proxy  = |p_sa - p_sb| / 2
        interval       = z * max(binomial_se, seed_se_proxy)

    ``seed_se_proxy`` is a two-point range, which is a weak estimator of a
    standard error — with n=2 it is the only one available, and it is used as a
    *floor* on the interval rather than as an estimate in its own right. That is
    the conservative direction: it can only widen the interval, so it can only
    make the selector more willing to call a difference a tie.

    The formula is frozen before any candidate is searched; the values are
    materialized from the **control's own** per-seed rates and then never change.
    Until then ``value`` is ``None`` and every selector that needs it raises — a
    fallback to a prior would reintroduce the second definition this replaces.
    """

    n_pooled: int
    rule_id: str = "seed_aware_max_binomial_seedrange"
    version: int = 2
    z: float = 2.0
    p_control: float | None = None
    p_sa: float | None = None
    p_sb: float | None = None
    binomial_se: float | None = None
    seed_se_proxy: float | None = None
    value: float | None = None

    def __post_init__(self) -> None:
        if self.n_pooled <= 0:
            raise ValueError("n_pooled must be positive")
        if (self.value is None) != (self.p_control is None):
            raise ValueError(
                "value and p_control are materialized together or not at all")

    def components(self, p_pool: float, p_sa: float, p_sb: float) -> dict[str, float]:
        for name, value in (("p_pool", p_pool), ("p_sa", p_sa), ("p_sb", p_sb)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value!r} is not a rate")
        binomial = math.sqrt(max(p_pool * (1 - p_pool), 0.0) / self.n_pooled)
        seed = abs(p_sa - p_sb) / 2.0
        return {"binomial_se": binomial, "seed_se_proxy": seed,
                "value": self.z * max(binomial, seed)}

    def materialize(self, *, p_pool: float, p_sa: float, p_sb: float) -> "EquivalenceRule":
        """Freeze the numeric value from the control's pooled and per-seed rates."""
        if self.value is not None:
            raise ValueError(
                f"already materialized at {self.value}; the interval is frozen once "
                "the control is characterized and does not move afterwards")
        parts = self.components(p_pool, p_sa, p_sb)
        return EquivalenceRule(
            n_pooled=self.n_pooled, rule_id=self.rule_id, version=self.version,
            z=self.z, p_control=p_pool, p_sa=p_sa, p_sb=p_sb,
            binomial_se=parts["binomial_se"], seed_se_proxy=parts["seed_se_proxy"],
            value=parts["value"])

    def require_value(self) -> float:
        if self.value is None:
            raise RecoveryAdmissionError(
                "the equivalence interval is not materialized: the canonical "
                "control has not been characterized on the recovery-search "
                "battery. Measure both seeds, materialize the rule, re-hash the "
                "plan, then select. There is no prior to fall back to by design.")
        return self.value

    def as_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "version": self.version,
                "formula": ("z * max(sqrt(p_pool*(1-p_pool)/n_pooled), "
                            "|p_sa - p_sb| / 2)"),
                "z": self.z, "n_pooled": self.n_pooled,
                "p_control_pooled": self.p_control, "p_sa": self.p_sa,
                "p_sb": self.p_sb, "binomial_se": self.binomial_se,
                "seed_se_proxy": self.seed_se_proxy, "value": self.value,
                "dominant_term": (None if self.value is None else
                                  ("seed_range" if (self.seed_se_proxy or 0)
                                   > (self.binomial_se or 0) else "binomial")),
                "status": "materialized" if self.value is not None
                          else "PENDING_CONTROL_CHARACTERIZATION",
                "seed_awareness": ("the two-point seed range is a floor on the "
                                   "interval, not an estimate; it can only widen "
                                   "it, which is the conservative direction"),
                "non_adaptive": ("the formula is frozen before any candidate is "
                                 "searched; only the control's own rates enter it")}


@dataclass(frozen=True)
class FeasibilityRule:
    """The usable-rollout floor. Seed-aware for the same reason.

        binomial_se = sqrt(u_pool * (1 - u_pool) / n_pooled)
        seed_proxy  = |u_sa - u_sb| / 2
        floor       = max(absolute_floor, u_pool - k * max(binomial_se, seed_proxy))

    The absolute floor guards "cannot hold a rollout at all" independently of how
    the control happens to score. The relative term guards against a candidate
    much less stable than the incumbent without demanding parity, which would turn
    feasibility into a second ranking.
    """

    n_pooled: int
    rule_id: str = "seed_aware_usable_floor"
    version: int = 2
    k: float = 3.0
    absolute_floor: float = 0.30
    u_pool: float | None = None
    u_sa: float | None = None
    u_sb: float | None = None
    value: float | None = None

    def components(self, u_pool: float, u_sa: float, u_sb: float) -> dict[str, float]:
        binomial = math.sqrt(max(u_pool * (1 - u_pool), 0.0) / self.n_pooled)
        seed = abs(u_sa - u_sb) / 2.0
        uncertainty = max(binomial, seed)
        return {"binomial_se": binomial, "seed_se_proxy": seed,
                "uncertainty": uncertainty,
                "value": max(self.absolute_floor, u_pool - self.k * uncertainty)}

    def materialize(self, *, u_pool: float, u_sa: float, u_sb: float) -> "FeasibilityRule":
        if self.value is not None:
            raise ValueError(f"already materialized at {self.value}")
        parts = self.components(u_pool, u_sa, u_sb)
        return FeasibilityRule(
            n_pooled=self.n_pooled, rule_id=self.rule_id, version=self.version,
            k=self.k, absolute_floor=self.absolute_floor, u_pool=u_pool,
            u_sa=u_sa, u_sb=u_sb, value=parts["value"])

    def require_value(self) -> float:
        if self.value is None:
            raise RecoveryAdmissionError(
                "the feasibility floor is not materialized: the canonical control "
                "has not been characterized on both seeds. There is no prior to "
                "fall back to by design.")
        return self.value

    def as_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "version": self.version, "k": self.k,
                "n_pooled": self.n_pooled, "absolute_floor": self.absolute_floor,
                "formula": ("max(absolute_floor, u_pool - k * max(binomial_se, "
                            "|u_sa - u_sb| / 2))"),
                "u_pool": self.u_pool, "u_sa": self.u_sa, "u_sb": self.u_sb,
                "value": self.value,
                "status": "materialized" if self.value is not None
                          else "PENDING_CONTROL_CHARACTERIZATION"}


@dataclass(frozen=True)
class CatastrophicCapabilityRule:
    """Excludes a candidate that collapses on one capability.

    The pooled feasibility rate can hide a total collapse: a candidate that never
    produces a usable tool rollout can still clear a global floor on the strength
    of its maths. Enforced mechanically at **both** rungs, with the capability
    name and both measured values in the exclusion reason — a rule that needs a
    human to notice it is not a rule.
    """

    candidate_max: float = 0.10
    control_min: float = 0.40
    metric: str = "usable_rollout_rate"
    rule_id: str = "per_capability_collapse"
    version: int = 1

    def violations(self, candidate: Mapping[str, Any],
                   control: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Capabilities on which this candidate has collapsed relative to control."""
        cand = dict(candidate.get("per_capability") or {})
        ctrl = dict(control.get("per_capability") or {}) if control else {}
        out = []
        for capability in sorted(set(cand) & set(ctrl)):
            c_val = float(cand[capability].get(self.metric, 1.0))
            k_val = float(ctrl[capability].get(self.metric, 0.0))
            if c_val < self.candidate_max and k_val > self.control_min:
                out.append({
                    "capability": capability, "metric": self.metric,
                    "candidate_value": c_val, "control_value": k_val,
                    "candidate_max": self.candidate_max,
                    "control_min": self.control_min,
                    "reason": (f"catastrophic collapse on {capability}: candidate "
                               f"{self.metric}={c_val:.4f} < {self.candidate_max} "
                               f"while the control has {k_val:.4f} > "
                               f"{self.control_min}"),
                })
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "version": self.version,
                "metric": self.metric, "candidate_max": self.candidate_max,
                "control_min": self.control_min,
                "statement": (f"candidate {self.metric} < {self.candidate_max} AND "
                              f"control {self.metric} > {self.control_min} "
                              "-> catastrophic failure, candidate excluded"),
                "enforced_at": ["rung1", "final"]}


CATASTROPHIC_V1 = CatastrophicCapabilityRule()


class CapabilitySchemaError(RecoveryAdmissionError):
    """A recovery result does not carry the capability breakdown it must."""


@dataclass(frozen=True)
class CapabilitySchema:
    """The capability breakdown every recovery result must carry. Fail closed.

    The catastrophic rule can only exclude a candidate on a capability it can
    *see*. If a scoring bug drops the `tool` breakdown, a naive rule silently
    passes every candidate on tool — a data defect converted into a clean bill of
    health. Defaults are worse still: "missing candidate usable -> 1.0" makes a
    broken pipeline look like a perfect candidate, and "missing control usable ->
    0.0" disables the rule entirely.

    So the expected capability set is part of the frozen policy, and a result that
    does not match it exactly raises. Missing capabilities, extra capabilities,
    absent metrics, NaN, Inf and out-of-range rates are all errors, never
    defaults.
    """

    expected: tuple[str, ...]
    required_metrics: tuple[str, ...] = ("usable_rollout_rate",)
    required_counts: tuple[str, ...] = ("n", "usable")
    version: int = 1

    def validate(self, result: Mapping[str, Any], *, label: str = "") -> None:
        who = label or result.get("state_id") or "<unnamed>"
        breakdown = result.get("per_capability")
        if breakdown is None:
            raise CapabilitySchemaError(
                f"{who}: no per_capability breakdown. The catastrophic rule cannot "
                "see what is not reported, and a missing breakdown must not read "
                "as a pass.")
        actual = set(breakdown)
        expected = set(self.expected)
        if actual != expected:
            raise CapabilitySchemaError(
                f"{who}: capability set {sorted(actual)} != expected "
                f"{sorted(expected)}; missing {sorted(expected - actual)}, "
                f"unexpected {sorted(actual - expected)}")
        for capability in sorted(expected):
            entry = breakdown[capability]
            if not isinstance(entry, Mapping):
                raise CapabilitySchemaError(
                    f"{who}/{capability}: breakdown entry is not a mapping")
            for metric in self.required_metrics:
                if metric not in entry:
                    raise CapabilitySchemaError(
                        f"{who}/{capability}: missing {metric!r}")
                value = entry[metric]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise CapabilitySchemaError(
                        f"{who}/{capability}: {metric}={value!r} is not numeric")
                if not math.isfinite(float(value)):
                    raise CapabilitySchemaError(
                        f"{who}/{capability}: {metric} is {value!r}; NaN and Inf are "
                        "scoring failures, not rates")
                if not 0.0 <= float(value) <= 1.0:
                    raise CapabilitySchemaError(
                        f"{who}/{capability}: {metric}={value} is outside [0, 1]")
            for count in self.required_counts:
                if count not in entry:
                    raise CapabilitySchemaError(
                        f"{who}/{capability}: missing count {count!r}, which pooled "
                        "aggregation needs")
                value = entry[count]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise CapabilitySchemaError(
                        f"{who}/{capability}: {count}={value!r} is not a "
                        "non-negative integer count")
            if {"n", "usable"} <= set(self.required_counts):
                if entry["usable"] > entry["n"]:
                    raise CapabilitySchemaError(
                        f"{who}/{capability}: usable={entry['usable']} exceeds "
                        f"n={entry['n']}")

    def validate_all(self, results: Sequence[Mapping[str, Any]]) -> None:
        for result in results:
            self.validate(result)

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "expected": list(self.expected),
                "required_metrics": list(self.required_metrics),
                "required_counts": list(self.required_counts),
                "policy": ("exact set match; missing/extra capabilities, absent "
                           "metrics, non-numeric values, NaN/Inf and out-of-range "
                           "rates all raise. No defaults.")}


#: The scorable capabilities of recovery_search_v1. `code` is behaviour-only and
#: therefore not a capability the catastrophic rule can rank on.
CAPABILITY_SCHEMA_V1 = CapabilitySchema(
    expected=("gsm8k", "math_verified", "multihop", "rag", "knowledge", "tool"))


#: Every implementation that materially defines the frozen recovery-search
#: metrics. One aggregate digest, not a single historical scorer-file hash.
#:
#: The v1 contract recorded `capability.py`'s hash alone, which is why a defect
#: living in the *composition* — `protocol_valid` rejecting a prompted tool call,
#: `usable_rollout` inheriting that, `correct => usable` propagating it — was
#: invisible to the identity that was supposed to pin the metric. A scoring
#: contract must cover everything that can move a number, including the rule that
#: relates two numbers to each other.
RECOVERY_SCORING_FILES_V2: tuple[str, ...] = (
    "scripts/autoinit/score_recovery_search.py",  # the recovery-search contract
    "src/aadistill/evaluation/usable_rollout.py",  # the five behaviour components
    "src/aadistill/evaluation/strict_answer.py",   # protocol validity, extraction
    "src/aadistill/evaluation/behavior.py",        # split, tool-call diagnostics
    "src/aadistill/evaluation/capability.py",      # per-set correctness scorers
    "src/aadistill/autoinit/recovery.py",          # correct=>usable, capability
                                                   # schema, pooled aggregation,
                                                   # thresholds
)
RECOVERY_SCORING_CONTRACT_ID = "recovery_search_scoring"
RECOVERY_SCORING_CONTRACT_VERSION = 2


def recovery_scoring_contract(repo_root: str | Path = ".", *,
                              files: Sequence[str] | None = None,
                              ) -> dict[str, Any]:
    """The aggregate scoring-contract digest a result binds to.

    Same shape and same failure mode as ``trainer_source_digest``: a missing
    declared file raises rather than yielding a digest over a smaller contract.
    """
    root = Path(repo_root)
    declared = tuple(files) if files is not None else RECOVERY_SCORING_FILES_V2
    entries = []
    for rel in sorted(declared):
        path = root / rel
        if not path.is_file():
            raise RecoveryAdmissionError(
                f"declared recovery-scoring source {rel!r} is missing; refusing "
                "to produce a contract digest that silently describes a smaller "
                "scorer than the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {
        "contract": f"{RECOVERY_SCORING_CONTRACT_ID}@v{RECOVERY_SCORING_CONTRACT_VERSION}",
        "contract_id": RECOVERY_SCORING_CONTRACT_ID,
        "version": RECOVERY_SCORING_CONTRACT_VERSION,
        "digest": digest,
        "files": entries,
        "rule": ("sha256 over sorted 'path:sha256' lines of the declared "
                 "recovery-scoring source set"),
        "covers": ["usable_rollout components and their composition",
                   "protocol validity, including the tools_offered relaxation",
                   "the tool structural-executability gate",
                   "per-set correctness scorers",
                   "correct => usable",
                   "capability schema and aggregation",
                   "pooled_counts seed aggregation",
                   "the threshold formulas materialized from control data"],
        "supersedes": {
            "contract": f"{RECOVERY_SCORING_CONTRACT_ID}@v1",
            "identity_was": ("a single hash of "
                             "src/aadistill/evaluation/capability.py"),
            "why": ("v1 could not see a defect in the composition. A prompted "
                    "tool call was rejected as unexpected_tool_call, so the tool "
                    "capability read a structural 0.0000 for every arm while "
                    "capability.py itself was unchanged and its hash matched."),
        },
    }


#: Source files whose contents can change what a recovery run produces.
#:
#: Explicit and versioned, because whole-repository ``git HEAD`` is the wrong
#: material identity: a docs-only commit would invalidate a control checkpoint
#: whose trainer never moved. Derived from what the recovery entry point actually
#: imports — the training loop and its loss/KD, the deterministic block ordering,
#: the packing helpers, model construction, and the LoRA/freeze policy the
#: trainable-parameter selection goes through — not from a guess.
TRAINER_SOURCE_FILES_V1: tuple[str, ...] = (
    "scripts/training/train_stage3.py",      # entry point, config -> run semantics
    "src/aadistill/training/train.py",       # loop, loss/KD, optimizer, resume
    "src/aadistill/training/lora.py",        # trainable-parameter selection policy
    "src/aadistill/data/ladder.py",          # deterministic block ordering
    "src/aadistill/data/dataset.py",         # packing, masks, encoding
    "src/aadistill/models/teacher.py",       # teacher load + dtype/attn selection
    "src/aadistill/models/student.py",       # student load + RoPE guard
)
TRAINER_SOURCE_SET_VERSION = 1


def trainer_source_digest(repo_root: str | Path = ".",
                          files: Sequence[str] = TRAINER_SOURCE_FILES_V1) -> dict[str, Any]:
    """A material identity for the trainer, independent of unrelated commits.

    Hashes the declared source set and nothing else. A documentation or STATE
    commit therefore leaves this unchanged and a control checkpoint stays matched;
    a change to the loss, the loop, the ordering or the optimizer construction
    changes it and the control stops being matched, which is the intent.

    The whole-repository commit is recorded separately as provenance
    (``repo_git_commit`` / ``repo_dirty``) and is never the equality predicate.
    """
    root = Path(repo_root)
    entries = []
    for rel in files:
        path = root / rel
        if not path.is_file():
            raise RecoveryAdmissionError(
                f"declared trainer source {rel} is missing; the digest would "
                "silently describe a smaller trainer than the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "set_version": TRAINER_SOURCE_SET_VERSION,
            "files": entries,
            "rule": ("sha256 over sorted 'path:sha256' lines of the declared "
                     "trainer source set; whole-repo commit is provenance only")}


@dataclass(frozen=True)
class RuntimeEnvironmentFingerprint:
    """The runtime a recovery run executes under.

    ``torch_version`` alone is not enough. The permanent controls generated during
    the preflight and the searched-leaf probes that are later compared against them
    must execute under the *same* frozen runtime, or the comparison carries a
    runtime difference alongside the initialization difference — which is exactly
    the confound that disqualified the historical controls.

    ``image_digest`` is the strongest field and the one that actually pins the
    rest; it is required for a run to be recorded as protocol-matched. The
    individual versions are recorded too so a mismatch says *what* moved.
    """

    image_digest: str | None
    python_version: str
    torch_version: str
    transformers_version: str
    cuda_runtime: str | None
    attention_backend: str
    version: int = 1

    @classmethod
    def observe(cls, *, image_digest: str | None = None,
                attention_backend: str = "sdpa") -> "RuntimeEnvironmentFingerprint":
        import platform

        import torch
        import transformers

        return cls(
            image_digest=image_digest,
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            cuda_runtime=getattr(torch.version, "cuda", None),
            attention_backend=attention_backend,
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "image_digest": self.image_digest,
                "python_version": self.python_version,
                "torch_version": self.torch_version,
                "transformers_version": self.transformers_version,
                "cuda_runtime": self.cuda_runtime,
                "attention_backend": self.attention_backend}

    def require_pinned(self) -> None:
        if not self.image_digest:
            raise RecoveryAdmissionError(
                "the runtime fingerprint has no image digest. The permanent "
                "controls and the later searched probes must execute under the "
                "same frozen image; without a digest that cannot be asserted.")


@dataclass(frozen=True)
class RecoveryProtocolFingerprint:
    """Everything that must be **identical** across control and searched leaves.

    Deliberately excludes the student initialization artifact and the seed. Those
    are the variables:

        control-sa     protocol = X, seed = sa, init = canonical
        searched-A-sa  protocol = X, seed = sa, init = searched-A

    which is the single-variable comparison the whole experiment is about. Putting
    the initialization inside the protocol identity would make every pair of arms
    "mismatched" by construction and make the equality predicate useless for its
    actual job.
    """

    # data
    pack: str
    pack_blocks_sha256: str | None
    rung: int
    train_blocks: int | None
    train_supervised_tokens: int | None
    block_len: int
    packing: str
    val_blocks: int
    block_ordering: str
    # objective
    ce_weight: float
    kd_weight: float
    kd_temperature: float
    kd_scope: str
    kd_chunk: int
    # optimizer
    optimizer: str
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    grad_clip: float
    # schedule
    total_steps: int
    warmup_steps: int
    min_lr_frac: float
    lr_schedule: str
    # batch semantics
    blocks_per_step: int
    micro_blocks: int
    # numerics
    dtype: str
    autocast_bf16: bool
    gradient_checkpointing: bool
    # trainable-parameter policy
    trainable_patterns: tuple[str, ...]
    trainable_params: int | None
    # teacher + tokenizer
    teacher_id: str
    teacher_revision: str
    teacher_dtype: str
    teacher_attn: str | None
    tokenizer_sha256: str | None
    # execution
    resume_semantics: str
    trainer_source_digest: str | None
    trainer_source_set_version: int | None
    runtime_digest: str | None

    #: Fields that could not be established. Compared as unverifiable, never as
    #: matched.
    unverifiable: tuple[str, ...] = ()

    #: Fields that must carry a real value before this protocol can take part in a
    #: MATCHED comparison. ``None`` on both sides means *unknown on both sides*,
    #: which is not the same statement as *verified identical* — and comparing two
    #: ``None``s with ``==`` silently turns the first into the second.
    MATERIALIZATION_REQUIRED: ClassVar[tuple[str, ...]] = (
        "trainer_source_digest", "trainer_source_set_version", "runtime_digest")

    def unmaterialized_fields(self) -> tuple[str, ...]:
        return tuple(f for f in self.MATERIALIZATION_REQUIRED
                     if getattr(self, f) is None)

    @property
    def is_materialized(self) -> bool:
        return not self.unmaterialized_fields()

    def require_materialized(self, *, context: str = "") -> None:
        """Raise unless every identity-bearing field carries a real value.

        Gate for *future* runs. A historical audit may still report fields as
        unverifiable and is unaffected: "never recorded" is a permanent property
        of a past run, while "not yet attested" is a stage the preflight passes
        through and must not be trained under.
        """
        missing = self.unmaterialized_fields()
        if missing:
            where = f" ({context})" if context else ""
            raise RecoveryAdmissionError(
                f"protocol is not materialized{where}: {', '.join(missing)} "
                "is unknown. Unknown on both sides is not verified identical; "
                "attest the runtime and trainer source at preflight Stage 0 "
                "before declaring any protocol matched.")

    def materialized(self, *, runtime: "RuntimeEnvironmentFingerprint",
                     trainer_source: Mapping[str, Any],
                     ) -> "RecoveryProtocolFingerprint":
        """Fill the environment fields from a Stage-0 attestation.

        Only fills what was unknown. If a field is already set and the
        attestation disagrees, that is protocol drift between preregistration and
        execution, and it raises rather than being overwritten.
        """
        runtime.require_pinned()
        proposed = {"trainer_source_digest": trainer_source["digest"],
                    "trainer_source_set_version": trainer_source["set_version"],
                    "runtime_digest": runtime.digest}
        for key, value in proposed.items():
            current = getattr(self, key)
            if current is not None and current != value:
                raise RecoveryAdmissionError(
                    f"attested {key} ({value!r}) contradicts the preregistered "
                    f"value ({current!r}). This is protocol drift between what "
                    "was preregistered and what is about to execute; resolve it "
                    "rather than overwriting the preregistration.")
        return replace(self, **proposed)

    @classmethod
    def from_run_artifacts(cls, run_dir: str | Path, *,
                           repo_root: str | Path | None = None,
                           pack_root: str | Path | None = None,
                           strict: bool = True) -> "RecoveryProtocolFingerprint":
        """Reconstruct the protocol a run **actually executed**, from its own files.

        The counterpart of ``phase_a_protocol()`` and the opposite of
        ``compare_recovery_fingerprints.historical_protocol()``. That helper is a
        forensic tool for runs whose evidence no longer exists, and it is
        deliberately permissive: it defaults ``kd_chunk`` to 512, hard-codes
        ``optimizer="AdamW"`` and the schedule and ordering literals, and fills
        ``pack_blocks_sha256`` from the *expected* frozen constant. Every one of
        those is a value the verifier supplies to itself, which is precisely the
        backfill that turns a paid control's verification into a tautology.

        Under ``strict=True`` nothing is defaulted, nothing is inherited from a
        preregistration, and the pack hash is **recomputed from the pack the run
        read**. A material field with no evidence raises
        :class:`ObservedProtocolError`; a run that cannot prove what it did is
        not a control.

        ``strict=False`` returns the same object with the unestablished fields
        listed in ``unverifiable``, where ``compare`` reports them as unknown —
        never as matched.

        The parameters an earlier sketch took — ``runtime=`` and
        ``trainer_source=`` — are deliberately **absent**. Passing them in would
        mean the caller supplying the two fields the comparison most needs to
        establish; they are read from the run's own execution record instead.
        """
        return observe_recovery_protocol(
            run_dir, repo_root=repo_root, pack_root=pack_root,
            strict=strict).protocol

    def identity(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "unverifiable"}
        # None-safe: a non-strict reconstruction of a run that established
        # neither field must still be describable and comparable — as unknown.
        d["betas"] = list(self.betas) if self.betas is not None else None
        d["trainable_patterns"] = (list(self.trainable_patterns)
                                   if self.trainable_patterns is not None else None)
        return d

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.identity())

    def compare(self, other: "RecoveryProtocolFingerprint") -> dict[str, Any]:
        mine, theirs = self.identity(), other.identity()
        # An unmaterialized required field is unknown, on whichever side it is
        # unknown — so `None == None` can never be reported as a matched field.
        unmaterialized = sorted(set(self.unmaterialized_fields())
                                | set(other.unmaterialized_fields()))
        unverifiable = (set(self.unverifiable) | set(other.unverifiable)
                        | set(unmaterialized))
        matched, mismatched, unknown = [], [], []
        for key in sorted(set(mine) | set(theirs)):
            if key in unverifiable:
                unknown.append({"field": key, "self": mine.get(key),
                                "other": theirs.get(key)})
            elif mine.get(key) == theirs.get(key):
                matched.append(key)
            else:
                mismatched.append({"field": key, "self": mine.get(key),
                                   "other": theirs.get(key)})
        return {
            "fingerprint_self": self.fingerprint,
            "fingerprint_other": other.fingerprint,
            "fingerprints_equal": self.fingerprint == other.fingerprint,
            "matched_fields": matched,
            "mismatched_fields": mismatched,
            "unverifiable_fields": unknown,
            "unmaterialized_fields": unmaterialized,
            "both_materialized": (self.is_materialized and other.is_materialized),
            "protocol_identical": (not mismatched and not unknown),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.identity(), "protocol_fingerprint": self.fingerprint,
                "unverifiable": list(self.unverifiable),
                "is_materialized": self.is_materialized,
                "unmaterialized_fields": list(self.unmaterialized_fields()),
                "materialization_required": list(self.MATERIALIZATION_REQUIRED),
                "excluded_by_design": ["student initialization artifact", "seed"],
                "why_excluded": ("initialization is the treatment variable and the "
                                 "seed is the intended replicate; including either "
                                 "would make every comparable pair of arms differ")}


class ObservedProtocolError(RecoveryAdmissionError):
    """A run's own artifacts do not establish a material protocol field."""


def normalize_trainable_patterns(value: Any) -> tuple[str, ...] | None:
    """`"all"` is one pattern, not three characters.

    `tuple("all")` is `('a', 'l', 'l')`, so a config using the string form would
    fingerprint differently depending on which side of the comparison built it.
    Every construction site goes through this.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(value)


class _Observation:
    """Collects observed values and the fields nothing established.

    Every read goes through here so a missing field is *recorded* rather than
    silently becoming a default. Reads are collected and reported together: one
    exception naming eight missing fields is a diagnosis, eight exceptions in
    sequence are a scavenger hunt.
    """

    def __init__(self) -> None:
        self.missing: list[str] = []
        self.problems: list[str] = []
        self.sources: dict[str, str] = {}

    def need(self, root: Mapping[str, Any] | None, path: str, *, field: str,
             cast=None) -> Any:
        """Read ``path`` (dotted) out of ``root``; record it as missing if absent."""
        node: Any = root
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if node is None:
            self.missing.append(f"{field} (no evidence at {path})")
            return None
        self.sources[field] = path
        return cast(node) if cast is not None else node

    def agree(self, field: str, observed: Any, declared: Any, *,
              what: str = "config") -> None:
        """Cross-check a resolved value against the value the run declared."""
        if observed is None or declared is None:
            return
        if observed != declared:
            self.problems.append(
                f"{field}: the run executed {observed!r} but its {what} says "
                f"{declared!r}; the run's own artifacts disagree about what ran")


@dataclass(frozen=True)
class ObservedRecoveryProtocol:
    """A protocol reconstructed from one run's artifacts, with its evidence."""

    protocol: "RecoveryProtocolFingerprint"
    seed: int | None
    initialization_source: str | None
    evidence: dict[str, Any]

    @property
    def is_strict(self) -> bool:
        return not self.protocol.unverifiable

    def as_dict(self) -> dict[str, Any]:
        return {"observed_protocol": self.protocol.as_dict(),
                "observed_protocol_fingerprint": self.protocol.fingerprint,
                "seed": self.seed,
                "initialization_source": self.initialization_source,
                "evidence": self.evidence}


def observe_recovery_protocol(run_dir: str | Path, *,
                              repo_root: str | Path | None = None,
                              pack_root: str | Path | None = None,
                              strict: bool = True) -> ObservedRecoveryProtocol:
    """Read a completed run's artifacts and reconstruct what it executed.

    Evidence, in order of authority:

    ``run_manifest.json``      the config as emitted, the ladder stats with the
                               pack hashes computed **at read time on the pod**,
                               the teacher identity resolved from the hub, the
                               tokenizer hash, the trainable-parameter count, and
                               the ``execution`` block the trainer wrote off its
                               own objects (optimizer, resolved KD chunk,
                               schedule and ordering rules, runtime, trainer
                               source digest).
    ``run_completion.json``    step and consumed-block accounting after the loop
                               returned. The manifest is written before the first
                               step, so on its own it describes an intention.
    the pack on disk           ``blocks.npz`` is re-hashed here and must equal
                               what the run recorded. A pinned constant is not
                               evidence that *this* run read *that* pack.
    """
    run = Path(run_dir)
    manifest_path = run / "run_manifest.json"
    completion_path = run / "run_completion.json"
    obs = _Observation()

    if not manifest_path.is_file():
        raise ObservedProtocolError(
            f"{manifest_path} does not exist; there is no evidence of what this "
            "run executed. A control is its artifacts, not its intention.")
    manifest = json.loads(manifest_path.read_text())
    completion = (json.loads(completion_path.read_text())
                  if completion_path.is_file() else None)
    if completion is None:
        obs.missing.append("run completion record (run_completion.json)")

    cfg = manifest.get("config") or {}
    execution = manifest.get("execution") or {}
    ladder = manifest.get("ladder") or {}
    if not execution:
        obs.missing.append(
            "execution record (run_manifest.execution) — this run was produced "
            "by a trainer that records no execution evidence")

    # -- data ------------------------------------------------------------
    data_dir = obs.need(cfg, "data_dir", field="pack")
    pack = Path(data_dir).name if data_dir else None
    packing = obs.need(cfg, "packing", field="packing")
    if packing is not None and packing != "ladder":
        obs.problems.append(
            f"packing is {packing!r}; the pack identity this protocol pins "
            "(blocks.npz) exists only for a ladder run")
    rung = obs.need(cfg, "rung", field="rung", cast=int)
    val_blocks = obs.need(cfg, "val_blocks", field="val_blocks", cast=int)
    train_blocks = obs.need(ladder, "train_blocks", field="train_blocks", cast=int)
    train_tokens = obs.need(ladder, "train_supervised_tokens",
                            field="train_supervised_tokens", cast=int)
    recorded_pack_sha = obs.need(ladder, "blocks_sha256",
                                 field="pack_blocks_sha256")
    obs.agree("rung", rung, ladder.get("rung_target_supervised_tokens"),
              what="ladder stats")

    # The pack hash is RECOMPUTED from the pack the run named. A value copied
    # from the frozen pin would prove only that the pin exists.
    pack_sha = None
    root = Path(pack_root) if pack_root is not None else (
        Path(repo_root) if repo_root is not None else None)
    if data_dir is not None:
        if root is None:
            obs.missing.append(
                "pack_blocks_sha256 (no repo_root given, so the pack the run "
                "consumed cannot be re-hashed)")
        else:
            blocks = root / data_dir / "blocks.npz"
            if not blocks.is_file():
                obs.missing.append(
                    f"pack_blocks_sha256 (the consumed pack {blocks} is not on "
                    "this machine, so it cannot be re-hashed)")
            else:
                pack_sha = sha256_file(blocks)
                if recorded_pack_sha is not None and pack_sha != recorded_pack_sha:
                    obs.problems.append(
                        f"pack_blocks_sha256: the pack on disk hashes to "
                        f"{pack_sha} but the run recorded {recorded_pack_sha}; "
                        "the pack changed after the run read it, or this is not "
                        "the pack it read")

    # -- objective, optimizer, schedule, batch, numerics: from the run's own
    # execution record, cross-checked against the config it was given ------
    ce = obs.need(execution, "ce_weight", field="ce_weight", cast=float)
    kd = obs.need(execution, "kd_weight", field="kd_weight", cast=float)
    kd_t = obs.need(execution, "kd_temperature", field="kd_temperature", cast=float)
    kd_scope = obs.need(execution, "kd_scope", field="kd_scope")
    kd_chunk = obs.need(execution, "kd_chunk", field="kd_chunk", cast=int)
    loss_cfg = cfg.get("loss") or {}
    for name, value in (("ce_weight", ce), ("kd_weight", kd),
                        ("kd_temperature", kd_t), ("kd_scope", kd_scope)):
        obs.agree(name, value, loss_cfg.get(name))

    optimizer = obs.need(execution, "optimizer", field="optimizer")
    lr = obs.need(execution, "optimizer_defaults.lr", field="lr", cast=float)
    weight_decay = obs.need(execution, "optimizer_defaults.weight_decay",
                            field="weight_decay", cast=float)
    betas = obs.need(execution, "optimizer_defaults.betas", field="betas",
                     cast=lambda b: tuple(float(x) for x in b))
    eps = obs.need(execution, "optimizer_defaults.eps", field="eps", cast=float)
    grad_clip = obs.need(execution, "grad_clip", field="grad_clip", cast=float)
    optim_cfg = cfg.get("optim") or {}
    obs.agree("lr", lr, optim_cfg.get("lr"))
    obs.agree("weight_decay", weight_decay, optim_cfg.get("weight_decay"))
    obs.agree("eps", eps, optim_cfg.get("eps"))
    obs.agree("betas", betas, (tuple(optim_cfg["betas"])
                               if optim_cfg.get("betas") else None))

    total_steps = obs.need(execution, "total_steps", field="total_steps", cast=int)
    warmup = obs.need(execution, "warmup_steps", field="warmup_steps", cast=int)
    min_lr_frac = obs.need(execution, "min_lr_frac", field="min_lr_frac", cast=float)
    lr_schedule = obs.need(execution, "lr_schedule", field="lr_schedule")
    block_ordering = obs.need(execution, "block_ordering", field="block_ordering")
    resume_semantics = obs.need(execution, "resume_semantics",
                                field="resume_semantics")
    blocks_per_step = obs.need(execution, "blocks_per_step",
                               field="blocks_per_step", cast=int)
    micro_blocks = obs.need(execution, "micro_blocks", field="micro_blocks", cast=int)
    block_len = obs.need(execution, "block_len", field="block_len", cast=int)
    obs.agree("block_len", block_len, cfg.get("block_len"))
    obs.agree("block_len", block_len, ladder.get("block_len"), what="ladder stats")

    dtype = obs.need(execution, "dtype", field="dtype")
    autocast = obs.need(execution, "autocast_bf16", field="autocast_bf16", cast=bool)
    grad_ckpt = obs.need(execution, "gradient_checkpointing",
                         field="gradient_checkpointing", cast=bool)

    patterns = obs.need(cfg, "trainable_patterns", field="trainable_patterns",
                        cast=normalize_trainable_patterns)
    trainable_params = obs.need(manifest, "trainable_params",
                                field="trainable_params", cast=int)
    obs.agree("trainable_params", trainable_params,
              execution.get("trainable_params"), what="execution record")

    # -- teacher, tokenizer ----------------------------------------------
    teacher_id = obs.need(manifest, "teacher.model_id", field="teacher_id")
    teacher_rev = obs.need(manifest, "teacher.revision", field="teacher_revision")
    teacher_dtype = obs.need(manifest, "teacher.dtype", field="teacher_dtype")
    teacher_attn = obs.need(manifest, "teacher.attn_implementation",
                            field="teacher_attn")
    tokenizer_sha = obs.need(manifest, "tokenizer_sha256", field="tokenizer_sha256")

    # -- execution identity ----------------------------------------------
    trainer_digest = obs.need(execution, "trainer_source.digest",
                              field="trainer_source_digest")
    trainer_set_version = obs.need(execution, "trainer_source.set_version",
                                   field="trainer_source_set_version", cast=int)
    runtime_digest = obs.need(execution, "runtime_digest", field="runtime_digest")
    image_digest = obs.need(execution, "runtime.image_digest",
                            field="runtime.image_digest")
    if runtime_digest is not None and image_digest is None:
        obs.problems.append(
            "the run recorded a runtime digest with no image digest; the "
            "runtime is not pinned, so two runs cannot be asserted to share it")

    # -- accounting: did the run finish what it declared? ------------------
    accounting: dict[str, Any] = {"present": completion is not None}
    if completion is not None:
        final_step = completion.get("final_step")
        planned = completion.get("planned_total_steps")
        consumed = completion.get("consumed_blocks")
        accounting.update({
            "final_step": final_step, "planned_total_steps": planned,
            "completed_all_steps": completion.get("completed_all_steps"),
            "consumed_blocks": consumed,
        })
        if not completion.get("completed_all_steps"):
            obs.problems.append(
                f"the run stopped at step {final_step} of {planned}; a partially "
                "trained control is not the control the protocol describes")
        if total_steps is not None and planned is not None and planned != total_steps:
            obs.problems.append(
                f"completion record plans {planned} steps but the execution "
                f"record says {total_steps}")
        if (consumed is not None and final_step is not None
                and blocks_per_step is not None
                and consumed != final_step * blocks_per_step):
            obs.problems.append(
                f"consumed blocks {consumed} != final step {final_step} x "
                f"blocks_per_step {blocks_per_step}; block accounting is broken")
        if completion.get("config_sha256") != manifest.get("config_sha256"):
            obs.problems.append(
                "the completion record was written under a different config hash "
                "than the manifest; these artifacts are not from one run")

    resumed_from = manifest.get("resumed_from")
    if resumed_from:
        obs.problems.append(
            f"this run resumed from {resumed_from}; its protocol evidence spans "
            "several manifests and is not reconstructed here. Refusing rather "
            "than reconstructing half of it.")

    seed = cfg.get("seed")
    obs.agree("seed", execution.get("seed"), seed, what="config")

    if obs.problems and strict:
        raise ObservedProtocolError(
            f"{run}: the run's artifacts are inconsistent with each other or "
            f"with the pack on disk:\n  - " + "\n  - ".join(obs.problems))
    if obs.missing and strict:
        raise ObservedProtocolError(
            f"{run}: {len(obs.missing)} material protocol field(s) have no "
            "evidence in the run's own artifacts, so this run's protocol cannot "
            "be established:\n  - " + "\n  - ".join(obs.missing)
            + "\n\nA value filled from the preregistration would make the "
              "comparison a tautology. Fail closed instead.")

    unverifiable = tuple(sorted(m.split(" (")[0] for m in obs.missing))
    protocol = RecoveryProtocolFingerprint(
        pack=pack, pack_blocks_sha256=pack_sha, rung=rung,
        train_blocks=train_blocks, train_supervised_tokens=train_tokens,
        block_len=block_len, packing=packing, val_blocks=val_blocks,
        block_ordering=block_ordering,
        ce_weight=ce, kd_weight=kd, kd_temperature=kd_t, kd_scope=kd_scope,
        kd_chunk=kd_chunk,
        optimizer=optimizer, lr=lr, weight_decay=weight_decay, betas=betas,
        eps=eps, grad_clip=grad_clip,
        total_steps=total_steps, warmup_steps=warmup, min_lr_frac=min_lr_frac,
        lr_schedule=lr_schedule,
        blocks_per_step=blocks_per_step, micro_blocks=micro_blocks,
        dtype=dtype, autocast_bf16=autocast, gradient_checkpointing=grad_ckpt,
        trainable_patterns=patterns, trainable_params=trainable_params,
        teacher_id=teacher_id, teacher_revision=teacher_rev,
        teacher_dtype=teacher_dtype, teacher_attn=teacher_attn,
        tokenizer_sha256=tokenizer_sha,
        resume_semantics=resume_semantics,
        trainer_source_digest=trainer_digest,
        trainer_source_set_version=trainer_set_version,
        runtime_digest=runtime_digest,
        unverifiable=unverifiable,
    )
    return ObservedRecoveryProtocol(
        protocol=protocol,
        seed=int(seed) if seed is not None else None,
        initialization_source=manifest.get("student_source"),
        evidence={
            "strict": strict,
            "run_dir": str(run),
            "run_manifest_sha256": sha256_file(manifest_path),
            "run_completion_sha256": (sha256_file(completion_path)
                                      if completion_path.is_file() else None),
            "config_sha256": manifest.get("config_sha256"),
            "command": manifest.get("command"),
            "resumed_from": resumed_from,
            "pack_blocks_sha256_recorded_by_run": recorded_pack_sha,
            "pack_blocks_sha256_recomputed": pack_sha,
            "pack_recompute_path": (str(root / data_dir / "blocks.npz")
                                    if (root is not None and data_dir) else None),
            "step_accounting": accounting,
            "image_digest": image_digest,
            "runtime": execution.get("runtime"),
            "trainer_source_files": (execution.get("trainer_source") or {}).get("files"),
            "field_sources": obs.sources,
            "missing_fields": obs.missing,
            "internal_inconsistencies": obs.problems,
            "rule": ("every material field is read from the run's own artifacts; "
                     "the pack hash is recomputed from the pack the run named; "
                     "nothing is defaulted or inherited from a preregistration"),
        })


@dataclass(frozen=True)
class RecoveryProbeIdentity:
    """One probe: a protocol, an initialization, and a seed.

    ``matched_against`` is the operational question — two probes are a valid
    single-variable comparison when their protocols and seeds agree and their
    initializations differ.
    """

    protocol: RecoveryProtocolFingerprint
    initialization_artifact_digest: str
    seed: int
    label: str = ""

    @property
    def probe_id(self) -> str:
        return sha256_json({
            "protocol": self.protocol.fingerprint,
            "init": self.initialization_artifact_digest,
            "seed": self.seed})

    def matched_against(self, other: "RecoveryProbeIdentity") -> dict[str, Any]:
        protocol = self.protocol.compare(other.protocol)
        materialized = protocol["both_materialized"]
        same_seed = self.seed == other.seed
        different_init = (self.initialization_artifact_digest
                          != other.initialization_artifact_digest)
        # Materialization is a precondition, not one vote among several. Two
        # protocols that are both unknown in the same place are *unverifiable*,
        # and an unverifiable pair is never MATCHED.
        ok = (materialized and protocol["protocol_identical"]
              and same_seed and different_init)
        return {
            "self": self.label or self.probe_id[:12],
            "other": other.label or other.probe_id[:12],
            "protocols_materialized": materialized,
            "unmaterialized_fields": protocol["unmaterialized_fields"],
            "protocol_identical": protocol["protocol_identical"],
            "same_seed": same_seed,
            "initializations_differ": different_init,
            "is_single_variable_comparison": ok,
            "protocol_comparison": protocol,
            "verdict": (
                "MATCHED: same protocol, same seed, different initialization — the "
                "only difference is the treatment."
                if ok else
                "NOT ELIGIBLE FOR MATCHED: protocol is not materialized ("
                + ", ".join(protocol["unmaterialized_fields"])
                + " unknown on one or both sides). Unknown on both sides is not "
                  "verified identical; attest at preflight Stage 0 first."
                if not materialized else
                "NOT MATCHED: "
                + ("; ".join(filter(None, [
                    None if protocol["protocol_identical"] else "protocol differs",
                    None if same_seed else "seeds differ",
                    None if different_init else
                    "initializations are identical, so there is no treatment"])))),
        }

    def require_attested(self, attested_protocol_fingerprint: str) -> None:
        """Raise unless this probe ran the attested protocol, byte for byte.

        Stage 2's check. The comparison target is the fingerprint frozen *after*
        Stage 0 attestation, not the preregistered object that still carries
        ``runtime_digest: null`` — comparing against the latter would accept a
        control trained under an unpinned runtime.
        """
        self.protocol.require_materialized(context=f"probe {self.label or self.probe_id[:12]}")
        actual = self.protocol.fingerprint
        if actual != attested_protocol_fingerprint:
            raise RecoveryAdmissionError(
                f"probe {self.label or self.probe_id[:12]} ran protocol {actual} "
                f"but the attested Phase-A protocol is "
                f"{attested_protocol_fingerprint}. A control that did not run the "
                "attested protocol is not a control.")

    def as_dict(self) -> dict[str, Any]:
        return {"probe_id": self.probe_id, "label": self.label, "seed": self.seed,
                "initialization_artifact_digest": self.initialization_artifact_digest,
                "protocol_fingerprint": self.protocol.fingerprint,
                "protocol": self.protocol.as_dict()}


@dataclass(frozen=True)
class HistoricalRunAudit:
    """The complete record of a past run, **including** init and seed.

    Kept separate from the protocol identity on purpose. This is what you use to
    ask "what exactly was that run?"; it is never the equality predicate for
    matched arms.
    """

    run_id: str
    protocol: RecoveryProtocolFingerprint
    initialization_artifact_digest: str
    seed: int
    repo_git_commit: str | None
    repo_dirty: bool | None
    repo_uncommitted_sha256: str | None
    weights_sha256: str | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "seed": self.seed,
            "initialization_artifact_digest": self.initialization_artifact_digest,
            "weights_sha256": self.weights_sha256,
            "protocol": self.protocol.as_dict(),
            "provenance_only": {
                "repo_git_commit": self.repo_git_commit,
                "repo_dirty": self.repo_dirty,
                "repo_uncommitted_sha256": self.repo_uncommitted_sha256,
                "note": ("whole-repository state is provenance, never the material "
                         "trainer identity; a docs-only commit must not invalidate "
                         "a control"),
            },
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class PreflightStage:
    stage: int
    name: str
    purpose: str
    produces: tuple[str, ...]
    blocking: bool
    stop_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # `("one item")` is a string, not a 1-tuple, and `list()` of it serializes
        # as one entry per character. Caught once in the real plan; refused here so
        # a missing comma is a construction error rather than a silent 47-condition
        # stage in a frozen artifact.
        for name in ("produces", "stop_conditions"):
            if isinstance(getattr(self, name), str):
                raise TypeError(
                    f"PreflightStage.{name} must be a tuple of strings, not a "
                    f"single string — a missing trailing comma serializes as a "
                    f"list of characters")

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "name": self.name, "purpose": self.purpose,
                "produces": list(self.produces), "blocking": self.blocking,
                "stop_conditions": list(self.stop_conditions)}


@dataclass(frozen=True)
class PreflightPlan:
    """Staged, fail-closed, and ordered so money is spent last.

    The ordering is the whole point. The permanent control checkpoints are the
    expensive artifact and the one whose value depends on the runtime staying
    fixed — so they are trained **only after** the runtime is attested and the
    cheap machine gates have passed. If Stage 1 says the hardware, evaluator
    tolerance or storage plan has to change, the session stops before spending
    anything on controls that would immediately stop being matched.

    ``advance_to`` is the executable form: a stage cannot start until every
    blocking stage before it has passed.
    """

    stages: tuple[PreflightStage, ...]
    plan_id: str = "autoinit.micro_preflight"
    version: int = 1

    @property
    def plan_hash(self) -> str:
        return sha256_json({"plan_id": self.plan_id, "version": self.version,
                            "stages": [s.as_dict() for s in self.stages]})

    def advance_to(self, stage: int, results: Mapping[int, Mapping[str, Any]]) -> None:
        """Raise unless every blocking earlier stage has recorded a pass."""
        for earlier in self.stages:
            if earlier.stage >= stage or not earlier.blocking:
                continue
            outcome = results.get(earlier.stage)
            if outcome is None:
                raise RecoveryAdmissionError(
                    f"stage {stage} cannot start: blocking stage {earlier.stage} "
                    f"({earlier.name}) has no recorded result")
            if not outcome.get("passed"):
                raise RecoveryAdmissionError(
                    f"stage {stage} cannot start: blocking stage {earlier.stage} "
                    f"({earlier.name}) did not pass "
                    f"({outcome.get('reason', 'no reason recorded')}). "
                    "Do not train permanent controls under a runtime or hardware "
                    "configuration that is about to change.")

    def as_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, "version": self.version,
                "plan_hash": self.plan_hash,
                "stages": [s.as_dict() for s in self.stages],
                "ordering_rationale": (
                    "controls are the expensive artifact and the one whose validity "
                    "depends on a fixed runtime, so they are produced only after "
                    "the runtime is attested and the cheap gates pass")}


PREFLIGHT_PLAN_V1 = PreflightPlan(stages=(
    PreflightStage(
        stage=0, name="runtime attestation", blocking=True,
        purpose=("pin what everything else will be measured and trained under, "
                 "before any of it happens"),
        produces=("image digest", "RuntimeEnvironmentFingerprint",
                  "trainer_source_digest + declared file set",
                  "input artifact hashes (canonical init, pack, suite, battery)",
                  "materialized RecoveryProtocolFingerprint",
                  "frozen attested protocol artifact: "
                  "logs/autoinit_phase_a_protocol_attested.json"),
        stop_conditions=("image digest unavailable",
                         "any input artifact hash mismatches its pin",
                         "trainer source file missing from the declared set",
                         "attested trainer digest or runtime contradicts a "
                         "preregistered value -> protocol drift, STOP")),
    PreflightStage(
        stage=1, name="cheap machine gates", blocking=True,
        purpose="find out whether this machine and runtime are usable at all",
        produces=("activation-statistics GPU/CPU split",
                  "GPU state-evaluator repeatability range",
                  "peak GPU resident memory on the widest operator",
                  "checkpoint write/read throughput"),
        stop_conditions=(
            "evaluator repeatability range >= declared epsilon -> "
            "conservative_review_gate fires, Phase A blocked, STOP",
            "peak resident memory > 40 GiB on an L40S -> hardware plan wrong, STOP",
            "disk throughput implies the working set is not feasible -> STOP")),
    PreflightStage(
        stage=2, name="permanent canonical controls", blocking=True,
        purpose=("produce the two matched control probes that Phase A will reuse "
                 "at rung 2"),
        produces=("canonical init -> 0.86M seed sa",
                  "canonical init -> 0.86M seed sb",
                  "per-run RecoveryProtocolFingerprint and RecoveryProbeIdentity",
                  "checkpoint artifact digests"),
        stop_conditions=(
            "a run's protocol fingerprint differs from the Stage-0 ATTESTED "
            "protocol hash (not the preregistered object, which still carries "
            "runtime_digest: null)",
            "a run's protocol is not materialized",
            "step time diverges from the priced 4.15 s/step by more than 25%")),
    PreflightStage(
        stage=3, name="control characterization", blocking=False,
        purpose="materialize the frozen threshold formulas from control data only",
        produces=("sa and sb on recovery_search_v1",
                  "pooled_counts@v1 aggregate + per-seed rates",
                  "materialized equivalence interval",
                  "materialized feasibility floor",
                  "per-capability control baselines"),
        stop_conditions=("capability schema validation fails -> scoring defect, STOP",)),
))


class ScoringContractError(RuntimeError):
    """A scored recovery row violates the metric contract."""


def score_recovery_row(*, usable: bool, scorer_correct: bool,
                       scorable: bool = True) -> dict[str, bool]:
    """One scored rollout, with ``correct => usable`` true **by construction**.

    The semantic question is real and is resolved here rather than left for the
    aggregator to trip over. A rollout can contain an extractable correct answer
    and still be unusable: it can hit the context limit after answering, or fall
    into a repetition loop, or break protocol. So a scorer alone *can* say
    "correct" about a rollout the behaviour metric calls unusable.

    **This battery defines ``correct`` as "correct in a usable rollout".** A
    checkpoint that emits the right answer and then loops forever cannot produce
    trajectories for Stage 5, and counting it as correct would let the primary
    metric reward the exact failure that dominates this project — ~31% of rollouts
    hitting the context limit. `correct_given_usable` then means what it says, and
    `correct_overall` means "answered correctly, in a rollout we could actually
    use".

    The rejected alternative is recorded: scoring correctness independently of
    usability would make `correct_overall` a measure of latent capability rather
    than of deployable behaviour, and would break `correct <= usable` in the
    aggregate.
    """
    correct = bool(scorable and usable and scorer_correct)
    return {
        "usable": bool(usable),
        "scorer_correct": bool(scorer_correct),
        "scorable": bool(scorable),
        "correct": correct,
        # Recorded so the gap between "the scorer found an answer" and "we counted
        # it" is visible rather than silently absorbed.
        "correct_but_unusable": bool(scorable and scorer_correct and not usable),
    }


def validate_scored_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Enforce the contract on scored rows **before** they are aggregated.

    `SeedAggregation.pool` also refuses `correct > usable`, but by then the
    offending row is invisible — the counts are already summed. This runs on rows,
    so a violation names the prompt.
    """
    violations = []
    counts = {"n": 0, "usable": 0, "correct": 0, "scorable": 0,
              "correct_but_unusable": 0}
    for row in rows:
        counts["n"] += 1
        counts["usable"] += bool(row.get("usable"))
        counts["correct"] += bool(row.get("correct"))
        counts["scorable"] += bool(row.get("scorable", True))
        counts["correct_but_unusable"] += bool(row.get("correct_but_unusable"))
        if row.get("correct") and not row.get("usable"):
            violations.append({
                "id": row.get("id"), "set": row.get("set"),
                "reason": ("correct=True with usable=False; this battery defines "
                           "correct as 'correct in a usable rollout'")})
        if row.get("correct") and not row.get("scorable", True):
            violations.append({
                "id": row.get("id"), "set": row.get("set"),
                "reason": "correct=True on a behaviour-only row"})
    if violations:
        raise ScoringContractError(
            f"{len(violations)} scored rows violate correct => usable: "
            f"{violations[:3]}")
    return counts

SEED_SA = 20260726
SEED_SB = 20260801
#: Tie-break seed. Used only for candidates that finish inside the preregistered
#: equivalence interval after two seeds — never as a third look at everything.
SEED_SC = 20260813


@dataclass(frozen=True)
class SuccessiveHalvingPlan:
    """A preregistered probe schedule. Freeze it, then run it."""

    plan_id: str
    recipe: RecoveryRecipe
    #: Searched leaves admitted at rung 1. The canonical control is additional.
    searched_leaves: int
    #: Searched leaves advancing to rung 2. The control advances unconditionally.
    survivors: int
    seeds: tuple[int, ...] = (SEED_SA, SEED_SB)
    tie_break_seed: int | None = SEED_SC
    include_canonical_control: bool = True
    #: Feasibility constraint. Blind to correctness by construction, so it gates
    #: rather than scores.
    feasibility_metric: str = "usable_rollout_rate"
    feasibility_min: float = 0.0
    #: The capability objective, applied among feasible candidates only.
    primary_metric: str = "correct_overall"
    #: Reported to explain a ranking; never changes one.
    secondary_metric: str = "correct_given_usable"
    reported_components: tuple[str, ...] = (
        "non_empty", "natural_termination", "no_severe_repetition",
        "no_context_limit", "protocol_valid")
    #: The behaviour equivalence interval. A rule, not a constant: the formula is
    #: frozen now, the numeric value comes from the control characterization.
    equivalence: EquivalenceRule = field(
        default_factory=lambda: EquivalenceRule(n_pooled=300))
    #: Per-capability collapse gate, enforced at both rungs.
    catastrophic: CatastrophicCapabilityRule = CATASTROPHIC_V1
    #: What every result must carry for that gate to be able to see anything.
    #:
    #: `None` means no per-capability contract is declared, in which case the
    #: catastrophic rule is **disabled** rather than silently passing: a rule that
    #: cannot see a capability must not report a verdict on it. Every selection
    #: result records `capability_schema_enforced` so the difference is visible,
    #: and the Phase-A plan always declares one.
    capability_schema: CapabilitySchema | None = CAPABILITY_SCHEMA_V1
    #: Seed-aware usable-rollout floor. Materialized from the control.
    feasibility: FeasibilityRule | None = None
    #: How per-seed counts become one number. Part of the plan hash.
    aggregation: SeedAggregation = POOLED_COUNTS_V1
    survivor_rule: str = ""
    winner_rule: str = ""
    battery_asset_id: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.searched_leaves < 2:
            raise ValueError("successive halving needs at least 2 searched leaves")
        if not 1 <= self.survivors < self.searched_leaves:
            raise ValueError(
                f"survivors ({self.survivors}) must be at least 1 and fewer than "
                f"searched_leaves ({self.searched_leaves}); otherwise rung 1 "
                "selects nothing")
        if len(self.seeds) < 2:
            raise ValueError(
                "one seed cannot rank close candidates: the behaviour metric's "
                "seed-only spread is 0.1290")
        if self.feasibility_metric == self.primary_metric:
            raise ValueError(
                "the feasibility constraint and the capability objective must be "
                "different metrics; usable_rollout is blind to correctness by "
                "construction, which is exactly why it gates rather than ranks")

        if not self.survivor_rule or not self.winner_rule:
            raise ValueError(
                "survivor_rule and winner_rule must be stated before the run; a rule "
                "written after the table is a rule chosen on the outcome")

    @property
    def rung1_probes(self) -> int:
        return self.searched_leaves + (1 if self.include_canonical_control else 0)

    @property
    def rung2_probes(self) -> int:
        return self.survivors + (1 if self.include_canonical_control else 0)

    @property
    def probe_count(self) -> int:
        """Probes excluding any tie-break rung, which is conditional."""
        return self.rung1_probes + self.rung2_probes

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "plan_id": self.plan_id,
            "recipe": self.recipe.as_dict(),
            "searched_leaves": self.searched_leaves,
            "survivors": self.survivors, "seeds": list(self.seeds),
            "tie_break_seed": self.tie_break_seed,
            "include_canonical_control": self.include_canonical_control,
            "rung1_probes": self.rung1_probes, "rung2_probes": self.rung2_probes,
            "probe_count": self.probe_count,
            "seed_aggregation": self.aggregation.as_dict(),
            "equivalence_rule": self.equivalence.as_dict(),
            "feasibility_rule": (self.feasibility.as_dict() if self.feasibility
                                 else {"status": "PENDING_CONTROL_CHARACTERIZATION"}),
            "catastrophic_capability_rule": self.catastrophic.as_dict(),
            "capability_schema": (self.capability_schema.as_dict()
                                  if self.capability_schema else None),
            "selection": {
                "feasibility_metric": self.feasibility_metric,
                "feasibility_min": self.feasibility_min,
                "primary_metric": self.primary_metric,
                "secondary_metric": self.secondary_metric,
                "equivalence_interval": self.equivalence.value,
                "rule": ("feasibility constraint, then primary objective among "
                         "feasible candidates; the secondary metric is reported and "
                         "never reorders. No weighted combination."),
                "control_eligible_to_win": True,
                "control_exempt_from_feasibility_gate": True,
            },
            "reported_components": list(self.reported_components),
            "survivor_rule": self.survivor_rule, "winner_rule": self.winner_rule,
            "battery_asset_id": self.battery_asset_id,
            "notes": dict(self.notes),
        }

    # --- selection: two different questions, two different functions ---------
    #
    # Rung 1 asks "which searched leaves are worth a second seed?" — the control
    # is not competing for those slots, it advances on its own.
    # The final asks "which initialization won?" — and the control is a
    # candidate like any other, because **"the incumbent won, AutoInitializer v1
    # did not improve recovered behaviour" has to be a reachable conclusion.**
    # A single generic `select()` that always excluded the control from the
    # winner list made the experiment asymmetric: it could confirm an
    # improvement and could never refute one.

    def feasibility_floor(self) -> float:
        """The materialized floor. Prefers the seed-aware rule when present."""
        if self.feasibility is not None:
            return self.feasibility.require_value()
        return self.feasibility_min

    def _gate(self, results: Sequence[Mapping[str, Any]]) -> tuple[list, list]:
        """Global feasibility floor **and** the per-capability collapse rule.

        Both are mechanical, and the capability schema is validated first: a rule
        that cannot see a capability must not report a pass on it.
        """
        enforced = self.capability_schema is not None
        if enforced:
            self.capability_schema.validate_all(results)
        floor = self.feasibility_floor()
        control = next((r for r in results if r.get("is_control")), None)
        feasible, excluded = [], []
        for row in results:
            if row.get("is_control"):
                # The control is never gated out. A baseline that fails the floor
                # is a finding about the floor or the baseline, and dropping it
                # would leave the searched candidates with nothing to beat.
                feasible.append(row)
                continue

            value = float(row.get(self.feasibility_metric, 0.0))
            if value < floor:
                excluded.append({**dict(row), "exclusion": "feasibility_floor",
                                 "reason": (
                    f"{self.feasibility_metric}={value:.4f} below the preregistered "
                    f"feasibility floor {floor:.4f}")})
                continue

            violations = (self.catastrophic.violations(row, control)
                          if (enforced and control is not None) else [])
            if violations:
                excluded.append({**dict(row), "exclusion": "catastrophic_capability",
                                 "violations": violations,
                                 "reason": "; ".join(v["reason"] for v in violations)})
                continue
            feasible.append(row)
        return feasible, excluded

    def _rank(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: (-float(r.get(self.primary_metric, 0.0)),
                                           str(r["state_id"])))

    def _tied_with_leader(self, ranked: Sequence[Mapping[str, Any]]) -> list[str]:
        if not ranked:
            return []
        interval = self.equivalence.require_value()
        best = float(ranked[0][self.primary_metric])
        return [r["state_id"] for r in ranked
                if abs(float(r[self.primary_metric]) - best) <= interval]

    def select_rung1_survivors(self, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Which **searched** leaves advance to seed sb.

        The control is not eligible for these slots and does not consume one; it
        advances unconditionally, which is what makes rung 2 a two-seed baseline
        comparison rather than a ranking of survivors against each other.
        """
        feasible, excluded = self._gate(results)
        ranked = self._rank(feasible)
        searched = [r for r in ranked if not r.get("is_control")]
        chosen = searched[:self.survivors]
        control = [r["state_id"] for r in ranked if r.get("is_control")]
        return {
            "rung": 1,
            "ranked": ranked,
            "selected_searched": [r["state_id"] for r in chosen],
            "auto_advanced_control": control,
            "advancing": [*control, *(r["state_id"] for r in chosen)],
            "excluded_by_feasibility": [e for e in excluded
                                        if e["exclusion"] == "feasibility_floor"],
            "excluded_by_catastrophic_capability": [
                e for e in excluded if e["exclusion"] == "catastrophic_capability"],
            "all_exclusions": excluded,
            "control_present": any(r.get("is_control") for r in results),
            "capability_schema_enforced": self.capability_schema is not None,
            "rule": self.survivor_rule,
        }

    def select_final_winner(self, results: Sequence[Mapping[str, Any]], *,
                            tie_break_completed: bool | None = None) -> dict[str, Any]:
        """Top-1 over the pooled seeds — or an explicit "no winner".

        ``results`` are pooled-count aggregates (see ``SeedAggregation``), one per
        finalist, the control included. The winner is whichever finalist leads on
        the primary metric among those clearing the gates — searched or canonical,
        with no asymmetry between them.

        **A tie is not a winner.** Three outcomes, and only one of them names a
        checkpoint:

        ``resolved``               one finalist leads by more than the interval.
        ``tie_pending``            finalists are equivalent after sa+sb; seed sc is
                                   owed, and ``winner`` is ``None``.
        ``unresolved_equivalence`` finalists are *still* equivalent after sc.
                                   ``winner`` is ``None``, and that is the result:
                                   **AutoInitializer v1 did not resolve a unique
                                   behavioural winner.**

        No fourth seed is requested, and the deterministic state-id ordering is
        *not* used to break a scientific tie — it orders the report, nothing more.
        Manufacturing a winner from a lexicographic id would dress a null result
        as a finding.
        """
        feasible, excluded = self._gate(results)
        ranked = self._rank(feasible)
        tied = self._tied_with_leader(ranked)
        leader = ranked[0] if ranked else None

        if tie_break_completed is None:
            # Derived from the data when the caller does not say: every tied
            # finalist having the tie-break seed in its pooled seed list means sc
            # has already run for them.
            seeds_seen = [set(r.get("seeds") or []) for r in ranked
                          if r["state_id"] in tied]
            tie_break_completed = bool(seeds_seen) and all(
                self.tie_break_seed in s for s in seeds_seen)

        is_tie = len(tied) > 1
        if not is_tie:
            status = "resolved"
        elif not tie_break_completed and self.tie_break_seed is not None:
            status = "tie_pending"
        else:
            status = "unresolved_equivalence"

        return {
            "rung": "final",
            "ranked": ranked,
            "decision_status": status,
            "winner": leader["state_id"] if (leader and status == "resolved") else None,
            "winner_is_control": bool(leader and status == "resolved"
                                      and leader.get("is_control")),
            "provisional_leader": leader["state_id"] if leader else None,
            "provisional_leader_is_control": bool(leader and leader.get("is_control")),
            "excluded_by_feasibility": [e for e in excluded
                                        if e["exclusion"] == "feasibility_floor"],
            "excluded_by_catastrophic_capability": [
                e for e in excluded if e["exclusion"] == "catastrophic_capability"],
            "all_exclusions": excluded,
            "equivalence_interval": self.equivalence.require_value(),
            "capability_schema_enforced": self.capability_schema is not None,
            "control_present": any(r.get("is_control") for r in results),
            "tied_within_equivalence": tied,
            "needs_tie_break_seed": status == "tie_pending",
            "tie_break_candidates": tied if status == "tie_pending" else [],
            "tie_break_completed": tie_break_completed,
            "interpretation": (
                "AutoInitializer v1 did not resolve a unique behavioural winner; "
                "whether to full-recover both tied candidates is a separate "
                "decision and a separate authorization."
                if status == "unresolved_equivalence" else None),
            "rule": self.winner_rule,
        }

    @property
    def plan_hash(self) -> str:
        return sha256_json(self.as_dict())

    def freeze(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            **self.as_dict(),
            "plan_hash": self.plan_hash,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True) + "\n")
        return p


def assert_preregistered(plan: SuccessiveHalvingPlan, path: str | Path) -> dict[str, Any]:
    """Refuse a selection whose plan is not the frozen one."""
    p = Path(path)
    if not p.is_file():
        raise RecoveryAdmissionError(
            f"no frozen plan at {p}; a Top-N selection needs its rules registered first")
    frozen = json.loads(p.read_text())
    if frozen.get("plan_hash") != plan.plan_hash:
        raise RecoveryAdmissionError(
            f"the plan in {p} hashes to {frozen.get('plan_hash', '')[:12]} but the "
            f"plan being executed hashes to {plan.plan_hash[:12]}; a threshold moved "
            "after freezing")
    return frozen


def admit_leaves(candidates: Sequence[InitializationState],
                 plan: SuccessiveHalvingPlan) -> list[InitializationState]:
    """The gate. Only complete, measured, target-size leaves get through.

    Refusing intermediates is not a formality. A depth-only intermediate is
    *larger* than the target and will usually look better on teacher KL than any
    fully compressed leaf — E8b measured exactly that reversal, DC beating DP by
    0.89–1.18 nats at full width while FC lost to FP by 0.90–2.82 nats once
    compressed. An unguarded Top-N would fill up with states that cannot be
    deployed and whose ranking does not transfer.
    """
    searched = [s for s in candidates if s.provenance != "retained_canonical"]
    for state in candidates:
        state.require_recovery_admissible()
    if len(searched) < plan.searched_leaves:
        raise RecoveryAdmissionError(
            f"plan asks for {plan.searched_leaves} searched leaves but only "
            f"{len(searched)} admissible ones exist; report the shortfall rather "
            "than shrinking N")
    if plan.include_canonical_control and len(searched) == len(candidates):
        raise RecoveryAdmissionError(
            "the plan includes the canonical control but none was supplied; a "
            "comparison whose baseline is a re-executed recipe rather than the "
            "retained checkpoint is not the comparison the plan describes")
    return list(candidates)


def probe_configs(selected: Sequence[InitializationState],
                  plan: SuccessiveHalvingPlan, *, rung: int = 1) -> list[dict[str, Any]]:
    """Probe descriptors: identical recovery, one per candidate, one seed.

    Descriptors only — nothing here launches anything.
    """
    if rung < 1 or rung > len(plan.seeds):
        raise RecoveryAdmissionError(
            f"rung {rung} has no seed in {list(plan.seeds)}")
    seed = plan.seeds[rung - 1]
    suffix = f"s{'abc'[rung - 1]}"
    return [
        {
            "probe_id": f"{plan.plan_id}.rung{rung}.{state.state_id[:12]}.{suffix}",
            "rung": rung,
            "state_id": state.state_id,
            "path": state.path_label,
            "is_control": state.provenance == "retained_canonical",
            "student_checkpoint": state.checkpoint_path,
            "student_artifact_digest": state.artifact_digest,
            "student_single_shard_sha256": state.checkpoint_sha256,
            "recipe": plan.recipe.recipe_id,
            "seed": seed,
            "feasibility_metric": plan.feasibility_metric,
            "primary_metric": plan.primary_metric,
            "_purpose": ("identical recovery across candidates; the only intended "
                         "difference between probes is the initialization"),
        }
        for state in selected
    ]
