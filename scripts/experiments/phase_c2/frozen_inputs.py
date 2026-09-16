"""The frozen candidate side of B->C, rehydrated for the comparison builder.

Attempt 4's beam completed and its five selected candidates were measured. The
baseline-completion session measures B and computes the comparison, so it needs
those five measurements as objects the existing
`experiments.phase_c2.comparison.build` already knows how to read -- without the
28.9 MB journal they were extracted from, and without the ability to change
them.

Two properties this module exists to enforce:

**The record is verified before it is used.** It carries its own `self_sha256`
over everything else in it, so a record edited after Attempt 4 froze it is
refused here rather than quietly compared against.

**Nothing is recomputed.** `StateEvaluation` is reconstructed field-for-field
from the stored mapping, which is why the freeze stores the COMPLETE
`as_dict()`. A loader that rebuilt a measurement from partial fields would be a
second measurement wearing the first one's identity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aadistill.infrastructure.manifest import sha256_json
from aadistill.initialization.specs.metrics import StateEvaluation

#: The only extraction this module knows how to read. A record written by a
#: later rule may mean something different, so it is refused rather than
#: interpreted optimistically.
SUPPORTED_EXTRACTION_RULES = ("c2.frozen_comparison_inputs/v1",)

SCHEMA = "aadistill.autoinit.c2_frozen_comparison_inputs/v1"


class FrozenInputError(RuntimeError):
    """The frozen candidate record cannot be used as a comparison input."""


@dataclass(frozen=True)
class FrozenCandidate:
    """One measured candidate, carrying exactly what the comparison reads.

    Deliberately not an `InitializationState`: that type owns a lifecycle --
    planned, materialized, validated, measured -- and this object has no
    lifecycle to own. It is a measurement that already happened, and it must not
    be advanceable, re-materializable or re-measurable.
    """

    state_id: str
    path_label: str
    artifact_digest: str
    num_parameters: int | None
    evaluation: StateEvaluation
    front: int | None = None
    lineage: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenInputError(message)


def load_record(path: str | Path) -> dict[str, Any]:
    """Read and verify the frozen-inputs record's own hash."""
    record = json.loads(Path(path).read_text())
    _require(record.get("schema") == SCHEMA,
             f"{path} is {record.get('schema')!r}, not {SCHEMA!r}")
    rule = record.get("extraction_rule")
    _require(rule in SUPPORTED_EXTRACTION_RULES,
             f"{path} was extracted by {rule!r}, which this loader does not know. "
             "A record written by a different rule may not mean what this one means.")
    stated = record.get("self_sha256")
    recomputed = sha256_json({k: v for k, v in record.items() if k != "self_sha256"})
    _require(stated == recomputed,
             f"{path} does not match its own self_sha256; it has been edited since "
             "the search's evidence was frozen")
    return record


def load_frozen_candidates(
    path: str | Path, *,
    expect_suite_hash: str | None = None,
    expect_policy_hash: str | None = None,
) -> tuple[FrozenCandidate, ...]:
    """The five measured candidates, in the order the selection committed.

    `expect_suite_hash` and `expect_policy_hash` are the caller's own frozen
    identities. They are optional arguments and not defaults read from here,
    because this module must not become a second place that declares what the
    experiment's suite is -- but a caller that supplies them gets the comparison
    refused rather than computed across two suites.
    """
    record = load_record(path)

    if expect_suite_hash is not None:
        actual = (record.get("suite") or {}).get("hash")
        _require(actual == expect_suite_hash,
                 f"the frozen candidates were measured on suite {actual} and this "
                 f"session's suite is {expect_suite_hash}. Values from two suites "
                 "are not comparable.")
    if expect_policy_hash is not None:
        actual = (record.get("policy") or {}).get("hash")
        _require(actual == expect_policy_hash,
                 f"the frozen candidates were ranked under policy {actual} and this "
                 f"session's policy is {expect_policy_hash}.")

    candidates = []
    for entry in record["candidates_in_committed_order"]:
        evaluation = _evaluation_of(entry)
        _require(evaluation.artifact_digest == entry["identity"]["artifact_digest"],
                 f"{entry['state_id']}: the stored evaluation measured "
                 f"{evaluation.artifact_digest} and the entry's identity is "
                 f"{entry['identity']['artifact_digest']}")
        candidates.append(FrozenCandidate(
            state_id=entry["state_id"],
            path_label=entry["path_label"],
            artifact_digest=entry["identity"]["artifact_digest"],
            num_parameters=entry["identity"].get("num_parameters"),
            evaluation=evaluation,
            front=entry.get("front"),
            lineage=entry.get("lineage"),
        ))
    _require(bool(candidates), f"{path} carries no candidates")
    return tuple(candidates)


def _evaluation_of(entry: Mapping[str, Any]) -> StateEvaluation:
    """Field-for-field, from the stored mapping. No metric is recomputed."""
    stored = dict(entry["evaluation"])
    try:
        return StateEvaluation(**stored)
    except TypeError as exc:
        raise FrozenInputError(
            f"{entry.get('state_id')}: the stored evaluation does not match "
            f"StateEvaluation's fields ({exc}). The freeze stores the complete "
            "as_dict() precisely so this reconstruction is exact.") from exc


def numerically_sensitive_pairs(
    baseline_values: Mapping[str, float],
    candidates: tuple[FrozenCandidate, ...],
    *, objectives: tuple[str, ...], threshold: float,
) -> list[dict[str, Any]]:
    """Where a measured B lands close enough to a candidate to disclose it.

    The protocol registers this BEFORE B exists: the ranking rule and its
    epsilon are untouched, and this only obliges the record to say when a front
    assignment rests on a margin smaller than the candidate side's own tightest
    observed gap. No repeat measurement exists anywhere in this project, so the
    numerical reproducibility of a single measurement across sessions is
    unmeasured -- and a verdict that depends on a margin below that scale should
    say so rather than read as decisive.
    """
    flagged = []
    for candidate in candidates:
        for key in objectives:
            margin = abs(float(baseline_values[key])
                         - float(candidate.evaluation.values[key]))
            if margin <= threshold:
                flagged.append({
                    "state_id": candidate.state_id,
                    "objective": key,
                    "margin": margin,
                    "threshold": threshold,
                    "why": ("this margin is at or below the tightest gap observed "
                            "among the frozen candidates, and cross-session "
                            "numerical reproducibility is unmeasured"),
                })
    return flagged
