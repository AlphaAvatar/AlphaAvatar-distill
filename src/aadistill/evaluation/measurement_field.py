"""Does a set of measurements form ONE admissible field?

A paired estimate pools several measurements and reports one interval. That is
only a quantity if every measurement was made the same way. Six valid row files
measured under different evaluation protocols are six measurements of different
things, and pooling them produces a number with no estimand.

This is the admission rule for that question, and it FAILS CLOSED. It sits
beside `paired_stats`, which does the arithmetic it guards, and it delegates the
generation-protocol comparison to `generation_compat` -- the rule that already
owns whether two runtimes are comparable -- rather than inventing a second
interpretation of "same protocol".

Nothing here names an experiment, an arm, a battery or a stage: the keys are
opaque and the field's name is a parameter, so every stage inherits it.

The failure it was written for: C2's behavioural confirmation field carried a
uniform battery, scoring contract and metric contract, and THREE distinct
generation protocol fingerprints -- four probes on one, the incumbent's third
seed on a second, the candidate's third seed on a third. The paired difference
at that seed was therefore computed across two protocols: a confound inside the
pair, on the seed with the largest magnitude. Nothing refused it, the interval
was reported, and the stage had to be closed without a promotion verdict.
"""
from __future__ import annotations

import json
from typing import Any, Mapping


class MeasurementFieldError(Exception):
    """The measurements do not form one admissible field."""


#: Compared by exact identity. A probe scored against a different battery,
#: scoring contract or metric contract is measuring something else, and no
#: runtime-comparability argument can rescue that.
PROTOCOL_IDENTITY_FIELDS = ("battery", "scoring_contract", "metric_contract")

#: Compared through the project's OWN rule, not by equality, and deliberately
#: separate from the fields above. `generation_compat` v2 demotes the NVIDIA
#: driver patch to recorded-not-material because the field named `image_digest`
#: is really `imageName@driver` and the provider assigns whatever host is free,
#: so exact equality over a fingerprint containing it is a host lottery. That
#: rule needs the expanded protocol and runtime blocks; a fingerprint alone
#: cannot be demoted, only compared.
GENERATION_IDENTITY_FIELD = "generation_protocol_fingerprint"


def _stable(value: Any) -> str:
    return (json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list)) else str(value))


def assert_one_measurement_protocol(
        protocols: Mapping[Any, Mapping[str, Any]], *,
        context: str = "measurement field") -> dict[str, Any]:
    """Refuse unless every measurement establishes ONE compatible protocol.

    FAILS CLOSED, in both directions that matter:

    * identities that are present and DIFFER -> refused;
    * identities that are ABSENT, or a generation protocol that differs and
      cannot be judged under `generation_compat` v2 because the expanded
      protocol and runtime blocks were never recorded -> also refused, because
      "not shown to be comparable" is not "comparable".

    `protocols` maps each measurement's key to what it recorded about how it was
    measured. An entry may additionally carry `protocol` and `runtime` blocks,
    in which case the generation protocol is compared through
    `require_comparable` instead of by fingerprint equality.

    Returns the comparison, so a caller can record what was equal and what was
    merely recorded.
    """
    from aadistill.initialization.planning.generation_compat import (
        ComparabilityError, comparable_generation_identity, require_comparable,
    )

    if not protocols:
        raise MeasurementFieldError(
            f"the {context} declares no measurement protocol for any entry. A "
            "paired interval over measurements whose protocols are unknown has "
            "no estimand; this refuses rather than producing a number.")

    report: dict[str, Any] = {"probes": {str(k): {} for k in protocols},
                              "uniform": {}, "compared_by": {}}

    missing: dict[str, list[str]] = {}
    for key, rec in protocols.items():
        absent = [f for f in (*PROTOCOL_IDENTITY_FIELDS, GENERATION_IDENTITY_FIELD)
                  if rec.get(f) in (None, "", {}, [])]
        if absent:
            missing[str(key)] = absent
    if missing:
        raise MeasurementFieldError(
            f"the {context} cannot be shown to share one measurement protocol: "
            f"{missing} record no value for those identities. Absent is not "
            "equal; this refuses rather than assuming they matched.")

    for field in PROTOCOL_IDENTITY_FIELDS:
        seen = {_stable(rec.get(field)) for rec in protocols.values()}
        report["uniform"][field] = len(seen) == 1
        report["compared_by"][field] = "exact identity"
        if len(seen) != 1:
            raise MeasurementFieldError(
                f"the {context} spans {len(seen)} distinct {field} identities. "
                "Measurements against different batteries, scoring contracts "
                "or metric contracts are measurements of different things: "
                f"{sorted(s[:80] for s in seen)}")

    #: The generation protocol, through the rule that owns it.
    fingerprints = {_stable(r.get(GENERATION_IDENTITY_FIELD))
                    for r in protocols.values()}
    report["generation_fingerprints"] = sorted(f[:64] for f in fingerprints)
    if len(fingerprints) == 1:
        report["uniform"][GENERATION_IDENTITY_FIELD] = True
        report["compared_by"][GENERATION_IDENTITY_FIELD] = "exact identity"
        return report

    report["uniform"][GENERATION_IDENTITY_FIELD] = False
    expanded = {k: r for k, r in protocols.items()
                if r.get("protocol") and r.get("runtime")}
    if len(expanded) != len(protocols):
        raise MeasurementFieldError(
            f"the {context} spans {len(fingerprints)} distinct generation "
            f"protocol fingerprints, and {len(protocols) - len(expanded)} of "
            f"{len(protocols)} entries did not record the expanded protocol "
            "and runtime blocks that `generation_compat` v2 needs to judge "
            "comparability. That rule demotes the NVIDIA driver patch to "
            "recorded-not-material, so differing fingerprints MIGHT be "
            "comparable -- but it cannot be applied to a fingerprint alone. "
            "Unjudgeable is refused, not assumed comparable. Fingerprints: "
            f"{report['generation_fingerprints']}")

    report["compared_by"][GENERATION_IDENTITY_FIELD] = (
        "generation_runtime_comparability@v2")
    keys = list(expanded)
    base = comparable_generation_identity(protocol=expanded[keys[0]]["protocol"],
                                          runtime=expanded[keys[0]]["runtime"])
    for other in keys[1:]:
        try:
            require_comparable(
                comparable_generation_identity(
                    protocol=expanded[other]["protocol"],
                    runtime=expanded[other]["runtime"]),
                base, context=f"{context}: {other} vs {keys[0]}")
        except ComparabilityError as exc:
            raise MeasurementFieldError(
                f"the {context} is not one measurement protocol: {exc}") from exc
    report["uniform"][GENERATION_IDENTITY_FIELD] = "comparable under v2"
    return report
