"""One identity for "was this measurement taken under my protocol?".

The resume path had grown two independent comparisons — the suite hash, then the
position-policy hash — and the list was going to keep growing: the reduction's
chunk boundaries matter (measured at ~9e-8 relative), the reference strategy
matters, the scoring content matters, and the numerical execution matters. A
restore that asks five separate questions is a restore where the sixth gets
forgotten, and the one that gets forgotten is the one that silently admits a
measurement of a different quantity.

So the terms are combined once, here, and the consumer asks one question.

``measurement_protocol_id`` binds:

``suite_structural_identity``
    which SUITE: its id, version, domains, sub-types and critical tags, which
    are what the required metrics and therefore the beam ranking read. This is
    ``StateEvalSuite.suite_hash``, and it is structural by this project's
    design — many committed records pin it, and folding content into it would
    reinterpret every one of them.
``suite_content_identity``
    which PROMPTS. Bound separately and required, because the structural hash
    cannot see them: two suites with one declared shape and different content
    are different measurements. This is the term that closes the hole, and it
    is closed HERE rather than in the structural hash precisely because nothing
    historical pins this identity.
``scoring_content_identity``
    which POSITIONS of those prompts, as content rather than as a policy name.
    A suite whose tokens are unchanged and whose supervised masks moved is a
    different measurement, and nothing else in the chain can see that.
``position_policy_hash``
    the rule that produced those positions. Redundant with the content
    identity for a given asset and not redundant in general: two policies can
    agree on one asset and disagree on the next, and a record should say which
    rule it ran rather than only what that rule happened to select.
``reduction_semantics``
    what the numbers MEAN — the chunk boundaries, the reference strategy, and
    the aggregation rule. Chunking is not a formality here: this project
    measured it mattering at ~9e-8 relative, which is the same order as the
    drift budget the evaluator is certified against.
``execution_fingerprint``
    how the forwards were issued. A batched state evaluation and an unbatched
    one compute the same estimand through different kernel shapes.

**A historical record is not reinterpreted.** A measurement taken before this
identity existed carries no ``measurement_protocol_id``, and
:func:`measurement_is_comparable` says so explicitly rather than computing what
its id "would have been" — which would require asserting a reduction and an
execution nobody recorded. Such a record is admissible only to a run using the
historical default semantics, and the function reports that as its reason.

Nothing here knows about an experiment, a stage, a family or a suite's subject.
It takes identities and returns one.

(Named ``protocol_identity`` rather than ``measurement``: the core-boundary
inventory treats ``measurement`` in a module name as an experiment-instance
marker, and that tripwire protects a real boundary. Renaming this file was the
cheaper correction.)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aadistill.infrastructure.manifest import sha256_json

SCHEMA = "aadistill.autoinit.measurement_protocol/v1"

#: The field a measurement carries its protocol identity under.
PROTOCOL_FIELD = "measurement_protocol_id"

#: What an UNBOUND suite content identity contributes. A literal rather than a
#: rejection: a toy or ad-hoc suite assembled in memory has no frozen content
#: record to bind, and refusing to identify it would make the protocol id
#: unavailable exactly where a test needs it most. Recorded explicitly so
#: "nobody bound the content" is itself part of the identity and cannot be
#: mistaken for a content-bound measurement.
UNBOUND_SUITE_CONTENT = "unbound"

#: What an UNDECLARED numerical environment contributes. A literal rather than
#: an omission, so that "nobody stated the numerics" is itself part of the
#: identity: a run that declares them and one that does not are measuring under
#: different stated conditions, and should not resume each other.
UNDECLARED_EXECUTION = "undeclared"

#: The aggregation rule the state evaluator implements. A version string, not a
#: description: the rule is two-level unweighted (equal sub-type mean inside an
#: equal domain mean) and lives in `domain_balanced_score`. If that rule ever
#: changes, this string changes with it and every protocol id moves — which is
#: the intended behaviour, because the metric would mean something else.
AGGREGATION_RULE = "equal_subtype_then_equal_domain/v1"


@dataclass(frozen=True)
class ReductionSemantics:
    """What the reduced numbers mean, as opposed to which inputs produced them.

    Deliberately small. These are the knobs that change the VALUE of a metric
    computed over fixed inputs, which is a different list from the knobs that
    change which inputs there are.
    """

    chunk: int
    reference_strategy: str
    aggregation_rule: str = AGGREGATION_RULE
    #: WHICH VOCABULARY ENTRIES THE DIVERGENCE IS REDUCED OVER. Optional, and
    #: ABSENT from `as_dict` when it is `None` or full-vocabulary — which is the
    #: whole design. `reference_topk_tail_v1` is not mathematically identical to
    #: full-vocabulary KL, so a measurement taken under it must not be comparable
    #: with one taken under the other; but 785 committed records cite protocol
    #: ids computed before this field existed, and a key added unconditionally —
    #: even carrying `"full_vocab_v1"` — would change every one of their hashes.
    #:
    #: So the serialization is VERSIONED BY ABSENCE: the historical contract
    #: serializes exactly as it did before this field, and only the new contract
    #: adds a key. See `support.DistributionSupport.as_dict`.
    distribution_support: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {"chunk": int(self.chunk),
               "reference_strategy": str(self.reference_strategy),
               "aggregation_rule": str(self.aggregation_rule)}
        support = self.distribution_support
        if support is not None and not getattr(support, "is_full_vocab", False):
            out["distribution_support"] = support.as_dict()
        return out


def measurement_protocol_id(*, suite_structural_identity: str,
                            suite_content_identity: str,
                            scoring_content_identity: str,
                            position_policy_hash: str,
                            reduction: ReductionSemantics,
                            execution_fingerprint: str) -> str:
    """One id for the protocol a state measurement was taken under.

    Every term is required. An absent term would make two measurements of
    different quantities agree, which is the failure the whole type exists to
    prevent — so there are no defaults to forget.
    """
    missing = [name for name, value in (
        ("suite_structural_identity", suite_structural_identity),
        ("suite_content_identity", suite_content_identity),
        ("scoring_content_identity", scoring_content_identity),
        ("position_policy_hash", position_policy_hash),
        ("execution_fingerprint", execution_fingerprint),
    ) if not value]
    if missing:
        raise ValueError(
            f"a measurement protocol identity needs {missing}; an absent term "
            "would let two measurements of different quantities agree")
    return sha256_json({
        "schema": SCHEMA,
        "suite_structural_identity": suite_structural_identity,
        "suite_content_identity": suite_content_identity,
        "scoring_content_identity": scoring_content_identity,
        "position_policy_hash": position_policy_hash,
        "reduction": reduction.as_dict(),
        "execution_fingerprint": execution_fingerprint,
    })[:32]


def measurement_is_comparable(record: Mapping[str, Any] | None, *,
                              protocol_id: str | None,
                              historical_suite_hash: str | None = None,
                              historical_policy_hash: str | None = None,
                              ) -> tuple[bool, str]:
    """``(ok, why)`` — may this recorded measurement be adopted by this run?

    ONE call site, two rules, and which one applies is decided by the record
    rather than by the caller:

    * the record carries a protocol id → compare protocol ids. Everything the
      old separate comparisons covered is inside both.
    * the record carries none → it predates the identity. It is admissible only
      to a run whose own semantics are the historical default, judged by the two
      fields such a record does carry. Its id is NOT reconstructed: doing so
      would mean asserting a reduction and an execution nobody wrote down.

    Returning a reason rather than a bare bool is the point. A resume that
    declines silently is indistinguishable from a resume that found nothing, and
    this project has spent whole rounds on that difference.
    """
    if record is None:
        return False, "no record"
    detail = dict(record.get("detail") or {})
    recorded = detail.get(PROTOCOL_FIELD)

    if recorded:
        if protocol_id is None:
            return False, (
                "the record was measured under protocol "
                f"{str(recorded)[:12]} and this run declares none, so it "
                "cannot show the record describes its own measurement")
        if str(recorded) != protocol_id:
            return False, (
                f"measured under protocol {str(recorded)[:12]}, this run is "
                f"{protocol_id[:12]}")
        return True, "same measurement protocol"

    #: No protocol id: a historical record.
    if protocol_id is not None:
        return False, (
            "the record predates measurement-protocol identity and this run "
            f"declares one ({protocol_id[:12]}); a record whose reduction and "
            "execution were never written down cannot be shown comparable to "
            "a declared protocol, and reconstructing its id would be asserting "
            "what it ran under")
    if historical_suite_hash is not None \
            and record.get("suite_hash") != historical_suite_hash:
        return False, (
            f"historical record's suite {str(record.get('suite_hash'))[:12]} "
            f"is not this run's {historical_suite_hash[:12]}")
    if historical_policy_hash is not None:
        recorded_policy = detail.get("position_policy_hash")
        if recorded_policy is not None \
                and str(recorded_policy) != historical_policy_hash:
            return False, (
                "historical record scored position policy "
                f"{str(recorded_policy)[:12]}, this run "
                f"{historical_policy_hash[:12]}")
    return True, "historical record under historical default semantics"
