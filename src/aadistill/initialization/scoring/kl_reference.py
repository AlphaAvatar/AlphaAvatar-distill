"""Which model a KL-scored selection measures its candidates against.

A KL-scored operator compares each candidate child with a REFERENCE
distribution. Historically that reference is the intact PARENT it was handed
(`parent_state`): DEPTH measured how far bypassing a layer subset moves the
model it is compressing. The alternative is the lineage ROOT -- the frozen
original model, before any compression -- read from a stored root reference
(`scoring.reference_store`), never re-forwarded.

The choice is OPERATOR CONFIGURATION, declared per invocation and therefore
part of `config_hash` and of every state id built on it:

* the historical default is ``parent_state`` and is NEVER WRITTEN into a
  config. An absent key means the parent, so every recorded config, hash and
  state id stays exactly what it was. Spelling the default explicitly is
  refused: two configs that compute the same thing under two hashes would
  manufacture distinct identities for one computation.
* ``root_reference/v1`` is declared under the key ``kl_reference`` and binds
  the stored root reference: its Top-K support, its tail, every real
  prediction position (the consumer's position policy selects among them).

A scoring objective that declares its own reference (`ScoringObjective.
kl_reference`) owns that choice for every operator that consumes it, so a
declaration here BESIDE a consumed objective is refused as a conflict rather
than resolved by precedence. Only implementations that declare
`accepts_kl_reference` may be handed one.

Nothing here names a model, an experiment or a size.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The operator-config key. Absent means the historical parent reference.
KL_REFERENCE_KEY = "kl_reference"
#: The historical reference: the intact parent the operator was handed.
#: Never declared -- see the module docstring.
PARENT_STATE = "parent_state"
#: The lineage root's stored reference. The same string the scoring
#: objectives use for their own root-referenced KL.
ROOT_REFERENCE_V1 = "root_reference/v1"
#: What a config may declare. The default is deliberately absent.
DECLARABLE = frozenset({ROOT_REFERENCE_V1})


class KLReferenceError(ValueError):
    """A KL-reference declaration that cannot be honoured as written."""


def declared_kl_reference(config: Mapping[str, Any] | None) -> str:
    """The reference a config declares: ``PARENT_STATE`` when the key is absent.

    Refuses an unknown value and the explicitly spelled default.
    """
    if not config or KL_REFERENCE_KEY not in config:
        return PARENT_STATE
    value = config[KL_REFERENCE_KEY]
    if value == PARENT_STATE:
        raise KLReferenceError(
            f"{KL_REFERENCE_KEY}={PARENT_STATE!r} is the historical default "
            "and is expressed by OMITTING the key; spelling it out would give "
            "the same computation a second config hash")
    if value not in DECLARABLE:
        raise KLReferenceError(
            f"{KL_REFERENCE_KEY}={value!r} is not a known reference; "
            f"declarable: {sorted(DECLARABLE)}")
    return str(value)


def validate_kl_reference(implementation: Any,
                          config: Mapping[str, Any] | None, *,
                          objective_consumed: bool) -> str:
    """Check a declaration against the implementation and the objective.

    Returns the declared reference (``PARENT_STATE`` when none). Raises
    `KLReferenceError` when the implementation does not accept one, or when a
    consumed scoring objective already decides the reference -- two
    declarations of one fact are refused, never ranked.
    """
    reference = declared_kl_reference(config)
    if reference == PARENT_STATE:
        return reference
    impl_id = getattr(implementation, "impl_id", implementation)
    if not getattr(implementation, "accepts_kl_reference", False):
        raise KLReferenceError(
            f"{impl_id} does not accept a {KL_REFERENCE_KEY} declaration; its "
            "selection reads no KL reference this key could move")
    if objective_consumed:
        raise KLReferenceError(
            f"{impl_id} declares {KL_REFERENCE_KEY}={reference!r} and also "
            "consumes a scoring objective, which declares its own KL "
            "reference. Two declarations of one reference are refused, not "
            "resolved by precedence")
    return reference


def requests_root_reference(implementation: Any,
                            config: Mapping[str, Any] | None) -> bool:
    """Whether a config's declaration needs the stored ROOT reference."""
    if not getattr(implementation, "accepts_kl_reference", False):
        return False
    return declared_kl_reference(config) == ROOT_REFERENCE_V1


__all__ = ["DECLARABLE", "KLReferenceError", "KL_REFERENCE_KEY", "PARENT_STATE",
           "ROOT_REFERENCE_V1", "declared_kl_reference",
           "requests_root_reference", "validate_kl_reference"]
