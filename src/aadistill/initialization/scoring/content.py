"""The identity of the position metadata a scoring policy actually consumes.

**The collision this closes.** A frozen calibration mixture's content identity
(``mixture_content_sha256``) hashes ``item_id`` and the token ids, and nothing
else. A policy that reads positions — ``positions.supervised_target_v1`` reads
each item's ``assistant`` tag — therefore consumes metadata that no identity in
the repository covers. Two assets could carry

* the same item ids,
* the same token ids,
* the same calibration profile hash,
* the same mixture content hash,
* the same position-policy hash,

and **different supervised masks**, and so produce different DEPTH, FFN, WIDTH
and ATTENTION decisions under one scientific path identity. That is a scientific
identity collision, not a bookkeeping gap.

**What is bound, and why each term.** Per item, in the asset's own order:

``item_id``
    which item this is.
``token content``
    the same digest rule the mixture identity already uses, so the two are
    recognisably siblings and a token change moves both.
``domain`` and ``subtype``
    because the reductions above the per-item level aggregate by subtype and
    then by domain. Two assets with identical tokens and identical masks but
    swapped subtypes give different domain-balanced scores, so the labels are
    part of what the objective is computed over.
``the position metadata the POLICY declares it reads``
    asked of the policy rather than written down here. ``reads_tags`` is the
    policy's own declaration, so a future policy reading ``final_answer``,
    ``think_close`` or a tag that does not exist yet binds the right thing with
    no edit in this file. ``assistant`` appears nowhere below.

**Item ORDER is content.** The lines are emitted in the asset's order, because
``original_order_v1`` groups consecutive items and the grouping is part of the
execution protocol. Sorting here would make two differently ordered assets
identical to an identity that the batcher can tell apart.

**Nothing arbitrary is hashed.** Not the JSON, not the key order, not the
whitespace, not fields the policy does not read. The input is a canonical line
per item built from the five terms above, which is why re-serializing an asset
cannot move the identity and adding a supervised position must.

**The degenerate case is the compatibility guarantee.**
``positions.all_v1`` declares ``reads_tags = ()``, so under the incumbent policy
this reduces to ids, tokens and labels — there is no mask for it to collide on,
and callers omit the field entirely so that every committed state id stays
derivable. See ``policy_config``'s omission, which follows the same rule.

**One pre-existing gap this does NOT close**, stated so it is not mistaken for
closed: ``domain`` and ``subtype`` affect the incumbent policy's aggregation too
and are not bound by ``mixture_content_sha256`` either. Binding them for the
incumbent would move every committed state id, so it is left alone; the
protection there is that a profile resolves to exactly one pinned content hash,
and a relabelled asset would be a new profile version.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from aadistill.initialization.scoring.positions import (
    ALL_POSITIONS_V1,
    PREDICTION_AXIS,
    ScoringPositionError,
    ScoringPositionPolicy,
    _item_view,
)

SCHEMA = "aadistill.autoinit.scoring_content/v1"

#: Field separator inside a line, and the line separator. Control characters
#: rather than punctuation, so a value containing a colon or a comma cannot
#: shift the field boundaries and make two different assets hash alike.
_FS = "\x1f"
_RS = "\n"

#: Digest width for the per-item token and per-tag components. Sixteen hex
#: characters, matching `mixture_content_sha256`'s `sha_ids`, so the two
#: identities are visibly the same family of construction.
_WIDTH = 16


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:_WIDTH]


def _token_component(item: Any) -> str:
    """The token digest, by the rule the mixture identity already uses.

    `mixture_content_sha256` hashes the comma-joined raw `ids`. Reproducing that
    rule rather than inventing a second one means a token change moves both
    identities, and a reader comparing them is comparing like with like.
    """
    raw = item.get("ids") if isinstance(item, Mapping) else None
    if raw is not None:
        return _digest(",".join(str(int(t)) for t in raw))
    ids = (item.get("input_ids") if isinstance(item, Mapping)
           else getattr(item, "input_ids", None))
    if ids is None:
        raise ScoringPositionError(
            "an item needs 'ids' or 'input_ids' before its token content can "
            "be identified")
    return _digest(",".join(str(int(t)) for t in ids.reshape(-1).tolist()))


def _position_component(policy: ScoringPositionPolicy,
                        tags: Mapping[str, Any], n_predictions: int) -> str:
    """The position metadata this policy reads, canonically.

    One entry per declared tag, in sorted tag order so the policy's declaration
    order cannot move the identity, and each entry a digest of the tag's sorted
    prediction-position indices. A tag the asset does not carry is recorded as
    absent rather than skipped: "no `assistant` tag" and "an empty `assistant`
    tag" are different assets and the supervised-target policy treats them
    differently, so the identity has to tell them apart.
    """
    parts = []
    for name in sorted(policy.reads_tags):
        mask = tags.get(name)
        if mask is None:
            parts.append(f"{name}=absent")
            continue
        indices = sorted(int(i) for i in mask.nonzero().flatten().tolist())
        parts.append(f"{name}={len(indices)}:{_digest(','.join(map(str, indices)))}")
    return ";".join(parts) + f"|n={n_predictions}"


def scoring_content_identity(items: Sequence[Any],
                             policy: ScoringPositionPolicy) -> str:
    """The identity of what ``policy`` will read from ``items``.

    Accepts either calibration-mixture mappings or
    :class:`~aadistill.initialization.specs.metrics.SuiteItem` objects; the
    position metadata is normalized by the same `_item_view` a policy uses, so
    the identity covers exactly what the policy will see.
    """
    if not items:
        raise ScoringPositionError(
            "refusing to identify the scoring content of an empty item set: a "
            "vacuous identity would make two different assets agree")
    lines = []
    for index, item in enumerate(items):
        n_tokens, tags, meta = _item_view(item)
        item_id = meta.get("item_id")
        if not item_id:
            raise ScoringPositionError(
                f"item {index} carries no `item_id`, so its scoring content "
                "cannot be attributed; an unidentified item would let two "
                "assets differing in one item hash alike")
        lines.append(_FS.join((
            str(item_id),
            _token_component(item),
            str(meta.get("domain") or ""),
            str(meta.get("subtype") or ""),
            _position_component(policy, tags, n_tokens - 1),
        )))
    return hashlib.sha256(
        (SCHEMA + _RS + _RS.join(lines) + _RS).encode()).hexdigest()


def scoring_content_report(items: Sequence[Any],
                           policy: ScoringPositionPolicy) -> dict[str, Any]:
    """The identity plus the totals a reader checks it against.

    Evidence, not identity: a record carrying only a hash gives a reader nothing
    to verify it with, and these three numbers are derivable from the asset and
    the policy id alone.
    """
    from aadistill.initialization.scoring.positions import weights_for_items

    weights = weights_for_items(items, policy, axis=PREDICTION_AXIS)
    return {
        "schema": SCHEMA,
        "scoring_content_sha256": scoring_content_identity(items, policy),
        "position_policy": policy.qualified_id,
        "position_policy_hash": policy.policy_hash,
        "reads_tags": sorted(policy.reads_tags),
        "n_items": len(items),
        "prediction_positions": sum(w.n_positions for w in weights),
        "active_prediction_positions": sum(w.n_active for w in weights),
    }


#: The config key a hashed operator config carries the identity under, beside
#: `position_policy` and `position_policy_hash`.
CONTENT_CONFIG_KEY = "scoring_content_sha256"


def scoring_content_config(items: Sequence[Any],
                           policy: ScoringPositionPolicy) -> dict[str, str]:
    """``{CONTENT_CONFIG_KEY: ...}``, or ``{}`` under the incumbent policy.

    Empty at :data:`ALL_POSITIONS_V1` for the same reason `policy_config` is
    empty there: every committed state hashed an operator config without these
    keys, and a historical state id must stay derivable from live code. It is
    also sound rather than merely convenient — the incumbent reads no position
    metadata, so it has no mask content to collide on.
    """
    if policy.policy_hash == ALL_POSITIONS_V1.policy_hash:
        return {}
    return {CONTENT_CONFIG_KEY: scoring_content_identity(items, policy)}
