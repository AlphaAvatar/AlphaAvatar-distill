"""When a searched initialization turns out to BE an imported one.

Phase-B attempt 5 completed its joint P=2 search and then died in Stage 2 on
``duplicate seeds in [20260726, 20260726]``. The pooling guard was right. What
was wrong was the candidate universe it was asked to pool.

The preregistration froze "5 searched + 2 imported + 1 control = **8 distinct**".
Two of the five searched leaves turned out to be **byte-identical** to two
retained Phase-A finalists: same content-derived state id, same re-derived
artifact digest, same bytes. State ids are content-derived, so a larger search
that rediscovers the same composition from the same root produces the same
initialization — which is the determinism the project wants, and which the
"8 distinct" assumption cannot express.

So the universe collapses to **6 distinct behavioural candidates**, and the two
collapsed ones each carry two *roles* — searched, and imported-as-evidence-alias
— while remaining **one** statistical observation per seed.

Two rules make this safe rather than convenient:

**Collapse only on materialized identity.** Both the canonical state identity and
the re-derived artifact digest must agree. Names, id prefixes, composition
descriptions, `sa` scores and behavioural outcomes are never inputs. A collapse
that consulted a score would let a measurement decide who counts as whom.

**Same id, different bytes is a refusal, not a merge.** If the state ids agree
and the digests do not, something is wrong with an identity this project depends
on, and the correct response is to stop.

The searched role is primary for an initialization the search rediscovered: it is
what the P=2 search produced and what the Top-5 admitted. The imported role
survives as provenance and as the source of citable historical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..infrastructure.manifest import sha256_json


class IdentityCollapseError(RuntimeError):
    """Two candidates share a state identity but not their bytes."""


#: Precedence when an initialization carries more than one role. The searched
#: role wins because it is what the search produced; the rest are provenance.
ROLE_PRECEDENCE = ("searched", "imported_finalist", "control")


@dataclass(frozen=True)
class CollapsedCandidate:
    """One distinct materialized initialization, with every role it plays."""

    state_id: str
    artifact_digest: str
    primary_role: str
    roles: tuple[str, ...]
    checkpoint_path: str | None = None
    aliases: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def is_collapsed(self) -> bool:
        return len(self.roles) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "artifact_digest": self.artifact_digest,
            "primary_role": self.primary_role,
            "roles": list(self.roles),
            "is_collapsed": self.is_collapsed,
            "checkpoint_path": self.checkpoint_path,
            "aliases": list(self.aliases),
            "detail": dict(self.detail),
        }


def _role_key(role: str) -> int:
    return ROLE_PRECEDENCE.index(role) if role in ROLE_PRECEDENCE else len(ROLE_PRECEDENCE)


def collapse(entries: list[dict[str, Any]]) -> list[CollapsedCandidate]:
    """Union candidates by (state_id, artifact_digest). Deterministic.

    `entries` are dicts carrying at least `state_id`, `artifact_digest` and
    `role`. Output order is by state id, so the same inputs always produce the
    same universe regardless of how the caller assembled them.

    Raises `IdentityCollapseError` when two entries share a state id and differ
    in artifact digest — the fail-closed case.
    """
    by_state: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        for required in ("state_id", "artifact_digest", "role"):
            if not entry.get(required):
                raise IdentityCollapseError(
                    f"candidate entry is missing {required!r}: {entry}. Collapse "
                    "decides on materialized identity and cannot infer it.")
        by_state.setdefault(entry["state_id"], []).append(entry)

    out: list[CollapsedCandidate] = []
    for state_id, group in sorted(by_state.items()):
        digests = {e["artifact_digest"] for e in group}
        if len(digests) > 1:
            raise IdentityCollapseError(
                f"{state_id} appears {len(group)} times with {len(digests)} distinct "
                f"artifact digests {sorted(d[:12] for d in digests)}. A content-derived "
                "state id that disagrees about its own bytes is a broken identity, not "
                "a duplicate to merge. Refusing.")
        roles = tuple(sorted({e["role"] for e in group}, key=_role_key))
        paths = [e.get("checkpoint_path") for e in group if e.get("checkpoint_path")]
        out.append(CollapsedCandidate(
            state_id=state_id,
            artifact_digest=digests.pop(),
            primary_role=roles[0],
            roles=roles,
            checkpoint_path=paths[0] if paths else None,
            aliases=tuple(sorted({e["alias"] for e in group if e.get("alias")})),
            detail={"n_entries": len(group)},
        ))
    return out


def observations_per_seed(candidates: list[CollapsedCandidate],
                          records: list[dict[str, Any]]) -> dict[str, list[dict]]:
    """One observation per (initialization, seed). The rule the guard enforces.

    A collapsed candidate may cite historical evidence under either role, but the
    statistics see **one** observation per seed. This drops exact duplicates —
    records agreeing on state id, seed *and* artifact digest — and refuses
    anything that disagrees, because two different measurements of one seed is a
    real conflict and not a bookkeeping artifact.
    """
    known = {c.state_id: c for c in candidates}
    out: dict[str, list[dict]] = {c.state_id: [] for c in candidates}
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        state_id, seed = record["state_id"], int(record["seed"])
        if state_id not in known:
            raise IdentityCollapseError(
                f"record for {state_id} is outside the collapsed universe")
        key = (state_id, seed)
        previous = seen.get(key)
        if previous is None:
            seen[key] = record
            out[state_id].append(record)
            continue
        a = previous.get("student_artifact_digest")
        b = record.get("student_artifact_digest")
        if a != b:
            raise IdentityCollapseError(
                f"{state_id} seed {seed} has two observations with different "
                f"artifact digests ({str(a)[:12]} vs {str(b)[:12]}); that is a "
                "conflict, not a duplicate role")
        # Same initialization, same seed, same bytes, cited twice under two
        # roles. One observation.
    return out


def universe_identity(candidates: list[CollapsedCandidate]) -> str:
    """A hash over the collapsed universe, for binding into an amendment."""
    return sha256_json([{"state_id": c.state_id,
                         "artifact_digest": c.artifact_digest,
                         "roles": list(c.roles)} for c in candidates])
