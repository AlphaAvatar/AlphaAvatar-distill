"""Calibration profiles — first-class, versioned, and chosen per operator.

A depth objective has been measured on one frozen domain-balanced mixture.
That was a fixed choice, and a fixed choice cannot be compared against the
alternative it excluded. Here the mixture is an argument: a path may run
``DEPTH(reasoning_heavy) -> ATTENTION(domain_balanced) -> FFN(reasoning_heavy) ->
WIDTH(stage0_current)`` and the manifest records which profile fed which
operator invocation.

Two rules are mechanical rather than editorial:

* a profile carries its ``DatasetRole``, and an operator may only be fed a
  profile whose role is ``OPERATOR_CALIBRATION`` (see ``datasets.py``);
* a profile that has not been built yet is ``materialized=False`` and has no
  content hash. It can be *represented*, priced and reasoned about, but
  ``resolve()`` refuses to hand it to a paid run. Inventing a hash for an
  unbuilt mixture is exactly the fake-record failure P7 forbids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aadistill.initialization.calibration.datasets import DatasetRole


class CalibrationError(RuntimeError):
    """A calibration profile cannot be used as requested."""


def mixture_content_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    """Token-level identity of a rendered mixture.

    Byte-for-byte the rule ``scripts/data/build_e8_calibration.py`` froze, so a
    profile that claims to be E8a's mixture can be *re-derived* rather than
    trusted: ``sha256`` over ``item_id:sha256(comma-joined ids)[:16]`` lines. A
    file hash would also move when the JSON is reformatted, which is a change the
    operator cannot see and must not be told about.
    """
    def sha_ids(ids) -> str:
        return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]

    missing = [i for i, item in enumerate(items)
               if "item_id" not in item or "ids" not in item]
    if missing:
        raise CalibrationError(
            f"{len(missing)} mixture items lack item_id/ids and cannot be hashed")
    return hashlib.sha256(
        "".join(f"{i['item_id']}:{sha_ids(i['ids'])}\n" for i in items).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CalibrationSource:
    """One dataset contributing to a mixture.

    ``revision`` is required even when it is the string ``"local"``: a source
    without a pinned revision cannot support reproduction, and leaving the
    field optional is how that gets forgotten.
    """

    dataset_id: str
    revision: str
    domain: str
    n_items: int
    content_sha256: str | None = None
    license: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "domain": self.domain,
            "n_items": self.n_items,
            "content_sha256": self.content_sha256,
            "license": self.license,
            "note": self.note,
        }


@dataclass(frozen=True)
class CalibrationProfile:
    """A versioned, hashable calibration mixture.

    The hash covers the *specification* — sources, revisions, weights, budget,
    sampling rule, seed and exclusions — not the sampled bytes. The sampled bytes
    are pinned separately by ``rendered_manifest_sha256`` when the profile has
    been materialized, so a spec change and a data change are distinguishable in
    a manifest instead of collapsing into one number.
    """

    profile_id: str
    version: int
    description: str
    sources: tuple[CalibrationSource, ...]
    domain_weights: Mapping[str, float]
    token_budget: int
    sample_rule: str
    seed: int
    role: DatasetRole = DatasetRole.OPERATOR_CALIBRATION
    materialized: bool = False
    items_path: str | None = None
    rendered_manifest_sha256: str | None = None
    #: Token-level mixture identity, ``sha256`` over ``item_id:sha_ids(ids)``
    #: lines — the same rule ``scripts/data/build_e8_calibration.py`` froze. It is
    #: **not** the items file's hash: JSON key order or whitespace can move the
    #: file hash without changing a single token the operator sees, and the
    #: mixture's identity is the tokens.
    content_sha256: str | None = None
    items_file_sha256: str | None = None
    leakage_exclusions: tuple[str, ...] = ()
    leakage_proof_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: True only for the ``NO_CALIBRATION`` sentinel. A profile that describes no
    #: mixture cannot satisfy the mixture invariants below, and exempting it is
    #: cheaper than making every invariant optional.
    is_null: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise CalibrationError("calibration profile needs an id")
        if self.is_null:
            return
        if self.version < 1:
            raise CalibrationError(f"{self.profile_id}: version must be >= 1")
        if not self.sources:
            raise CalibrationError(f"{self.profile_id}: declares no sources")
        if self.token_budget <= 0:
            raise CalibrationError(f"{self.profile_id}: token_budget must be positive")
        weights = dict(self.domain_weights)
        if not weights:
            raise CalibrationError(f"{self.profile_id}: declares no domain weights")
        if any(w < 0 for w in weights.values()):
            raise CalibrationError(f"{self.profile_id}: negative domain weight")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise CalibrationError(
                f"{self.profile_id}: domain weights sum to {sum(weights.values())!r}, not 1.0")
        declared = {s.domain for s in self.sources}
        if declared != set(weights):
            raise CalibrationError(
                f"{self.profile_id}: source domains {sorted(declared)} do not match "
                f"weighted domains {sorted(weights)}")
        if self.materialized and not self.content_sha256:
            raise CalibrationError(
                f"{self.profile_id}: declares materialized=True with no content_sha256")
        if not self.materialized and self.content_sha256:
            raise CalibrationError(
                f"{self.profile_id}: carries a content hash but is not materialized")

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "sources": [s.as_dict() for s in self.sources],
            "domain_weights": dict(sorted(self.domain_weights.items())),
            "token_budget": self.token_budget,
            "sample_rule": self.sample_rule,
            "seed": self.seed,
            "role": self.role.value,
            "leakage_exclusions": list(self.leakage_exclusions),
        }

    @property
    def profile_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def qualified_id(self) -> str:
        return f"{self.profile_id}@v{self.version}"

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.spec,
            "description": self.description,
            "materialized": self.materialized,
            "items_path": self.items_path,
            "content_sha256": self.content_sha256,
            "items_file_sha256": self.items_file_sha256,
            "rendered_manifest_sha256": self.rendered_manifest_sha256,
            "leakage_proof_path": self.leakage_proof_path,
            "profile_hash": self.profile_hash,
            "metadata": dict(self.metadata),
        }

    @property
    def is_no_calibration(self) -> bool:
        return self.is_null

    def resolve(self, repo_root: str | Path = ".") -> list[dict[str, Any]]:
        """Load the sampled items, or raise.

        Deliberately strict. A search that silently ran on an empty or
        unmaterialized mixture would still produce a ranking, and the ranking
        would look like evidence.
        """
        if not self.materialized:
            raise CalibrationError(
                f"{self.qualified_id} is declared but not built; build it and pin "
                "its content hash before using it in a run")
        if not self.items_path:
            raise CalibrationError(f"{self.qualified_id} is materialized but has no items_path")
        path = Path(repo_root) / self.items_path
        if not path.is_file():
            raise CalibrationError(f"{self.qualified_id}: {path} is missing")
        if self.items_file_sha256:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != self.items_file_sha256:
                raise CalibrationError(
                    f"{self.qualified_id}: {path} hashes to {digest} but the profile "
                    f"pins {self.items_file_sha256}")
        items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not items:
            raise CalibrationError(f"{self.qualified_id}: {path} contains no items")
        derived = mixture_content_sha256(items)
        if derived != self.content_sha256:
            raise CalibrationError(
                f"{self.qualified_id}: the loaded mixture's token content hashes to "
                f"{derived} but the profile pins {self.content_sha256}")
        return items


#: The single canonical stand-in for "this operator reads no calibration data".
#:
#: An implementation declaring ``CalibrationNeed.NONE`` — ``depth.positional_v0``
#: is a fixed positional heuristic, ``attention.weight_proxy_v0`` scores weights —
#: has no mechanism by which a mixture could change its output. Branching it over
#: the active profiles would manufacture states that are byte-identical, measure
#: identically, and occupy beam slots that distinct hypotheses should hold. It
#: would also inflate the search-space count by a factor that means nothing.
#:
#: So every such operator is invoked with this one sentinel, and because a state's
#: identity is derived from its steps' profile hashes, the resulting states are
#: automatically profile-independent rather than deduplicated after the fact.
NO_CALIBRATION = CalibrationProfile(
    profile_id="calib.none",
    version=1,
    description=("canonical sentinel for operators that consume no calibration "
                 "data; never resolves to items"),
    sources=(),
    domain_weights={},
    token_budget=0,
    sample_rule="none",
    seed=0,
    materialized=False,
    is_null=True,
)


class CalibrationNeed(Enum):
    """What an implementation must be fed to make its decision.

    Defined here rather than beside the operator base class because both
    `profile_for` and `consumes_calibration` below branch on it, and importing
    it from `operators` closed a `calibration <-> operators` cycle. The two
    functions were reaching for it through local imports inside their bodies,
    which hid the cycle from the import graph without removing it.
    """

    NONE = "none"
    ACTIVATION_STATS = "activation_stats"
    FORWARD_LOGITS = "forward_logits"


def profile_for(implementation, profile: CalibrationProfile) -> CalibrationProfile:
    """The profile an implementation is actually invoked with.

    One place decides this, so the search engine, the cost model and the
    branching estimate cannot disagree about how many states exist.
    """
    if implementation.calibration is CalibrationNeed.NONE:
        return NO_CALIBRATION
    return profile


def consumes_calibration(implementation) -> bool:
    return implementation.calibration is not CalibrationNeed.NONE


# --- profile registry ------------------------------------------------------

_PROFILES: dict[str, CalibrationProfile] = {}


def register_profile(profile: CalibrationProfile, *, replace: bool = False) -> CalibrationProfile:
    key = profile.qualified_id
    existing = _PROFILES.get(key)
    if existing is not None and not replace:
        if existing.profile_hash != profile.profile_hash:
            raise CalibrationError(
                f"{key} is already registered with a different specification; a "
                "changed mixture needs a new version, not a redefinition")
        return existing
    _PROFILES[key] = profile
    return profile


def get_profile(qualified_id: str) -> CalibrationProfile:
    if qualified_id not in _PROFILES:
        raise KeyError(f"no calibration profile {qualified_id!r}; "
                       f"registered: {sorted(_PROFILES)}")
    return _PROFILES[qualified_id]


def registered_profiles() -> list[str]:
    return sorted(_PROFILES)


def unregister_profile(qualified_id: str) -> None:
    """Test-only."""
    _PROFILES.pop(qualified_id, None)


# --- loading concrete profiles ---------------------------------------------
#
# The four concrete mixtures this project has defined used to be written out
# here and registered at import. That put experiment DATA -- specific token
# budgets, seeds, item paths, source datasets and leakage proofs -- inside the
# reusable core, and made merely importing this module change global state.
#
# They now live in `configs/calibration/profiles.json`, and a caller loads them.
# This module keeps only the mechanism: what a profile IS, and how to build one
# from a document.


def profile_from_dict(doc: Mapping[str, Any]) -> CalibrationProfile:
    """One profile from its serialized form. Field names are the dataclass's."""
    fields = dict(doc)
    fields["sources"] = tuple(CalibrationSource(**s) for s in fields.get("sources", ()))
    if fields.get("role") is not None:
        fields["role"] = DatasetRole(fields["role"])
    for key in ("domain_weights", "metadata"):
        if fields.get(key) is not None:
            fields[key] = dict(fields[key])
    for key in ("leakage_exclusions",):
        if fields.get(key) is not None:
            fields[key] = tuple(fields[key])
    return CalibrationProfile(**fields)


def load_profiles(document: Mapping[str, Any], *,
                  register: bool = True) -> tuple[CalibrationProfile, ...]:
    """Build the profiles a document declares, and register them by default.

    Registration stays refusable: re-registering a qualified id whose
    specification differs raises, so loading a document that redefines an
    existing mixture is an error rather than a silent swap.
    """
    if document.get("schema") != PROFILE_DOCUMENT_SCHEMA:
        raise CalibrationError(
            f"expected a {PROFILE_DOCUMENT_SCHEMA} document, got "
            f"{document.get('schema')!r}")
    built = tuple(profile_from_dict(d) for d in document["profiles"])
    if register:
        built = tuple(register_profile(p) for p in built)
    return built


PROFILE_DOCUMENT_SCHEMA = "aadistill.calibration_profiles/v1"


def buildable_profiles(profiles: Sequence[CalibrationProfile]) -> list[CalibrationProfile]:
    """Of the profiles given, those a run can actually use today."""
    return [p for p in profiles if p.materialized]


def profile_summary(profiles: Sequence[CalibrationProfile]) -> list[dict[str, Any]]:
    return [
        {
            "qualified_id": p.qualified_id,
            "materialized": p.materialized,
            "token_budget": p.token_budget,
            "domains": sorted(p.domain_weights),
            "profile_hash": p.profile_hash,
        }
        for p in profiles
    ]
