"""Calibration profiles — first-class, versioned, and chosen per operator.

E8a measured its depth objective on one frozen 67-item domain-balanced mixture.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .datasets import DatasetRole


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
    without a pinned revision cannot support P4 reproduction, and leaving the
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

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise CalibrationError("calibration profile needs an id")
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


# --- the three v1 profiles -------------------------------------------------
#
# Only the second is built. The other two are representable so a path may name
# them and the cost model may price them; `resolve()` refuses them until they
# exist.

E8A_DOMAINS = ("general", "math", "rag_multihop", "code", "tool")

STAGE0_CURRENT_V1 = register_profile(CalibrationProfile(
    profile_id="calib.stage0_current",
    version=1,
    description=(
        "The historical Stage-0 warm-up mixture that produced "
        "artifacts/stage0/qwen3_4b_thinking_v1 (949,859 tokens). Represented so a "
        "search path can select the incumbent mixture; the item list still has to "
        "be re-rendered from scripts/data/build_warmup_v1.py before use."),
    sources=(
        CalibrationSource("aadistill/warmup_v1", "local", "mixed", 0,
                          note="rebuild via scripts/data/build_warmup_v1.py"),
    ),
    domain_weights={"mixed": 1.0},
    token_budget=949_859,
    sample_rule="warmup_v1 builder order, seed-fixed",
    seed=20260713,
    materialized=False,
    metadata={"activation_stats": "artifacts/stage0/qwen3_4b_thinking_v1/activation_stats.safetensors",
              "activation_stats_sha256_prefix": "aaeb2e4c"},
))

DOMAIN_BALANCED_V1 = register_profile(CalibrationProfile(
    profile_id="calib.domain_balanced",
    version=1,
    description=(
        "E8a's frozen 67-item mixture: five domains, 59,763 prediction positions, "
        "leakage-checked against the recovery rung, the validation slice and "
        "prompt-content collisions. The only v1 profile that is already built."),
    sources=tuple(
        CalibrationSource("aadistill/e8_calibration_v1", "2026-08-10", d, 0,
                          note="per-domain counts in the mixture manifest")
        for d in E8A_DOMAINS
    ),
    domain_weights={d: 1.0 / len(E8A_DOMAINS) for d in E8A_DOMAINS},
    token_budget=59_763,
    sample_rule="frozen item list; unweighted mean over domains of the unweighted "
                "mean over each domain's sub-types",
    seed=20260810,
    materialized=True,
    items_path="artifacts/stage1/e8_calibration_v1/items.jsonl",
    content_sha256="d65c1f40e4837ea1bd5bcc33c68041a13b797c68f5be3c0686e0142ed761028f",
    items_file_sha256="c7202338109e459b17b70456461e8f304fadea7929ea547accee21adbbe7fd0b",
    leakage_exclusions=("ladder_uniform_probe.rung", "ladder_uniform_probe.val",
                        "prompt_content_collisions"),
    leakage_proof_path="artifacts/stage1/e8_calibration_v1/leakage.json",
    metadata={"manifest": "artifacts/stage1/e8_calibration_v1/manifest.json",
              "frozen_by": "E8a"},
))

REASONING_HEAVY_V1 = register_profile(CalibrationProfile(
    profile_id="calib.reasoning_heavy",
    version=1,
    description=(
        "A reasoning-weighted counterpart to the domain-balanced mixture, for the "
        "hypothesis that an operator should be calibrated on the capability the "
        "recipe actually targets (AGENTS.md P3/P10.1) rather than on a uniform "
        "domain average. Not built."),
    sources=tuple(
        CalibrationSource("aadistill/e8_calibration_v1", "2026-08-10", d, 0,
                          note="reweighted draw from the same item pool")
        for d in E8A_DOMAINS
    ),
    domain_weights={"general": 0.10, "math": 0.35, "rag_multihop": 0.25,
                    "code": 0.20, "tool": 0.10},
    token_budget=59_763,
    sample_rule="weighted draw from the domain-balanced pool, deterministic by seed",
    seed=20260812,
    materialized=False,
    metadata={"pool": "calib.domain_balanced@v1"},
))

V1_PROFILES = (STAGE0_CURRENT_V1, DOMAIN_BALANCED_V1, REASONING_HEAVY_V1)


def buildable_profiles() -> list[CalibrationProfile]:
    """Profiles a run can actually use today."""
    return [p for p in V1_PROFILES if p.materialized]


def profile_summary(profiles: Sequence[CalibrationProfile] = V1_PROFILES) -> list[dict[str, Any]]:
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
