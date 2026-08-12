"""Architecture description and the adapter boundary.

The AutoInitializer searches over *structural transformations*. Which structural
fields exist, what they are called, where the blocks live and which linear reads
the residual stream are **family knowledge**, and family knowledge lives here —
never in the search engine and never in an operator's control flow.

Three things are separated on purpose:

``ArchSpec``
    A hashable, family-tagged bag of structural fields. It is deliberately not a
    fixed dataclass of ``hidden_size``/``num_hidden_layers``/...: a future MoE or
    MLA family declares field names this module has never heard of, and the
    engine still has to compare, diff and hash specs. Comparison is therefore
    field-name based, and an adapter owns the names.

``Capability``
    What a family *can do*, as opaque strings. An operator declares the
    capabilities it needs; the registry dispatches on that instead of on
    ``isinstance`` or on a family name. This is what lets a new attention family
    arrive with its own ``ATTENTION`` implementation and no core edit.

``ArchitectureAdapter``
    The only object allowed to know that a Qwen3 block keeps its FFN under
    ``.mlp`` and its input norm under ``.input_layernorm``. Methods that only make
    sense under a capability raise ``UnsupportedCapability`` by default, so a
    partial adapter is a loud failure rather than a silent one.

Nothing here imports an operator, the search engine, or a model family. The
dependency arrow points one way: engine -> operators -> adapters -> this module.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


class UnsupportedCapability(NotImplementedError):
    """An adapter was asked for something its declared capabilities exclude."""


class Capability:
    """Opaque capability tags.

    Strings rather than an enum: a family defined outside this repository must be
    able to declare a capability the core has never seen, and an enum would force
    a core edit to do it. The constants below are the ones v1 dispatches on; they
    are a convenience, not a closed set.
    """

    RESIDUAL_STREAM = "residual_stream"
    PRENORM_BLOCKS = "prenorm_blocks"
    BLOCK_LIST = "block_list"
    DENSE_FFN = "dense_ffn"
    MOE_FFN = "moe_ffn"
    MOE_ROUTER = "moe_router"
    MOE_SHARED_EXPERT = "moe_shared_expert"
    ATTENTION_MHA = "attention.mha"
    ATTENTION_GQA = "attention.gqa"
    ATTENTION_MLA = "attention.mla"
    ATTENTION_LINEAR = "attention.linear"
    TIED_EMBEDDINGS = "tied_embeddings"
    RMS_NORM = "rms_norm"
    ACTIVATION_STATS = "activation_stats"
    LOGIT_COMPARABLE = "logit_comparable"


@dataclass(frozen=True)
class ArchSpec:
    """An immutable structural description of one model.

    ``fields`` is stored as a sorted tuple of pairs so the spec is hashable and
    its serialization is canonical — two specs built in different field orders
    hash identically, which the state store depends on.
    """

    family: str
    fields: tuple[tuple[str, Any], ...]

    @classmethod
    def of(cls, family: str, fields: Mapping[str, Any]) -> ArchSpec:
        return cls(family, tuple(sorted((str(k), v) for k, v in fields.items())))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.fields)

    def __getitem__(self, key: str) -> Any:
        d = self.as_dict()
        if key not in d:
            raise KeyError(f"{self.family} spec has no structural field {key!r}; "
                           f"declared: {sorted(d)}")
        return d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def replace(self, **updates: Any) -> ArchSpec:
        """A new spec with some fields changed. Unknown field names are refused.

        Adding a field would silently widen the structural description, and the
        diff machinery would then read the addition as a change no operator
        declared. New fields belong to a new adapter version.
        """
        known = self.as_dict()
        unknown = sorted(set(updates) - set(known))
        if unknown:
            raise KeyError(
                f"{self.family} spec has no structural fields {unknown}; a "
                "transformation that introduces a field needs an adapter version "
                "that declares it")
        return ArchSpec.of(self.family, {**known, **updates})

    def diff(self, other: ArchSpec) -> frozenset[str]:
        """Structural field names whose values differ (family mismatch included)."""
        if self.family != other.family:
            raise ValueError(
                f"cannot diff a {self.family} spec against a {other.family} spec")
        a, b = self.as_dict(), other.as_dict()
        keys = set(a) | set(b)
        return frozenset(k for k in keys if a.get(k, _MISSING) != b.get(k, _MISSING))

    def matches(self, other: ArchSpec) -> bool:
        return self.family == other.family and self.fields == other.fields

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(
            json.dumps({"family": self.family, "fields": self.as_dict()},
                       sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()

    def describe(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.fields)
        return f"{self.family}({inner})"


_MISSING = object()


class ArchitectureAdapter(ABC):
    """Everything the engine and the operators are allowed to know about a family.

    Subclasses implement only what their ``capabilities`` promise. The default
    bodies raise ``UnsupportedCapability`` so that an operator which skipped its
    capability check fails at the boundary instead of reaching into a module tree
    that does not have the shape it assumed.
    """

    family: str = ""
    adapter_version: str = ""
    capabilities: frozenset[str] = frozenset()
    #: The structural field names this adapter manages. An operator may only
    #: declare `modifies`/`preserves` over these.
    structural_fields: tuple[str, ...] = ()

    # --- spec algebra ------------------------------------------------------

    @abstractmethod
    def spec_from_config(self, config: Any) -> ArchSpec:
        """Read a model config into a family-tagged structural spec."""

    @abstractmethod
    def build_config(self, base_config: Any, spec: ArchSpec) -> Any:
        """A config at ``spec``, inheriting non-structural settings from ``base``."""

    @abstractmethod
    def param_count(self, spec: ArchSpec) -> int:
        """Exact parameter count implied by a spec, without building the model."""

    @abstractmethod
    def validate_target(self, spec: ArchSpec) -> None:
        """Raise if ``spec`` is not a coherent target for this family."""

    def spec_of(self, model: Any) -> ArchSpec:
        return self.spec_from_config(model.config)

    # --- model lifecycle ---------------------------------------------------

    @abstractmethod
    def build_model(self, config: Any, dtype: Any, seed: int) -> Any:
        """A fresh model at ``config``; deterministic given ``seed``."""

    @abstractmethod
    def save(self, model: Any, path: str, *, max_shard_size: str | int | None = None) -> None:
        """Write a checkpoint. ``max_shard_size`` is honoured when the format shards."""

    @abstractmethod
    def load(self, path: str, dtype: Any = None, device: str = "cpu") -> Any:
        """Canonical reload. The engine hashes and measures whatever this returns."""

    @abstractmethod
    def weight_files(self, path: str) -> list[str]:
        """Every weight shard in the directory, as filenames relative to it.

        A list rather than a single name because a 30B-class teacher is always
        sharded and a depth-only 4B intermediate already can be. The caller sorts;
        the adapter only has to find them.
        """

    def index_file(self, path: str) -> str | None:
        """The shard index filename, when the format uses one."""
        return None

    # --- structure accessors ----------------------------------------------

    def blocks(self, model: Any) -> list[Any]:
        raise UnsupportedCapability(
            f"{self.family} adapter does not expose a block list")

    def set_blocks(self, model: Any, blocks: Iterable[Any]) -> None:
        raise UnsupportedCapability(
            f"{self.family} adapter does not expose a block list")

    def attention(self, block: Any) -> Any:
        raise UnsupportedCapability(f"{self.family} adapter has no attention accessor")

    def ffn(self, block: Any) -> Any:
        raise UnsupportedCapability(f"{self.family} adapter has no FFN accessor")

    def stream_in_projections(self, block: Any) -> dict[str, tuple[Any, Any]]:
        """Linears that *read* the residual stream: role -> (linear, preceding norm).

        The norm is part of the answer because a width transformation folds the
        preceding elementwise norm weight into the linear (``W diag(w) P``) and
        then sets that norm to ones. Which norm precedes which projection is
        family knowledge; pairing them here keeps the operator from guessing.
        """
        raise UnsupportedCapability(f"{self.family} adapter has no stream readers")

    def stream_out_projections(self, block: Any) -> dict[str, Any]:
        """Linears that *write* the residual stream, by role name."""
        raise UnsupportedCapability(f"{self.family} adapter has no stream writers")

    def attn_norm(self, block: Any) -> Any:
        raise UnsupportedCapability(f"{self.family} adapter has no attention norm")

    def ffn_norm(self, block: Any) -> Any:
        raise UnsupportedCapability(f"{self.family} adapter has no FFN norm")

    def embedding(self, model: Any) -> Any:
        raise UnsupportedCapability(f"{self.family} adapter has no embedding accessor")

    def final_norm(self, model: Any) -> Any:
        raise UnsupportedCapability(f"{self.family} adapter has no final norm accessor")

    def head_groups(self, spec: ArchSpec) -> tuple[int, int, int]:
        """(query heads, key/value heads, head dim) for grouped attention."""
        raise UnsupportedCapability(f"{self.family} adapter is not a grouped-attention family")

    def stats_collector(self, model: Any) -> Any:
        """A streaming activation-statistics collector for this family."""
        raise UnsupportedCapability(
            f"{self.family} adapter cannot collect activation statistics")

    # --- helpers -----------------------------------------------------------

    def requires(self, *capabilities: str) -> None:
        missing = sorted(set(capabilities) - set(self.capabilities))
        if missing:
            raise UnsupportedCapability(
                f"{self.family} adapter lacks capabilities {missing}")

    def identity(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "adapter_version": self.adapter_version,
            "capabilities": sorted(self.capabilities),
            "structural_fields": list(self.structural_fields),
        }


# --- adapter registry ------------------------------------------------------

_ADAPTERS: dict[str, ArchitectureAdapter] = {}


def register_adapter(adapter: ArchitectureAdapter, *, replace: bool = False) -> ArchitectureAdapter:
    """Register an adapter under its family name.

    Re-registering the same family with a different adapter version is refused
    unless ``replace`` is passed, because a search manifest records the adapter
    version it ran under and a silent swap would invalidate that record.
    """
    if not adapter.family:
        raise ValueError("adapter must declare a family")
    if not adapter.adapter_version:
        raise ValueError(f"{adapter.family} adapter must declare an adapter_version")
    existing = _ADAPTERS.get(adapter.family)
    if existing is not None and not replace:
        if existing.adapter_version != adapter.adapter_version:
            raise ValueError(
                f"family {adapter.family!r} is already registered at version "
                f"{existing.adapter_version!r}; registering {adapter.adapter_version!r} "
                "would change what an existing manifest describes")
        return existing
    _ADAPTERS[adapter.family] = adapter
    return adapter


def get_adapter(family: str) -> ArchitectureAdapter:
    if family not in _ADAPTERS:
        raise KeyError(
            f"no architecture adapter registered for family {family!r}; "
            f"registered: {sorted(_ADAPTERS)}")
    return _ADAPTERS[family]


def adapter_for_config(config: Any) -> ArchitectureAdapter:
    family = getattr(config, "model_type", None)
    if family is None:
        raise ValueError("config has no model_type; cannot select an adapter")
    return get_adapter(family)


def registered_families() -> list[str]:
    return sorted(_ADAPTERS)


def unregister_adapter(family: str) -> None:
    """Test-only: drop a family so a fixture adapter cannot leak between tests."""
    _ADAPTERS.pop(family, None)
