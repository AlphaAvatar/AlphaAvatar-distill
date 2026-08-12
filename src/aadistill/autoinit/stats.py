"""Activation-statistics cache, with a deliberately narrow reuse boundary.

Two operators need residual second moments and FFN activation magnitudes from the
*same* parent state: ``width.global_pca_v0`` and ``ffn.activation_importance_v0``.
When the search expands one parent into both, that is one statistics pass, not
two — and at the 4B teacher a pass is 1.81 GiB of float64 accumulation and the
single largest unmeasured term in the cost model.

What must **never** happen is reuse across parents. Re-collecting statistics on
every produced checkpoint is the entire scientific point: after a depth or
attention operator the residual second moments are no longer the teacher's, and
E8a's central negative result is precisely that a statistic taken before
composition mispredicts the composed model. A cache that keyed on "the teacher
and the mixture" would silently reintroduce the error the architecture exists to
avoid.

So the key is everything that could change the numbers:

    parent artifact digest
  + calibration profile hash
  + statistics specification hash   (which points, which quantities, dtype)
  + adapter version
  + numerical configuration         (device, accumulation dtype, batch rule)

Change any one and it is a different cache entry. The parent artifact digest is
first because it is the one that makes cross-parent reuse impossible by
construction rather than by discipline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from ..infrastructure.manifest import sha256_json


@dataclass(frozen=True)
class StatsSpec:
    """What is collected, independent of what it is collected from."""

    spec_id: str = "residual_moments_and_ffn_magnitude"
    version: int = 1
    accumulation_dtype: str = "float64"
    quantities: tuple[str, ...] = ("residual_sum", "residual_sqsum",
                                   "residual_count", "ffn_abs_sum", "ffn_sq_sum",
                                   "token_counts")

    @property
    def spec_hash(self) -> str:
        return sha256_json({"spec_id": self.spec_id, "version": self.version,
                            "accumulation_dtype": self.accumulation_dtype,
                            "quantities": list(self.quantities)})

    def as_dict(self) -> dict[str, Any]:
        return {"spec_id": self.spec_id, "version": self.version,
                "accumulation_dtype": self.accumulation_dtype,
                "quantities": list(self.quantities), "spec_hash": self.spec_hash}


DEFAULT_STATS_SPEC = StatsSpec()


def stats_cache_key(*, parent_artifact_digest: str, profile_hash: str,
                    stats_spec: StatsSpec, adapter_version: str,
                    numerical_config: Mapping[str, Any]) -> str:
    if not parent_artifact_digest:
        raise ValueError(
            "refusing to key an activation-statistics cache without the parent "
            "artifact digest: that is the term that makes cross-parent reuse "
            "impossible, and every other term is insufficient without it")
    return sha256_json({
        "parent_artifact_digest": parent_artifact_digest,
        "profile_hash": profile_hash,
        "stats_spec": stats_spec.spec_hash,
        "adapter_version": adapter_version,
        "numerical_config": dict(sorted(numerical_config.items())),
    })


@dataclass
class StatsCache:
    """In-memory, per-run. Small enough to hold one parent's statistics at a time.

    ``max_entries`` defaults to 1 because the search expands one parent at a time
    and a 4B parent's statistics are ~1.8 GiB — holding several would trade a
    measured memory cost for a saving the access pattern does not produce.
    """

    stats_spec: StatsSpec = DEFAULT_STATS_SPEC
    max_entries: int = 1
    _entries: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    passes_saved: int = 0

    def get_or_collect(self, key: str, collect) -> dict[str, torch.Tensor]:
        if key in self._entries:
            self.hits += 1
            self.passes_saved += 1
            return self._entries[key]
        self.misses += 1
        state = collect()
        self._entries[key] = state
        self._order.append(key)
        while len(self._order) > max(1, self.max_entries):
            self._entries.pop(self._order.pop(0), None)
        return state

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()

    def report(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses,
                "passes_saved": self.passes_saved,
                "resident_entries": len(self._entries),
                "stats_spec": self.stats_spec.as_dict()}
