"""Resolve a frozen record's path spelling to the object's current home.

Frozen configs and records keep the spellings they were written with; the
2026-10-08/09 information-architecture migration moved the objects. A test
that follows such a pointer resolves it here — a thin reader over
`logs/index.json :: historical_paths.map` (exact match, then longest
prefix), the same table `shared.run_layout.resolve_historical` serves the
application layer from. Core tests read the committed data file directly so
the core suite keeps importing nothing from `scripts/`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    doc = json.loads((REPO / "logs/index.json").read_text())
    return doc["historical_paths"]["map"]


def resolve(rel: str) -> str:
    table = _table()
    probe, suffix = rel, ""
    while probe:
        if probe in table:
            return table[probe] + suffix
        probe, _, tail = probe.rpartition("/")
        suffix = "/" + tail + suffix
    return rel
