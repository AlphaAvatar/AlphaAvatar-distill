"""Recipe data a core test passes to the core explicitly, read from the committed config (never a core default).

`STREAM_WEIGHTING_V0` is the historical end-point weighting (`configs/shared/initialization/stream_weighting.json ::
endpoints_9_1_8_v0`): what every test that reproduces a pre-2026-11-26 projection, or runs `width.global_pca_v0` /
`composite.stage1_sandwich_v0` as they ran then, declares. Read as a data file so the core suite imports nothing from
`scripts/`.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_DOC = json.loads((REPO / "configs/shared/initialization/stream_weighting.json").read_text())

STREAM_WEIGHTING_V0 = {k: v for k, v in _DOC["recipes"]["endpoints_9_1_8_v0"].items()
                       if k in ("endpoint_weights", "source")}
#: the operators' declared config carrying it (a fixed-path step's ``config`` / `SearchConfig.operator_configs`)
WEIGHTED_OPERATOR_CONFIGS = {impl: {"stream_weighting": STREAM_WEIGHTING_V0}
                             for impl in ("width.global_pca_v0", "composite.stage1_sandwich_v0")}
#: the declaration as an operator context's / fixed-path step's config entry (an operator that reads none ignores it)
DECLARED = {"stream_weighting": STREAM_WEIGHTING_V0}


def declared(config=None) -> dict:
    """``config`` with the historical stream weighting declared beside it."""
    return {**DECLARED, **dict(config or {})}
