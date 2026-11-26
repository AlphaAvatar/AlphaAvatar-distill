"""The stream weightings a recipe may declare (`configs/shared/initialization/stream_weighting.json`), for the
application layer: `recipe(name)` returns the declaration the core validates (`transforms.project.stream_weighting`)
-- its weights and its source -- and `operator_configs(name)` the search / fixed-path declared config of the two
operators that read one. The core holds no weighting; this module only reads the committed record."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "configs/shared/initialization/stream_weighting.json"
#: the recipe every Stage-1 config written before 2026-11-26 ran under (its projection diagnostics record the
#: weights [9, 1, ..., 1, 8]); an application names it explicitly for those configs
HISTORICAL = "endpoints_9_1_8_v0"
OPERATORS = ("width.global_pca_v0", "composite.stage1_sandwich_v0")


def recipe(name: str) -> dict:
    doc = json.loads(RECORD.read_text())["recipes"]
    if name not in doc:
        raise KeyError(f"no stream weighting {name!r} in {RECORD.relative_to(REPO)}; known: {sorted(doc)}")
    r = doc[name]
    return {**({"endpoint_weights": r["endpoint_weights"]} if "endpoint_weights" in r
               else {"points": r["points"], "weights": r["weights"]}), "source": f"{name}: {r['source']}"}


def operator_configs(name: str) -> dict:
    return {impl: {"stream_weighting": recipe(name)} for impl in OPERATORS}
