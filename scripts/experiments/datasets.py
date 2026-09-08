"""The concrete datasets this project has frozen.

The application layer. The assets themselves are DATA and live in
`configs/datasets/assets.json`; what an asset *is*, and the role isolation that
governs it, live in `aadistill.initialization.calibration.datasets`; and the
decision to load these particular three is made here.

They used to be written out inside `src/aadistill` and registered at import, so
importing the reusable core both carried experiment data — prompt counts,
artifact paths, content hashes, leakage proofs — and mutated global state.

Importing THIS module registers them, which is what an application bootstrap is
for. Anything wanting an empty registry uses the core directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.calibration.datasets import (  # noqa: E402
    get_asset, load_assets)

#: Where the assets are declared. Data, not code.
ASSET_CONFIG = REPO / "configs/datasets/assets.json"


def register_builtin_assets(config: Path | None = None) -> tuple:
    """Load and register the declared assets. Idempotent."""
    document = json.loads((config or ASSET_CONFIG).read_text())
    return load_assets(document)


#: Registered on import, because this module IS the application bootstrap.
ASSETS = register_builtin_assets()

FROZEN_PROMOTION_BATTERY = get_asset("battery.frozen_promotion_150")
CAPABILITY_BATTERY_V2 = get_asset("battery.capability_v2_846")
E8A_CALIBRATION = get_asset("calib.e8a_domain_balanced_67")

__all__ = ["ASSET_CONFIG", "register_builtin_assets", "ASSETS",
           "FROZEN_PROMOTION_BATTERY", "CAPABILITY_BATTERY_V2",
           "E8A_CALIBRATION"]
