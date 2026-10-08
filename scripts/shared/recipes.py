"""The concrete recovery recipes this project has frozen.

The application layer. The recipes are DATA and live in
`configs/recipes/recovery.json`; what a recipe *is* lives in
`aadistill.initialization.planning.recovery`; and the decision to load these
is made here, where an experiment-instance decision belongs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.planning.recovery import load_recipes  # noqa: E402

RECIPE_CONFIG = REPO / "configs/recipes/recovery.json"

RECIPES = load_recipes(json.loads(RECIPE_CONFIG.read_text()))
BY_ID = {r.recipe_id: r for r in RECIPES}

#: The recipe every E1 arm used, at the rung the frozen battery was sampled
#: from. Held fixed so a difference between two probes is a difference between
#: two initializations.
E1_KD_HEAVY_0860K = BY_ID["e1_p1_kd_heavy@0.86M"]

__all__ = ["RECIPE_CONFIG", "RECIPES", "BY_ID", "E1_KD_HEAVY_0860K"]
