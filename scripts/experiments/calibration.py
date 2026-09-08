"""The concrete calibration mixtures this project has defined.

This is the application layer. The mixtures themselves are DATA and live in
`configs/calibration/profiles.json`; what a profile *is* lives in
`aadistill.initialization.calibration.profiles`; and the decision to load these
particular four is made here, where an experiment-instance decision belongs.

They used to be written out inside `src/aadistill` and registered at import, so
importing the reusable core both carried experiment data and mutated global
state. Nothing in `src` names them now.

Importing THIS module does register them, which is the point of an application
bootstrap: a caller that wants the project's mixtures asks for them by importing
the module that owns them. Anything wanting an empty registry uses the core
directly and loads its own document.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.calibration.profiles import (  # noqa: E402
    get_profile, load_profiles)

#: Where the mixtures are declared. Data, not code.
PROFILE_CONFIG = REPO / "configs/calibration/profiles.json"


def register_builtin_profiles(config: Path | None = None) -> tuple:
    """Load and register the declared mixtures. Idempotent.

    Idempotent because `register_profile` returns the existing entry when the
    same qualified id is registered with an identical specification, and raises
    when it differs -- so calling this from several entry points in one process
    is safe, while a document that redefines a mixture is an error.
    """
    document = json.loads((config or PROFILE_CONFIG).read_text())
    return load_profiles(document)


#: Registered on import, because this module IS the application bootstrap.
PROFILES = register_builtin_profiles()

STAGE0_CURRENT_V1 = get_profile("calib.stage0_current@v1")
DOMAIN_BALANCED_V1 = get_profile("calib.domain_balanced@v1")
REASONING_HEAVY_V1 = get_profile("calib.reasoning_heavy@v1")
REASONING_HEAVY_V2 = get_profile("calib.reasoning_heavy@v2")

#: The three that existed before `reasoning_heavy@v2`. Several callers summarize
#: exactly this set, and the distinction is theirs rather than the core's.
V1_PROFILES = (STAGE0_CURRENT_V1, DOMAIN_BALANCED_V1, REASONING_HEAVY_V1)

__all__ = ["PROFILE_CONFIG", "register_builtin_profiles", "PROFILES",
           "V1_PROFILES", "STAGE0_CURRENT_V1", "DOMAIN_BALANCED_V1",
           "REASONING_HEAVY_V1", "REASONING_HEAVY_V2"]
