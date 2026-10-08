"""The micro-preflight session authorization.

The application layer. The grant is DATA and lives in
`configs/experiments/micro_preflight/authorization.json`; what an authorization
IS lives in `aadistill.governance.authorization`.

It used to be written out inside `src/aadistill`, which put one session's
dollar amounts, granted date and plan hash inside the reusable core.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.governance.authorization import authorization_from_dict  # noqa: E402
from experiments.preflight import PREFLIGHT_POLICY  # noqa: E402

AUTHORIZATION_CONFIG = (REPO
                        / "configs/experiments/micro_preflight/authorization.json")

#: Under the preflight's wire contract: this session's artifact carries the
#: same schema and the same `preflight_plan_hash` key, so it reads and writes
#: identically. The POLICY still grants nothing -- `allowed` is empty -- which
#: is the part that decides permission; the wire contract only decides how the
#: artifact is spelled.
MICRO_PREFLIGHT_AUTHORIZATION = authorization_from_dict(
    json.loads(AUTHORIZATION_CONFIG.read_text())["authorization"],
    policy=PREFLIGHT_POLICY)

__all__ = ["AUTHORIZATION_CONFIG", "MICRO_PREFLIGHT_AUTHORIZATION"]
