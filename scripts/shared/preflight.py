"""The preflight session: its harness, its action policy, its authorization.

The application layer. `aadistill.governance.authorization` holds the mechanism
-- what an authorization IS, how a harness digest is computed, how a claim is
refused -- and knows none of the names below.

Three things used to live in that core module and are experiment facts:

* `HARNESS_SOURCE_FILES_V1`, which names seven pod scripts. Which scripts a
  session executes is the session's fact; a governance primitive that listed
  them could not serve a second experiment without being edited.
* `allows_phase_a` and `automatic_phase_a_start`, hard-`False` properties. Phase
  A is one project's stage name.
* `refuse_phase_a`, whose refusal text named the preflight.

The refusal is unchanged in effect: `PREFLIGHT_POLICY` grants nothing, and
absence of permission is denial.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.governance.authorization import (  # noqa: E402
    ActionPolicy, SpendAuthorization, harness_source_digest)

#: Where the harness is declared. Data, not code.
HARNESS_CONFIG = REPO / "configs/experiments/preflight/harness.json"

_HARNESS = json.loads(HARNESS_CONFIG.read_text())

#: The executable a preflight session runs. Same rule and same failure mode as
#: every other declared source set: a missing declared file raises rather than
#: yielding a digest over a smaller harness than the one that runs.
HARNESS_SOURCE_FILES_V1: tuple[str, ...] = tuple(_HARNESS["files"])
HARNESS_SOURCE_SET_VERSION: int = _HARNESS["set_version"]

#: What a preflight authorization may express: nothing. Phase A is separately
#: unauthorized and is not reachable from the preflight, so `phase_a` is absent
#: from `allowed` rather than present-and-false -- under this mechanism absence
#: IS the denial, and there is no flag anyone could set.
PREFLIGHT_POLICY = ActionPolicy(
    policy_id="preflight",
    allowed=frozenset(),
    wire_claims={"phase_a_authorized": "phase_a",
                 "automatic_phase_a_start": "automatic_phase_a_start"},
    #: The on-disk identity, transcribed from the serializer that used to hold
    #: it. Every byte is what it always was: the schema string, the key the
    #: plan hash appears under, and the enforcement sentence -- including the
    #: Phase-A clause, which is a statement about THIS session and belongs to
    #: it, not to a governance primitive that must serve other experiments.
    wire_schema="aadistill.autoinit.spend_authorization/v1",
    plan_hash_key="preflight_plan_hash",
    enforcement=("the launcher loads this artifact and refuses to create a pod "
                 "whose priced hard threshold exceeds hard_cap_usd, refuses a "
                 "stage not in authorized_stages, and has no code path to "
                 "Phase A"),
    refusal_notes={
        "phase_a": ("Phase A is separately unauthorized and is not reachable "
                    "from the preflight. Stop, report, and obtain a new "
                    "authorization."),
    },
)


class PreflightAuthorization(SpendAuthorization):
    """A spend authorization under the preflight policy."""

    POLICY = PREFLIGHT_POLICY


def preflight_harness_digest(repo_root: str | Path = ".") -> dict:
    """The preflight harness digest, over the files this session declares."""
    return harness_source_digest(repo_root, files=HARNESS_SOURCE_FILES_V1,
                                 set_version=HARNESS_SOURCE_SET_VERSION)


__all__ = ["HARNESS_CONFIG", "HARNESS_SOURCE_FILES_V1",
           "HARNESS_SOURCE_SET_VERSION", "PREFLIGHT_POLICY",
           "PreflightAuthorization", "preflight_harness_digest"]
