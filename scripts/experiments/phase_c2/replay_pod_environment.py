"""The pod-environment and readiness contract for ONE replay session.

Same generic machinery every other session uses — the sweep runner, the record
contract and the group rules all live in `aadistill.runtime.pod_environment`.
What differs is everything the record BINDS, which is the part that decides
whether a launch may rest on it.

The pod SELECTION is this session's own, `tests/c2_replay_preflight`. Search-1's
asserts the canonical 0.6B control this session does not stage; the full
search's asserts that session's staged assets and its beam. Both would fail a
correct replay tree, and narrowing either to fit would stop it guarding the
session it was written for — the full search learned that on its first
launch-bound sweep and the baseline completion before it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.runtime import pod_environment as _pe  # noqa: E402
from aadistill.runtime.pod_environment import (  # noqa: E402,F401
    LAUNCH_BOUND,
    RECORD_KINDS,
    ReadinessGroups,
    RecordContract,
    SweepContract,
    pod_test_environment_digest,
)

from experiments.phase_c2 import replay as RG
from experiments.phase_c2.pod_environment import (  # noqa: F401
    POD_TEST_ENVIRONMENT_FILES_V1, ReadinessError,
)

#: This session's OWN pod selection, named ONCE here and read by the launcher
#: through this module.
POD_TEST_SELECTION = "tests/c2_replay_preflight"

#: Its OWN schema. This is the one-line check that makes a full-search,
#: Search-1 or baseline-completion readiness record unusable here, and this one
#: unusable for them.
SCHEMA = "aadistill.autoinit.c2_replay_pod_environment/v1"

EXPERIMENT_ID = "phase_c2_replay"

LAUNCHER_MODULE = "autoinit_c2_replay_launch"

RUN_READINESS_ROLE = "governance/readiness.json"

#: Everything in the selection passes and nothing skips, enforced by the
#: machinery rather than by a list somebody maintains. The selection was written
#: with no conditional skips for exactly this reason.
READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(POD_TEST_SELECTION,),
    refusal_notes={},
)


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where a replay readiness record lives, repository-relative.

    A run id is REQUIRED. Launch-bound readiness is evidence about ONE attempt,
    and a shared default location is how the evidence a run launched under comes
    to live at a path the next run replaces.
    """
    if not run_id:
        raise ReadinessError(
            "a replay readiness record has no location without a run id: it is "
            "evidence about one attempt, and a shared path is how one attempt's "
            "evidence comes to describe another's tree")
    from experiments.run_layout import rel_run_dir

    return f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id)}/{RUN_READINESS_ROLE}"


def harness(repo_root: Any = REPO) -> dict[str, Any]:
    """The LIVE replay closure, as the record's harness.

    A callable, evaluated when the sweep runs: a value captured at import time
    would describe whatever the tree was when somebody imported this module.
    """
    live = RG.current_executable(repo_root)
    return {"digest": live["digest"], "n_files": live["n_files"],
            "files": [row["path"] for row in live["files"]]}


def harness_digest(repo_root: Any = REPO) -> str:
    return harness(repo_root)["digest"]


def record_contract(run_id: str | None = None,
                    stage_id: str | None = None) -> RecordContract:
    return RecordContract(
        schema=SCHEMA,
        harness_field="replay_harness_digest",
        harness_digest=harness_digest,
        record_path=record_path_for(run_id, stage_id),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="replay harness",
    )


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str = LAUNCH_BOUND) -> SweepContract:
    """How to DRIVE a replay readiness sweep, for the generic recorder."""

    def bundle_name(commit: str) -> str:
        from experiments.phase_c2.replay_bundle import canonical_bundle_name

        return canonical_bundle_name(commit)

    return SweepContract(
        experiment_id=EXPERIMENT_ID,
        record=record_contract(run_id, stage_id),
        groups=READINESS_GROUPS,
        launcher_module=LAUNCHER_MODULE,
        session_id=RG.SESSION_ID,
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="replay_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of the replay's own pod selection, "
            f"{POD_TEST_SELECTION}/ — everything the paid session can fail "
            "before its first operator runs, and nothing else. It checks that "
            "the committed evidence arrived and parses, that five specs build "
            "with all twenty steps pinned, that no operator has moved since "
            "attempt 3's commit, that both calibration mixtures resolve from "
            "disk and hash to what the selection was searched against, that "
            "the driver leaves its profile registry populated in a FRESH "
            "interpreter, and that the plan fits its authorized money. What "
            "this record BINDS is the REPLAY session: its launcher, its "
            "session id and its executable closure. A full-search readiness "
            "record cannot satisfy it, and it cannot satisfy a full search's."),
    )


def verify_record(record: dict, repo_root: Any = ".", *, run_id: str,
                  stage_id: str | None = None, **kwargs):
    """The replay record verification. `run_id` is required."""
    kwargs.setdefault("contract", record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root: Any = ".", *, run_id: str,
                stage_id: str | None = None) -> dict[str, Any]:
    """Read this run's replay readiness record, or raise."""
    import json

    path = Path(repo_root) / record_path_for(run_id, stage_id)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())
