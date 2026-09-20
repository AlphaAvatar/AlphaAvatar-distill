"""The pod-environment and readiness contract for ONE behavioural session.

Same generic machinery every other session uses — the sweep runner, the record
contract and the group rules all live in `aadistill.runtime.pod_environment`.
What differs is everything the record BINDS, which is the part that decides
whether a launch may rest on it.

The pod SELECTION is this session's own, `tests/c2_behavioural_preflight`.
The replay's asserts five reconstructed leaves and no training; Search-1's
asserts a canonical control this session does not stage; the full search's
asserts a beam. All three would fail a correct behavioural tree, and narrowing
any of them to fit would stop it guarding the session it was written for — a
lesson two C2 sessions have already paid for.
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
    LAUNCH_BOUND, RECORD_KINDS, ReadinessGroups, RecordContract, SweepContract,
    pod_test_environment_digest,
)

from experiments.phase_c2 import behavioural_governance as BG
from experiments.phase_c2.pod_environment import (  # noqa: F401
    POD_TEST_ENVIRONMENT_FILES_V1, ReadinessError,
)

#: This session's OWN pod selection, named ONCE here and read by the launcher
#: through this module.
POD_TEST_SELECTION = "tests/c2_behavioural_preflight"

#: Its OWN schema. The one-line check that makes a full-search, Search-1,
#: baseline-completion or replay readiness record unusable here, and this one
#: unusable for them.
SCHEMA = "aadistill.autoinit.c2_behavioural_pod_environment/v1"

EXPERIMENT_ID = "phase_c2_behavioural"

LAUNCHER_MODULE = "autoinit_c2_behavioural_launch"

RUN_READINESS_ROLE = "governance/readiness.json"

#: Everything in the selection passes and nothing skips, enforced by the
#: machinery rather than by a list somebody maintains. The selection is written
#: with no conditional skips for exactly this reason: a `skipif` keyed on a
#: simulator-set marker is INVERTED on the pod, and a gate can check THAT a
#: test skipped but never WHY.
READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(POD_TEST_SELECTION,),
    refusal_notes={},
)


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where a behavioural readiness record lives, repository-relative.

    A run id is REQUIRED. Launch-bound readiness is evidence about ONE attempt,
    and a shared default location is how the evidence a run launched under
    comes to live at a path the next run replaces.
    """
    if not run_id:
        raise ReadinessError(
            "a behavioural readiness record has no location without a run id: "
            "it is evidence about one attempt, and a shared path is how one "
            "attempt's evidence comes to describe another's tree")
    from experiments.run_layout import rel_run_dir

    return f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id)}/{RUN_READINESS_ROLE}"


def harness(repo_root: Any = REPO) -> dict[str, Any]:
    """The LIVE behavioural closure, as the record's harness.

    A callable, evaluated when the sweep runs: a value captured at import time
    would describe whatever the tree was when somebody imported this module.
    """
    live = BG.current_executable(repo_root)
    return {"digest": live["digest"], "n_files": live["n_files"],
            "files": [row["path"] for row in live["files"]]}


def harness_digest(repo_root: Any = REPO) -> str:
    return harness(repo_root)["digest"]


def record_contract(run_id: str | None = None,
                    stage_id: str | None = None) -> RecordContract:
    return RecordContract(
        schema=SCHEMA,
        harness_field="behavioural_harness_digest",
        harness_digest=harness_digest,
        record_path=record_path_for(run_id, stage_id),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="behavioural harness",
    )


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str = LAUNCH_BOUND) -> SweepContract:
    """How to DRIVE a behavioural readiness sweep, for the generic recorder."""

    def bundle_name(commit: str) -> str:
        from experiments.phase_c2.behavioural_bundle import canonical_bundle_name

        return canonical_bundle_name(commit)

    return SweepContract(
        experiment_id=EXPERIMENT_ID,
        record=record_contract(run_id, stage_id),
        groups=READINESS_GROUPS,
        launcher_module=LAUNCHER_MODULE,
        session_id=BG.SESSION_ID,
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="behavioural_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of the behavioural session's own pod "
            f"selection, {POD_TEST_SELECTION}/ — everything the paid session "
            "can fail before its first arm is built, and nothing else. It "
            "checks that the frozen protocol parses and matches its own hash, "
            "that the decision rule reads C2's three confirmation seeds and "
            "C2's own bootstrap seed rather than C1's default, that all six "
            "arms' specs build with every step digest-pinned and every root "
            "equal to the verified teacher binding, that both batteries "
            "validate against their identity records, that the screening "
            "scorer refuses the confirmation battery and vice versa, that the "
            "schedule refuses a partial rung and an anchor that advances, and "
            "that the plan fits its authorized money. What this record BINDS "
            "is the BEHAVIOURAL session: its launcher, its session id and its "
            "executable closure. No other C2 readiness record can satisfy it, "
            "and it cannot satisfy theirs."),
    )


def verify_record(record: dict, repo_root: Any = ".", *, run_id: str,
                  stage_id: str | None = None, **kwargs):
    """The behavioural record verification. `run_id` is required."""
    kwargs.setdefault("contract", record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root: Any = ".", *, run_id: str,
                stage_id: str | None = None) -> dict[str, Any]:
    """Read this run's behavioural readiness record, or raise."""
    import json

    path = Path(repo_root) / record_path_for(run_id, stage_id)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())
