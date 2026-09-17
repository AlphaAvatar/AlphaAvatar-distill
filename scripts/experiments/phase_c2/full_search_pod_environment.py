"""The pod-environment and readiness contract for ONE full-search session.

Same generic machinery every other session uses: the sweep runner, the record
contract and the group rules all live in `aadistill.runtime.pod_environment`.
What differs is everything the record BINDS, which is the part that decides
whether a launch may rest on it.

Search-1's readiness record cannot satisfy this one and this cannot satisfy
Search-1's: `verify_record` compares the record's schema against the contract's,
and the two strings differ. That is the same one-line check that stops a
baseline-completion record standing in for either.

The pod SELECTION is Search-1's, deliberately. Both sessions stage the same
assets — the two calibration mixtures and the frozen metric suite — and run the
same beam machinery against the same teacher, so what a pod can fail before the
search starts is the same set of things. Inventing a second selection would mean
maintaining two answers to one question, and a selection that asserted staged
paths this session does not stage is exactly how a correct test fails a correct
session.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.runtime import pod_environment as _pe  # noqa: E402
#: The generic names, re-exported so a caller reads one namespace.
from aadistill.runtime.pod_environment import (  # noqa: E402,F401
    LAUNCH_BOUND,
    RECORD_KINDS,
    ReadinessGroups,
    RecordContract,
    SweepContract,
    pod_test_environment_digest,
)

from experiments.phase_c2 import full_search as FSG
from experiments.phase_c2.pod_environment import (  # noqa: F401
    POD_TEST_ENVIRONMENT_FILES_V1, POD_TEST_SELECTION, ReadinessError,
)

#: Its OWN schema. This is the check that makes a Search-1 or a
#: baseline-completion readiness record unusable here.
SCHEMA = "aadistill.autoinit.c2_full_search_pod_environment/v1"

EXPERIMENT_ID = "phase_c2_full_search"

LAUNCHER_MODULE = "autoinit_phase_c2_full_search_launch"

#: The run's declared role for its readiness evidence, same convention as every
#: other role in a run's `governance/` area.
RUN_READINESS_ROLE = "governance/readiness.json"

#: Everything in the selection passes and nothing skips, enforced by the
#: machinery rather than by a list somebody maintains.
READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(POD_TEST_SELECTION,),
    refusal_notes={},
)


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where a full-search readiness record lives, repository-relative.

    A run id is REQUIRED. Launch-bound readiness is evidence about ONE attempt,
    and a shared default location is how the evidence a run launched under comes
    to live at a path the next run replaces. There is no phase-level fallback
    here and no diagnostic path either.
    """
    if not run_id:
        raise ReadinessError(
            "a full-search readiness record has no location without a run id: "
            "it is evidence about one attempt, and a shared path is how one "
            "attempt's evidence comes to describe another's tree")
    from experiments.run_layout import rel_run_dir

    return f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id)}/{RUN_READINESS_ROLE}"


def harness(repo_root: Any = REPO) -> dict[str, Any]:
    """The LIVE full-search closure, as the record's harness.

    A callable, evaluated when the sweep runs, because a value captured at
    import time would describe whatever the tree was when somebody imported
    this module.
    """
    live = FSG.current_executable(repo_root)
    return {"digest": live["digest"], "n_files": live["n_files"],
            "files": [row["path"] for row in live["files"]]}


def harness_digest(repo_root: Any = REPO) -> str:
    return harness(repo_root)["digest"]


def record_contract(run_id: str | None = None,
                    stage_id: str | None = None) -> RecordContract:
    return RecordContract(
        schema=SCHEMA,
        harness_field="full_search_harness_digest",
        harness_digest=harness_digest,
        record_path=record_path_for(run_id, stage_id),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="full-search harness",
    )


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str = LAUNCH_BOUND) -> SweepContract:
    """How to DRIVE a full-search readiness sweep, for the generic recorder.

    `kind` is accepted for interface parity and changes nothing here: this
    experiment has no phase-level path, so a diagnostic and a launch-bound sweep
    both write into the run. What the KIND decides is what the record CLAIMS,
    and only a `launch_bound` record may be what a launch rests on -- which the
    launcher's gate requires.
    """

    def bundle_name(commit: str) -> str:
        from experiments.phase_c2.full_search_bundle import canonical_bundle_name

        return canonical_bundle_name(commit)

    return SweepContract(
        experiment_id=EXPERIMENT_ID,
        record=record_contract(run_id, stage_id),
        groups=READINESS_GROUPS,
        launcher_module=LAUNCHER_MODULE,
        session_id=FSG.SESSION_ID,
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="full_search_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of the C2 pod selection, "
            f"{POD_TEST_SELECTION}/ -- everything the paid session can fail "
            "before its beam starts, and nothing else. The selection is "
            "Search-1's because both sessions stage the same assets and run the "
            "same beam machinery against the same teacher, so what a pod can "
            "fail beforehand is the same set of things. What this record BINDS "
            "is the FULL-SEARCH session: its launcher, its session id, its "
            "staging contract and its executable closure, whose derived joint "
            "space is a different search from Search-1's restricted one. A "
            "Search-1 readiness record cannot satisfy it, and it cannot satisfy "
            "Search-1's."),
    )


def verify_record(record: dict, repo_root: Any = ".", *, run_id: str,
                  stage_id: str | None = None, **kwargs):
    """The full-search record verification. `run_id` is required."""
    kwargs.setdefault("contract", record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root: Any = ".", *, run_id: str,
                stage_id: str | None = None) -> dict[str, Any]:
    """Read this run's full-search readiness record, or raise."""
    import json

    path = Path(repo_root) / record_path_for(run_id, stage_id)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


__all__ = ["EXPERIMENT_ID", "LAUNCHER_MODULE", "LAUNCH_BOUND",
           "POD_TEST_SELECTION", "READINESS_GROUPS", "RECORD_KINDS",
           "RUN_READINESS_ROLE", "SCHEMA", "ReadinessError", "harness",
           "harness_digest", "load_record", "record_contract",
           "record_path_for", "sweep_contract", "verify_record"]
