"""Baseline completion's pod-environment readiness: its record, its harness.

A thin contract over `aadistill.runtime.pod_environment`, which owns the whole
mechanism -- record verification, the test-environment digest, JUnit parsing,
the skip comparison, the lineage rule, the diagnostic/launch_bound distinction.
None of that is repeated here and none of it is this experiment's.

**Why a second instance rather than Search-1's.** The completion launcher was
calling the GENERIC module's `record_path_for(run_id, stage_id)` and
`load_record(...)` directly. Those are experiment-shaped questions -- which
experiment's runs, which schema, which harness, which staging contract -- and
the generic runtime does not own the answers. Asking it anyway meant the
completion session would have verified a record written under Search-1's
experiment id, against Search-1's harness digest and Search-1's staging
contract, while describing itself as ready.

So this instance declares what is the COMPLETION's:

* its own schema, so a Search-1 record cannot satisfy it;
* experiment id `phase_c2_baseline_completion`, so the record is owned by a
  completion run and a glob over one experiment's runs cannot reach another's;
* the completion launcher and session id, so the staged view is derived from the
  manifest the completion session actually declares;
* the LIVE completion closure as its harness, so the record binds the code that
  would run;
* its own harness field names, so a record's self-hash covers what it claims.

The pod test SELECTION is deliberately the same `tests/c2_preflight/` Search-1
uses: the environmental question -- can a fresh pod of this image run this
repository's preflight -- is identical, and inventing a second selection would
mean maintaining two answers to one question. What differs is everything the
record BINDS, which is the part that decides whether a launch may rest on it.
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

from experiments.phase_c2 import baseline_completion as BC
from experiments.phase_c2.pod_environment import (  # noqa: F401
    POD_TEST_ENVIRONMENT_FILES_V1, POD_TEST_SELECTION, ReadinessError,
)

#: Its OWN schema. This is the check that makes a Search-1 readiness record
#: unusable here: `verify_record` compares the record's schema against the
#: contract's, and the two strings differ.
SCHEMA = "aadistill.autoinit.c2_baseline_completion_pod_environment/v1"

EXPERIMENT_ID = "phase_c2_baseline_completion"

LAUNCHER_MODULE = "autoinit_phase_c2_baseline_launch"

#: The run's declared role for its readiness evidence, same convention as every
#: other role in a run's `governance/` area.
RUN_READINESS_ROLE = "governance/readiness.json"

#: Everything in the completion selection passes and nothing skips, enforced by
#: the machinery rather than by a list somebody maintains.
READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(POD_TEST_SELECTION,),
    refusal_notes={},
)


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where a completion readiness record lives, repository-relative.

    A run id is REQUIRED. Launch-bound readiness is evidence about ONE attempt,
    and a shared default location is how the evidence a run launched under comes
    to live at a path the next run replaces. There is no phase-level fallback
    here, and no diagnostic path either: this experiment has no history of one.
    """
    if not run_id:
        raise ReadinessError(
            "a baseline-completion readiness record has no location without a "
            "run id: it is evidence about one attempt, and a shared path is how "
            "one attempt's evidence comes to describe another's tree")
    from experiments.run_layout import rel_run_dir

    return f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id)}/{RUN_READINESS_ROLE}"


def harness(repo_root: Any = REPO) -> dict[str, Any]:
    """The LIVE completion closure, as the record's harness.

    A callable, evaluated when the sweep runs, because a value captured at
    import time would describe whatever the tree was when somebody imported
    this module.
    """
    live = BC.current_executable(repo_root)
    return {"digest": live["digest"], "n_files": live["n_files"],
            "files": [row["path"] for row in live["files"]]}


def harness_digest(repo_root: Any = REPO) -> str:
    return harness(repo_root)["digest"]


def record_contract(run_id: str | None = None,
                    stage_id: str | None = None) -> RecordContract:
    return RecordContract(
        schema=SCHEMA,
        harness_field="completion_harness_digest",
        harness_digest=harness_digest,
        record_path=record_path_for(run_id, stage_id),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="baseline-completion harness",
    )


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str = LAUNCH_BOUND) -> SweepContract:
    """How to DRIVE a completion readiness sweep, for the generic recorder.

    `kind` is accepted for interface parity with the other experiments and
    changes nothing here: this experiment has no phase-level path, so a
    diagnostic and a launch-bound sweep both write into the run. What the KIND
    decides is what the record CLAIMS, and only a `launch_bound` record may be
    what a launch rests on -- which the launcher's gate requires.
    """

    def bundle_name(commit: str) -> str:
        from experiments.phase_c2.baseline_completion_bundle import (
            canonical_bundle_name)

        return canonical_bundle_name(commit)

    return SweepContract(
        experiment_id=EXPERIMENT_ID,
        record=record_contract(run_id, stage_id),
        groups=READINESS_GROUPS,
        launcher_module=LAUNCHER_MODULE,
        session_id=BC.SESSION_ID,
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="completion_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of the baseline-completion pod "
            f"selection. The selection is {POD_TEST_SELECTION}/ -- the same "
            "environmental question Search-1 asks, because it is the same "
            "question about the same image and the same repository. What this "
            "record BINDS is the completion session: its launcher, its session "
            "id, its staging contract and its executable closure. A Search-1 "
            "readiness record cannot satisfy it, and it cannot satisfy "
            "Search-1's."),
        #: No pointer. The record is run-owned and there is nothing else to keep
        #: in step with it.
    )


def verify_record(record: dict, repo_root: Any = ".", *, run_id: str,
                  stage_id: str | None = None, **kwargs):
    """The completion's record verification. `run_id` is required."""
    kwargs.setdefault("contract", record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root: Any = ".", *, run_id: str,
                stage_id: str | None = None) -> dict[str, Any]:
    """Read this run's completion readiness record, or raise."""
    import json

    path = Path(repo_root) / record_path_for(run_id, stage_id)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())
