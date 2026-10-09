"""D1's BEHAVIOURAL pod-environment readiness: which record, which files.

The generic recorder (`scripts/shared/pod/record_pod_environment.py`) drives
the real `simulate_pod_env.sh` and writes the run's readiness record;
everything experiment-specific arrives through the `SweepContract` declared
here. The search's `pod_environment.py` serves the SEARCH launcher; this one
serves `autoinit_d1_behavioural_launch`, whose session id, pod test selection
and setup environment are a different session's -- a sweep certified under the
search's SESSION_KIND would describe an environment the behavioural pod is
never in.

**Without the registry entry the chain is unusable at exactly the step a
launch rests on**: the search's own launch-bound sweep was once refused as an
unknown experiment, so the sweep AGENTS.md P8.3 requires before a launch could
not be expressed at all. `phase_d1_behavioural` is that entry for this rung.

**Nothing here is transcribed from the launcher.** The session id, the pod
test selection and the staged view come from the real
`autoinit_d1_behavioural_launch.spec(...)`, the executable closure from
`behavioural_authorization`, and the canonical bundle name from the transport
owner. RUN-OWNED, no pointer: `pointer_path=None`, so no later run overwrites
this evidence and no global document drifts from it.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from aadistill.runtime import pod_environment as _pe
from aadistill.runtime import pod_environment as _runtime
from aadistill.runtime.pod_environment import (
    LAUNCH_BOUND,
    ReadinessGroups,
    RecordContract,
    SweepContract,
)

from stages.phase_d1 import behavioural_authorization as BA
from stages.phase_d1 import d1_session as D1S

REPO = Path(__file__).resolve().parents[4]

EXPERIMENT_ID = D1S.EXPERIMENT_ID
STAGE_ID = D1S.STAGE_ID

LAUNCHER_MODULE = "autoinit_d1_behavioural_launch"
SCHEMA = "aadistill.autoinit.d1_behavioural_pod_environment/v1"

#: The run-relative role. A sibling of `launch_readiness.json`, not a
#: replacement: this answers "can the pod's blocking CPU test gate pass under
#: the pod's own environment", the other answers "is this launch the one that
#: was prepared".
RUN_READINESS_ROLE = "governance/readiness.json"


class D1BehaviouralReadinessError(RuntimeError):
    """A behavioural readiness premise does not hold."""


@functools.lru_cache(maxsize=1)
def _spec() -> Any:
    """The REAL behavioural SessionSpec, loaded the way the recorder loads it."""
    import sys

    for extra in ("scripts/pod", "tests"):
        p = str(REPO / extra)
        if p not in sys.path:
            sys.path.insert(0, p)
    from support.session_specs import load_session_launcher, session_args

    launcher = load_session_launcher(LAUNCHER_MODULE)
    return launcher.spec(session_args(launcher))


def session_id() -> str:
    return _spec().session_id


def pod_test_selection() -> tuple[str, ...]:
    """What the behavioural pod gate actually collects:
    `SetupManifest.test_paths`. Derived, and REFUSED when empty -- a readiness
    group over zero modules passes vacuously."""
    paths = tuple(_spec().setup.test_paths)
    if not paths:
        raise D1BehaviouralReadinessError(
            "the behavioural session declares no test_paths, so its pod gate "
            "would collect the core suite and this contract would watch "
            "nothing")
    return paths


def readiness_groups() -> ReadinessGroups:
    """The whole claim: the declared pod test selection passes, with no
    unexpected environment skips."""
    return ReadinessGroups(
        expected_skips={},
        must_pass={},
        staged_role_nodeid=None,
        known_non_environment_skips=(),
        watched=pod_test_selection(),
        refusal_notes={},
    )


def pod_test_environment_files() -> tuple[str, ...]:
    """Files that decide the gate outcome and lie OUTSIDE the behavioural
    harness: the simulator, the two conftests that apply to a selection
    outside `tests/`, and every `.py` under the declared selection."""
    named = [
        "scripts/shared/pod/simulate_pod_env.sh",
        "conftest.py",
        "scripts/conftest.py",
    ]
    for selection in pod_test_selection():
        root = REPO / selection
        if not root.is_dir():
            raise D1BehaviouralReadinessError(
                f"the declared pod test selection {selection!r} is not a "
                "directory; digesting a smaller environment than the gate "
                "runs would certify a tree the pod does not have")
        named += [str(p.relative_to(REPO)) for p in sorted(root.rglob("*.py"))]
    missing = [rel for rel in named if not (REPO / rel).is_file()]
    if missing:
        raise D1BehaviouralReadinessError(
            f"declared pod test-environment sources are missing: {missing}")
    return tuple(sorted(set(named)))


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where THIS run's readiness record lives. A run id is REQUIRED."""
    if not run_id:
        raise D1BehaviouralReadinessError(
            "a behavioural readiness record has no location without a run "
            "id: it is evidence about one attempt")
    from shared.run_layout import rel_run_dir

    return (f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id or STAGE_ID)}"
            f"/{RUN_READINESS_ROLE}")


def harness(repo_root: Any = REPO) -> dict[str, Any]:
    """The LIVE behavioural executable closure, as the record's harness."""
    return BA.behavioural_current_executable(repo_root)


def harness_digest(repo_root: Any = REPO) -> str:
    return harness(repo_root)["digest"]


def bundle_name(commit: str) -> str:
    """D1's own canonical bundle name -- the behavioural session ships through
    the same transport the search does."""
    return D1S.canonical_bundle_name(commit)


def behavioural_record_contract(run_id: str | None = None,
                                stage_id: str | None = None) -> RecordContract:
    return RecordContract(
        schema=SCHEMA,
        harness_field="d1_behavioural_harness_digest",
        harness_digest=harness_digest,
        record_path=record_path_for(run_id, stage_id),
        named_files=pod_test_environment_files(),
        harness_label="D1 behavioural harness",
    )


def evaluate_sweep(outcomes, skip_reasons=None, *, groups=None):
    return _runtime.evaluate_sweep(outcomes, skip_reasons,
                                   groups=groups or readiness_groups())


def verify_record(record: dict, repo_root: Any = ".", *, run_id: str,
                  stage_id: str | None = None, **kwargs) -> tuple[bool, str]:
    kwargs.setdefault("contract",
                      behavioural_record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root: Any = ".", *, run_id: str,
                stage_id: str | None = None) -> dict[str, Any]:
    return _pe.load_record(repo_root,
                           record_path=record_path_for(run_id, stage_id))


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str | None = None) -> SweepContract:
    """How to DRIVE a behavioural readiness sweep, for the generic recorder."""
    return SweepContract(
        experiment_id=EXPERIMENT_ID,
        record=behavioural_record_contract(run_id, stage_id),
        groups=readiness_groups(),
        launcher_module=LAUNCHER_MODULE,
        session_id=session_id(),
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="d1_behavioural_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of D1's own declared pod selection — "
            f"{', '.join(pod_test_selection())} — under the environment the "
            "behavioural SessionRunner really hands setup: empty HOME, "
            "isolated HF_HOME, synthetic HF_TOKEN, every gitignored artifact "
            "this session does NOT stage hidden, and "
            "SESSION_KIND=d1_behavioural. It is the condition a fresh "
            "behavioural pod is actually in before its first arm "
            "materializes. RUN-OWNED: there is no pointer and no global "
            "readiness file, so no later run overwrites this evidence."),
        pointer_path=None,
    )


__all__ = ["EXPERIMENT_ID", "LAUNCHER_MODULE", "LAUNCH_BOUND",
           "RUN_READINESS_ROLE", "SCHEMA", "STAGE_ID",
           "D1BehaviouralReadinessError", "behavioural_record_contract",
           "bundle_name", "evaluate_sweep", "harness", "harness_digest",
           "load_record", "pod_test_environment_files", "pod_test_selection",
           "readiness_groups", "record_path_for", "session_id",
           "sweep_contract", "verify_record"]
