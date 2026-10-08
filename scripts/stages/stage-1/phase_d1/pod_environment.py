"""D1's pod-environment readiness: which record, which non-harness files.

The generic recorder (`scripts/shared/pod/record_pod_environment.py`) drives the
real `simulate_pod_env.sh` — empty HOME, isolated `HF_HOME`, synthetic
`HF_TOKEN`, gitignored artifacts hidden, the session's own pytest selection —
and writes the run's readiness record. Everything experiment-specific arrives
through the `SweepContract` declared here, which is why D1 needs this module and
not a copy of eight hundred lines.

**Without the registry entry the chain is unusable at exactly the step a launch
rests on**, which the recorder's own registry says five times over. D1's
launch-bound sweep was not merely unrun: `--experiment phase_d1` was refused as
an unknown experiment, so the sweep AGENTS.md P8.3 requires before a launch
could not be expressed at all.

**Nothing here is transcribed from the launcher.** The session id, the pod test
selection, the setup environment and the staged view all come from the real
`autoinit_d1_launch.spec(...)`, and the executable closure and the canonical
bundle name from the modules that own them. A second copy of any of those is a
second thing that can disagree with the session that actually launches.

**RUN-OWNED, with no pointer.** `SweepContract.pointer_path=None`, so there is
no repository-level file for a later run to overwrite and no global document to
drift from the evidence. `SweepContract`'s own docstring calls that "the better
default for anything new", and D1 is new. In particular this record does NOT
touch `logs/state/current.json`'s `latest_verification`, which is derived from
the Phase-C1 readiness pointer: making D1 answer to C1's pointer would be
making D1 pretend to be C1. D1's authority is this record.

**Two records, two different questions.** This writes `governance/readiness.json`
— can the pod's blocking CPU test gate pass under the pod's own environment.
`autoinit_d1_launch.readiness_path_for` gates on
`governance/launch_readiness.json` — is this launch the one that was prepared.
Neither substitutes for the other, and they are deliberately different files.
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

from stages.phase_d1 import d1_authorization as A
from stages.phase_d1 import d1_session as D1S

REPO = Path(__file__).resolve().parents[4]

#: From the session module, never retyped: `rel_run_dir` keys on both.
EXPERIMENT_ID = D1S.EXPERIMENT_ID
STAGE_ID = D1S.STAGE_ID

LAUNCHER_MODULE = "autoinit_d1_launch"
SCHEMA = "aadistill.autoinit.d1_pod_environment/v1"

#: The run-relative role. A sibling of `launch_readiness.json`, not a
#: replacement for it; see the module docstring.
RUN_READINESS_ROLE = "governance/readiness.json"


class D1ReadinessError(RuntimeError):
    """A D1 readiness premise does not hold. The recorder reports it as a
    refusal naming this experiment, never as a traceback."""


@functools.lru_cache(maxsize=1)
def _spec() -> Any:
    """The REAL `autoinit_d1_launch` SessionSpec, loaded the way the recorder
    loads it.

    This is what makes "derive, do not duplicate" true rather than intended:
    the session id, the pod test selection and the staged view below are read
    off the same object `SessionRunner` consumes. Memoized because
    `load_session_launcher` builds a fresh module per call and the recorder
    loads it again for its own purposes.
    """
    import sys

    for extra in ("scripts/pod", "tests"):
        p = str(REPO / extra)
        if p not in sys.path:
            sys.path.insert(0, p)
    from support.session_specs import load_session_launcher, session_args

    launcher = load_session_launcher(LAUNCHER_MODULE)
    return launcher.spec(session_args(launcher))


def session_id() -> str:
    """D1's session id, from the spec. C3 and A3 type theirs as a literal;
    D1 reads it, because a launcher that renamed its session would otherwise
    have its staged view derived under the old name."""
    return _spec().session_id


def pod_test_selection() -> tuple[str, ...]:
    """What D1's pod gate actually collects: `SetupManifest.test_paths`.

    Derived, and REFUSED when empty. An empty selection means the pod would run
    `tests/` — the core suite — and a readiness group watching nothing passes
    vacuously, which is the `battery_v2` defect in another costume.
    """
    paths = tuple(_spec().setup.test_paths)
    if not paths:
        raise D1ReadinessError(
            "the D1 session declares no test_paths, so its pod gate would "
            "collect the core suite and this contract would watch nothing. A "
            "readiness group over zero modules passes vacuously.")
    return paths


def readiness_groups() -> ReadinessGroups:
    """D1's readiness contract, and it is deliberately the simple one.

    No `expected_skips`, no `must_pass` and no `staged_role_nodeid`: D1 stages
    three small frozen assets whose own pins its driver verifies, and it has no
    battery and no C1-style readiness roles to inherit. Inheriting C1's groups
    would assert another experiment's expectations about another experiment's
    tests.

    What remains is the whole claim: **D1's declared pod test selection passes,
    with no unexpected environment skips.** `evaluate_sweep` fails the verdict
    on any failure or error, and `watched` makes any OTHER skip inside that
    selection a finding — a test that quietly stops running under an empty HOME
    is indistinguishable from one that never existed.
    """
    return ReadinessGroups(
        expected_skips={},
        must_pass={},
        staged_role_nodeid=None,
        known_non_environment_skips=(),
        watched=pod_test_selection(),
        refusal_notes={},
    )


def pod_test_environment_files() -> tuple[str, ...]:
    """Files that decide D1's gate outcome and lie OUTSIDE the D1 harness.

    `pod_test_environment_digest` always digests `tests/**/*.py` plus whatever
    is named here. For a session whose selection is INSIDE `tests/` that covers
    the tests the gate runs; D1's selection is not, and since the 2026-10-03
    boundary decision no new experiment's is.

    So the gate's real inputs are named here and DERIVED rather than listed:

    * the simulator, because a readiness record whose command was typed by hand
      is a claim rather than evidence;
    * the two conftests that apply to a selection outside `tests/` — the rootdir
      one and `scripts/conftest.py` — which is where the fixtures
      and the path setup for that selection come from;
    * every `.py` under D1's declared selection. Zero of them are in the
      executable closure (it is derived from the launcher, driver, session,
      authorization and collector, none of which import a test) and zero are
      under `tests/`, so without this a D1 test could change and the sweep that
      certified it would still verify. A new test is exactly as capable of
      failing on a pod as a new line of production code.

    Globbed at contract-construction time, which is live: the contract is built
    fresh for every sweep and every verification.
    """
    named = [
        "scripts/shared/pod/simulate_pod_env.sh",
        "conftest.py",
        "scripts/conftest.py",
    ]
    for selection in pod_test_selection():
        root = REPO / selection
        if not root.is_dir():
            raise D1ReadinessError(
                f"the declared pod test selection {selection!r} is not a "
                "directory; digesting a smaller environment than the gate runs "
                "would certify a tree the pod does not have")
        named += [str(p.relative_to(REPO)) for p in sorted(root.rglob("*.py"))]
    missing = [rel for rel in named if not (REPO / rel).is_file()]
    if missing:
        raise D1ReadinessError(
            f"declared pod test-environment sources are missing: {missing}")
    #: Sorted and de-duplicated: the digest is over sorted entries anyway, and a
    #: duplicate would be hashed twice.
    return tuple(sorted(set(named)))


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where THIS run's readiness record lives, repository-relative.

    A run id is REQUIRED. Launch-bound readiness is evidence about ONE attempt,
    and a shared default location is how the evidence a run launched under comes
    to live at a path the next run replaces. There is no phase-level fallback
    and no pointer.
    """
    if not run_id:
        raise D1ReadinessError(
            "a D1 readiness record has no location without a run id: it is "
            "evidence about one attempt, and a shared path is how one attempt's "
            "evidence comes to describe another's tree")
    from shared.run_layout import rel_run_dir

    return (f"{rel_run_dir(EXPERIMENT_ID, run_id, stage_id or STAGE_ID)}"
            f"/{RUN_READINESS_ROLE}")


def harness(repo_root: Any = REPO) -> dict[str, Any]:
    """The LIVE D1 executable closure, as the record's harness.

    A callable, evaluated when the sweep runs: a value captured at import time
    would describe whatever the tree was when somebody imported this module.
    `d1_current_executable` already returns `digest` and `n_files`, so no count
    is reconstructed here.
    """
    return A.d1_current_executable(repo_root)


def harness_digest(repo_root: Any = REPO) -> str:
    return harness(repo_root)["digest"]


def bundle_name(commit: str) -> str:
    """D1's OWN canonical bundle name.

    C3 and A3 both borrow C1's `canonical_bundle_name`. D1 has its own
    `TransportSpec`, and the sweep builds the production setup environment from
    this — so borrowing another experiment's deriver would put another
    experiment's bundle name into the environment the swept pytest runs under.
    Nothing is staged and nothing is uploaded by a sweep.
    """
    return D1S.canonical_bundle_name(commit)


def d1_record_contract(run_id: str | None = None,
                       stage_id: str | None = None) -> RecordContract:
    """D1's wire contract for a given run."""
    return RecordContract(
        schema=SCHEMA,
        harness_field="d1_harness_digest",
        harness_digest=harness_digest,
        record_path=record_path_for(run_id, stage_id),
        named_files=pod_test_environment_files(),
        harness_label="D1 harness",
    )


def evaluate_sweep(outcomes, skip_reasons=None, *, groups=None):
    """`evaluate_sweep` under D1's readiness contract."""
    return _runtime.evaluate_sweep(outcomes, skip_reasons,
                                   groups=groups or readiness_groups())


def verify_record(record: dict, repo_root: Any = ".", *, run_id: str,
                  stage_id: str | None = None, **kwargs) -> tuple[bool, str]:
    """D1's record verification. `run_id` is required: the record is run-owned."""
    kwargs.setdefault("contract", d1_record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root: Any = ".", *, run_id: str,
                stage_id: str | None = None) -> dict[str, Any]:
    """Read THIS run's readiness record, or raise."""
    return _pe.load_record(repo_root,
                           record_path=record_path_for(run_id, stage_id))


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str | None = None) -> SweepContract:
    """How to DRIVE a D1 readiness sweep, for the generic recorder.

    `kind` is accepted and changes nothing here: D1 has no phase-level path, so
    a diagnostic and a launch-bound sweep both write into the run. What the kind
    decides is what the record CLAIMS, and only a `launch_bound` record is one a
    funded launch rests on.
    """
    return SweepContract(
        experiment_id=EXPERIMENT_ID,
        record=d1_record_contract(run_id, stage_id),
        groups=readiness_groups(),
        launcher_module=LAUNCHER_MODULE,
        session_id=session_id(),
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="d1_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of D1's own declared pod selection — "
            f"{', '.join(pod_test_selection())} — under the environment the D1 "
            "SessionRunner really hands setup: empty HOME, isolated HF_HOME, "
            "synthetic HF_TOKEN, every gitignored artifact this session does "
            "NOT stage hidden, and SESSION_KIND=d1. It is the condition a fresh "
            "D1 pod is actually in before its beam starts. C1 attempt 3R's "
            "setup test gate refused for $0.3482 with zero scientific stages "
            "run, which is what this record exists to make impossible to "
            "discover on a billing machine. RUN-OWNED: there is no pointer and "
            "no global readiness file, so no later run overwrites this evidence "
            "and this record makes no claim about any other experiment."),
        #: NO POINTER. See the module docstring: D1's authority is this
        #: run-owned record, and it must not borrow C1's global pointer.
        pointer_path=None,
    )


__all__ = ["EXPERIMENT_ID", "LAUNCHER_MODULE", "LAUNCH_BOUND",
           "RUN_READINESS_ROLE", "SCHEMA", "STAGE_ID", "D1ReadinessError",
           "bundle_name", "d1_record_contract", "evaluate_sweep", "harness",
           "harness_digest", "load_record", "pod_test_environment_files",
           "pod_test_selection", "readiness_groups", "record_path_for",
           "session_id", "sweep_contract", "verify_record"]
