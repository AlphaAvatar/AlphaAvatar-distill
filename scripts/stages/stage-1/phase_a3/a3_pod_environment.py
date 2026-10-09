"""A3's pod-environment readiness: which record, which non-harness files.

A SEPARATE module rather than a mode of `pod_environment`, for the reason the
recorder's own registry gives five times over: A3 binds its own launcher, its
own session id, its own executable closure, its own pod test selection
(`scripts/stages/stage-1/phase_c3/tests`, which C3's sweep would not run) and its own record
schema. A C3 record cannot satisfy an A3 verifier or the reverse.

**Without the registry entry the chain is unusable at exactly the step a
launch rests on.** The acquisition loop's second step is
`record_pod_environment.py --experiment phase_a3 --kind launch_bound`, and
with no entry that call is refused with a message about an unknown experiment
— while `readiness_gate` would refuse forever afterwards with a message about
a missing record rather than a missing registration. `SESSION_KIND=phase_b`
had no branch in the setup script and cost `$0.23` to discover; this one was
found at `$0`.

**The harness digest is carried as a CALLABLE, not a value**, because C1's
real launch path once refused with "no harness_digest provider was supplied"
while every test passed one explicitly and never saw it.

`kind` is accepted and ignored: A3's record path does not depend on it.
"""

from __future__ import annotations

from typing import Any

from aadistill.runtime import pod_environment as _pe
from aadistill.runtime import pod_environment as _runtime
from aadistill.runtime.pod_environment import ReadinessGroups, RecordContract

#: The module whose outcome decides whether an A3 pod can run at all: the
#: launcher/driver CLI seam in both directions, the two shelled-out seams, the
#: imports stage H needs after three trainings and three evaluations, the
#: asymmetric digest gates, the absent stage I, and a real GEMM.
RUNTIME_CONTRACT_MODULE = "scripts/stages/stage-1/phase_c3/tests/test_a3_runtime_contract.py"

A3_READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(RUNTIME_CONTRACT_MODULE,),
    refusal_notes={},
)

SCHEMA = "aadistill.autoinit.a3_pod_environment_verification/v1"

#: The GLOBAL entry point: a POINTER, not a second editable record.
RECORD_POINTER = ("logs/stages/stage-1/phase_a3/analyses/"
                  "a3_pod_environment_verification.json")
RECORD_PATH = RECORD_POINTER

#: Files that decide the pod test gate's outcome and are OUTSIDE the A3
#: harness. Disjoint from it by construction, because `verify_record` checks
#: both digests and a file in both would be measured twice.
POD_TEST_ENVIRONMENT_FILES_V1: tuple[str, ...] = (
    "scripts/shared/pod/simulate_pod_env.sh",
    "tests/conftest.py",
)


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where THIS run's readiness record lives, repository-relative.

    The same path `autoinit_a3_launch.readiness_path_for` gates on, derived
    through the same `rel_run_dir`, so the recorder writes where the launcher
    looks. Two owners of one path is how a record gets written somewhere
    nothing reads.
    """
    if not run_id:
        return RECORD_POINTER
    from shared.run_layout import rel_run_dir

    return (f"{rel_run_dir('phase_a3', run_id, stage_id or '1')}"
            "/governance/readiness.json")


def a3_harness_digest_value(repo_root: str = ".") -> str:
    """A3's harness digest, as the contract's callable wants it."""
    from stages.phase_a3.a3_authorization import a3_harness_digest

    return a3_harness_digest(repo_root)["digest"]


def a3_record_contract(run_id: str | None = None,
                       stage_id: str | None = None) -> RecordContract:
    """A3's wire contract for a given run."""
    return RecordContract(
        schema=SCHEMA,
        harness_field="a3_harness_digest",
        harness_digest=a3_harness_digest_value,
        record_path=record_path_for(run_id, stage_id),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="A3 harness",
    )


#: The no-run contract, for callers with no session in hand.
A3_RECORD_CONTRACT = a3_record_contract()


def evaluate_sweep(outcomes, skip_reasons=None, *, groups=None):
    return _runtime.evaluate_sweep(outcomes, skip_reasons,
                                   groups=groups or A3_READINESS_GROUPS)


def verify_record(record: dict, repo_root=".", **kwargs) -> tuple[bool, str]:
    kwargs.setdefault("contract", A3_RECORD_CONTRACT)
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root=".", *, run_id: str | None = None,
                stage_id: str | None = None, record_path: str | None = None):
    """THIS run's readiness record, or the pointer's when no run is named."""
    path = record_path or record_path_for(run_id, stage_id)
    return _pe.load_record(repo_root, record_path=path)


def sweep_contract(run_id: str | None = None, stage_id: str | None = None,
                   kind: str | None = None) -> Any:
    """How to DRIVE an A3 readiness sweep, for the generic recorder."""
    from aadistill.runtime.pod_environment import SweepContract

    def bundle_name(commit: str) -> str:
        from stages.phase_c1.bundle import canonical_bundle_name

        return canonical_bundle_name(commit)

    def harness(repo_root):
        from stages.phase_a3.a3_authorization import a3_harness_digest

        #: The recorder records a COUNT beside the digest, under the field
        #: named below. C3's harness comes from `closure.derive`, which
        #: returns `n_files`; A3's comes from `harness_source_digest` over an
        #: explicitly DECLARED file set, which returns `{digest, files,
        #: set_version}` and no count -- and the recorder raised `KeyError:
        #: 'n_files'` after running the whole sweep. Counted here rather than
        #: switching A3 to `derive`, because the declared set is the thing
        #: the frozen-set declarations and the grant both bind, and changing
        #: how it is digested to obtain one integer would move that digest.
        h = dict(a3_harness_digest(repo_root))
        h["n_files"] = len(h["files"])
        return h

    return SweepContract(
        experiment_id="phase_a3",
        record=a3_record_contract(run_id, stage_id),
        groups=A3_READINESS_GROUPS,
        launcher_module="autoinit_a3_launch",
        session_id="autoinit-a3",
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="a3_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of A3's own preflight selection: the "
            "condition a fresh A3 pod is actually in. A3's launcher declared "
            "NO pod test selection, so without the derived complement this "
            "sweep would have described the whole repository suite running on "
            "a billing L40S -- which is also what the pod would have done."),
        pointer_path=RECORD_POINTER,
        pointer_schema="aadistill.autoinit.a3_readiness_pointer/v1",
        pointer_history=(
            "logs/stages/stage-1/phase_a3/history/readiness_history.json"),
    )
