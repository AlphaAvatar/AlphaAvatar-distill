"""C3's pod-environment readiness: which record, which non-harness files.

The generic recorder (`scripts/autoinit/record_pod_environment.py`) drives the
real `simulate_pod_env.sh` — empty HOME, isolated `HF_HOME`, synthetic
`HF_TOKEN`, gitignored artifacts hidden, the session's own pytest selection —
and writes the run's readiness record. Everything experiment-specific arrives
through the `SweepContract` declared here, which is why C3 needs this module
and not a copy of seven hundred lines.

**The harness digest is carried as a CALLABLE, not a value.** C1's real launch
path once refused with "no harness_digest provider was supplied" while every
test passed one explicitly and never saw it; carrying the callable in the
contract is what makes the tested path and the launch path the same path.

`kind` is accepted and ignored: C3's record path does not depend on it. A
`launch_bound` record is the one a funded launch rests on and is owed when the
grant is issued; a `diagnostic` one says only that the pod-like suite passes
on this tree.
"""

from __future__ import annotations

from typing import Any

from aadistill.runtime import pod_environment as _pe
from aadistill.runtime.pod_environment import (
    ReadinessGroups,
    RecordContract,
)
from aadistill.runtime import pod_environment as _runtime

#: The module whose outcome decides whether a C3 pod can run at all. It is the
#: preflight directory's runtime contract: the launcher/driver CLI seam, the
#: imports stage H needs after ten hours of training, the two digest gates, the
#: three-arm agreement and a real GEMM.
RUNTIME_CONTRACT_MODULE = "tests/c3_preflight/test_c3_runtime_contract.py"

C3_READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(RUNTIME_CONTRACT_MODULE,),
    refusal_notes={},
)

SCHEMA = "aadistill.autoinit.c3_pod_environment_verification/v1"

#: The GLOBAL entry point: a POINTER, not a second editable record. It names
#: the run whose governance area owns the live readiness evidence together
#: with that record's hash, so a reader has one stable place to start without
#: every run in turn overwriting one file.
RECORD_POINTER = ("logs/stages/stage-1/phase_c3/analyses/"
                  "c3_pod_environment_verification.json")
RECORD_PATH = RECORD_POINTER

#: Files that decide the pod test gate's outcome and are OUTSIDE the C3
#: harness. Disjoint from it by construction: `verify_record` checks both
#: digests, so a file in both would be measured twice and could satisfy one
#: check while failing the other.
POD_TEST_ENVIRONMENT_FILES_V1: tuple[str, ...] = (
    #: The simulator itself. It is what the record's command field ran, and a
    #: readiness record whose command was typed by hand is a claim rather than
    #: evidence.
    "scripts/pod/simulate_pod_env.sh",
    #: The suite's fixtures. A conftest change can turn a passing pod
    #: selection red without touching a single test the harness measures.
    "tests/conftest.py",
    #: The preflight directory's own contents are NOT here: they are reached
    #: by the sweep and reported by nodeid, and naming them would make the
    #: environment digest move every time a preflight test is added -- which
    #: is ordinary work, not a change in the pod's environment.
)


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where THIS run's readiness record lives, repository-relative."""
    if not run_id:
        return RECORD_POINTER
    from experiments.run_layout import rel_run_dir

    return f"{rel_run_dir('phase_c3', run_id, stage_id or '1')}/governance/readiness.json"


def c3_harness_digest_value(repo_root: str = ".") -> str:
    """C3's harness digest, as the contract's callable wants it."""
    from experiments.phase_c3.authorization import c3_harness_digest

    return c3_harness_digest(repo_root)["digest"]


def c3_record_contract(run_id: str | None = None,
                       stage_id: str | None = None) -> RecordContract:
    """C3's wire contract for a given run.

    The record path is the ONLY thing that varies, and everything derived
    from it follows -- including `permitted_post_sweep_paths`, which stays
    exactly as narrow as C1's: it names one file, this run's readiness
    record, and not the `governance/` directory it sits in.
    """
    return RecordContract(
        schema=SCHEMA,
        harness_field="c3_harness_digest",
        harness_digest=c3_harness_digest_value,
        record_path=record_path_for(run_id, stage_id),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="C3 harness",
    )


#: The no-run contract, for callers with no session in hand.
C3_RECORD_CONTRACT = c3_record_contract()


def evaluate_sweep(outcomes, skip_reasons=None, *, groups=None):
    """`evaluate_sweep` under C3's readiness contract."""
    return _runtime.evaluate_sweep(outcomes, skip_reasons,
                                   groups=groups or C3_READINESS_GROUPS)


def verify_record(record: dict, repo_root=".", **kwargs) -> tuple[bool, str]:
    kwargs.setdefault("contract", C3_RECORD_CONTRACT)
    return _pe.verify_record(record, repo_root, **kwargs)


def load_record(repo_root=".", *, run_id: str | None = None,
                stage_id: str | None = None, record_path: str | None = None):
    """THIS run's readiness record, or the pointer's when no run is named."""
    path = record_path or record_path_for(run_id, stage_id)
    return _pe.load_record(repo_root, record_path=path)


def c3_sweep_contract(run_id: str | None = None,
                      stage_id: str | None = None,
                      kind: str | None = None) -> Any:
    """How to DRIVE a C3 readiness sweep, for the generic recorder."""
    from aadistill.runtime.pod_environment import SweepContract

    def bundle_name(commit: str) -> str:
        from experiments.phase_c1.bundle import canonical_bundle_name

        return canonical_bundle_name(commit)

    def harness(repo_root):
        from experiments.phase_c3.authorization import c3_harness_digest

        return c3_harness_digest(repo_root)

    return SweepContract(
        experiment_id="phase_c3",
        record=c3_record_contract(run_id, stage_id),
        groups=C3_READINESS_GROUPS,
        launcher_module="autoinit_c3_launch",
        session_id="autoinit-c3",
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="c3_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of C3's own preflight selection: the "
            "condition a fresh C3 pod is actually in. C1 attempt 3R's setup "
            "test gate refused for $0.3482 with zero scientific stages run, "
            "which is what this record exists to make impossible to discover "
            "on a billing machine."),
        pointer_path=RECORD_POINTER,
        pointer_schema="aadistill.autoinit.c3_readiness_pointer/v1",
        pointer_history=(
            "logs/stages/stage-1/phase_c3/history/readiness_history.json"),
    )
