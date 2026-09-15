"""C1's pod-environment readiness: which record, which non-harness files.

`aadistill.runtime.pod_environment` holds the mechanism -- how a sweep record is
verified, how the test-environment digest is computed, what lineage a session
commit must have -- and now names none of these. It used to carry C1's record
path and C1's tool list, which put one experiment inside a reusable runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from aadistill.runtime import pod_environment as _pe  # noqa: E402
#: The generic names, re-exported so a caller reads one namespace. This module
#: is C1's VIEW of the runtime mechanism: the mechanism is unchanged and lives
#: in `aadistill.runtime.pod_environment`; what is added here is which record
#: and which non-harness files C1 owns.
from aadistill.runtime.pod_environment import (  # noqa: E402,F401
    LAUNCH_BOUND,
    RECORD_KINDS,
    ReadinessGroups,
    RecordContract,
    compare_skip_sets,
    head_commit,
    lineage_from_swept_base,
    read_junit,
    self_hash,
    skip_set_digest,
    tree_is_clean,
)
from aadistill.runtime import pod_environment as _runtime  # noqa: E402

# --- C1's readiness contract -------------------------------------------------
#
# Nine node-id groups lived here until 2026-09-13: which of the repository's
# 3892 tests were expected to skip on a pod, which had to pass, which skipped for
# reasons that were not environmental, and prose explaining each. All of it
# existed to make one design work — running the whole repository on a billing
# GPU — and that design is gone. A C1 pod now runs `tests/c1_preflight/`, thirteen
# tests, and the contract fits in one screen.
#
# The groups are DELETED rather than left in place. They named tests the pod no
# longer collects, so every one of them would have reported "expected skip
# missing" forever, and a contract that cannot be satisfied is not a contract.
# The tests themselves are untouched and still run in development and CI.

#: There is no GPU half, and there must not be one. The shared CPU-test contract
#: sets `CUDA_VISIBLE_DEVICES=""` for the pytest command on BOTH machines, on
#: purpose, so that a predicate asking `torch.cuda.is_available()` cannot decide
#: one way in the sweep and the other on the pod. A "GPU-only" preflight module
#: would therefore skip on the pod too — it would test nothing and would look
#: like it had. The card is asserted where it can actually be seen: the setup
#: script checks `torch.cuda.is_available()` after the gate, outside the scope,
#: and the driver measures headroom against the real device before it trains.
RUNTIME_CONTRACT_MODULE = "tests/c1_preflight/test_c1_runtime_contract.py"

#: So the contract is the simplest one there is: every preflight test passes, on
#: the dev box, in the launch-bound sweep and on the pod, and NOTHING skips. Any
#: skip anywhere is a divergence, and `compare_skip_sets` names it.
C1_READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(RUNTIME_CONTRACT_MODULE,),
    refusal_notes={},
)


def evaluate_sweep(outcomes, skip_reasons=None, *, groups=None):
    """`evaluate_sweep` under C1's readiness contract."""
    return _runtime.evaluate_sweep(outcomes, skip_reasons,
                                   groups=groups or C1_READINESS_GROUPS)


#: THE WIRE FORMAT OF C1'S READINESS RECORD.
#:
#: `SCHEMA` and the `c1_harness_digest` key were defined in
#: `aadistill.runtime.pod_environment`, and `verify_record` compared against
#: them by name. A reusable runtime therefore accepted exactly one experiment's
#: schema and looked for its harness under exactly one experiment's key: a
#: second caller could not have had a readiness record at all.
#:
#: **Neither value moves.** The string is what every committed C1 record already
#: carries, and the field name is the key their self-hashes were computed over.
#: What changed is who owns them.
SCHEMA = "aadistill.autoinit.c1_pod_environment_verification/v1"

#: The GLOBAL entry point. A POINTER, not a second editable record: it names
#: the run whose governance area owns the live readiness evidence, together with
#: that record's hash. Kept because a reader needs one stable place to start.
#:
#: It was the canonical record, which meant every run in turn overwrote one file
#: and each closeout copied it away afterwards -- so the evidence a run was
#: launched under lived at a path the next run would replace.
RECORD_POINTER = "logs/stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json"

#: Back-compatible alias. Sweeps that name no run still write here, and every
#: record committed before 2026-09-12 is here.
RECORD_PATH = RECORD_POINTER

#: The run-owned location. Matches `C1_RUN_ROLES["readiness_record"]` in the
#: launcher, which is the run's own declared role for it.
RUN_READINESS_ROLE = "governance/readiness.json"


def record_path_for(run_id: str | None, stage_id: str | None = None) -> str:
    """Where THIS run's readiness record lives, repository-relative.

    `None` is the global pointer path, which is what a sweep with no run writes
    and what every pre-2026-09-12 record is at. A run id resolves through the
    same convention the launcher uses, so the sweep, the gate and the run's own
    manifest cannot disagree about where the record is.
    """
    if not run_id:
        return RECORD_POINTER
    from experiments.run_layout import rel_run_dir

    return f"{rel_run_dir('phase_c1', run_id, stage_id)}/{RUN_READINESS_ROLE}"

#: Files that decide the pod test gate's outcome and are OUTSIDE the C1 harness.
#: A list, not a glob, so each entry is a decision somebody made. Strictly
#: disjoint from the harness set: `verify_record` checks both digests, so a file
#: named in both buys nothing and only makes two lists look like independent
#: evidence when they are not.
POD_TEST_ENVIRONMENT_FILES_V1: tuple[str, ...] = (
    #: The simulator that creates the pod-like conditions. Not executed on a
    #: pod, so it has no place in the harness, but a change to it changes what
    #: the recorded sweep MEANT.
    "scripts/pod/simulate_pod_env.sh",
    #: A dev-box publishing tool the paid session never runs -- and whose tests
    #: the pod's setup gate does. That asymmetry is why it is measured here.
    "scripts/autoinit/publish_selected_leaves.py",
    #: The recorder decides what the record CLAIMS the sweep found. A parser
    #: that mislabelled a skip as a pass would certify a failing gate.
    "scripts/autoinit/record_pod_environment.py",
)


def pod_test_environment_digest(repo_root=".") -> dict:
    return _pe.pod_test_environment_digest(
        repo_root, named_files=POD_TEST_ENVIRONMENT_FILES_V1)


def load_record(repo_root=".", *, run_id: str | None = None,
                stage_id: str | None = None) -> dict:
    """This run's readiness record, or the global one when no run is named."""
    return _pe.load_record(repo_root,
                           record_path=record_path_for(run_id, stage_id))


def c1_harness_digest_value(repo_root=".") -> str:
    """C1's harness digest, as the contract's callable wants it.

    Imported inside the function: `experiments.phase_c1.authorization` imports
    from this module's neighbours, and a module-level import here would make the
    cycle real.
    """
    from experiments.phase_c1.authorization import c1_harness_digest

    return c1_harness_digest(repo_root)["digest"]


#: C1's complete wire contract, in one place. The launcher's
#: `pod_environment_gate` never passed a harness digest -- it called this
#: module's wrapper, which set only `named_files` and `record_path` -- so the
#: real launch path would have refused with "no harness_digest provider was
#: supplied" while every test passed one explicitly and never saw it. Carrying
#: the callable in the contract is what makes the tested path and the launch
#: path the same path.
def c1_record_contract(run_id: str | None = None,
                       stage_id: str | None = None) -> RecordContract:
    """C1's wire contract for a given run.

    The record path is the ONLY thing that varies, and everything derived from
    it follows -- including `permitted_post_sweep_paths`, which is why the
    lineage rule stays exactly as narrow as before: it names one file, this
    run's readiness record, and not the `governance/` directory it sits in.
    """
    return RecordContract(
        schema=SCHEMA,
        harness_field="c1_harness_digest",
        harness_digest=c1_harness_digest_value,
        record_path=record_path_for(run_id, stage_id),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="C1 harness",
    )


#: The no-run contract, for callers that have no session in hand.
C1_RECORD_CONTRACT = c1_record_contract()


def verify_record(record: dict, repo_root=".", **kwargs):
    kwargs.setdefault("contract", C1_RECORD_CONTRACT)
    return _pe.verify_record(record, repo_root, **kwargs)


def c1_sweep_contract(run_id: str | None = None,
                      stage_id: str | None = None,
                      kind: str | None = None):
    """How to DRIVE a C1 readiness sweep, for the generic recorder.

    `kind` is accepted and ignored: C1's record path does not depend on it,
    because `None` already resolves to the historical repository-root pointer
    path that every pre-2026-09-12 sweep wrote to.

    Every value here was a module-level constant, a hardcoded string or a
    top-level import inside `scripts/autoinit/record_pod_environment.py`. None of
    them changes: the schema, the two record key names, the pointer path and its
    schema, the session id and the prose are exactly what C1's existing records
    carry, because their self-hashes were computed over those bytes.
    """
    from aadistill.runtime.pod_environment import SweepContract

    def bundle_name(commit: str) -> str:
        from experiments.phase_c1.bundle import canonical_bundle_name

        return canonical_bundle_name(commit)

    def harness(repo_root):
        from experiments.phase_c1.authorization import c1_harness_digest

        return c1_harness_digest(repo_root)

    return SweepContract(
        experiment_id="phase_c1",
        record=c1_record_contract(run_id, stage_id),
        groups=C1_READINESS_GROUPS,
        launcher_module="autoinit_c1_launch",
        session_id="autoinit-c1",
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="c1_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of the CPU test suite: the condition a "
            "fresh C1 pod is actually in, which is what C1 attempt 3R's setup "
            "test gate refused for $0.3482 with zero scientific stages run."),
        pointer_path=RECORD_POINTER,
        pointer_schema="aadistill.autoinit.c1_readiness_pointer/v1",
        pointer_history=(
            "logs/stages/stage-1/phase_c1/history/readiness_history.json"),
        extra_record_fields={
            "renderer_parity_is_proved_by":
                "logs/stages/stage-1/phase_c1/validations/renderer-parity/"
                "c1_renderer_parity.json",
        },
    )


def permitted_post_sweep_paths(run_id: str | None = None,
                               stage_id: str | None = None) -> tuple[str, ...]:
    """The tracked paths that may differ after THIS run's sweep.

    Exactly two things, as before: the readiness record itself, and -- added by
    the caller -- the issued authorization. The record is now run-owned, so the
    permitted path is this run's readiness file. The `governance/` directory is
    NOT exempt: a grant committed after the sweep still invalidates it, which is
    the rule that forces grant-then-sweep.
    """
    return _pe.permitted_post_sweep_paths(record_path_for(run_id, stage_id))


#: C1's permitted post-sweep path when no run is named.
PERMITTED_POST_SWEEP_PATHS = permitted_post_sweep_paths()

__all__ = ["C1_RECORD_CONTRACT", "PERMITTED_POST_SWEEP_PATHS",
           "POD_TEST_ENVIRONMENT_FILES_V1", "RECORD_PATH", "SCHEMA",
           "c1_harness_digest_value", "c1_sweep_contract", "load_record",
           "pod_test_environment_digest", "verify_record"]
