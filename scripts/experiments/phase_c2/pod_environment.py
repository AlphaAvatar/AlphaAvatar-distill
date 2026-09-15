"""C2's pod-environment readiness: which record, which non-harness files.

A thin contract over `aadistill.runtime.pod_environment`, which owns the whole
mechanism — the record's verification, the test-environment digest, the JUnit
parsing, the skip comparison, the lineage rule, the diagnostic/launch_bound
distinction. Nothing of that is repeated here and nothing of it is C2's.

Three things make this shorter than C1's, and each is a consequence of what C2
runs rather than a simplification:

**No expected-skip map.** C1's version once declared nine node-id groups
describing which of 3892 tests should skip on a pod. C2's pod selection is
`tests/c2_preflight/`, whose last test refuses any conditional marker in the
directory, so the contract is: everything passes and nothing skips. `watched`
carries the selection, which makes any skip there an unexpected environment skip
and the verdict FAIL — enforced by the machinery, not by a list somebody keeps.

**No pointer.** C1's readiness record was a repository-root file that every
attempt overwrote, and the pointer exists to unwind that: the record moved into
the run and the old path became navigation. C2 starts where C1 arrived — the
record is run-owned, `record_path_for` refuses without a run, and there is no
second file to drift from it.

**No GPU half, deliberately.** The shared CPU-test contract sets
`CUDA_VISIBLE_DEVICES=""` for the pytest command on both the dev box and the
pod, so a predicate asking `torch.cuda.is_available()` answers the same way in
both and a "GPU-only" preflight module would skip on the pod as well — testing
nothing while looking like it had. The card is asserted where it can be seen:
the setup script checks it after the gate, and the driver measures the real
device before it searches.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.runtime import pod_environment as _pe  # noqa: E402
#: The generic names, re-exported so a caller reads one namespace. The mechanism
#: is unchanged and lives in `aadistill.runtime.pod_environment`; what is added
#: here is which record and which non-harness files C2 owns.
from aadistill.runtime.pod_environment import (  # noqa: E402,F401
    LAUNCH_BOUND,
    RECORD_KINDS,
    ReadinessGroups,
    RecordContract,
    SweepContract,
    compare_skip_sets,
    head_commit,
    lineage_from_swept_base,
    read_junit,
    self_hash,
    skip_set_digest,
    tree_is_clean,
)

#: THE WIRE FORMAT OF C2'S READINESS RECORD. Its own schema, because a C1 record
#: measures a different harness under a different staging contract and
#: `verify_record` compares this string for equality.
SCHEMA = "aadistill.autoinit.c2_pod_environment_verification/v1"

#: The pod selection this contract is about, named once. `ignores_for_selection`
#: in the launcher derives the complement of the same directory; a test holds
#: the two equal so the readiness contract cannot describe a different suite
#: from the one the pod runs.
POD_TEST_SELECTION = "tests/c2_preflight"

#: The run's declared role for its readiness evidence. Same convention as every
#: other role in a run's `governance/` area.
RUN_READINESS_ROLE = "governance/readiness.json"

#: Files that decide the pod test gate's outcome and are OUTSIDE the C2 harness.
#: A list, not a glob, so each entry is a decision somebody made. Strictly
#: disjoint from the harness closure: `verify_record` checks both digests, so a
#: file in both buys nothing and only makes two lists look like independent
#: evidence when they are not.
#:
#: C1 names a third file here, `scripts/autoinit/publish_selected_leaves.py`: a
#: dev-box tool the paid session never runs and whose tests its pod gate did.
#: C2's gate collects `tests/c2_preflight/` only, so that asymmetry does not
#: exist here and declaring the file would measure something this session's gate
#: cannot reach.
POD_TEST_ENVIRONMENT_FILES_V1: tuple[str, ...] = (
    #: The simulator that creates the pod-like conditions. Never executed on a
    #: pod, so it has no place in the harness — but a change to it changes what
    #: a recorded sweep MEANT.
    "scripts/pod/simulate_pod_env.sh",
    #: The recorder decides what the record CLAIMS the sweep found. A parser
    #: that mislabelled a skip as a pass would certify a failing gate.
    "scripts/autoinit/record_pod_environment.py",
)

#: The contract, and the whole of it: every selected test passes, on the dev
#: box, in the launch-bound sweep and on the pod, and nothing skips.
C2_READINESS_GROUPS = ReadinessGroups(
    expected_skips={},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(POD_TEST_SELECTION,),
    refusal_notes={},
)


class ReadinessError(RuntimeError):
    """The readiness record cannot be located or does not describe this tree."""


#: Where a sweep that belongs to no attempt writes. A `diagnostic` record
#: proves the machinery and that the selection passes under pod conditions on a
#: given tree; it is a fact about the PHASE, not about one attempt, and filing
#: it inside a run would attribute evidence to an attempt that did not produce
#: it. A `launch_bound` record is the opposite — it is what one attempt's money
#: rests on — and may never live here.
PHASE_DIAGNOSTIC_RECORD = ("logs/stages/stage-1/phase_c2/analyses/"
                           "c2_pod_environment_diagnostic.json")


def record_path_for(run_id: str | None, stage_id: str | None = None, *,
                    phase_diagnostic: bool = False) -> str:
    """Where a readiness record lives, repository-relative.

    A run id is REQUIRED for anything a launch can rest on, unlike C1's, where
    `None` resolves to the historical repository-root path. C2 has no such
    history and no such file: launch-bound readiness is evidence about one
    attempt, and a shared default location is how the evidence a run launched
    under came to live at a path the next run replaced.

    `phase_diagnostic=True` is the one exception, and it is narrow: a sweep with
    no attempt writes the phase-level diagnostic above. The recorder passes it
    only for `--kind diagnostic`, so a launch-bound sweep still cannot be taken
    without a run to own it.
    """
    if not run_id:
        if phase_diagnostic:
            return PHASE_DIAGNOSTIC_RECORD
        raise ReadinessError(
            "a launch-bound C2 readiness record belongs to a run: pass the run "
            "id (and the stage its experiment declares). There is no "
            "repository-level readiness file for C2, deliberately — a shared "
            "path is how one attempt's evidence gets overwritten by the next.")
    from experiments.run_layout import rel_run_dir

    return f"{rel_run_dir('phase_c2', run_id, stage_id)}/{RUN_READINESS_ROLE}"


def pod_test_environment_digest(repo_root=".") -> dict:
    return _pe.pod_test_environment_digest(
        repo_root, named_files=POD_TEST_ENVIRONMENT_FILES_V1)


def load_record(repo_root=".", *, run_id: str, stage_id: str | None = None) -> dict:
    return _pe.load_record(repo_root,
                           record_path=record_path_for(run_id, stage_id))


def evaluate_sweep(outcomes, skip_reasons=None, *, groups=None):
    """`evaluate_sweep` under C2's readiness contract."""
    return _pe.evaluate_sweep(outcomes, skip_reasons,
                              groups=groups or C2_READINESS_GROUPS)


def c2_harness_digest_value(repo_root=".") -> str:
    """C2's harness digest — the LIVE derived executable closure.

    Imported inside the function because `experiments.phase_c2.session` reaches
    this module's neighbours; a module-level import would make the cycle real.
    """
    from experiments.phase_c2.session import c2_harness_digest

    return c2_harness_digest(repo_root)["digest"]


def c2_record_contract(run_id: str | None = None,
                       stage_id: str | None = None, *,
                       phase_diagnostic: bool = False) -> RecordContract:
    """C2's wire contract for a given run.

    The harness digest is carried as the callable, not supplied per call site:
    C1's launcher gate once called a wrapper that set only `named_files` and
    `record_path`, so the real launch path refused with "no harness_digest
    provider was supplied" while every test passed one explicitly and never saw
    it.
    """
    return RecordContract(
        schema=SCHEMA,
        harness_field="c2_harness_digest",
        harness_digest=c2_harness_digest_value,
        record_path=record_path_for(run_id, stage_id,
                                    phase_diagnostic=phase_diagnostic),
        named_files=POD_TEST_ENVIRONMENT_FILES_V1,
        harness_label="C2 harness",
    )


def c2_sweep_contract(run_id: str | None = None,
                      stage_id: str | None = None,
                      kind: str = LAUNCH_BOUND) -> SweepContract:
    """How to DRIVE a C2 readiness sweep, for the generic recorder.

    `kind` decides only where a run-less sweep writes: a `diagnostic` with no
    attempt lands at the phase-level path, and anything a launch could rest on
    still needs a run to own it.
    """

    def bundle_name(commit: str) -> str:
        from experiments.phase_c2.bundle import canonical_bundle_name

        return canonical_bundle_name(commit)

    def harness(repo_root):
        from experiments.phase_c2.session import c2_harness_digest

        return c2_harness_digest(repo_root)

    return SweepContract(
        experiment_id="phase_c2",
        #: `None` means the caller said nothing about the kind, which is the
        #: STRICT reading: only an explicit `diagnostic` may write outside a
        #: run. Treating an unstated kind as a diagnostic would let a caller
        #: reach the phase-level path by omission.
        record=c2_record_contract(
            run_id, stage_id,
            phase_diagnostic=kind is not None and kind != LAUNCH_BOUND),
        groups=C2_READINESS_GROUPS,
        launcher_module="autoinit_phase_c2_launch",
        session_id="autoinit-phase-c2-search1",
        bundle_name=bundle_name,
        harness=harness,
        harness_n_files_field="c2_harness_n_files",
        what_this_is=(
            "one complete pod-like sweep of the C2 pod selection: the condition "
            "a fresh Phase-C2 Search-1 pod is actually in. The selection is "
            f"{POD_TEST_SELECTION}/ — everything the paid session can fail "
            "before its search begins, and nothing else. A $0 probe measured "
            "what the previous whole-repository selection cost: 3976 tests, 65 "
            "skips, 907 s of an L40S gate."),
        #: No pointer. The record is run-owned and there is nothing else to
        #: keep in step with it.
    )


def verify_record(record: dict, repo_root=".", *, run_id: str,
                  stage_id: str | None = None, **kwargs):
    """C2's record verification. `run_id` is required, as everywhere here."""
    kwargs.setdefault("contract", c2_record_contract(run_id, stage_id))
    return _pe.verify_record(record, repo_root, **kwargs)


def permitted_post_sweep_paths(run_id: str, stage_id: str | None = None
                               ) -> tuple[str, ...]:
    """The tracked paths that may differ after THIS run's sweep.

    One file: this run's readiness record. The `governance/` directory it sits
    in is NOT exempt — a grant committed after the sweep still invalidates it,
    which is the rule that forces grant-then-sweep-then-issue.
    """
    return _pe.permitted_post_sweep_paths(record_path_for(run_id, stage_id))


__all__ = ["C2_READINESS_GROUPS", "POD_TEST_ENVIRONMENT_FILES_V1",
           "POD_TEST_SELECTION", "SCHEMA", "ReadinessError",
           "c2_harness_digest_value", "c2_record_contract",
           "c2_sweep_contract", "evaluate_sweep", "load_record",
           "permitted_post_sweep_paths", "pod_test_environment_digest",
           "record_path_for", "verify_record"]
