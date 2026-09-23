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

#: THE ONE CLASS OF CASE THIS SCOPE CANNOT RUN ON A POD, declared rather
#: than discovered. Each of these reads the out-of-tree durable store --
#: the five reconstructed replay products and the campaign's completed
#: probes, under `$HOME/aad-artifacts`. A session stages an asset's BYTES
#: and never the store they were frozen in, so no pod has it.
#:
#: This list was 28 and is now 8, and the shrinking is the point. Twenty of
#: those cases were not host-dependent at all: the CODE they covered asked
#: `candidate_manifest` for a parameter count the frozen selection already
#: commits, so it reached for host bytes a pod never receives. Marking them
#: made the sweep agree with the pod and silenced the tests that would have
#: caught that -- and stage C then failed on exactly that line, on a pod,
#: for $1.20. A skip that hides a defect is worse than a failure that shows
#: one.
#:
#: What remains genuinely needs the durable products' BYTE SIZES, not just
#: their identities: `storage_requirement` joins the selection to the
#: checkpoints on the machine that froze them, and no pod path reaches it --
#: the launcher's container gate runs on the dev box. Each is keyed on the
#: CONDITION rather than on any simulator marker, and the expectation is
#: pinned in BOTH directions: declaring a case here means the gate refuses
#: if it RUNS on a pod too.
HOST_LOCAL_STORE_CASES: tuple[str, ...] = (
    f"{POD_TEST_SELECTION}/test_behavioural_continuation.py::test_the_destination_is_charged_for_the_probes_this_session_produces",
    f"{POD_TEST_SELECTION}/test_behavioural_continuation.py::test_a_fresh_campaign_is_still_charged_for_every_probe",
    f"{POD_TEST_SELECTION}/test_behavioural_storage_lifecycle.py::TestTheGenericByteModel::test_the_footprint_reproduces_a_real_probe_on_disk",
    f"{POD_TEST_SELECTION}/test_behavioural_storage_lifecycle.py::test_the_probe_is_charged_at_its_save_size_not_its_leaf_size",
    f"{POD_TEST_SELECTION}/test_behavioural_storage_lifecycle.py::test_container_and_durable_are_separate_bounds",
    f"{POD_TEST_SELECTION}/test_behavioural_storage_lifecycle.py::TestTheContainerGate::test_it_passes_the_authorized_provision",
    f"{POD_TEST_SELECTION}/test_behavioural_storage_lifecycle.py::TestTheContainerGate::test_it_refuses_a_provision_that_cannot_hold_the_work",
    f"{POD_TEST_SELECTION}/test_behavioural_storage_lifecycle.py::TestTheContainerGate::test_it_reads_the_flag_rather_than_the_constant",
)


#: Everything else in the selection passes. A `skipif` keyed on a simulator-set
#: marker would be INVERTED on the pod and a gate can check THAT a test skipped
#: but never WHY, so the only skips admitted here are keyed on the CONDITION --
#: whether this machine has the store -- and each one is named above.
READINESS_GROUPS = ReadinessGroups(
    expected_skips={"host_local_durable_stores": HOST_LOCAL_STORE_CASES},
    must_pass={},
    staged_role_nodeid=None,
    known_non_environment_skips=(),
    watched=(POD_TEST_SELECTION,),
    refusal_notes={"host_local_durable_stores": (
        "these cases read $HOME/aad-artifacts, which a pod never "
        "receives; they must skip there and must run on a dev box "
        "that has it")},
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
