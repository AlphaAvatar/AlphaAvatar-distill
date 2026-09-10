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
# These nine node-id groups lived in `aadistill.runtime.pod_environment` until
# the Milestone-A closure. They are C1's: concrete test node ids, one session's
# expectation about which of its own tests skip on a pod and which must pass. A
# reusable runtime cannot own that, and its refusals cannot say "C1".
#
# Moved verbatim -- every node id, every reason and every count is unchanged, so
# the recorded sweeps stay comparable.

#: The seven parametrized cases that legitimately skip on a pod: they re-open the
#: pinned Hugging Face source snapshots, which are a dev-box readiness input and
#: never a C1 runtime or scientific one. Renderer parity itself is proved at $0 by
#: `scripts/autoinit/renderer_parity_gate.py`, which refuses a skip.
RENDERER_PARITY_NODEIDS: tuple[str, ...] = tuple(
    "tests/data/test_c1_battery.py::"
    f"test_the_shared_renderers_reproduce_the_frozen_battery_byte_for_byte[{g}]"
    for g in ("code", "gsm8k", "knowledge", "math_verified", "multihop", "rag",
              "tool"))

#: Five tests that must PASS, never skip, under an empty HOME and an isolated
#: HF cache. Every one uses monkeypatched network calls, so none is a test of
#: possessing a real credential.
#:
#: They are NOT the five unexplained attempt-3R failures. A $0 reproduction once
#: attributed those to this module, but it ran with no `HF_TOKEN` — a state no
#: pod is in, since setup exports one before the gate — and under the real pod
#: condition all five PASS. That attribution is WITHDRAWN and the five actual
#: failure identities remain unknown. This stays a mandatory regression set on
#: its own merits: it is the shape of failure that aborted recovery continuation
#: attempt 3 at $0.2011.
LEAF_TRANSPORT_NODEIDS: tuple[str, ...] = tuple(
    f"tests/autoinit/test_leaf_transport_publish.py::{n}" for n in (
        "test_a_corrupted_remote_file_is_caught_by_the_round_trip",
        "test_a_size_mismatch_at_the_far_end_is_caught",
        "test_an_lfs_oid_that_disagrees_is_caught_without_downloading",
        "test_a_file_absent_from_the_far_end_is_caught",
        "test_the_round_trip_needs_no_dev_box_directory"))

#: C1 construction-source cases that intentionally skip when their historical
#: source role is absent — which is every pod, because those roles are isolation
#: evidence and not C1 runtime inputs. They SKIP with the role named; they must
#: never pass vacuously, which is exactly what the `battery_v2` parameter did
#: until 2026-09-05 (an absent directory globbed to zero rows and the
#: disjointness assertions held trivially).
#:
#: `recovery_search_v2` is deliberately NOT here: C1 stages it as a local asset,
#: so its parameter of the same test must PASS on a pod.
BATTERY_SOURCE_NODEIDS: tuple[str, ...] = (
    "tests/data/test_c1_battery.py::"
    "test_it_is_disjoint_from_each_jsonl_role_by_id_and_by_content"
    "[artifacts/eval/battery_v2]",
    "tests/data/test_c1_battery.py::test_it_is_disjoint_from_the_recovery_training_corpus",
    "tests/data/test_c1_battery.py::test_final_promotion_is_still_intact_and_was_only_read",
)

#: The parameter that must still PASS on a pod, because C1 stages its source.
BATTERY_STAGED_ROLE_NODEID = (
    "tests/data/test_c1_battery.py::"
    "test_it_is_disjoint_from_each_jsonl_role_by_id_and_by_content"
    "[artifacts/stage3/recovery_search_v2]")

#: The staging-contract self-tests that describe the DEV BOX's staged/hidden
#: split. A pod never received the unstaged artifacts and the simulation has
#: moved them aside, so in both there is nothing left to prove hidden and they
#: skip. Declared here so they are exact rather than merely unnoticed.
#:
#: They skip because each ASKS THE FILESYSTEM for its own premise. Until the C1
#: attempt-5 postmortem they skipped because of `AAD_SYNTHETIC_HF_TOKEN`, a flag
#: the simulator sets and the pod does not — so this group was recorded
#: `skipped_as_expected: true` by a sweep while the same tests RAN on the pod and
#: failed. A group that names WHICH tests skipped cannot ask WHY, which is why
#: the reason now lives in the test and the complete skip set is recorded beside
#: this one.
DEVBOX_ONLY_NODEIDS: tuple[str, ...] = (
    "tests/autoinit/test_staging_contract.py::"
    "test_an_artifact_c1_does_not_stage_is_invisible",
    "tests/autoinit/test_staging_contract.py::"
    "test_an_undeclared_file_inside_a_staged_destination_stays_hidden",
    "tests/autoinit/test_staging_contract.py::"
    "test_the_dev_box_satisfies_both_premises_and_so_both_tests_run",
)

#: Host-local Phase-A integration cases, scoped by `SESSION_KIND=c1`. Their
#: premise is the retained leaf store at an absolute dev-box path that is
#: deliberately never staged to a pod. They are named SEPARATELY from the other
#: expected skips rather than folded into that count, because they skip for a
#: different reason: not "this source is absent here" but "this session does not
#: own that store". On a real pod they also skip, via their own store check.
HOST_LOCAL_C1_NODEIDS: tuple[str, ...] = (
    # Added 2026-09-06 with the CPU-test parity contract. `launcher.CKPT_STORE`
    # is located through `$HOME` now, so the canonical leaf store is invisible
    # under the contract's fresh empty HOME -- on the dev box AND on the pod,
    # which is the point. Before that it was a hardcoded absolute path, so this
    # case RAN in the diagnostic and would have SKIPPED on the pod: an
    # `unexpected_skip` that the strict comparison would have refused a healthy
    # pod for.
    "tests/pod/test_recovery_continuation_session.py::"
    "test_the_leaf_gate_reflects_whether_a_verified_transport_exists",
    "tests/pod/test_recovery_continuation_session.py::"
    "test_the_real_stage1_entrypoint_imports_measures_admits_and_hands_off",
    "tests/pod/test_recovery_continuation_session.py::"
    "test_the_entrypoint_refuses_a_substituted_leaf",
)

#: A legitimate skip inside a WATCHED module that is not environment-driven and
#: so is not part of the readiness-owned expected set. It is named rather than
#: silenced: widening the watch to that module surfaced it, and the honest answer
#: is an exemption with a reason, not a narrower watch that would also stop
#: noticing real staging skips there.
#:
#: `test_an_unverified_transport_declares_no_leaf_inputs` covers the branch taken
#: when NO verified transport exists. The transport IS verified, so the branch is
#: not live and the case skips — on the dev box exactly as in the simulation. It
#: has nothing to do with HOME, HF or staging.
KNOWN_NON_ENVIRONMENT_SKIPS: tuple[str, ...] = (
    "tests/pod/test_recovery_continuation_session.py::"
    "test_an_unverified_transport_declares_no_leaf_inputs",
)

#: Structural tests that failed on the pod for repository-state reasons and are
#: reported separately, because "fixed" was claimed for them once already.
REPOSITORY_STATE_NODEIDS: tuple[str, ...] = (
    "tests/docs/test_repository_structure.py::"
    "test_every_log_is_classified_in_the_catalog",
    "tests/pod/test_continuation_b_one_probe_contract.py::"
    "test_the_live_snapshot_records_the_terminal_phase_b_state",
)


#: What `evaluate_sweep` needs, assembled from the groups above. The group NAMES
#: are the record's key prefixes, so these reproduce the existing schema exactly
#: -- `renderer_parity_expected_skips`, `leaf_transport_all_passed` and the rest
#: are unchanged, which is what keeps the committed record readable by the gate.
C1_READINESS_GROUPS = ReadinessGroups(
    expected_skips={
        "renderer_parity": RENDERER_PARITY_NODEIDS,
        "battery_source": BATTERY_SOURCE_NODEIDS,
        "devbox_only": DEVBOX_ONLY_NODEIDS,
        "host_local_c1": HOST_LOCAL_C1_NODEIDS,
    },
    must_pass={
        "leaf_transport": LEAF_TRANSPORT_NODEIDS,
        "repository_state": REPOSITORY_STATE_NODEIDS,
    },
    staged_role_nodeid=BATTERY_STAGED_ROLE_NODEID,
    known_non_environment_skips=KNOWN_NON_ENVIRONMENT_SKIPS,
    #: The modules whose skips are watched at all. Widening this surfaced one
    #: legitimate non-environment skip, which is named above rather than hidden
    #: by narrowing the watch again.
    watched=("tests/data/test_c1_battery.py",
             "tests/autoinit/test_leaf_transport_publish.py",
             "tests/autoinit/test_staging_contract.py",
             "tests/pod/test_recovery_continuation_session.py"),
    refusal_notes={
        "host_local_c1": "They skip under SESSION_KIND=c1 because this session "
                         "does not own the retained leaf store.",
        "staged_role": "recovery_search_v2 is a local asset C1 DOES stage, so "
                       "its disjointness parameter must run on a pod, not skip.",
    },
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

#: Where the sweep writes what it found.
RECORD_PATH = "logs/c1_pod_environment_verification.json"

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


def load_record(repo_root=".") -> dict:
    return _pe.load_record(repo_root, record_path=RECORD_PATH)


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
C1_RECORD_CONTRACT = RecordContract(
    schema=SCHEMA,
    harness_field="c1_harness_digest",
    harness_digest=c1_harness_digest_value,
    record_path=RECORD_PATH,
    named_files=POD_TEST_ENVIRONMENT_FILES_V1,
    harness_label="C1 harness",
)


def verify_record(record: dict, repo_root=".", **kwargs):
    kwargs.setdefault("contract", C1_RECORD_CONTRACT)
    return _pe.verify_record(record, repo_root, **kwargs)


#: C1's permitted post-sweep path, derived from the record it owns.
PERMITTED_POST_SWEEP_PATHS = _pe.permitted_post_sweep_paths(RECORD_PATH)

__all__ = ["C1_RECORD_CONTRACT", "PERMITTED_POST_SWEEP_PATHS",
           "POD_TEST_ENVIRONMENT_FILES_V1", "RECORD_PATH", "SCHEMA",
           "c1_harness_digest_value", "load_record",
           "pod_test_environment_digest", "verify_record"]
