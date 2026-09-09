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
    BATTERY_SOURCE_NODEIDS,
    BATTERY_STAGED_ROLE_NODEID,
    DEVBOX_ONLY_NODEIDS,
    HOST_LOCAL_C1_NODEIDS,
    KNOWN_NON_ENVIRONMENT_SKIPS,
    LAUNCH_BOUND,
    LEAF_TRANSPORT_NODEIDS,
    RECORD_KINDS,
    RENDERER_PARITY_NODEIDS,
    REPOSITORY_STATE_NODEIDS,
    SCHEMA,
    compare_skip_sets,
    evaluate_sweep,
    head_commit,
    lineage_from_swept_base,
    read_junit,
    self_hash,
    skip_set_digest,
    tree_is_clean,
)

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


def verify_record(record: dict, repo_root=".", **kwargs):
    kwargs.setdefault("named_files", POD_TEST_ENVIRONMENT_FILES_V1)
    kwargs.setdefault("record_path", RECORD_PATH)
    return _pe.verify_record(record, repo_root, **kwargs)


#: C1's permitted post-sweep path, derived from the record it owns.
PERMITTED_POST_SWEEP_PATHS = _pe.permitted_post_sweep_paths(RECORD_PATH)

__all__ = ["RECORD_PATH", "POD_TEST_ENVIRONMENT_FILES_V1",
           "PERMITTED_POST_SWEEP_PATHS", "pod_test_environment_digest",
           "load_record", "verify_record"]
