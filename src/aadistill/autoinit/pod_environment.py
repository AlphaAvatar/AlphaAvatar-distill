"""Proof that the pod's CPU test gate can pass, bound to the tree that proved it.

C1 attempt 3R reached `VLLM_READY → TEACHER_READY → ROPE_OK` and then died at the
setup test gate: `14 failed, 2650 passed`, `$0.3482`, no scientific stage. Twelve
of the fourteen were environment, not code — tests that read `$HOME` for a
Hugging Face cache or credential the pod does not have. They passed on the dev box
and could not pass on a pod, and three launches went by without anyone finding
out, because no C1 attempt had ever reached `TESTS_OK` before.

The suite that answers "would this pass on a pod?" takes roughly three quarters
of an hour, which is far too slow to run inside a pre-provider gate while a pod
waits. So it is run once, deliberately, and what it produced is recorded here —
and the cheap gate checks that the recording still describes the code that would
actually run.

That binding is to the **executable**, never to `HEAD`. The normal order of work
commits the executable first, runs the sweep against it, then writes a
preregistration and updates documentation in later commits; a gate keyed on the
commit hash would go stale the moment the paperwork landed, and the obvious way
to make it green again would be to re-run an hour-long sweep that had not changed
in any way that mattered. Two digests decide instead:

* the C1 harness digest — everything the paid session executes;
* the pod **test environment** digest — everything that decides what the pod's
  test gate does and is outside that harness: the setup script that runs the
  gate, the simulator that models it, the whole test tree, and the modules whose
  `$HOME` assumptions caused the abort in the first place.

Change any of them and the record no longer describes the gate; the sweep is owed
again. Change a log, a decision record or a README, and it does not.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SCHEMA = "aadistill.autoinit.c1_pod_environment_verification/v1"
RECORD_PATH = "logs/c1_pod_environment_verification.json"

#: Files outside `C1_HARNESS_SOURCE_FILES_V1` that decide the pod test gate's
#: outcome. A list, not a glob, so each entry is a decision somebody made.
#:
#: `autoinit_preflight_setup.sh` is the pod setup the session runner actually
#: uploads and runs — `session_runner.py:528`. It is NOT in the C1 harness set,
#: although every other phase's set names it. Until that is resolved, binding it
#: here is what stops an edit to the pod's own test gate from leaving a readiness
#: record that no longer describes it.
POD_TEST_ENVIRONMENT_FILES_V1: tuple[str, ...] = (
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/simulate_pod_env.sh",
    "scripts/data/battery_render.py",
    "scripts/autoinit/publish_selected_leaves.py",
    "scripts/autoinit/renderer_parity_gate.py",
)

#: The seven parametrized cases that legitimately skip on a pod: they re-open the
#: pinned Hugging Face source snapshots, which are a dev-box readiness input and
#: never a C1 runtime or scientific one. Renderer parity itself is proved at $0 by
#: `scripts/autoinit/renderer_parity_gate.py`, which refuses a skip.
RENDERER_PARITY_NODEIDS: tuple[str, ...] = tuple(
    "tests/data/test_c1_battery.py::"
    f"test_the_shared_renderers_reproduce_the_frozen_battery_byte_for_byte[{g}]"
    for g in ("code", "gsm8k", "knowledge", "math_verified", "multihop", "rag",
              "tool"))

#: The five that must PASS, never skip. They drove attempt 3R's diagnosis and
#: every one of them uses monkeypatched network calls, so none is a test of
#: possessing a real credential.
LEAF_TRANSPORT_NODEIDS: tuple[str, ...] = tuple(
    f"tests/autoinit/test_leaf_transport_publish.py::{n}" for n in (
        "test_a_corrupted_remote_file_is_caught_by_the_round_trip",
        "test_a_size_mismatch_at_the_far_end_is_caught",
        "test_an_lfs_oid_that_disagrees_is_caught_without_downloading",
        "test_a_file_absent_from_the_far_end_is_caught",
        "test_the_round_trip_needs_no_dev_box_directory"))

#: Structural tests that failed on the pod for repository-state reasons and are
#: reported separately, because "fixed" was claimed for them once already.
REPOSITORY_STATE_NODEIDS: tuple[str, ...] = (
    "tests/docs/test_repository_structure.py::"
    "test_every_log_is_classified_in_the_catalog",
    "tests/pod/test_continuation_b_one_probe_contract.py::"
    "test_the_live_snapshot_records_the_terminal_phase_b_state",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pod_test_environment_digest(repo_root: str | Path = ".") -> dict[str, Any]:
    """Digest over everything that decides the pod test gate's outcome.

    The whole test tree is in here on purpose. A new test is exactly as capable
    of failing on a pod as a new line of production code — that is the entire
    lesson of attempt 3R — so a readiness record must not survive one.
    """
    root = Path(repo_root)
    rels = sorted(
        {str(p.relative_to(root)) for p in (root / "tests").rglob("*.py")}
        | set(POD_TEST_ENVIRONMENT_FILES_V1))
    entries = []
    for rel in rels:
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(
                f"declared pod test-environment source {rel!r} is missing; "
                "refusing to digest a smaller environment than the one the pod "
                "gate would run")
        entries.append({"path": rel, "sha256": _sha256_file(p)})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "n_files": len(entries),
            "named_files": list(POD_TEST_ENVIRONMENT_FILES_V1),
            "rule": ("sha256 over sorted 'path:sha256' lines of tests/**/*.py "
                     "plus the named non-harness files")}


# --- reading what the sweep actually did ------------------------------------

def _nodeid(case: ET.Element) -> str:
    """Reconstruct a pytest nodeid from a JUnit `testcase` element."""
    f = case.get("file") or ""
    cls = case.get("classname") or ""
    name = case.get("name") or ""
    mod = f[:-3].replace("/", ".") if f.endswith(".py") else ""
    if cls and mod and cls.startswith(mod + "."):
        inner = cls[len(mod) + 1:].replace(".", "::")
        return f"{f}::{inner}::{name}"
    return f"{f}::{name}"


def read_junit(path: str | Path) -> dict[str, Any]:
    """Per-nodeid outcomes from a JUnit XML report.

    JUnit is a *reporting* flag: it changes nothing about which tests are
    selected or how they run, so the sweep still executes the pod's own command.
    It is the only way to name every skip and every pass exactly, and naming them
    is the point — attempt 3R's four-line tail is why fourteen failures arrived
    as three.
    """
    tree = ET.parse(str(path))
    outcomes: dict[str, str] = {}
    for case in tree.iter("testcase"):
        nid = _nodeid(case)
        status = "passed"
        for child in case:
            tag = child.tag.lower()
            if tag in ("failure", "error", "skipped"):
                status = {"failure": "failed", "error": "error",
                          "skipped": "skipped"}[tag]
                break
        outcomes[nid] = status
    counts = {s: sum(1 for v in outcomes.values() if v == s)
              for s in ("passed", "skipped", "failed", "error")}
    return {"outcomes": outcomes, "counts": counts, "total": len(outcomes)}


def evaluate_sweep(outcomes: dict[str, str]) -> dict[str, Any]:
    """Turn per-nodeid outcomes into the pass/fail findings the record asserts."""
    counts = {s: sum(1 for v in outcomes.values() if v == s)
              for s in ("passed", "skipped", "failed", "error")}
    failed = sorted(n for n, s in outcomes.items() if s in ("failed", "error"))

    renderer = {n: outcomes.get(n, "ABSENT") for n in RENDERER_PARITY_NODEIDS}
    renderer_ok = all(s == "skipped" for s in renderer.values())

    leaf = {n: outcomes.get(n, "ABSENT") for n in LEAF_TRANSPORT_NODEIDS}
    leaf_ok = all(s == "passed" for s in leaf.values())

    repo_state = {n: outcomes.get(n, "ABSENT") for n in REPOSITORY_STATE_NODEIDS}
    repo_state_ok = all(s == "passed" for s in repo_state.values())

    # Any OTHER skip in the two modules the environment repair touched is an
    # unexpected environment skip: a test that quietly stopped running under an
    # empty HOME is indistinguishable from one that never existed.
    watched = ("tests/data/test_c1_battery.py",
               "tests/autoinit/test_leaf_transport_publish.py")
    unexpected = sorted(
        n for n, s in outcomes.items()
        if s == "skipped" and n.startswith(watched)
        and n not in RENDERER_PARITY_NODEIDS)

    problems: list[str] = []
    if failed:
        problems.append(f"{len(failed)} failed/errored: {failed[:10]}")
    if not renderer_ok:
        problems.append(f"renderer-parity skip set is not the expected 7: {renderer}")
    if not leaf_ok:
        problems.append(f"leaf transport did not pass 5/5: {leaf}")
    if not repo_state_ok:
        problems.append(f"repository-state tests did not pass: {repo_state}")
    if unexpected:
        problems.append(f"unexpected environment skips: {unexpected}")

    return {
        "counts": counts,
        "failed_nodeids": failed,
        "renderer_parity_expected_skips": renderer,
        "renderer_parity_skipped_as_expected": renderer_ok,
        "leaf_transport": leaf,
        "leaf_transport_all_passed": leaf_ok,
        "repository_state": repo_state,
        "repository_state_all_passed": repo_state_ok,
        "unexpected_environment_skips": unexpected,
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
    }


# --- the record and the gate that reads it ----------------------------------

def self_hash(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "self_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def head_commit(repo_root: str | Path = ".") -> str:
    out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip() or "unknown"


def tree_is_clean(repo_root: str | Path = ".") -> bool:
    out = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                         capture_output=True, text=True, timeout=60)
    return out.returncode == 0 and not out.stdout.strip()


def verify_record(record: dict[str, Any], repo_root: str | Path = ".",
                  ) -> tuple[bool, str]:
    """The cheap pre-provider check: does this record still describe live code?

    Never re-runs the sweep. It answers one question — is the thing that was
    proved still the thing that would run — and refuses when it cannot tell.
    """
    from .c1_authorization import c1_harness_digest

    if record.get("schema") != SCHEMA:
        return False, f"unexpected schema {record.get('schema')!r}"
    stored = record.get("self_sha256")
    if not stored or stored != self_hash(record):
        return False, "the record's self-hash does not match its contents"
    if record.get("verdict") != "PASS":
        return False, (f"the recorded sweep verdict is {record.get('verdict')!r}: "
                       f"{record.get('problems')}")

    try:
        live_harness = c1_harness_digest(repo_root)["digest"]
        live_env = pod_test_environment_digest(repo_root)["digest"]
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot digest the live tree: {exc}"

    if record.get("c1_harness_digest") != live_harness:
        return False, (f"the record was made against C1 harness "
                       f"{str(record.get('c1_harness_digest'))[:12]}…, the live tree "
                       f"is {live_harness[:12]}… — the pod sweep is owed again")
    if record.get("pod_test_environment_digest") != live_env:
        return False, (f"the record was made against pod test environment "
                       f"{str(record.get('pod_test_environment_digest'))[:12]}…, the "
                       f"live tree is {live_env[:12]}… — the pod sweep is owed again")
    if not record.get("tree_clean"):
        return False, "the sweep was recorded against a dirty working tree"

    c = record.get("counts") or {}
    return True, (f"pod sweep {c.get('passed')} passed / {c.get('skipped')} skipped "
                  f"/ {c.get('failed', 0) + c.get('error', 0)} failed, 7 renderer "
                  f"skips, leaf transport 5/5, binds harness {live_harness[:12]}… "
                  f"and environment {live_env[:12]}…")


def load_record(repo_root: str | Path = ".") -> dict[str, Any]:
    return json.loads((Path(repo_root) / RECORD_PATH).read_text())
