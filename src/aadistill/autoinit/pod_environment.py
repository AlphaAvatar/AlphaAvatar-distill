"""Proof that the pod's CPU test gate can pass, bound to the tree that proved it.

C1 attempt 3R reached `VLLM_READY → TEACHER_READY → ROPE_OK` and then died at the
setup test gate: `14 failed, 2650 passed`, `$0.3482`, no scientific stage. Seven
were renderer-parity cases reading `$HOME` for a Hugging Face cache no pod has,
and two were repository state. **The other five remain UNEXPLAINED**: they were
attributed to leaf transport by a `$0` reproduction that ran with no `HF_TOKEN`,
which is a state no pod is in, and that attribution does not reproduce.

The seven passed on the dev box and could not pass on a pod, and three launches
went by without anyone finding out, because no C1 attempt had ever reached
`TESTS_OK` before.

The suite that answers "would this pass on a pod?" takes about thirteen minutes,
which is far too slow to run inside a pre-provider gate while a pod waits. So it
is run once, deliberately, and what it produced is recorded here — and the cheap
gate checks that the recording still describes the code that would actually run.

That binding is to the **executable**, never to `HEAD`. The normal order of work
commits the executable first, runs the sweep against it, then writes a
preregistration and updates documentation in later commits; a gate keyed on the
commit hash would go stale the moment the paperwork landed, and the obvious way
to make it green again would be to re-run a sweep that had not changed in any way
that mattered. Two digests decide instead:

* the C1 harness digest — everything the paid session executes;
* the pod **test environment** digest — everything that decides what the pod's
  test gate does and is outside that harness: the simulator that models it, the
  whole test tree, the publisher whose tests the gate runs, and the recorder
  that writes this file. The pod's own setup script is NOT here: it is executed
  on the pod, so the grant measures it, in the harness.

Neither digest covers `logs/` or `docs/`, and the pod suite reads repository
state — so a third check, `swept_base_commit` lineage, closes that gap.

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

#: Files that decide the pod test gate's outcome and are **outside**
#: `C1_HARNESS_SOURCE_FILES_V1`. A list, not a glob, so each entry is a decision
#: somebody made.
#:
#: Strictly disjoint from the harness set, and `test_c1_readiness_gates` asserts
#: it. `verify_record` checks BOTH digests, so a file named in both buys nothing:
#: editing it already invalidates the record through the harness. Redundant
#: binding only makes the two lists look like independent evidence when they are
#: not. `autoinit_preflight_setup.sh`, `battery_render.py` and
#: `renderer_parity_gate.py` were all listed here until 2026-09-04 and are now
#: covered by the harness itself.
POD_TEST_ENVIRONMENT_FILES_V1: tuple[str, ...] = (
    #: The simulator that creates the pod-like conditions. Not executed on a pod,
    #: so it has no place in the harness, but a change to it changes what the
    #: recorded sweep MEANT.
    "scripts/pod/simulate_pod_env.sh",
    #: A dev-box publishing tool the paid session never runs — and whose tests the
    #: pod's setup gate does. That asymmetry is exactly why it is measured here
    #: and not in the harness.
    "scripts/autoinit/publish_selected_leaves.py",
    #: The recorder decides what the record CLAIMS the sweep found. A parser that
    #: mislabelled a skip as a pass would certify a failing gate, and nothing else
    #: here would notice.
    "scripts/autoinit/record_pod_environment.py",
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

#: The two staging-contract self-tests that describe the DEV BOX's staged/hidden
#: split. Inside the simulation that split has already been applied, so there is
#: nothing left to prove hidden and they skip. Declared here so they are exact
#: rather than merely unnoticed.
DEVBOX_ONLY_NODEIDS: tuple[str, ...] = (
    "tests/autoinit/test_staging_contract.py::"
    "test_an_artifact_c1_does_not_stage_is_invisible",
    "tests/autoinit/test_staging_contract.py::"
    "test_an_undeclared_file_inside_a_staged_destination_stays_hidden",
)

#: Host-local Phase-A integration cases, scoped by `SESSION_KIND=c1`. Their
#: premise is the retained leaf store at an absolute dev-box path that is
#: deliberately never staged to a pod. They are named SEPARATELY from the other
#: expected skips rather than folded into that count, because they skip for a
#: different reason: not "this source is absent here" but "this session does not
#: own that store". On a real pod they also skip, via their own store check.
HOST_LOCAL_C1_NODEIDS: tuple[str, ...] = (
    "tests/pod/test_recovery_continuation_session.py::"
    "test_the_real_stage1_entrypoint_imports_measures_admits_and_hands_off",
    "tests/pod/test_recovery_continuation_session.py::"
    "test_the_entrypoint_refuses_a_substituted_leaf",
)

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

def _nodeid(case: ET.Element, repo_root: Path) -> str:
    """Reconstruct a pytest nodeid from a JUnit `testcase` element.

    pytest writes **no `file` attribute** — only `classname` and `name`. Assuming
    otherwise produced nodeids like `::test_x` for every case, so the record's
    lookups all read `ABSENT` and it could not tell a passing sweep from a
    failing one. Found by running the sweep, which is the only reason it was
    found at all.

    `classname` is the dotted module path, with any test class appended:
    `tests.pod.test_x` or `tests.pod.test_x.TestGroup`. The split point is
    resolved against the filesystem — the longest dotted prefix that is a real
    `.py` file is the module — rather than guessed from naming convention.
    """
    cls = case.get("classname") or ""
    name = case.get("name") or ""
    parts = cls.split(".") if cls else []
    for cut in range(len(parts), 0, -1):
        rel = "/".join(parts[:cut]) + ".py"
        if (repo_root / rel).is_file():
            inner = "::".join(parts[cut:])
            return f"{rel}::{inner}::{name}" if inner else f"{rel}::{name}"
    # No module resolved: keep the raw classname rather than silently emitting a
    # nodeid with an empty path, which is what hid the defect the first time.
    return f"{cls}::{name}" if cls else name


def read_junit(path: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    """Per-nodeid outcomes from a JUnit XML report.

    JUnit is a *reporting* flag: it changes nothing about which tests are
    selected or how they run, so the sweep still executes the pod's own command.
    It is the only way to name every skip and every pass exactly, and naming them
    is the point — attempt 3R's four-line tail is why fourteen failures arrived
    as three.
    """
    root = Path(repo_root)
    tree = ET.parse(str(path))
    outcomes: dict[str, str] = {}
    n_cases = 0
    for case in tree.iter("testcase"):
        n_cases += 1
        nid = _nodeid(case, root)
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
    # A collision means two testcases mapped to one nodeid and one outcome was
    # silently dropped — which is exactly how a failure would disappear.
    if n_cases != len(outcomes):
        raise ValueError(
            f"{n_cases} testcases collapsed to {len(outcomes)} nodeids; the "
            "reconstruction is lossy and an outcome would be lost")
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

    hostlocal = {n: outcomes.get(n, "ABSENT") for n in HOST_LOCAL_C1_NODEIDS}
    hostlocal_ok = all(v == "skipped" for v in hostlocal.values())
    devbox = {n: outcomes.get(n, "ABSENT") for n in DEVBOX_ONLY_NODEIDS}
    devbox_ok = all(v == "skipped" for v in devbox.values())
    battery = {n: outcomes.get(n, "ABSENT") for n in BATTERY_SOURCE_NODEIDS}
    battery_ok = all(s == "skipped" for s in battery.values())
    staged_role = outcomes.get(BATTERY_STAGED_ROLE_NODEID, "ABSENT")
    staged_role_ok = staged_role == "passed"

    # Any OTHER skip in the two modules the environment repair touched is an
    # unexpected environment skip: a test that quietly stopped running under an
    # empty HOME is indistinguishable from one that never existed.
    watched = ("tests/data/test_c1_battery.py",
               "tests/autoinit/test_leaf_transport_publish.py",
               "tests/autoinit/test_staging_contract.py",
               "tests/pod/test_recovery_continuation_session.py")
    expected_skips = (set(RENDERER_PARITY_NODEIDS) | set(BATTERY_SOURCE_NODEIDS)
                      | set(DEVBOX_ONLY_NODEIDS) | set(HOST_LOCAL_C1_NODEIDS))
    unexpected = sorted(
        n for n, s in outcomes.items()
        if s == "skipped" and n.startswith(watched) and n not in expected_skips)

    problems: list[str] = []
    if failed:
        problems.append(f"{len(failed)} failed/errored: {failed[:10]}")
    if not renderer_ok:
        problems.append(f"renderer-parity skip set is not the expected 7: {renderer}")
    if not leaf_ok:
        problems.append(f"leaf transport did not pass 5/5: {leaf}")
    if not repo_state_ok:
        problems.append(f"repository-state tests did not pass: {repo_state}")
    if not hostlocal_ok:
        problems.append(f"host-local Phase-A cases did not skip under "
                        f"SESSION_KIND=c1: {hostlocal}")
    if not devbox_ok:
        problems.append(f"dev-box-only staging self-tests did not skip inside the "
                        f"simulation: {devbox}")
    if not battery_ok:
        problems.append(f"battery construction-source skip set is not the "
                        f"expected {len(BATTERY_SOURCE_NODEIDS)}: {battery}")
    if not staged_role_ok:
        problems.append(
            f"the battery role C1 DOES stage did not pass: {staged_role!r}. "
            "recovery_search_v2 is a local asset, so its disjointness parameter "
            "must run on a pod, not skip.")
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
        "host_local_c1_expected_skips": hostlocal,
        "host_local_c1_skipped_as_expected": hostlocal_ok,
        "devbox_only_expected_skips": devbox,
        "devbox_only_skipped_as_expected": devbox_ok,
        "battery_source_expected_skips": battery,
        "battery_source_skipped_as_expected": battery_ok,
        "battery_staged_role_nodeid": BATTERY_STAGED_ROLE_NODEID,
        "battery_staged_role_outcome": staged_role,
        "expected_environment_skips": sorted(expected_skips),
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


#: The only tracked path that may differ between the swept commit and the
#: session commit before an authorization exists. The record is written after the
#: sweep by construction, so its own commit can never be part of the swept tree.
PERMITTED_POST_SWEEP_PATHS: tuple[str, ...] = (RECORD_PATH,)


def lineage_from_swept_base(repo_root: Path, base: str | None, commit: str,
                            allowed_paths: tuple[str, ...]) -> dict[str, Any]:
    """`lineage_from_authorized_base`, asked about more than one permitted path.

    It **delegates**: the git plumbing — does the commit exist, does `commit`
    descend from `base`, what differs between them — runs once, in the frozen
    function, and only the permitted-set predicate is re-evaluated here on the
    `changed_paths` it already returned. There is no second copy of the risky
    part, which is what the session gates learned the hard way when two
    hand-written copies of a lineage rule drifted.

    It lives HERE rather than in `session_prechecks` on purpose.
    `session_prechecks.py` is a member of Phase B's and continuation B's frozen
    executable sets, and generalizing it in place moved both of those digests —
    for a feature neither closed phase will ever use. This module is in the C1
    harness alone, so the cost lands where the benefit does.
    """
    from ..infrastructure.session_prechecks import lineage_from_authorized_base

    allowed = tuple(allowed_paths)
    out = lineage_from_authorized_base(repo_root, base, commit,
                                       allowed[0] if allowed else "")
    if out["changed_paths"] is None:
        return out                      # refused before the diff; nothing to widen
    unexpected = [p for p in out["changed_paths"] if p not in allowed]
    out["allowed_paths"] = list(allowed)
    out["unexpected_paths"] = unexpected
    out["ok"] = not unexpected
    if unexpected:
        out["reason"] = (
            f"{len(unexpected)} path(s) other than {list(allowed)} changed "
            f"between the swept base and the session commit: {unexpected[:8]}")
    else:
        out["reason"] = (f"only {sorted(set(out['changed_paths']))} differs from "
                         "the swept base" if out["changed_paths"]
                         else "identical to the swept base")
    return out


#: What a readiness record may claim to be.
#:
#: `diagnostic` proves the machinery and that the suite passes on a tree. It is
#: real readiness evidence and stays valid as such. `launch_bound` is the sweep a
#: maintainer-approved paid launch rests on — the same procedure, run deliberately
#: as the thing the money will rest on rather than as a check that the plumbing
#: works.
#:
#: The distinction was defined on 2026-09-04 and enforced by nothing: the paid
#: gate accepted any PASS record and merely copied the kind into evidence, so a
#: diagnostic sweep could have satisfied gate 12 and the promised launch-bound
#: sweep need never have happened.
RECORD_KINDS: tuple[str, ...] = ("diagnostic", "launch_bound")

#: What `pod_environment_gate` requires before a provider resource is created.
LAUNCH_BOUND: str = "launch_bound"


def verify_record(record: dict[str, Any], repo_root: str | Path = ".", *,
                  session_commit: str | None = None,
                  authorization_path: str | None = None,
                  required_kind: str | None = None,
                  staging_contract_digest: str | None = None,
                  ) -> tuple[bool, str]:
    """The cheap pre-provider check: does this record still describe live code?

    Never re-runs the sweep. It answers one question — is the thing that was
    proved still the thing that would run — and refuses when it cannot tell.

    Two digests are not enough on their own. They cover the harness and the pod
    test environment, and deliberately ignore `logs/` and `docs/` — but the pod's
    pytest suite READS repository state: `current_state.json`, `STATE.md`,
    `CATALOG.md`. The 2026-09-04 sweep was recorded at `0457bab` and four
    documentation commits landed afterwards without invalidating anything, so the
    record described a suite that had not been run against the tree it was
    certifying.

    Enumerating every pytest data dependency is a losing game — the next one
    added would silently not be in the list. So this asks git instead, reusing
    the session-lineage rule verbatim: the session commit must descend from
    `swept_base_commit`, and the ONLY tracked paths permitted to differ are the
    readiness record itself and, once a session is issued, the canonical
    authorization artifact the existing lineage contract already allows. Any
    other change — `logs/**`, docs, README, preregistration, state, tests,
    source — means the sweep is owed again.
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

    # AFTER the self-hash check, deliberately: editing `record_kind` in place to
    # promote a diagnostic record is tampering, and must be reported as tampering
    # rather than as a kind mismatch.
    kind = record.get("record_kind")
    if kind not in RECORD_KINDS:
        return False, (f"the record declares record_kind {kind!r}, which is not "
                       f"one of {list(RECORD_KINDS)}; a record that cannot say "
                       "what it is cannot be relied on for anything")
    # The staged view the sweep ran under must be the one this session stages.
    # Attempt 4's sweep used simulate_pod_env.sh's GENERIC default HIDDEN_PATHS,
    # so it modelled a machine 55 tests more generous than the pod and certified
    # a tree that then failed six ways. A launch-bound record must carry a
    # contract derived from the session's own SetupManifest, and it must still
    # describe the live one.
    recorded_staging = record.get("staging_contract_digest")
    if required_kind == LAUNCH_BOUND and not recorded_staging:
        return False, (
            "the record carries no staging_contract_digest, so the sweep it "
            "describes may have used the generic simulator default rather than "
            "this session's staging manifest. That is what aborted attempt 4.")
    if (staging_contract_digest is not None and recorded_staging
            and staging_contract_digest != recorded_staging):
        return False, (
            f"the record was swept under staging contract {recorded_staging[:12]}…, "
            f"the live session stages {staging_contract_digest[:12]}… — relay "
            "inputs, local assets, test ignores or the session kind moved, so the "
            "sweep is owed again")

    if required_kind is not None and kind != required_kind:
        return False, (
            f"this record is {kind!r} and {required_kind!r} is required. A "
            "diagnostic sweep proves the machinery works on a tree; it is not the "
            "sweep a paid launch rests on. Re-run "
            "`record_pod_environment.py --kind launch_bound` on the final "
            "authorized tree.")

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

    # --- post-sweep lineage -------------------------------------------------
    root = Path(repo_root)
    base = record.get("swept_base_commit")
    if not base:
        return False, ("the record names no swept_base_commit, so nothing "
                       "constrains what changed after the sweep")
    target = session_commit or head_commit(root)
    allowed = list(PERMITTED_POST_SWEEP_PATHS)
    if authorization_path:
        allowed.append(authorization_path)
    lineage = lineage_from_swept_base(root, base, target, tuple(allowed))
    if not lineage["ok"]:
        return False, (f"post-sweep drift: {lineage['reason']}. The pod suite "
                       "reads repository state, so anything beyond the permitted "
                       "paths means the recorded sweep no longer describes the "
                       "tree that would run")

    c = record.get("counts") or {}
    return True, (f"pod sweep {c.get('passed')} passed / {c.get('skipped')} skipped "
                  f"/ {c.get('failed', 0) + c.get('error', 0)} failed, 7 renderer "
                  f"skips, leaf transport 5/5, binds harness {live_harness[:12]}… "
                  f"and environment {live_env[:12]}…, staging contract "
                  f"{str(recorded_staging)[:12]}…, swept at {base[:8]} with "
                  f"{lineage['reason']}")


def load_record(repo_root: str | Path = ".") -> dict[str, Any]:
    return json.loads((Path(repo_root) / RECORD_PATH).read_text())
