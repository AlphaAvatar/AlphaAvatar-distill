"""The two pre-provider gates added after C1 attempt 3R aborted at $0.3482.

Attempt 3R cleared `VLLM_READY → TEACHER_READY → ROPE_OK` and then died on the
pod's CPU test suite: `14 failed, 2650 passed`, no scientific stage. Seven were
renderer-parity cases reading a `$HOME` Hugging Face cache no pod has, and two
were repository state. **The other five remain UNEXPLAINED** — the leaf-transport
attribution was withdrawn when it failed to reproduce under the pod's real
`HF_TOKEN` contract. Ten pre-provider gates had passed.

Gate 11 (`renderer_parity_gate`) keeps the byte-for-byte rendering guarantee that
the now-skippable pytest cases used to carry alone. Gate 12
(`pod_environment_gate`) refuses unless one complete pod-like sweep has been
recorded against this exact executable.

A gate that has only ever been seen to pass is not evidence. Every case here
drives a refusal.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit import pod_environment as pe  # noqa: E402
from renderer_parity_gate import (EXPECTED_GROUPS, gate_verdict,  # noqa: E402
                                  run_parity)


# --- gate 11: renderer parity -----------------------------------------------

def _group(name: str, status: str, **over) -> dict:
    base = {"group": name, "status": status, "repo_id": f"org/{name}",
            "revision": "0" * 40, "file": "x.parquet",
            "resolved_snapshot": f"/nowhere/{name}", "n_frozen": 30,
            "n_checked": 30 if status == "PASS" else 0,
            "mismatches": [], "missing": []}
    base.update(over)
    return base


def _all_pass() -> dict:
    return {"groups": [_group(g, "PASS") for g in EXPECTED_GROUPS]}


def test_seven_passing_groups_are_accepted():
    ok, why = gate_verdict(_all_pass())
    assert ok, why
    assert "7/7" in why and "0 skipped" in why


def test_a_missing_snapshot_is_refused():
    record = _all_pass()
    record["groups"][3] = _group(EXPECTED_GROUPS[3], "SOURCE_ABSENT")
    ok, why = gate_verdict(record)
    assert not ok
    assert "not proved" in why and EXPECTED_GROUPS[3] in why.replace("org/", "")


def test_six_of_seven_is_refused_even_though_none_failed():
    """The dangerous shape: nothing failed, so a count-the-failures gate would
    pass it. Six proofs are not the guarantee."""
    record = {"groups": [_group(g, "PASS") for g in EXPECTED_GROUPS[:6]]
              + [_group(EXPECTED_GROUPS[6], "SOURCE_ABSENT")]}
    ok, why = gate_verdict(record)
    assert not ok and "not proved" in why


def test_a_group_that_was_never_run_is_refused():
    record = {"groups": [_group(g, "PASS") for g in EXPECTED_GROUPS[:6]]}
    ok, why = gate_verdict(record)
    assert not ok and "expected the 7 frozen groups" in why


def test_a_broken_rendering_is_refused():
    record = _all_pass()
    record["groups"][0] = _group(EXPECTED_GROUPS[0], "FAIL",
                                 mismatches=[{"id": "x", "field": "prompt_text"}])
    ok, why = gate_verdict(record)
    assert not ok and "BROKEN" in why


def test_the_gate_runs_the_seven_real_groups_on_this_host():
    """The dev box holds all seven pinned snapshots; this is the host the gate
    exists to run on. Skips if they are absent, since that is a statement about
    the machine and not about the code."""
    record = run_parity()
    if record["counts"]["SOURCE_ABSENT"]:
        pytest.skip("pinned source snapshots are not on this host")
    ok, why = gate_verdict(record)
    assert ok, why
    assert record["counts"] == {"PASS": 7, "FAIL": 0, "SOURCE_ABSENT": 0}


# --- gate 12: the pod environment record ------------------------------------

def _valid_record(tmp_path: Path, kind: str = "diagnostic") -> dict:
    """A record that binds the LIVE tree, so only the field under test differs."""
    from aadistill.autoinit.c1_authorization import c1_harness_digest

    rec = {
        "schema": pe.SCHEMA,
        # HEAD itself: a record whose swept base IS the tip has, by definition,
        # nothing after it, which isolates the digest checks from the lineage one.
        "swept_base_commit": pe.head_commit(REPO),
        "tree_clean": True,
        "c1_harness_digest": c1_harness_digest(REPO)["digest"],
        "pod_test_environment_digest": pe.pod_test_environment_digest(REPO)["digest"],
        "counts": {"passed": 2700, "skipped": 48, "failed": 0, "error": 0},
        "verdict": "PASS",
        "record_kind": kind,
        # Required for launch_bound since 2026-09-05: a record that cannot say
        # which staged view it was swept under may have used the generic default.
        "staging_contract_digest": "a" * 64,
        "problems": [],
    }
    rec["self_sha256"] = pe.self_hash(rec)
    return rec


def test_a_record_binding_the_live_tree_is_accepted(tmp_path):
    ok, why = pe.verify_record(_valid_record(tmp_path), REPO)
    assert ok, why


def test_a_record_made_against_a_different_harness_is_refused(tmp_path):
    rec = _valid_record(tmp_path)
    rec["c1_harness_digest"] = "f" * 64
    rec["self_sha256"] = pe.self_hash(rec)
    ok, why = pe.verify_record(rec, REPO)
    assert not ok and "harness" in why and "owed again" in why


def test_a_record_made_against_a_different_test_environment_is_refused(tmp_path):
    """The case a harness-only binding would miss: the pod's setup script, the
    simulator or any test file changed after the sweep."""
    rec = _valid_record(tmp_path)
    rec["pod_test_environment_digest"] = "e" * 64
    rec["self_sha256"] = pe.self_hash(rec)
    ok, why = pe.verify_record(rec, REPO)
    assert not ok and "pod test environment" in why and "owed again" in why


def test_a_tampered_record_is_refused(tmp_path):
    rec = _valid_record(tmp_path)
    rec["counts"]["failed"] = 0
    rec["verdict"] = "PASS"
    rec["problems"] = []
    rec["self_sha256"] = "0" * 64
    ok, why = pe.verify_record(rec, REPO)
    assert not ok and "self-hash" in why


def test_a_failed_sweep_is_refused(tmp_path):
    rec = _valid_record(tmp_path)
    rec["verdict"] = "FAIL"
    rec["problems"] = ["3 failed/errored"]
    rec["self_sha256"] = pe.self_hash(rec)
    ok, why = pe.verify_record(rec, REPO)
    assert not ok and "FAIL" in why


def test_a_sweep_recorded_against_a_dirty_tree_is_refused(tmp_path):
    rec = _valid_record(tmp_path)
    rec["tree_clean"] = False
    rec["self_sha256"] = pe.self_hash(rec)
    ok, why = pe.verify_record(rec, REPO)
    assert not ok and "dirty" in why


def test_the_two_measured_sets_are_disjoint():
    """No file may be bound by both digests.

    `verify_record` checks the harness digest AND the pod test-environment
    digest, so a file named in both is already caught by the first — the second
    binding adds nothing and makes two lists look like independent evidence when
    they are not. `autoinit_preflight_setup.sh`, `battery_render.py` and
    `renderer_parity_gate.py` were double-bound until 2026-09-04 and now live in
    the harness alone, where the paid session's own grant measures them.
    """
    from aadistill.autoinit.c1_authorization import C1_HARNESS_SOURCE_FILES_V1

    overlap = sorted(set(pe.POD_TEST_ENVIRONMENT_FILES_V1)
                     & set(C1_HARNESS_SOURCE_FILES_V1))
    assert not overlap, (
        f"double-bound in both measured sets: {overlap}. Pick the one that "
        "describes why the file matters and delete the other.")


def test_the_pod_setup_script_is_measured_by_the_harness_not_by_this_record():
    """It moved sets on 2026-09-04, and the direction matters.

    The pod EXECUTES it, so the grant must measure it. Binding it only in the
    readiness record left a paid session whose own setup script could change
    without moving the digest the authorization checks.
    """
    from aadistill.autoinit.c1_authorization import C1_HARNESS_SOURCE_FILES_V1

    assert "scripts/pod/autoinit_preflight_setup.sh" in C1_HARNESS_SOURCE_FILES_V1
    assert "scripts/pod/autoinit_preflight_setup.sh" \
        not in pe.POD_TEST_ENVIRONMENT_FILES_V1


def test_the_environment_digest_covers_the_whole_test_tree():
    """A new test is exactly as able to fail on a pod as new production code."""
    d = pe.pod_test_environment_digest(REPO)
    n_tests = len(list((REPO / "tests").rglob("*.py")))
    assert d["n_files"] >= n_tests
    for named in pe.POD_TEST_ENVIRONMENT_FILES_V1:
        assert (REPO / named).is_file(), named


def test_the_environment_digest_moves_when_a_measured_file_moves(tmp_path):
    """Copy the tree, touch one measured file, require a different digest."""
    import shutil

    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    for named in pe.POD_TEST_ENVIRONMENT_FILES_V1:
        dest = root / named
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / named, dest)
    (root / "tests/test_x.py").write_text("def test_a(): pass\n")
    before = pe.pod_test_environment_digest(root)["digest"]
    (root / "tests/test_x.py").write_text("def test_a(): assert True\n")
    assert pe.pod_test_environment_digest(root)["digest"] != before


def test_a_missing_measured_file_refuses_rather_than_digesting_less(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        pe.pod_test_environment_digest(root)


# --- what the record must assert about the sweep ----------------------------

def test_the_expected_skip_set_is_exactly_the_seven_renderer_cases():
    assert len(pe.RENDERER_PARITY_NODEIDS) == 7
    groups = {n.split("[")[1].rstrip("]") for n in pe.RENDERER_PARITY_NODEIDS}
    assert groups == set(EXPECTED_GROUPS)
    assert all(n.startswith("tests/data/test_c1_battery.py::")
               for n in pe.RENDERER_PARITY_NODEIDS)


def test_the_five_leaf_transport_nodeids_are_the_historical_ones():
    """The five names are pinned so the regression set cannot quietly shrink.

    They are NOT "the five that failed on the pod": that attribution was
    withdrawn when it failed to reproduce under the pod's real HF_TOKEN
    contract. The five actual attempt-3R identities are still unknown.
    """
    assert len(pe.LEAF_TRANSPORT_NODEIDS) == 5
    historical = {
        "test_a_corrupted_remote_file_is_caught_by_the_round_trip",
        "test_a_size_mismatch_at_the_far_end_is_caught",
        "test_an_lfs_oid_that_disagrees_is_caught_without_downloading",
        "test_a_file_absent_from_the_far_end_is_caught",
        "test_the_round_trip_needs_no_dev_box_directory"}
    assert {n.split("::")[1] for n in pe.LEAF_TRANSPORT_NODEIDS} == historical


def _outcomes(**over) -> dict[str, str]:
    o = {n: "skipped" for n in pe.RENDERER_PARITY_NODEIDS}
    # The C1 construction-source cases skip because their historical roles are
    # not staged; the one role C1 DOES stage must pass.
    o.update({n: "skipped" for n in pe.BATTERY_SOURCE_NODEIDS})
    o[pe.BATTERY_STAGED_ROLE_NODEID] = "passed"
    o.update({n: "skipped" for n in pe.DEVBOX_ONLY_NODEIDS})
    o.update({n: "skipped" for n in pe.HOST_LOCAL_C1_NODEIDS})
    o.update({n: "passed" for n in pe.LEAF_TRANSPORT_NODEIDS})
    o.update({n: "passed" for n in pe.REPOSITORY_STATE_NODEIDS})
    o["tests/other/test_thing.py::test_ok"] = "passed"
    o.update(over)
    return o


def test_the_expected_sweep_shape_passes():
    findings = pe.evaluate_sweep(_outcomes())
    assert findings["verdict"] == "PASS", findings["problems"]
    assert findings["leaf_transport_all_passed"]
    assert findings["renderer_parity_skipped_as_expected"]


def test_a_leaf_transport_skip_is_a_problem_not_a_pass():
    """The exact regression: a skipped transport check on a pod is
    indistinguishable from one that never existed."""
    findings = pe.evaluate_sweep(
        _outcomes(**{pe.LEAF_TRANSPORT_NODEIDS[0]: "skipped"}))
    assert findings["verdict"] == "FAIL"
    assert any("leaf transport" in p for p in findings["problems"])


def test_an_unexpected_environment_skip_is_detected():
    """Not global zero-skips — the suite has legitimate unrelated skips — but a
    NEW skip in the two modules the repair touched."""
    findings = pe.evaluate_sweep(_outcomes(**{
        "tests/data/test_c1_battery.py::test_the_counts_are_exactly_the_frozen_mixture":
            "skipped"}))
    assert findings["verdict"] == "FAIL"
    assert any("unexpected environment skips" in p for p in findings["problems"])


def test_an_unrelated_skip_elsewhere_is_not_a_problem():
    findings = pe.evaluate_sweep(
        _outcomes(**{"tests/models/test_big.py::test_needs_gpu": "skipped"}))
    assert findings["verdict"] == "PASS", findings["problems"]


def test_a_renderer_case_that_passed_is_also_wrong():
    """If it PASSED under the simulator, the dev box's dataset cache leaked into
    the isolated environment and the sweep did not simulate a pod at all."""
    findings = pe.evaluate_sweep(
        _outcomes(**{pe.RENDERER_PARITY_NODEIDS[0]: "passed"}))
    assert findings["verdict"] == "FAIL"
    assert any("renderer-parity skip set" in p for p in findings["problems"])


def test_any_failure_is_a_problem():
    findings = pe.evaluate_sweep(
        _outcomes(**{"tests/other/test_thing.py::test_ok": "failed"}))
    assert findings["verdict"] == "FAIL"
    assert findings["failed_nodeids"] == ["tests/other/test_thing.py::test_ok"]


def test_junit_nodeids_round_trip_without_a_file_attribute(tmp_path):
    """pytest writes NO `file` attribute — only `classname` and `name`.

    Assuming otherwise produced `::test_x` for all 2820 cases, so every lookup in
    the record read `ABSENT` and it could not distinguish a green sweep from a
    red one. It reported `renderer-parity skip set is not the expected 7` on a
    sweep where all seven had skipped correctly. Only running the sweep found it.
    """
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.data.test_c1_battery" name="test_a[tool]"/>'
        '<testcase classname="tests.pod.test_simulator_restore" name="test_b">'
        '<skipped/></testcase>'
        '<testcase classname="tests.data.test_c1_battery" name="test_c">'
        '<failure/></testcase>'
        '</testsuite></testsuites>')
    got = pe.read_junit(xml, REPO)
    assert got["outcomes"] == {
        "tests/data/test_c1_battery.py::test_a[tool]": "passed",
        "tests/pod/test_simulator_restore.py::test_b": "skipped",
        "tests/data/test_c1_battery.py::test_c": "failed"}
    assert got["counts"] == {"passed": 1, "skipped": 1, "failed": 1, "error": 0}


def test_a_test_class_is_resolved_against_the_real_module(tmp_path):
    """`classname` appends the class, so the module boundary is resolved from the
    filesystem rather than guessed from naming convention."""
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.data.test_c1_battery.TestGroup" name="test_b"/>'
        '</testsuite></testsuites>')
    got = pe.read_junit(xml, REPO)
    assert list(got["outcomes"]) == [
        "tests/data/test_c1_battery.py::TestGroup::test_b"]


def test_two_cases_collapsing_to_one_nodeid_is_refused(tmp_path):
    """A lossy reconstruction would silently drop an outcome — and the one it
    dropped could be the failure."""
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.data.test_c1_battery" name="test_same"/>'
        '<testcase classname="tests.data.test_c1_battery" name="test_same">'
        '<failure/></testcase>'
        '</testsuite></testsuites>')
    with pytest.raises(ValueError, match="lossy"):
        pe.read_junit(xml, REPO)


def test_the_parsed_count_matches_the_suite_total(tmp_path):
    """The arithmetic that caught the lossy case: 2820 testcases must produce
    2820 nodeids."""
    xml = tmp_path / "j.xml"
    cases = "".join(
        f'<testcase classname="tests.data.test_c1_battery" name="test_{i}"/>'
        for i in range(50))
    xml.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    got = pe.read_junit(xml, REPO)
    assert got["total"] == 50 and got["counts"]["passed"] == 50


# --- the pod gate's own diagnostics -----------------------------------------

def test_the_pod_gate_names_every_failing_nodeid_before_it_exits():
    """Attempt 3R reported `14 failed, 2650 passed` and brought home THREE names.

    The gate tails four lines and `/workspace/pytest.log` dies with the pod, so
    the other eleven had to be reconstructed at $0 afterwards — and that
    reconstruction attributed five of them to a cause that does not survive
    re-testing. One grep is free.
    """
    text = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    block = text[text.index("CPU test suite"):]
    grep_at = block.index("grep -E '^(FAILED|ERROR) ' /workspace/pytest.log")
    tail_at = block.index("tail -4 /workspace/pytest.log")
    exit_at = block.index('[ "$RC" -eq 0 ] || {')
    assert grep_at < tail_at, "the failure list must precede the four-line tail"
    assert grep_at < exit_at, "the gate exits before it names the failures"
    # `|| true` so a log with no summary line cannot replace a test failure with a
    # different, less informative one.
    assert "|| true" in block[grep_at:grep_at + 120]


# --- the record on disk, once it exists -------------------------------------

C1_AUTH_PATH = "logs/autoinit_c1_authorization.json"


def test_the_committed_record_still_binds_the_live_executable():
    """Gate 12 itself, run as a test — with the gate's OWN argument contract.

    Until 2026-09-05 this called `verify_record(record, REPO)` and omitted
    `authorization_path`, so only the readiness record counted as a permitted
    post-sweep path. That is not the shape a pod is ever in: a pod checks out the
    SESSION commit, which by construction also carries
    `logs/autoinit_c1_authorization.json`, so the authorization read as post-sweep
    drift and the record was refused. It cost one of attempt 4's six failures.

    The production gate was already correct and passed — `pod_environment_gate`
    passes `authorization_path=AUTH_PATH`. Only the test disagreed with it, and no
    pre-issuance sweep could have caught that, because before issuance the
    authorization is not in the tree to be objected to. So the test now asks the
    question the gate asks, and `test_the_test_matches_the_paid_gate` below keeps
    the two from drifting apart again.
    """
    path = REPO / pe.RECORD_PATH
    if not path.is_file():
        pytest.skip(f"{pe.RECORD_PATH} has not been produced yet")
    ok, why = pe.verify_record(json.loads(path.read_text()), REPO,
                               authorization_path=C1_AUTH_PATH)
    assert ok, why


def test_the_test_matches_the_paid_gate_argument_for_argument():
    """The launcher's permitted-path set and this module's must be the same one."""
    src = (REPO / "scripts/pod/autoinit_c1_launch.py").read_text()
    assert 'AUTH_PATH = "logs/autoinit_c1_authorization.json"' in src
    assert C1_AUTH_PATH == "logs/autoinit_c1_authorization.json"
    assert "authorization_path=AUTH_PATH" in src, (
        "the paid gate no longer permits the authorization path; this test would "
        "then be stricter than the gate rather than equal to it")


def test_a_pre_authorization_tree_is_accepted_with_the_record_alone(tmp_path,
                                                                    monkeypatch):
    """Before issuance: the readiness record is the only permitted delta."""
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    head = _commit(root, pe.RECORD_PATH, '{"r": 1}\n', "record the sweep")
    ok, why = pe.verify_record(rec, root, session_commit=head,
                               authorization_path=C1_AUTH_PATH)
    assert ok, why


def test_an_issued_session_tree_is_accepted_with_record_and_authorization(
        tmp_path, monkeypatch):
    """After issuance: exactly the shape a pod checks out. This is the case
    attempt 4 refused."""
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    _commit(root, pe.RECORD_PATH, '{"r": 1}\n', "record the sweep")
    head = _commit(root, C1_AUTH_PATH, '{"auth": 1}\n', "issue")
    ok, why = pe.verify_record(rec, root, session_commit=head,
                               authorization_path=C1_AUTH_PATH)
    assert ok, why
    # And the omission that caused the abort still refuses, so the fix is the
    # argument and not a loosened rule.
    ok2, why2 = pe.verify_record(rec, root, session_commit=head)
    assert not ok2 and "post-sweep drift" in why2


def test_gate_twelve_is_excluded_from_the_candidate_sweep_for_a_stated_reason():
    """The exclusion must stay narrow and stay justified.

    `pod_environment_gate` consumes the pod sweep's own output, so it cannot also
    be a precondition of the suite that sweep runs. That is a circularity, not a
    convenience — but an exclusion list is exactly where a real gate goes to die
    quietly, so both the membership and the reason are pinned here.
    """
    src = (REPO / "tests/pod/test_c1_session_contract.py").read_text()
    assert '"pod_environment_gate"]' in src, (
        "the exclusion set changed shape; re-read why each member is in it")
    assert "genuine\n    circularity" in src, "the reason for excluding gate 12 is gone"
    assert "It is not left unexercised" in src, (
        "the pointer to gate 12's real coverage is gone")


def test_the_only_conditional_exclusion_is_the_synthetic_credential():
    """Two gates are excluded ONLY inside the simulation, and only there.

    `rope_input_gate` authenticates to the private relay (401 with a fake token);
    `renderer_parity_gate` reads the seven pinned snapshots, which the isolated
    cache deliberately lacks. Both must stay conditional: on a real dev box they
    run, and an unconditional exclusion would retire the check that closed attempt
    2 and the guarantee that replaced the skippable parity cases.
    """
    src = (REPO / "tests/pod/test_c1_session_contract.py").read_text()
    assert 'if os.environ.get("AAD_SYNTHETIC_HF_TOKEN"):' in src
    assert 'structurally_unavailable += ["rope_input_gate", "renderer_parity_gate"]' in src
    # Conditional, not a member of the base list.
    base = src[src.index("structurally_unavailable = ["):
               src.index('if os.environ.get("AAD_SYNTHETIC_HF_TOKEN")')]
    assert "rope_input_gate" not in base, (
        "rope_input_gate became an unconditional exclusion; attempt 2 died at "
        "ROPE_OK and this gate is what closed that gap")


def test_the_simulator_announces_that_its_credential_is_synthetic():
    sim = (REPO / "scripts/pod/simulate_pod_env.sh").read_text()
    assert "export AAD_SYNTHETIC_HF_TOKEN=1" in sim
    # It must survive the PODSIM_* unset, or the suite never sees it.
    unset_line = sim[sim.index("unset PODSIM_JUNIT"):]
    assert "AAD_SYNTHETIC_HF_TOKEN" not in unset_line.split("\n\n")[0], (
        "the flag is unset before the suite runs, so nothing can read it")


# --- gate 12: post-sweep repository lineage ---------------------------------
#
# The two digests cover the harness and the pod test environment and deliberately
# ignore logs/ and docs/. But the pod's pytest suite READS repository state --
# current_state.json, STATE.md, CATALOG.md -- so a record could certify a suite
# that had never been run against the tree it was certifying. That is exactly
# what happened: the 2026-09-04 sweep was recorded at 0457bab and four
# documentation commits landed after it without invalidating anything.
#
# Enumerating every pytest data dependency is a losing game; the next one added
# would silently not be in the list. So the rule is git lineage, reusing
# `lineage_from_authorized_base` verbatim. These drive it against a real
# temporary repository, with the digest checks stubbed so only the lineage
# behaviour is under test.

import subprocess


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _swept_repo(tmp_path, monkeypatch):
    """A repo at a clean 'swept' commit, plus a record that binds it."""
    root = tmp_path / "repo"
    (root / "logs").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "logs/STATE.md").write_text("state\n")
    (root / "logs/current_state.json").write_text('{"a": 1}\n')
    (root / "logs/CATALOG.md").write_text("catalog\n")
    (root / "tests/test_x.py").write_text("def test_a(): pass\n")
    _git(root.parent, "init", "-q", str(root)) if False else subprocess.run(
        ["git", "init", "-q", str(root)], check=True, capture_output=True)
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "swept base")
    base = _git(root, "rev-parse", "HEAD")

    # Only the lineage is under test; make both digest checks agree by fiat.
    monkeypatch.setattr(pe, "pod_test_environment_digest",
                        lambda r=".": {"digest": "e" * 64, "n_files": 1})
    from aadistill.autoinit import c1_authorization as ca
    monkeypatch.setattr(ca, "c1_harness_digest",
                        lambda r=".", files=None: {"digest": "h" * 64, "n_files": 1})

    rec = {"schema": pe.SCHEMA, "swept_base_commit": base, "tree_clean": True,
           "c1_harness_digest": "h" * 64, "pod_test_environment_digest": "e" * 64,
           "counts": {"passed": 10, "skipped": 1, "failed": 0, "error": 0},
           "verdict": "PASS", "record_kind": "diagnostic", "problems": []}
    rec["self_sha256"] = pe.self_hash(rec)
    return root, rec, base


def _commit(root, rel, text, msg):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)
    return _git(root, "rev-parse", "HEAD")


def test_a_tree_identical_to_the_swept_commit_is_accepted(tmp_path, monkeypatch):
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    ok, why = pe.verify_record(rec, root, session_commit=base)
    assert ok, why


def test_a_readiness_record_only_delta_is_accepted(tmp_path, monkeypatch):
    """The one change that is inherent: the record is written AFTER the sweep,
    so its own commit can never be inside the swept tree."""
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    head = _commit(root, pe.RECORD_PATH, '{"record": true}\n', "record the sweep")
    ok, why = pe.verify_record(rec, root, session_commit=head)
    assert ok, why


def test_the_canonical_authorization_is_accepted_only_in_the_issued_shape(
        tmp_path, monkeypatch):
    """An issued session commits its authorization after the sweep. That is the
    ONE extra path, and only when a session actually carries an authorization."""
    auth = "logs/autoinit_c1_authorization.json"
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    _commit(root, pe.RECORD_PATH, '{"record": true}\n', "record the sweep")
    head = _commit(root, auth, '{"authorization": true}\n', "issue")

    ok, why = pe.verify_record(rec, root, session_commit=head,
                               authorization_path=auth)
    assert ok, why

    # Without an issued session it is just another log file, and refused.
    ok2, why2 = pe.verify_record(rec, root, session_commit=head)
    assert not ok2 and "post-sweep drift" in why2 and auth in why2


def test_changing_current_state_after_the_sweep_is_refused(tmp_path, monkeypatch):
    """The exact 2026-09-04 hole: the pod suite reads this file."""
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    head = _commit(root, "logs/current_state.json", '{"a": 2}\n', "normalize state")
    ok, why = pe.verify_record(rec, root, session_commit=head)
    assert not ok
    assert "post-sweep drift" in why and "logs/current_state.json" in why


def test_changing_state_md_after_the_sweep_is_refused(tmp_path, monkeypatch):
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    head = _commit(root, "logs/STATE.md", "rewritten\n", "update the handoff")
    ok, why = pe.verify_record(rec, root, session_commit=head)
    assert not ok
    assert "post-sweep drift" in why and "logs/STATE.md" in why


def test_an_arbitrary_second_log_or_doc_file_is_refused(tmp_path, monkeypatch):
    """`logs/**` is not a permitted class. Exactly two paths are, by name."""
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    _commit(root, pe.RECORD_PATH, '{"record": true}\n', "record the sweep")
    head = _commit(root, "logs/some_other_note.json", "{}\n", "add a note")
    ok, why = pe.verify_record(rec, root, session_commit=head,
                               authorization_path="logs/autoinit_c1_authorization.json")
    assert not ok
    assert "post-sweep drift" in why and "logs/some_other_note.json" in why


def test_changing_a_test_or_source_file_is_refused_by_the_digests(tmp_path,
                                                                  monkeypatch):
    """Not by lineage — by the digest that already covered it.

    The lineage rule is an ADDITION. If a test file change ever reached the
    lineage check instead of being stopped by the pod test-environment digest,
    the digest would have stopped covering the test tree.
    """
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    head = _commit(root, "tests/test_x.py", "def test_a(): assert True\n", "edit")
    monkeypatch.setattr(pe, "pod_test_environment_digest",
                        lambda r=".": {"digest": "9" * 64, "n_files": 1})
    ok, why = pe.verify_record(rec, root, session_commit=head)
    assert not ok
    assert "pod test environment" in why and "owed again" in why


def test_a_record_without_a_swept_base_commit_is_refused(tmp_path, monkeypatch):
    """A record that names no base constrains nothing after the sweep."""
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    rec.pop("swept_base_commit")
    rec["self_sha256"] = pe.self_hash(rec)
    ok, why = pe.verify_record(rec, root, session_commit=base)
    assert not ok and "swept_base_commit" in why


def test_a_session_commit_off_the_swept_line_of_history_is_refused(tmp_path,
                                                                   monkeypatch):
    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    _git(root, "checkout", "-q", "-b", "other", f"{base}~0")
    _git(root, "checkout", "-q", "--orphan", "elsewhere")
    _commit(root, "logs/STATE.md", "different history\n", "orphan")
    head = _git(root, "rev-parse", "HEAD")
    ok, why = pe.verify_record(rec, root, session_commit=head)
    assert not ok and "post-sweep drift" in why


def test_the_permitted_set_is_exactly_the_record_before_issuance():
    assert pe.PERMITTED_POST_SWEEP_PATHS == (pe.RECORD_PATH,)


def test_the_session_lineage_gate_kept_its_single_permitted_path():
    """The generalization must not have loosened the caller it was extracted for."""
    src = (REPO / "src/aadistill/infrastructure/session_prechecks.py").read_text()
    assert "lineage_from_authorized_base(\n                repo_root, ctx.auth." \
           "authorized_session_commit, commit, auth_path)" in src, (
        "the session gate no longer passes exactly one permitted path")


def test_the_snapshot_does_not_duplicate_the_swept_commit():
    """One fact, one owner.

    `current_state.json` carried `baseline_commit: 4f85ecda…` meaning "the tree
    the sweep ran against" while the record said `0457bab…`. A copy of a hash the
    snapshot cannot know until after it is written could only ever go stale — and
    gate 12 now forbids editing state after the sweep, so it could not even be
    corrected in place. The record owns it.
    """
    snap = json.loads((REPO / "logs/current_state.json").read_text())
    baseline = snap["baseline_commit"]
    assert not re.fullmatch(r"[0-9a-f]{7,40}", baseline), (
        f"baseline_commit is a bare hash ({baseline}); it must point at the "
        "readiness record, which is the only thing that can know it in time")
    assert pe.RECORD_PATH in baseline and "swept_base_commit" in baseline


def test_the_recorded_swept_commit_is_a_real_commit_in_this_repository():
    path = REPO / pe.RECORD_PATH
    if not path.is_file():
        pytest.skip(f"{pe.RECORD_PATH} has not been produced yet")
    base = json.loads(path.read_text()).get("swept_base_commit")
    assert base, "the record names no swept_base_commit"
    got = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                          f"{base}^{{commit}}"], capture_output=True)
    assert got.returncode == 0, f"swept_base_commit {base} is not a commit here"


def test_the_swept_base_helper_delegates_rather_than_reimplementing():
    """One copy of the git plumbing, and a proof that it agrees.

    `lineage_from_swept_base` exists because gate 12 permits two paths and the
    session gate permits one. It must not become a second hand-written lineage
    rule: that is precisely how the two session gates drifted before
    `lineage_from_authorized_base` was extracted. It delegates, and on the
    single-path case it must agree with what it delegates to.
    """
    from aadistill.infrastructure import session_prechecks as sp

    src = (REPO / "src/aadistill/autoinit/pod_environment.py").read_text()
    assert "from ..infrastructure.session_prechecks import " \
           "lineage_from_authorized_base" in src, "it no longer delegates"
    for forbidden in ("merge-base", "--is-ancestor", "git diff"):
        assert forbidden not in src, (
            f"pod_environment re-implements git plumbing ({forbidden!r}); it must "
            "delegate to lineage_from_authorized_base")

    # And the session gate it delegates to is untouched: `session_prechecks.py`
    # is inside Phase B's and continuation B's FROZEN executable sets, so
    # generalizing it in place would move two closed phases' digests.
    import inspect
    sig = inspect.signature(sp.lineage_from_authorized_base)
    assert sig.parameters["auth_path"].annotation in ("str", str), (
        "session_prechecks was generalized in place; that moves frozen digests "
        "for phases that will never use the feature")


def test_the_delegate_agrees_with_the_frozen_rule_on_one_path(tmp_path,
                                                              monkeypatch):
    """Same inputs, same verdict — accepted and refused alike."""
    from aadistill.infrastructure.session_prechecks import (
        lineage_from_authorized_base)

    root, rec, base = _swept_repo(tmp_path, monkeypatch)
    head = _commit(root, pe.RECORD_PATH, '{"r": 1}\n', "record")
    for allowed in (pe.RECORD_PATH, "logs/STATE.md"):
        a = lineage_from_authorized_base(root, base, head, allowed)
        b = pe.lineage_from_swept_base(root, base, head, (allowed,))
        assert a["ok"] == b["ok"], allowed
        assert a["unexpected_paths"] == b["unexpected_paths"], allowed
        assert a["changed_paths"] == b["changed_paths"], allowed

    # And a pre-diff refusal passes straight through, unwidened.
    a = lineage_from_authorized_base(root, None, head, pe.RECORD_PATH)
    b = pe.lineage_from_swept_base(root, None, head, (pe.RECORD_PATH,))
    assert a["ok"] is False and b["ok"] is False
    assert b["changed_paths"] is None


def test_the_snapshot_does_not_duplicate_the_sweep_result():
    """`latest_verification` points; it does not copy.

    It carried `commit cd5cb7d` and `2768/61/0` while the readiness record said
    `4231b42` and `2787/62/0` — the same failure mode as `baseline_commit`, one
    field over. The record owns kind, swept base, counts, both digests and the
    verdict; a second copy in a file that is edited on a different cadence can
    only go stale.
    """
    lv = json.loads((REPO / "logs/current_state.json").read_text())[
        "latest_verification"]
    assert lv["owned_by"] == pe.RECORD_PATH
    blob = json.dumps(lv)
    assert not re.search(r"\b\d{3,4}\s*/\s*\d{1,3}\s*/\s*\d\b", blob.replace(
        "2768/61/0", "").replace("2787/62/0", "")), (
        f"latest_verification carries a pass/skip/fail triple: {blob}")
    for key in ("suite", "counts", "swept_base_commit", "record_kind",
                "c1_harness_digest", "pod_test_environment_digest", "verdict"):
        assert key not in lv, (
            f"latest_verification duplicates {key!r}, which the readiness record "
            "owns")


# --- gate 12: diagnostic is not launch-bound --------------------------------
#
# The two kinds were defined on 2026-09-04 and enforced by nothing. `verify_record`
# accepted any PASS record and the launcher merely copied `record_kind` into
# evidence, so the diagnostic sweep sitting in the repository could have satisfied
# gate 12 and the promised launch-bound sweep need never have happened. The
# distinction existed only in prose.

def test_a_diagnostic_record_is_valid_readiness_evidence(tmp_path):
    """It must keep verifying for anyone asking whether the tree is sound.

    The repair must not make diagnostic verification impossible — that would
    delete the cheap check the machinery was built around.
    """
    ok, why = pe.verify_record(_valid_record(tmp_path, "diagnostic"), REPO)
    assert ok, why


def test_a_diagnostic_record_is_refused_by_the_paid_launch_gate(tmp_path):
    """The defect this closes: sound tree, wrong warrant."""
    ok, why = pe.verify_record(_valid_record(tmp_path, "diagnostic"), REPO,
                               required_kind=pe.LAUNCH_BOUND)
    assert not ok
    assert "'diagnostic'" in why and "launch_bound" in why
    assert "--kind launch_bound" in why, "the refusal must say how to fix it"


def test_a_launch_bound_record_is_accepted_by_the_paid_launch_gate(tmp_path):
    ok, why = pe.verify_record(_valid_record(tmp_path, pe.LAUNCH_BOUND), REPO,
                               required_kind=pe.LAUNCH_BOUND)
    assert ok, why


def test_a_missing_record_kind_is_refused(tmp_path):
    rec = _valid_record(tmp_path)
    rec.pop("record_kind")
    rec["self_sha256"] = pe.self_hash(rec)
    for required in (None, pe.LAUNCH_BOUND):
        ok, why = pe.verify_record(rec, REPO, required_kind=required)
        assert not ok and "record_kind" in why and "cannot say what it is" in why


def test_an_unknown_record_kind_is_refused(tmp_path):
    """Not silently treated as diagnostic, and not as launch-bound either."""
    rec = _valid_record(tmp_path)
    rec["record_kind"] = "provisional"
    rec["self_sha256"] = pe.self_hash(rec)
    for required in (None, pe.LAUNCH_BOUND):
        ok, why = pe.verify_record(rec, REPO, required_kind=required)
        assert not ok and "'provisional'" in why


def test_promoting_the_kind_in_place_is_caught_as_tampering(tmp_path):
    """Editing `record_kind` without recomputing the self-hash must read as
    TAMPERING, not as a kind mismatch — the self-hash check comes first, so the
    message names the right offence."""
    rec = _valid_record(tmp_path, "diagnostic")
    rec["record_kind"] = pe.LAUNCH_BOUND            # no re-hash: the whole point
    ok, why = pe.verify_record(rec, REPO, required_kind=pe.LAUNCH_BOUND)
    assert not ok
    assert "self-hash" in why, why
    assert "launch_bound" not in why, (
        "a tampered record must be refused as tampered, not as a kind mismatch")


def test_the_recorder_can_only_write_the_two_known_kinds():
    """`--kind` choices and RECORD_KINDS must not drift apart."""
    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    assert 'choices=("diagnostic", "launch_bound")' in src
    assert pe.RECORD_KINDS == ("diagnostic", "launch_bound")
    assert pe.LAUNCH_BOUND in pe.RECORD_KINDS
    assert 'default="diagnostic"' in src, (
        "the recorder must default to the WEAKER claim; a default of "
        "launch_bound would make every casual sweep look launch-authorising")


def test_the_paid_gate_requires_launch_bound():
    """Read from the launcher, so the requirement cannot quietly be dropped."""
    src = (REPO / "scripts/pod/autoinit_c1_launch.py").read_text()
    assert "required_kind=LAUNCH_BOUND" in src, (
        "pod_environment_gate no longer requires a launch-bound record")


# --- the launcher's price default -------------------------------------------

def test_the_max_price_default_comes_from_the_pricing_record():
    """One hand-maintained price, and it is the one the pricing gate verifies.

    The literal `0.99` survived the reprice to the accepted secure L40S rate of
    `1.09`. It was never a spend risk — the runner refuses a live quote ABOVE
    `--max-price` before any provider resource exists, so a stale low value can
    only refuse a launch — but the operator had to remember the real rate, and the
    price lived in two places.
    """
    import sys as _sys
    from aadistill.autoinit.c1_authorization import c1_price_per_hour_usd

    _sys.path.insert(0, str(REPO / "tests/pod"))
    from session_specs import load_session_launcher, session_args

    rate = c1_price_per_hour_usd(REPO)
    assert rate == 1.09, f"the accepted secure L40S rate moved: {rate}"
    args = session_args(load_session_launcher("autoinit_c1_launch"))
    assert args.max_price == rate, (
        f"--max-price defaults to {args.max_price}, the pricing record says {rate}")

    src = (REPO / "scripts/pod/autoinit_c1_launch.py").read_text()
    assert "default=c1_price_per_hour_usd(REPO_ROOT)" in src
    assert "default=0.99" not in src, "the stale literal is back"


def test_the_pricing_record_is_hash_verified_before_the_rate_is_used():
    """A rate read from a tampered record would be worse than a stale literal."""
    from aadistill.autoinit.c1_authorization import PRICING_PATH, load_pricing

    src = (REPO / "src/aadistill/autoinit/c1_authorization.py").read_text()
    assert "pricing_sha256" in src
    doc = load_pricing(REPO)
    assert doc["hardware"]["price_per_hour_usd"] == 1.09
    assert PRICING_PATH == "logs/phase_c1_pricing.json"


def test_a_battery_source_case_that_passed_means_the_role_leaked(tmp_path):
    """If a construction-source case PASSES on a pod, the staged view was too
    generous — or, worse, it passed vacuously against a role it never read."""
    findings = pe.evaluate_sweep(
        _outcomes(**{pe.BATTERY_SOURCE_NODEIDS[0]: "passed"}))
    assert findings["verdict"] == "FAIL"
    assert any("battery construction-source skip set" in p
               for p in findings["problems"])


def test_the_staged_battery_role_must_not_skip(tmp_path):
    """C1 stages recovery_search_v2, so its disjointness parameter has to RUN.
    A skip there would mean the local asset stopped being staged."""
    findings = pe.evaluate_sweep(
        _outcomes(**{pe.BATTERY_STAGED_ROLE_NODEID: "skipped"}))
    assert findings["verdict"] == "FAIL"
    assert any("role C1 DOES stage" in p for p in findings["problems"])


def test_the_c1_readiness_owned_skip_set_is_exact():
    """FOURTEEN, in four separately-named groups — and this is the set C1
    READINESS owns, not a claim about the whole repository, which has other
    legitimate historical and environment skips of its own.

    Named separately because they skip for different reasons: a pinned dataset
    absent, a historical source role absent, a dev-box-only self-test, a
    host-local store this session does not own.
    """
    expected = (set(pe.RENDERER_PARITY_NODEIDS) | set(pe.BATTERY_SOURCE_NODEIDS)
                | set(pe.DEVBOX_ONLY_NODEIDS) | set(pe.HOST_LOCAL_C1_NODEIDS))
    assert len(pe.RENDERER_PARITY_NODEIDS) == 7
    assert len(pe.BATTERY_SOURCE_NODEIDS) == 3
    assert len(pe.DEVBOX_ONLY_NODEIDS) == 2
    assert len(pe.HOST_LOCAL_C1_NODEIDS) == 2
    assert len(expected) == 14
    assert pe.BATTERY_STAGED_ROLE_NODEID not in expected
    findings = pe.evaluate_sweep(_outcomes())
    assert findings["expected_environment_skips"] == sorted(expected)


def test_the_non_environment_exemption_is_named_and_narrow():
    """One nodeid, with a reason, and it is NOT in the readiness-owned set.

    Widening the watch to test_recovery_continuation_session.py surfaced a
    pre-existing skip that has nothing to do with HOME, HF or staging: the branch
    it covers is only live when no verified transport exists, and one does. The
    answer is a named exemption, not a narrower watch that would also stop
    noticing real staging skips in that module.
    """
    assert len(pe.KNOWN_NON_ENVIRONMENT_SKIPS) == 1
    expected = (set(pe.RENDERER_PARITY_NODEIDS) | set(pe.BATTERY_SOURCE_NODEIDS)
                | set(pe.DEVBOX_ONLY_NODEIDS) | set(pe.HOST_LOCAL_C1_NODEIDS))
    assert not set(pe.KNOWN_NON_ENVIRONMENT_SKIPS) & expected
    findings = pe.evaluate_sweep(
        _outcomes(**{pe.KNOWN_NON_ENVIRONMENT_SKIPS[0]: "skipped"}))
    assert findings["verdict"] == "PASS", findings["problems"]
    # A different unexpected skip in that same module is still caught.
    findings = pe.evaluate_sweep(_outcomes(**{
        "tests/pod/test_recovery_continuation_session.py::test_something_new":
            "skipped"}))
    assert findings["verdict"] == "FAIL"


# --- the preregistration's operational contract ------------------------------

def test_the_prereg_gate_count_and_order_equal_the_live_session():
    """It said 10 while `spec.precheck` had held 12 since the two readiness gates
    were added — a false execution fact in the document an authorization binds."""
    import importlib.util

    doc = json.loads(
        (REPO / "logs/phase_c1_execution_preregistration.json").read_text())
    transport = doc["transport"]

    sys.path.insert(0, str(REPO / "scripts/pod"))
    loader = importlib.util.spec_from_file_location(
        "_c1_launch_for_test", REPO / "scripts/pod/autoinit_c1_launch.py")
    mod = importlib.util.module_from_spec(loader)
    sys.modules["_c1_launch_for_test"] = mod
    loader.loader.exec_module(mod)
    args = mod.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40,
         "--bundle", "aad_autoinit_00000000.bundle"])
    live = mod.spec(args).precheck
    names = [getattr(g, "__name__", "session_commit_and_lineage") for g in live]

    assert transport["n_pre_provider_gates"] == len(live) == 12
    assert transport["pre_provider_gate_order"] == names
    assert names[-2:] == ["renderer_parity_gate", "pod_environment_gate"]


def test_the_prereg_states_the_canonical_issuance_ordering():
    """Sweep BEFORE issuance. Stated backwards once, and it would refuse at $0."""
    doc = json.loads(
        (REPO / "logs/phase_c1_execution_preregistration.json").read_text())
    steps = doc["transport"]["ordering"]
    assert len(steps) == 9, steps
    joined = " ".join(steps).lower()
    sweep_at = next(i for i, s in enumerate(steps) if "launch_bound" in s)
    record_at = next(i for i, s in enumerate(steps) if "readiness record" in s)
    issue_at = next(i for i, s in enumerate(steps) if "issue the authorization" in s)
    bundle_at = next(i for i, s in enumerate(steps) if "canonical bundle" in s)
    gates_at = next(i for i, s in enumerate(steps) if "12 $0 pre-provider" in s)
    provider_at = next(i for i, s in enumerate(steps) if "provider resource" in s)
    assert sweep_at < record_at < issue_at < bundle_at < gates_at < provider_at
    assert "why_the_sweep_precedes_issuance" in doc["transport"]
    assert "ninth gate that closes it" not in json.dumps(doc)


def test_the_launch_bound_refusal_names_the_pre_authorization_tree():
    """Load-bearing operator guidance: the wrong wording produced a backwards
    issuance order that session_commit_gate would have refused at $0."""
    src = (REPO / "src/aadistill/autoinit/pod_environment.py").read_text()
    assert "PRE-AUTHORIZATION tree" in src
    assert "on the final authorized tree" not in src
    rec = _valid_record(Path("/tmp"), "diagnostic")
    ok, why = pe.verify_record(rec, REPO, required_kind=pe.LAUNCH_BOUND)
    assert not ok
    assert "PRE-AUTHORIZATION" in why and "BEFORE the authorization is issued" in why
