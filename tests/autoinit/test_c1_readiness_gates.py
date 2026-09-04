"""The two pre-provider gates added after C1 attempt 3R aborted at $0.3482.

Attempt 3R cleared `VLLM_READY → TEACHER_READY → ROPE_OK` and then died on the
pod's CPU test suite: `14 failed, 2650 passed`, no scientific stage. Twelve of the
fourteen were environment — tests reading a `$HOME` Hugging Face cache or
credential that no pod has. Ten pre-provider gates had passed.

Gate 11 (`renderer_parity_gate`) keeps the byte-for-byte rendering guarantee that
the now-skippable pytest cases used to carry alone. Gate 12
(`pod_environment_gate`) refuses unless one complete pod-like sweep has been
recorded against this exact executable.

A gate that has only ever been seen to pass is not evidence. Every case here
drives a refusal.
"""

from __future__ import annotations

import json
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

def _valid_record(tmp_path: Path) -> dict:
    """A record that binds the LIVE tree, so only the field under test differs."""
    from aadistill.autoinit.c1_authorization import c1_harness_digest

    rec = {
        "schema": pe.SCHEMA,
        "executable_head": "0" * 40,
        "tree_clean": True,
        "c1_harness_digest": c1_harness_digest(REPO)["digest"],
        "pod_test_environment_digest": pe.pod_test_environment_digest(REPO)["digest"],
        "counts": {"passed": 2700, "skipped": 48, "failed": 0, "error": 0},
        "verdict": "PASS",
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
    """Exactly the five that failed on the pod, by name."""
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


def test_junit_nodeids_round_trip(tmp_path):
    """The record names nodeids; JUnit stores file/classname/name."""
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<testsuites><testsuite>'
        '<testcase classname="tests.data.test_c1_battery" file="tests/data/test_c1_battery.py" name="test_a[tool]"/>'
        '<testcase classname="tests.pod.test_x.TestGroup" file="tests/pod/test_x.py" name="test_b"><skipped/></testcase>'
        '<testcase classname="tests.pod.test_y" file="tests/pod/test_y.py" name="test_c"><failure/></testcase>'
        '</testsuite></testsuites>')
    got = pe.read_junit(xml)
    assert got["outcomes"] == {
        "tests/data/test_c1_battery.py::test_a[tool]": "passed",
        "tests/pod/test_x.py::TestGroup::test_b": "skipped",
        "tests/pod/test_y.py::test_c": "failed"}
    assert got["counts"] == {"passed": 1, "skipped": 1, "failed": 1, "error": 0}


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

def test_the_committed_record_still_binds_the_live_executable():
    """This is gate 12 itself, run as a test. If it fails, the pod sweep is owed
    again before any C1 launch — which is the point."""
    path = REPO / pe.RECORD_PATH
    if not path.is_file():
        pytest.skip(f"{pe.RECORD_PATH} has not been produced yet")
    ok, why = pe.verify_record(json.loads(path.read_text()), REPO)
    assert ok, why
