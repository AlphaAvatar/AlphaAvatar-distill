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

def _valid_record(tmp_path: Path) -> dict:
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

def test_the_committed_record_still_binds_the_live_executable():
    """This is gate 12 itself, run as a test. If it fails, the pod sweep is owed
    again before any C1 launch — which is the point."""
    path = REPO / pe.RECORD_PATH
    if not path.is_file():
        pytest.skip(f"{pe.RECORD_PATH} has not been produced yet")
    ok, why = pe.verify_record(json.loads(path.read_text()), REPO)
    assert ok, why


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
           "verdict": "PASS", "problems": []}
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
