"""The pod CPU gate must preserve its COMPLETE outcome, not only its failures.

C1 attempt 3R lost fourteen failures to a four-line tail. The `grep '^FAILED'`
added afterwards fixed that half, and attempt 5 proved the other half was still
missing: its diagnostics named both failing nodeids exactly and not one of the
99 skips. The counts force a third divergence — a test the sweep PASSED that the
pod SKIPPED — and it cannot be named, because that list died with the pod.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))

from aadistill.infrastructure.session import ExecutionCommands  # noqa: E402
from aadistill.runtime import pod_environment as pe
import summarize_pytest_outcomes as S  # noqa: E402

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"


def _junit(tmp_path: Path, cases: dict[str, tuple[str, str]]) -> Path:
    """cases: (classname, name) -> (status, reason).

    The modules are created for real: `_nodeid` reconstructs `tests/a.py::test_x`
    from the dotted classname by RESOLVING it against the filesystem, because
    pytest writes no `file` attribute. A fixture that skips that resolution would
    not exercise the parser the pod uses.
    """
    for cls, _name in cases:
        f = tmp_path / (cls.replace(".", "/") + ".py")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    parts = ['<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite>']
    for (cls, name), (status, reason) in cases.items():
        body = ""
        if status == "skipped":
            body = f'<skipped message="{reason}"/>'
        elif status == "failed":
            body = f'<failure message="{reason}"/>'
        elif status == "error":
            body = f'<error message="{reason}"/>'
        parts.append(f'<testcase classname="{cls}" name="{name}">{body}</testcase>')
    parts.append("</testsuite></testsuites>")
    p = tmp_path / "junit.xml"
    p.write_text("".join(parts))
    return p


# --- the gate actually produces the evidence ---------------------------------

def test_the_setup_gate_writes_a_junit_report():
    text = SETUP.read_text()
    assert "--junitxml=/workspace/pytest_junit.xml" in text, (
        "without a JUnit report the gate can name failures and never skips")


def test_the_setup_gate_summarizes_on_both_paths():
    """A PASSING gate whose skip set differs from the sweep's is as informative
    as a failing one, and cheaper to learn at setup cost."""
    text = SETUP.read_text()
    assert "summarize_pytest_outcomes.py" in text
    assert "--out /workspace/pytest_outcomes.json" in text
    assert "--strict" in text
    # It runs BEFORE the pass/fail branch, so it happens either way.
    assert text.index("summarize_pytest_outcomes.py") < text.index(
        '[ "$RC" -eq 0 ] || { say "test suite failed rc=$RC"; exit 1; }')


def test_the_summary_survives_a_setup_abort():
    """A setup abort never reaches artifact collection, and the launcher's own
    window is `tail -40`. The file must be pulled while the pod still exists."""
    sys.path.insert(0, str(REPO / "tests/pod"))
    from session_specs import load_session_launcher, session_args
    mod = load_session_launcher("autoinit_c1_launch")
    spec = mod.spec(session_args(mod))
    assert "/workspace/pytest_outcomes.json" in spec.setup_failure_files

    runner = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert "_collect_setup_failure_evidence(target, draw)" in runner
    assert 'return "setup_failed"' in runner
    # Collected BEFORE the return that leads to teardown.
    assert runner.index("_collect_setup_failure_evidence(target, draw)") < \
        runner.index('return "setup_failed"')


def test_the_collector_never_raises_on_a_billing_pod(monkeypatch):
    """It runs on an already-failing path while a pod bills; losing the evidence
    must not also lose the teardown."""
    import types

    from aadistill.infrastructure.session_runner import SessionRunner

    class Boom:
        def run(self, *a, **k):
            raise OSError("ssh died")

    fake = types.SimpleNamespace(
        spec=types.SimpleNamespace(
        #: The REAL type, not a SimpleNamespace. A fake that names the fields
        #: it happens to need goes stale silently the moment the type gains
        #: one -- which is exactly what happened when `workspace_root`,
        #: `checkout_root` and `min_cuda_version` were added: twenty tests
        #: failed with AttributeError on a double, not on the code.
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            remote_python="/opt/train/bin/python",
            workspace_root="/workspace", checkout_root="/workspace/aad"),
setup_failure_files=("/workspace/x.json",)),
        ev={}, say=lambda m: None)
    SessionRunner._collect_setup_failure_evidence(fake, Boom(), 1)
    got = fake.ev["setup_failure_evidence"][0]["files"]["/workspace/x.json"]
    assert "unreadable" in got and "OSError" in got


# --- the summary itself -------------------------------------------------------

def test_every_outcome_class_is_named(tmp_path):
    junit = _junit(tmp_path, {
        ("tests.a", "test_p"): ("passed", ""),
        ("tests.a", "test_s"): ("skipped", "premise absent: no artifact"),
        ("tests.a", "test_f"): ("failed", "boom"),
        ("tests.a", "test_e"): ("error", "collect"),
    })
    out = S.summarize(junit, None, tmp_path)
    assert out["counts"] == {"passed": 1, "skipped": 1, "failed": 1, "error": 1}
    assert out["failed_nodeids"] and out["error_nodeids"]
    assert len(out["all_skipped_nodeids"]) == 1
    assert "premise absent: no artifact" in json.dumps(out["skip_reasons"])
    assert len(out["skip_set_digest"]) == 64


def test_the_digest_is_order_independent_and_set_sensitive():
    a = pe.skip_set_digest(["x::b", "x::a"])
    assert a == pe.skip_set_digest(["x::a", "x::b", "x::a"])
    assert a != pe.skip_set_digest(["x::a"])


def test_the_two_divergence_shapes_are_reported_separately():
    """`expected_but_ran` is attempt 5's shape; `unexpected_skip` is the shape of
    the divergence attempt 5 never identified. Folding them together loses which
    happened."""
    cmp = pe.compare_skip_sets(["a", "b", "c"], ["b", "c", "d"])
    assert cmp["expected_but_ran"] == ["a"]
    assert cmp["unexpected_skip"] == ["d"]
    assert not cmp["identical"]
    assert pe.compare_skip_sets(["a"], ["a"])["identical"]


def test_a_divergent_skip_set_is_refused_and_the_difference_printed(tmp_path):
    """The whole point: --strict must exit non-zero and print the exact set."""
    record = tmp_path / "record.json"
    record.write_text(json.dumps(
        {"findings": {"all_skipped_nodeids": ["tests/a.py::test_s",
                                              "tests/a.py::test_only_in_sweep"]}}))
    junit = _junit(tmp_path, {
        ("tests.a", "test_s"): ("skipped", "r"),
        ("tests.a", "test_only_in_sweep"): ("passed", ""),
        ("tests.a", "test_pod_only"): ("skipped", "r"),
    })
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/pod/summarize_pytest_outcomes.py"),
         "--junit", str(junit), "--out", str(tmp_path / "o.json"),
         "--expected", str(record), "--repo", str(tmp_path), "--strict"],
        capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout
    assert "expected-skip-but-RAN" in proc.stdout
    assert "test_only_in_sweep" in proc.stdout
    assert "unexpected POD-ONLY skip" in proc.stdout
    assert "test_pod_only" in proc.stdout


def test_an_identical_skip_set_is_accepted(tmp_path):
    record = tmp_path / "record.json"
    record.write_text(json.dumps(
        {"findings": {"all_skipped_nodeids": ["tests/a.py::test_s"]}}))
    junit = _junit(tmp_path, {("tests.a", "test_s"): ("skipped", "r"),
                              ("tests.a", "test_p"): ("passed", "")})
    out = S.summarize(junit, record, tmp_path)
    assert out["comparison"]["identical"] is True


def test_a_record_without_the_skip_set_says_so_rather_than_passing(tmp_path):
    """A comparison that silently succeeds against a missing expectation is the
    vacuous-pass shape this project has been bitten by before."""
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"findings": {"counts": {"skipped": 3}}}))
    junit = _junit(tmp_path, {("tests.a", "test_s"): ("skipped", "r")})
    out = S.summarize(junit, record, tmp_path)
    assert out["comparison"]["available"] is False
    assert "only the named groups" in out["comparison"]["why"]


def test_a_missing_junit_report_does_not_mask_the_suite_exit_code(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/pod/summarize_pytest_outcomes.py"),
         "--junit", str(tmp_path / "nope.xml"), "--out", str(tmp_path / "o.json"),
         "--strict"], capture_output=True, text=True)
    assert proc.returncode == 0, "the summary must never invent a gate failure"
    assert "no JUnit report" in proc.stdout


#: A synthetic readiness contract. `evaluate_sweep` no longer owns C1's nine
#: node-id groups, so these tests declare their own -- which is the stronger
#: test: a mechanism that only ever runs against one session's contract has not
#: been shown to be a mechanism.
GROUPS = pe.ReadinessGroups(
    expected_skips={"renderer_parity": ("tests/g.py::r",),
                    "battery_source": ("tests/g.py::b",),
                    "devbox_only": ("tests/g.py::d",),
                    "host_local_c1": ("tests/g.py::h",)},
    must_pass={"leaf_transport": ("tests/g.py::l",)},
    staged_role_nodeid="tests/g.py::staged",
    known_non_environment_skips=(),
    watched=("tests/watched.py",),
)


# --- the readiness record carries the whole set ------------------------------

def test_the_record_carries_the_complete_skip_set_and_its_digest():
    outcomes = {"tests/a.py::test_p": "passed",
                "tests/a.py::test_s": "skipped",
                "tests/b.py::test_s2": "skipped"}
    findings = pe.evaluate_sweep(outcomes, {"tests/a.py::test_s": "because"},
                                groups=GROUPS)
    assert findings["all_skipped_nodeids"] == ["tests/a.py::test_s",
                                               "tests/b.py::test_s2"]
    assert findings["n_skipped"] == 2
    assert findings["skip_set_digest"] == pe.skip_set_digest(
        ["tests/a.py::test_s", "tests/b.py::test_s2"])
    assert findings["skip_reasons"]["tests/a.py::test_s"] == "because"


def test_the_named_groups_are_kept_beside_the_raw_set():
    """The complete list is forensic. The named groups are the ones that express
    the session's expectations, and a raw total must not replace them.

    The group NAMES come from the caller, and the record's keys are derived from
    them -- which is how C1 reproduces the existing schema while the mechanism
    itself names no experiment."""
    findings = pe.evaluate_sweep({}, groups=GROUPS)
    for group in ("renderer_parity_skipped_as_expected",
                  "battery_source_skipped_as_expected",
                  "devbox_only_skipped_as_expected",
                  "host_local_c1_skipped_as_expected",
                  "expected_environment_skips"):
        assert group in findings
    assert "not a target" in findings["skip_set_is_forensic_not_scientific"]


def test_a_new_unclassified_skip_moves_the_digest(tmp_path):
    """Mutation: one extra skip must change the outcome contract.

    If the digest did not move, a sweep and a pod could differ by a whole test
    and compare equal — which is the failure this whole section exists to end.
    """
    base = {"tests/a.py::test_p": "passed", "tests/a.py::test_s": "skipped"}
    before = pe.evaluate_sweep(base, groups=GROUPS)["skip_set_digest"]
    after = pe.evaluate_sweep({**base, "tests/a.py::test_p": "skipped"},
                              groups=GROUPS)
    assert after["skip_set_digest"] != before
    assert after["n_skipped"] == 2
    cmp = pe.compare_skip_sets(
        [n for n, s in base.items() if s == "skipped"],
        after["all_skipped_nodeids"])
    assert cmp["unexpected_skip"] == ["tests/a.py::test_p"]
    assert not cmp["identical"]


def test_the_junit_parser_keeps_the_reason_not_just_the_status(tmp_path):
    junit = _junit(tmp_path, {
        ("tests.a", "test_s"): ("skipped", "premise absent: no corpus_v2 here")})
    parsed = pe.read_junit(junit, tmp_path)
    assert "no corpus_v2 here" in list(parsed["skip_reasons"].values())[0]


def test_the_summariser_is_inside_the_measured_harness():
    """It can refuse a pod whose suite passed, so a grant must measure it."""
    from experiments.phase_c1.authorization import C1_HARNESS_SOURCE_FILES_V1
    assert "scripts/pod/summarize_pytest_outcomes.py" in C1_HARNESS_SOURCE_FILES_V1


@pytest.mark.parametrize("field", ["all_skipped_nodeids", "skip_set_digest",
                                   "n_skipped", "skip_reasons"])
def test_the_recorder_writes_the_new_fields(field):
    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    assert '"findings": findings,' in src, "the record embeds findings wholesale"
    assert field in json.dumps(pe.evaluate_sweep({}, {}, groups=GROUPS))


# --- failure DETAIL, not just failure names ----------------------------------
#
# C1 attempt 6 preserved 18 exact failing nodeids and not one reason, so its
# mechanism is attributed rather than proven. That is the same gap as attempt
# 3R's four-line tail and attempt 5's missing skip list, one layer further in.

def _junit_with_failure(tmp_path: Path, message: str, body: str) -> Path:
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/a.py").touch()
    p = tmp_path / "junit.xml"
    p.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite>'
        f'<testcase classname="tests.a" name="test_boom">'
        f'<failure type="AssertionError" message="{message}">{body}</failure>'
        "</testcase></testsuite></testsuites>")
    return p


def test_a_failure_message_and_traceback_survive_into_the_record(tmp_path):
    body = ("Traceback (most recent call last):\n"
            '  File "tests/a.py", line 3, in test_boom\n'
            "    assert shutil.which('python3')\n"
            "AssertionError: no interpreter on this machine")
    junit = _junit_with_failure(tmp_path, "assert 0 == 1", body)
    out = S.summarize(junit, None, tmp_path)
    nodeid = out["failed_nodeids"][0]
    detail = out["failure_details"][nodeid]
    assert detail["kind"] == "failed"
    assert detail["type"] == "AssertionError"
    assert "assert 0 == 1" in detail["message"]
    assert "no interpreter on this machine" in detail["body"]
    assert "Traceback" in detail["body"]


def test_an_error_body_survives_too(tmp_path):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/a.py").touch()
    p = tmp_path / "junit.xml"
    p.write_text('<?xml version="1.0"?><testsuites><testsuite>'
                 '<testcase classname="tests.a" name="test_collect">'
                 '<error type="CollectError" message="import failure">'
                 'ModuleNotFoundError: no module named x</error>'
                 "</testcase></testsuite></testsuites>")
    out = S.summarize(p, None, tmp_path)
    d = out["failure_details"][out["error_nodeids"][0]]
    assert d["kind"] == "error" and "ModuleNotFoundError" in d["body"]


def test_the_reason_is_printed_not_only_stored(tmp_path, capsys):
    """The launcher's window is `tail -40`; a mechanism must reach it."""
    junit = _junit_with_failure(tmp_path, "no interpreter on this machine", "x")
    S.report(S.summarize(junit, None, tmp_path))
    printed = capsys.readouterr().out
    assert "WHY test_boom:" in printed
    assert "no interpreter on this machine" in printed


def test_mutation_dropping_failure_capture_is_caught(tmp_path, monkeypatch):
    """If read_junit stopped recording bodies, these tests must go red."""
    from aadistill.runtime import pod_environment as pe_mod
    real = pe_mod.read_junit

    def stripped(*a, **k):
        out = real(*a, **k)
        out["failure_details"] = {}
        return out

    monkeypatch.setattr(S.pe, "read_junit", stripped)
    junit = _junit_with_failure(tmp_path, "m", "b")
    out = S.summarize(junit, None, tmp_path)
    assert out["failure_details"] == {}, "the mutation did not take"
    with pytest.raises(KeyError):
        _ = out["failure_details"][out["failed_nodeids"][0]]


def test_the_raw_cpu_test_artifacts_are_retrieved_before_teardown():
    """A parser bug must not again be the only surviving evidence."""
    sys.path.insert(0, str(REPO / "tests/pod"))
    from session_specs import load_session_launcher, session_args
    mod = load_session_launcher("autoinit_c1_launch")
    files = mod.spec(session_args(mod)).setup_failure_files
    for raw in ("/workspace/pytest_outcomes.json", "/workspace/pytest_junit.xml",
                "/workspace/pytest.log"):
        assert raw in files, f"{raw} would not survive a setup abort"
    assert not any("token" in f.lower() or "credential" in f.lower() for f in files)
