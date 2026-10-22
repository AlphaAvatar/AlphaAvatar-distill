"""Code targets are verified by execution: every declared test must RUN and pass,
the whole output is judged, and model-written code runs without the host."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.code_exec import (  # noqa: E402
    ESTABLISHES, Limits, bwrap_sandbox, probe_sandbox, run_asserts, run_stdio)

GOOD = "def add(a, b):\n    return a + b\n"
FAST = Limits(wall_seconds=5, cpu_seconds=3, memory_mb=256)
TESTS = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]


def test_a_correct_program_passes_and_says_only_what_tests_establish():
    r = run_asserts(GOOD, TESTS, limits=FAST)
    assert r["passed"] and r["status"] == "passed"
    assert r["tests_run"] == r["tests_passed"] == r["tests_total"] == 2
    assert r["establishes"] == ESTABLISHES and "not full semantic correctness" in ESTABLISHES


def test_a_wrong_program_fails_and_says_which_test():
    r = run_asserts(GOOD, ["assert add(2, 3) == 5", "assert add(2, 2) == 5"], limits=FAST)
    assert not r["passed"] and r["status"] == "failed"
    assert r["tests_run"] == 2 and r["tests_passed"] == 1
    assert r["failures"][0]["test"] == 1 and "AssertionError" in r["failures"][0]["error"]
    assert r["establishes"] is None


def test_zero_tests_is_not_a_pass():
    r = run_asserts(GOOD, [], limits=FAST)
    assert not r["passed"] and r["status"] == "no_tests"
    assert not run_stdio(GOOD, [], limits=FAST)["passed"]


@pytest.mark.parametrize("leave", ["import sys\nsys.exit(0)\n", "raise SystemExit\n"])
def test_a_program_that_exits_before_its_tests_ran_none_of_them(leave):
    r = run_asserts(GOOD + leave, TESTS, limits=FAST)
    assert r["returncode"] == 0                       # exit status 0 ...
    assert not r["passed"] and r["status"] == "exited_early" and r["tests_run"] == 0


def test_a_hard_exit_leaves_no_report_and_fails():
    r = run_asserts(GOOD + "import os\nos._exit(0)\n", TESTS, limits=FAST)
    assert r["returncode"] == 0 and not r["passed"] and r["status"] == "no_report"


def test_an_exit_inside_a_test_stops_the_count_and_fails():
    code = GOOD + "def bail():\n    import sys\n    sys.exit(0)\n"
    r = run_asserts(code, ["assert add(1, 1) == 2", "bail()", "assert add(2, 2) == 4"], limits=FAST)
    assert not r["passed"] and r["tests_run"] == 2 and r["tests_passed"] == 1


def test_a_forged_report_without_the_nonce_is_refused():
    forged = ('import os, sys, json\nos.write(int(sys.argv[1]), json.dumps({"nonce": "x", '
              '"stage": "done", "tests_total": 2, "tests_run": 2, "tests_passed": 2, '
              '"failures": []}).encode())\nos._exit(0)\n')
    r = run_asserts(GOOD + forged, TESTS, limits=FAST)
    assert not r["passed"] and r["status"] == "no_report"


def test_the_program_cannot_rewrite_its_tests():
    code = GOOD + "import json\nopen('job.json', 'w').write(json.dumps({'tests': []}))\n"
    r = run_asserts(code, ["assert add(2, 2) == 5"], limits=FAST)
    assert not r["passed"] and r["tests_run"] == 1


def test_an_infinite_loop_is_stopped():
    r = run_asserts("while True:\n    pass\n", ["assert True"], limits=Limits(wall_seconds=3, cpu_seconds=1))
    assert not r["passed"] and r["status"] in ("no_report", "timeout")


def test_a_memory_blowup_is_stopped():
    r = run_asserts("x = bytearray(2_000_000_000)\n", ["assert True"], limits=FAST)
    assert not r["passed"] and r["status"] == "program_error"


def test_the_environment_is_not_inherited(monkeypatch):
    monkeypatch.setenv("AAD_SECRET_FOR_TEST", "s3cret")
    r = run_asserts("import os\n", ["assert 'AAD_SECRET_FOR_TEST' not in os.environ"], limits=FAST)
    assert r["passed"]


SUM = "a, b = map(int, input().split())\nprint(a + b)\n"


def test_stdio_cases_are_compared_whitespace_normalized():
    assert run_stdio(SUM, [("1 2\n", "3"), ("5 5\n", "10\n")], limits=FAST)["passed"]
    bad = run_stdio(SUM, [("1 2\n", "4"), ("5 5\n", "10")], limits=FAST)
    assert not bad["passed"] and bad["cases_run"] == 1 and bad["status"] == "wrong_output"


def test_stdio_a_correct_prefix_with_more_or_less_output_fails():
    more = run_stdio("print(1, 2, 3, 4)\n", [("", "1 2 3")], limits=FAST)
    less = run_stdio("print(1, 2)\n", [("", "1 2 3")], limits=FAST)
    assert not more["passed"] and not less["passed"]


N = 50_000
LONG = " ".join(str(i) for i in range(N))


def test_stdio_a_correct_long_output_is_judged_whole_and_passes():
    assert len(LONG) > 100_000
    assert run_stdio(f"print(*range({N}))\n", [("", LONG)], limits=FAST)["passed"]


def test_stdio_a_wrong_long_output_with_the_right_tail_fails():
    # the old verifier compared a 4000-character TAIL: this program passed it
    code = f"print(-1, *range(1, {N}))\n"
    r = run_stdio(code, [("", LONG)], limits=FAST)
    assert not r["passed"] and r["status"] == "wrong_output"


def test_stdio_output_over_the_limit_fails_as_such_and_is_never_compared():
    small = Limits(wall_seconds=5, cpu_seconds=3, memory_mb=256, output_bytes=10_000)
    r = run_stdio(f"print(*range({N}))\n", [("", LONG)], limits=small)
    assert not r["passed"] and r["status"] == "output_limit"
    endless = run_stdio("while True:\n    print('y' * 1000)\n", [("", "y")], limits=small)
    assert not endless["passed"] and endless["status"] == "output_limit"


def test_stdio_right_output_but_a_failing_exit_fails():
    r = run_stdio("print(3)\nraise SystemExit(2)\n", [("", "3")], limits=FAST)
    assert not r["passed"] and r["status"] == "error"


# --- the sandbox ---------------------------------------------------------------

def _sandbox_or_skip(protected):
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed here")
    sb = bwrap_sandbox()
    probe = probe_sandbox(sb, protected, limits=FAST)
    if "error" in probe:
        pytest.skip(f"bubblewrap cannot create namespaces here: {probe['error']}")
    return sb, probe


def test_the_sandbox_hides_the_host(tmp_path):
    secret = tmp_path / "credentials.toml"
    secret.write_text("api_key = 'x'\n")
    sb, probe = _sandbox_or_skip([str(secret), str(tmp_path)])
    assert probe["isolated"], probe
    assert probe["seen"]["network_interfaces"] == ["lo"]
    code = f"import os\nVISIBLE = os.path.exists({str(secret)!r})\n"
    r = run_asserts(code, ["assert not VISIBLE"], limits=FAST, sandbox=sb)
    assert r["passed"] and r["sandboxed"], r


def test_a_correct_program_passes_inside_the_sandbox():
    sb, _ = _sandbox_or_skip([])
    assert run_asserts(GOOD, TESTS, limits=FAST, sandbox=sb)["passed"]
    assert run_stdio(SUM, [("1 2\n", "3")], limits=FAST, sandbox=sb)["passed"]


def test_no_sandbox_reaches_the_host_and_the_probe_says_so(tmp_path):
    secret = tmp_path / "credentials.toml"
    secret.write_text("x")
    probe = probe_sandbox(lambda workdir: [], [str(secret)], limits=FAST)
    assert not probe["isolated"] and str(secret) in probe["reached"]["visible"]


def test_a_missing_sandbox_binary_is_not_isolated():
    probe = probe_sandbox(bwrap_sandbox(bwrap="/nonexistent/bwrap"), [], limits=FAST)
    assert not probe["isolated"] and "error" in probe
    assert not os.path.exists("/nonexistent/bwrap")
