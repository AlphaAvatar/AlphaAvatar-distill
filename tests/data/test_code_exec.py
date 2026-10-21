"""Code targets are verified by execution, and a bad program cannot hang the verifier."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.code_exec import Limits, run_asserts, run_stdio  # noqa: E402

GOOD = "def add(a, b):\n    return a + b\n"
FAST = Limits(wall_seconds=5, cpu_seconds=3, memory_mb=256)


def test_a_correct_program_passes_its_asserts():
    assert run_asserts(GOOD, ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"], limits=FAST)["passed"]


def test_a_wrong_program_fails_and_says_why():
    r = run_asserts(GOOD, ["assert add(2, 2) == 5"], limits=FAST)
    assert not r["passed"] and r["status"] == "error" and "AssertionError" in r["stderr_tail"]


def test_an_infinite_loop_is_stopped():
    r = run_asserts("while True:\n    pass\n", [], limits=Limits(wall_seconds=3, cpu_seconds=1))
    assert not r["passed"] and r["status"] in ("killed", "timeout")


def test_a_memory_blowup_is_stopped():
    r = run_asserts("x = bytearray(2_000_000_000)\n", [], limits=FAST)
    assert not r["passed"]


def test_the_environment_is_not_inherited(monkeypatch):
    monkeypatch.setenv("AAD_SECRET_FOR_TEST", "s3cret")
    r = run_asserts("import os\nassert 'AAD_SECRET_FOR_TEST' not in os.environ\n", [], limits=FAST)
    assert r["passed"]


def test_stdio_cases_are_compared_whitespace_normalized():
    code = "a, b = map(int, input().split())\nprint(a + b)\n"
    assert run_stdio(code, [("1 2\n", "3"), ("5 5\n", "10\n")], limits=FAST)["passed"]
    bad = run_stdio(code, [("1 2\n", "4"), ("5 5\n", "10")], limits=FAST)
    assert not bad["passed"] and bad["cases_run"] == 1
