"""Run a candidate program against its tests in a separate, limited process.

A code target is verified by EXECUTION, never by reading it: the candidate and
its tests run in a fresh ``python -I`` interpreter (isolated: no user site, no
PYTHON* environment variables), in an empty temporary directory, with CPU-time,
address-space, file-size and open-file limits and a wall-clock timeout, and with
an environment that carries nothing but ``PATH``.

Two modes, because benchmarks state tests in two ways:

* ``asserts`` -- the tests are Python statements appended to the program
  (MBPP-style); the candidate passes when the process exits 0;
* ``stdio`` -- each case is ``(stdin, expected_stdout)``; the candidate passes
  a case when its whitespace-normalized stdout equals the expectation.

This is containment for an honest mistake (an infinite loop, a memory blowup),
NOT a security sandbox: it does not block network access or filesystem reads.
Untrusted model-written code belongs on a disposable machine; the caller owns
that choice. Nothing here names a dataset.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

CODE_EXEC_SCHEMA = "aadistill.data.code_exec/v1"


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 10.0
    cpu_seconds: int = 10
    memory_mb: int = 1024
    file_mb: int = 16
    open_files: int = 64


def _limiter(lim: Limits):
    def apply():
        mb = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (lim.memory_mb * mb, lim.memory_mb * mb))
        resource.setrlimit(resource.RLIMIT_FSIZE, (lim.file_mb * mb, lim.file_mb * mb))
        resource.setrlimit(resource.RLIMIT_NOFILE, (lim.open_files, lim.open_files))
        os.setsid()
    return apply


def _run(source: str, stdin: str, lim: Limits) -> tuple[str, int | None, str, str]:
    with tempfile.TemporaryDirectory(prefix="aad_code_") as d:
        path = os.path.join(d, "candidate.py")
        with open(path, "w") as f:
            f.write(source)
        try:
            p = subprocess.run([sys.executable, "-I", path], input=stdin, cwd=d,
                               capture_output=True, text=True, timeout=lim.wall_seconds,
                               env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                               preexec_fn=_limiter(lim))
        except subprocess.TimeoutExpired:
            return "timeout", None, "", ""
        status = "ok" if p.returncode == 0 else ("killed" if p.returncode < 0 else "error")
        return status, p.returncode, p.stdout[-4000:], p.stderr[-2000:]


def run_asserts(code: str, tests: Sequence[str], *, setup: str = "",
                limits: Limits = Limits()) -> dict:
    """Pass when the program followed by every test statement exits 0."""
    source = "\n".join([setup, code, "", *tests, ""])
    status, rc, out, err = _run(source, "", limits)
    return {"schema": CODE_EXEC_SCHEMA, "mode": "asserts", "passed": status == "ok",
            "status": status, "returncode": rc, "stderr_tail": err}


def run_stdio(code: str, cases: Sequence[tuple[str, str]], *,
              limits: Limits = Limits()) -> dict:
    """Pass when every case's normalized stdout equals its expectation."""
    results = []
    for stdin, expected in cases:
        status, rc, out, err = _run(code, stdin, limits)
        ok = status == "ok" and out.split() == expected.split()
        results.append({"status": status, "passed": ok})
        if not ok:
            break                                   # the first failure decides
    return {"schema": CODE_EXEC_SCHEMA, "mode": "stdio",
            "passed": bool(results) and all(r["passed"] for r in results),
            "cases_run": len(results), "cases_total": len(cases),
            "first_failure": next((r["status"] for r in results if not r["passed"]), None)}
