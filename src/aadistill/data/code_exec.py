"""Run a candidate program against its tests in a separate, limited process.

A code target is verified by EXECUTION, never by reading it. The candidate runs
in a fresh interpreter (``-I``: no user site, no PYTHON* variables, and the base
interpreter rather than any virtualenv, so only the standard library), in an
empty temporary directory, with CPU-time, address-space, file-size and
open-file limits, a wall-clock timeout, and an environment holding only
``PATH``.

Two modes, because benchmarks state tests in two ways:

* ``asserts`` -- the tests are Python statements (MBPP-style). A driver runs the
  setup, the program and then each test in turn in one namespace, and reports
  over a private pipe, under a per-run nonce, how many tests RAN and how many
  PASSED. A run passes only when it declares at least one test, the report
  arrives with its nonce, and every declared test ran and passed. Exit status 0
  is not evidence: a program that leaves before its tests (``sys.exit``,
  ``os._exit``) ran none of them, and fails saying so.
* ``stdio`` -- each case is ``(stdin, expected_stdout)``. The WHOLE stdout is
  compared, whitespace-normalized, and the program must exit 0. An output that
  reaches ``Limits.output_bytes`` fails as ``output_limit``; it is never
  compared on a truncated prefix or tail.

A pass establishes that the declared tests passed (``ESTABLISHES``). That is
evidence bounded by those tests, not full semantic correctness, and a result
never says more.

Isolation. The limits contain an honest mistake -- an infinite loop, a memory
blowup, a flood of output. They do not stop a program from reading files or
opening sockets. Model-written code therefore runs inside a ``sandbox``:
``bwrap_sandbox`` gives it no network, no inherited environment, and a
filesystem holding only the system's ``/usr``, the interpreter's own prefix
and the work directory -- no home directory, no credentials, no mounted assets.
``probe_sandbox`` runs a canary inside a sandbox and reports what it could
reach; a caller verifying untrusted code requires a clean probe first. Without
a sandbox, run only trusted code, such as a dataset's human reference
solutions. Nothing here names a dataset or a path to protect: the caller
supplies them.
"""
from __future__ import annotations

import json
import os
import resource
import secrets
import select
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

CODE_EXEC_SCHEMA = "aadistill.data.code_exec/v2"
ESTABLISHES = ("the declared tests ran and passed: evidence bounded by those tests, "
               "not full semantic correctness")

#: work dir -> argv prefix that runs a command inside the sandbox with that
#: directory as its working directory.
Sandbox = Callable[[str], list[str]]

_PATH = "/usr/bin:/bin"
_WORK = "/work"
_MAX_FAILURES = 20


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 10.0
    cpu_seconds: int = 10
    memory_mb: int = 1024
    #: bounds stdout, stderr and every file the program writes
    output_bytes: int = 4 * 1024 * 1024
    open_files: int = 64


def _limiter(lim: Limits):
    def apply():
        mb = 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (lim.memory_mb * mb, lim.memory_mb * mb))
        # one byte over the bound, so a program that reached it is distinguishable
        cap = lim.output_bytes + 1
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap, cap))
        resource.setrlimit(resource.RLIMIT_NOFILE, (lim.open_files, lim.open_files))
        os.setsid()
    return apply


def _interpreter() -> str:
    """The base interpreter: a virtualenv's packages are not the candidate's."""
    return os.path.realpath(sys.executable)


def bwrap_sandbox(*, bwrap: str = "bwrap", python_prefix: str | None = None,
                  read_only: Sequence[str] = ()) -> Sandbox:
    """A bubblewrap sandbox: new user, network, PID, IPC and UTS namespaces, and
    a filesystem of ``/usr`` (with the usual ``/lib``, ``/bin`` links), the
    interpreter's prefix, any ``read_only`` paths, a private ``/tmp`` and the
    work directory -- nothing else."""
    prefixes = sorted({python_prefix or sys.base_prefix,
                       os.path.dirname(os.path.dirname(_interpreter()))})

    def wrap(workdir: str) -> list[str]:
        args = [bwrap, "--unshare-all", "--die-with-parent", "--new-session",
                "--ro-bind", "/usr", "/usr"]
        for top in ("/lib", "/lib64", "/bin", "/sbin"):
            if os.path.islink(top):
                args += ["--symlink", os.readlink(top), top]
            elif os.path.isdir(top):
                args += ["--ro-bind", top, top]
        for p in [*prefixes, *read_only]:
            args += ["--ro-bind", p, p]
        return args + ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                       "--bind", workdir, _WORK, "--chdir", _WORK]
    return wrap


def _run(argv: list[str], workdir: str, stdin: str, lim: Limits,
         sandbox: Sandbox | None, pass_fds: Sequence[int] = ()) -> dict:
    """One process. stdout and stderr go to files outside the work directory,
    bounded by the file-size limit, and are read back whole."""
    with tempfile.TemporaryDirectory(prefix="aad_cap_") as cap:
        out_path, err_path = os.path.join(cap, "out"), os.path.join(cap, "err")
        cmd = (sandbox(workdir) if sandbox else []) + [_interpreter(), "-I", *argv]
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                 cwd=workdir, env={"PATH": _PATH},
                                 preexec_fn=_limiter(lim), pass_fds=tuple(pass_fds))
            try:
                p.communicate(stdin.encode(), timeout=lim.wall_seconds)
                status = ("exited" if p.returncode == 0 else
                          "killed" if p.returncode < 0 else "error")
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                status = "timeout"
        size = os.path.getsize(out_path)
        if size > lim.output_bytes or os.path.getsize(err_path) > lim.output_bytes:
            status = "output_limit"
        with open(out_path, "rb") as f:
            stdout = f.read(lim.output_bytes + 1).decode("utf-8", "replace")
        with open(err_path, "rb") as f:
            f.seek(max(0, os.path.getsize(err_path) - 2000))
            stderr_tail = f.read().decode("utf-8", "replace")
    return {"status": status, "returncode": p.returncode, "stdout": stdout,
            "stdout_bytes": size, "stderr_tail": stderr_tail}


_DRIVER = r'''
import json, os, sys
fd = int(sys.argv[1])
nonce = sys.stdin.readline().strip()
with open("job.json") as f:
    job = json.load(f)
os.remove("job.json")
report = {"nonce": nonce, "stage": "setup", "tests_total": len(job["tests"]),
          "tests_run": 0, "tests_passed": 0, "failures": []}
def send():
    os.write(fd, json.dumps(report).encode())
    os.close(fd)
    os._exit(0)
def why(e):
    return f"{type(e).__name__}: {str(e)[:300]}"
ns = {"__name__": "__main__", "__builtins__": __builtins__}
try:
    exec(compile(job["setup"], "setup.py", "exec"), ns)
    report["stage"] = "program"
    exec(compile(job["code"], "candidate.py", "exec"), ns)
except SystemExit as e:
    report["exited"] = f"SystemExit({e.code!r}) before the tests"
    send()
except BaseException as e:
    report["error"] = why(e)
    send()
report["stage"] = "tests"
for i, t in enumerate(job["tests"]):
    report["tests_run"] += 1
    try:
        exec(compile(t, f"test_{i}", "exec"), ns)
        report["tests_passed"] += 1
    except SystemExit as e:
        report["failures"].append({"test": i, "error": f"SystemExit({e.code!r})"})
        break
    except BaseException as e:
        if len(report["failures"]) < job["max_failures"]:
            report["failures"].append({"test": i, "error": why(e)})
report["stage"] = "done"
send()
'''


def _drain(fd: int, *, grace: float = 1.0, limit: int = 1 << 20) -> bytes:
    """What the pipe holds, without waiting on a descendant that kept it open."""
    chunks, got, deadline = [], 0, time.monotonic() + grace
    while got < limit:
        ready, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
        if not ready:
            break
        b = os.read(fd, 65536)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def run_asserts(code: str, tests: Sequence[str], *, setup: str = "",
                limits: Limits = Limits(), sandbox: Sandbox | None = None) -> dict:
    """Pass only when every declared test ran and passed (see the module doc)."""
    tests = list(tests)
    base = {"schema": CODE_EXEC_SCHEMA, "mode": "asserts", "sandboxed": sandbox is not None,
            "tests_total": len(tests), "tests_run": 0, "tests_passed": 0}
    if not tests:
        return {**base, "passed": False, "status": "no_tests", "establishes": None}
    nonce = secrets.token_hex(16)
    rfd, wfd = os.pipe()
    try:
        with tempfile.TemporaryDirectory(prefix="aad_code_") as d:
            with open(os.path.join(d, "driver.py"), "w") as f:
                f.write(_DRIVER)
            with open(os.path.join(d, "job.json"), "w") as f:
                json.dump({"setup": setup, "code": code, "tests": tests,
                           "max_failures": _MAX_FAILURES}, f)
            run = _run(["driver.py", str(wfd)], d, nonce + "\n", limits, sandbox,
                       pass_fds=(wfd,))
        os.close(wfd)
        wfd = -1
        raw = _drain(rfd)
    finally:
        for fd in (rfd, wfd):
            if fd >= 0:
                os.close(fd)
    try:
        rep = json.loads(raw)
        rep = rep if isinstance(rep, dict) and rep.get("nonce") == nonce else None
    except ValueError:
        rep = None
    out = {**base, "returncode": run["returncode"], "stderr_tail": run["stderr_tail"]}
    if run["status"] in ("timeout", "output_limit"):
        status = run["status"]
    elif rep is None:
        status = "no_report"                    # left without the driver: os._exit, a crash, a limit
    elif "exited" in rep:
        status = "exited_early"
    elif "error" in rep:
        status = "program_error"
    elif rep["stage"] == "done" and rep["tests_run"] == rep["tests_passed"] == len(tests):
        status = "passed"
    else:
        status = "failed"
    if rep is not None:
        out.update(tests_run=rep["tests_run"], tests_passed=rep["tests_passed"],
                   failures=rep["failures"], detail=rep.get("exited") or rep.get("error"))
    passed = status == "passed"
    return {**out, "passed": passed, "status": status,
            "establishes": ESTABLISHES if passed else None}


def run_stdio(code: str, cases: Sequence[tuple[str, str]], *,
              limits: Limits = Limits(), sandbox: Sandbox | None = None) -> dict:
    """Pass only when every case exits 0 and its WHOLE normalized stdout equals
    the expectation. The first failing case ends the run."""
    cases = list(cases)
    base = {"schema": CODE_EXEC_SCHEMA, "mode": "stdio", "sandboxed": sandbox is not None,
            "cases_total": len(cases)}
    if not cases:
        return {**base, "passed": False, "status": "no_tests", "cases_run": 0,
                "cases_passed": 0, "first_failure": None, "establishes": None}
    run_n = passed_n = 0
    first_failure = None
    for i, (stdin, expected) in enumerate(cases):
        with tempfile.TemporaryDirectory(prefix="aad_code_") as d:
            with open(os.path.join(d, "candidate.py"), "w") as f:
                f.write(code)
            run = _run(["candidate.py"], d, stdin, limits, sandbox)
        run_n += 1
        if run["status"] != "exited":
            status = run["status"]
        elif run["stdout"].split() != expected.split():
            status = "wrong_output"
        else:
            passed_n += 1
            continue
        first_failure = {"case": i, "status": status, "returncode": run["returncode"],
                         "stdout_bytes": run["stdout_bytes"], "stderr_tail": run["stderr_tail"]}
        break
    passed = passed_n == len(cases)
    return {**base, "passed": passed, "status": "passed" if passed else first_failure["status"],
            "cases_run": run_n, "cases_passed": passed_n, "first_failure": first_failure,
            "establishes": ESTABLISHES if passed else None}


_CANARY = r'''
import json, os, socket, sys
paths = json.loads(sys.stdin.read())
print(json.dumps({
    "network_interfaces": sorted(n for _i, n in socket.if_nameindex()),
    "visible": [p for p in paths if os.path.exists(p)],
    "environment": sorted(os.environ),
}))
'''

#: what an interpreter sets for itself; anything else came from outside
_OWN_ENVIRONMENT = frozenset({"PATH", "PWD", "LC_CTYPE", "SHLVL", "_"})


def probe_sandbox(sandbox: Sandbox, protected: Sequence[str], *,
                  limits: Limits = Limits()) -> dict:
    """Run a canary inside ``sandbox`` and report what it reached: network
    interfaces other than loopback, any of the ``protected`` paths, and
    environment variables the interpreter did not set itself. ``isolated`` is
    the conjunction; a probe that cannot run is not isolated. Sends no traffic."""
    with tempfile.TemporaryDirectory(prefix="aad_probe_") as d:
        with open(os.path.join(d, "canary.py"), "w") as f:
            f.write(_CANARY)
        try:
            run = _run(["canary.py"], d, json.dumps(list(protected)), limits, sandbox)
        except OSError as e:                    # the sandbox binary itself is missing
            return {"isolated": False, "error": f"{type(e).__name__}: {e}"}
    if run["status"] != "exited":
        return {"isolated": False, "error": f"canary {run['status']}: {run['stderr_tail'][-300:]}"}
    seen = json.loads(run["stdout"])
    leaks = {"network_interfaces": [n for n in seen["network_interfaces"] if n != "lo"],
             "visible": seen["visible"],
             "environment": [k for k in seen["environment"] if k not in _OWN_ENVIRONMENT]}
    return {"isolated": not any(leaks.values()), "reached": leaks, "seen": seen}
