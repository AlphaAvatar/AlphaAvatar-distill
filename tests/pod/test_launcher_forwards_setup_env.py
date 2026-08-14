"""A setup script's environment is a contract with its launcher. Check it.

`e6b_setup.sh` reads `TEACHER_REVISION` because E6b computes KD and must pin the
teacher. Its launcher was derived from E6's, which had no teacher and therefore
never forwarded that variable. The omission survived the derivation, and the pod
died with a bare `KeyError: 'TEACHER_REVISION'` at `INIT_READY` — after both
venvs were built and the Stage 1 init downloaded and hash-verified.

That failure is invisible to every other guard: the pod simulation runs the test
suite, not the setup script; the shell parses fine; the variable is simply absent
at runtime. So the contract is checked statically here — for every setup script,
every variable it reads must be forwarded by the launcher that runs it, or have a
default inside the setup itself.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
POD = REPO / "scripts/pod"

# Provided by the pod image or by the setup script's own preamble, not by the
# launcher's ssh invocation.
AMBIENT = {
    "HF_TOKEN", "HF_HOME", "PATH", "HOME", "DEBIAN_FRONTEND", "PYTHONPATH",
    "UV_PROJECT_ENVIRONMENT", "REPO", "WS", "STATUS", "LOG", "PY",
    "UV_PID", "TRIP_S", "GRACE_S", "UV_TRIP_S", "UV_GRACE_S",
    # Derived inside the setup's own preamble from $E8B_SESSION, which IS
    # forwarded. Listing them here is not a loophole: the launcher cannot forward
    # a value the setup computes for itself.
    "NEED_DEPTH", "NEED_COMPRESSED", "SESSION", "RC", "UV_MAX_S", "TESTS_MAX_S",
}

def _launcher_for(setup: Path):
    """A launcher may be bash or Python; both forward the same contract.

    E7's orchestrator is `e7_launch.py`. A pairing rule that only knew about
    `.sh` would silently stop checking the very contract whose omission killed
    the E6b setup at INIT_READY after both venvs were built.
    """
    stem = setup.name[: -len("_setup.sh")]
    for cand in (POD / f"{stem}_launch.sh", POD / f"{stem}_launch.py"):
        if cand.is_file():
            return cand
    return None


PAIRS = [(_launcher_for(p), p) for p in sorted(POD.glob("*_setup.sh"))
         if _launcher_for(p) is not None]


def required_env(setup_text: str) -> set[str]:
    """Variables the setup consumes without defaulting them itself."""
    read = set(re.findall(r"os\.environ\[['\"]([A-Z][A-Z0-9_]+)['\"]\]", setup_text))
    read |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]+)\}", setup_text))
    read |= set(re.findall(r"\$([A-Z][A-Z0-9_]+)\b", setup_text))
    # Anything the setup gives a default to, or assigns, is not the launcher's job.
    defaulted = set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):-", setup_text))
    assigned = set(re.findall(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]+)=", setup_text, re.M))
    return read - defaulted - assigned - AMBIENT


def test_pairs_are_discovered():
    """A discovery bug here would make every assertion below vacuous."""
    assert PAIRS, "no launcher/setup pairs found; the detector is broken"


def forwards(launch: Path, var: str) -> bool:
    """Is `var` set on the command line that runs the setup?

    Two shapes, because launchers come in two languages. Bash writes
    `VAR=$VAR`; Python interpolates, `f"VAR={self.a.thing}"`. A detector that
    knew only the first would silently stop checking the moment an orchestrator
    was rewritten — which is exactly how the E6b setup came to die on an
    unforwarded TEACHER_REVISION after both venvs were built.
    """
    text = launch.read_text()
    if re.search(rf"\b{var}=\$\{{?{var}\b", text):
        return True
    return bool(re.search(rf"\b{var}=\{{[^}}]+\}}", text))


def python_source_is_non_empty(launch: Path, var: str) -> bool:
    """The interpolated value must come from a required or defaulted option."""
    text = launch.read_text()
    m = re.search(rf"\b{var}=\{{\s*self\.a\.([a-z0-9_]+)", text)
    if not m:
        return False
    flag = "--" + m.group(1).replace("_", "-")
    for decl in re.findall(rf'add_argument\(\s*"{re.escape(flag)}"(.*?)\)',
                           text, re.S):
        if "required=True" in decl:
            return True
        d = re.search(r'default=("([^"]*)"|[^,\s)]+)', decl)
        if d and d.group(2) != "" and d.group(1) not in ("None", '""'):
            return True
    return False


@pytest.mark.parametrize("launch,setup", PAIRS,
                         ids=lambda p: p.name if isinstance(p, Path) else str(p))
def test_launcher_forwards_every_variable_the_setup_reads(launch, setup):
    needed = required_env(setup.read_text())
    missing = sorted(v for v in needed if not forwards(launch, v))
    assert not missing, (
        f"{setup.name} reads {missing} but {launch.name} never forwards "
        f"{'it' if len(missing) == 1 else 'them'} over ssh. The pod fails at "
        "runtime with a bare KeyError, after setup has already been paid for.")


@pytest.mark.parametrize("launch,setup", PAIRS,
                         ids=lambda p: p.name if isinstance(p, Path) else str(p))
def test_forwarded_variables_have_a_launcher_side_default(launch, setup):
    """Forwarding an unset variable forwards an empty string, which is worse."""
    needed = required_env(setup.read_text())
    # ...unless the setup itself refuses an empty one. `${VAR:?msg}` exits
    # before the value is ever consumed, which is a stronger guarantee than a
    # launcher-side default rather than a weaker one: the failure is immediate
    # and named, instead of an empty string flowing into a command. The unset
    # cases are executed, not assumed, in
    # tests/pod/test_continuation_rehearsal.py.
    # `test_launcher_forwards_every_variable_the_setup_reads` still requires the
    # launcher to forward these; only the default requirement is lifted.
    needed -= set(re.findall(r"\$\{([A-Z][A-Z0-9_]+):\?", setup.read_text()))
    launch_text = launch.read_text()
    if launch.suffix == ".py":
        undefaulted = sorted(v for v in needed
                             if not python_source_is_non_empty(launch, v))
    else:
        undefaulted = sorted(
            v for v in needed
            if re.search(rf"\b{v}=\$\{{?{v}\b", launch_text)
            and not re.search(rf"^{v}=\$\{{{v}:[-?]", launch_text, re.M)
            and not re.search(rf"^{v}=\$\{{{v}:\?", launch_text, re.M))
    assert not undefaulted, (
        f"{launch.name} forwards {undefaulted} without a default or a required "
        "marker; if the caller does not export it the pod receives an empty "
        "value and fails in a way that looks like a data problem")
