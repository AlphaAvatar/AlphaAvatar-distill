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
}

PAIRS = [(POD / f"{p.name[:-len('_setup.sh')]}_launch.sh", p)
         for p in sorted(POD.glob("*_setup.sh"))
         if (POD / f"{p.name[:-len('_setup.sh')]}_launch.sh").is_file()]


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


@pytest.mark.parametrize("launch,setup", PAIRS,
                         ids=lambda p: p.name if isinstance(p, Path) else str(p))
def test_launcher_forwards_every_variable_the_setup_reads(launch, setup):
    needed = required_env(setup.read_text())
    launch_text = launch.read_text()
    # The forwarding happens on the ssh line that runs the setup script.
    missing = sorted(v for v in needed
                     if not re.search(rf"\b{v}=\$\{{?{v}\b", launch_text))
    assert not missing, (
        f"{setup.name} reads {missing} but {launch.name} never forwards "
        f"{'it' if len(missing) == 1 else 'them'} over ssh. The pod fails at "
        "runtime with a bare KeyError, after setup has already been paid for.")


@pytest.mark.parametrize("launch,setup", PAIRS,
                         ids=lambda p: p.name if isinstance(p, Path) else str(p))
def test_forwarded_variables_have_a_launcher_side_default(launch, setup):
    """Forwarding an unset variable forwards an empty string, which is worse."""
    needed = required_env(setup.read_text())
    launch_text = launch.read_text()
    undefaulted = sorted(
        v for v in needed
        if re.search(rf"\b{v}=\$\{{?{v}\b", launch_text)
        and not re.search(rf"^{v}=\$\{{{v}:[-?]", launch_text, re.M)
        and not re.search(rf"^{v}=\$\{{{v}:\?", launch_text, re.M))
    assert not undefaulted, (
        f"{launch.name} forwards {undefaulted} without a default or a required "
        "marker; if the caller does not export it the pod receives an empty "
        "value and fails in a way that looks like a data problem")
