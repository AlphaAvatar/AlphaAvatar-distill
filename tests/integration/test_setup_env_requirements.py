"""Every variable the SHELL refuses to run without, asked of every session.

The shared setup script states its own requirements in the only form that
cannot drift from the code that uses them:

    : "${SESSION_FROZEN_EXPECT:?a session declaring ASSETS_READY must name ...}"

An unset variable there is `exit 1` on a billing pod, after the repository has
been cloned, both virtualenvs built and the teacher fetched. A3 declared
`ASSETS_READY` and did not set `SESSION_FROZEN_EXPECT`, and the acquisition
loop discovered that **twenty-one times**, one pod each, `$2.74` in total --
because every `$0` check in the repository looked at the MARKERS a session
declares and none looked at the VARIABLES the script needs in order to reach
them.

So this extracts the requirements from the shell and asks each launcher's
`SetupManifest` whether the environment it actually exports satisfies them.
Parsed rather than listed: a requirement added to the script tomorrow is
checked tomorrow, which is the difference between this and the constant it
would otherwise be.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))

from support.session_specs import SESSION_LAUNCHERS, load_session_launcher, session_args  # noqa: E402

SETUP = REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"

#: Launchers that PREDATE a requirement, named with the reason rather than
#: silently excluded. `: "${SESSION_FROZEN_EXPECT:?}"` landed on 2026-09-16
#: ("Setup follows the session's declaration, and C2 asks its own frozen
#: question"); every session here completed before that and none is scheduled
#: to run again. They are recorded because "this check does not apply to them"
#: is a fact worth stating, and because the test below requires the set to
#: SHRINK: a launcher written after the requirement may not join it.
PREDATE_THE_REQUIREMENT: dict[str, str] = {
    "autoinit_preflight_launch": "the micro-preflight, complete",
    "autoinit_phase_a_launch": "Phase A, COMPLETE and FROZEN",
    "autoinit_continuation_launch": "the Stage-3 continuation, complete",
    "autoinit_device_canary_launch": "the device canary, complete",
    "autoinit_measurement_launch": "the bounded runtime measurement, complete",
    "autoinit_recovery_continuation_launch":
        "the recovery continuation, complete",
}

#: `: "${VAR:?message}"` — the shell's own way of saying "required".
_REQUIRED = re.compile(r':\s*"\$\{([A-Z_][A-Z0-9_]*):\?([^}]*)\}"')

#: `if step_declared NAME; then` — the condition a requirement may sit under.
_STEP = re.compile(r"^\s*if\s+step_declared\s+([A-Z_][A-Z0-9_]*)\s*;\s*then")


def requirements() -> list[tuple[str, str | None, str]]:
    """`(variable, gating step or None, message)` for the whole script.

    The gating step is tracked by walking the file and remembering the most
    recent `step_declared` guard whose block has not closed. Approximate on
    purpose: a requirement attributed to a step it does not really sit under
    makes this check STRICTER, never weaker, and the unconditional ones are
    the ones every session owes regardless.
    """
    out: list[tuple[str, str | None, str]] = []
    stack: list[tuple[str, int]] = []
    for line in SETUP.read_text().splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        while stack and stripped.startswith("fi") and indent <= stack[-1][1]:
            stack.pop()
        m = _STEP.match(line)
        if m:
            stack.append((m.group(1), indent))
            continue
        r = _REQUIRED.search(line)
        if r:
            out.append((r.group(1), stack[-1][0] if stack else None,
                        r.group(2)))
    return out


def test_the_script_states_requirements_this_way_at_all():
    """If the form changed, every case below would vacuously pass."""
    reqs = requirements()
    assert len(reqs) >= 3, (
        f"only {len(reqs)} `${{VAR:?}}` requirements found in "
        f"{SETUP.name}; the extraction no longer matches the script")
    names = {v for v, _, _ in reqs}
    assert "SESSION_FROZEN_EXPECT" in names, (
        "the frozen-asset requirement is no longer stated as a shell "
        "refusal; this check was written because that one cost 21 pods")


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_every_variable_the_shell_requires_is_one_the_session_exports(
        name, extra):
    """The session's REAL exported environment, from its real spec.

    `setup_environment` is the same production method `SessionRunner._launch`
    calls, so what is checked here is what the pod receives -- not the
    `env` dict the launcher happens to pass, which omits everything the
    runner injects.
    """
    mod = load_session_launcher(name)
    args = session_args(mod, extra)
    spec = mod.spec(args)
    env = spec.setup_environment(session_commit="0" * 40,
                                 bundle="aad_test.bundle")
    declared = set(spec.setup.setup_markers or ())

    missing = []
    for var, step, message in requirements():
        if step is not None and step not in declared:
            continue            # the branch that needs it never runs
        if not str(env.get(var, "")).strip():
            missing.append((var, step, message[:90]))

    if name in PREDATE_THE_REQUIREMENT:
        #: Declared, not skipped: the expectation is stated and checked, so a
        #: session that quietly started satisfying it would be noticed and
        #: removed from the list rather than left there forever.
        assert missing, (
            f"{name} is listed as predating the requirement and now satisfies "
            f"it; remove it from PREDATE_THE_REQUIREMENT ({PREDATE_THE_REQUIREMENT[name]})")
        return

    assert not missing, (
        f"{name} declares the steps that need {[m[0] for m in missing]} and "
        f"its environment does not set them: {missing}. The shell exits 1 on "
        f"a billing pod, after the clone, both virtualenvs and the teacher "
        f"fetch.")


def test_the_predating_list_only_shrinks():
    """A launcher written after 2026-09-16 may not join it.

    The exemption is for sessions that completed before the requirement
    existed. Adding a NEW launcher to it would convert a guard into a habit,
    which is how A3 came to launch twenty-one times without the variable.
    """
    import subprocess

    for launcher in PREDATE_THE_REQUIREMENT:
        path = f"scripts/pod/{launcher}.py"
        first = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%ad", "--date=short",
             "--", path],
            cwd=REPO, capture_output=True, text=True).stdout.strip()
        assert first, f"{path} has no add commit; is it tracked?"
        added = first.splitlines()[-1]
        assert added < "2026-09-16", (
            f"{launcher} was added on {added}, after the requirement landed on "
            f"2026-09-16; it may not be exempt from it")
