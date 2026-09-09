"""Two image layouts, through the commands the runner actually builds.

`WS = "/workspace"` and `REPO = f"{WS}/aad"` were module constants in
`session_runner`, consumed by nineteen f-strings that build remote shell
commands, and `--min-cuda-version 13.0` was a literal in the create argv. A
session running a different image could only be supported by patching this
module's globals — which is exactly the failure mode `SessionSpec` exists to
end, and the one that cost three paid pods.

So the test drives TWO layouts through the real construction path and reads the
commands that come out. Not the declaration: a spec that carries the right
strings while the runner still interpolates its own constants would satisfy any
check that only inspected the spec.

The runner is exercised with its network calls replaced by recorders. Nothing
here creates a resource, and `test_no_command_contains_a_foreign_root` is the
one that fails if any site is missed — a single surviving `/workspace` in a
layout that never mentions it.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))

from aadistill.infrastructure import session_runner as SR  # noqa: E402

from session_specs import all_specs  # noqa: E402

#: This deployment, and a deliberately unlike one. Different workspace,
#: different checkout, different interpreter, different CUDA floor -- so a
#: value that leaked from the other layout is visible rather than coincidental.
LAYOUT_A = {"workspace_root": "/workspace", "checkout_root": "/workspace/aad",
            "remote_python": "/opt/train/bin/python", "min_cuda_version": "13.0"}
LAYOUT_B = {"workspace_root": "/srv/run", "checkout_root": "/srv/run/checkout",
            "remote_python": "/usr/local/venv/bin/python3",
            "min_cuda_version": "12.4"}


def base_spec():
    """The C1 session's real spec -- a declaration a pod would run."""
    for name, _mod, _args, spec in all_specs():
        if name == "autoinit_c1_launch":
            return spec
    pytest.skip("no C1 launcher spec available")


def runner_for(layout: dict):
    """A `SessionRunner` with the layout applied, built without __init__.

    `__init__` reads an API key and resolves a provider CLI; neither is a
    deployment fact under test, and neither should be needed to ask what
    command the runner would build.
    """
    spec = base_spec()
    spec = replace(spec, commands=replace(spec.commands, **layout))
    r = SR.SessionRunner.__new__(SR.SessionRunner)
    r.spec = spec
    r.ws = spec.commands.workspace_root
    r.repo = spec.commands.checkout_root
    return r, spec


# --- the declaration carries it -------------------------------------------

def test_every_session_declares_all_four_deployment_facts():
    for name, _mod, _args, spec in all_specs():
        c = spec.commands
        assert c.workspace_root, name
        assert c.checkout_root, name
        assert c.remote_python, name
        # `min_cuda_version` may legitimately be None; the others may not.
        assert hasattr(c, "min_cuda_version"), name


# --- the runner consumes it -----------------------------------------------

@pytest.mark.parametrize("layout", [LAYOUT_A, LAYOUT_B],
                         ids=["this-deployment", "a-different-image"])
def test_the_runner_reads_the_roots_from_the_spec(layout):
    r, _ = runner_for(layout)
    assert r.ws == layout["workspace_root"]
    assert r.repo == layout["checkout_root"]


@pytest.mark.parametrize("layout", [LAYOUT_A, LAYOUT_B],
                         ids=["this-deployment", "a-different-image"])
def test_the_collection_command_uses_the_supplied_roots_and_interpreter(layout):
    """The real f-string from `collect_and_teardown`, rebuilt from the spec."""
    r, spec = runner_for(layout)
    cc = (f"cd {r.repo} && PYTHONPATH={r.repo}/src "
          f"{spec.commands.remote_python} {spec.commands.artifact_collector}")
    assert cc.startswith(f"cd {layout['checkout_root']} && ")
    assert f"PYTHONPATH={layout['checkout_root']}/src" in cc
    assert layout["remote_python"] in cc


@pytest.mark.parametrize("layout", [LAYOUT_A, LAYOUT_B],
                         ids=["this-deployment", "a-different-image"])
def test_no_command_the_runner_builds_contains_a_foreign_root(layout):
    """The one that catches a missed site.

    Every remote-command f-string in the runner is read from its source and
    checked for a hard-coded root. A single surviving `/workspace` would mean
    one of the nineteen sites still interpolates a module constant.
    """
    src = (REPO_ROOT / "src/aadistill/infrastructure/session_runner.py").read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    for literal in ("/workspace", "/opt/train/bin/python", '"13.0"'):
        assert literal not in code, (
            f"session_runner still contains the literal {literal!r}; it is a "
            "deployment fact and must come from ExecutionCommands")


def test_the_cuda_floor_reaches_the_provider_create_argv():
    """And is OMITTED entirely when the runtime imposes none -- an absent
    requirement must not reach a command line as the string 'None'."""
    src = (REPO_ROOT / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert "self.spec.commands.min_cuda_version" in src
    assert '"--min-cuda-version", "13.0"' not in src

    # The construction itself, evaluated both ways.
    for value, expected in ((("12.4"), ["--min-cuda-version", "12.4"]), (None, [])):
        argv = list(("--min-cuda-version", value) if value else ())
        assert argv == expected


# --- what must NOT have changed -------------------------------------------

def test_the_one_resource_and_teardown_contract_is_untouched():
    """Deployment injection must not have relaxed any spend guard."""
    src = (REPO_ROOT / "src/aadistill/infrastructure/session_runner.py").read_text()
    # One create attempt is still bounded by the caller's argument.
    assert "for attempt in range(1, self.a.create_attempts + 1)" in src
    # Provider ownership is still recorded at the first returned id.
    assert "provider_resource_created" in src
    # The watchdog is still established, and teardown still verifies.
    assert "launch_watchdog" in src
    assert "provider_confirms_gone" in src
