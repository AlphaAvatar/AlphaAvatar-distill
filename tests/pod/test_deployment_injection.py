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

from aadistill.infrastructure import session as SESSION  # noqa: E402
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
    #: Not a skip. Every launcher this repository can run is enumerated by
    #: `all_specs`, so a missing C1 spec is a broken enumeration, and a skip
    #: here would hide it -- as well as adding an unregistered skip predicate
    #: to a suite whose pod/dev-box parity is audited.
    raise AssertionError(
        "all_specs() enumerates no autoinit_c1_launch spec; the launcher "
        "enumeration is broken, which is a failure and not a skip")


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


# --- the canonical binding, and what makes two deployments different -------

class TestTheCanonicalDeploymentBinding:
    """The session record must be able to tell two deployments apart.

    `ExecutionCommands.as_dict()` serialized four of its seven
    behaviour-affecting fields, and `SessionSpec.as_dict()` did not call it at
    all -- so the artifact a later reader reproduces a run from recorded
    neither the workspace, the checkout root nor the host CUDA floor. Two
    layouts that share three script paths were byte-identical in the record.
    """

    def test_every_behaviour_affecting_field_is_in_the_binding(self):
        r, spec = runner_for(LAYOUT_A)
        got = spec.commands.as_dict()
        for f in ("workspace_root", "checkout_root", "remote_python",
                  "min_cuda_version", "watchdog", "setup_script",
                  "artifact_collector"):
            assert f in got, f
        assert "provider_cli_candidates" in got

    def test_the_declared_field_list_matches_what_is_serialized(self):
        """A field added to the type but not to `BEHAVIOUR_FIELDS` would drop
        out of the identity silently, which is the defect being fixed."""
        from dataclasses import fields as dc_fields
        _, spec = runner_for(LAYOUT_A)
        declared = {f.name for f in dc_fields(spec.commands)}
        covered = set(spec.commands.BEHAVIOUR_FIELDS) | {"provider_cli_candidates"}
        assert declared == covered, declared ^ covered

    def test_two_deployments_receive_different_identities(self):
        a = runner_for(LAYOUT_A)[1].commands.binding().digest
        b = runner_for(LAYOUT_B)[1].commands.binding().digest
        assert a != b

    @pytest.mark.parametrize(
        "field_name,other",
        [("workspace_root", "/srv/run"),
         ("checkout_root", "/srv/run/checkout"),
         ("remote_python", "/usr/local/venv/bin/python3"),
         ("min_cuda_version", "12.4")])
    def test_each_field_alone_changes_the_identity(self, field_name, other):
        """One at a time. Four fields differing together could hide three that
        do not participate."""
        base = runner_for(LAYOUT_A)[1].commands
        assert base.binding().digest != replace(
            base, **{field_name: other}).binding().digest

    def test_a_null_cuda_floor_is_distinguishable_from_a_declared_one(self):
        """`None` is a decision -- this runtime imposes no floor -- and must not
        collide with any version string."""
        base = runner_for(LAYOUT_A)[1].commands
        assert (replace(base, min_cuda_version=None).binding().digest
                != base.binding().digest)

    def test_the_resolved_provider_cli_participates(self):
        """Which binary actually existed on the launching machine is knowable
        only at runtime, so it is supplied rather than stored."""
        c = runner_for(LAYOUT_A)[1].commands
        assert (c.binding(selected_provider_cli="/usr/bin/runpodctl").digest
                != c.binding(selected_provider_cli="/home/u/bin/runpodctl").digest)

    def test_the_binding_reaches_the_session_record(self):
        rec = runner_for(LAYOUT_B)[1].as_dict(selected_provider_cli="/usr/bin/x")
        assert rec["deployment"]["workspace_root"] == "/srv/run"
        assert rec["deployment"]["remote_python"] == "/usr/local/venv/bin/python3"
        assert rec["deployment"]["min_cuda_version"] == "12.4"
        assert rec["deployment"]["selected_provider_cli"] == "/usr/bin/x"
        assert len(rec["deployment_digest"]) == 64

    def test_the_runner_records_the_cli_it_actually_resolved(self):
        src = (REPO_ROOT / "src/aadistill/infrastructure/session_runner.py").read_text()
        assert "spec.as_dict(selected_provider_cli=self.cli)" in src


class TestMissingDeploymentDataFailsClosed:
    """Not silently defaulted. A blank interpreter does not fail where it is
    missing; it fails inside a remote command on a pod that is already billing.
    """

    @pytest.mark.parametrize("field_name", ["remote_python", "workspace_root",
                                            "checkout_root", "watchdog",
                                            "setup_script", "artifact_collector"])
    def test_a_blank_required_field_is_refused(self, field_name):
        base = runner_for(LAYOUT_A)[1].commands
        with pytest.raises(SESSION.SessionSpecError, match=field_name):
            replace(base, **{field_name: ""}).validate()

    def test_the_refusal_happens_during_spec_validation(self):
        """Before pricing, before the provider is contacted, before a pod."""
        _, spec = runner_for(LAYOUT_A)
        broken = replace(spec, commands=replace(spec.commands, remote_python=""))
        with pytest.raises(SESSION.SessionSpecError, match="remote_python"):
            broken.validate()

    def test_the_interpreter_has_no_default_at_all(self):
        """It defaulted to one image's build path, so every session that never
        mentioned an interpreter silently claimed that one."""
        with pytest.raises(TypeError, match="remote_python"):
            SESSION.ExecutionCommands(watchdog="w", setup_script="s",
                                 artifact_collector="c",
                                 workspace_root="/ws", checkout_root="/ws/r",
                                 min_cuda_version=None)

    def test_a_null_cuda_floor_is_still_accepted_when_stated(self):
        SESSION.ExecutionCommands(watchdog="w", setup_script="s",
                             artifact_collector="c", remote_python="/p",
                             workspace_root="/ws", checkout_root="/ws/r",
                             min_cuda_version=None).validate()


def test_no_concrete_image_interpreter_remains_in_core():
    """Anywhere in `src/aadistill`, not just in the runner."""
    core = REPO_ROOT / "src/aadistill"
    for p in sorted(core.rglob("*.py")):
        assert "/opt/train/bin/python" not in p.read_text(), p


def test_neither_layout_relies_on_a_module_global():
    """Both are driven through the real construction path, and the module holds
    no workspace, checkout or interpreter constant for either to fall back on."""
    for mod in (SR, SESSION):
        for name, value in vars(mod).items():
            if name.isupper() and isinstance(value, str):
                assert "/workspace" not in value, f"{mod.__name__}.{name}"
                assert "/opt/train" not in value, f"{mod.__name__}.{name}"


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
