"""Every session's namespace must satisfy the machinery that runs it.

Device-canary attempt 1 was lost at $0.0603 to
`AttributeError: 'Namespace' object has no attribute 'teacher_revision'`, raised
between "ssh reachable" and "running setup". The canary script never ran.

Subclassing a launcher inherited its **argument contract** as well as its
methods, and that contract was invisible from the subclass: nothing in the
wrapper mentioned `teacher_revision`, and nothing had to. The six $0 checks that
ran before launch — `make_plan`, `relay_precheck`, `driver_command`,
`event_streams`, `fetch_products`, the artifact spec — all passed, because the
attribute is read by the setup step, which needs a live host.

The subclassing is gone (2026-08-18), and with it the invisible half of the
contract: `session.RUNNER_ARGUMENT_CONTRACT` names what the runner reads, and
`SessionRunner.__init__` refuses a namespace missing any of it before a pod can
exist. This file is what keeps that honest, and it checks **every** session
rather than only the one that paid to find the problem:

* the declaration matches what the runner's own syntax tree reads;
* every launcher's REAL parser produces a namespace that satisfies it;
* the canary is still a canary — it fetches nothing, waits on nothing, and
  cannot authorize Phase A.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from session_specs import (
    SESSION_LAUNCHERS, all_specs, load_session_launcher, session_args,
)

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "src/aadistill/infrastructure/session_runner.py"
CANARY = "autoinit_device_canary_launch"


def attributes_read_off_the_namespace(source: Path) -> set[str]:
    """Every `self.a.<name>` and `args.<name>` in a module, from the AST.

    Not a regex over the text: a regex would also match the string `self.a.x`
    inside a docstring, and would miss nothing only by luck.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if (isinstance(base, ast.Attribute) and base.attr == "a"
                and isinstance(base.value, ast.Name) and base.value.id == "self"):
            found.add(node.attr)
        elif isinstance(base, ast.Name) and base.id == "args":
            found.add(node.attr)
    return found


def test_the_declared_contract_is_what_the_runner_actually_reads():
    """The declaration is only useful if it cannot drift from the code.

    `RUNNER_ARGUMENT_CONTRACT` exists so a launcher can be checked without a
    host. If the runner grew a read the declaration does not name, every such
    check would pass while the pod died on the new attribute — which is attempt 1
    again, with an extra layer of paperwork.
    """
    from aadistill.infrastructure.session import RUNNER_ARGUMENT_CONTRACT

    read = attributes_read_off_the_namespace(RUNNER)
    declared = set(RUNNER_ARGUMENT_CONTRACT)
    undeclared = sorted(read - declared)
    assert not undeclared, (
        f"the runner reads {undeclared} off the argument namespace and "
        "RUNNER_ARGUMENT_CONTRACT does not declare them; every launcher would "
        "pass its contract check and the pod would still die on them")
    assert len(declared) >= 18, (
        f"only {len(declared)} attributes declared; the contract has shrunk and "
        "would now under-report what a session must supply")
    unread = sorted(declared - read)
    assert not unread, (
        f"the contract declares {unread}, which the runner never reads. A "
        "contract with surplus entries makes every launcher carry an argument "
        "for nothing, which is how the canary came to declare a teacher")


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_every_session_namespace_carries_what_the_runner_reads(name, extra):
    """Checked from the REAL parser, for every session, not just the canary.

    Working out which code path reads which attribute is exactly the reasoning
    that produced attempt 1: `fetch_products` was overridden, so `ckpt_store`
    looked unreachable, and `teacher_revision` was not thought about at all. The
    whole contract is checked, for all four sessions, and it costs nothing.
    """
    from aadistill.infrastructure.session import missing_arguments

    mod = load_session_launcher(name)
    args = session_args(mod, extra)
    missing = missing_arguments(args)
    assert not missing, (
        f"{name}'s namespace is missing {missing}, which the runner reads. "
        "Attempt 1 died on exactly this, after the pod was created and billing.")


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_the_runner_refuses_a_namespace_missing_an_argument(name, extra, tmp_path):
    """The check runs before a pod exists, and it is the runner that runs it.

    A test-only check would be a second transcription of the contract. This
    exercises the refusal `SessionRunner.__init__` performs, which is the one
    that stands between a bad namespace and a billing pod.
    """
    from aadistill.autoinit.authorization import AuthorizationError
    from aadistill.infrastructure.session_runner import SessionRunner

    mod = load_session_launcher(name)
    args = session_args(mod, extra, scr=str(tmp_path))
    delattr(args, "poll_seconds")
    with pytest.raises(AuthorizationError, match="poll_seconds"):
        SessionRunner(mod.spec(args), args, REPO)


def test_the_canary_pins_the_same_teacher_revision_as_every_other_session():
    """Present is necessary; *appropriate* is the other half.

    `TEACHER_REVISION` is forwarded to the SHARED setup script. The canary needs
    no teacher, but setup's behaviour must not change because a canary is driving
    it — so the value must be the frozen one every other session pins, not a
    canary-specific stand-in.
    """
    revisions = {name: spec.setup.teacher_revision
                 for name, _mod, _args, spec in all_specs()}
    assert len(set(revisions.values())) == 1, (
        f"sessions pin different teacher revisions: {revisions}. The shared "
        "setup would behave differently depending on which session drove it")
    assert revisions[CANARY] == "768f209d9ea81521153ed38c47d515654e938aea"


def test_the_canary_declares_no_local_assets_and_that_is_now_honoured():
    """The retry's $0.0637: a correct declaration the setup script ignored.

    The canary reads the calibration mixture and the canonical student, both from
    the relay. It wants no dev-box asset, said so, and the shared setup copied
    two anyway out of a directory the launcher had therefore left empty.
    """
    specs = {name: spec for name, _m, _a, spec in all_specs()}
    canary = specs[CANARY]
    assert canary.setup.local_assets == (), (
        "the canary has acquired a local asset; it needs none")
    env = canary.setup_environment(session_commit="0" * 40, bundle="b.bundle")
    assert env["SESSION_ASSETS"] == "", (
        "the canary would tell setup to install something")
    # And the other sessions still declare theirs, so an empty value means
    # "none declared" rather than "the mechanism is broken".
    assert specs["autoinit_preflight_launch"].setup.local_assets, (
        "no session declares a local asset; SESSION_ASSETS is empty for everyone "
        "and this test proves nothing")


def test_the_canary_still_fetches_nothing_and_cannot_authorize_phase_a():
    """The two properties the session must not have quietly changed."""
    from aadistill.autoinit.authorization import SpendAuthorization
    from aadistill.infrastructure.session import SessionContext

    specs = {name: spec for name, _m, _a, spec in all_specs()}
    canary = specs[CANARY]
    ctx = SessionContext(scr=Path("/tmp"), args=None, auth=None, evidence={},
                         say=lambda m: None, stage2_passed=True)
    assert canary.artifacts.fetch_products(ctx) == []
    assert canary.artifacts.event_streams(ctx) == ()
    #: The type is the guarantee. `SpendAuthorization.allows_phase_a` is a hard
    #: False, so naming it makes "this session cannot start Phase A" a property
    #: of the declaration rather than a promise about the code.
    # `is` would compare two freshly-bound method objects and always fail;
    # `__self__` is the class the loader is bound to, which is the fact.
    assert canary.authorization_loader.__self__ is SpendAuthorization

    auth = SpendAuthorization.load(
        REPO / "logs/autoinit_device_canary_authorization.json")
    assert auth.allows_phase_a is False, (
        "a canary grant must never permit Phase A")


def test_the_canary_session_path_is_recorded_as_terminated():
    """Converting the launcher must not quietly prepare a run of it."""
    specs = {name: spec for name, _m, _a, spec in all_specs()}
    assert "TERMINATED" in specs[CANARY].evidence_fields["session_path_status"]
