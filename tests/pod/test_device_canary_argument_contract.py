"""The canary launcher's namespace must satisfy the base it subclasses.

Device-canary attempt 1 was lost at $0.0603 to
`AttributeError: 'Namespace' object has no attribute 'teacher_revision'`, raised
between "ssh reachable" and "running setup". The canary script never ran.

Subclassing a launcher inherits its **argument contract** as well as its
methods, and that contract is invisible from the subclass: nothing in the
wrapper mentions `teacher_revision`, and nothing had to. The six $0 checks that
ran before launch — `make_plan`, `relay_precheck`, `driver_command`,
`event_streams`, `fetch_products`, the artifact spec — all passed, because the
attribute is read by `run_setup`, which needs a live host.

This closes it with the two things that do not need a host: the **real** parser,
and the attribute set read off `self.a` anywhere in the base, discovered from the
base's own syntax tree rather than transcribed.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

BASE = REPO / "scripts/pod/autoinit_preflight_launch.py"
CANARY = REPO / "scripts/pod/autoinit_device_canary_launch.py"


def attributes_read_off_self_a(source: Path) -> set[str]:
    """Every `self.a.<name>` in a module, from the AST.

    Not a regex over the text: a regex would also match the string `self.a.x`
    inside a docstring, and would miss nothing only by luck.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "a"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"):
            found.add(node.attr)
    return found


def canary_namespace():
    """The namespace the REAL parser produces for a real invocation."""
    spec = importlib.util.spec_from_file_location("device_canary_launch", CANARY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["device_canary_launch"] = mod
    spec.loader.exec_module(mod)
    return mod, mod.build_parser().parse_args(
        ["--scr", "/tmp/unused", "--session-commit", "0" * 40,
         "--bundle", "unused.bundle"])


def test_the_canary_namespace_carries_every_attribute_the_base_reads():
    """Deliberately the FULL set, not just the methods the canary reaches.

    Working out which inherited methods run on which path is exactly the
    reasoning that produced attempt 1: `fetch_products` is overridden, so
    `ckpt_store` looked unreachable, and `teacher_revision` was not thought about
    at all. A superset cannot under-cover, and the three surplus attributes cost
    one line each.
    """
    _, args = canary_namespace()
    needed = attributes_read_off_self_a(BASE)
    assert len(needed) >= 21, (
        f"only {len(needed)} attributes found in the base; the AST walk is not "
        "seeing what it used to and would now under-report the contract")

    missing = sorted(a for a in needed if not hasattr(args, a))
    assert not missing, (
        f"the canary launcher's namespace is missing {missing}, which the "
        f"inherited base reads off self.a. Attempt 1 died on exactly this, "
        f"after the pod was created and billing.")


def test_the_three_attributes_attempt_1_lacked_are_present_and_canary_shaped():
    """Present is necessary; *appropriate* is the other half.

    A canary that quietly acquired a real checkpoint store, or a long fetch
    timeout for a fetch it never performs, would satisfy the test above while
    making the session something other than a canary.
    """
    _, args = canary_namespace()

    # Forwarded to the SHARED setup script, so it must be the same frozen
    # revision every other session pins: setup's behaviour must not change
    # because a canary is driving it.
    assert args.teacher_revision == "768f209d9ea81521153ed38c47d515654e938aea"

    # Read only by the base's `fetch_products`, which this session overrides to
    # return nothing. Canary-scoped so a future edit could not land real
    # checkpoints there.
    assert "device_canary" in args.ckpt_store
    assert "aad-artifacts" not in args.ckpt_store, (
        "the canary points at the real checkpoint store; it fetches nothing and "
        "must not be able to write there")
    assert args.ckpt_fetch_limit_min == 1


def test_the_canary_still_fetches_nothing_and_cannot_authorize_phase_a():
    """The two properties the extra arguments must not have quietly changed."""
    mod, _ = canary_namespace()
    session = mod.DeviceCanary.__new__(mod.DeviceCanary)
    assert mod.DeviceCanary.fetch_products(session, "host", None, True) == []
    assert mod.DeviceCanary.event_streams(session) == ()

    from aadistill.autoinit.authorization import SpendAuthorization

    auth = SpendAuthorization.load(
        REPO / "logs/autoinit_device_canary_authorization.json")
    assert auth.allows_phase_a is False, (
        "a canary grant must never permit Phase A")
