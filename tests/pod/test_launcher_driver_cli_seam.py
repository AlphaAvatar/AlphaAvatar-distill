"""The string the launcher hands the driver, parsed by the driver's own parser.

C1 attempt 7 cleared the pod CPU test gate — the first attempt ever to — and then
died at `$0.4231` because `driver_command` emitted `--stage all` and
`autoinit_c1_driver.build_parser()` has no such option. argparse exited 2 before a
line of the driver ran.

**No `$0` gate could have caught it, and that is the point of this file.** Every
pre-provider gate checks the LAUNCHER's inputs: its own argument namespace, the
harness digest, the authorization, the bundle, the staged view, the readiness
record. Nothing parsed the string the launcher hands the DRIVER with the driver's
parser. It is the device-canary failure one level out — that one produced
`missing_arguments(args)` for the runner's namespace, and this seam was left
without the equivalent.

Two rules make this a real check rather than a restatement:

* **the driver's parser is the authority.** This file never lists the driver's
  allowed options. It hands over the tokens and lets `parse_args` rule.
* **it is not a new gate.** Gate 12 binds the complete launch-bound sweep to the
  final executable tree, so a passing sweep containing this test IS the `$0`
  pre-provider evidence for the seam. The preregistered gate count stays at 12.
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests/pod"))

from session_specs import load_session_launcher, session_args  # noqa: E402

DRIVER_PATH = REPO / "scripts/pod/autoinit_c1_driver.py"


def driver_module():
    """The REAL driver module, loaded from the file the launcher names."""
    spec = importlib.util.spec_from_file_location("c1_driver_under_test", DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: What the launcher's context and plan supply. Checked against the parsed values,
#: so a silently reordered or dropped flag cannot pass.
IMAGE_DIGEST = "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.126.09"
PRICE = 1.09
SPENT = 0.3512
SOFT_STOP = 14.7841
HARD_CAP = 15.1475


@pytest.fixture(scope="module")
def command() -> str:
    """The real `driver_command`, from the real launcher, on a realistic ctx."""
    mod = load_session_launcher("autoinit_c1_launch")
    args = session_args(mod)
    ctx = types.SimpleNamespace(
        image_digest=IMAGE_DIGEST,
        price=PRICE,
        spent_usd=SPENT,
        args=args,
        auth=types.SimpleNamespace(hard_cap_usd=HARD_CAP),
        evidence={},
        scr=Path("/tmp/c1seam"),
    )
    plan = types.SimpleNamespace(soft_stop_usd=SOFT_STOP)
    return mod.driver_command(ctx, plan)


def test_the_generated_command_parses_under_the_drivers_own_parser(command):
    """THE attempt-7 failure, as a test. The parser rules; this file does not."""
    tokens = shlex.split(command)

    assert tokens[0] == "/opt/train/bin/python", tokens[0]
    assert tokens[1].endswith("scripts/pod/autoinit_c1_driver.py"), tokens[1]
    assert Path(tokens[1]).name == DRIVER_PATH.name

    parser = driver_module().build_parser()
    # `parse_known_args`, so an unknown flag is REPORTED rather than exiting the
    # test process — `parse_args` would raise SystemExit(2) and say nothing about
    # which token was wrong, which is exactly what attempt 7 saw.
    parsed, unknown = parser.parse_known_args(tokens[2:])
    assert unknown == [], (
        f"the launcher passes arguments the driver's parser does not accept: "
        f"{unknown}. That is the attempt-7 abort, at $0.")

    assert parsed.image_digest == IMAGE_DIGEST
    assert parsed.rate == pytest.approx(PRICE)
    assert parsed.spent_usd == pytest.approx(SPENT)
    assert parsed.soft_stop_usd == pytest.approx(SOFT_STOP)
    assert parsed.authorized_usd == pytest.approx(HARD_CAP)


def test_the_command_carries_no_stage_flag(command):
    """`run()` executes the whole fixed sequence B → C → DE → F → G → H → I.

    A `--stage` would be a stage-selection surface, and partial C1 execution is
    not an authorized control — so the flag is removed from the launcher rather
    than accepted by the driver.
    """
    assert "--stage" not in command, command


def test_the_driver_parser_still_refuses_stage(command):
    """Mutation: re-inject the exact attempt-7 argv and require SystemExit(2).

    If this ever stops raising, someone has added `--stage` to the driver — which
    is the repair this postmortem explicitly rejected.
    """
    tokens = shlex.split(command)
    # Exactly attempt 7's argv: `--stage all` first, before --image-digest.
    poisoned = ["--stage", "all"] + tokens[2:]
    parser = driver_module().build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(poisoned)
    assert exc.value.code == 2

    _parsed, unknown = parser.parse_known_args(poisoned)
    assert unknown == ["--stage", "all"], unknown


def test_the_authority_is_the_drivers_parser_not_a_copy_of_its_options():
    """A test that hardcodes the allowed flags drifts exactly like the bug did.

    Checked by MECHANISM rather than by scanning for option strings: this file
    mentions some of them in prose, and a scan that trips on its own commentary
    is noise rather than a guard.
    """
    src = Path(__file__).read_text()
    assert "build_parser()" in src and "parse_known_args" in src, (
        "acceptance must be decided by the driver's own parser")
    # No literal allow-list anywhere: the only place option names may appear as
    # data is the poisoned argv, which is the REJECTED one.
    import ast
    tree = ast.parse(src)
    prefix = "-" * 2                       # not a literal this scan can match
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.startswith(prefix) and len(n.value) > len(prefix)}
    assert literals <= {"--stage"}, (
        f"this file names driver options as data: {sorted(literals - {'--stage'})}. "
        "The parser is the authority; hand it the tokens instead.")


def test_the_seam_is_covered_without_a_thirteenth_gate():
    """Gate 12 binds the launch-bound sweep to the final tree, so a sweep that
    contains this module IS the $0 pre-provider evidence for the seam."""
    mod = load_session_launcher("autoinit_c1_launch")
    spec = mod.spec(session_args(mod))
    assert len(spec.precheck) == 12, (
        f"the preregistered pre-provider gate count moved to {len(spec.precheck)}")
    assert str(Path(__file__).relative_to(REPO)) not in spec.setup.test_ignores
