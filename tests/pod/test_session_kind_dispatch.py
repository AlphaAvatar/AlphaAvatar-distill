"""Every launchable `SESSION_KIND` must have the branch its launcher expects.

Phase-B attempt 2 died here for `$0.2300`. `autoinit_preflight_setup.sh` picks the
authorization type by `SESSION_KIND`; it had branches for `phase_a`,
`recovery_continuation` and a `spend` default. The Phase-B launcher declares
`SESSION_KIND=phase_b`, which matched nothing, so a `PhaseBAuthorization` was
loaded by `SpendAuthorization.load` and raised `KeyError: 'preflight_plan_hash'`
one step after the pod's test suite passed.

The dispatch is bash that only ever runs on a pod, which is why no Python test saw
it. So this file asserts the **mapping**, not the presence of a string: it builds
each launcher's real `SessionSpec`, reads the `SESSION_KIND` it will export and
the `authorization_loader` it will use, parses the branches out of the shell
script, and requires the two to name the *same class*. Grepping for `"phase_b"`
would pass against a branch that loads the wrong type — which is the failure this
exists to prevent, since routing Phase B through `SpendAuthorization` would fall
back to `HARNESS_SOURCE_FILES_V1` and bind Phase B to Phase A's file list while
reporting success.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"

#: Enough to construct a `SessionSpec`; no pod, no provider, no network.
BASE_ARGS = ["--scr", "/tmp/does-not-matter", "--session-commit", "d" * 40,
             "--bundle", "b.bundle"]
#: Some launchers require an extra required flag to parse at all.
ARG_VARIANTS = ([], ["--transport", "relay"])

DEFAULT_KIND = "spend"
#: Kinds that must NOT share the generic loader. Each names a distinct
#: authorization type whose ceiling prices a distinct amount of work.
DEDICATED_KINDS = ("phase_a", "phase_b", "recovery_continuation", "continuation_b")
BRANCH_RE = re.compile(r'^(?:el)?if \[ "\$SESSION_KIND" = "([a-z_]+)" \]; then$',
                       re.MULTILINE)


def dispatch_region() -> str:
    text = SETUP.read_text()
    start = text.index('SESSION_KIND="${SESSION_KIND:-')
    end = text.index("mark AUTHORIZATION_OK", start)
    return text[start:end]


def code_only(body: str) -> str:
    """Drop comment lines.

    The contract is about what executes. A branch may *explain* why it does not
    use `SpendAuthorization`, and a naive substring check would read that
    explanation as the thing it forbids.
    """
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#"))


def branches() -> dict[str, str]:
    """`SESSION_KIND` -> the shell body that handles it, `else` included."""
    region = dispatch_region()
    marks = [(m.group(1), m.start(), m.end()) for m in BRANCH_RE.finditer(region)]
    assert marks, "the dispatcher has no explicit SESSION_KIND branches at all"
    out: dict[str, str] = {}
    for i, (kind, _, body_start) in enumerate(marks):
        body_end = marks[i + 1][1] if i + 1 < len(marks) else region.index("\nelse\n")
        out[kind] = region[body_start:body_end]
    out[DEFAULT_KIND] = region[region.index("\nelse\n"):]
    return out


def default_kind_from_script() -> str:
    m = re.search(r'SESSION_KIND="\$\{SESSION_KIND:-([a-z_]+)\}"', SETUP.read_text())
    assert m, "the dispatcher no longer declares a default SESSION_KIND"
    return m.group(1)


def launchers() -> dict[str, tuple[str, str]]:
    """launcher module -> (SESSION_KIND it exports, authorization class name)."""
    found = {}
    for path in sorted((REPO / "scripts/pod").glob("*_launch.py")):
        mod_name = path.stem
        try:
            mod = __import__(mod_name)
        except BaseException:                     # noqa: BLE001 - not a launcher we can build
            continue
        if not (hasattr(mod, "spec") and hasattr(mod, "build_parser")):
            continue
        args = None
        for extra in ARG_VARIANTS:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    args = mod.build_parser().parse_args(BASE_ARGS + extra)
                break
            except SystemExit:
                continue
        if args is None:
            continue
        session = mod.spec(args)
        loader = session.authorization_loader
        cls = getattr(loader, "__self__", None)
        found[mod_name] = (session.setup.env.get("SESSION_KIND", DEFAULT_KIND),
                           getattr(cls, "__name__", repr(loader)))
    assert found, "no launcher could be constructed; the probe itself is broken"
    return found


LAUNCHERS = launchers()
BRANCHES = branches()


def test_the_probe_sees_the_sessions_this_repository_can_actually_launch():
    """Guards the guard: a silently empty enumeration would make everything pass."""
    kinds = {k for k, _ in LAUNCHERS.values()}
    assert kinds >= {*DEDICATED_KINDS, DEFAULT_KIND}
    assert len(LAUNCHERS) >= 7, sorted(LAUNCHERS)


def test_the_scripts_default_is_the_narrow_spend_path():
    assert default_kind_from_script() == DEFAULT_KIND
    assert DEFAULT_KIND in BRANCHES
    assert "SpendAuthorization" in code_only(BRANCHES[DEFAULT_KIND])


@pytest.mark.parametrize("module", sorted(LAUNCHERS))
def test_every_launchable_kind_has_a_branch_that_loads_ITS_OWN_type(module):
    kind, loader_class = LAUNCHERS[module]
    assert kind in BRANCHES, (
        f"{module} exports SESSION_KIND={kind!r}, which no branch in "
        f"{SETUP.name} handles. It would fall through to the {DEFAULT_KIND} path "
        f"and be loaded by SpendAuthorization — the $0.2300 attempt-2 failure.")
    body = code_only(BRANCHES[kind])
    assert f"{loader_class}.load(" in body, (
        f"{module} loads its authorization with {loader_class}, but the {kind!r} "
        f"branch does not. A branch that exists and reads the artifact through "
        f"the wrong type is worse than a missing one: it can succeed.")


@pytest.mark.parametrize("kind", DEDICATED_KINDS)
def test_each_dedicated_branch_binds_the_plan_and_fails_closed(kind):
    body = code_only(BRANCHES[kind])
    assert "require_plan(os.environ['SESSION_PLAN_HASH'])" in body, kind
    assert "automatic_followon_start is False" in body, kind
    assert 'mark "AUTHORIZATION_MISMATCH"; exit 98' in body, (
        f"the {kind!r} branch does not fail closed; a python failure there would "
        "be ignored and the driver would start unauthorized")
    assert "SpendAuthorization" not in body, (
        f"the {kind!r} branch reaches for the generic loader, whose "
        "harness_source_files falls back to Phase A's file list")


def test_the_dedicated_branches_do_not_share_a_type():
    dedicated = {k: code_only(BRANCHES[k]) for k in DEDICATED_KINDS}
    classes = {k: re.findall(r"(\w+Authorization)\.load\(", v)
               for k, v in dedicated.items()}
    flat = [c for v in classes.values() for c in v]
    assert len(flat) == len(set(flat)) == len(DEDICATED_KINDS), classes


def test_the_phase_b_branch_asserts_the_phase_b_permission_contract():
    body = code_only(BRANCHES["phase_b"])
    assert "PhaseBAuthorization" in body
    assert "allows_phase_b is True" in body
    assert "allows_phase_a is False" in body
    # The two fields whose absence produced the KeyError. Naming them here keeps
    # the reason the branch exists attached to the branch.
    assert "expected_usd" not in body and "preflight_plan_hash" not in body


def test_the_continuation_branch_asserts_it_cannot_repurchase_the_search():
    """The one contract that distinguishes this session from a full Phase B.

    The continuation's ceiling prices one missing `sb` and at most two conditional
    `sc`. The full Phase-B artifact prices those *plus* a 16.5 h P=2 search that
    attempt 5 already bought. Loading either through the other's branch would be a
    silent re-purchase, so the branch states the permission explicitly rather than
    inferring it from the schema check inside `load`.
    """
    body = code_only(BRANCHES["continuation_b"])
    assert "ContinuationAuthorization" in body
    assert "a.runs_search is False" in body, (
        "the branch does not assert the property that makes this session cheap; "
        "runs_search is False by type, and the branch is where that becomes a "
        "checked precondition rather than an assumption")
    assert "allows_phase_b is True" in body
    assert "PhaseBAuthorization" not in body, (
        "the continuation branch reaches for the full Phase-B loader, whose grant "
        "authorizes the search this session must not run")
