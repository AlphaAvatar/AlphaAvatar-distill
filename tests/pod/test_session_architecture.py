"""The structural checks the session specification must ship with.

Three paid pods were lost to one shape of defect — a session inheriting a
requirement it never declared — and the composition design in
`docs/SESSION_ARCHITECTURE.md` is only worth the churn if the properties it
claims are checked rather than asserted. Each test below corresponds to one line
of that document's "structural checks the replacement must ship with", or to one
of the three failures:

    Phase-A attempt 1        $0.1075   SESSION_KIND leaked between two sessions
                                       sharing one setup script
    device canary attempt 1  $0.0603   the base read `self.a.teacher_revision`;
                                       the subclass had never heard of it
    device canary retry      $0.0637   the shared setup copies two assets; the
                                       subclass had declared LOCAL_ASSETS = ()

The argument contract is checked in `test_device_canary_argument_contract.py` and
the setup-environment contract in `test_launcher_forwards_setup_env.py`, because
each belongs beside the failure it prevents. What is here is everything else.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from session_specs import SESSION_LAUNCHERS, all_specs, load_session_launcher

REPO = Path(__file__).resolve().parents[2]
POD = REPO / "scripts/pod"
SETUP = POD / "autoinit_preflight_setup.sh"


def test_every_session_validates():
    """`validate()` runs before anything is priced, which is the only point at
    which refusing an incomplete declaration is free."""
    for name, _mod, _args, spec in all_specs():
        spec.validate()          # raises SessionSpecError if incomplete
        assert spec.session_id and spec.schema, name


@pytest.mark.parametrize("field,value,fragment", [
    ("status_path", "", "status_path"),
    ("run_log_path", "", "run_log_path"),
    ("driver_job_id", "", "driver_job_id"),
    ("plan_hash", "", "plan_hash"),
    ("authorization_path", "", "authorization_path"),
])
def test_validate_refuses_an_incomplete_declaration(field, value, fragment):
    """A spec missing any of these would create a pod and then fail on it."""
    from aadistill.infrastructure.session import SessionSpecError

    _name, _mod, _args, spec = all_specs()[0]
    with pytest.raises(SessionSpecError, match=fragment):
        dataclasses.replace(spec, **{field: value}).validate()


def test_validate_refuses_a_spec_that_could_never_finish_or_fail():
    from aadistill.infrastructure.session import (
        ArtifactPolicy, MarkerPolicy, SessionSpecError,
    )

    _name, _mod, _args, spec = all_specs()[0]
    with pytest.raises(SessionSpecError, match="no success marker"):
        dataclasses.replace(spec, markers=MarkerPolicy(
            success="", failure=("X",))).validate()
    with pytest.raises(SessionSpecError, match="no failure markers"):
        dataclasses.replace(spec, markers=MarkerPolicy(failure=())).validate()
    with pytest.raises(SessionSpecError, match="subset"):
        dataclasses.replace(spec, markers=MarkerPolicy(
            failure=("A",), incomplete=("B",))).validate()
    with pytest.raises(SessionSpecError, match="evidence_filename"):
        dataclasses.replace(spec, artifacts=ArtifactPolicy(
            "aud", "", "a.tar.gz", "s.json", "f.json")).validate()
    with pytest.raises(SessionSpecError, match="artifact specs"):
        dataclasses.replace(spec, artifacts=ArtifactPolicy(
            "aud", "ev.json", "a.tar.gz", "", "f.json")).validate()


def setup_installed_asset_names() -> set[str]:
    """Asset names the setup script copies out of `$WS/assets` by name.

    Empty is the correct answer, and the reason this test exists: on
    2026-08-17 the script contained two `cp -r "$WS/assets/<name>"` lines, so a
    session that declared no local asset had two copied into it anyway and died
    at $0.0637 under `set -e`.
    """
    # Comments are stripped first. The script explains the two lines it used to
    # contain, and an extractor that reads the explanation as the offence would
    # make this test impossible to satisfy honestly.
    code = "\n".join(l for l in SETUP.read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    return set(re.findall(r'\$WS/assets/([A-Za-z0-9_.-]+)"', code))


def test_the_setup_script_installs_only_what_a_session_declares():
    hardcoded = setup_installed_asset_names()
    assert not hardcoded, (
        f"autoinit_preflight_setup.sh names {sorted(hardcoded)} itself. A "
        "session that declares no local asset would have them installed anyway, "
        "which is the $0.0637 the device-canary retry paid.")
    assert "SESSION_ASSETS" in SETUP.read_text(), (
        "the setup script no longer reads the session's asset declaration")


def test_every_declared_asset_is_a_directory_the_launcher_can_actually_send():
    """A declaration the launcher cannot satisfy fails after ssh, not at $0."""
    for name, _mod, _args, spec in all_specs():
        for asset in spec.setup.local_assets:
            path = REPO / asset.repo_path
            if not path.exists():
                pytest.skip(f"{asset.repo_path} is a gitignored local artifact")
            root = asset.install_to.strip("/").split("/")[0]
            assert root in ("artifacts", "logs", "configs"), (
                f"{name} installs {asset.dest_name} to {asset.install_to}, "
                "outside the trees a session may write")


@pytest.mark.parametrize("name,extra", SESSION_LAUNCHERS,
                         ids=lambda v: v if isinstance(v, str) else "")
def test_no_launcher_mutates_another_modules_globals(name, extra):
    """The mechanism `SESSION_KIND` leaked through, removed by construction.

    Every session used to `import` the preflight launcher and assign to its
    module globals — `_preflight.AUTH_PATH`, `_preflight.STATUS`,
    `_preflight.LOCAL_ASSETS`, `_preflight.SpendAuthorization` — before
    constructing a subclass. Two sessions sharing one module is two sessions
    sharing one set of globals.
    """
    src = (POD / f"{name}.py").read_text()
    # `_preflight.` as an attribute access, not the substring — every session
    # legitimately names `autoinit_preflight_setup.sh` and its own status file.
    assert not re.search(r"(?<![A-Za-z0-9])_preflight\.", src), (
        f"{name} still retargets a shared module")
    assert "importlib.util.spec_from_file_location" not in src, (
        f"{name} loads another launcher as a module; sessions are specifications "
        "now and share the runner, not each other")
    assert not re.search(r"^\s*class \w+\(.*Preflight.*\):", src, re.M), (
        f"{name} subclasses a launcher again")


def test_the_runner_is_not_subclassed_anywhere():
    """One runner, no inheritance. That is the whole design."""
    for path in sorted(REPO.rglob("*.py")):
        if any(p in path.parts for p in (".git", ".venv", "__pycache__")):
            continue
        if path.name in ("session_runner.py", "test_session_architecture.py"):
            continue
        text = path.read_text(errors="ignore")
        assert not re.search(r"class \w+\(\s*SessionRunner\s*\)", text), (
            f"{path.relative_to(REPO)} subclasses SessionRunner")


def test_no_attempt_specific_grant_prose_in_executable_source():
    """A grant is a one-use decision and goes stale where code does not.

    `src/aadistill/autoinit/phase_a.py` carried attempt-7's grant — which attempt
    it covered, the cumulative spend at approval, what it did not authorize —
    inside the authorization constant, where it still read as current after the
    attempt was over. The schema stays; the grant arrives at issue time.
    """
    from aadistill.autoinit.phase_a import (
        GRANT_PROSE_REQUIRED, PHASE_A_AUTHORIZATION,
    )

    assert PHASE_A_AUTHORIZATION.granted_by == GRANT_PROSE_REQUIRED, (
        "the Phase-A authorization schema carries grant prose again")
    for attr in ("granted_utc", "science_plan_hash"):
        assert getattr(PHASE_A_AUTHORIZATION, attr) == "PLACEHOLDER", attr
    assert PHASE_A_AUTHORIZATION.authorized_session_commit is None
    assert PHASE_A_AUTHORIZATION.harness_source_digest is None

    # And no authorization CONSTANT in the core carries a grant. Checked on the
    # `granted_by` field rather than on free text: a module docstring listing
    # what past attempts cost is failure history, which AGENTS.md P11 requires
    # to stay, and is not a permission.
    from aadistill.autoinit.authorization import MICRO_PREFLIGHT_AUTHORIZATION
    from aadistill.autoinit.continuation import CONTINUATION_AUTHORIZATION

    for label, constant in (("micro-preflight", MICRO_PREFLIGHT_AUTHORIZATION),
                            ("continuation", CONTINUATION_AUTHORIZATION),
                            ("phase A", PHASE_A_AUTHORIZATION)):
        prose = constant.granted_by
        offenders = re.findall(
            r"attempt \d+[^.\n]{0,80}\$\d|\$\d[^.\n]{0,80}attempt \d+", prose)
        assert not offenders, (
            f"the {label} authorization constant names an attempt beside a "
            f"dollar figure: {offenders[:2]}. That is a one-use grant living in "
            "executable source.")


def test_the_issuer_refuses_to_issue_without_a_grant_document():
    """Issuing must not be possible by running the script."""
    src = (REPO / "scripts/autoinit/issue_phase_a_authorization.py").read_text()
    assert '"--grant", required=True' in src or '--grant", required=True' in src, (
        "the grant document is optional again")
    assert "no grant document at" in src
    assert "which this script\n" in src or "which this script " in src, (
        "the issuer no longer refuses a grant that asserts a derived identity")


def test_every_session_names_a_distinct_status_file_and_authorization():
    """Two sessions sharing either one is the $0.1324 and the $0.1369."""
    specs = [spec for _n, _m, _a, spec in all_specs()]
    for field in ("status_path", "run_log_path", "authorization_path",
                  "driver_job_id", "plan_hash"):
        values = [getattr(s, field) for s in specs]
        assert len(set(values)) == len(values), f"sessions share {field}: {values}"


def test_the_specs_are_frozen():
    """A spec that could be mutated after validation is a spec that can differ
    from the one that was validated."""
    from aadistill.infrastructure.session import SessionSpec

    _n, _m, _a, spec = all_specs()[0]
    assert dataclasses.is_dataclass(SessionSpec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.status_path = "/workspace/somewhere_else.status"
