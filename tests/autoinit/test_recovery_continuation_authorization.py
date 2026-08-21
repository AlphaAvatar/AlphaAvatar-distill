"""The continuation must be authorized against the executable it actually runs.

The gap these close: the continuation had its own `SessionSpec`, driver and
budget, but the only issuer bound `PHASE_A_HARNESS_SOURCE_FILES_V1` — a set that
measures the full launcher, driver and beam search and contains **neither**
continuation file. Issuing with `--out logs/autoinit_recovery_continuation_…`
would have produced a green harness digest over code the paid run does not
execute, while the launcher, driver and strict importer it *does* execute went
unmeasured. It would also have carried the search's $23.0484 ceiling into a
session priced at $16.7456.

Every mutation below is a *passing* state made to fail. A digest that does not
move when its executable moves is not measuring it.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in ("src", "scripts/pod", "scripts/autoinit"):
    sys.path.insert(0, str(REPO / p))

from aadistill.autoinit.authorization import AuthorizationError  # noqa: E402
from aadistill.autoinit.phase_a import (  # noqa: E402
    PHASE_A_AUTHORIZATION, PHASE_A_HARNESS_SOURCE_FILES_V1, PHASE_A_PLAN_V1,
    PhaseAAuthorization, phase_a_harness_digest,
)
from aadistill.autoinit.recovery_continuation import (  # noqa: E402
    CONTINUATION_ONLY_HARNESS_FILES, RECOVERY_CONTINUATION_AUTHORIZATION,
    RECOVERY_CONTINUATION_HARNESS_FILES_V1, SCHEMA, SEARCH_ONLY_HARNESS_FILES,
    RecoveryContinuationAuthorization, recovery_continuation_harness_digest,
)

ISSUER = REPO / "scripts/autoinit/issue_recovery_continuation_authorization.py"
LAUNCH = "scripts/pod/autoinit_recovery_continuation_launch.py"
DRIVER = "scripts/pod/autoinit_recovery_continuation_driver.py"
IMPORTER = "src/aadistill/autoinit/stage1_import.py"
HANDOFF = "src/aadistill/autoinit/device_handoff.py"
SEARCH = "scripts/autoinit/phase_a_search.py"


def load_module(rel, name):
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return load_module(LAUNCH, "rca_launch")


@pytest.fixture
def mirror(tmp_path):
    """A copy of every harness file, so a mutation never touches the real tree.

    Includes the search module, which the continuation set excludes — a test
    that it is excluded needs the file to exist and to be mutable.
    """
    root = tmp_path / "repo"
    for rel in set(RECOVERY_CONTINUATION_HARNESS_FILES_V1) | {SEARCH}:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    return root


def digest_of(root) -> str:
    return recovery_continuation_harness_digest(root)["digest"]


def issued(root, **over):
    """An authorization as the issuer would build it, against `root`."""
    harness = recovery_continuation_harness_digest(root)
    auth = replace(
        RECOVERY_CONTINUATION_AUTHORIZATION,
        authorization_id="autoinit.recovery_continuation.TEST",
        granted_utc="2026-08-21T00:00:00Z", granted_by="test",
        science_plan_hash="sci-hash",
        expected_usd=14.9233, hard_cap_usd=16.7456,
        per_launch_hard_usd=16.7456,
        authorized_session_commit="0" * 40,
        harness_source_digest=harness["digest"], **over)
    return auth


# --- the set itself ---------------------------------------------------------

def test_the_continuation_set_is_the_phase_a_set_minus_search_plus_its_own():
    """Derived, not transcribed. Two hand-maintained copies of fourteen shared
    paths would drift, and the failure is silent: whichever was forgotten
    certifies a smaller harness than the one that runs."""
    assert set(RECOVERY_CONTINUATION_HARNESS_FILES_V1) == (
        (set(PHASE_A_HARNESS_SOURCE_FILES_V1) - set(SEARCH_ONLY_HARNESS_FILES))
        | set(CONTINUATION_ONLY_HARNESS_FILES))


def test_the_continuation_set_covers_what_this_session_executes():
    for rel in (LAUNCH, DRIVER, IMPORTER, HANDOFF,
                "src/aadistill/autoinit/leaf_durability.py",
                "scripts/autoinit/phase_a_frozen.py",
                "src/aadistill/autoinit/recovery_continuation.py"):
        assert rel in RECOVERY_CONTINUATION_HARNESS_FILES_V1, rel


def test_every_declared_file_exists():
    """A missing declared file raises rather than digesting a smaller harness."""
    for rel in RECOVERY_CONTINUATION_HARNESS_FILES_V1:
        assert (REPO / rel).is_file(), rel


def test_phase_a_was_not_broadened_to_include_this_session():
    """Full Phase A and the continuation are distinct operational harnesses and
    stay independently measured. Deriving one from the other is not the same as
    widening the other."""
    for rel in (LAUNCH, DRIVER, IMPORTER, HANDOFF):
        assert rel not in PHASE_A_HARNESS_SOURCE_FILES_V1, (
            f"{rel} reached the Phase-A set; a continuation edit would now "
            "revoke a full-Phase-A authorization")


def test_the_two_harnesses_are_different_numbers():
    a = phase_a_harness_digest(REPO)["digest"]
    b = recovery_continuation_harness_digest(REPO)["digest"]
    assert a != b


# --- MUTATION: the four executables that must move the digest ---------------

@pytest.mark.parametrize("rel, what", [
    (LAUNCH, "the continuation launcher"),
    (DRIVER, "the continuation driver"),
    (IMPORTER, "the strict Stage-1 importer"),
    (HANDOFF, "the device handoff"),
])
def test_editing_an_executable_dependency_invalidates_the_authorization(
        mirror, rel, what):
    """The whole point of the new set. Under the Phase-A set none of these four
    files is digested at all, so all four of these mutations would pass."""
    auth = issued(mirror)
    auth.require_harness(mirror)                       # green before

    before_phase_a = phase_a_harness_digest(
        mirror, files=PHASE_A_HARNESS_SOURCE_FILES_V1)["digest"]

    target = mirror / rel
    target.write_text(target.read_text() + "\n# mutated\n")

    with pytest.raises(AuthorizationError) as exc:
        auth.require_harness(mirror)
    assert "harness" in str(exc.value).lower(), (
        f"editing {what} did not invalidate the authorization")

    # The positive control, and the gap itself: under the Phase-A file set this
    # same edit is invisible. That is what a continuation authorization issued
    # by the Phase-A issuer would have certified.
    assert phase_a_harness_digest(
        mirror, files=PHASE_A_HARNESS_SOURCE_FILES_V1)["digest"] == before_phase_a


def test_editing_the_search_alone_does_not_change_the_continuation_identity(
        mirror):
    """It is unreachable from here: `PhaseADriver.stage1` imports
    `run_phase_a_search` *inside the method*, and the continuation overrides that
    method. Digesting it anyway would let an edit to code this session cannot
    execute revoke a valid authorization."""
    before = digest_of(mirror)
    search = mirror / SEARCH
    search.write_text(search.read_text() + "\n# mutated\n")
    assert digest_of(mirror) == before

    auth = issued(mirror)
    auth.require_harness(mirror)                       # still valid

    # And the exclusion is only defensible while the search stays unreachable.
    tree = ast.parse((REPO / DRIVER).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert not any("phase_a_search" in m for m in imported)


def test_the_search_is_not_in_the_continuation_set():
    assert SEARCH not in RECOVERY_CONTINUATION_HARNESS_FILES_V1
    assert SEARCH in PHASE_A_HARNESS_SOURCE_FILES_V1


# --- MUTATION: a Phase-A artifact cannot stand in for a continuation one -----

def test_a_full_phase_a_authorization_cannot_load_as_a_continuation_one(tmp_path):
    """Refused by SCHEMA, not by convention — the same rule that stops a
    `SpendAuthorization` being read as a Phase-A grant."""
    phase_a = replace(
        PHASE_A_AUTHORIZATION,
        authorization_id="autoinit.phase_a.TEST", granted_utc="2026-08-21T00:00:00Z",
        granted_by="test", science_plan_hash="sci-hash",
        authorized_session_commit="0" * 40,
        harness_source_digest=phase_a_harness_digest(REPO)["digest"])
    path = tmp_path / "phase_a.json"
    path.write_text(json.dumps(phase_a.as_dict(), indent=2))

    # It is a perfectly valid Phase-A authorization...
    assert PhaseAAuthorization.load(path).hard_cap_usd == 23.0484
    # ...and cannot authorize the continuation.
    with pytest.raises(AuthorizationError) as exc:
        RecoveryContinuationAuthorization.load(path)
    assert "declares schema" in str(exc.value)


def test_the_schema_string_is_what_does_the_refusing(tmp_path):
    """Isolates the discriminator. Deleting the schema check left the earlier
    test green, because a Phase-A artifact also lacks
    `authorizes_recovery_continuation` and was refused one line later — the
    protection held, but the test was not measuring the thing it named.

    This payload is a valid continuation authorization in every respect except
    the schema string, so nothing else can refuse it.
    """
    from aadistill.autoinit.phase_a import SCHEMA as PHASE_A_SCHEMA
    from aadistill.infrastructure.manifest import sha256_json

    payload = issued(REPO).as_dict()
    assert payload["authorizes_recovery_continuation"] is True
    assert payload["allows_beam_search"] is False
    payload["schema"] = PHASE_A_SCHEMA
    payload.pop("authorization_sha256")
    payload["authorization_sha256"] = sha256_json(payload)
    path = tmp_path / "wrong_schema.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(AuthorizationError) as exc:
        RecoveryContinuationAuthorization.load(path)
    assert "declares schema" in str(exc.value)


def test_the_full_session_ceiling_cannot_silently_price_the_continuation(tmp_path):
    """$23.0484 funds a beam search this session does not run. Substituting it
    would authorize 6.3 GPU-hours of headroom for work that was priced without
    them."""
    phase_a = replace(
        PHASE_A_AUTHORIZATION, granted_utc="x", granted_by="t",
        science_plan_hash="s", authorized_session_commit="0" * 40,
        harness_source_digest=phase_a_harness_digest(REPO)["digest"])
    path = tmp_path / "phase_a.json"
    path.write_text(json.dumps(phase_a.as_dict(), indent=2))
    with pytest.raises(AuthorizationError):
        RecoveryContinuationAuthorization.load(path)

    # And what the continuation's own type carries is the derived pair.
    auth = issued(REPO)
    assert (auth.expected_usd, auth.hard_cap_usd) == (14.9233, 16.7456)
    assert auth.per_launch_hard_usd == 16.7456
    with pytest.raises(AuthorizationError):
        auth.require_within_cap(23.0484)


def test_a_continuation_artifact_cannot_claim_a_search(tmp_path):
    auth = issued(REPO)
    payload = auth.as_dict()
    payload["allows_beam_search"] = True
    payload.pop("authorization_sha256")
    from aadistill.infrastructure.manifest import sha256_json
    payload["authorization_sha256"] = sha256_json(payload)
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(AuthorizationError) as exc:
        RecoveryContinuationAuthorization.load(path)
    assert "search" in str(exc.value).lower()


def test_the_artifact_round_trips_and_is_tamper_evident(tmp_path):
    auth = issued(REPO)
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(auth.as_dict(), indent=2))
    back = RecoveryContinuationAuthorization.load(path)
    assert back.hard_cap_usd == 16.7456
    assert back.harness_source_files == RECOVERY_CONTINUATION_HARNESS_FILES_V1
    assert back.allows_beam_search is False
    assert back.authorizes_recovery_continuation is True
    back.require_harness(REPO)

    raw = json.loads(path.read_text())
    raw["hard_cap_usd"] = 23.0484
    path.write_text(json.dumps(raw))
    with pytest.raises(AuthorizationError) as exc:
        RecoveryContinuationAuthorization.load(path)
    assert "edited" in str(exc.value)


def test_the_schema_string_is_distinct():
    from aadistill.autoinit.phase_a import SCHEMA as PHASE_A_SCHEMA
    assert SCHEMA != PHASE_A_SCHEMA
    assert RECOVERY_CONTINUATION_AUTHORIZATION.as_dict()["schema"] == SCHEMA


# --- the production path uses it --------------------------------------------

def test_the_launcher_loads_the_continuation_type(launcher):
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"])
    spec = launcher.spec(args)
    assert spec.authorization_loader == RecoveryContinuationAuthorization.load, (
        "the production launcher still loads the Phase-A type")
    assert spec.authorization_path == (
        "logs/autoinit_recovery_continuation_authorization.json")


def test_the_precheck_recomputes_the_continuation_harness(launcher):
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"])
    names = [getattr(c, "__name__", "") for c in launcher.spec(args).precheck]
    assert "continuation_harness_gate" in names


class _Ctx:
    def __init__(self, auth):
        self.auth = auth
        self.evidence = {}


def test_the_gate_passes_on_the_real_tree(launcher):
    ok, why = launcher.continuation_harness_gate(_Ctx(issued(REPO)))
    assert ok, why


def test_the_gate_refuses_an_artifact_declaring_the_phase_a_file_list(launcher):
    """The one thing `require_harness()` cannot catch on its own: it digests
    whatever list the artifact *stores*, so an artifact carrying the Phase-A list
    verifies perfectly against the Phase-A files while this launcher, this driver
    and the importer go unmeasured. The gate asks from the other side."""
    smuggled = replace(
        issued(REPO),
        harness_source_files=PHASE_A_HARNESS_SOURCE_FILES_V1,
        harness_source_digest=phase_a_harness_digest(REPO)["digest"])
    smuggled.require_harness(REPO)          # its own check is satisfied...
    ok, why = launcher.continuation_harness_gate(_Ctx(smuggled))
    assert not ok and "different harness set" in why


def test_the_pod_driver_loads_the_continuation_artifact_not_the_phase_a_one():
    """The consumer that actually governs spend, and the one this work nearly
    missed. `PhaseADriver.__init__` loaded a hard-coded
    `logs/autoinit_phase_a_authorization.json` — a file that IS committed,
    holding attempt 12's consumed $23.0484 authorization. The continuation
    subclasses that driver, so on the pod it would have enforced
    `require_within_cap` against the search's ceiling, not its own, and recorded
    the wrong grant as the thing that authorized the run. Not a crash: a
    silently wrong number, 38% too high.
    """
    drv = load_module(DRIVER, "rca_driver")
    from aadistill.autoinit.phase_a import PhaseAAuthorization as PA

    assert drv.RecoveryContinuationDriver.AUTHORIZATION_TYPE is (
        RecoveryContinuationAuthorization)
    assert drv.RecoveryContinuationDriver.AUTHORIZATION_PATH == (
        "logs/autoinit_recovery_continuation_authorization.json")
    # And the parent is unchanged, so full Phase A still loads its own.
    assert drv.PhaseADriver.AUTHORIZATION_TYPE is PA
    assert drv.PhaseADriver.AUTHORIZATION_PATH == (
        "logs/autoinit_phase_a_authorization.json")
    assert (drv.RecoveryContinuationDriver.AUTHORIZATION_PATH
            != drv.PhaseADriver.AUTHORIZATION_PATH)


def test_the_committed_phase_a_artifact_cannot_be_read_as_a_continuation_one():
    """Concrete, against the real file the driver used to load."""
    stale = REPO / "logs/autoinit_phase_a_authorization.json"
    if not stale.is_file():
        pytest.skip("no committed Phase-A authorization to check against")
    assert PhaseAAuthorization.load(stale).hard_cap_usd == 23.0484
    with pytest.raises(AuthorizationError) as exc:
        RecoveryContinuationAuthorization.load(stale)
    assert "declares schema" in str(exc.value)


def test_the_setup_gate_has_a_branch_for_this_session(launcher):
    """`SESSION_KIND` selects the loader inside the shared setup script. The
    continuation declared none, so it fell to `spend`, whose `SpendAuthorization`
    refuses any artifact asserting `phase_a_authorized` — the session would have
    died at setup with exit 98 before doing any work."""
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"])
    setup = launcher.spec(args).setup
    assert setup.env.get("SESSION_KIND") == "recovery_continuation"
    assert "SESSION_KIND" in setup.required_env

    sh = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    assert '"$SESSION_KIND" = "recovery_continuation"' in sh
    assert "RecoveryContinuationAuthorization.load" in sh
    # The branch must assert the two properties that distinguish this session.
    branch = sh.split('"$SESSION_KIND" = "recovery_continuation"')[1].split("else")[0]
    assert "allows_beam_search is False" in branch
    assert "authorizes_recovery_continuation is True" in branch


def test_the_setup_gate_block_actually_runs_against_a_real_artifact(tmp_path):
    """Executes the real shell block, not a transcription of it.

    String-matching the branch proves it was typed, not that it works. This
    lifts the block out of the setup script verbatim, runs it under bash with
    `SESSION_KIND=recovery_continuation`, and requires the three outcomes that
    matter: this session's artifact reaches the driver, a full-Phase-A artifact
    is refused with the classified exit 98, and a plan hash that is not this
    session's is refused too.
    """
    import os
    import tempfile

    setup = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    lines = setup.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines)
                 if l.startswith('SESSION_KIND="${SESSION_KIND:-spend}"'))
    end = next(i for i, l in enumerate(lines) if l.rstrip() == "mark AUTHORIZATION_OK")
    block = "".join(lines[start:end + 1]).replace("/opt/train/bin/python",
                                                  sys.executable)

    ours = tmp_path / "continuation.json"
    ours.write_text(json.dumps(issued(REPO).as_dict(), indent=2))
    theirs = tmp_path / "phase_a.json"
    theirs.write_text(json.dumps(replace(
        PHASE_A_AUTHORIZATION, granted_utc="x", granted_by="t",
        science_plan_hash="s", authorized_session_commit="0" * 40,
        harness_source_digest=phase_a_harness_digest(REPO)["digest"]
    ).as_dict(), indent=2))

    with tempfile.TemporaryDirectory() as tmp:
        markers = Path(tmp) / "markers"
        script = Path(tmp) / "gate.sh"
        script.write_text(
            'set -euo pipefail\n'
            'say() { echo "  $*"; }\n'
            f'mark() {{ echo "$*" >> "{markers}"; }}\n'
            f'REPO="{REPO}"\n' + block + 'echo REACHED_THE_DRIVER\n')

        def run(auth, plan, kind="recovery_continuation"):
            env = dict(os.environ)
            # Controlled, never inherited: leaving SESSION_KIND to the invoking
            # environment made an earlier gate test pass on the dev box and fail
            # on a pod.
            env["SESSION_KIND"] = kind
            env["SESSION_AUTH_PATH"] = str(auth)
            env["SESSION_PLAN_HASH"] = plan
            markers.write_text("")
            r = subprocess.run(["bash", str(script)], capture_output=True,
                               text=True, env=env, cwd=str(REPO))
            return r, markers.read_text()

        ok, marks = run(ours, PHASE_A_PLAN_V1.plan_hash)
        assert ok.returncode == 0, ok.stdout + ok.stderr
        assert "REACHED_THE_DRIVER" in ok.stdout
        assert marks.strip() == "AUTHORIZATION_OK"
        assert "16.7456" in ok.stdout, ok.stdout

        # A full-Phase-A artifact, refused by the branch that runs here.
        bad, marks = run(theirs, PHASE_A_PLAN_V1.plan_hash)
        assert bad.returncode == 98, f"rc={bad.returncode}: {bad.stdout}{bad.stderr}"
        assert "REACHED_THE_DRIVER" not in bad.stdout
        assert marks.strip() == "AUTHORIZATION_MISMATCH"

        # And a plan that is not this session's.
        bad, marks = run(ours, "0" * 64)
        assert bad.returncode == 98
        assert marks.strip() == "AUTHORIZATION_MISMATCH"

        # The defect as it stood: with no SESSION_KIND declared, the block falls
        # to `spend`, and this session dies at setup before doing any work.
        fell, marks = run(ours, PHASE_A_PLAN_V1.plan_hash, kind="spend")
        assert fell.returncode == 98
        assert marks.strip() == "AUTHORIZATION_MISMATCH"


def test_the_gate_refuses_a_stale_digest(launcher):
    stale = replace(issued(REPO), harness_source_digest="0" * 64)
    ok, why = launcher.continuation_harness_gate(_Ctx(stale))
    assert not ok and "re-issue" in why


# --- the issuer -------------------------------------------------------------

def grant(**over):
    doc = {"granted_by": "a test maintainer", "covers": "one continuation",
           "cumulative_spend_at_approval_usd": 213.4714,
           "cumulative_cap_usd": 234.0,
           "does_not_authorize": "a search, a fresh stage 1, anything after",
           "grant_type": "recovery_continuation"}
    doc.update(over)
    return doc


def run_issuer(tmp_path, doc, out="logs/_test_continuation_auth.json"):
    g = tmp_path / "grant.json"
    g.write_text(json.dumps(doc))
    return subprocess.run(
        [sys.executable, str(ISSUER), "--grant", str(g), "--out", out],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})


def test_the_issuer_requires_a_continuation_grant(tmp_path):
    """A Phase-A grant approved a beam search at a search's price; it does not
    carry over to a session that imports Stage 1."""
    r = run_issuer(tmp_path, grant(grant_type="phase_a"))
    assert r.returncode != 0
    assert "not a 'recovery_continuation'" in r.stderr


def test_the_issuer_refuses_a_grant_that_asserts_its_own_price(tmp_path):
    r = run_issuer(tmp_path, grant(hard_cap_usd=23.0484))
    assert r.returncode != 0 and "derives" in r.stderr


def test_the_issuer_refuses_without_a_grant():
    r = subprocess.run([sys.executable, str(ISSUER)], capture_output=True,
                       text=True, cwd=REPO, env={"PYTHONPATH": "src"})
    assert r.returncode != 0
    assert "--grant" in r.stderr


def test_the_issuer_derives_the_price_and_the_continuation_harness(tmp_path):
    out = "logs/_test_continuation_auth.json"
    r = run_issuer(tmp_path, grant(), out=out)
    try:
        assert r.returncode == 0, r.stderr
        report = json.loads(r.stdout)
        assert report["expected_usd"] == 14.9233
        assert report["hard_cap_usd"] == 16.7456
        assert report["per_launch_hard_usd"] == 16.7456
        assert report["harness_covers_search"] is False
        assert report["launched"] is False
        assert (report["harness_source_digest"]
                == recovery_continuation_harness_digest(REPO)["digest"])
        assert set(report["harness_source_files"]) == set(
            RECOVERY_CONTINUATION_HARNESS_FILES_V1)
        # The written artifact loads through the launcher's loader.
        auth = RecoveryContinuationAuthorization.load(REPO / out)
        assert auth.hard_cap_usd == 16.7456
        assert auth.plan_hash == PHASE_A_PLAN_V1.plan_hash
    finally:
        (REPO / out).unlink(missing_ok=True)


def test_the_issuer_writes_no_dollar_figure_of_its_own():
    """A written number would drift from the plan it claims to derive from."""
    src = ISSUER.read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "derived_pricing")
    code = ast.unparse(fn).split('"""')[-1]
    for token in ("14.9", "16.7", "23.0", "904"):
        assert token not in code, f"{token!r} is written into the derivation"


def test_the_issuer_defaults_to_the_continuation_artifact_path():
    src = ISSUER.read_text()
    assert 'logs/autoinit_recovery_continuation_authorization.json' in src


def test_the_module_carries_no_grant_prose():
    """A grant is a one-use decision about a particular attempt at a particular
    cumulative spend; in executable source it goes stale silently."""
    from aadistill.autoinit.recovery_continuation import (
        CONTINUATION_GRANT_PROSE_REQUIRED)
    assert (RECOVERY_CONTINUATION_AUTHORIZATION.granted_by
            == CONTINUATION_GRANT_PROSE_REQUIRED)
    assert "NO GRANT" in CONTINUATION_GRANT_PROSE_REQUIRED


def test_any_continuation_authorization_present_is_a_spent_one():
    """Until 2026-08-21 this asserted the artifact did not exist, because the
    work was readiness rather than permission. The maintainer then authorized one
    continuation and it was issued, so that form of the test has done its job.

    What still has to hold is the property it was protecting: **no session
    authorizes itself.** An artifact may exist only as the record of a permission
    that was granted and is now spent — bound to a base commit that is no longer
    HEAD, so its lineage gate refuses the current tree by construction.
    """
    path = REPO / "logs/autoinit_recovery_continuation_authorization.json"
    if not path.exists():
        return
    auth = RecoveryContinuationAuthorization.load(path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO).stdout.strip()
    assert auth.authorized_session_commit not in ("", None, head), (
        "the authorization names the current HEAD as its authorized base, which "
        "is what a self-issued artifact looks like")
    # And it is still the narrow thing that was granted.
    assert auth.hard_cap_usd == 16.7456 and auth.allows_beam_search is False
