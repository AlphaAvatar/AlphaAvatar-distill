"""The shared setup script's `c2_full_search` authorization branch, EXECUTED.

`tests/pod/test_session_kind_dispatch.py` already maps every launcher's
`SESSION_KIND` to a shell branch and requires the two to name the same loader
class. It is a good guard and it WOULD have caught the missing
`c2_full_search` branch — verified by removing the branch and watching it fail.
What it did not do was run the branch. This file does, because the difference
matters: a branch can name the right class and still assert the wrong things,
or assert nothing.

The branch is the LAST thing a formal pod does before `AUTHORIZATION_OK`, after
paid setup and the whole test gate. A missing or wrong branch is therefore not a
type error — it is a late refusal on a billing machine, which is what
`SESSION_KIND=phase_b` cost `$0.2300` to establish.

So the python body is extracted from the shell source and run against real
artifacts: one valid full-search authorization, which must be admitted, and one
per governance boundary, each of which must be refused. Extracted rather than
retyped, because a copy of the assertions here would pass while the shell's
copy said something else — which is exactly the failure mode being guarded.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO / extra) not in sys.path:
        sys.path.insert(0, str(REPO / extra))

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"
KIND = "c2_full_search"


def branch_body(kind: str = KIND) -> str:
    """The python the shell runs for `kind`, lifted out of the shell source.

    The shell embeds it in a double-quoted `python -c "..."`, so `\\$` is an
    escaped dollar the shell would have unescaped before python saw it. Undone
    here for the same reason: what is under test is the code the pod runs, not
    the shell's quoting of it.
    """
    source = SETUP.read_text()
    start = source.index(f'elif [ "$SESSION_KIND" = "{kind}" ]; then')
    #: Up to the next branch, so a body is never read past its own `elif`.
    rest = source[start:]
    nxt = rest.find('\nelif [ "$SESSION_KIND"', 1)
    if nxt == -1:
        nxt = rest.find('\nelse', 1)
    block = rest[:nxt if nxt != -1 else len(rest)]
    body = re.search(r'python -c "\n(.*?)\n" \|\|', block, re.S)
    assert body, f"no python -c body found in the {kind} branch"
    return body.group(1).replace("\\$", "$")


def run_body(auth_path: Path, plan_hash: str, body: str | None = None):
    """Run the branch body exactly as the pod would, in a fresh process.

    A fresh process because the body imports the authorization module and
    `full_search.staged_assets` registers calibration profiles: a registry a
    sibling test already filled would make an assertion here pass for the wrong
    reason.
    """
    program = body if body is not None else branch_body()
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True, text=True, cwd=str(REPO), timeout=600,
        env={**os.environ,
             "PYTHONPATH": f"{REPO}/src:{REPO}/scripts",
             "SESSION_AUTH_PATH": str(auth_path),
             "SESSION_PLAN_HASH": plan_hash,
             "HF_HUB_OFFLINE": "1"})


def artifact(tmp_path: Path, name: str = "auth.json", **over) -> Path:
    """A real `FullSearchAuthorization` artifact, through its own serializer.

    Not a hand-written dict: the serializer reads the permission flags FROM the
    class's properties, so an artifact built any other way could claim a
    combination the type cannot actually produce — and the branch would then be
    tested against something no issuer can emit.
    """
    from experiments.phase_c2 import full_search as FSG
    from experiments.phase_c2.session import C2ResourceScope

    auth = FSG.FullSearchAuthorization(
        authorization_id="autoinit.v1.phase_c2.full_search",
        granted_by="a test", granted_utc="2026-01-01T00:00:00+00:00",
        plan_id=FSG.PLAN_ID,
        plan_hash=FSG.plan_hash(REPO),
        science_plan_hash=FSG.plan_hash(REPO),
        expected_usd=16.0998, hard_cap_usd=33.1827,
        authorized_stages=FSG.AUTHORIZED_STAGES,
        stage_conditions={k: "as declared" for k in FSG.AUTHORIZED_STAGES},
        scope_note="one beam over the derived joint space, one committed Top-5",
        authorized_session_commit="a" * 40,
        harness_source_digest="d" * 64,
        harness_source_files=("scripts/pod/autoinit_phase_c2_full_search_driver.py",),
        resource_scope=C2ResourceScope(
            run_id="attempt1", issuances_permitted=1,
            launch_attempts_permitted=1, provider_resources_permitted=2,
            one_billing_resource_at_a_time=True),
        per_launch_hard_usd=33.1827, provenance_commit="a" * 40)
    doc = auth.as_dict()
    doc.update(over)
    if over:
        from aadistill.infrastructure.manifest import sha256_json
        doc.pop("authorization_sha256", None)
        doc["authorization_sha256"] = sha256_json(doc)
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


# --- the branch exists and is reachable -------------------------------------

def test_the_shell_has_a_branch_for_the_kind_the_launcher_exports():
    """The launcher's own value, read from the launcher, not retyped."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fs_launch_for_dispatch",
        REPO / "scripts/pod/autoinit_phase_c2_full_search_launch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fs_launch_for_dispatch"] = module
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "a" * 40, "--bundle", "b",
         "--run-id", "attempt1", "--max-price", "1.09"])
    exported = module.spec(args).setup.env["SESSION_KIND"]
    assert exported == KIND
    assert f'elif [ "$SESSION_KIND" = "{exported}" ]; then' in SETUP.read_text()


def test_the_shell_still_parses():
    assert subprocess.run(["bash", "-n", str(SETUP)],
                          capture_output=True).returncode == 0


# --- it ADMITS a valid artifact ---------------------------------------------

def test_the_branch_admits_a_valid_full_search_authorization(tmp_path):
    from experiments.phase_c2 import full_search as FSG

    out = run_body(artifact(tmp_path), FSG.plan_hash(REPO))
    assert out.returncode == 0, out.stdout + out.stderr
    #: And it PRINTS the bound scope and ceiling, which is what a reader of a
    #: pod transcript has to see to know what was permitted.
    assert "autoinit.v1.phase_c2.full_search" in out.stdout
    assert "33.1827" in out.stdout
    assert "full search True" in out.stdout
    for boundary in ("search1 False", "completion False", "behavioural False",
                     "training False"):
        assert boundary in out.stdout, f"the transcript does not state {boundary!r}"
    for stage in FSG.AUTHORIZED_STAGES:
        assert stage in out.stdout


# --- it REFUSES every boundary ----------------------------------------------

def test_the_branch_refuses_a_foreign_plan_hash(tmp_path):
    out = run_body(artifact(tmp_path), "0" * 64)
    assert out.returncode != 0
    assert "plan" in (out.stdout + out.stderr).lower()


@pytest.mark.parametrize("claim,why", [
    ("authorizes_c2_search1", "a full-search grant must not authorize the Search-1 beam"),
    ("authorizes_c2_baseline_completion", "must not authorize a baseline rebuild"),
    ("authorizes_behavioural_selection", "nothing may authorize behavioural work"),
    ("allows_phase_a", "claims Phase A authorization"),
    ("allows_recovery_training", "trains nothing"),
])
def test_the_branch_refuses_each_forbidden_claim(tmp_path, claim, why):
    """One at a time, so a single assertion cannot appear to cover all six.

    These are the governance boundaries in their FINAL position: whatever the
    launcher's `$0` gates concluded, this is the last check before a pod is
    allowed to start work.
    """
    from experiments.phase_c2 import full_search as FSG

    path = artifact(tmp_path, name=f"{claim}.json", **{claim: True})
    out = run_body(path, FSG.plan_hash(REPO))
    assert out.returncode != 0, (
        f"the branch admitted an artifact claiming {claim}:\n{out.stdout}")
    combined = out.stdout + out.stderr
    #: Either the loader refused it or the branch's own assertion did; both are
    #: correct, and requiring one specifically would make the test brittle about
    #: which layer catches it.
    assert claim in combined or why.split()[0] in combined.lower(), combined


def test_the_followon_assertion_cannot_fail_and_that_is_STRONGER(tmp_path):
    """`automatic_followon_start` is not a field, so no artifact can claim it.

    This test started out asserting that the branch REFUSES an artifact setting
    it, and the branch admitted one -- correctly. The attribute is a property
    hardcoded to False on the base type ("Not a field. Phase A stops for
    review; nothing chains off it"), so the dict key is inert and the shell's
    `assert a.automatic_followon_start is False` can never trip.

    That is a stronger guarantee than a field, not a weaker one: a field could
    be set, a property cannot. But an assertion that cannot fail is also not
    evidence, so what is checked here is the actual property -- and the
    artifact override is confirmed inert rather than left looking enforced.
    """
    from experiments.phase_c2 import full_search as FSG

    prop = type(FSG.FullSearchAuthorization.automatic_followon_start)
    assert prop is property, "it became a field; the shell assertion now matters"
    path = artifact(tmp_path, name="followon.json",
                    automatic_followon_start=True)
    loaded = FSG.FullSearchAuthorization.load(path)
    assert loaded.automatic_followon_start is False, (
        "the artifact's claim reached the object, so the property became a "
        "field and the branch's assertion must now be tested for refusal")
    #: And the branch admits it, because there is nothing to refuse.
    out = run_body(path, FSG.plan_hash(REPO))
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_branch_refuses_an_artifact_of_another_type(tmp_path):
    """A baseline-completion artifact reaching the full-search branch.

    The symmetric direction is already covered by the completion's own branch;
    this is the one that matters here, because a completion grant is cheap
    ($1.1950) and a full search is not.
    """
    from experiments.phase_c2 import baseline_completion as BC
    from experiments.phase_c2 import full_search as FSG
    from experiments.phase_c2.session import C2ResourceScope

    other = BC.BaselineCompletionAuthorization(
        authorization_id="autoinit.v1.phase_c2.baseline_completion",
        granted_by="a test", granted_utc="2026-01-01T00:00:00+00:00",
        plan_id=BC.PLAN_ID, plan_hash=BC.plan_hash(REPO),
        science_plan_hash=BC.plan_hash(REPO),
        expected_usd=0.5, hard_cap_usd=1.1950,
        authorized_stages=("bind_identities", "rebuild_measure_compare"),
        stage_conditions={"bind_identities": "x",
                          "rebuild_measure_compare": "y"},
        scope_note="one rebuild",
        authorized_session_commit="a" * 40,
        harness_source_digest="d" * 64, harness_source_files=("x",),
        resource_scope=C2ResourceScope(
            run_id="attempt1", issuances_permitted=1,
            launch_attempts_permitted=1, provider_resources_permitted=1,
            one_billing_resource_at_a_time=True),
        per_launch_hard_usd=1.1950, provenance_commit="a" * 40)
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(other.as_dict(), indent=1) + "\n")

    out = run_body(path, FSG.plan_hash(REPO))
    assert out.returncode != 0, (
        "the full-search branch admitted a baseline-completion artifact:\n"
        + out.stdout)
    assert "schema" in (out.stdout + out.stderr).lower()


def test_the_branch_refuses_a_tampered_artifact(tmp_path):
    """Edited after issuance: the self-hash no longer verifies."""
    from experiments.phase_c2 import full_search as FSG

    path = artifact(tmp_path)
    doc = json.loads(path.read_text())
    doc["hard_cap_usd"] = 99.0            # raise the ceiling, keep the hash
    path.write_text(json.dumps(doc, indent=1))
    out = run_body(path, FSG.plan_hash(REPO))
    assert out.returncode != 0
    assert "authorization_sha256" in (out.stdout + out.stderr)


# --- the body under test is the shell's, not a copy --------------------------

def test_the_body_is_extracted_from_the_shell_and_not_retyped():
    """The property that makes every test above mean something.

    A copy of these assertions living in this file would pass while the shell
    said something else -- which is precisely the failure mode being guarded.
    So the extractor is checked: it must find the branch, the body must import
    the full-search type, and it must contain the four permission assertions.
    """
    body = branch_body()
    assert "FullSearchAuthorization" in body
    assert "require_plan" in body
    #: As ASSERTIONS, not as names. A first version of this test looked for the
    #: bare attribute name and passed against a branch whose assertion had been
    #: deleted -- because every name also appears in the branch's `print`, which
    #: reports the flags for the pod transcript. So "the branch mentions
    #: `authorizes_behavioural_selection`" was true of a branch that merely
    #: printed it, and a mutation confirmed the hole.
    import ast

    tree = ast.parse(body)
    asserted: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and len(test.comparators) == 1
                and isinstance(test.ops[0], ast.Is)
                and isinstance(test.left, ast.Attribute)):
            asserted[test.left.attr] = ast.literal_eval(test.comparators[0])

    assert asserted.get("authorizes_c2_full_search") is True, (
        "the branch does not ASSERT that this artifact authorizes the search")
    for claim in ("authorizes_c2_search1", "authorizes_c2_baseline_completion",
                  "authorizes_behavioural_selection", "allows_phase_a",
                  "allows_recovery_training", "automatic_followon_start"):
        assert claim in asserted, (
            f"the shell branch does not ASSERT {claim}; mentioning it in the "
            "print is not a check")
        assert asserted[claim] is False, (
            f"the branch asserts {claim} is {asserted[claim]!r}, not False")
    #: And it is the branch for THIS kind: the extractor must not run past its
    #: own `elif` into a neighbour's body.
    assert "BaselineCompletionAuthorization" not in body
    assert "C2Authorization" not in body
