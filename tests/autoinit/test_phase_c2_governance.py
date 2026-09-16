"""The C2 launch chain: closure, readiness, issuer, transport, gates.

Nothing here spends anything, creates a pod, stages a real bundle or issues an
authorization. What it does do is execute each link for real — the derived
closure against the live tree, the payload builder against a candidate grant in
`tmp_path`, the transport round-trip against a bundle of this repository and a
local fake download, and the launcher's gates against those artifacts — because
the alternative is a chain whose links are only described.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for _root in ("src", "scripts", "scripts/pod", "scripts/autoinit", "tests/pod"):
    if str(REPO / _root) not in sys.path:
        sys.path.insert(0, str(REPO / _root))

from session_specs import load_session_launcher, session_args  # noqa: E402

RUN_ID = "c2gov_test"
STAGE_ID = "1"



def code_without_prose(path: Path) -> str:
    """A module's source with its docstring and comments removed.

    By LINE RANGE from the parsed tree, not by `source.replace(docstring, "")`:
    a docstring containing a line continuation — which both modules checked
    below have, in their usage examples — has a VALUE that differs from its
    source text, so the replace silently removed nothing and the check passed
    for the wrong reason.
    """
    import ast

    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    body = tree.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        first, last = body[0].lineno - 1, body[0].end_lineno
        lines = lines[:first] + lines[last:]
    return "\n".join(line for line in lines
                      if not line.strip().startswith("#"))


@pytest.fixture(scope="module")
def launcher():
    return load_session_launcher("autoinit_phase_c2_launch")


# --- item 3: the derived executable closure ---------------------------------

def test_the_closure_derives_with_no_unresolved_internal_imports():
    """`derive` raises on an unresolved internal import, so a successful call
    already proves the set is closed. Asked of `walk` directly as well, because
    "it did not raise" and "the unresolved list is empty" are different
    statements and only one of them is reportable."""
    from aadistill.governance.closure import walk
    from experiments.phase_c2.session import (
        C2_ENTRY_POINTS, C2_SOURCE_ROOTS, c2_current_executable,
    )

    from experiments.phase_c2.session import C2_DECLARED_INPUTS

    files, unresolved = walk(REPO, C2_ENTRY_POINTS, C2_SOURCE_ROOTS)
    assert unresolved == [], unresolved
    live = c2_current_executable(REPO)
    #: The union, not the sum: `autoinit_preflight_setup.sh` is BOTH a declared
    #: input and a subprocess target the walk finds, and declaring it anyway is
    #: correct -- the declaration is what says its bytes matter even if the
    #: literal that names it is ever moved into a variable.
    assert live["n_files"] == len(set(files) | set(C2_DECLARED_INPUTS))
    assert len(live["digest"]) == 64


def test_the_closure_contains_what_a_c2_session_actually_runs():
    """Named individually: each of these decides what the session does, and a
    closure that omitted one would bind a digest that could not detect its
    edit."""
    from experiments.phase_c2.session import c2_current_executable

    paths = {f["path"] for f in c2_current_executable(REPO)["files"]}
    for rel in (
            #: the session, end to end
            "scripts/pod/autoinit_phase_c2_launch.py",
            "scripts/pod/autoinit_phase_c2_driver.py",
            "scripts/pod/collect_artifacts.py",
            "scripts/pod/watchdog.py",
            #: the experiment layer
            "scripts/experiments/phase_c2/search_space.py",
            "scripts/experiments/phase_c2/baseline.py",
            "scripts/experiments/phase_c2/comparison.py",
            "scripts/experiments/phase_c2/session.py",
            "scripts/experiments/phase_c2/bundle.py",
            "scripts/experiments/phase_c2/pod_environment.py",
            "scripts/experiments/phase_c2/authorization_payload.py",
            "scripts/autoinit/issue_c2_authorization.py",
            #: the search seam and the frozen identities it resolves
            "scripts/autoinit/phase_a_search.py",
            "scripts/autoinit/phase_a_frozen.py",
            "scripts/autoinit/load_state_eval.py",
            "scripts/experiments/phase_c1/session.py",
            #: everything that can spend money
            "src/aadistill/infrastructure/session_runner.py",
            "src/aadistill/infrastructure/provider.py",
            "src/aadistill/infrastructure/remote.py",
            "src/aadistill/infrastructure/budget.py",
            #: the operators the search composes
            "src/aadistill/initialization/operators/attention_activation.py",
            "src/aadistill/initialization/operators/depth.py",
            "src/aadistill/initialization/operators/ffn.py",
            "src/aadistill/initialization/operators/width.py",
            "src/aadistill/initialization/planning/ranking.py",
            #: the declared non-python inputs
            "scripts/pod/autoinit_preflight_setup.sh",
            "configs/autoinit/c2_artifacts.json",
            "configs/autoinit/c2_artifacts_failed.json",
            "configs/experiments/phase_c2/authorization.json"):
        assert rel in paths, rel


def test_the_closure_excludes_what_a_c2_session_does_not_run():
    """A closure is only useful if it is tight. C1's probe training, its
    isolation plan and Phase A's launcher are not on the C2 path, and a set
    that swept them in would move C2's digest for C1's reasons."""
    from experiments.phase_c2.session import c2_current_executable

    paths = {f["path"] for f in c2_current_executable(REPO)["files"]}
    for rel in ("scripts/pod/autoinit_c1_launch.py",
                "scripts/pod/autoinit_c1_driver.py",
                "scripts/pod/autoinit_phase_a_launch.py",
                "scripts/experiments/phase_c1/isolation.py",
                "scripts/experiments/phase_c1/scoring.py",
                "scripts/pod/start_job.py"):
        assert rel not in paths, rel


def test_the_derived_set_is_the_one_the_digest_covers():
    """The invariant C1 violated for two weeks: it declared 73 pre-migration
    paths while binding a digest over 97 real ones, and the only gate that reads
    the field was excluded from its candidate sweep."""
    from experiments.phase_c2.session import c2_current_executable

    live = c2_current_executable(REPO)
    from aadistill.governance.closure import digest_of

    assert digest_of(live["files"]) == live["digest"]
    assert len({f["path"] for f in live["files"]}) == live["n_files"]


def test_the_historical_declaration_is_kept_and_no_longer_current():
    """The 18-path set is preserved as the description of the superseded
    attempt-1 grant, and it is NOT what a session now executes."""
    from experiments.phase_c2.session import (
        C2_HARNESS_SOURCE_FILES_V1, c2_current_executable,
        c2_historical_harness_digest,
    )

    assert len(C2_HARNESS_SOURCE_FILES_V1) == 18
    hist = c2_historical_harness_digest(REPO)
    live = c2_current_executable(REPO)
    assert hist["n_files"] == 18 and live["n_files"] > 18
    assert hist["digest"] != live["digest"]


def test_the_superseded_grant_is_preserved_byte_for_byte():
    """It is historical evidence. Its own hash still verifies, and the identity
    it binds is no longer the one a session would run under — which is what
    makes attempt 1 superseded rather than merely old."""
    from aadistill.infrastructure.manifest import sha256_json
    from experiments.phase_c2.session import c2_current_executable

    rel = "logs/stages/stage-1/phase_c2/runs/attempt1/governance/grant.json"
    doc = json.loads((REPO / rel).read_text())
    body = {k: v for k, v in doc.items() if k != "grant_sha256"}
    assert doc["grant_sha256"] == sha256_json(body), (
        f"{rel} no longer matches its own hash; it must not be rewritten")
    assert doc["grant_sha256"] == (
        "d43013edf6a9d8242b4778c570c14f054d203685c41e418e4c3bf9f10f2982c9")

    bound = doc["bound_identities_the_issuer_must_reproduce"]
    assert bound["c2_harness_n_files"] == 18
    assert bound["c2_harness_digest"] != c2_current_executable(REPO)["digest"]


def test_the_pod_test_gate_probe_is_preserved_byte_for_byte():
    """The $0 measurement that motivated the preflight. Raw evidence."""
    rel = ("logs/stages/stage-1/phase_c2/runs/attempt1/analyses/"
           "pod_test_gate_probe.json")
    doc = json.loads((REPO / rel).read_text())
    assert doc["schema"] == "aadistill.autoinit.c2_pod_test_gate_probe/v1"
    assert doc["result"]["passed"] == 3976
    assert doc["result"]["skipped"] == 65
    assert doc["result"]["returncode"] == 0


# --- item 4: one readiness mechanism, two thin contracts --------------------

def test_the_c2_readiness_contract_is_a_view_of_the_generic_runtime():
    from aadistill.runtime import pod_environment as generic
    from experiments.phase_c2 import pod_environment as PE

    contract = PE.c2_record_contract(RUN_ID, STAGE_ID)
    assert isinstance(contract, generic.RecordContract)
    assert contract.schema == PE.SCHEMA != (
        "aadistill.autoinit.c1_pod_environment_verification/v1")
    assert contract.harness_field == "c2_harness_digest"
    assert callable(contract.harness_digest)
    assert contract.record_path == (
        f"logs/stages/stage-{STAGE_ID}/phase_c2/runs/{RUN_ID}/"
        "governance/readiness.json")


def test_the_c2_readiness_record_must_belong_to_a_run():
    from experiments.phase_c2 import pod_environment as PE

    with pytest.raises(PE.ReadinessError, match="belongs to a run"):
        PE.record_path_for(None)
    assert PE.permitted_post_sweep_paths(RUN_ID, STAGE_ID) == (
        PE.record_path_for(RUN_ID, STAGE_ID),)


def test_the_c2_readiness_groups_declare_zero_expected_skips():
    """Not an empty contract: `watched` carries the pod selection, so ANY skip
    inside it becomes an unexpected environment skip and the verdict FAILs."""
    from aadistill.runtime.pod_environment import evaluate_sweep
    from experiments.phase_c2.pod_environment import (
        C2_READINESS_GROUPS as G, POD_TEST_SELECTION,
    )

    assert G.expected_skips == {} and G.must_pass == {}
    assert G.staged_role_nodeid is None
    assert tuple(G.watched) == (POD_TEST_SELECTION,)

    nid = f"{POD_TEST_SELECTION}/test_c2_execution_preflight.py::test_x"
    assert evaluate_sweep({nid: "passed"}, groups=G)["verdict"] == "PASS"
    failed = evaluate_sweep({nid: "skipped"}, groups=G)
    assert failed["verdict"] == "FAIL"
    assert failed["unexpected_environment_skips"] == [nid]


def test_every_registered_experiment_drives_the_same_recorder():
    """One mechanism. The recorder holds no experiment's names.

    Asserted as a PROPERTY of the registry rather than as an exhaustive list:
    registering a new experiment is the supported way to add one, and a list
    that has to be edited for every addition tests the list rather than the
    mechanism. What must stay true is that EVERY entry resolves to a contract
    built outside this file, and that the recorder names none of them.
    """
    import record_pod_environment as REC

    assert {"phase_c1", "phase_c2"} <= set(REC.EXPERIMENTS)
    for experiment, (module, factory) in REC.EXPERIMENTS.items():
        assert module.startswith("experiments."), (experiment, module)
        assert factory, experiment
    #: The module docstring NAMES the strings it used to hardcode, because that
    #: is the change it is describing.
    code = code_without_prose(Path(REC.__file__))
    for leaked in ("c1_harness_digest", "autoinit-c1", "c1_harness_n_files",
                   "c1_readiness_pointer", "autoinit_c1_launch",
                   "renderer_parity_is_proved_by"):
        assert leaked not in code, (
            f"{leaked!r} is still hardcoded in the recorder; it belongs to an "
            "experiment's SweepContract")

    c1 = REC.sweep_contract("phase_c1", None, None)
    c2 = REC.sweep_contract("phase_c2", RUN_ID, STAGE_ID)
    #: C1 keeps every byte of its wire format: its records' self-hashes were
    #: computed over these exact strings.
    assert c1.record.schema == (
        "aadistill.autoinit.c1_pod_environment_verification/v1")
    assert c1.record.harness_field == "c1_harness_digest"
    assert c1.harness_n_files_field == "c1_harness_n_files"
    assert c1.session_id == "autoinit-c1"
    assert c1.launcher_module == "autoinit_c1_launch"
    assert c1.pointer_schema == "aadistill.autoinit.c1_readiness_pointer/v1"
    assert "renderer_parity_is_proved_by" in c1.extra_record_fields
    #: C2 has no pointer at all: its record is run-owned and nothing else names
    #: it, so there is no second file to drift.
    assert c2.pointer_path is None and c2.extra_record_fields == {}
    assert c2.experiment_id == "phase_c2"

    #: And baseline completion, whose whole reason for being a third entry is
    #: that it binds a DIFFERENT launcher, session, harness and schema. If any
    #: of these equalled C2's, one experiment's readiness record could satisfy
    #: the other's verifier.
    completion = REC.sweep_contract(
        "phase_c2_baseline_completion", RUN_ID, STAGE_ID)
    assert completion.experiment_id == "phase_c2_baseline_completion"
    assert completion.record.schema != c2.record.schema
    assert completion.launcher_module != c2.launcher_module
    assert completion.session_id != c2.session_id
    assert completion.harness_n_files_field != c2.harness_n_files_field
    assert completion.pointer_path is None and completion.extra_record_fields == {}


def test_the_recorder_assembles_a_c2_record_end_to_end(tmp_path):
    """The whole post-sweep path, executed.

    It is the part of the recorder that NO $0 check reached: everything after
    the simulator returns runs only when a sweep has completed, and the first
    real C2 sweep died there on an unbound name — `_evaluate_sweep`, dropped
    when the module's C1 imports were replaced by the contract. A crash after
    the suite has run is the cheapest possible version of that failure here and
    the most expensive one on a pod, where the sweep is the thing being paid
    for.

    `--from-existing` is the route: a JUnit report of the right SHAPE is
    trivially cheap to build, and the consumer does not care that no suite
    produced it.
    """
    import subprocess

    from experiments.phase_c2 import pod_environment as PE

    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite '
        'name="pytest" errors="0" failures="0" skipped="0" tests="2">'
        '<testcase classname="tests.c2_preflight.test_c2_execution_preflight" '
        'name="test_the_launcher_and_driver_modules_load" time="0.01"/>'
        '<testcase classname="tests.c2_preflight.test_c2_execution_preflight" '
        'name="test_the_session_declaration_validates" time="0.01"/>'
        "</testsuite></testsuites>")
    (tmp_path / "pytest.log").write_text("2 passed\n")

    out = REPO / PE.PHASE_DIAGNOSTIC_RECORD
    previous = out.read_bytes() if out.is_file() else None
    #: The DIRECTORY too, not just the file. The recorder creates the parent,
    #: and restoring only the file left an empty `analyses/` behind — which the
    #: log navigation renderer derives an area row from, so this test made a
    #: DIFFERENT test report `phase_c2/README.md is stale`. A test that dirties
    #: the tree it runs in is a test that fails somebody else's assertion.
    existed = out.parent.is_dir()
    try:
        done = subprocess.run(
            [sys.executable, "scripts/autoinit/record_pod_environment.py",
             "--experiment", "phase_c2", "--kind", "diagnostic",
             "--from-existing", "--junit", str(junit)],
            cwd=REPO, capture_output=True, text=True, timeout=900,
            env={**__import__("os").environ, "PYTHONPATH": "src"})
        assert done.returncode == 0, done.stdout + done.stderr
        record = json.loads(out.read_text())
    finally:
        if previous is None:
            out.unlink(missing_ok=True)
            if not existed and out.parent.is_dir():
                try:
                    out.parent.rmdir()
                except OSError:
                    pass          # something else put a file there; leave it
        else:
            out.write_bytes(previous)

    assert record["schema"] == PE.SCHEMA
    assert record["record_kind"] == "diagnostic"
    assert record["verdict"] == "PASS", record["problems"]
    assert record["counts"] == {"passed": 2, "skipped": 0, "failed": 0,
                                "error": 0}
    #: The keys the contract names, under the contract's own names.
    assert record["c2_harness_digest"] and record["c2_harness_n_files"]
    assert "c1_harness_digest" not in record
    assert record["setup_environment"]["SESSION_KIND"] == "c2"
    assert record["staging_contract_digest"]
    assert record["self_sha256"]
    from aadistill.runtime.pod_environment import self_hash

    assert self_hash(record) == record["self_sha256"]


def test_an_unknown_experiment_is_refused_by_name():
    import record_pod_environment as REC

    with pytest.raises(SystemExit, match="not declared"):
        REC.sweep_contract("phase_c9", None, None)
    #: And C2 without a run is refused with the experiment's own reason rather
    #: than a traceback.
    with pytest.raises(SystemExit, match="belongs to a run"):
        REC.sweep_contract("phase_c2", None, None)


def test_repoint_is_scoped_to_one_experiment(tmp_path):
    """The glob was `logs/stages/*/*/runs/*`, across every experiment. Harmless
    while one experiment had run-owned readiness records; wrong the moment a
    second did, because C2's newest record would become C1's pointer target."""
    import record_pod_environment as REC

    for experiment in ("phase_c1", "phase_c2"):
        d = (tmp_path / "logs/stages/stage-1" / experiment
             / "runs/attempt7/governance")
        d.mkdir(parents=True)
        (d / "readiness.json").write_text(json.dumps(
            {"self_sha256": experiment, "record_kind": "diagnostic",
             "verdict": "PASS", "swept_base_commit": "0" * 40}))

    c1 = REC.sweep_contract("phase_c1", None, None)
    (tmp_path / c1.pointer_path).parent.mkdir(parents=True, exist_ok=True)
    assert REC.repoint(c1, tmp_path) == 0
    pointer = json.loads((tmp_path / c1.pointer_path).read_text())
    assert pointer["record_self_sha256"] == "phase_c1", (
        "C1's pointer names another experiment's readiness record")
    assert "phase_c1" in pointer["record"] and "phase_c2" not in pointer["record"]

    #: And C2, having no pointer, writes nothing at all.
    c2 = REC.sweep_contract("phase_c2", RUN_ID, STAGE_ID)
    before = sorted(p.name for p in tmp_path.rglob("*.json"))
    assert REC.repoint(c2, tmp_path) == 0
    assert sorted(p.name for p in tmp_path.rglob("*.json")) == before


# --- item 5: the issuer ------------------------------------------------------

def candidate_grant(repo_root=REPO) -> dict:
    """A grant asserting exactly the identities this tree derives.

    Built from the live derivation rather than transcribed, so this fixture
    cannot be the reason a test passes: every identity is re-derived inside the
    builder and compared to what is written here.
    """
    from experiments.phase_c2.authorization_payload import live_identities

    #: No commit argument: `live_identities` takes none, because no commit is a
    #: verified identity. See the chronology test at the end of this section.
    live = live_identities(repo_root)
    return {
        "granted_by": "TEST FIXTURE — not a maintainer, not a grant",
        "covers": ("an ephemeral candidate used to drive the real pre-provider "
                   "gates at $0. It permits nothing and is never written to "
                   "logs/."),
        "explicitly_not_authorized": ["anything at all"],
        "approved_money": {"expected_usd": 7.1787, "soft_stop_usd": 14.4996,
                           "hard_cap_usd": 15.0446,
                           "price_basis_usd_per_hour": 1.09},
        #: All four keys: the issuer refuses a partial `one_use`, because none
        #: of these limits has a safe default and a permissive one is the shape
        #: this must not have.
        "one_use": {"issuances_permitted": 1,
                    "launch_attempts_permitted": 1,
                    "provider_resources_permitted": 3,
                    "one_billing_resource_at_a_time": True},
        "budget_context_at_approval": {"cumulative_spend_usd": 290.5174,
                                       "authorized_cap_usd": 320.0},
        "bound_identities_the_issuer_must_reproduce": {
            "_rule": "re-derived by the issuer; never trusted",
            **{k: v for k, v in live.items() if not k.startswith("_")},
        },
    }


def build(**over):
    from experiments.phase_c2.authorization_payload import (
        build_c2_authorization_payload,
    )

    kwargs = dict(grant=candidate_grant(), session_commit="0" * 40,
                  granted_utc="2026-01-01T00:00:00+00:00", run_id=RUN_ID,
                  stage_id=STAGE_ID, repo_root=REPO,
                  require_readiness_record=False,
                  grant_path="<test fixture>")
    kwargs.update(over)
    return build_c2_authorization_payload(**kwargs)


def test_the_payload_binds_the_live_derivation():
    from experiments.phase_c2.session import (
        C2_SESSION_CONTRACT, c2_current_executable, c2_plan_hash,
    )

    payload = build()
    bound = payload["bound"]
    live = c2_current_executable(REPO)
    assert bound["c2_harness_digest"] == live["digest"]
    assert bound["c2_harness_n_files"] == live["n_files"]
    assert payload["harness_source_files"] == [f["path"] for f in live["files"]]
    assert bound["c2_plan_hash"] == c2_plan_hash()
    assert bound["c2_session_contract_hash"] == C2_SESSION_CONTRACT.contract_hash
    assert payload["hard_cap_usd"] == 15.0446
    assert payload["expected_usd"] == 7.1787
    #: The two properties the type refuses to be talked out of.
    assert payload["allows_phase_a"] is False
    assert payload["allows_recovery_training"] is False
    assert payload["authorizes_c2_search1"] is True


def test_the_declared_set_is_the_set_the_digest_covers():
    from aadistill.governance.closure import digest_of, sha256_of

    payload = build()
    rows = [{"path": p, "sha256": sha256_of(REPO / p)}
            for p in payload["harness_source_files"]]
    assert digest_of(rows) == payload["harness_source_digest"], (
        "the authorization declares one file set and binds a digest over "
        "another; session_commit_gate re-digests the declared set and could "
        "never match")


def test_the_payload_round_trips_through_the_real_loader(tmp_path):
    from experiments.phase_c2.session import C2Authorization

    payload = build()
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    auth = C2Authorization.load(path)
    assert auth.hard_cap_usd == 15.0446
    assert auth.plan_hash == payload["bound"]["c2_plan_hash"]
    assert auth.science_plan_hash == payload["bound"]["c2_session_contract_hash"]
    assert auth.harness_source_digest == payload["harness_source_digest"]
    assert tuple(auth.harness_source_files) == tuple(
        payload["harness_source_files"])
    assert auth.authorizes_c2_search1 is True


#: Every genuine verified identity, one parametrisation each. `reviewed_commit`
#: was in this list until 2026-09-16; it is not an identity of the tree but a
#: point in its history, and requiring a grant to reproduce it made the chain
#: unsatisfiable — see `test_the_real_grant_then_readiness_then_issuance_chronology_works`.
@pytest.mark.parametrize("identity", [
    "c2_harness_digest", "c2_harness_n_files", "c2_plan_hash",
    "c2_session_contract_hash", "pricing_sha256", "baseline_spec_hash",
    "baseline_artifact_digest",
])
def test_an_identity_that_no_longer_reproduces_is_refused(identity):
    """Each one separately, because "it refuses something" is not the claim —
    the claim is that it refuses THIS binding, and a single check that only ever
    exercised one field would pass for a builder that ignored the rest.

    These are the identities that still bind: the executable closure and its
    file count, the plan hash, the session-contract hash, the pricing hash and
    B's spec hash and artifact digest. Each is a property of the TREE, so a
    grant written at any time before issuance can state it truthfully."""
    from experiments.phase_c2.authorization_payload import C2AuthorizationRefused

    grant = candidate_grant()
    grant["bound_identities_the_issuer_must_reproduce"][identity] = "f" * 64
    with pytest.raises(C2AuthorizationRefused, match="no longer reproduce"):
        build(grant=grant)


def test_an_identity_the_issuer_cannot_derive_is_refused():
    """The superseded grant's `c2_harness_set_version` is exactly this case: it
    names a mechanism that has been replaced. A binding nobody re-computes is
    not a binding, and a grant may not introduce one."""
    from experiments.phase_c2.authorization_payload import C2AuthorizationRefused

    grant = candidate_grant()
    grant["bound_identities_the_issuer_must_reproduce"][
        "c2_harness_set_version"] = 1
    with pytest.raises(C2AuthorizationRefused, match="does not derive"):
        build(grant=grant)


def test_the_superseded_attempt1_grant_is_refused_by_the_issuer():
    """Executed, not asserted. That grant was valid at the commit it binds, and
    three separate things now refuse it: an eighteen-path harness identity the
    derived closure does not reproduce, a `c2_harness_set_version` naming a
    mechanism that no longer exists, and a `reviewed_commit` inside the
    identities block, which is no longer a machine-verified identity at all."""
    from experiments.phase_c2.authorization_payload import C2AuthorizationRefused

    grant = json.loads((REPO / "logs/stages/stage-1/phase_c2/runs/attempt1/"
                        "governance/grant.json").read_text())
    bound = grant["bound_identities_the_issuer_must_reproduce"]
    #: Read from the historical document, which still carries all three.
    assert bound["reviewed_commit"].startswith("7d0a35af")
    assert bound["c2_harness_n_files"] == 18
    assert "c2_harness_set_version" in bound

    with pytest.raises(C2AuthorizationRefused) as refused:
        build(grant=grant, session_commit=bound["reviewed_commit"])
    #: The FIRST thing it hits is the block it may not contain, which is the
    #: refusal a reader of that grant needs to see.
    assert "reviewed_commit" in str(refused.value)


def test_the_real_grant_then_readiness_then_issuance_chronology_works():
    """THE chronology, which every fixture above flattens into one commit.

    The formal order has three distinct trees, and no two of them are the same
    commit:

        1. a maintainer reviews some tree and writes the grant
        2. the grant is committed                          -> commit G
        3. the launch_bound sweep runs on G and records it as swept_base
        4. only the readiness record is committed          -> commit R
        5. the authorization is issued against R           -> session_commit = R

    `reviewed_commit` was a verified identity derived as the issuer's own
    `session_commit`, i.e. R. A grant written at step 1 cannot state R: it does
    not exist, it is two commits in the future, and it depends on the hash of a
    record produced by sweeping the grant itself. Amending the grant at step 4
    to add R would change G, and the readiness record's `swept_base_commit`
    would no longer describe the tree — so there was no order in which the chain
    could be satisfied. The same impossibility C1 attempt 13 hit from the other
    direction, and it was in the contract rather than in the code.

    This asserts the repair on the smallest seam there is: the payload builder,
    called with a `session_commit` that is a REAL and DIFFERENT commit from the
    one the grant's prose names. No git plumbing, no fixture repository, no new
    framework — the builder is where the impossibility lived, so it is where the
    possibility has to be shown.
    """
    import subprocess

    from experiments.phase_c2.authorization_payload import (
        build_c2_authorization_payload,
    )

    def rev(spec: str) -> str:
        return subprocess.run(["git", "rev-parse", spec], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()

    #: Two real commits of this repository, standing in for "the tree the
    #: maintainer reviewed" and "the clean HEAD after the readiness record was
    #: committed". Real, so the test cannot be satisfied by a string that could
    #: never be a commit; different, which is the whole point.
    reviewed, issuance = rev("HEAD~1"), rev("HEAD")
    assert reviewed != issuance

    grant = candidate_grant()
    #: Step 1: the grant's own basis is PROSE, outside the identities block.
    grant["granted_by"] = (
        "TEST FIXTURE — standing in for a maintainer decision taken after "
        f"independent review of commit {reviewed}")
    bound = grant["bound_identities_the_issuer_must_reproduce"]
    assert "reviewed_commit" not in bound, (
        "the fixture is back to asserting a commit; the chronology this test "
        "describes would be unsatisfiable again")

    #: Step 5: issuance happens against a LATER commit, and the grant said
    #: nothing about it — because it could not have.
    payload = build_c2_authorization_payload(
        grant=grant, session_commit=issuance,
        granted_utc="2026-01-01T00:00:00+00:00", run_id=RUN_ID,
        stage_id=STAGE_ID, repo_root=REPO, require_readiness_record=False,
        grant_path="<test fixture>")

    #: It issued, and the tree identities — unchanged across all three commits —
    #: are what permitted it.
    from experiments.phase_c2.session import (
        C2_SESSION_CONTRACT, c2_current_executable, c2_plan_hash,
    )

    live = c2_current_executable(REPO)
    assert payload["bound"]["c2_harness_digest"] == live["digest"]
    assert payload["bound"]["c2_plan_hash"] == c2_plan_hash()
    assert payload["bound"]["c2_session_contract_hash"] == (
        C2_SESSION_CONTRACT.contract_hash)
    assert payload["hard_cap_usd"] == 15.0446

    #: The commit binding lives where it can: the authorization records the
    #: issuance HEAD, and NO grant field had to predict it.
    assert payload["authorized_session_commit"] == issuance
    assert payload["provenance_commit"] == issuance
    grant_text = json.dumps(payload["grant"])
    assert issuance not in grant_text, (
        "a grant field carries the issuance commit, so the grant would again "
        "have to be written after the readiness sweep it precedes")
    assert reviewed in grant_text, (
        "the reviewed commit is not recorded even as provenance; a reader "
        "cannot tell which tree the decision was taken against")
    #: And prose is all it is: nothing compares it to anything.
    assert reviewed != payload["authorized_session_commit"]


def test_the_chronology_does_not_weaken_the_identities_that_do_bind():
    """Removing a commit from the contract must not remove anything else.

    Asked as one statement over the whole set, because the parametrised test
    above proves each identity refuses individually and this proves the SET is
    still the one the config declares — a repair that quietly emptied
    `verified_identities` would satisfy every test above.
    """
    from experiments.phase_c2.authorization_payload import (
        live_identities, load_config,
    )

    declared = tuple(load_config(REPO)["grant_contract"]["verified_identities"])
    assert declared == ("c2_harness_digest", "c2_harness_n_files",
                        "c2_plan_hash", "c2_session_contract_hash",
                        "pricing_sha256", "baseline_spec_hash",
                        "baseline_artifact_digest")
    assert "reviewed_commit" not in declared

    #: Every declared identity is one the issuer actually derives, and the
    #: derivation offers nothing the contract does not verify.
    derivable = {k for k in live_identities(REPO) if not k.startswith("_")}
    assert derivable == set(declared), sorted(derivable ^ set(declared))
    #: And none of them is a commit-shaped value from this repository's history.
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    live = live_identities(REPO)
    assert head not in {str(v) for v in live.values()}


def test_a_grant_with_no_identities_block_is_refused():
    from experiments.phase_c2.authorization_payload import C2AuthorizationRefused

    grant = candidate_grant()
    del grant["bound_identities_the_issuer_must_reproduce"]
    with pytest.raises(C2AuthorizationRefused, match="carries no"):
        build(grant=grant)


def test_a_grant_whose_ceiling_would_exceed_the_cap_is_refused():
    from experiments.phase_c2.authorization_payload import C2AuthorizationRefused

    grant = candidate_grant()
    grant["budget_context_at_approval"]["cumulative_spend_usd"] = 319.0
    with pytest.raises(C2AuthorizationRefused, match="exceeds"):
        build(grant=grant)


def test_a_grant_naming_another_ceiling_is_refused():
    """C1's $15.1475 belongs to a different package and a different experiment,
    and the maintainer forbade substituting it."""
    from experiments.phase_c2.authorization_payload import C2AuthorizationRefused

    grant = candidate_grant()
    grant["approved_money"]["hard_cap_usd"] = 15.1475
    with pytest.raises(C2AuthorizationRefused, match="approved_money"):
        build(grant=grant)


def test_the_issuer_requires_a_launch_bound_readiness_record():
    """By default, not by option. A candidate may be built without one; an
    issuance may not."""
    import inspect

    from experiments.phase_c2 import authorization_payload as AP

    sig = inspect.signature(AP.build_c2_authorization_payload)
    assert sig.parameters["require_readiness_record"].default is True
    with pytest.raises(AP.C2AuthorizationRefused, match="readiness"):
        build(require_readiness_record=True)
    #: And the issuer never passes it.
    issuer = (REPO / "scripts/autoinit/issue_c2_authorization.py").read_text()
    assert "require_readiness_record" not in issuer


# --- item 6: generic transport, thin C2 wrapper ------------------------------

def test_the_transport_spec_derives_names_and_refuses_aliases():
    from aadistill.infrastructure.bundle_transport import BundleTransportError
    from experiments.phase_c2.bundle import (
        C2_TRANSPORT, canonical_bundle_name, canonical_repo_path,
        require_canonical_bundle_arg,
    )

    commit = "a6e8063bef05130693a6759df19fc2c0103ec7da"
    assert canonical_bundle_name(commit) == "aad_autoinit_a6e8063b.bundle"
    assert canonical_repo_path(commit) == "transfer/aad_autoinit_a6e8063b.bundle"
    assert require_canonical_bundle_arg(canonical_bundle_name(commit), commit)
    with pytest.raises(BundleTransportError, match="alias for nothing"):
        require_canonical_bundle_arg("c2", commit)
    with pytest.raises(BundleTransportError, match="not a hex commit"):
        canonical_bundle_name("not-a-commit")
    assert C2_TRANSPORT.relay_repo == "AlphaAvatar/aadistill-artifacts"


def test_the_generic_transport_carries_no_experiment():
    """C2 supplies the relay, the prefix and the label; the module names none of
    them, and it does not name C1 either except in the comments that explain
    why the older copy stays."""
    from aadistill.infrastructure import bundle_transport as BT

    #: The module docstring EXPLAINS why C1's older copy stays, so it names it.
    code = code_without_prose(Path(BT.__file__))
    for leaked in ("AlphaAvatar/aadistill-artifacts", "phase_c1", "phase_c2",
                   "c1_artifacts", "autoinit_c1"):
        assert leaked not in code, f"{leaked!r} leaked into generic transport"


def test_an_empty_executable_set_cannot_pass_the_round_trip(tmp_path):
    """An empty set digests to a constant and would match a record that also
    computed it over nothing — a gate that passes vacuously."""
    from aadistill.infrastructure import bundle_transport as BT
    from experiments.phase_c2.bundle import C2_TRANSPORT

    with pytest.raises(BT.BundleTransportError, match="vacuously"):
        BT.roundtrip(C2_TRANSPORT, session_commit="0" * 40,
                     local_bundle_sha256="x", authorization_bytes=b"",
                     authorization_path="x", expected_harness_digest="x",
                     harness_files=(), download=lambda *a: None,
                     workdir=tmp_path)


@pytest.fixture(scope="module")
def local_bundle(tmp_path_factory):
    """A real bundle of this repository's HEAD, built by the real builder."""
    import subprocess

    from experiments.phase_c2.bundle import C2_TRANSPORT, build_bundle

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    work = tmp_path_factory.mktemp("bundle")
    built = build_bundle(C2_TRANSPORT, REPO, commit,
                         work / C2_TRANSPORT.bundle_name(commit))
    return SimpleNamespace(commit=commit, built=built, work=work)


def test_the_round_trip_verifies_a_real_bundle_of_this_repository(
        local_bundle, tmp_path):
    """The whole gate, end to end, with the network replaced by a local copy.

    The authorization is a file that exists at HEAD — this test uses the
    superseded attempt-1 grant, because it needs any committed blob whose bytes
    it can compare, and the point of the check is the equality, not the file.
    """
    from experiments.phase_c2.bundle import c2_executable_set, roundtrip

    auth_rel = ("logs/stages/stage-1/phase_c2/runs/attempt1/governance/"
                "grant.json")
    digest, files = c2_executable_set(REPO)
    #: Only the files that exist AT HEAD can be digested from a checkout of it.
    #: Anything uncommitted is not in the bundle by construction, which is
    #: exactly what the gate should say — so the set is narrowed to HEAD's and
    #: the expected digest recomputed over it, rather than asserting a digest
    #: that a clean tree would satisfy and a dirty one would not.
    import subprocess

    from aadistill.governance.closure import digest_of

    tracked = []
    for rel in sorted(files):
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO,
                              capture_output=True)
        if blob.returncode == 0:
            import hashlib
            tracked.append({"path": rel,
                            "sha256": hashlib.sha256(blob.stdout).hexdigest()})
    expected = digest_of(tracked)

    def fake_download(repo_id, path_in_repo, dest_dir):
        assert repo_id == "AlphaAvatar/aadistill-artifacts"
        assert path_in_repo.endswith(local_bundle.built["canonical_name"])
        return local_bundle.built["path"]

    evidence = roundtrip(
        session_commit=local_bundle.commit,
        local_bundle_sha256=local_bundle.built["sha256"],
        authorization_bytes=(REPO / auth_rel).read_bytes(),
        authorization_path=auth_rel,
        expected_harness_digest=expected,
        harness_files=tuple(r["path"] for r in tracked),
        download=fake_download, workdir=tmp_path)

    assert evidence["ok"] is True
    assert evidence["roundtrip_head"] == local_bundle.commit
    assert evidence["authorization_matches"] is True
    assert evidence["roundtrip_harness_digest"] == expected
    assert evidence["n_harness_files"] == len(tracked)


def test_the_round_trip_refuses_the_wrong_authorization_bytes(
        local_bundle, tmp_path):
    from aadistill.infrastructure.bundle_transport import BundleTransportError
    from experiments.phase_c2.bundle import roundtrip

    auth_rel = ("logs/stages/stage-1/phase_c2/runs/attempt1/governance/"
                "grant.json")

    def fake_download(repo_id, path_in_repo, dest_dir):
        return local_bundle.built["path"]

    with pytest.raises(BundleTransportError, match="not the artifact"):
        roundtrip(session_commit=local_bundle.commit,
                  local_bundle_sha256=local_bundle.built["sha256"],
                  authorization_bytes=b"{}\n",
                  authorization_path=auth_rel,
                  expected_harness_digest="0" * 64,
                  harness_files=("AGENTS.md",),
                  download=fake_download, workdir=tmp_path)


def test_the_round_trip_refuses_bytes_that_are_not_the_staged_bundle(
        local_bundle, tmp_path):
    from aadistill.infrastructure.bundle_transport import BundleTransportError
    from experiments.phase_c2.bundle import roundtrip

    def fake_download(repo_id, path_in_repo, dest_dir):
        return local_bundle.built["path"]

    with pytest.raises(BundleTransportError, match="different bytes"):
        roundtrip(session_commit=local_bundle.commit,
                  local_bundle_sha256="0" * 64,
                  authorization_bytes=b"", authorization_path="AGENTS.md",
                  expected_harness_digest="0" * 64,
                  harness_files=("AGENTS.md",),
                  download=fake_download, workdir=tmp_path)


# --- item 7: the pre-provider gates -----------------------------------------

def test_every_gate_runs_before_a_provider_resource_exists(launcher):
    """Named, in order, so a gate cannot be added or dropped silently — and
    `run_session` runs the whole tuple before `create()`."""
    spec = launcher.spec(session_args(launcher)).validate()
    names = [getattr(g, "__name__", str(g)) for g in spec.precheck]
    assert names == [
        "session_commit_and_lineage",
        "c2_executable_gate",
        #: The run identity and the provider-resource COUNT the grant stated.
        "resource_scope_gate",
        #: The frozen-asset expectation the pod's setup will check, verified
        #: here first: attempt 2 discovered it at SETUP_RC=91 for $0.0552.
        "frozen_assets_gate",
        "storage_gate",
        "pricing_identity_gate",
        "plan_identity_gate",
        "pod_environment_gate",
        "bundle_staged_gate",
    ]


def test_the_executable_gate_refuses_an_artifact_declaring_another_set(launcher):
    """Every other check digests the set the ARTIFACT declares, so this is the
    only one that can catch an artifact declaring the wrong one."""
    from experiments.phase_c2.session import (
        C2_HARNESS_SOURCE_FILES_V1, c2_current_executable,
    )

    live = c2_current_executable(REPO)
    args = session_args(launcher)

    def ctx(**auth):
        return SimpleNamespace(args=args, evidence={},
                              auth=SimpleNamespace(**auth))

    ok, why = launcher.c2_executable_gate(ctx(
        harness_source_files=tuple(f["path"] for f in live["files"]),
        harness_source_digest=live["digest"]))
    assert ok, why

    ok, why = launcher.c2_executable_gate(ctx(
        harness_source_files=C2_HARNESS_SOURCE_FILES_V1,
        harness_source_digest=live["digest"]))
    assert not ok and "SUPERSEDED" in why

    ok, why = launcher.c2_executable_gate(ctx(
        harness_source_files=(), harness_source_digest=live["digest"]))
    assert not ok and "NO harness file set" in why

    ok, why = launcher.c2_executable_gate(ctx(
        harness_source_files=tuple(f["path"] for f in live["files"]),
        harness_source_digest="f" * 64))
    assert not ok and "does not match the live tree" in why


def test_the_readiness_gate_refuses_when_no_record_exists(launcher):
    args = session_args(launcher)
    ok, why = launcher.pod_environment_gate(
        SimpleNamespace(args=args, evidence={}, auth=SimpleNamespace()))
    assert not ok
    assert "does not exist" in why and "launch_bound" in why


def test_the_bundle_gate_refuses_an_alias_and_a_missing_record(launcher):
    args = session_args(launcher, bundle="c2")
    ok, why = launcher.bundle_staged_gate(
        SimpleNamespace(args=args, evidence={}, auth=SimpleNamespace()))
    assert not ok and "alias for nothing" in why

    commit = args.session_commit
    from experiments.phase_c2.bundle import canonical_bundle_name

    args2 = session_args(launcher, bundle=canonical_bundle_name(commit))
    ok, why = launcher.bundle_staged_gate(
        SimpleNamespace(args=args2, evidence={}, auth=SimpleNamespace()))
    assert not ok and "stage_c2_bundle.py" in why


def test_every_governance_artifact_is_owned_by_the_run(launcher):
    """No repository-level path for any of them. C1's authorization, readiness
    record and bundle record were each one root file that every attempt
    overwrote, and unwinding that took a pointer, a history file and a
    migration."""
    from experiments.phase_c2.session import C2_RUN_ROLES, c2_run_path

    prefix = f"logs/stages/stage-{STAGE_ID}/phase_c2/runs/{RUN_ID}/"
    for role in C2_RUN_ROLES:
        assert c2_run_path(RUN_ID, role, STAGE_ID).startswith(prefix)
    assert launcher.auth_path_for(RUN_ID) == (
        prefix + "governance/authorization.json")
    assert launcher.bundle_record_for(RUN_ID) == prefix + "governance/bundle.json"

    source = (REPO / "scripts/pod/autoinit_phase_c2_launch.py").read_text()
    assert "logs/budget/approvals" not in source, (
        "a repository-level authorization path is back in the C2 launcher")


def test_the_session_declaration_carries_the_run_scoped_authorization(launcher):
    args = session_args(launcher)
    spec = launcher.spec(args).validate()
    assert spec.authorization_path == launcher.auth_path_for(args.run_id)
    #: And the driver is told the same path, not a different one.
    ctx = SimpleNamespace(args=args, image_digest="x", price=1.09, spent_usd=0.0)
    plan = spec.budget.plan(price_per_hour=1.09, authorized_usd=15.0446)
    command = spec.driver_command(ctx, plan)
    assert f"--authorization-path {spec.authorization_path}" in command


# --- item 8: the science and the ceiling did not move ------------------------

def test_the_science_and_the_ceiling_are_unchanged():
    """The values the maintainer approved, asserted here so this session's
    engineering cannot have moved one of them unnoticed."""
    from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1
    from experiments.phase_c2 import baseline as B
    from experiments.phase_c2.search_space import (
        C2_ALLOWED_IMPLS, C2_PROFILE_IDS,
    )
    from experiments.phase_c2.session import (
        c2_budget_spec, c2_hard_ceiling_usd, c2_price_per_hour_usd,
    )

    assert sorted(C2_ALLOWED_IMPLS) == [
        "attention.activation_importance_v1", "depth.causal_kl_greedy_v1",
        "ffn.activation_importance_v0", "width.global_pca_v0"]
    assert C2_PROFILE_IDS == ("calib.domain_balanced@v1",
                              "calib.reasoning_heavy@v2")
    assert SCHEDULE_V1.width == 6 and SCHEDULE_V1.warmup_levels == 1
    assert PARETO_V1.qualified_id.startswith("beam.pareto_multi_objective@")
    assert B.B_SPEC_HASH.startswith("3a233a9017b3")
    assert B.B_ARTIFACT_DIGEST.startswith("53e30566c5f7")

    assert c2_hard_ceiling_usd(REPO) == 15.0446
    assert c2_price_per_hour_usd(REPO) == 1.09
    budget = c2_budget_spec(REPO)
    reserves = {r.name: r.minutes for r in budget.soft_stop_reserves}
    assert reserves["beam_composition_risk"] == 335.8
    assert reserves["baseline_rebuild_reserve"] == 27.665
    assert budget.artifact_recovery_reserve_minutes == 30.0
    base = next(p.minutes for p in budget.other_phases
                if p.name == "beam_search_depth_early")
    assert round(base + reserves["beam_composition_risk"], 3) == 635.96
