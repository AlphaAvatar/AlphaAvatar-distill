"""The staged view a readiness sweep runs under, derived from the manifest.

C1 attempt 4 died at the pod CPU test gate for `$0.6986` with six failures, and
the sweep that had certified the same tree passed. It ran with
`simulate_pod_env.sh`'s generic default `HIDDEN_PATHS` — a hand-maintained
complement whose own comment claimed every pod session stages
`artifacts/stage3/corpus_v2`. C1 stages no such thing. The simulation was 55 tests
more generous than the pod: 49 extra skips plus 6 failures, exactly the pass delta.

The direction was the defect. A complement cannot be checked against anything, so
it drifts every time a session's staging changes. These pin the inversion.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests/pod"))

from aadistill.autoinit import staging_contract as sc  # noqa: E402
from session_specs import load_session_launcher, session_args  # noqa: E402


@pytest.fixture(scope="module")
def c1_setup():
    mod = load_session_launcher("autoinit_c1_launch")
    return mod.spec(session_args(mod)).setup


@pytest.fixture(scope="module")
def contract(c1_setup):
    return sc.derive_contract(c1_setup, session_id="autoinit-c1")


# --- what the pod can see ---------------------------------------------------

#: These two describe the DEV BOX's staged/hidden split, so they cannot run
#: inside the simulation that split produces: there, the unstaged artifacts have
#: already been moved aside, so `gitignored_files()` no longer lists them and the
#: hidden set is legitimately empty. The simulator sets this flag, and it is the
#: same one that tells the credential-dependent gates they hold a fake token.
in_simulation = pytest.mark.skipif(
    bool(os.environ.get("AAD_SYNTHETIC_HF_TOKEN")),
    reason="runs on the dev box: inside the pod simulation the unstaged files "
           "are already hidden, so there is nothing left to prove hidden")


@in_simulation
def test_an_artifact_c1_does_not_stage_is_invisible(contract):
    """`corpus_v2` and `eval/battery_v2` are exactly what attempt 4 tripped on."""
    hidden = set(sc.hidden_files(contract, REPO))
    staged = sc.staged_files(contract, REPO)
    for role in ("artifacts/stage3/corpus_v2", "artifacts/eval/battery_v2"):
        present = [p for p in hidden if p.startswith(role)]
        assert present, f"{role} is not hidden; the simulation would be generous"
        assert not any(p.startswith(role) for p in staged), (
            f"{role} is marked staged, and C1's manifest does not stage it")


@in_simulation
def test_an_undeclared_file_inside_a_staged_destination_stays_hidden(contract):
    """FILE granularity, and the reason it matters.

    A `RelayInput` stages ONE NAMED FILE into its destination, not the directory.
    C1 puts four files into the checkpoint dir; the dev box holds six. Modelling
    the destination as wholly present is how a sweep certifies a machine that has
    `model.safetensors` when the pod does not.
    """
    ck = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    staged = sc.staged_files(contract, REPO)
    hidden = set(sc.hidden_files(contract, REPO))
    declared = {f"{ck}/{n}" for n in ("tokenizer.json", "tokenizer_config.json",
                                      "chat_template.jinja", "config.json")}
    assert declared <= staged, sorted(declared - staged)
    on_disk = {f"{ck}/{p.name}" for p in (REPO / ck).iterdir() if p.is_file()}
    undeclared = on_disk - declared
    assert undeclared, "the dev box no longer holds an undeclared file here"
    for p in undeclared:
        assert p not in staged, f"{p} is visible but C1 does not stage it"
        assert p in hidden, f"{p} is neither staged nor hidden"


def test_a_local_asset_is_a_whole_tree_and_a_relay_input_is_one_file(contract):
    """The two staging kinds differ, and the contract must not flatten them."""
    staged = sc.staged_files(contract, REPO)
    tree = "artifacts/stage3/c1_confirmation_v1"
    assert sum(1 for p in staged if p.startswith(tree + "/")) > 1
    relay = [r for r in contract["relay_inputs"] if r.get("staged")]
    for r in relay:
        assert r["staged_path"].endswith(Path(r["path"]).name)


def test_install_to_alone_would_over_stage(c1_setup):
    """The bug this contract had for one draft: `install_to` is the PARENT.

    Reading it alone marks all of `artifacts/stage1` and `artifacts/stage3` as
    staged — which would have hidden nothing that matters and reproduced the very
    over-generous model being removed.
    """
    parents = {a.install_to for a in c1_setup.local_assets}
    assert parents == {"artifacts/stage1", "artifacts/stage3"}
    for a in c1_setup.local_assets:
        assert a.repo_path == f"{a.install_to}/{a.dest_name}", (
            "the staged tree is install_to/dest_name; if that stops holding, "
            "derive_contract's join is wrong")


# --- the digest moves when the staging does ---------------------------------

def test_removing_one_declared_staged_file_changes_the_contract(c1_setup):
    import copy
    before = sc.derive_contract(c1_setup)["digest"]
    trimmed = copy.copy(c1_setup)
    object.__setattr__(trimmed, "relay_inputs", tuple(c1_setup.relay_inputs[1:]))
    assert sc.derive_contract(trimmed)["digest"] != before


def test_changing_test_ignores_changes_the_contract(c1_setup):
    import copy
    before = sc.derive_contract(c1_setup)["digest"]
    changed = copy.copy(c1_setup)
    object.__setattr__(changed, "test_ignores",
                       tuple(c1_setup.test_ignores) + ("tests/other.py",))
    assert sc.derive_contract(changed)["digest"] != before


def test_changing_the_session_kind_changes_the_contract(c1_setup):
    import copy
    before = sc.derive_contract(c1_setup)["digest"]
    changed = copy.copy(c1_setup)
    object.__setattr__(changed, "env", {"SESSION_KIND": "not_c1"})
    assert sc.derive_contract(changed)["digest"] != before


def test_changing_a_local_asset_destination_changes_the_contract(c1_setup):
    import copy
    from aadistill.infrastructure.session import LocalAsset
    before = sc.derive_contract(c1_setup)["digest"]
    changed = copy.copy(c1_setup)
    first = c1_setup.local_assets[0]
    object.__setattr__(changed, "local_assets",
                       (LocalAsset(first.repo_path, first.dest_name,
                                   "artifacts/elsewhere"),)
                       + tuple(c1_setup.local_assets[1:]))
    assert sc.derive_contract(changed)["digest"] != before


# --- the generic default must not be able to produce a launch-bound record ---

def test_the_recorder_derives_and_never_falls_back(contract):
    """No default path. A sweep that cannot say what this session stages must
    not run at all, because that is exactly what attempt 4 did."""
    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    assert "derive_c1_session()" in src
    assert '"HIDDEN_PATHS"' in src and '"PODSIM_CMD"' in src
    assert "hidden_files(contract, REPO_ROOT)" in src
    # No except-and-continue around the derivation.
    head = src[src.index("def derive_c1_session"):
               src.index("def check_invocation_matches")]
    assert "except" not in head, (
        "derive_c1_session swallows an error and would let the sweep fall back "
        "to the generic simulator default")


def test_a_launch_bound_record_without_a_staging_contract_is_refused():
    from aadistill.autoinit import pod_environment as pe
    from aadistill.autoinit.c1_authorization import c1_harness_digest

    rec = {"schema": pe.SCHEMA, "swept_base_commit": pe.head_commit(REPO),
           "tree_clean": True,
           "c1_harness_digest": c1_harness_digest(REPO)["digest"],
           "pod_test_environment_digest": pe.pod_test_environment_digest(REPO)["digest"],
           "counts": {"passed": 1, "skipped": 0, "failed": 0, "error": 0},
           "verdict": "PASS", "record_kind": pe.LAUNCH_BOUND, "problems": []}
    rec["self_sha256"] = pe.self_hash(rec)
    ok, why = pe.verify_record(rec, REPO, required_kind=pe.LAUNCH_BOUND)
    assert not ok and "staging_contract_digest" in why


def test_a_record_swept_under_a_different_staging_contract_is_refused(contract):
    from aadistill.autoinit import pod_environment as pe
    from aadistill.autoinit.c1_authorization import c1_harness_digest

    rec = {"schema": pe.SCHEMA, "swept_base_commit": pe.head_commit(REPO),
           "tree_clean": True,
           "c1_harness_digest": c1_harness_digest(REPO)["digest"],
           "pod_test_environment_digest": pe.pod_test_environment_digest(REPO)["digest"],
           "counts": {"passed": 1, "skipped": 0, "failed": 0, "error": 0},
           "verdict": "PASS", "record_kind": pe.LAUNCH_BOUND, "problems": [],
           "staging_contract_digest": "0" * 64}
    rec["self_sha256"] = pe.self_hash(rec)
    ok, why = pe.verify_record(rec, REPO, required_kind=pe.LAUNCH_BOUND,
                               staging_contract_digest=contract["digest"])
    assert not ok and "staging contract" in why and "owed again" in why


def test_the_paid_gate_passes_the_live_staging_digest():
    src = (REPO / "scripts/pod/autoinit_c1_launch.py").read_text()
    assert "staging_contract_digest=live_staging" in src
    assert "derive_contract(spec(ctx.args).setup" in src


# --- the host-side isolation verifier must fail closed ----------------------

def test_an_absent_or_empty_role_fails_the_isolation_verifier(tmp_path):
    """`zero collisions` against a role that was never read is not evidence."""
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    import verify_c1_battery_isolation as v

    with pytest.raises(v.RoleUnavailable, match="does not exist"):
        v.role_identities("artifacts/does_not_exist_at_all", "jsonl_dir")

    # `REPO_ROOT / rel` yields rel itself when rel is absolute, so a tmp_path
    # outside the repo still exercises the real function.
    empty = tmp_path / "empty_role"
    empty.mkdir()
    with pytest.raises(v.RoleUnavailable, match="no \\*.jsonl"):
        v.role_identities(str(empty), "jsonl_dir")

    # Present, with a jsonl file, but no rows: still not evidence.
    blank = tmp_path / "blank_role"
    blank.mkdir()
    (blank / "a.jsonl").write_text("\n\n")
    with pytest.raises(v.RoleUnavailable, match="zero rows"):
        v.role_identities(str(blank), "jsonl_dir")


def test_the_verifier_declares_every_role_and_cross_checks_frozen_counts():
    sys.path.insert(0, str(REPO / "scripts/autoinit"))
    import verify_c1_battery_isolation as v

    assert set(v.EXPECTED) == set(v.ROLES), (
        "a role has no frozen expectation, so a substituted nonempty asset would "
        "satisfy it")
    manifest = json.loads(
        (REPO / "artifacts/stage3/c1_confirmation_v1/manifest.json").read_text())
    for role, (key, rows_field, _ids) in v.EXPECTED.items():
        assert key in manifest["isolation"], role
        assert rows_field in manifest["isolation"][key], (role, rows_field)


# --- the environment the child process actually runs under -------------------
#
# Attempt 4's record hashed the SetupManifest and then launched pytest under
# nothing but PODSIM variables. The contract was hashed but never REALIZED: the
# child never saw SESSION_KIND=c1, so any test keying on it behaved as though it
# were some other session. These pin the realization, not the declaration.

def _c1_session():
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts/autoinit"))
    from record_pod_environment import derive_c1_session
    return derive_c1_session()


def test_the_setup_environment_comes_from_the_production_method():
    """`SessionSpec.setup_environment`, not a reconstructed subset."""
    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    assert "spec.setup_environment(session_commit=" in src
    assert "**setup_env," in src, "the production env is not merged into the child"
    spec, contract, view, env = _c1_session()
    for key in ("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                "SESSION_AUTH_PATH", "SESSION_PLAN_HASH", "SESSION_ASSETS",
                "SESSION_RELAY_INPUTS", "SESSION_TEST_IGNORES", "UV_MAX_S",
                "TESTS_MAX_S", "TEACHER_REVISION", "SESSION_KIND"):
        assert key in env, key
    assert env["SESSION_KIND"] == "c1"


def test_the_simulated_environment_equals_the_production_session_values():
    """SESSION_TEST_IGNORES, SESSION_ASSETS and SESSION_RELAY_INPUTS observed in
    simulation must be the SessionSpec's own values."""
    spec, contract, view, env = _c1_session()
    assert env["SESSION_TEST_IGNORES"] == spec.setup.test_ignores_env()
    assert env["SESSION_ASSETS"] == spec.setup.assets_env()
    assert env["SESSION_RELAY_INPUTS"] == spec.setup.relay_env()
    assert env["SESSION_PLAN_HASH"] == spec.plan_hash
    assert env["SESSION_AUTH_PATH"] == spec.authorization_path


def test_changing_session_kind_moves_the_digest_and_the_child_environment(tmp_path):
    """Both halves, because attempt 4 had the first without the second."""
    import copy
    import subprocess as sp
    spec, contract, view, env = _c1_session()

    changed = copy.copy(spec.setup)
    object.__setattr__(changed, "env", {"SESSION_KIND": "not_c1"})
    assert sc.derive_contract(changed)["digest"] != contract["digest"]

    probe = "import os,sys; sys.stdout.write(os.environ.get('SESSION_KIND','<unset>'))"
    seen = sp.run([sys.executable, "-c", probe], capture_output=True, text=True,
                  env={**os.environ, **env}).stdout
    assert seen == "c1", f"a child process observed SESSION_KIND={seen!r}"
    other = sp.run([sys.executable, "-c", probe], capture_output=True, text=True,
                   env={**os.environ, **env, "SESSION_KIND": "not_c1"}).stdout
    assert other == "not_c1", "the probe cannot see the variable at all"


def test_the_recorder_records_the_command_it_ran_not_a_transcription():
    """A record that restates its command cannot be checked against its JUnit.
    Attempt 4's said two ignores while the sweep ran a different selection."""
    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    assert '"pytest_command": pytest_cmd,' in src
    assert '"--ignore=tests/data/test_recovery_corpus_pipeline.py "' not in src, (
        "a hand-transcribed pytest command is back in the record")
    assert "the simulator's default HIDDEN_PATHS" not in src, (
        "the record still describes the generic default it no longer uses")


# --- item 8: hashed-but-not-realized must be impossible ----------------------

def test_a_mismatched_invocation_refuses_before_a_pass_record_exists():
    """THE attempt-4 failure mode, as a test.

    Each of the three facts a manifest declares -- test selection, environment,
    staging -- is compared against what was handed to the subprocess, and any
    disagreement becomes a problem that forces the verdict to FAIL.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts/autoinit"))
    from record_pod_environment import check_invocation_matches

    spec, contract, view, env = _c1_session()
    cmd = (".venv/bin/python -m pytest tests/ -q "
           + " ".join(f"--ignore={i}" for i in contract["test_ignores"]))
    good = {**env, "HIDDEN_PATHS": "a\nb", "PODSIM_CMD": cmd}
    assert check_invocation_matches(contract, env, cmd, good)["problems"] == []

    # 1. test selection differs from the manifest
    fewer = ".venv/bin/python -m pytest tests/ -q --ignore=tests/only_one.py"
    r = check_invocation_matches(contract, env, fewer,
                                 {**good, "PODSIM_CMD": fewer})
    assert any("test selection mismatch" in p for p in r["problems"])

    # 2. the environment is not realized in the child
    r = check_invocation_matches(contract, env, cmd,
                                 {**good, "SESSION_KIND": "phase_a"})
    assert any("setup environment mismatch" in p for p in r["problems"])

    # 3. no derived staged view was passed -- the attempt-4 fallback
    no_hidden = {k: v for k, v in good.items() if k != "HIDDEN_PATHS"}
    r = check_invocation_matches(contract, env, cmd, no_hidden)
    assert any("generic default" in p for p in r["problems"])

    # and the recorder turns any of those into a FAIL verdict
    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    assert 'if realization["problems"]:' in src
    assert 'record["verdict"] = "FAIL"' in src


def test_the_host_local_cases_are_named_separately_from_the_source_skips():
    """They skip for a different reason and must not be folded into that count."""
    from aadistill.autoinit import pod_environment as pe
    assert len(pe.HOST_LOCAL_C1_NODEIDS) == 2
    assert not (set(pe.HOST_LOCAL_C1_NODEIDS)
                & (set(pe.RENDERER_PARITY_NODEIDS) | set(pe.BATTERY_SOURCE_NODEIDS)
                   | set(pe.DEVBOX_ONLY_NODEIDS)))


def test_c1_ignores_exactly_the_four_whole_modules():
    spec, contract, view, env = _c1_session()
    assert list(contract["test_ignores"]) == [
        "tests/data/test_recovery_corpus_pipeline.py",
        "tests/pod/test_phase_a_stages1_5_execute.py",
        "tests/autoinit/test_phase_b_reuse_hostlocal.py",
        "tests/autoinit/test_stage1_import.py",
    ]


def test_the_record_embeds_the_whole_findings_block():
    """Cherry-picking findings into the record is how the battery, host-local and
    dev-box skip groups came to be computed but never written: a new group had to
    be remembered in two places and the second was forgotten."""
    src = (REPO / "scripts/autoinit/record_pod_environment.py").read_text()
    assert '"findings": findings,' in src
    from aadistill.autoinit import pod_environment as pe
    keys = set(pe.evaluate_sweep({}))
    for group in ("battery_source_skipped_as_expected",
                  "host_local_c1_skipped_as_expected",
                  "devbox_only_skipped_as_expected",
                  "expected_environment_skips",
                  "repository_state_all_passed"):
        assert group in keys, group
