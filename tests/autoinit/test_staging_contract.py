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
from _pytest.outcomes import Skipped

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests/pod"))

from aadistill.runtime import staging_contract as sc
from session_specs import load_session_launcher, session_args  # noqa: E402

#: The C1 experiment's harness digest, injected into the generic verifier.
#: `verify_record` no longer imports an experiment package to find this out --
#: which harness a readiness record describes is the caller's fact.
def c1_digest(repo_root):
    from experiments.phase_c1.authorization import c1_harness_digest
    return c1_harness_digest(repo_root)["digest"]




@pytest.fixture(scope="module")
def c1_setup():
    mod = load_session_launcher("autoinit_c1_launch")
    return mod.spec(session_args(mod)).setup


@pytest.fixture(scope="module")
def contract(c1_setup):
    return sc.derive_contract(c1_setup, session_id="autoinit-c1")


# --- what the pod can see ---------------------------------------------------
#
# These two describe a machine's staged/hidden SPLIT, so each needs its own
# premise present before it can assert anything. Attempt 5 guarded both with
# `skipif(AAD_SYNTHETIC_HF_TOKEN)` — the SIMULATOR's flag — and died at the pod
# test gate for `$0.3150`: the pod is behaviourally identical to the simulation
# (the unstaged artifacts are not there) while carrying none of its markers, so
# a simulator-keyed guard skips where the assertion holds and runs where it
# cannot. Each now asks the filesystem the question it actually depends on, and
# they ask DIFFERENT questions — one coarse shared condition would re-create the
# same class of defect one level up.

#: The unstaged source roots the hidden-set assertion needs to inspect. Exactly
#: what attempt 4 tripped on: the generic default claimed every session stages
#: `corpus_v2`, and C1 stages neither of these.
UNSTAGED_ROLES = ("artifacts/stage3/corpus_v2", "artifacts/eval/battery_v2")

#: The four files C1's manifest stages into the checkpoint destination. The dev
#: box holds more; a correctly staged pod holds exactly these.
CHECKPOINT_DEST = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
CHECKPOINT_DECLARED = ("tokenizer.json", "tokenizer_config.json",
                       "chat_template.jinja", "config.json")


def test_an_artifact_c1_does_not_stage_is_invisible(contract):
    """`corpus_v2` and `eval/battery_v2` must be hidden where they exist.

    Premise: this machine holds them. A pod never received them and the
    simulation has already moved them aside, so in both the hidden set is
    legitimately empty and there is nothing to prove.
    """
    present, absent = sc.sources_on_disk(REPO, UNSTAGED_ROLES)
    if absent:
        pytest.skip(
            "premise absent: this machine holds no files under "
            + ", ".join(absent)
            + " — nothing can be proven hidden that was never here. That is the "
              "correct state of a pod and of the simulation, asked of the "
              "filesystem rather than of a simulator marker.")

    hidden = set(sc.hidden_files(contract, REPO))
    staged = sc.staged_files(contract, REPO)
    for role in present:
        assert any(p.startswith(role) for p in hidden), (
            f"{role} is on disk and not staged, so it must be hidden; a "
            "simulation that leaves it visible is more generous than the pod")
        assert not any(p.startswith(role) for p in staged), (
            f"{role} is marked staged, and C1's manifest does not stage it")


def test_an_undeclared_file_inside_a_staged_destination_stays_hidden(contract):
    """FILE granularity, and the reason it matters.

    A `RelayInput` stages ONE NAMED FILE into its destination, not the directory.
    C1 puts four files into the checkpoint dir; the dev box holds six. Modelling
    the destination as wholly present is how a sweep certifies a machine that has
    `model.safetensors` when the pod does not.

    Two distinct premises can be absent, and they mean different things: the
    destination may not exist at all, or it may exist holding exactly the four
    declared files — which is a correctly staged pod, and must skip rather than
    fail.
    """
    staged = sc.staged_files(contract, REPO)
    declared = {f"{CHECKPOINT_DEST}/{n}" for n in CHECKPOINT_DECLARED}
    # Manifest-derived and disk-independent, so it holds on every machine.
    assert declared <= staged, sorted(declared - staged)

    undeclared = sc.undeclared_in_destination(REPO, CHECKPOINT_DEST,
                                              CHECKPOINT_DECLARED)
    if undeclared is None:
        pytest.skip(f"premise absent: {CHECKPOINT_DEST} does not exist on this "
                    "machine, so it holds nothing to classify")
    if not undeclared:
        pytest.skip(
            f"premise absent: {CHECKPOINT_DEST} holds exactly the "
            f"{len(CHECKPOINT_DECLARED)} declared files and nothing undeclared. "
            "That is a correctly staged pod, and the simulation after hiding — "
            "the same property, not a marker.")

    hidden = set(sc.hidden_files(contract, REPO))
    for name in undeclared:
        p = f"{CHECKPOINT_DEST}/{name}"
        assert p not in staged, f"{p} is visible but C1 does not stage it"
        assert p in hidden, f"{p} is neither staged nor hidden"


def test_the_manifest_stages_neither_unstaged_role_anywhere(contract):
    """The manifest half of the claim above, with no premise at all.

    Portable by construction — it reads the contract, not the disk — so it runs
    and holds on the dev box, in the simulation AND on the pod. The hidden-set
    half cannot be checked where the files are absent; this half always can, and
    it is the half that would have caught attempt 4's generous default.
    """
    declared_dests = (
        [r["staged_path"] for r in contract["relay_inputs"] if r.get("staged")]
        + [a["staged_tree"] for a in contract["local_assets"]])
    for role in UNSTAGED_ROLES:
        for dest in declared_dests:
            assert not (dest == role or dest.startswith(role + "/")
                        or role.startswith(dest + "/")), (
                f"C1's manifest stages {dest}, which overlaps {role}")


# --- the three environments, each getting the right verdict from one question -
#
# Dev box: premises present, both RUN and PASS.
# Pod: correctly staged, premises absent, both SKIP.
# Simulation after hiding: the SAME condition as the pod, so the SAME skip — for
# the property, never because a simulator marker exists.

def _pod_like_root(tmp_path):
    """A correctly staged pod: the four declared files, and nothing else."""
    ck = tmp_path / CHECKPOINT_DEST
    ck.mkdir(parents=True)
    for n in CHECKPOINT_DECLARED:
        (ck / n).write_text("{}")
    return tmp_path


def test_the_dev_box_satisfies_both_premises_and_so_both_tests_run():
    """Environment 1: on the machine that HOLDS the artifacts, both must run.

    Guarded by the same property as the two tests it is about — otherwise it
    would be the third guard in this module to fail on a pod for asserting a
    dev-box fact, which is the defect being repaired.
    """
    present, absent = sc.sources_on_disk(REPO, UNSTAGED_ROLES)
    undeclared = sc.undeclared_in_destination(REPO, CHECKPOINT_DEST,
                                              CHECKPOINT_DECLARED)
    if absent or not undeclared:
        pytest.skip(
            "premise absent: this machine is a pod or the simulation — it holds "
            f"neither {', '.join(absent) or 'the unstaged roles'} nor an "
            f"undeclared file in {CHECKPOINT_DEST}, so there is no dev-box "
            "premise to confirm here.")
    assert present == list(UNSTAGED_ROLES)
    assert undeclared, ("the dev box no longer holds an undeclared file in "
                        f"{CHECKPOINT_DEST}")


def test_a_correctly_staged_pod_skips_both_for_the_property(tmp_path, contract,
                                                            monkeypatch):
    """Environment 2 — executed, not reasoned about.

    The two real test functions are called against a root in exactly the state a
    correctly staged pod is in. Both must raise Skipped. A FAIL here is attempt
    5 reproduced at `$0`.
    """
    root = _pod_like_root(tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "REPO", root)

    with pytest.raises(Skipped) as a:
        test_an_artifact_c1_does_not_stage_is_invisible(contract)
    with pytest.raises(Skipped) as b:
        test_an_undeclared_file_inside_a_staged_destination_stays_hidden(contract)

    for exc, expected in ((a, "holds no files under"),
                          (b, "holds exactly the 4 declared files")):
        reason = str(exc.value)
        assert "premise absent" in reason and expected in reason, reason

    # The two premises are DIFFERENT: the pod has the destination but not the
    # roles. One shared condition could not tell these apart.
    assert sc.sources_on_disk(root, UNSTAGED_ROLES)[1] == list(UNSTAGED_ROLES)
    assert sc.undeclared_in_destination(root, CHECKPOINT_DEST,
                                        CHECKPOINT_DECLARED) == set()


def test_the_simulation_after_hiding_is_the_same_condition_as_the_pod(tmp_path,
                                                                     contract,
                                                                     monkeypatch):
    """Environment 3, and the whole point.

    Build a dev-box-shaped root, then hide exactly what `hidden_files` would move
    aside. The premises must then read IDENTICALLY to the pod's — no marker is
    consulted, and none exists in this root."""
    root = tmp_path / "devbox"
    ck = root / CHECKPOINT_DEST
    ck.mkdir(parents=True)
    for n in CHECKPOINT_DECLARED:
        (ck / n).write_text("{}")
    (ck / "model.safetensors").write_bytes(b"\0")          # the undeclared extra
    for role in UNSTAGED_ROLES:
        (root / role).mkdir(parents=True)
        (root / role / "rows.jsonl").write_text("{}\n")

    before = (sc.sources_on_disk(root, UNSTAGED_ROLES),
              sc.undeclared_in_destination(root, CHECKPOINT_DEST,
                                           CHECKPOINT_DECLARED))
    assert before[0][1] == [] and before[1] == {"model.safetensors"}, (
        "the dev-box-shaped root does not satisfy the premises, so hiding "
        "proves nothing")

    hide = tmp_path / "hidden"
    hide.mkdir()
    (ck / "model.safetensors").rename(hide / "model.safetensors")
    for role in UNSTAGED_ROLES:                            # deepest-first prune
        for p in sorted((root / role).rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        (root / role).rmdir()

    pod = _pod_like_root(tmp_path / "pod")
    assert (sc.sources_on_disk(root, UNSTAGED_ROLES)
            == sc.sources_on_disk(pod, UNSTAGED_ROLES))
    assert (sc.undeclared_in_destination(root, CHECKPOINT_DEST, CHECKPOINT_DECLARED)
            == sc.undeclared_in_destination(pod, CHECKPOINT_DEST, CHECKPOINT_DECLARED))

    monkeypatch.setattr(sys.modules[__name__], "REPO", root)
    with pytest.raises(Skipped):
        test_an_artifact_c1_does_not_stage_is_invisible(contract)
    with pytest.raises(Skipped):
        test_an_undeclared_file_inside_a_staged_destination_stays_hidden(contract)


def test_an_absent_destination_is_a_different_premise_from_an_empty_one(tmp_path,
                                                                       contract,
                                                                       monkeypatch):
    """`None` vs `set()`. Collapsing them loses which machine you are on."""
    assert sc.undeclared_in_destination(tmp_path, CHECKPOINT_DEST,
                                        CHECKPOINT_DECLARED) is None
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    with pytest.raises(Skipped) as e:
        test_an_undeclared_file_inside_a_staged_destination_stays_hidden(contract)
    assert "does not exist on this machine" in str(e.value)


def test_these_staging_tests_never_consult_the_simulator_flag():
    """`AAD_SYNTHETIC_HF_TOKEN` describes a fake CREDENTIAL, which is a real
    property of the simulation — but not this one. It must not gate a staging
    assertion again."""
    src = Path(__file__).read_text()
    # Comments are allowed to NAME the defect; only executable lines are checked.
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    flag = "AAD_SYNTHETIC" + "_HF_TOKEN"          # split so this test is not a hit
    for pattern in (f'environ.get("{flag}")', f"environ.get('{flag}')",
                    f'environ["{flag}"]', f"getenv('{flag}')", f'getenv("{flag}")'):
        assert pattern not in code, (
            f"{pattern} is back: a staging premise keyed on the simulator marker. "
            "Ask the filesystem what it holds instead.")
    marker_decorator = "skip" + "if"              # split, for the same reason
    assert marker_decorator not in code, (
        "a module-level mark is back on these tests; each premise belongs "
        "inside the test that depends on it")
    # Both guards ask the contract module's filesystem helpers.
    assert src.count("sc.sources_on_disk(REPO") >= 1
    assert src.count("sc.undeclared_in_destination(REPO") >= 1


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
    #
    # Read from the SYNTAX TREE, not by slicing between two function names.
    # The slice covered everything written between them, so an unrelated helper
    # added later -- one whose `except FileExistsError` is how it allocates a
    # fresh sweep directory -- failed this. The subject is one function.
    import ast

    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "derive_c1_session")
    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
    assert not handlers, (
        "derive_c1_session swallows an error and would let the sweep fall back "
        "to the generic simulator default")


def test_a_launch_bound_record_without_a_staging_contract_is_refused():
    from experiments.phase_c1 import pod_environment as pe
    from experiments.phase_c1.authorization import c1_harness_digest

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
    from experiments.phase_c1 import pod_environment as pe
    from experiments.phase_c1.authorization import c1_harness_digest

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
    from experiments.phase_c1 import pod_environment as pe
    assert len(pe.HOST_LOCAL_C1_NODEIDS) == 3
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
    from experiments.phase_c1 import pod_environment as pe
    keys = set(pe.evaluate_sweep({}))
    for group in ("battery_source_skipped_as_expected",
                  "host_local_c1_skipped_as_expected",
                  "devbox_only_skipped_as_expected",
                  "expected_environment_skips",
                  "repository_state_all_passed"):
        assert group in keys, group
