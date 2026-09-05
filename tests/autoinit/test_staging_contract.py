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
    assert "derive_c1_staging()" in src
    assert '"HIDDEN_PATHS"' in src and '"PODSIM_CMD"' in src
    assert "hidden_files(contract, REPO_ROOT)" in src
    # No except-and-continue around the derivation.
    head = src[src.index("def derive_c1_staging"):src.index("def main(")]
    assert "except" not in head, (
        "derive_c1_staging swallows an error and would let the sweep fall back "
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
