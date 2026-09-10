"""The launchers that will actually run next use one run identity.

A convention that only tests obey is not a convention. `RunLayout` and
`build_run_manifest` existed for a while with **no production caller at all**:
every C1 attempt wrote its session record to the flat `logs/autoinit_c1_session.json`,
which the next attempt overwrote, and `logs/autoinit_c1_attempt9/` was assembled
by hand afterwards. Meanwhile the three CUDA stage-F subruns wrote real
directories under `logs/runs/` and recorded no manifest, so `logs/runs/index.json`
reported `runs_current: 0` with three runs sitting on disk.

So these tests execute the real functions — `open_c1_run`, `close_c1_run`,
`Engineering.write_evidence` — against a temporary repository root, rather than
asserting on their source. `write_evidence` in particular had no coverage at
all, and it is the function whose last defect was found by a dry run because it
is only reached by a *completed* session.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from experiments.run_layout import RUNS_ROOT, RunConventionError, read_run
from session_specs import load_session_launcher

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def L():
    return load_session_launcher("autoinit_c1_launch")


def _args(tmp_path, run_id="attempt10", **over):
    scr = tmp_path / "scr"
    scr.mkdir(exist_ok=True)
    return types.SimpleNamespace(
        run_id=run_id, scr=str(scr), session_commit="a" * 40,
        bundle="aad_autoinit_aaaaaaaa.bundle", **over)


def _fake_repo(tmp_path, L, *, governance=True):
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True, exist_ok=True)
    #: The real pricing record, because `build_parser` derives `--max-price`
    #: from it rather than carrying a second copy of the rate.
    (repo / L.PRICING).write_bytes((REPO / L.PRICING).read_bytes())
    if governance:
        (repo / L.AUTH_PATH).write_text('{"authorization_id": "test"}')
        (repo / L.BUNDLE_RECORD).write_text('{"bundle": "aad_test"}')
    return repo


def _write_session(repo, args, **over):
    """What `SessionRunner.save()` leaves at `args.out`."""
    body = {"session_id": "autoinit-c1", "session_plan_hash": "ph",
            "harness_source_digest": "hd", "passed": False,
            "terminal": "C1_INCOMPLETE", "pod_id": "podabc",
            "cost": {"actual_usd": 1.044}, "provider_confirms_gone": True}
    body.update(over)
    path = repo / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


# --- the command line ------------------------------------------------------

def test_the_launcher_requires_a_run_id(L):
    with pytest.raises(SystemExit):
        L.build_parser().parse_args(
            ["--scr", "/tmp/x", "--session-commit", "0" * 40,
             "--bundle", "aad_test.bundle"])


def test_there_is_no_out_flag_to_point_somewhere_else(L):
    """`--out` is how nine attempts shared one path. It is gone, not defaulted."""
    flags = {o for a in L.build_parser()._actions for o in (a.option_strings or ())}
    assert "--run-id" in flags
    assert "--out" not in flags


def test_the_parser_still_produces_every_attribute_the_runner_reads(L):
    """`out` is DERIVED by `--run-id`, not filled in afterwards.

    `SessionRunner` reads `args.out`. Device-canary attempt 1 died at `$0.0603`
    on an attribute a hand-written namespace had and the real parser did not,
    after the pod was created and billing, so the parser's namespace must be
    complete on its own — `missing_arguments` is asked here as well as in
    `test_device_canary_argument_contract`, because that is the property this
    change could have broken.
    """
    from aadistill.infrastructure.session import missing_arguments
    from session_specs import session_args

    args = session_args(L)
    assert not missing_arguments(args)
    assert args.out == f"{RUNS_ROOT}/phase_c1/{args.run_id}/runtime/session.json"


def test_the_parser_and_the_open_derive_the_same_path(tmp_path, L):
    """One rule. Two derivations of one path is how they drift apart."""
    repo = _fake_repo(tmp_path, L)
    parsed = L.build_parser().parse_args(
        ["--scr", str(tmp_path / "scr"), "--session-commit", "a" * 40,
         "--bundle", "aad_autoinit_aaaaaaaa.bundle", "--run-id", "attempt10"])
    from_parser = parsed.out
    L.open_c1_run(parsed, repo)
    assert parsed.out == from_parser
    assert (repo / parsed.out).parent.is_dir()


# --- opening the run -------------------------------------------------------

def test_the_session_record_is_written_inside_the_run(tmp_path, L):
    repo = _fake_repo(tmp_path, L)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    assert args.out == f"{RUNS_ROOT}/phase_c1/attempt10/runtime/session.json"
    assert layout.rel_root == "phase_c1/attempt10"
    #: The parent exists, which `SessionRunner.save()` does not create.
    assert (repo / args.out).parent.is_dir()


def test_the_one_use_governance_artifacts_are_snapshotted_at_open(tmp_path, L):
    """Snapshotted BEFORE the session, while they still describe this attempt.

    The live authorization and bundle record are rewritten by the next issuance,
    so a copy taken afterwards can describe a different attempt entirely.
    """
    repo = _fake_repo(tmp_path, L)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    assert json.loads(layout.path("governance/authorization.json").read_text()
                      )["authorization_id"] == "test"
    (repo / L.AUTH_PATH).write_text('{"authorization_id": "the NEXT attempt"}')
    assert json.loads(layout.path("governance/authorization.json").read_text()
                      )["authorization_id"] == "test"


def test_an_absent_governance_artifact_does_not_stop_the_run(tmp_path, L):
    """A pre-issuance dry run has no authorization file and must still open."""
    repo = _fake_repo(tmp_path, L, governance=False)
    layout = L.open_c1_run(_args(tmp_path), repo)
    assert not layout.path("governance/authorization.json").exists()


def test_a_run_id_that_collides_is_refused(tmp_path, L):
    repo = _fake_repo(tmp_path, L)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args)
    L.close_c1_run(layout, args, repo)
    with pytest.raises(RunConventionError):
        L.open_c1_run(_args(tmp_path), repo)


def test_the_refusal_happens_before_the_session_spec_is_built(tmp_path, L,
                                                              monkeypatch):
    """`$0`, and before any provider call.

    Executed, not read off the source: `main` is run with `spec` and
    `run_session` replaced by recorders, and the collision must raise before
    either is reached.
    """
    repo = _fake_repo(tmp_path, L)
    monkeypatch.setattr(L, "REPO_ROOT", repo)
    calls = []
    monkeypatch.setattr(L, "spec", lambda a: calls.append("spec"))
    monkeypatch.setattr(L, "run_session", lambda *a, **k: calls.append("run"))
    argv = ["--scr", str(tmp_path / "scr"), "--session-commit", "a" * 40,
            "--bundle", "aad_autoinit_aaaaaaaa.bundle", "--run-id", "attempt10"]
    monkeypatch.setattr("sys.argv", ["autoinit_c1_launch.py", *argv])

    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args)
    L.close_c1_run(layout, args, repo)

    with pytest.raises(RunConventionError):
        L.main()
    assert calls == [], f"work happened before the collision was caught: {calls}"


# --- closing the run -------------------------------------------------------

def test_a_completed_session_records_every_role_it_produced(tmp_path, L):
    repo = _fake_repo(tmp_path, L)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args, passed=True, terminal="ALL_DONE")
    scr = Path(args.scr)
    (scr / "launch.log").write_text("line\n")
    (scr / "watchdog.jsonl").write_text("{}\n")
    (scr / "relay").mkdir()
    #: Named the way the relay names them, from the spec, so this fixture
    #: cannot drift away from what a real session leaves.
    for src, _role in L._RUN_COLLECT:
        if src.startswith("relay/"):
            (scr / src).write_text("{}\n")
    (scr / "store").mkdir()
    (scr / "store" / "manifest.json").write_text("{}")

    doc = L.close_c1_run(layout, args, repo)
    assert set(doc["roles"]) == {
        "session_record", "launcher_log", "watchdog_journal", "authorization",
        "bundle_record", "driver_evidence", "driver_log", "driver_status",
        "artifact_manifest"}
    assert layout.path("evidence/driver_status.txt").is_file()
    assert doc["status"]["terminal"] == "ALL_DONE"
    assert doc["status"]["passed"] is True
    assert doc["status"]["cost"]["actual_usd"] == 1.044
    assert doc["authorizes"] == "nothing"
    assert read_run(repo, "phase_c1", "attempt10")["self_sha256"] == doc["self_sha256"]


def test_a_setup_abort_records_the_session_record_alone(tmp_path, L):
    """No driver evidence, no artifact manifest — six C1 attempts ended here."""
    repo = _fake_repo(tmp_path, L, governance=False)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args, terminal=None, pod_id="")
    doc = L.close_c1_run(layout, args, repo)
    assert list(doc["roles"]) == ["session_record"]
    assert doc["status"]["pod_id"] is None
    assert doc["status"]["passed"] is False


def test_the_manifest_references_the_scratch_root_rather_than_copying_it(
        tmp_path, L):
    """Large artifacts stay outside git; the manifest carries the reference."""
    repo = _fake_repo(tmp_path, L)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args)
    scr = Path(args.scr)
    (scr / "store").mkdir()
    (scr / "store" / "manifest.json").write_text("{}")
    (scr / "store" / "c1_artifacts.tar.gz").write_bytes(b"\x00" * 4096)

    doc = L.close_c1_run(layout, args, repo)
    assert doc["plan"]["scratch_root"] == str(scr)
    copied = [p for p in layout.root.rglob("*") if p.suffix == ".gz"]
    assert not copied, f"an archive was copied into the repository: {copied}"


def test_what_is_collected_is_what_the_relay_actually_leaves(L):
    """Asked of the built spec, not of the three names written in the launcher.

    `LogRelay` names each local copy `Path(remote).name` and puts it under
    `<scr>/relay/`. The collection list is a second statement of those names,
    and it runs once — after teardown, where a name that stopped matching loses
    the evidence silently rather than failing. So the names are compared to the
    spec the runner is handed.
    """
    from session_specs import session_args

    spec = L.spec(session_args(L))
    relayed = {Path(spec.run_log_path).name, Path(spec.status_path).name,
               spec.artifacts.evidence_filename}
    collected = {src.split("/", 1)[1] for src, _ in L._RUN_COLLECT
                 if src.startswith("relay/")}
    assert collected == relayed, (
        f"the run collects {collected} from relay/, the relay writes {relayed}")


def test_a_recording_failure_is_loud_and_does_not_overwrite_the_session_result(
        tmp_path, L, monkeypatch, capsys):
    """A session that passed and could not record itself must not exit 0.

    And a session that already failed keeps its own code: the pod outcome is
    what an operator acts on, and replacing it with a bookkeeping code would
    hide the thing that actually happened.
    """
    repo = _fake_repo(tmp_path, L)
    monkeypatch.setattr(L, "REPO_ROOT", repo)
    monkeypatch.setattr(L, "spec", lambda a: None)
    monkeypatch.setattr(L, "close_c1_run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    for session_rc, expected in ((0, L.RUN_NOT_RECORDED), (11, 11)):
        monkeypatch.setattr(L, "run_session", lambda *a, rc=session_rc, **k: rc)
        argv = ["--scr", str(tmp_path / "scr"), "--session-commit", "a" * 40,
                "--bundle", "aad_autoinit_aaaaaaaa.bundle",
                "--run-id", f"attempt_rc{session_rc}"]
        monkeypatch.setattr("sys.argv", ["autoinit_c1_launch.py", *argv])
        assert L.main() == expected
        assert "RUN NOT RECORDED" in capsys.readouterr().out


def test_the_status_names_the_key_the_runner_actually_writes(tmp_path, L):
    """`terminal`, not `terminal_marker`.

    A key the session record does not have reads as `None`, which is
    indistinguishable from a session that produced no marker at all.
    """
    repo = _fake_repo(tmp_path, L, governance=False)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args, terminal="C1_REPLAY_MISMATCH")
    assert L.close_c1_run(layout, args, repo)["status"][
        "terminal"] == "C1_REPLAY_MISMATCH"


# --- the second consumer: a different experiment, different roles ----------

def test_the_cuda_validation_records_its_run_too(tmp_path):
    """The real `write_evidence`, executed. It had no test at all.

    Built with `object.__new__` because the constructor needs a provider API key
    and a CLI on disk; the body under test reads only `self.a`, `self.scr` and
    `self.ev`, and it is the body that matters.
    """
    import importlib.util

    path = REPO / "scripts/validation/cuda_engineering_launch.py"
    spec = importlib.util.spec_from_file_location("cuda_engineering_launch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    repo, scr = tmp_path / "repo", tmp_path / "scr"
    (repo / "logs").mkdir(parents=True)
    (scr / "artifacts" / "cuda_engineering").mkdir(parents=True)
    (scr / "artifacts" / "cuda_engineering" / "suffix_evidence.json").write_text("{}")
    (scr / "validation_stdout.txt").write_text("ok\n")

    eng = object.__new__(mod.Engineering)
    eng.a = types.SimpleNamespace(run_id="cuda_stage_f_20260911_s1",
                                  execution_sha="a" * 40, image="img:tag")
    eng.scr = scr
    eng.ev = {"verdict": "CUDA ENGINEERING VALIDATION PASS", "pod_id": "p1",
              "subrun_cost_usd": 0.0182, "campaign_cost_after_usd": 0.04}
    eng.write_evidence(repo)

    doc = read_run(repo, "cuda_stage_f", "cuda_stage_f_20260911_s1")
    assert set(doc["roles"]) == {"evidence", "validation_stdout", "artifacts"}
    assert doc["status"]["verdict"] == "CUDA ENGINEERING VALIDATION PASS"
    assert doc["status"]["subrun_cost_usd"] == 0.0182
    assert doc["artifact_spec"] == "cuda_engineering_run_v1"
    #: A different vocabulary from C1's, through the same functions.
    assert "session_record" not in doc["roles"]


def test_the_two_consumers_share_no_role_name():
    """If they ever converge, the convention has grown an experiment's opinion."""
    import importlib.util

    L = load_session_launcher("autoinit_c1_launch")
    path = REPO / "scripts/validation/cuda_engineering_launch.py"
    spec = importlib.util.spec_from_file_location("cuda_engineering_launch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(L.C1_RUN_ROLES) & set(mod.RUN_ROLES) == set()
    assert L.RUN_EXPERIMENT_ID != mod.RUN_EXPERIMENT_ID


# --- the index sees both, and says so when it cannot -----------------------

def test_the_index_finds_a_recorded_run_and_reports_an_unrecorded_one(tmp_path,
                                                                      L):
    import importlib.util

    path = REPO / "scripts/architecture/record_run_index.py"
    spec = importlib.util.spec_from_file_location("record_run_index", path)
    ri = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ri)

    repo = _fake_repo(tmp_path, L)
    args = _args(tmp_path)
    layout = L.open_c1_run(args, repo)
    _write_session(repo, args)
    L.close_c1_run(layout, args, repo)

    orphan = repo / RUNS_ROOT / "phase_c1" / "attempt11"
    (orphan / "runtime").mkdir(parents=True)
    (orphan / "runtime" / "session.json").write_text("{}")

    found = ri.discover_v3(repo)
    assert [(r["experiment_id"], r["run_id"]) for r in found] == [
        ("phase_c1", "attempt10")]
    orphans = ri.discover_unrecorded(repo)
    assert [(o["experiment_id"], o["run_id"]) for o in orphans] == [
        ("phase_c1", "attempt11")]
    assert orphans[0]["n_files"] == 1 and orphans[0]["digest"]


def test_an_empty_run_directory_is_not_reported_as_an_orphan(tmp_path):
    import importlib.util

    path = REPO / "scripts/architecture/record_run_index.py"
    spec = importlib.util.spec_from_file_location("record_run_index", path)
    ri = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ri)
    (tmp_path / RUNS_ROOT / "phase_c1" / "attempt12").mkdir(parents=True)
    assert ri.discover_unrecorded(tmp_path) == []


def test_this_repositorys_index_still_accounts_for_every_run_on_disk():
    """Against the real tree. The committed index must not silently drop a run.

    `logs/runs/index.json` read `runs_current: 0` while three CUDA stage-F
    subruns existed under `logs/runs/`, because a directory without a manifest
    matched neither discovery rule. Whatever is on disk is either recorded or
    reported.
    """
    import importlib.util

    path = REPO / "scripts/architecture/record_run_index.py"
    spec = importlib.util.spec_from_file_location("record_run_index", path)
    ri = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ri)

    on_disk = {(p.parent.name, p.name)
               for p in (REPO / "logs/runs").glob("*/*") if p.is_dir()
               and any(q.is_file() for q in p.rglob("*"))}
    index = ri.build_index(REPO)
    accounted = {(r["experiment_id"], r["run_id"])
                 for r in index["runs"] if r["layout_version"] != 1}
    accounted |= {(u["experiment_id"], u["run_id"]) for u in index["unrecorded"]}
    assert on_disk <= accounted, f"unaccounted run directories: {on_disk - accounted}"
    committed = json.loads((REPO / "logs/runs/index.json").read_text())
    assert committed["counts"].get("runs_unrecorded") == len(index["unrecorded"])
