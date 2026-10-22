"""P1's launch wiring at $0: what binds a session, what refuses it, what it brings home.

No pod, no provider call, no grant: artifacts are issued in tmp_path from a
test grant document, and the provider reconciliation is injected.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import governance as GOV

from aadistill.governance.authorization import AuthorizationError, harness_source_digest
from support.session_specs import load_session_launcher, session_args

L = load_session_launcher("h1_generation_launch")


def _grant(tmp_path, **over) -> Path:
    g = {"granted_by": "test", "decision": "test only", "phase": "P1", "launch_identity_sha256": GOV.launch_hash(),
         "expected_usd": 1.0, "hard_cap_usd": 2.0, **over}
    p = tmp_path / "grant.json"
    p.write_text(json.dumps(g))
    return p


def test_the_launch_identity_binds_the_protocol_and_every_input(tmp_path):
    ident = GOV.launch_identity()
    assert ident["generation_protocol"]["path"] == str(GOV.PROTOCOL)
    assert ident["inputs"] and ident["pools"]
    repo = tmp_path / "repo"
    for f in [ident["generation_plan"], ident["generation_protocol"], *ident["inputs"], *ident["pools"]]:
        rel = f.get("path") or f["file"]
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes((A.REPO / rel).read_bytes())
    assert GOV.launch_hash(repo) == GOV.launch_hash()
    proto = repo / GOV.PROTOCOL                       # a protocol edit moves the identity: it is bound
    proto.write_text(proto.read_text().replace('"v2"', '"v1"', 1))
    assert GOV.launch_hash(repo) != GOV.launch_hash()
    inp = repo / ident["inputs"][0]["file"]            # an input that differs from the plan refuses
    inp.write_text(inp.read_text() + "\n")
    with pytest.raises(AuthorizationError, match="not the plan's"):
        GOV.launch_identity(repo)


def test_the_issuer_needs_a_grant_naming_this_launch_identity(tmp_path):
    out = tmp_path / "authorization.json"
    with pytest.raises(AuthorizationError, match="no grant document at"):
        GOV.issue("r1", grant_path=tmp_path / "missing.json", out=out)
    with pytest.raises(AuthorizationError, match="does not derive"):
        GOV.issue("r1", grant_path=_grant(tmp_path, launch_identity_sha256="0" * 64), out=out)
    with pytest.raises(AuthorizationError, match="not P1"):
        GOV.issue("r1", grant_path=_grant(tmp_path, phase="P2"), out=out)
    path = GOV.issue("r1", grant_path=_grant(tmp_path), out=out)
    auth = GOV.load(path)                              # the artifact binds identity and harness
    auth.require_plan(GOV.launch_hash())
    assert auth.require_harness(A.REPO)["digest"] == harness_source_digest(A.REPO, files=GOV.HARNESS)["digest"]
    raw = json.loads(path.read_text())
    assert raw["authorizes_teacher_generation"] and not raw["authorizes_training"] and not raw["authorizes_paid_storage"]
    with pytest.raises(AuthorizationError, match="exists"):
        GOV.issue("r1", grant_path=_grant(tmp_path), out=out)


def test_an_artifact_claiming_more_or_edited_after_issue_is_refused(tmp_path):
    path = GOV.issue("r2", grant_path=_grant(tmp_path), out=tmp_path / "a.json")
    raw = json.loads(path.read_text())
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps({**raw, "hard_cap_usd": 99.0}))
    with pytest.raises(AuthorizationError, match="edited"):
        GOV.load(edited)
    with pytest.raises(AuthorizationError, match="no authorization at"):
        GOV.load(tmp_path / "none.json")


def test_the_harness_covers_the_pod_paths_whole_import_closure():
    code = ("import sys, importlib\n"
            "for m in ('stages.reasoning_data.generation_driver', 'stages.reasoning_data.generation_adapter',"
            " 'stages.reasoning_data.checkers', 'stages.reasoning_data.supporting_verify',"
            " 'stages.calibration_volume.run_battery', 'stages.calibration_volume.score_battery',"
            " 'shared.evaluation.uncapped_eval', 'aadistill.evaluation.degeneration'):\n"
            "    importlib.import_module(m)\n"
            "print('\\n'.join(sorted(getattr(m, '__file__', '') or '' for m in list(sys.modules.values()))))\n")
    got = subprocess.run([sys.executable, "-c", code], cwd=A.REPO, capture_output=True, text=True,
                         env={"PYTHONPATH": "src:scripts", "PATH": "/usr/bin:/bin"}, check=True).stdout.split()
    root = str(A.REPO) + "/"
    closure = {p[len(root):] for p in got if p.startswith(root) and "/.venv/" not in p
               and not p.endswith("__init__.py")}
    missing = sorted(p for p in closure - set(GOV.HARNESS) if not p.startswith(GOV.HARNESS_EXEMPT_PREFIXES))
    assert not missing, f"executed on the pod but not in the authorized harness: {missing}"


def test_the_session_spec_validates_and_binds_the_launch_identity():
    args = session_args(L)
    spec = L.spec(args)
    spec.validate()
    assert spec.plan_hash == GOV.launch_hash()
    assert "TESTS_OK" not in spec.setup.setup_markers and not spec.setup.test_paths   # no suite on the paid pod
    assert spec.authorization_path == str(GOV.authorization_path(args.run_id))
    cmd = spec.driver_command(type("Ctx", (), {"args": args})(), None)
    assert "-m stages.reasoning_data.generation_driver" in cmd and "--resume-from" not in cmd
    b = spec.budget
    assert b.contingency_fraction > 0 and b.other_phases[0].minutes > 0


def test_the_prechecks_refuse_a_plan_or_commit_that_is_not_the_bound_one(monkeypatch):
    ok, why = L.plan_is_the_chosen_admitted_one(None)
    assert ok, why
    readme = subprocess.run(["git", "show", "HEAD:README.md"], cwd=A.REPO, capture_output=True).stdout
    ctx = type("Ctx", (), {"args": type("A", (), {"session_commit": "HEAD"})()})()
    ident = {"generation_plan": {"path": "README.md", "sha256": hashlib.sha256(readme).hexdigest()},
             "generation_protocol": {"path": "README.md", "sha256": hashlib.sha256(readme).hexdigest()},
             "inputs": [], "pools": []}
    monkeypatch.setattr(GOV, "launch_identity", lambda root=None: ident)
    assert L.bound_files_are_in_the_session_commit(ctx)[0]
    ident["pools"] = [{"path": "README.md", "sha256": "0" * 64}]
    ok, why = L.bound_files_are_in_the_session_commit(ctx)
    assert not ok and "README.md" in why


def test_a_launch_without_an_authorization_is_refused_before_any_provider_call(tmp_path, capsys):
    rc = L.main(["--run-id", "no_such_run", "--scr", str(tmp_path), "--session-commit", "0" * 40,
                 "--bundle", "x", "--max-price", "1.0", "--dry-run"])
    assert rc == 11 and "REFUSED at $0" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no network volume"):
        L.main(["--run-id", "r", "--scr", str(tmp_path), "--session-commit", "0" * 40, "--bundle", "x",
                "--max-price", "1.0", "--network-volume-id", "v"])


def test_closeout_rehashes_every_shard_and_reconciles_the_provider(tmp_path, monkeypatch):
    scr, store = tmp_path / "scr", tmp_path / "scr/store"
    store.mkdir(parents=True)
    good = store / "pass0_g0.part0000.jsonl"
    good.write_text('{"task_id": "a"}\n{"task_id": "b"}\n')
    good.with_suffix(".done").write_text(hashlib.sha256(good.read_bytes()).hexdigest() + "\n")
    (store / "pass0_g0.complete").write_text(json.dumps({"label": "pass0_g0", "tasks": 2, "shards": [
        {"name": good.name, "sha256": hashlib.sha256(good.read_bytes()).hexdigest()}]}))
    (store / L.EVIDENCE).write_text(json.dumps({"terminal": "ALL_DONE"}))
    monkeypatch.setattr(GOV, "run_dir", lambda run_id: Path("runs") / run_id)
    out = L.closeout("r", scr, repo_root=tmp_path, reconcile=lambda: {"ok": True, "why": "0 pods"})
    assert out["complete"] and out["shards"] == {"secured": 1, "unverified": [], "records": 2}
    assert out["provider_reconciliation_after"]["ok"]
    good.write_text('{"task_id": "a"}\n')                 # a shard that no longer matches its marker
    out = L.closeout("r", scr, repo_root=tmp_path, reconcile=lambda: {"ok": True, "why": "0 pods"})
    assert not out["complete"] and out["shards"]["unverified"] == [good.name]


def _setup_branch() -> str:
    text = (A.REPO / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    start = text.index('elif [ "$SESSION_KIND" = "stage2_h1_generation" ]; then\n') + len(
        'elif [ "$SESSION_KIND" = "stage2_h1_generation" ]; then\n')
    return text[start:text.index("\nelif [", start)]


@pytest.mark.parametrize("plan_hash,rc", [("current", 0), ("0" * 64, 98)])
def test_the_pods_setup_branch_runs_and_fails_closed(tmp_path, plan_hash, rc):
    """The branch the pod executes, executed: it loads the artifact by its own
    type, recomputes the launch identity and harness from the checkout, and a
    mismatch marks AUTHORIZATION_MISMATCH and exits 98."""
    path = GOV.issue("setup_probe", grant_path=_grant(tmp_path), out=tmp_path / "auth.json")
    script = ('say() { echo "SAY $*"; }; mark() { echo "MARK $*"; }\n' + _setup_branch() + "\nexit 0\n")
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "REPO": str(A.REPO),
           "SESSION_AUTH_PATH": str(path),
           "SESSION_PLAN_HASH": GOV.launch_hash() if plan_hash == "current" else plan_hash}
    script = script.replace("/opt/train/bin/python", sys.executable)
    res = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, cwd=A.REPO)
    assert res.returncode == rc, res.stdout + res.stderr
    if rc == 0:
        assert "setup_probe" in res.stdout and f"launch {GOV.launch_hash()[:12]}" in res.stdout
    else:
        assert "MARK AUTHORIZATION_MISMATCH" in res.stdout
