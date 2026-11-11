"""Research A CLOSED (maintainer 2026-11-11, 'Research A 最终关闭与 Squash Merge Main', section 1): the grants' study-level
closure (`study_engineering.require_open`) refuses every A-TM kind at issue, a dry run included; every launch, before
the artifact, the bundle or the provider is read; an issued artifact no run has used; and the runner's first precheck.
A-E and Research A v1 are closed the same way. Read-only work still reads the closed grant, and the closure appended a
block without changing anything the grant said before it."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from shared import study_engineering as SE
from stages.teacher_modes import ae_launch as AE
from stages.teacher_modes import phase as P1
from stages.teacher_modes import token_matched_phase as TM

REPO = TM.REPO_ROOT
A_MIX = "a_train_sr_mix_20261109_193709"            #: an artifact the issuer really wrote (used once, 2026-11-09)
LAUNCH = ["--scr", "{scr}", "--session-commit", "HEAD", "--bundle", "b", "--max-price", "1.19"]


@pytest.fixture
def unreachable(monkeypatch):
    """Everything past the closure: the provider, the session, the runner and the relay. Reaching any one fails."""
    reached = []

    def trap(name):
        def f(*a, **k):
            reached.append(name)
            raise AssertionError(f"{name} was reached under a CLOSED study")
        return f
    from aadistill.infrastructure import session_runner as SR
    from stages.reasoning_data import h1_generation_launch as GL
    for mod, name in ((SE, "provider_snapshot"), (SE, "live_position"), (SE, "session_spec"), (SR, "SessionRunner"),
                      (SR, "run_session"), (GL, "bundle_on_relay")):
        monkeypatch.setattr(mod, name, trap(name))
    return reached


def launch_argv(run_id: str, tmp_path) -> list[str]:
    return ["--run-id", run_id, *(a.format(scr=tmp_path / "scr") for a in LAUNCH)]


# ---- issue ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("kind", TM.STUDY.kinds)
def test_every_a_tm_kind_is_refused_at_issue_and_nothing_is_written(kind, dry_run, unreachable, capsys, tmp_path):
    run = f"closure_probe_{kind}"
    argv = ["issue", "--kind", kind, "--run-id", run, "--expected-usd", "0.01", "--hard-cap-usd", "0.01"]
    if dry_run:
        argv += ["--dry-run", "--out", str(tmp_path / "auth.json")]
    assert TM.main(argv) == 11
    assert "A-TM is CLOSED" in capsys.readouterr().out
    #: the generic issuer, whoever calls it, refuses at its grant read
    with pytest.raises(SE.AuthorizationError, match="A-TM is CLOSED"):
        SE.issue(TM.STUDY, run, expected_usd=0.01, hard_cap_usd=0.01, kind=kind, dry_run=dry_run,
                 out=tmp_path / "auth.json")
    assert not (REPO / TM.STUDY.run_dir(run)).exists() and not (tmp_path / "auth.json").exists()
    assert unreachable == []


# ---- launch -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("kind", TM.STUDY.kinds)
def test_every_a_tm_kind_is_refused_at_launch_before_anything_is_read(kind, dry_run, unreachable, capsys, tmp_path):
    argv = ["launch", "--kind", kind, "--minutes", "1", *launch_argv(A_MIX, tmp_path)]
    if dry_run:
        argv += ["--dry-run", "--authorization", str(REPO / TM.STUDY.authorization_path(A_MIX))]
    assert TM.main(argv) == 11
    out = capsys.readouterr().out
    assert "A-TM is CLOSED" in out and "No provider resource was created." in out
    #: the shared launcher, reached directly with this kind's launch, refuses the same way
    assert SE.launch_main(TM.STUDY, TM.launch_spec(kind), launch_argv(A_MIX, tmp_path), gpu=TM.GPU) == 11
    assert "A-TM is CLOSED" in capsys.readouterr().out
    assert unreachable == [] and not (tmp_path / "scr").exists()


def test_the_runners_first_precheck_refuses_a_closed_study():
    """A session built by any other caller still meets the closure before its create: the first precheck, and the
    funding precheck's grant read."""
    checks = {c.__name__: c for c in SE.prechecks(TM.STUDY, TM.launch_spec("train_sr_mix"))}
    assert next(iter(checks)) == "the_study_is_open"
    ok, why = checks["the_study_is_open"](SimpleNamespace())
    assert not ok and "A-TM is CLOSED" in why
    ok, why = checks["the_project_cap_holds_now"](SimpleNamespace(args=SimpleNamespace(dry_run=False)),
                                                  snapshot={"unused": True})
    assert not ok and "A-TM is CLOSED" in why


# ---- an artifact issued before the closure ----------------------------------------------------------------------

def _root(tmp_path, name: str, grant: bytes, run_id: str):
    """A root holding the grant (as given) and a real issued artifact under a run no launch has used."""
    root = tmp_path / name
    (root / TM.STUDY.grant).parent.mkdir(parents=True)
    (root / TM.STUDY.grant).write_bytes(grant)
    dest = root / TM.STUDY.authorization_path(run_id)
    dest.parent.mkdir(parents=True)
    shutil.copy2(REPO / TM.STUDY.authorization_path(A_MIX), dest)
    return root


def test_an_issued_unused_artifact_cannot_launch_once_the_study_is_closed(unreachable, capsys, tmp_path, monkeypatch):
    """The artifact the issuer wrote for A-MIX, placed under a run that never launched (no session, no closeout): the
    closed grant refuses it before it is read. Under the grant's historical snapshot (its content before the closure)
    the same launch passes the gate and goes on to build the session -- so it is the closure that refuses."""
    run = "a_train_sr_mix_unused_probe"
    closed = (REPO / TM.STUDY.grant).read_bytes()
    root = _root(tmp_path, "closed", closed, run)
    assert not (root / TM.STUDY.run_dir(run) / "runtime/session.json").exists()
    argv = launch_argv(run, tmp_path) + ["--network-volume-id", "thprcgh7s9", "--volume-mount-path", "/runpod-volume"]
    assert SE.launch_main(TM.STUDY, TM.launch_spec("train_sr_mix"), argv, root=root, gpu=TM.GPU) == 11
    assert "A-TM is CLOSED" in capsys.readouterr().out and unreachable == []

    snapshot = {k: v for k, v in json.loads(closed).items() if k != "closed"}
    root = _root(tmp_path, "historical", json.dumps(snapshot, indent=1, ensure_ascii=False).encode(), run)
    built = []

    def session_spec(*a, **k):
        built.append(a[0].key)
        raise SE.AuthorizationError("stopped here: the closure gate was passed")
    monkeypatch.setattr(SE, "session_spec", session_spec)
    assert SE.launch_main(TM.STUDY, TM.launch_spec("train_sr_mix"), argv, root=root, gpu=TM.GPU) == 11
    assert built == ["A-TM"] and "stopped here" in capsys.readouterr().out


# ---- the rest of Research A -------------------------------------------------------------------------------------

def test_a_e_and_research_a_v1_are_closed_at_issue_and_launch(unreachable, capsys, tmp_path):
    assert AE.main(["issue", "--run-id", "closure_probe_ae", "--expected-usd", "0.01", "--hard-cap-usd", "0.01"]) == 11
    assert "A-E is CLOSED" in capsys.readouterr().out
    assert AE.main(["launch", *launch_argv("ae_20261028_093428", tmp_path)]) == 11
    assert "A-E is CLOSED" in capsys.readouterr().out
    for study in (AE.STUDY, P1.STUDY):
        with pytest.raises(SE.AuthorizationError, match=f"{study.key} is CLOSED"):
            SE.grant(study)
    assert P1.main(["launch", "--kind", "recovery", "--minutes", "1", *launch_argv("x", tmp_path)]) == 11
    assert "A-D4 is CLOSED" in capsys.readouterr().out
    assert not (REPO / AE.STUDY.run_dir("closure_probe_ae")).exists() and unreachable == []


# ---- what the closure leaves ------------------------------------------------------------------------------------

def test_read_only_work_still_reads_the_closed_grant(unreachable):
    from stages.teacher_modes import tool_pilot as TPL
    g = SE.grant_record(TM.STUDY)
    c = g["closed"]
    env = SE.envelope(TM.STUDY, g)
    #: nothing can be issued after the closure, so the envelope it recorded is the envelope now
    assert env["spent_usd"] == c["spent_usd"] and env["remaining_usd"] == c["unspent_usd"]
    assert {k: TM.kind_spent(k)["usd"] for k in TM.STUDY.kinds} == c["by_kind_usd"]
    assert set(c["by_kind_usd"]) == set(TM.STUDY.kinds)
    assert TPL.envelope_reconciliation()["envelope"]["spent_usd"] == c["spent_usd"]
    #: an identity review of a historical artifact
    auth = SE.load_authorization(TM.STUDY, REPO / TM.STUDY.authorization_path(A_MIX))
    assert auth.stage_conditions["kind"] == "train_sr_mix" and auth.plan_id == TM.STUDY.plan_id
    #: the funding position answers with the closure instead of a position
    assert TM.main(["position", "--hard-cap-usd", "1"]) == 11
    assert unreachable == []


@pytest.mark.parametrize("study", [TM.STUDY, AE.STUDY], ids=["A-TM", "A-E"])
def test_the_closure_is_appended_and_changes_nothing_before_it(study):
    g = json.loads((REPO / study.grant).read_text())
    c = g.pop("closed")
    assert c["status"] == "CLOSED" and c["utc"].endswith("Z") and c["by"] == "the maintainer"
    assert "2026-11-11" in c["decision"] and c["source"] and "require_open" in c["enforced_by"]
    assert g["status"] == "granted"                    #: the grant was valid when it funded what it funded
    assert c["unspent_usd"] == round(float(g["ceiling_usd"]) - c["spent_usd"], 4)
    assert "not carried forward" in c["remainder"] and "not refunded" in c["remainder"]
    b = c["grant_before_closure"]
    before = subprocess.run(["git", "show", f"{b['commit']}:{study.grant}"], cwd=REPO, capture_output=True)
    if before.returncode != 0:
        pytest.skip("the grant's pre-closure commit is not in this checkout")
    assert hashlib.sha256(before.stdout).hexdigest() == b["sha256"]
    assert json.loads(before.stdout) == g
