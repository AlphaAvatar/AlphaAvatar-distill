"""A study's engineering validation: grant, envelope, one-use authorization, live funding, launch gates, closeout.

Every refusal is exercised on a temporary repository with a fake study; the
provider position is supplied (its own rule is `test_project_position.py`'s).
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from aadistill.governance.authorization import AuthorizationError
from shared import study_engineering as SE

POS = {"fetched_utc": "2026-10-28T05:37:32+00:00", "provider_billed_usd": 720.2558, "committed_usd": 724.3538,
       "project_cap_usd": 740.0, "project_remaining_usd": 15.6462, "account_pods": [], "account_volumes": ["v"],
       "balance_usd": 139.18}


def _study(root: Path) -> SE.Study:
    for rel, text in {"configs/x/design.json": "{}", "configs/x/check.json": "{}", "scripts/x/driver.py": "x = 1\n",
                      "src/aadistill/__init__.py": ""}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return SE.Study(key="X-E", plan_id="stage9.x.engineering", logs=Path("logs/stages/stage-9/x"),
                    grant=Path("logs/budget/approvals/x_engineering_grant.json"),
                    proposal=Path("logs/stages/stage-9/x/plans/engineering_grant_proposal.json"),
                    identity_files=("configs/x/design.json", "configs/x/check.json"), scripts=("scripts/x/driver.py",),
                    wire_schema="aadistill.stage9.x.engineering_authorization/v1")


def _proposal(root: Path, study: SE.Study, **over) -> dict:
    doc = {"schema": "aadistill.study_engineering_grant/v1", "status": "proposed", "study": study.key,
           "granted_by": "nobody: a proposal, not a grant", "decision": "PENDING", "ceiling_usd": 3.0,
           "per_session_hard_cap_usd": 3.0, "project_cumulative_cap_usd": 740.0, "gpu": "NVIDIA L40S",
           "max_price_usd_per_hour": 1.14, **over}
    (root / study.proposal).parent.mkdir(parents=True, exist_ok=True)
    (root / study.proposal).write_text(json.dumps(doc))
    return doc


@pytest.fixture
def live(monkeypatch):
    state = dict(POS)
    monkeypatch.setattr(SE, "live_position", lambda root=None, snapshot=None: dict(state))
    return state


def test_no_grant_refuses_and_a_proposal_funds_only_a_dry_run(tmp_path, live):
    st = _study(tmp_path)
    with pytest.raises(AuthorizationError, match="not authorized"):
        SE.issue(st, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path)
    _proposal(tmp_path, st)
    with pytest.raises(AuthorizationError, match="not authorized"):
        SE.issue(st, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path)
    with pytest.raises(AuthorizationError, match="explicit out path"):
        SE.issue(st, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True)
    with pytest.raises(AuthorizationError, match="never lands in the study's run directory"):
        SE.issue(st, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True,
                 out=tmp_path / st.authorization_path("r1"))
    p = SE.issue(st, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True, out=tmp_path / "s/a.json")
    assert not (tmp_path / st.run_dir("r1")).exists()
    with pytest.raises(AuthorizationError, match="DRY-RUN"):
        SE.load_authorization(st, p)
    a = SE.load_authorization(st, p, allow_dry_run=True)
    assert a.stage_conditions["dry_run"] == "true" and a.allows("engineering_validation")
    assert not any(a.allows(x) for x in ("formal", "training", "evaluation", "teacher_generation", "paid_storage"))


def test_a_granted_artifact_is_one_use_bound_and_tamper_evident(tmp_path, live):
    st = _study(tmp_path)
    _proposal(tmp_path, st)
    with pytest.raises(AuthorizationError, match="quotes the maintainer's decision"):
        SE.write_grant(st, decision=" ", granted_by="maintainer", root=tmp_path)
    g = SE.write_grant(st, decision="approved: X-E $3.00", granted_by="maintainer", root=tmp_path)
    doc = json.loads(g.read_text())
    assert doc["status"] == "granted" and doc["study_identity_sha256"] == SE.identity_hash(st, tmp_path)
    with pytest.raises(AuthorizationError, match="written once"):
        SE.write_grant(st, decision="again", granted_by="maintainer", root=tmp_path)
    p = SE.issue(st, "r1", expected_usd=1.5, hard_cap_usd=3.0, root=tmp_path)
    assert p == tmp_path / st.authorization_path("r1")
    a = SE.load_authorization(st, p)
    assert a.plan_hash == SE.identity_hash(st, tmp_path) and a.stage_conditions["dry_run"] == "false"
    assert a.stage_conditions["gpu"] == "NVIDIA L40S" and json.loads(a.stage_conditions["live_position"])["committed_usd"]
    with pytest.raises(AuthorizationError, match="one artifact per run"):
        SE.issue(st, "r1", expected_usd=1.5, hard_cap_usd=3.0, root=tmp_path)
    raw = json.loads(p.read_text())
    raw["hard_cap_usd"] = 30.0
    p.write_text(json.dumps(raw))
    with pytest.raises(AuthorizationError, match="edited since it was granted"):
        SE.load_authorization(st, p)
    # the study changed after the grant: the grant's identity no longer names it
    (tmp_path / "configs/x/check.json").write_text('{"threshold": 3}')
    with pytest.raises(AuthorizationError, match="names study identity"):
        SE.issue(st, "r2", expected_usd=1.5, hard_cap_usd=3.0, root=tmp_path)


@pytest.mark.parametrize("change, match", [
    ({"hard": 3.01}, "per-session hard cap"),
    ({"spent": 2.5}, "envelope leaves"),
    ({"pos": {"project_cap_usd": 790.0, "project_remaining_usd": 65.6}}, "a new maintainer decision"),
    ({"pos": {"project_remaining_usd": 2.99}}, "does not fit"),
    ({"pos": {"balance_usd": 2.0}}, "less than the"),
])
def test_each_funding_condition_refuses_on_its_own(change, match):
    g = {"per_session_hard_cap_usd": 3.0, "project_cumulative_cap_usd": 740.0}
    env = {"remaining_usd": round((3.5 if change.get("hard") else 3.0) - change.get("spent", 0.0), 4)}
    bad = SE.funding_problems(change.get("hard", 3.0), g, env, {**POS, **change.get("pos", {})})
    assert len(bad) >= 1 and any(match in b for b in bad), bad
    assert SE.funding_problems(3.0, {"per_session_hard_cap_usd": 3.0, "project_cumulative_cap_usd": 740.0},
                               {"remaining_usd": 3.0}, POS) == []


def _run(root: Path, st: SE.Study, run: str, *, closeout: dict | None = None, created: bool = True) -> None:
    d = root / st.run_dir(run)
    (d / "governance").mkdir(parents=True)
    (d / "governance/authorization.json").write_text(json.dumps({"plan_id": st.plan_id}))
    (d / "runtime").mkdir()
    (d / "runtime/session.json").write_text(json.dumps({"provider_resource_created": created}))
    if closeout is not None:
        (d / "closeout").mkdir()
        (d / "closeout/outcome.json").write_text(json.dumps(closeout))


def test_the_envelope_counts_every_session_and_an_unknown_refuses(tmp_path):
    st = _study(tmp_path)
    g = {"ceiling_usd": 3.0}
    _run(tmp_path, st, "a", closeout={"money": {"all_in_usd": 0.75}, "provider_resource_created": True})
    _run(tmp_path, st, "b", created=False)                       # refused at $0: nothing billed
    env = SE.envelope(st, g, tmp_path)
    assert env["spent_usd"] == 0.75 and env["remaining_usd"] == 2.25
    _run(tmp_path, st, "c")                                      # a pod, and no closeout cost yet
    with pytest.raises(AuthorizationError, match="record no closeout cost"):
        SE.envelope(st, g, tmp_path)


def _ctx(st, auth, run="r9", **args):
    a = SimpleNamespace(run_id=run, dry_run=False, gpu="NVIDIA L40S", max_price=1.09, disk_gb=120, **args)
    return SimpleNamespace(auth=auth, args=a, evidence={}, plan=None)


def test_launch_gates_refuse_a_dry_run_artifact_a_used_run_and_another_card(tmp_path, live):
    st = _study(tmp_path)
    _proposal(tmp_path, st)
    launch = SE.Launch(session_kind="x", authorization=SE.EngineeringAuthorization, teacher_model="m", teacher_revision="r", driver="scripts/x/driver.py",
                       validation_minutes=10.0)
    unused, gpu = SE.prechecks(st, launch, tmp_path)[:2]
    dry = SE.load_authorization(st, SE.issue(st, "r9", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path,
                                             dry_run=True, out=tmp_path / "s/a.json"), allow_dry_run=True)
    ok, why = unused(_ctx(st, dry))
    assert not ok and "dry-run" in why
    ctx = _ctx(st, dry)
    ctx.args.dry_run = True
    assert unused(ctx)[0]
    assert gpu(_ctx(st, dry))[0]                                  # the granted card at the granted price
    bad = _ctx(st, dry)
    bad.args.gpu, bad.args.max_price = "NVIDIA H100 80GB HBM3", 1.09
    assert not gpu(bad)[0]
    bad.args.gpu, bad.args.max_price = "NVIDIA L40S", 1.20
    assert not gpu(bad)[0]
    granted = SE.load_authorization(st, SE.issue(st, "r9", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path,
                                                 dry_run=True, out=tmp_path / "s/b.json"), allow_dry_run=True)
    real = _ctx(st, granted)
    real.auth = type("A", (), {"plan_id": st.plan_id, "stage_conditions": {"study": st.key, "dry_run": "false",
                                                                          "kind": "engineering_validation"}})()
    assert unused(real)[0]
    _run(tmp_path, st, "r9", created=False)                      # an earlier launch refused at $0: consumed
    ok, why = unused(real)
    assert not ok and "already used its artifact" in why


def _store(scr: Path, report: dict, *, status="MARKER:ALL_DONE", tamper=False) -> None:
    ev = scr / "ev"
    ev.mkdir(parents=True)
    (ev / SE.REPORT).write_text(json.dumps(report))
    st = scr / "store"
    st.mkdir()
    with tarfile.open(st / SE.EVIDENCE_ARCHIVE, "w:gz") as t:
        t.add(ev / SE.REPORT, arcname=f"./{SE.REPORT}")
    digest = hashlib.sha256((st / SE.EVIDENCE_ARCHIVE).read_bytes()).hexdigest()
    (st / "evidence.sha256").write_text(("0" * 64 if tamper else digest) + "\n")
    (st / "driver.status").write_text(f"2026-10-28T06:00:00Z {status}\n")


@pytest.mark.parametrize("tamper", [False, True])
def test_closeout_rehashes_the_evidence_and_rederives_the_verdict(tmp_path, tamper):
    st = _study(tmp_path)
    report = {"passed": True, "checks": {"a": {"status": "PASS"}, "vllm": {"status": "PASS"}}}
    launch = SE.Launch(session_kind="x", authorization=SE.EngineeringAuthorization, teacher_model="m", teacher_revision="r", driver="scripts/x/driver.py",
                       validation_minutes=10.0, verdict=lambda d: json.loads((d / SE.REPORT).read_text()))
    _run(tmp_path, st, "r1")
    sess = tmp_path / st.run_dir("r1") / "runtime/session.json"
    sess.write_text(json.dumps({"provider_resource_created": True, "provider_confirms_gone": True,
                                "cost": {"actual_usd": 1.2, "elapsed_minutes": 66.0, "price_per_hour": 1.09}}))
    _store(tmp_path / "scr", report, tamper=tamper)
    out = SE.closeout(st, launch, "r1", tmp_path / "scr", root=tmp_path, reconcile=lambda: {"ok": True})
    assert out["money"]["all_in_usd"] == pytest.approx(1.2 + 120 * 0.10 * 66 / (60 * 24 * 30.4375), abs=1e-4)
    assert out["evidence"]["verified"] is (not tamper)
    if tamper:
        assert out["status"] == "INCOMPLETE" and out["verdict_rederived"] is None and not out["validation_passed"]
    else:
        assert out["status"] == "COMPLETE" and out["validation_passed"]
        assert out["verdict_rederived"]["agrees_with_the_pod"]
    assert SE.envelope(st, {"ceiling_usd": 3.0}, tmp_path)["spent_usd"] == out["money"]["all_in_usd"]


@pytest.mark.parametrize("secured", [True, False])
def test_a_kind_that_secures_its_products_closes_on_them(tmp_path, secured):
    """A formal kind keeps no evidence archive: its products are secured on every poll, and the launcher's
    record of that (with the driver's ALL_DONE) is what makes its closeout COMPLETE."""
    st = _study(tmp_path)
    launch = SE.Launch(session_kind="x", authorization=SE.EngineeringAuthorization, teacher_model="m",
                       teacher_revision="r", driver="scripts/x/driver.py", validation_minutes=10.0,
                       products_secured=lambda ctx, *a, **k: (True, ""))
    _run(tmp_path, st, "r1")
    (tmp_path / st.run_dir("r1") / "runtime/session.json").write_text(json.dumps(
        {"provider_resource_created": True, "provider_confirms_gone": True,
         "required_products_secured": {"ok": secured, "why": "74 finished item(s) secured off-pod"},
         "cost": {"actual_usd": 5.2, "elapsed_minutes": 288.0, "price_per_hour": 1.09}}))
    (tmp_path / "scr/store").mkdir(parents=True)
    (tmp_path / "scr/store/driver.status").write_text("2026-10-30T01:06:39Z MARKER:ALL_DONE\n")
    out = SE.closeout(st, launch, "r1", tmp_path / "scr", root=tmp_path, reconcile=lambda: {"ok": True})
    assert out["status"] == ("COMPLETE" if secured else "INCOMPLETE") and out["evidence"] is None
    assert out["products_secured"]["ok"] is secured and not out["validation_passed"]


def test_each_study_has_its_own_type_and_setup_branch():
    text = (SE.REPO_ROOT / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    for kind, mod, cls in (("stage2_teacher_modes", "stages.teacher_modes.ae_launch", "AEAuthorization"),
                           ("stage1_projected_residual", "stages.projected_residual.be_launch", "BEAuthorization"),
                           ("stage1_independent_gated", "stages.projected_residual.igv_launch", "IGEAuthorization"),
                           ("stage2_teacher_modes_phase", "stages.teacher_modes.phase", "PhaseAuthorization"),
                           ("stage2_teacher_modes_tm", "stages.teacher_modes.token_matched_phase", "TMAuthorization")):
        body = text[text.index(f'elif [ "$SESSION_KIND" = "{kind}" ]; then'):]
        body = body[:body.index("elif [", 10)]
        for needle in (f"from {mod} import {cls}", f"{cls}.load(", "SE.require_on_pod(", "AUTHORIZATION_MISMATCH"):
            assert needle in body, (kind, needle)
    assert 'TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"' in text


def test_the_pod_check_refuses_a_dry_run_and_another_plan(tmp_path, live):
    st = _study(tmp_path)
    _proposal(tmp_path, st)

    class XAuthorization(SE.EngineeringAuthorization):
        STUDY = st
    p = SE.issue(st, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True, out=tmp_path / "s/a.json")
    with pytest.raises(AuthorizationError, match="DRY-RUN"):
        XAuthorization.load(p)
    a = XAuthorization.load(p, allow_dry_run=True)
    assert type(a) is XAuthorization
    with pytest.raises(AuthorizationError, match="dry-run artifact funds nothing"):
        SE.require_on_pod(a, st, SE.identity_hash(st, tmp_path), root=tmp_path)
    with pytest.raises(AuthorizationError, match="bound to plan"):
        SE.require_on_pod(a, st, "0" * 64, root=tmp_path)


def test_a_real_launch_without_its_own_artifact_refuses_at_zero(tmp_path, monkeypatch, capsys):
    """No grant, so no artifact in the run directory: the launcher refuses with the reason before the
    runner, and a dry-run artifact is never accepted by a real launch."""
    from stages.reasoning_data import h1_generation_launch as GL
    st = _study(tmp_path)
    launch = SE.Launch(session_kind="x", authorization=SE.EngineeringAuthorization, teacher_model="m",
                       teacher_revision="r", driver="scripts/x/driver.py", validation_minutes=10.0)
    monkeypatch.setattr(GL, "bundle_on_relay", lambda args: (True, "on the relay"))
    monkeypatch.setattr(SE, "session_spec", lambda study, launch, args, root=None: SimpleNamespace(
        authorization_path=str(study.authorization_path(args.run_id)), plan_hash="0" * 64,
        authorization_loader=lambda p: SE.load_authorization(study, p)))
    import aadistill.infrastructure.session_runner as SR
    monkeypatch.setattr(SR, "run_session", lambda *a, **k: pytest.fail("the runner was reached"))
    argv = ["--run-id", "r1", "--scr", str(tmp_path / "scr"), "--session-commit", "0" * 40,
            "--bundle", "b", "--max-price", "1.09"]
    assert SE.launch_main(st, launch, argv, root=tmp_path, gpu="NVIDIA L40S") == 11
    assert "REFUSED at $0: no authorization" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="dry-run artifact"):
        SE.launch_main(st, launch, argv + ["--authorization", "x.json"], root=tmp_path, gpu="NVIDIA L40S")


def test_a_formal_phase_names_its_kinds_and_its_artifacts_claim_exactly_the_phase(tmp_path, live, monkeypatch):
    st = _study(tmp_path)
    import dataclasses
    ph = dataclasses.replace(st, key="X-P", plan_id="stage9.x.phase", claims=frozenset({"formal", "training", "evaluation"}),
                             kinds=("teacher", "recovery"), wire_schema="aadistill.stage9.x.phase_authorization/v1")
    _proposal(tmp_path, ph, gpus={"NVIDIA A100-SXM4-80GB": 1.89}, gpu_count_max=2, ceiling_usd=10.0,
              per_session_hard_cap_usd=10.0, gpu=None, max_price_usd_per_hour=None)
    live["project_remaining_usd"] = 20.0
    with pytest.raises(AuthorizationError, match="not a session kind"):
        SE.issue(ph, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True, out=tmp_path / "s/a.json",
                 kind="engineering_validation")
    p = SE.issue(ph, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True, out=tmp_path / "s/a.json",
                 kind="recovery", conditions={"admission_sha256": "a" * 64})
    a = SE.load_authorization(ph, p, allow_dry_run=True)
    assert SE.claimed(a) == ph.claims and a.stage_conditions["kind"] == "recovery"
    assert a.stage_conditions["admission_sha256"] == "a" * 64 and json.loads(a.stage_conditions["gpus"])
    # an engineering study cannot load the phase's artifact, nor the phase an engineering one
    with pytest.raises(AuthorizationError):
        SE.load_authorization(st, p, allow_dry_run=True)
    # on the pod: identity, harness, claims and the kind, never a dry run
    with pytest.raises(AuthorizationError, match="dry-run"):
        SE.require_on_pod(a, ph, a.plan_hash, str(tmp_path), kind="recovery")
    monkeypatch.setitem(a.stage_conditions, "dry_run", "false")
    with pytest.raises(AuthorizationError, match="not 'teacher'"):
        SE.require_on_pod(a, ph, a.plan_hash, str(tmp_path), kind="teacher")
    assert SE.require_on_pod(a, ph, a.plan_hash, str(tmp_path), kind="recovery")["kind"] == "recovery"
    # the hardware boundary: a granted card, at most its price per GPU x the count, at most gpu_count_max GPUs
    launch = SE.Launch(session_kind="x", authorization=SE.EngineeringAuthorization, teacher_model="m",
                       teacher_revision="r", driver="scripts/x/driver.py", validation_minutes=10.0, kind="recovery")
    unused, gpu = SE.prechecks(ph, launch, tmp_path)[:2]
    ctx = SimpleNamespace(auth=a, args=SimpleNamespace(run_id="r1", dry_run=True, gpu="NVIDIA A100-SXM4-80GB",
                                                       max_price=3.7, gpu_count=2, disk_gb=150), evidence={}, plan=None)
    assert gpu(ctx)[0] and unused(ctx)[0]
    ctx.args.gpu_count = 3
    assert not gpu(ctx)[0]
    ctx.args.gpu_count, ctx.args.max_price = 2, 3.9
    assert not gpu(ctx)[0]
    ctx.args.gpu, ctx.args.max_price = "NVIDIA L40S", 1.0
    assert not gpu(ctx)[0]
    teacher = dataclasses.replace(launch, kind="teacher")
    assert not SE.prechecks(ph, teacher, tmp_path)[0](ctx)[0]           # a recovery artifact funds no teacher session


def test_a_phase_kind_may_claim_less_and_each_artifact_claims_exactly_its_kinds(tmp_path, live, monkeypatch):
    """A formal phase's engineering check claims engineering_validation only, under the phase's one grant; a check
    artifact is never a formal one, and neither passes as the other's kind on the pod."""
    st = _study(tmp_path)
    import dataclasses
    ph = dataclasses.replace(st, key="X-P", plan_id="stage9.x.phase", claims=frozenset({"formal", "training"}),
                             kinds=("check", "teacher"), kind_claims=(("check", frozenset({"engineering_validation"})),),
                             wire_schema="aadistill.stage9.x.phase_authorization/v1")
    _proposal(tmp_path, ph, gpus={"NVIDIA L40S": 1.19}, gpu_count_max=2, ceiling_usd=10.0,
              per_session_hard_cap_usd=10.0, gpu=None, max_price_usd_per_hour=None)
    live["project_remaining_usd"] = 20.0
    assert ph.formal and ph.claims_of("check") == {"engineering_validation"} and ph.claims_of("teacher") == ph.claims
    chk = SE.issue(ph, "r1", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True, out=tmp_path / "s/c.json",
                   kind="check")
    tea = SE.issue(ph, "r2", expected_usd=1.0, hard_cap_usd=3.0, root=tmp_path, dry_run=True, out=tmp_path / "s/t.json",
                   kind="teacher")
    c, t = (SE.load_authorization(ph, p, allow_dry_run=True) for p in (chk, tea))
    assert SE.claimed(c) == {"engineering_validation"} and not c.allows("formal")
    assert SE.claimed(t) == {"formal", "training"}
    # a check artifact relabelled as a teacher one is refused: its claims are not the teacher kind's
    raw = json.loads(chk.read_text())
    raw["stage_conditions"]["kind"] = "teacher"
    raw.pop("authorization_sha256")
    from aadistill.infrastructure.manifest import sha256_json
    raw["authorization_sha256"] = sha256_json(raw)
    forged = tmp_path / "s/forged.json"
    forged.write_text(json.dumps(raw))
    with pytest.raises(AuthorizationError, match="claims"):
        SE.load_authorization(ph, forged, allow_dry_run=True)
    for a, kind, other in ((c, "check", "teacher"), (t, "teacher", "check")):
        monkeypatch.setitem(a.stage_conditions, "dry_run", "false")
        assert SE.require_on_pod(a, ph, a.plan_hash, str(tmp_path), kind=kind)["kind"] == kind
        with pytest.raises(AuthorizationError, match="claims|not"):
            SE.require_on_pod(a, ph, a.plan_hash, str(tmp_path), kind=other)
