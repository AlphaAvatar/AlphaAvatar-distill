"""The mixed-calibration study's execution path, at $0.

One test per property the maintainer's 2026-10-26 corrections or the grant
claim: posted volume rows netted only on sole-volume days and never beyond a
volume's own hours, unposted sessions counted inside the posting window, an
unknown or foreign resource refusing, the first funded issue holding the whole
envelope, MC1 retries leaving MC2 fundable, both sessions' worst cases inside
their caps, Mix-Augmented's plan being D4-Augmented's with only the
initialization changed, H1-PA's schedule rule untouched, and both closeouts
and the Top-1 read-back executed on real-shaped inputs before any paid run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from stages.calibration_volume import governance as G
from stages.calibration_volume import mixed_governance as MG
from stages.calibration_volume import mixed_launch as ML

REPO = Path(__file__).resolve().parents[5]
UTC = timezone.utc


def _vol(vid, size, created, deleted=None, reserved=4.2, actual=None):
    return {"volume": vid, "size_gb": float(size), "created": created, "deleted": deleted,
            "reserved_usd": reserved, "actual_usd": actual, "record": f"logs/x/{vid}.json"}


def _row(day, gbh):
    return {"startDate": f"{day} 00:00:00", "diskSpaceBilledGb": gbh, "amount": gbh * 0.07 / 720}


def test_netting_only_on_sole_volume_days_capped_by_own_hours():
    """The 2026-10-24..26 rows: the shared day nets nothing, 10-25's 870 GB-h nets 720."""
    h1 = _vol("1980hnh75i", 30, datetime(2026, 10, 23, 18, 35, tzinfo=UTC), datetime(2026, 10, 24, 9, 57, tzinfo=UTC))
    pa = _vol("am5m373kuq", 30, datetime(2026, 10, 24, 19, 1, tzinfo=UTC))
    rows = [_row("2026-10-23", 210), _row("2026-10-24", 540), _row("2026-10-25", 870), _row("2026-10-26", 300)]
    n = MG.netted_posted(pa, rows, [h1, pa], datetime(2026, 10, 26, 9, 28, tzinfo=UTC))
    by = {d["day"]: d for d in n["days"]}
    assert "2026-10-23" not in by
    assert by["2026-10-24"]["netted_gb_h"] == 0 and by["2026-10-24"]["shared_with"] == ["1980hnh75i"]
    assert by["2026-10-25"]["netted_gb_h"] == 720 and by["2026-10-25"]["row_gb_h"] == 870
    assert by["2026-10-26"]["netted_gb_h"] == 300
    assert n["netted_gb_h"] == 1020 and n["netted_usd"] == pytest.approx(1020 * 0.07 / 720, abs=1e-6)


def _snapshot(*, pods=(), vols=("am5m373kuq",), pod_rows=(), vol_rows=(), fetched="2026-10-26T09:28:51+00:00"):
    return {"fetched_utc": fetched, "pods": list(pod_rows), "networkvolumes": list(vol_rows),
            "account": {"balance_known": True, "resources_known": True, "under_balance": False,
                        "pods": [{"id": p} for p in pods], "network_volumes": [{"id": v} for v in vols],
                        "client_balance_usd": 50.0, "current_spend_per_hr": 0.003}}


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / MG.GRANT).parent.mkdir(parents=True)
    shutil.copy(REPO / MG.GRANT, root / MG.GRANT)
    store = root / "logs/stages/stage-2/reasoning_data/storage"
    store.mkdir(parents=True)
    shutil.copy(REPO / "logs/stages/stage-2/reasoning_data/storage/h1pa_volume_am5m373kuq.json", store)
    return root


def _session(root: Path, name: str, pod: str, usd: float, ended: datetime):
    run = root / "logs/stages/stage-2/reasoning_data/runs" / name
    (run / "runtime").mkdir(parents=True)
    (run / "closeout").mkdir()
    (run / "runtime/session.json").write_text(json.dumps({"pod_id": pod, "provider_resource_created": True,
                                                          "provider_confirms_gone": True}))
    (run / "closeout/outcome.json").write_text(json.dumps({"recorded_utc": ended.isoformat(),
                                                           "money": {"all_in_usd": usd}}))


def test_position_counts_unposted_inside_the_window_and_lists_older(tmp_path):
    root = _repo(tmp_path)
    now = datetime(2026, 10, 26, 9, 28, 51, tzinfo=UTC)
    _session(root, "recent", "podA", 1.5, now - timedelta(hours=3))
    _session(root, "old", "podB", 0.83, now - timedelta(days=20))
    _session(root, "posted", "podC", 9.9, now - timedelta(hours=1))
    snap = _snapshot(pod_rows=[{"podId": "podC", "amount": 9.9}], vol_rows=[_row("2026-10-26", 300)])
    pos = MG.position(root, snapshot=snap)
    assert [u["pod_id"] for u in pos["unposted_sessions"]] == ["podA"]
    assert [u["pod_id"] for u in pos["never_posted_sessions_not_counted"]] == ["podB"]
    am = next(s for s in pos["storage"] if s["volume"] == "am5m373kuq")
    assert am["future_usd"] == pytest.approx(4.2 - 300 * 0.07 / 720, abs=1e-4)
    assert pos["committed_usd"] == pytest.approx(9.9 + 300 * 0.07 / 720 + 1.5 + am["future_usd"], abs=1e-3)


def test_position_refuses_pods_foreign_volumes_and_unknown_reads(tmp_path):
    from aadistill.governance.authorization import AuthorizationError
    root = _repo(tmp_path)
    with pytest.raises(AuthorizationError, match="at most one billing pod"):
        MG.position(root, snapshot=_snapshot(pods=("live",)))
    with pytest.raises(AuthorizationError, match="no live record owns"):
        MG.position(root, snapshot=_snapshot(vols=("am5m373kuq", "stranger")))
    bad = _snapshot()
    bad["account"]["balance_known"] = False
    with pytest.raises(AuthorizationError, match="cannot be confirmed"):
        MG.position(root, snapshot=bad)
    assert MG.position(root, snapshot=_snapshot(pods=("mine",)), expect_pods=("mine",))["account_pods"] == []


def _env(sessions=(), storage=(), remaining=33.05):
    return {"sessions": list(sessions), "storage": list(storage), "remaining_usd": remaining}


def test_first_funded_issue_holds_the_whole_envelope_then_the_session_cap():
    g = MG.grant(REPO)
    ok = {"committed_usd": 706.9114, "project_cap_usd": 740.0, "balance_usd": 56.0}
    assert MG.funding_problems("search", 15.0, _env(), ok, g) == ([], True)
    over = {**ok, "committed_usd": 707.0}
    probs, first = MG.funding_problems("search", 15.0, _env(), over, g)
    assert first and any("HARD BOUNDARY" in p and "whole remaining envelope" in p for p in probs)
    #: a $0 dry run does not make the next issue a later one
    assert MG.funding_problems("search", 15.0, _env(sessions=[{"usd": 0.0, "phase": "MC1"}]), over, g)[1] is True
    later = _env(sessions=[{"usd": 6.0, "phase": "MC1"}], storage=[{"usd": 1.05}], remaining=26.0)
    assert MG.funding_problems("recovery", 14.0, later, {**ok, "committed_usd": 714.0}, g) == ([], False)
    probs, _ = MG.funding_problems("recovery", 14.0, later, {**ok, "committed_usd": 726.5}, g)
    assert any("this session" in p for p in probs)


def test_an_mc1_retry_must_leave_mc2_fundable():
    g = MG.grant(REPO)
    pos = {"committed_usd": 700.0, "project_cap_usd": 740.0, "balance_usd": 56.0}
    env = _env(sessions=[{"usd": 15.0, "phase": "MC1"}], storage=[{"usd": 1.05}], remaining=17.0)
    probs, _ = MG.funding_problems("search", 4.0, env, pos, g)
    assert any("kept for MC2" in p for p in probs)
    assert MG.funding_problems("search", 3.0, env, pos, g)[0] == []


def test_both_worst_cases_fit_their_caps_at_the_priced_rates():
    from stages.reasoning_data import h1_generation_launch as GL
    g = MG.grant(REPO)
    plan = G.plan_doc(MG.SEARCH_PLAN)
    mc1 = ML.mc1_budget(plan, "vol").plan(price_per_hour=1.79, authorized_usd=15.0)
    assert GL.all_in(mc1, 1.79, ML.DEFAULTS["MC1"]["disk_gb"])["all_in_usd"] <= g["phases"]["MC1"]["session_hard_cap_usd"]
    assert mc1.hard_terminate_minutes >= sum(plan["budget_phases"].values())
    args = SimpleNamespace(unit=None)
    mc2 = ML.mc2_budget(args).plan(price_per_hour=1.09, authorized_usd=14.0)
    assert GL.all_in(mc2, 1.09, ML.DEFAULTS["MC2"]["disk_gb"])["all_in_usd"] <= g["phases"]["MC2"]["session_hard_cap_usd"]
    m = ML.measured_recovery_minutes()
    assert mc2.hard_terminate_minutes >= 2 * (m["train"] + m["evaluate"]) * ML.RECOVERY_BOUND_FACTOR


def _recovery_driver():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mixed_recovery_driver", REPO / "scripts/stages/stage-0/calibration_volume/mixed_recovery_driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stand_in_plan(tmp_path, M):
    st = json.loads(M.STUDY.read_text())
    st["initialization"]["record"] = "logs/stages/stage-2/reasoning_data/assets/d4_top1_materialization.json"
    p = tmp_path / "mixed_recovery.json"
    p.write_text(json.dumps(st))
    return M.load_plan(p)


def _pack(tmp_path: Path, n_blocks=2468) -> Path:
    d = tmp_path / "packs/e1_2p96M_plus_fluid"
    d.mkdir(parents=True)
    (d / "ladder.json").write_text(json.dumps({"rungs": [{"n_blocks": n_blocks, "target_supervised_tokens": 5923021}]}))
    return tmp_path / "packs"


def test_mix_augmented_is_d4_augmented_with_only_the_initialization_changed(tmp_path):
    M = _recovery_driver()
    plan = _stand_in_plan(tmp_path, M)
    base = json.loads((REPO / "configs/stages/stage-2/reasoning_data/h1pa.json").read_text())
    arm = plan["study"]["arms"]["Mix-Augmented"]
    ctl = base["arms"]["D4-Augmented"]
    assert {k: v for k, v in arm.items() if k not in ("initialization", "execution")} == \
        {k: v for k, v in ctl.items() if k not in ("initialization", "execution")}
    assert [u["unit"] for u in plan["units"]] == ["Mix-Augmented-s20260726", "Mix-Augmented-s20260801"]
    assert M.control_refusals(plan, _pack(tmp_path)) == []
    #: a set with another sequence count would train another schedule: refused before anything trains
    other = tmp_path / "other"
    assert M.control_refusals(plan, _pack(other, n_blocks=2400))


def test_a_changed_recipe_is_refused(tmp_path):
    M = _recovery_driver()
    plan = _stand_in_plan(tmp_path, M)
    plan["recipes"][20260801]["optim"]["lr"] = 1e-4
    bad = M.control_refusals(plan, _pack(tmp_path))
    assert len(bad) == 1 and "Mix-Augmented-s20260801" in bad[0] and "recipe identity" in bad[0]


def test_h1pa_schedule_rule_is_unchanged(tmp_path):
    from stages.reasoning_data import h1pa_driver as HD
    M = _recovery_driver()
    plan = _stand_in_plan(tmp_path, M)
    packs = _pack(tmp_path)
    h1pa_like = {**plan, "study": {k: v for k, v in plan["study"].items() if k != "derived_schedule_arms"}}
    h1pa_like["study"]["arms"] = {**plan["study"]["arms"], "D4-Augmented": plan["study"]["arms"]["Mix-Augmented"],
                                  "D4-Original": {"dataset": "e1_2p96M"}}
    assert HD.schedule_for(h1pa_like, "D4-Original", packs) is None
    assert HD.schedule_for(h1pa_like, "Mix-Augmented", packs) is None
    assert HD.schedule_for(h1pa_like, "D4-Augmented", packs)["total_steps"] == 3702


def test_the_stage0_programme_never_books_this_studys_runs(tmp_path):
    runs = tmp_path / G.EXPERIMENT_LOGS / "runs"
    for name, usd in (("dose_x", 2.0), ("mixs_1", 5.0)):
        (runs / name / "closeout").mkdir(parents=True)
        (runs / name / "closeout/outcome.json").write_text(json.dumps({"money": {"all_in_usd": usd}}))
    assert G.programme_spent(tmp_path)["spent_usd"] == 2.0


def _fake_ctx_store(tmp_path: Path, phase: str) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    run = root / MG.run_dir(f"mix{'s' if phase == 'MC1' else 'r'}_t")
    (run / "runtime").mkdir(parents=True)
    (run / "runtime/session.json").write_text(json.dumps({
        "provider_resource_created": True, "pod_id": "p1", "provider_confirms_gone": True, "terminal": "ALL_DONE",
        "passed": True, "required_products_secured": {"ok": True},
        "cost": {"actual_usd": 5.5, "elapsed_minutes": 200.0, "price_per_hour": 1.79, "authorized_usd": 15.0}}))
    scr = tmp_path / "scr"
    st = scr / "store"
    if phase == "MC1":
        (st / "mc1").mkdir(parents=True)
        (st / "mc1/mixed_search_record.json").write_text(json.dumps({"status": "COMPLETE"}))
    else:
        st.mkdir(parents=True)
        (st / "driver.status").write_text("x MARKER:ALL_DONE\n")
        import tarfile
        src = tmp_path / "unit"
        (src / "report").mkdir(parents=True)
        (src / "evaluation/rows").mkdir(parents=True)
        (src / "evaluation/generations").mkdir(parents=True)
        (src / "report/report.json").write_text("{}")
        (src / "evaluation/rows/fluid_reasoning_720__dev_v1.rows.jsonl").write_text("{}\n")
        (src / "evaluation/generations/x.jsonl").write_text("a reply\n")
        t = st / "Mix-Augmented-s20260726.evidence.tgz"
        with tarfile.open(t, "w:gz") as tf:
            tf.add(src, arcname=".")
        t.with_suffix(".sha256").write_text(hashlib.sha256(t.read_bytes()).hexdigest() + "\n")
    return root, scr


@pytest.mark.parametrize("phase,disk", [("MC1", 160), ("MC2", 150)])
def test_success_closeouts_execute_before_any_paid_success(tmp_path, phase, disk):
    root, scr = _fake_ctx_store(tmp_path, phase)
    out = ML.closeout(phase, f"mix{'s' if phase == 'MC1' else 'r'}_t", scr, repo_root=root, disk_gb=disk,
                      reconcile=lambda: {"ok": True, "why": "stub"})
    assert out["status"] == "COMPLETE" and out["money"]["gpu_usd"] == 5.5
    assert out["money"]["all_in_usd"] == pytest.approx(5.5 + disk * 0.10 * 200 / (60 * 24 * 30.4375), abs=1e-4)
    landed = ML.land_evidence(phase, out["run_id"], scr, repo_root=root)
    if phase == "MC2":
        assert out["files"]["Mix-Augmented-s20260726.evidence.tgz"]["verified"] is True
        assert landed == {"landed": ["Mix-Augmented-s20260726"]}
        log = root / ML.UNITS_LOG / "Mix-Augmented-s20260726"
        assert (log / "report.json").is_file() and (log / "evaluation/rows/fluid_reasoning_720__dev_v1.rows.jsonl").is_file()
        assert not list(log.rglob("generations*")), "a generation never goes into git"
        assert (root / ML.EVIDENCE_MIX / "Mix-Augmented-s20260726/evaluation/generations/x.jsonl").is_file()
    else:
        assert "mc1/mixed_search_record.json" in landed["landed"]
    assert json.loads((root / MG.run_dir(out["run_id"]) / "closeout/outcome.json").read_text())["phase"] == phase


def test_top1_identity_matches_the_journal_row(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "msd", REPO / "scripts/stages/stage-0/calibration_volume/mixed_search_driver.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    unit = tmp_path / "2x"
    (unit / "search").mkdir(parents=True)
    art = {"artifact_digest": "a" * 64, "weights_digest": "w" * 64, "arch_signature": "s", "num_parameters": 1}
    (unit / "stage1_selection.json").write_text(json.dumps({"selected": [{"state_id": "S", "artifact_digest": "a" * 64,
                                                                          "checkpoint_path": "/x"}]}))
    (unit / "search/states.jsonl").write_text(json.dumps({"state_id": "S", "artifact": art, "checkpoint_sha256": "c"}) + "\n"
                                              + json.dumps({"state_id": "S", "artifact": {}}) + "\n")
    top, ident = D.top1_identity(unit)
    assert top["state_id"] == "S" and ident["weights_digest"] == "w" * 64 and ident["checkpoint_sha256"] == "c"
    (unit / "stage1_selection.json").write_text(json.dumps({"selected": [{"state_id": "S", "artifact_digest": "b" * 64}]}))
    with pytest.raises(RuntimeError, match="does not match"):
        D.top1_identity(unit)


D4_DIR = REPO / "artifacts/stages/stage-0/calibration_volume/candidates/d4/1301deb71685aa1ca5c7b6cf2c491ac4"


@pytest.mark.skipif(not (D4_DIR / "model.safetensors").is_file(), reason="D4 Top-1 bytes are a local artifact")
def test_read_back_rebuilds_a_real_checkpoints_identity(tmp_path, monkeypatch):
    """The read-back on real bytes: D4 Top-1's relayed files stand in for the MC1 Top-1."""
    import huggingface_hub
    rec = json.loads((REPO / "logs/stages/stage-2/reasoning_data/assets/d4_top1_materialization.json").read_text())
    ident = rec["materialization_identity"]
    manifest = json.loads((REPO / "logs/stages/stage-0/calibration_volume/runs/dose_d4_20261020_212310/evidence/d4/"
                                  "finalists/finalists_manifest.json").read_text())
    files = [f for f in manifest["files"] if f["state_id"] == ident["state_id"]]
    root = tmp_path / "repo"
    (root / MG.RECOVERY_STUDY).parent.mkdir(parents=True)
    shutil.copy(REPO / MG.RECOVERY_STUDY, root / MG.RECOVERY_STUDY)
    store = tmp_path / "store/2x"
    store.mkdir(parents=True)
    (store / "search_record.json").write_text("{}")
    (store / "top1_manifest.json").write_text(json.dumps({
        "study_id": "s", "run_id": "mixs_t", "search_run_id": "r", "search_config_hash": "h",
        "search_record_sha256": hashlib.sha256(b"{}").hexdigest(), "selection_sha256": "x",
        "selected": {"state_id": ident["state_id"]}, "relay_repo": "AlphaAvatar/aadistill-artifacts",
        "identity": {k: ident[k] for k in ("artifact_digest", "weights_digest", "config_sha256", "arch_signature",
                                           "num_parameters", "checkpoint_sha256")},
        "files": files}))

    def fake_download(repo, path, token=None, local_dir=None):
        src = D4_DIR / Path(path).name
        dst = Path(local_dir) / Path(path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        return str(dst)

    class FakeApi:
        def __init__(self, token=None):
            pass

        def get_paths_info(self, repo, paths, expand=True):
            return [SimpleNamespace(path=p, size=0, lfs=None, blob_id=None) for p in paths]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    out = ML.read_back_top1(tmp_path / "store", repo_root=root, token="t")
    got = json.loads(out.read_text())
    assert got["read_back"]["verified"] is True
    assert got["materialization_identity"]["artifact_digest"] == ident["artifact_digest"]
    assert [f["name"] for f in got["relay_files"]] == [f["name"] for f in files]


CONTROL = REPO / "artifacts/stages/stage-2/reasoning_data/h1pa/units/evidence"


@pytest.mark.skipif(not (CONTROL / "D4-Augmented-s20260726/report/report.json").is_file(),
                    reason="H1-PA's unit evidence is a local artifact")
def test_the_analysis_executes_before_any_result_exists(tmp_path, monkeypatch):
    """D4-Augmented's evidence stands in for Mix-Augmented: every reader, the bound rule
    and the full eval-set-1 table run; identical rows must not resolve M."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mixed_analysis", REPO / "scripts/stages/stage-0/calibration_volume/mixed_analysis.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    mix = tmp_path / "mix"
    mix.mkdir()
    for s in (20260726, 20260801):
        os.symlink(CONTROL / f"D4-Augmented-s{s}", mix / f"Mix-Augmented-s{s}")
    monkeypatch.setattr(A, "EVIDENCE_MIX", mix)
    real_driver = A._driver

    def driver():
        M = real_driver()
        st = json.loads(M.STUDY.read_text())
        st["initialization"]["record"] = "logs/stages/stage-2/reasoning_data/assets/d4_top1_materialization.json"
        p = tmp_path / "study.json"
        p.write_text(json.dumps(st))
        load = M.load_plan
        M.load_plan = lambda *a, **k: load(p)
        return M
    ctl = {s: json.loads((CONTROL / f"D4-Augmented-s{s}/report/report.json").read_text()) for s in (20260726, 20260801)}
    monkeypatch.setattr(A, "_driver", driver)
    monkeypatch.setattr(A, "expected_mix_inputs", lambda plan, u: ctl[u["seed"]]["inputs"])
    real_load = A.load_unit

    def load_unit(d, dec):
        ev = real_load(d, dec)
        if d.name.startswith("Mix"):
            ev["report"] = {**ev["report"], "unit": d.name}
        return ev
    monkeypatch.setattr(A, "load_unit", load_unit)
    doc = A.analyse()
    assert doc["decision"]["comparisons"]["M"]["primary"]["label"] == "NOT_RESOLVED"
    assert doc["decision"]["comparisons"]["E"]["label"] is None
    assert set(doc["eval_set_1_correct_and_usable"]) == set(doc["units"]) and len(doc["units"]) == 6
    assert all(doc["units"][n]["reused"] for n in doc["units"] if not n.startswith("Mix"))


def test_a_launch_refuses_an_authorization_outside_the_session_commit(tmp_path):
    """mixs_20261026_102914: the authorization was issued after the session commit and never reached the pod."""
    import subprocess as sp
    root = tmp_path / "repo"
    rel = MG.authorization_path("mixs_t")
    (root / rel).parent.mkdir(parents=True)
    git = lambda *a: sp.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=root,  # noqa: E731
                            check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (root / "README").write_text("x")
    git("add", "-A")
    git("commit", "-qm", "before the issue")
    before = git("rev-parse", "HEAD")
    (root / rel).write_text('{"issued": true}')
    ctx = lambda c: SimpleNamespace(args=SimpleNamespace(run_id="mixs_t", session_commit=c))  # noqa: E731
    assert ML.the_authorization_is_in_the_session_commit(ctx(before), repo_root=root)[0] is False
    git("add", "-A")
    git("commit", "-qm", "the session commit")
    assert ML.the_authorization_is_in_the_session_commit(ctx(git("rev-parse", "HEAD")), repo_root=root)[0] is True
    (root / rel).write_text('{"issued": "edited"}')
    assert ML.the_authorization_is_in_the_session_commit(ctx(git("rev-parse", "HEAD")), repo_root=root)[0] is False
