"""H1-PA's session governance and launcher: the grant, the live project cap, the PA2v gate, the volume, the price.

The grant does not exist until the maintainer decides; these tests write a
synthetic one, synthetic provider snapshots and a synthetic PA2v run under
``tmp_path`` (never in the repository).
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from aadistill.governance.authorization import AuthorizationError
from aadistill.infrastructure.manifest import sha256_file
from stages.reasoning_data import assets as A
from stages.reasoning_data import h1pa_driver as D
from stages.reasoning_data import h1pa_governance as HG
from stages.reasoning_data import h1pa_launch as L

from .test_h1pa_validation import MUTATIONS, _edit, build_evidence


def snap(billed=680.67, pods=(), volumes=(), **acct):
    return {"fetched_utc": "2026-10-25T00:00:00+00:00", "pods": [{"amount": billed, "podId": "old"}],
            "networkvolumes": [], "account": {"pods": list(pods), "network_volumes": list(volumes),
                                              "current_spend_per_hr": 0.0, "client_balance_usd": 78.7,
                                              "under_balance": False, "known": True, "balance_known": True,
                                              "resources_known": True, **acct}}


SNAP = snap()


def write_grant(tmp_path, **over):
    g = {"granted_by": "test", "ceiling_usd": 50.0, "project_cumulative_cap_usd": 731.0, "occupancy_basis_usd": 680.67,
         "phases": {"PA2v": {"scope": "test validation"}, "PA2": {"scope": "test formal"}}, **over}
    p = tmp_path / "grant.json"
    p.write_text(json.dumps(g))
    return p


@pytest.fixture(autouse=True)
def own_records(tmp_path, monkeypatch):
    """Runs and storage records under tmp_path, never the repository's real ones
    (an open real session must not decide a synthetic test)."""
    monkeypatch.setattr(HG, "EXPERIMENT_LOGS", tmp_path / HG.EXPERIMENT_LOGS)


@pytest.fixture
def grant(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "GRANT", write_grant(tmp_path))


def test_without_the_grant_nothing_is_issued(tmp_path, monkeypatch):
    monkeypatch.setattr(HG, "GRANT", tmp_path / "absent.json")
    with pytest.raises(AuthorizationError, match="not authorized"):
        HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "a.json", snapshot=SNAP)


def test_each_kind_loads_only_under_its_own_policy(tmp_path, grant):
    v = HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "v.json", snapshot=SNAP)
    assert HG.load_for("validation", v).plan_hash == HG.study_hash()
    with pytest.raises(AuthorizationError):
        HG.load_for("formal", v)


def test_a_cap_beyond_the_envelope_refuses(tmp_path, grant):
    with pytest.raises(AuthorizationError, match="does not fit the envelope"):
        HG.issue("validation", "t_f", expected_usd=40.0, hard_cap_usd=60.0, out=tmp_path / "f.json", snapshot=SNAP)


def test_the_project_cap_is_checked_against_the_providers_billing_now(tmp_path, grant):
    # the application's snapshot left $50.33; a late-posted $40 leaves $10.33 now
    with pytest.raises(AuthorizationError, match="as the provider bills it now"):
        HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=12.0, out=tmp_path / "a.json",
                 snapshot=snap(billed=720.67))
    with pytest.raises(AuthorizationError, match="does not own"):
        HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "a.json",
                 snapshot=snap(pods=[{"id": "stray"}]))


def test_a_closed_session_the_provider_has_not_posted_is_still_counted(tmp_path, grant):
    d = tmp_path / HG.run_dir("h1pa_closed")
    (d / "governance").mkdir(parents=True)
    (d / "governance/authorization.json").write_text(json.dumps({"plan_id": HG.PLAN_ID}))
    (d / "closeout").mkdir()
    (d / "closeout/outcome.json").write_text(json.dumps({"money": {"all_in_usd": 16.5}}))
    (d / "runtime").mkdir()
    (d / "runtime/session.json").write_text(json.dumps({"pod_id": "newpod", "provider_resource_created": True}))
    live = HG.live_position(tmp_path, snapshot=SNAP)
    assert live["unposted_sessions"] == [{"run_id": "h1pa_closed", "pod_id": "newpod", "usd": 16.5}]
    assert live["committed_usd"] == pytest.approx(680.67 + 16.5)
    posted = snap()
    posted["pods"].append({"amount": 16.4, "podId": "newpod"})
    assert HG.live_position(tmp_path, snapshot=posted)["unposted_sessions"] == []


UNCONFIRMED = {
    "resources unknown": lambda s: s["account"].update(resources_known=False, resources_error="HTTP 502"),
    "balance unknown": lambda s: s["account"].update(balance_known=False, balance_error="timeout"),
    "an old merged snapshot": lambda s: [s["account"].pop(k) for k in ("balance_known", "resources_known")],
    "under its minimum balance": lambda s: s["account"].update(under_balance=True),
    "an error body for billing": lambda s: s.update(pods={"error": "unauthorized"}),
    "no resource lists": lambda s: s["account"].pop("pods"),
}


@pytest.mark.parametrize("case", sorted(UNCONFIRMED))
def test_an_unconfirmed_provider_position_authorizes_nothing(tmp_path, grant, case):
    s = snap()
    UNCONFIRMED[case](s)
    with pytest.raises(AuthorizationError, match="cannot be confirmed"):
        HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "a.json", snapshot=s)
    ctx = SimpleNamespace(args=args("--phase", "PA2v"), auth=SimpleNamespace(hard_cap_usd=2.0), evidence={})
    assert not L.the_project_cap_holds_now(ctx, snapshot=s)[0]


def test_a_failed_provider_read_is_an_authorization_refusal(monkeypatch):
    from maintenance.consolidation import provider_billing as PB
    def boom(*_a, **_k):
        raise OSError("connection reset")
    monkeypatch.setattr(PB, "fetch", boom)
    monkeypatch.setattr("aadistill.infrastructure.provider.read_api_key", lambda _p: "k")
    with pytest.raises(AuthorizationError, match="could not be read"):
        HG.provider_snapshot()


def test_fetch_keeps_the_resource_listings_own_known_state(monkeypatch):
    from aadistill.infrastructure import provider as P
    from maintenance.consolidation import provider_billing as PB
    class Prov:
        def __init__(self, _k): pass
        def account_balance(self): return P.AccountBalance(known=True, client_balance_usd=78.7, under_balance=False)
        def account_resources(self): return P.AccountResources(known=False, error="HTTP 502")
    monkeypatch.setattr(P, "RunPodProvider", Prov)
    monkeypatch.setattr(PB.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b"[]"))
    acct = PB.fetch("k", "2026-10-01", "2026-10-25")["account"]
    assert acct["resources_known"] is False and acct["balance_known"] is True and acct["known"] is False
    assert HG.snapshot_problems({"account": acct, "pods": [], "networkvolumes": []})


def test_a_balance_below_the_hard_cap_refuses(tmp_path, grant):
    with pytest.raises(AuthorizationError, match="less than"):
        HG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=12.0, out=tmp_path / "a.json",
                 snapshot=snap(client_balance_usd=10.0))


# ---- the PA2v gate ---------------------------------------------------------------------------

def pa2v_run(root: Path, plan, *, rid="h1pav_t", mutate=None, other_code=None) -> Path:
    """A complete synthetic PA2v run under `root`: authorization, closeout, secured evidence.
    `other_code` (a monkeypatch) issues the validation artifact against a different harness."""
    d = root / HG.run_dir(rid)
    if other_code is not None:
        real = HG.harness_source_digest
        other_code.setattr(HG, "harness_source_digest", lambda *a, **k: {**real(*a, **k), "digest": "0" * 64})
    HG.issue("validation", rid, expected_usd=1.6, hard_cap_usd=3.75, out=d / "governance/authorization.json",
             snapshot=SNAP)
    if other_code is not None:
        other_code.undo()
    ev = build_evidence(root / "work/evidence", plan, root / "work")
    if mutate:
        _edit(ev / MUTATIONS[mutate][0], MUTATIONS[mutate][1])
    store = root / "store"
    store.mkdir(parents=True, exist_ok=True)
    tgz = store / HG.PA2V_EVIDENCE
    with tarfile.open(tgz, "w:gz") as t:
        rep = json.dumps({"passed": True}).encode()
        info = tarfile.TarInfo("./pa2v_report.json")
        info.size = len(rep)
        t.addfile(info, io.BytesIO(rep))
        t.add(ev, arcname="./evidence")
    (d / "closeout").mkdir(parents=True, exist_ok=True)
    (d / "closeout/outcome.json").write_text(json.dumps({
        "status": "COMPLETE", "provider_confirms_gone": True, "store": str(store),
        "evidence": {HG.PA2V_EVIDENCE: {"sha256": sha256_file(tgz), "verified": True}}}))
    HG.pa2v_gate(rid, repo_root=root)
    return d / "closeout/pa2v_gate.json"


@pytest.fixture(scope="module")
def plan():
    return D.load_plan()


def test_a_formal_session_needs_a_pa2v_gate(tmp_path, grant):
    with pytest.raises(AuthorizationError, match="no PA2v gate"):
        HG.issue("formal", "t_f", expected_usd=16.0, hard_cap_usd=26.0, out=tmp_path / "f.json", snapshot=SNAP)


def test_a_passed_gate_funds_a_formal_session_and_is_bound_into_it(tmp_path, grant, plan):
    gate = pa2v_run(tmp_path / "root", plan)
    assert json.loads(gate.read_text())["passed"] is True
    f = HG.issue("formal", "t_f", expected_usd=16.0, hard_cap_usd=26.0, out=tmp_path / "f.json", snapshot=SNAP,
                 pa2v=gate)
    named = json.loads(json.loads(f.read_text())["stage_conditions"]["pa2v_gate"])
    assert named["sha256"] == sha256_file(gate) and named["measured"]["median_step_seconds"] == 3.4


def test_a_failed_validation_never_funds_a_formal_session(tmp_path, grant, plan):
    gate = pa2v_run(tmp_path / "root", plan, mutate="memory")
    with pytest.raises(AuthorizationError, match="did not pass"):
        HG.issue("formal", "t_f", expected_usd=16.0, hard_cap_usd=26.0, out=tmp_path / "f.json", snapshot=SNAP,
                 pa2v=gate)


def test_a_gate_edited_by_hand_is_caught_by_its_re_derivation(tmp_path, grant, plan):
    gate = pa2v_run(tmp_path / "root", plan, mutate="decoding")
    g = json.loads(gate.read_text())
    gate.write_text(json.dumps({**g, "passed": True}))
    with pytest.raises(AuthorizationError, match="disagrees with its re-derivation"):
        HG.issue("formal", "t_f", expected_usd=16.0, hard_cap_usd=26.0, out=tmp_path / "f.json", snapshot=SNAP,
                 pa2v=gate)


def test_a_validation_of_other_code_does_not_fund_this_code(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(HG, "GRANT", write_grant(tmp_path))
    gate = pa2v_run(tmp_path / "root", plan, other_code=pytest.MonkeyPatch())
    with pytest.raises(AuthorizationError, match="the code changed since the validation"):
        HG.issue("formal", "t_f", expected_usd=16.0, hard_cap_usd=26.0, out=tmp_path / "f.json", snapshot=SNAP,
                 pa2v=gate)


def test_the_launch_re_checks_the_gate_the_authorization_names(tmp_path, grant, plan):
    gate = pa2v_run(tmp_path / "root", plan)
    f = HG.issue("formal", "t_f", expected_usd=16.0, hard_cap_usd=26.0, out=tmp_path / "f.json", snapshot=SNAP,
                 pa2v=gate)
    auth = HG.load_for("formal", f)
    ctx = SimpleNamespace(args=args("--phase", "PA2"), auth=auth, evidence={})
    assert L.the_pa2v_gate_holds(ctx)[0]
    gate.write_text(gate.read_text().replace('"run_id"', '"run_id" ', 1))       # same content, other bytes
    ok, why = L.the_pa2v_gate_holds(ctx)
    assert not ok and "changed after the authorization" in why


# ---- setup, launcher prechecks, pricing ----------------------------------------------------------

def test_setups_branch_binds_the_artifact_on_the_pods_own_files(tmp_path, grant):
    text = (A.REPO / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    body = text[text.index('elif [ "$SESSION_KIND" = "stage2_h1pa" ]'):]
    code = re.search(r'/opt/train/bin/python -c "\n(.*?)\n" \|\|', body, re.S).group(1)
    code = code.replace('\\"', '"').replace("\\$", "$")
    v = HG.issue("validation", "t_s", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "a.json", snapshot=SNAP)
    env = {**os.environ, "PYTHONPATH": "src:scripts", "SESSION_AUTH_PATH": str(v), "SESSION_PLAN_HASH": HG.study_hash()}
    r = subprocess.run([sys.executable, "-c", code], cwd=A.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "validation" in r.stdout, r.stderr[-800:]
    env["SESSION_PLAN_HASH"] = "0" * 64
    assert subprocess.run([sys.executable, "-c", code], cwd=A.REPO, env=env, capture_output=True).returncode != 0


def args(*extra):
    return L.build_parser().parse_args(["--run-id", "t", "--scr", "/tmp/x", "--session-commit", "d" * 40,
                                        "--bundle", "b.bundle", "--network-volume-id", "vol1",
                                        "--volume-mount-path", "/runpod-volume", "--data-center-ids", "US-TX-3", *extra])


class Provider:
    def __init__(self, vols):
        self.vols = vols

    def account_resources(self):
        return SimpleNamespace(known=True, network_volumes=tuple(self.vols), error=None)


def test_the_volume_is_taken_from_the_provider_and_must_be_the_planned_30_gb():
    assert args("--phase", "PA2v").volume_gb == 30 == L.study()["storage"]["volume_gb"]
    good = {"id": "vol1", "size": 30, "dataCenterId": "US-TX-3"}
    ctx = SimpleNamespace(args=args("--phase", "PA2v"), evidence={})
    assert L.the_volume_is_the_planned_one(ctx, Provider([good]))[0]
    assert L.volume_bytes(ctx) == 30 * 10 ** 9
    for bad, why in (({**good, "size": 50}, "not the planned 30"), ({**good, "dataCenterId": "EU-NL-1"}, "must name it")):
        ok, msg = L.the_volume_is_the_planned_one(SimpleNamespace(args=args("--phase", "PA2v"), evidence={}),
                                                  Provider([bad]))
        assert not ok and why in msg
    ok, msg = L.the_volume_is_the_planned_one(SimpleNamespace(args=args("--phase", "PA2v", "--volume-gb", "50"),
                                                              evidence={}), Provider([good]))
    assert not ok and "--volume-gb 50" in msg


def test_the_launch_re_reads_the_project_cap(tmp_path, grant):
    ctx = SimpleNamespace(args=args("--phase", "PA2v"), auth=SimpleNamespace(hard_cap_usd=26.0), evidence={})
    assert L.the_project_cap_holds_now(ctx, snapshot=SNAP)[0]
    ok, why = L.the_project_cap_holds_now(ctx, snapshot=snap(billed=710.0))
    assert not ok and "as the provider bills it now" in why


@pytest.mark.parametrize("measured", [3.3, 4.1])
def test_formal_training_is_priced_at_the_slower_of_committed_and_pa2v(tmp_path, measured):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"measured": {"median_step_seconds": measured}}))
    a = args("--phase", "PA2", "--pa2v-gate", str(gate), "--unit", "D4-Augmented-s20260726")
    pr = L.pricing(a)
    step = max(measured, 1.7836 * 2)
    assert pr["step_seconds"] == pytest.approx(step) and pr["training_fixed_minutes"] == 10.0
    t = {p.name: p.minutes for p in L.spec(a).budget.other_phases}
    assert t["D4-Augmented-s20260726_train"] == round(3702 * step / 60 + 10.0, 1)
    assert L.pricing(args("--phase", "PA2"))["step_seconds"] == pytest.approx(1.7836 * 2)


def test_a_measured_full_training_only_raises_its_arms_price(monkeypatch):
    price = {"step_seconds": 3.5672, "training_fixed_minutes": 10.0, "evaluation_minutes": 99.2,
             "training_measured_minutes": {"D4-Augmented": 400.0, "D4-Original": 60.0}}
    assert L.unit_minutes("D4-Augmented-s20260801", price)[0] == 400.0
    assert L.unit_minutes("D4-Original-s20260801", price)[0] == round(2916 * 3.5672 / 60 + 10.0, 1)
    assert L.unit_minutes("PCA-Original-s20260801", price)[0] == 0


def test_a_measured_formal_evaluation_reprices_the_next_session(tmp_path, monkeypatch):
    d = tmp_path / HG.run_dir("h1pa_s1")
    (d / "governance").mkdir(parents=True)
    (d / "governance/authorization.json").write_text(json.dumps({"plan_id": HG.PLAN_ID,
                                                                 "stage_conditions": {"kind": "formal"}}))
    (d / "closeout").mkdir()
    (d / "closeout/outcome.json").write_text(json.dumps({"units": {
        "PCA-Original-s20260726": {"seconds": {"evaluate": 3000.0}},
        "D4-Augmented-s20260726": {"seconds": {"train": 13000.0, "evaluate": 3600.0}}}}))
    assert L.measured_evaluation_seconds(tmp_path) == 3600.0
    assert L.measured_training_seconds(tmp_path) == {"D4-Augmented": 13000.0}
    monkeypatch.setattr(L, "measured_evaluation_seconds", lambda repo_root=None: 3600.0)
    assert L.pricing(args("--phase", "PA2"))["evaluation_minutes"] == round(3600.0 * 1.25 / 60, 1)


def test_a_session_stages_only_what_its_units_read():
    s = L.spec(args("--phase", "PA2", "--unit", "PCA-Original-s20260801", "--unit", "D4-Original-s20260801"))
    dests = {r.dest for r in s.setup.relay_inputs}
    assert f"{L.CHECKPOINTS}/20260801" in dests and f"{L.PACKS}/e1_2p96M" in dests
    assert f"{L.CHECKPOINTS}/20260726" not in dests and f"{L.PACKS}/e1_2p96M_plus_fluid" not in dests
    assert s.setup.env["SESSION_KIND"] == "stage2_h1pa"


def test_pa2v_stages_the_pack_of_every_arm_its_runs_train():
    # h1pav_20261024_190211: O trains D4-Original, gpu_validation.units names only D4-Augmented,
    # and the pod refused at the plan for the missing e1_2p96M pack
    from stages.reasoning_data import h1pa_validation as V
    dests = {r.dest for r in L.spec(args("--phase", "PA2v")).setup.relay_inputs}
    for arm, _ in V.RUNS.values():
        assert f"{L.PACKS}/{L.study()['arms'][arm]['dataset']}" in dests, arm
    assert f"{L.CHECKPOINTS}/20260726" in dests and f"{L.CHECKPOINTS}/20260801" not in dests


def test_a_pca_checkpoint_is_staged_without_its_tokenizer_sidecars():
    # h1pav_20261024_192448: the evaluation package refused E1's checkpoint for carrying its own
    # tokenizer files; they are the frozen sidecars byte for byte, so only the model files are staged
    from stages.phase_c1.autoinit_c1_driver import TOKENIZER_SIDECAR_SHA256
    from stages.phase_c1.packaging import EVAL_TOKENIZER_SIDECARS
    rel = L.spec(args("--phase", "PA2v")).setup.relay_inputs
    pca = [r.path.rsplit("/", 1)[1] for r in rel if r.dest == f"{L.CHECKPOINTS}/20260726"]
    assert "model.safetensors" in pca and "config.json" in pca
    assert not set(pca) & set(EVAL_TOKENIZER_SIDECARS)
    pins = L.study()["relay_inputs"]["pca_original"]
    for seed in pins:
        assert {n: pins[seed]["files"][n] for n in EVAL_TOKENIZER_SIDECARS} == TOKENIZER_SIDECAR_SHA256
    with pytest.raises(RuntimeError, match="not the frozen evaluation sidecars"):
        L.model_files_only({**pins["20260726"]["files"], "tokenizer.json": "0" * 64}, "20260726")


def test_the_driver_commands_take_the_providers_size():
    for phase, script in (("PA2v", "h1pa_validation.py"), ("PA2", "h1pa_driver.py")):
        a = args("--phase", phase)
        cmd = L.spec(a).driver_command(SimpleNamespace(args=a, evidence={"volume": {"size_gb": 30}}), None)
        assert script in cmd and f"--volume-bytes {30 * 10 ** 9}" in cmd and "MARKER:" in cmd
        with pytest.raises(RuntimeError, match="no provider-reported volume size"):
            L.spec(a).driver_command(SimpleNamespace(args=a, evidence={}), None)


def _unit_tgz(store: Path, name: str, report: dict, ckpt: dict | None) -> None:
    store.mkdir(parents=True, exist_ok=True)
    with tarfile.open(store / f"{name}.evidence.tgz", "w:gz") as tf:
        for arc, doc in (("./report/report.json", report), ("./checkpoint.descriptor.json", ckpt)):
            if doc is None:
                continue
            raw = json.dumps(doc).encode()
            info = tarfile.TarInfo(arc)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


def test_a_formal_closeout_records_each_units_seconds_and_kept_checkpoint(tmp_path, monkeypatch):
    store = tmp_path / "scr" / "store"
    _unit_tgz(store, "D4-Original-s20260726", {"execution": "train", "seconds": {"train": 11000.0, "evaluate": 5000.0}},
              {"artifact_digest": "a" * 64, "weights_digest": "w" * 64, "destination": "/v/h1pa/x", "persisted": True})
    _unit_tgz(store, "PCA-Original-s20260726", {"execution": "evaluate", "seconds": {"evaluate": 4800.0},
                                                "checkpoint": {"sha256": "p" * 64}}, None)
    _unit_tgz(store, "D4-Augmented-s20260726", {"execution": "train", "seconds": {"train": 1.0}}, {})
    verified = {f"{u}.evidence.tgz": {"verified": True} for u in ("D4-Original-s20260726", "PCA-Original-s20260726")}
    monkeypatch.setattr(L.HUL, "closeout", lambda *a, **k: {"evidence": verified, "money": {"all_in_usd": 1.0}})
    (tmp_path / HG.run_dir("h1pa_f") / "closeout").mkdir(parents=True)
    out = L.closeout("h1pa_f", tmp_path / "scr", repo_root=tmp_path, reconcile=lambda: {})
    assert set(out["units"]) == {"D4-Original-s20260726", "PCA-Original-s20260726"}   # an unverified archive is not read
    assert out["units"]["D4-Original-s20260726"]["checkpoint"]["weights_digest"] == "w" * 64
    assert out["units"]["PCA-Original-s20260726"]["seconds"] == {"evaluate": 4800.0}
