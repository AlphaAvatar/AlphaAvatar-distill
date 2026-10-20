"""A probe session across lost pods: what a replacement restores, what it brings
home, and that a retrained student never inherits another student's generations.

Real control flow, governance-free: only the GPU steps (recovery training, the
vLLM eval) and the identity lookups are stubbed. The $0 rehearsal of 2026-10-20
found both failures these pin: a restored unit came home without its recovery
records, and a student retrained after a failed restore was scored on the
previous student's fluid generations."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from stages.calibration_volume import launch

REAL_RUN = subprocess.run


@pytest.fixture
def probe(tmp_path, monkeypatch):
    from aadistill.evaluation import eval_registry as ER
    from shared import eval_sets as ES
    from stages.calibration_volume import governance as G
    from stages.calibration_volume import probe_driver as PD
    from stages.phase_c1.autoinit_c1_driver import trained_model_dir

    repo, vol = tmp_path / "repo", tmp_path / "vol"
    (repo / PD.PREREG).parent.mkdir(parents=True)
    shutil.copyfile(PD.REPO / PD.PREREG, repo / PD.PREREG)
    prereg = json.loads((repo / PD.PREREG).read_text())
    units = []
    for i in range(3):
        cand = repo / f"cand/{i}"
        cand.mkdir(parents=True)
        (cand / "model.safetensors").write_bytes(b"candidate %d" % i)
        units.append({"label": f"u{i}", "treatment": "d1", "rank": i + 1, "quality_order": i + 1,
                      "state_id": f"s{i}", "seed": 20260726, "candidate_dir": f"cand/{i}",
                      "checkpoint_sha256": hashlib.sha256(b"candidate %d" % i).hexdigest()})
    monkeypatch.setattr(PD, "REPO", repo)
    monkeypatch.setattr(G, "plan_doc", lambda name: {"units": units})
    monkeypatch.setattr(G, "plan_hash", lambda name: "h")
    monkeypatch.setattr(G, "load_for", lambda kind, path: SimpleNamespace(
        authorization_id="a", require_plan=lambda h: None))
    monkeypatch.setattr(ER, "current_identity",
                        lambda e: prereg["evaluation"]["eval_set_identity"])
    monkeypatch.setattr(ER, "entry", lambda reg, set_id: None)
    monkeypatch.setattr(ES, "registry", lambda: None)
    monkeypatch.setattr(ES, "manifest", lambda set_id: None)
    monkeypatch.setattr(PD, "gpu", lambda: {})
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: (
        SimpleNamespace(returncode=0) if cmd[0] == "mountpoint" else REAL_RUN(cmd, *a, **k)))
    calls, fail = [], {"label": None}

    def train(cand, rec_out, cfg, log, *, env, run_name, seed, purpose):
        calls.append(("train", run_name))
        cfg.write_text(json.dumps({"seed": seed}) + "\n")
        log.write_text("recovery\n")
        shutil.rmtree(rec_out, ignore_errors=True)
        (rec_out / "checkpoints/t/model").mkdir(parents=True)
        (rec_out / "checkpoints/t/model/model.safetensors").write_bytes(
            f"{run_name} {len(calls)}".encode())
        (rec_out / "checkpoints/latest.txt").write_text("t\n")
        (rec_out / "train_log.jsonl").write_text("{}\n")
        return trained_model_dir(rec_out)

    def evaluate(model_dir, out, vol_gens, label, prereg, env):
        calls.append(("eval", label))
        gens = out / "generations"
        gens.mkdir(exist_ok=True)
        PD.sync("pull", vol_gens, gens, out / "sync.log")
        runs = []
        if not (gens / "fluid.jsonl").is_file():
            (gens / "fluid.jsonl").write_text((model_dir / "model.safetensors").read_text())
            PD.sync("push", gens, vol_gens, out / "sync.log")
            runs.append("fluid")
        (out / "eval_fluid.log").write_text("fluid\n")
        if label == fail["label"]:
            raise RuntimeError("lost the pod in the capability eval")
        (gens / "cap.jsonl").write_text((model_dir / "model.safetensors").read_text())
        (out / "eval_capability.log").write_text("cap\n")
        (out / "rows").mkdir(exist_ok=True)
        (out / "rows/r.jsonl").write_text("{}\n")
        return {"units": {"fluid_reasoning_720/dev_v1": {"correct": 1}}}, runs + ["capability"]

    monkeypatch.setattr(PD, "train_recovery", train)
    monkeypatch.setattr(PD, "evaluate_all", evaluate)

    def session(fail_label=None):
        """One pod: its local disk starts empty, the volume persists."""
        shutil.rmtree(repo / "artifacts", ignore_errors=True)
        fail["label"] = fail_label
        calls.clear()
        rc = PD.main(["--run-id", "r", "--auth", "a.json", "--plan", "p",
                      "--status", str(tmp_path / "status"), "--volume", str(vol)])
        audit = repo / PD.AUDIT
        return rc, json.loads((audit / "probe_session.json").read_text()), list(calls), audit

    return SimpleNamespace(session=session, unit_vol=lambda label: (
        vol / "stage0_calvol" / prereg["study_id"] / "probes" / label))


def _fetchable(audit):
    """Every unit marked done holds what the launcher will fetch for it."""
    for done in sorted((audit / "records").glob("*.done")):
        out = audit / done.name[:-5]
        assert [f for f in launch.PROBE_UNIT_FILES if not (out / f).exists()] == [], out.name
        assert hashlib.sha256((out / "report.json").read_bytes()).hexdigest() == done.read_text().strip()


def test_a_replacement_session_restores_and_brings_every_unit_home(probe):
    rc, rec, _, _ = probe.session(fail_label="u1")
    assert rc == 1 and rec["units"]["u0"]["status"] == "ok"

    rc, rec, calls, audit = probe.session(fail_label="u2")
    assert rec["units"]["u0"]["restored_complete"] is True
    assert rec["units"]["u1"]["recovery"] == "restored from the volume"
    assert rec["units"]["u1"]["generated"] == ["capability"]      # fluid rescored, not regenerated
    assert ("train", "u0") not in calls and ("train", "u1") not in calls
    assert sorted(p.name for p in (audit / "records").glob("*.done")) == ["u0.done", "u1.done"]
    _fetchable(audit)


def test_a_retrained_student_never_inherits_the_previous_students_generations(probe):
    probe.session(fail_label="u1")
    model = next(probe.unit_vol("u1").glob("recovery/checkpoints/*/model/model.safetensors"))
    model.write_bytes(b"cut short")
    rc, rec, calls, audit = probe.session()
    assert rc == 0 and rec["status"] == "COMPLETE"
    u1 = rec["units"]["u1"]
    assert "recovery_restore_failed" in u1 and ("train", "u1") in calls
    assert u1["generated"] == ["fluid", "capability"]
    gens = audit / "u1/generations"
    assert (gens / "fluid.jsonl").read_text() == (gens / "cap.jsonl").read_text()
    assert (probe.unit_vol("u1") / u1["generations_superseded"] / "fluid.jsonl").is_file()
    _fetchable(audit)


def test_report_json_reaches_the_volume_after_the_rest_of_the_unit(probe, monkeypatch):
    """report.json is what makes a replacement treat a unit as finished, so it
    lands only once everything else of the unit already has."""
    from stages.calibration_volume import probe_driver as PD
    pushes, real = [], PD.sync

    def sync(mode, src, dst, log):
        if mode == "push" and dst == probe.unit_vol("u0") / "out":
            pushes.append(sorted(str(p.relative_to(src)) for p in src.rglob("*") if p.is_file()))
        return real(mode, src, dst, log)

    monkeypatch.setattr(PD, "sync", sync)
    probe.session()
    assert pushes[-1] == ["report.json"]
    assert all("report.json" not in p for p in pushes[:-1])
    assert {"recovery_config.json", "recovery.log", "train_log.jsonl", "rows/r.jsonl"} <= set(
        sum(pushes[:-1], []))
