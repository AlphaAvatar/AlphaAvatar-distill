"""PA2v's checks on synthetic evidence: everything the gate needs passes, and each failure it guards fails.

`build_evidence` writes the files `h1pa_validation.main` leaves under
``evidence/`` -- real derived recipes, per-step train logs, generations whose
seeds are the protocol's -- so `checks` runs exactly as the gate runs it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.evaluation import decoding as DEC
from stages.reasoning_data import h1pa_driver as D
from stages.reasoning_data import h1pa_validation as V


@pytest.fixture(scope="module")
def plan():
    return D.load_plan()


def _w(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)


def build_evidence(ev: Path, plan: dict, tmp: Path) -> Path:
    gv, st = plan["study"]["gpu_validation"], plan["study"]["storage"]
    proto = DEC.load(V.REPO / plan["prereg"]["evaluation"]["decoding"]["path"])
    _w(ev / "gpu.json", {"name": "NVIDIA L40S", "total_gb": 47.7})
    _w(ev / "runs.json", {"X1": [{"rc": 0, "killed": False}], "X2": [{"rc": 0, "killed": False}],
                          "Y": [{"rc": -9, "killed": True}, {"rc": 0, "killed": False}], "O": [{"rc": 0, "killed": False}]})
    for name, (arm, steps) in V.RUNS.items():
        cfg = V.validation_recipe(plan, arm, steps=steps, save_every=gv["training"]["save_every"] if name != "O" else 0,
                                  out_dir=tmp / name, data_dir=tmp / "packs", init=tmp / "init", name=f"t_{name}")
        _w(ev / "training" / name / "recovery_config.json", cfg)
        rows = [{"event": "train_step", "step": s, "lr": 1e-5 * s, "executed_positions": 16384,
                 "executed_nonpad_tokens": 9000, "supervised_tokens": 5000, "kd_positions": 9000,
                 "loss": 2.0 + 1e-3 * s + (1e-4 if name == "X2" else 0.0), "ce": 1.0, "kd": 1.5, "grad_norm": 1.0,
                 "gpu_mem_gb": 40.0, "seconds": 3.4} for s in range(1, steps + 1)]
        _w(ev / "training" / name / "train_log.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _w(ev / "initialization.json", {"reidentified": True})
    sha = plan["study"]["arms"]["PCA-Original"]["checkpoints"]["20260726"]["model_safetensors_sha256"]
    _w(ev / "pca_checkpoint.json", {"seed": 20260726, "sha256": sha})
    _w(ev / "weights.json", {"resumed": {"same_keys": True, "max_abs_diff": 1e-7},
                             "floor": {"same_keys": True, "max_abs_diff": 1e-6}})
    _w(ev / "persistence.json", {"reidentified_from_landed_bytes": True, "save_point_bytes": st["bytes"]["save_point"]})
    _w(ev / "probe.json", {"bytes": gv["volume_probe_bytes"], "read_back_ok": True,
                           "provider_bytes": st["volume_gb"] * 10 ** 9})
    g = ev / "evaluation" / "generations"
    fluid = [{"task_id": f"f{i}", "turns": [{"turn": 0, "seed": DEC.request_seed(proto, V.FLUID_KEY, f"f{i}", 0),
                                             "stop_reason": "eos", "output_token_ids": [1, 2]}]} for i in range(3)]
    _w(g / f"{V.FLUID_KEY}.jsonl", "".join(json.dumps(r) + "\n" for r in fluid))
    _w(g / f"{V.FLUID_KEY}.summary.json", {"identity": {"decoding_protocol": {"sha256": proto["sha256"]},
                                                        "sampling": DEC.sampling(proto)}})
    gsm = [{"id": f"g{i}", "sampling_seed": DEC.request_seed(proto, V.GSM8K_KEY, f"g{i}"), "stop_reason": "degeneration",
            "output_token_ids": [3]} for i in range(3)]
    _w(g / f"{V.GSM8K_KEY}.jsonl", "".join(json.dumps(r) + "\n" for r in gsm))
    _w(g / f"{V.GSM8K_KEY}.summary.json", {"decoding_protocol": {"sha256": proto["sha256"]},
                                           "sampling": {**DEC.sampling(proto), "max_tokens_rule": "context - prompt"}})
    _w(ev / "evaluation" / "timing.json", {"seconds": {"fluid": 300.0, "gsm8k": 120.0}})
    return ev


def _edit(path: Path, fn) -> None:
    rows = path.read_text().splitlines()
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(fn(json.loads(r))) + "\n" for r in rows))
    else:
        path.write_text(json.dumps(fn(json.loads(path.read_text()))))


def test_complete_evidence_passes_and_measures_the_step(plan, tmp_path):
    r = V.checks(build_evidence(tmp_path / "ev", plan, tmp_path), plan)
    assert r["passed"], {k: v for k, v in r["checks"].items() if not v.get("ok")}
    assert r["measured"]["median_step_seconds"] == 3.4 and r["measured"]["peak_gpu_mem_gb"] == 40.0
    assert all(v["ok"] for v in r["checks"]["config"]["formal_recipes"].values())


MUTATIONS = {
    "gpu": ("gpu.json", lambda d: {**d, "name": "NVIDIA A100-SXM4-80GB"}),
    "memory": ("training/X1/train_log.jsonl", lambda r: {**r, "gpu_mem_gb": 46.0}),
    "training": ("training/O/train_log.jsonl", lambda r: {**r, "loss": float("nan")}),
    "config": ("training/X1/recovery_config.json", lambda d: {**d, "optim": {**d["optim"], "lr": 1e-4}}),
    "resume": ("training/Y/train_log.jsonl", lambda r: {**r, "lr": r["lr"] * (2 if r["step"] == 15 else 1)}),
    "pca_checkpoint": ("pca_checkpoint.json", lambda d: {**d, "sha256": "0" * 64}),
    "persistence": ("probe.json", lambda d: {**d, "read_back_ok": False}),
    "capacity": ("probe.json", lambda d: {**d, "provider_bytes": 50 * 10 ** 9}),
    "decoding": ("evaluation/generations/fluid_reasoning_720__dev_v1.summary.json",
                 lambda d: {"identity": {**d["identity"], "sampling": {**d["identity"]["sampling"], "temperature": 0.0}}}),
    "initialization": ("initialization.json", lambda d: {"reidentified": False}),
}


@pytest.mark.parametrize("check", sorted(MUTATIONS))
def test_each_guarded_failure_fails_its_check_and_the_gate(plan, tmp_path, check):
    ev = build_evidence(tmp_path / "ev", plan, tmp_path)
    f, fn = MUTATIONS[check]
    _edit(ev / f, fn)
    r = V.checks(ev, plan)
    assert not r["checks"][check]["ok"] and not r["passed"]


def test_a_seed_that_is_not_the_protocols_or_a_missing_stop_reason_fails(plan, tmp_path):
    ev = build_evidence(tmp_path / "ev", plan, tmp_path)
    _edit(ev / f"evaluation/generations/{V.GSM8K_KEY}.jsonl", lambda r: {**r, "sampling_seed": 7})
    assert V.checks(ev, plan)["checks"]["decoding"]["seed_mismatch"]
    ev2 = build_evidence(tmp_path / "ev2", plan, tmp_path)
    _edit(ev2 / f"evaluation/generations/{V.FLUID_KEY}.jsonl",
          lambda r: {**r, "turns": [{**t, "stop_reason": "length_limited"} for t in r["turns"]]})
    assert V.checks(ev2, plan)["checks"]["decoding"]["bad_stop_reason"]


def test_a_save_point_larger_than_planned_fails_capacity(plan, tmp_path):
    ev = build_evidence(tmp_path / "ev", plan, tmp_path)
    _edit(ev / "persistence.json", lambda d: {**d, "save_point_bytes": int(d["save_point_bytes"] * 1.1)})
    assert not V.checks(ev, plan)["checks"]["capacity"]["ok"]


def test_the_validation_recipe_changes_only_the_declared_keys(plan, tmp_path):
    cfg = V.validation_recipe(plan, "D4-Augmented", steps=24, save_every=12, out_dir=tmp_path, data_dir=tmp_path,
                              init=tmp_path, name="n")
    assert V.recipe_outside(plan, cfg) == []
    assert cfg["batch"] == plan["recipes"][20260726]["batch"] and cfg["optim"] == plan["recipes"][20260726]["optim"]
    assert cfg["intervals"]["log_every"] == 1 and cfg["checkpoint"]["save_every"] == 12


def _leftovers(pa2v_root: Path, run: str) -> None:
    for rel in V.VALIDATION_BYTES:
        (pa2v_root / run / rel).mkdir(parents=True)
        (pa2v_root / run / rel / "model.safetensors").write_bytes(b"x" * 10)
    _w(pa2v_root / run / V.REPORT, {"run_id": run})
    _w(pa2v_root / run / "evidence" / "gpu.json", {"name": "NVIDIA L40S"})


def test_pa2v_releases_validation_bytes_on_a_failed_path_and_an_earlier_attempts(tmp_path):
    # h1pav_20261024_192448 failed at evaluation, before its release, and left 8.29 GB on the volume
    durable = tmp_path / "vol" / "h1pa"
    root = durable / "validation" / "pa2v"
    _leftovers(root, "h1pav_earlier")
    rc = V.main(["--run-id", "h1pav_now", "--durable-root", str(durable), "--volume-mount", str(tmp_path / "vol"),
                 "--volume-bytes", str(30 * 10 ** 9), "--status", str(tmp_path / "status"),
                 "--pack-root", str(tmp_path / "no_packs"), "--work", str(tmp_path / "work")])
    assert rc == 3 and "RUN_FAILED" in (tmp_path / "status").read_text()
    rep = json.loads((root / "h1pav_now" / V.REPORT).read_text())
    assert rep["error"]["stage"] == "plan"
    assert sorted(r["path"] for r in rep["released_earlier_validation_bytes"]) == \
        sorted(str(root / "h1pav_earlier" / rel) for rel in V.VALIDATION_BYTES)
    for rel in V.VALIDATION_BYTES:
        assert not (root / "h1pav_earlier" / rel).exists()
    assert (root / "h1pav_earlier" / V.REPORT).is_file() and (root / "h1pav_earlier" / "evidence/gpu.json").is_file()


def test_the_release_runs_in_the_terminal_path_of_every_run(tmp_path, monkeypatch):
    durable = tmp_path / "vol" / "h1pa"
    root = durable / "validation" / "pa2v"
    calls = []
    real = V.release_validation_bytes
    monkeypatch.setattr(V, "release_validation_bytes", lambda r: calls.append(Path(r)) or real(r))
    V.main(["--run-id", "h1pav_now", "--durable-root", str(durable), "--volume-mount", str(tmp_path / "vol"),
            "--volume-bytes", str(30 * 10 ** 9), "--status", str(tmp_path / "status"),
            "--pack-root", str(tmp_path / "no_packs"), "--work", str(tmp_path / "work")])
    assert calls == [root, root]
