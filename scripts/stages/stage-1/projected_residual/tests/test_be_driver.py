"""B-E's driver: the parent/child orchestration on the host, and the verdict that re-derives every check.

The host path runs the real orchestration -- the PyTorch checks in a child
process that exits, the engine step NOT_RUN without a GPU -- on a random small
teacher. What only a GPU can answer (the L40S numerics, the engine) is tested
through `verdict` on evidence edited to look like a device run, and through
`vllm_phase` with a fake engine child: a value past a fixed threshold, a card
other than the configured one, or an engine that echoes other prompt ids must
each turn that check to FAIL.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from shared.validation import vllm_check as VC
from stages.projected_residual import engineering_validation as EV

CFG = json.loads((EV.REPO / EV.CONFIG).read_text())
CLEAR = {"known": True, "name": "NVIDIA L40S", "driver": "580.0", "used_mib": 3, "total_mib": 46068,
         "compute_processes": []}


@pytest.fixture(scope="module")
def host_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("be")
    rep = EV.run(d / "ev", d / "work", run_id="host", device="cpu", tiny=True, status=str(d / "status"))
    return d, rep


def test_the_host_rehearsal_runs_every_check_through_the_child(host_run):
    d, rep = host_run
    c = rep["checks"]
    assert rep["passed"] and rep["study"] == "B-E" and rep["scientific_use"] is False
    assert {k: v["status"] for k, v in c.items()} == {
        "initialization": "PASS", "structural_fold": "PASS", "fold_cuda": "NOT_RUN", "trainer_smoke": "PASS",
        "resume": "PASS", "checkpoint_identity": "PASS", "gpu_isolation": "NOT_RUN", "vllm": "NOT_RUN"}
    # the CPU numbers are a rehearsal, never presented as the CUDA result
    assert c["fold_cuda"]["device_name"] == "cpu" and c["fold_cuda"]["rehearsal_criterion_met"]
    assert c["gpu_isolation"]["applicable"] is False
    smoke = c["trainer_smoke"]
    assert smoke["projections_with_a_gradient"] == smoke["projections"] > 0 and smoke["embedding_unchanged"]
    assert smoke["orthogonality_rel_fro_max"]["after"] > smoke["orthogonality_rel_fro_max"]["before"]
    assert c["resume"]["max_abs_param_diff"] == 0.0 and c["resume"]["basis"] == "bit-identical"
    # the declared basis is measured against the outputs' subspaces too
    layer = json.loads((d / "ev/materialization.json").read_text())["per_layer"][0]
    assert {"attn_out_energy_under_declared_basis", "ffn_out_energy_under_declared_basis",
            "residual_in_energy_under_declared_basis"} <= set(layer)
    # one terminal marker; the child exited; weights gone, evidence small
    assert (d / "status").read_text().count("MARKER:") == 1 and "MARKER:ALL_DONE" in (d / "status").read_text()
    assert json.loads((d / "ev/isolation.json").read_text())["torch_child"]["returncode"] == 0
    assert not any((d / "work").iterdir())
    assert not list((d / "ev").rglob("*.safetensors"))
    assert rep["config"]["sha256"] and json.loads((d / "ev/report.json").read_text()) == rep


def test_the_verdict_is_rederived_from_the_evidence_alone(host_run):
    d, rep = host_run
    assert EV.verdict(d / "ev", CFG)["checks"] == rep["checks"]


def _as_device_run(src: Path, dst: Path) -> Path:
    """The host evidence edited to look like a run on the configured card with the fixed probe."""
    shutil.copytree(src, dst)
    t = json.loads((dst / "torch.json").read_text())
    t.update(tiny=False, device_name="NVIDIA L40S")
    rows = t["measurements"]["probe"]["rows"]
    ids = [f"{i['component']}/{i['unit']}/{i['item_id']}" for i in CFG["probe"]["items"]]
    t["measurements"]["probe"]["rows"] = [{**r, "id": i, "token_ids_sha256": it["token_ids_sha256"]}
                                          for r, i, it in zip(rows * 3, ids, CFG["probe"]["items"])]
    (dst / "torch.json").write_text(json.dumps(t))
    return dst


def _edit(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def test_the_device_bound_fold_check_passes_and_fails_on_the_fixed_criteria(host_run, tmp_path):
    d, _ = host_run
    ev = _as_device_run(d / "ev", tmp_path / "ev")
    v = EV.verdict(ev, CFG)
    assert v["checks"]["fold_cuda"]["status"] == "PASS"
    # a device session whose engine never ran is not a pass
    assert v["checks"]["vllm"]["status"] == "NOT_RUN" and v["passed"] is False

    f = lambda doc: doc["measurements"]["fold_cuda"]  # noqa: E731
    _edit(ev / "torch.json", lambda t: f(t).update(max_abs_logit_diff_bf16=2.01 * f(t)["bf16_vs_fp32_same_model"]))
    assert EV.verdict(ev, CFG)["checks"]["fold_cuda"]["status"] == "FAIL"
    _edit(ev / "torch.json", lambda t: f(t).update(max_abs_logit_diff_bf16=0.0))
    assert EV.verdict(ev, CFG)["checks"]["fold_cuda"]["status"] == "PASS"
    _edit(ev / "torch.json", lambda t: t.update(device_name="NVIDIA A100-SXM4-80GB"))
    assert "not the NVIDIA L40S" in " ".join(EV.verdict(ev, CFG)["checks"]["fold_cuda"]["problems"])
    _edit(ev / "torch.json", lambda t: (t.update(device_name="NVIDIA L40S"),
                                        t["numeric_identity_applied"].update(matmul_allow_tf32=True)))
    assert EV.verdict(ev, CFG)["checks"]["fold_cuda"]["status"] == "FAIL"
    _edit(ev / "torch.json", lambda t: (t["numeric_identity_applied"].update(matmul_allow_tf32=False),
                                        t["measurements"]["probe"]["rows"][0].update(token_ids_sha256="0" * 64)))
    assert "differ from the fixed probe" in " ".join(EV.verdict(ev, CFG)["checks"]["fold_cuda"]["problems"])


FAKE_ENGINE = r'''
import json, sys
from pathlib import Path
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
side = json.loads((Path(__file__).parent / "fake_side.json").read_text())
req, ref = json.loads(Path(a["--request"]).read_text()), {r["id"]: r for r in side["ref"]}
rows = []
for q in req["requests"]:
    ids, h = list(q["prompt_token_ids"]), ref[q["id"]]
    pl = [None] + [[[h["argmax"][i - 1], 0.0, 1], [ids[i], h["realized_logprob"][i - 1] + side["shift"], 2]]
                   for i in range(1, len(ids))]
    echo = ids[:-1] + [ids[-1] + 1] if side["bad_echo"] else ids
    rows.append({"id": q["id"], "prompt_token_ids": echo, "max_tokens": 4, "prompt_logprobs": pl,
                 "outputs": [{"token_ids": [1, 2], "finish_reason": "length", "stop_reason": None,
                              "cumulative_logprob": -1.0, "logprobs": [[[1, -0.5, 1]], [[2, -0.5, 1]]]}]})
Path(a["--out"]).write_text(json.dumps({"schema": "aadistill.vllm_check/v1", "versions": {"vllm": "fake"},
    "device": "NVIDIA L40S", "engine_args": {"max_model_len": req["max_model_len"]},
    "load_seconds": 0.0, "generate_seconds": 0.0, "requests": rows}))
'''


def _engine_run(ev: Path, tmp_path: Path, monkeypatch, *, shift=0.0, bad_echo=False, gpu=CLEAR) -> dict:
    """The prepared request through vllm_phase with a fake engine child, on a GPU that reads `gpu`."""
    ids = {r["id"]: i for i, r in enumerate(json.loads((ev / "torch.json").read_text())
                                            ["measurements"]["probe"]["rows"])}
    names = list(ids)
    # the tiny request and reference carry the tiny ids; give them the device run's (the probe rows')
    _edit(ev / "vllm_request.prepared.json",
          lambda q: [r.update(id=names[k]) for k, r in enumerate(q["requests"])])
    _edit(ev / "hf_reference.json", lambda h: [r.update(id=names[k]) for k, r in enumerate(h["prompts"])])
    fake = tmp_path / "fake_engine.py"
    fake.write_text(FAKE_ENGINE)
    (tmp_path / "fake_side.json").write_text(json.dumps({"ref": json.loads((ev / "hf_reference.json").read_text())
                                                         ["prompts"], "shift": shift, "bad_echo": bad_echo}))
    monkeypatch.setattr(VC, "gpu_state", lambda index=0: dict(gpu))
    (ev / "isolation.json").write_text(json.dumps({"torch_child": {"pid": 1, "returncode": 0},
                                                   "parent_imported_torch": False}))
    EV.vllm_phase(ev, CFG, sys.executable, script=fake)
    return EV.verdict(ev, {**CFG, "vllm": {**CFG["vllm"], "criteria": {**CFG["vllm"]["criteria"],
                                                                       "confident_margin_nats": 0.0}}})


def test_the_engine_check_against_the_reference_with_a_fake_engine(host_run, tmp_path, monkeypatch):
    d, _ = host_run
    ok = _engine_run(_as_device_run(d / "ev", tmp_path / "a"), tmp_path / "a", monkeypatch)
    assert ok["checks"]["gpu_isolation"]["status"] == "PASS"
    assert ok["checks"]["vllm"]["status"] == "PASS", ok["checks"]["vllm"]["problems"]
    assert ok["checks"]["vllm"]["median_abs_realized_logprob_delta"] == 0.0
    assert ok["passed"] is True

    far = _engine_run(_as_device_run(d / "ev", tmp_path / "b"), tmp_path / "b", monkeypatch, shift=0.5)
    assert far["checks"]["vllm"]["status"] == "FAIL"
    assert "median |delta|" in " ".join(far["checks"]["vllm"]["problems"])

    echo = _engine_run(_as_device_run(d / "ev", tmp_path / "c"), tmp_path / "c", monkeypatch, bad_echo=True)
    assert echo["checks"]["vllm"]["status"] == "FAIL"
    assert "prompt ids are not the request's" in " ".join(echo["checks"]["vllm"]["problems"])


def test_a_gpu_still_held_refuses_the_engine(host_run, tmp_path, monkeypatch):
    d, _ = host_run
    busy = {**CLEAR, "used_mib": 9000, "compute_processes": [{"pid": 42, "used_mib": 9000}]}
    v = _engine_run(_as_device_run(d / "ev", tmp_path / "a"), tmp_path / "a", monkeypatch, gpu=busy)
    assert json.loads((tmp_path / "a/vllm.json").read_text())["engine_started"] is False
    assert v["checks"]["gpu_isolation"]["status"] == "FAIL" and v["checks"]["vllm"]["status"] == "FAIL"
    assert any("compute processes" in p for p in v["checks"]["gpu_isolation"]["problems"])


def test_the_fixed_probe_renders_to_its_recorded_ids():
    """The probe the config fixes is what the pod will render (needs the cached tokenizer)."""
    try:
        rows, ids = EV.render_probe(CFG)
    except OSError as exc:
        pytest.skip(f"the teacher's tokenizer is not cached here: {exc}")
    assert EV._probe_problems({"measurements": {"probe": {"rows": rows}}}, CFG) == []
    assert [len(x) for x in ids] == [i["tokens"] for i in CFG["probe"]["items"]]


def test_a_smoke_that_ran_off_the_requested_device_fails(host_run, tmp_path):
    """`Trainer(device=...)` is the only way the trainer reaches the GPU; it ignores cfg["device"] and
    defaults to the CPU, so a smoke on the CPU of a GPU pod is recorded and refused, never timed as the card's."""
    d, rep = host_run
    assert rep["checks"]["trainer_smoke"]["on_the_requested_device"] is True
    ev = tmp_path / "ev"
    shutil.copytree(d / "ev", ev)

    def off_device(doc):
        doc["measurements"]["trainer_smoke"].update(requested_device="cuda:0", trainer_device="cpu",
                                                    student_device="cpu")
    _edit(ev / "torch.json", off_device)
    c = EV.verdict(ev, CFG)["checks"]["trainer_smoke"]
    assert c["status"] == "FAIL" and c["on_the_requested_device"] is False
