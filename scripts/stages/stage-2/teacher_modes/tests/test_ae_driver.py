"""A-E's driver end to end on the host, and its verdict's re-derivation ($0).

The host path runs the real orchestration: the parent renders, calls the engine
through `vllm_check.run_isolated` (a stand-in engine script here, the GPU read
faked clear), builds documents, and runs the teacher and train phases as child
processes on CPU with random small models at the real vocabulary. `verdict`
must reproduce `report.json` from the evidence alone, and a mutated evidence
file must flip its check to FAIL.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from shared.validation import vllm_check as VC
from stages.teacher_modes import engineering_validation as EV

CLEAR = {"known": True, "name": "stand-in", "driver": "0", "used_mib": 0, "total_mib": 1, "compute_processes": []}

#: a stand-in for the vLLM child: echoes the prompt ids, writes the mode's well-formed turn, and answers a
#: teacher-forced request with the tiny teacher's own fp32 log-probabilities (so the engine check can pass)
ENGINE = r'''
import json, sys
from pathlib import Path
req = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text())
out = Path(sys.argv[sys.argv.index("--out") + 1])
R, A = [25139, 287, 911, 279, 5912, 13], [785, 4226, 374, 1124, 79075, 90, 16, 17, 7810]
model = None
rows = []
for r in req["requests"]:
    ids, s = r["prompt_token_ids"], dict(r["sampling"])
    mt = s.get("max_tokens") if s.get("max_tokens") is not None else req["max_model_len"] - len(ids)
    row = {"id": r["id"], "prompt_token_ids": ids, "max_tokens": mt, "prompt_logprobs": None, "outputs": []}
    if r["id"].startswith("gen:"):
        o = ([151667, 198, *R, 198, 151668, 271, *A, 151645] if r["id"].startswith("gen:H-T") else [*A, 151645])
        row["outputs"] = [{"token_ids": o, "finish_reason": "stop", "stop_reason": None,
                           "cumulative_logprob": None, "logprobs": None}]
    else:
        import torch
        from transformers import AutoModelForCausalLM
        model = model or AutoModelForCausalLM.from_pretrained(req["model"], dtype=torch.float32,
                                                              attn_implementation="sdpa").eval()
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids]), use_cache=False).logits[0], -1)
        row["prompt_logprobs"] = [None] + [[[ids[i], float(lp[i - 1, ids[i]]), 1]] for i in range(1, len(ids))]
        row["outputs"] = [{"token_ids": [0], "finish_reason": "length", "stop_reason": None,
                           "cumulative_logprob": None, "logprobs": None}]
    rows.append(row)
out.write_text(json.dumps({"schema": "aadistill.vllm_check/v1", "versions": {"vllm": "stand-in"},
                           "device": "cpu", "requests": rows}))
'''


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("ae")
    script = root / "engine.py"
    script.write_text(ENGINE)
    mp = pytest.MonkeyPatch()
    mp.setattr(VC, "gpu_state", lambda index=0: dict(CLEAR))
    try:
        rc = EV.main(["--run-id", "ae_tiny", "--out", str(root / "out"), "--work", str(root / "work"),
                      "--status", str(root / "status"), "--tiny", "--vllm-python", sys.executable,
                      "--engine-script", str(script)])
    finally:
        mp.undo()
    return root, rc


def test_the_tiny_run_passes_every_check_and_writes_its_marker(tiny_run):
    root, rc = tiny_run
    rep = json.loads((root / "out/report.json").read_text())
    assert rc == 0 and rep["passed"] and rep["study"] == "A-E" and rep["scientific_use"] is False
    assert {k: v["status"] for k, v in rep["checks"].items()} == dict.fromkeys(EV.CHECKS, "PASS")
    assert (root / "status").read_text().strip().endswith("MARKER:ALL_DONE")
    assert (root / "status").read_text().count("MARKER:") == 1
    tf = rep["checks"]["teacher_forward"]["per_mode"]
    assert all(tf[m]["engine"]["mean_abs"] <= tf[m]["engine"]["limit"] for m in EV.MODES)
    gen = rep["checks"]["generation"]["per_mode"]
    assert all(gen[m]["documents"] == 4 for m in EV.MODES) and gen["H-T"]["ht_opened_think"] == gen["H-T"]["turns"] == 8
    # the parent never held the engine's GPU: the torch work ran in children, each recorded
    ph = json.loads((root / "out/phases.json").read_text())
    assert ph["teacher"]["returncode"] == 0 and ph["train"]["returncode"] == 0
    assert rep["checks"]["trainer_smoke"]["documents_synthetic"] == [False]


def test_the_verdict_reproduces_the_report_from_the_evidence(tiny_run):
    root, _ = tiny_run
    cfg = EV.load_config()
    assert json.loads(json.dumps(EV.verdict(root / "out", cfg))) == json.loads((root / "out/report.json").read_text())


def _mutated(root: Path, tmp_path: Path, name: str, edit) -> dict:
    out = tmp_path / "out"
    import shutil
    shutil.copytree(root / "out", out)
    doc = json.loads((out / EV.EV[name]).read_text())
    edit(doc)
    (out / EV.EV[name]).write_text(json.dumps(doc))
    return EV.verdict(out, EV.load_config())


def test_an_hn_output_holding_a_think_delimiter_fails_generation(tiny_run, tmp_path):
    def edit(doc):
        r = next(r for r in doc["engine"]["result"]["requests"] if r["id"].startswith("gen:H-N:t0"))
        r["outputs"][0]["token_ids"].insert(0, 151668)
    rep = _mutated(tiny_run[0], tmp_path, "generation", edit)
    g = rep["checks"]["generation"]
    assert g["status"] == "FAIL" and any("think delimiter" in p for p in g["problems"]) and not rep["passed"]


def test_a_packed_difference_above_its_bound_fails_teacher_forward(tiny_run, tmp_path):
    def edit(doc):
        m = doc["modes"]["H-T"]["sequences"][0]["members"][0]
        m["d_bf16"] = 10 * m["ref_bf16_alone"]
    rep = _mutated(tiny_run[0], tmp_path, "teacher_forward", edit)
    assert rep["checks"]["teacher_forward"]["status"] == "FAIL" and not rep["passed"]


def test_a_kd_position_on_the_system_block_fails_the_assistant_mask(tiny_run, tmp_path):
    def edit(doc):
        doc["modes"]["H-N"]["sequences"][0]["predicted"].insert(0, 0)
    rep = _mutated(tiny_run[0], tmp_path, "masks", edit)
    c = rep["checks"]["assistant_target_kl"]
    assert c["status"] == "FAIL" and any("generated-token layout" in p for p in c["problems"])


def test_a_rescored_record_that_differs_from_its_result_fails_scoring(tiny_run, tmp_path):
    def edit(doc):
        r = next(r for r in doc["records"] if r["mode"] == "H-N")
        r["raw"] = "</think>" + r["raw"]
    rep = _mutated(tiny_run[0], tmp_path, "scoring", edit)
    assert rep["checks"]["preclosed_scoring"]["status"] == "FAIL"


def test_a_smoke_that_ran_off_the_requested_device_fails(tiny_run, tmp_path):
    """The trainer reaches the GPU only through Trainer(device=...); a CPU smoke on a GPU pod is refused."""
    def edit(doc):
        doc.update(requested_device="cuda:0", trainer_device="cpu", student_device="cpu")
    rep = _mutated(tiny_run[0], tmp_path, "trainer_smoke", edit)
    c = rep["checks"]["trainer_smoke"]
    assert c["status"] == "FAIL" and any("not the requested" in p for p in c["problems"])


def test_a_busy_gpu_refuses_the_engine_and_the_later_checks_still_get_turns(tmp_path, monkeypatch):
    cfg = EV.load_config()
    busy = dict(CLEAR, compute_processes=[{"pid": 1, "used_mib": 9000}], used_mib=9000)
    monkeypatch.setattr(VC, "gpu_state", lambda index=0: busy)
    args = EV.build_parser().parse_args(["--run-id", "x", "--out", str(tmp_path), "--work", str(tmp_path),
                                         "--tiny", "--engine-script", "unused.py"])
    tok = EV.StandInTokenizer()
    prompts = EV.render_prompts(cfg, tok, tiny=True)
    gen = EV.generate(cfg, args, prompts, EV.sizes(cfg, True))
    assert gen["engine"]["status"] == "FAIL" and gen["engine"]["engine_started"] is False
    rows = EV.turns(cfg, prompts, gen)
    assert rows and all(r["synthetic"] for r in rows)
    docs, refused = EV.documents(cfg, prompts, rows)
    assert all(len(docs[m]) == len(cfg["t0"]["task_ids"]) for m in EV.MODES) and not refused


def test_the_stand_in_tokenizer_keeps_the_hybrid_generation_prompts():
    from aadistill.data.token_documents import generation_suffix
    cfg, tok = EV.load_config(), EV.StandInTokenizer()
    for mode in EV.MODES:
        spec = EV.mode_spec(cfg, mode)
        assert list(generation_suffix(tok, spec["template_kwargs"])) == spec["generation_suffix_ids"]


def test_the_config_binds_the_design_and_never_authorizes():
    cfg = EV.load_config()
    design = EV._json(cfg["design"])
    assert cfg["teacher"] == {k: design["teacher"][k] for k in ("model", "revision")} | {"_": cfg["teacher"]["_"]}
    assert {x["path"]: x["sha256"] for x in design["tasks"]["T0"]["lists"]}[cfg["t0"]["list"]] == cfg["t0"]["list_sha256"]
    assert cfg["modes"]["H-N"]["think_state"] == "preclosed" and cfg["authorizes"] == "nothing"
    smoke = copy.deepcopy(cfg["trainer_smoke"]["declared_changes"])
    assert "batch.context_strategy" in smoke and "loss.kd_scope" in smoke
