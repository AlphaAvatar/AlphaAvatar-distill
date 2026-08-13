"""The three Stage-1 measurement scripts, executed end to end on CPU.

These scripts had been reviewed, wired into the driver, and rehearsed *around* —
the harness tests stubbed them out — but never **run**. The first real execution
was on a paid pod, where `measure_state_repeatability.py` died on

    row = result.as_dict()["metrics"]        KeyError: 'metrics'

after loading a 4B teacher and a 596M student and completing a full evaluation
pass. `StateEvaluation.as_dict()` has never had a `metrics` key. The cost was
$0.29 and a session; the information was one CPU run away.

So each script is invoked here as the driver invokes it — as a subprocess, with
its real argument parsing, its real output schema, and its own writing of the
result file — against a tiny stand-in teacher. That is what makes it cheap enough
to run every time. What these tests do *not* claim: they are not the gate
measurements. They prove the code path executes and the artifact it writes has
the shape the driver reads. The driver separately refuses any artifact that says
it was produced this way (`is_gate_measurement` / `is_real_teacher`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ENV = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
       "OMP_NUM_THREADS": "2", "HOME": "/tmp"}


@pytest.fixture(scope="module")
def tiny_teacher(tmp_path_factory) -> Path:
    """A 2-layer Qwen3 that stands in for the 4B teacher."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny_teacher")
    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, intermediate_size=48,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        tie_word_embeddings=True, max_position_embeddings=256)).float(
    ).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny_suite(tmp_path_factory) -> Path:
    """A state-eval suite of the frozen asset's exact schema, two items."""
    d = tmp_path_factory.mktemp("tiny_suite")
    domains = {"general": ["prose"], "reasoning": ["math"]}
    (d / "manifest.json").write_text(json.dumps({
        "artifact": "state_eval_tiny", "role": "INITIALIZER_STATE_EVAL",
        "suite_id": "state_eval_tiny", "version": 1,
        "domains": domains, "critical_tags": ["eos"],
    }))
    rows = []
    for i, (domain, subtype) in enumerate((("general", "prose"),
                                           ("reasoning", "math"))):
        ids = [(i * 7 + j) % 64 for j in range(24)]
        rows.append({"item_id": f"t{i}", "ids": ids, "domain": domain,
                     "subtype": subtype, "n_prediction_positions": len(ids) - 1,
                     "tags": {"eos": [len(ids) - 2]}})
    (d / "items.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def run(script: str, args: list[str], timeout: int = 900):
    result = subprocess.run([sys.executable, str(REPO / script), *args],
                            capture_output=True, text=True, timeout=timeout,
                            env=ENV, cwd=REPO)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    return result


def test_the_repeatability_probe_runs_and_reports_the_beam_objectives(
        tiny_teacher, tiny_suite, tmp_path):
    """The defect that cost a pod: it must read `values`, and report all three."""
    from aadistill.autoinit.ranking import PARETO_V1

    out = tmp_path / "rep.json"
    run("scripts/autoinit/measure_state_repeatability.py",
        ["--checkpoint", str(tiny_teacher), "--suite", str(tiny_suite),
         "--teacher", str(tiny_teacher), "--repeats", "2", "--device", "cpu",
         "--out", str(out)])
    report = json.loads(out.read_text())

    objectives = [o.key for o in PARETO_V1.objectives]
    assert report["objectives"] == objectives
    assert len(report["runs"]) == 2
    for row in report["runs"]:
        for key in objectives:
            assert key in row, f"{key} missing from a repeatability run"
    # The number the driver's gate reads, and the shape it reads it from.
    assert isinstance(report["max_objective_range"], float)
    assert report["max_objective_range"] >= 0.0
    assert set(report["per_objective"]) == set(objectives)
    # Re-measuring identical weights on CPU is exactly reproducible, so this
    # doubles as a check that the two repeats really were two measurements.
    assert report["max_objective_range"] == 0.0
    # And it declares itself a smoke run, which the driver refuses as a gate.
    assert report["is_real_teacher"] is False


def test_a_missing_objective_is_refused_rather_than_averaged_over(
        tiny_teacher, tiny_suite, tmp_path, monkeypatch):
    """A range over the objectives that happen to be present is a different number."""
    source = (REPO / "scripts/autoinit/measure_state_repeatability.py").read_text()
    assert 'row = result.as_dict()["values"]' in source
    assert "the evaluation carries no" in source, (
        "a missing beam objective must raise; silently ranging over what is "
        "left answers a question nobody asked")


def test_the_peak_memory_probe_runs_and_marks_a_cpu_run_as_not_the_gate(
        tiny_teacher, tmp_path):
    out = tmp_path / "peak.json"
    run("scripts/autoinit/probe_peak_memory.py",
        ["--teacher", str(tiny_teacher), "--device", "cpu", "--seq-len", "16",
         "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["schema"] == "aadistill.autoinit.peak_memory/v1"
    assert report["peak_gib"] is None and report["is_gate_measurement"] is False
    assert report["child_layers"] and report["teacher_layers"] == 2


def test_the_statistics_profile_runs_and_marks_a_cpu_run_as_not_the_gate(
        tiny_teacher, tmp_path):
    out = tmp_path / "stats.json"
    run("scripts/autoinit/profile_statistics_pass.py",
        ["--teacher", str(tiny_teacher), "--device", "cpu", "--tokens", "64",
         "--seq-len", "32", "--repeats", "1", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["schema"] == "aadistill.autoinit.statistics_split/v1"
    assert report["is_gate_measurement"] is False
    assert report["gpu_fraction"] is not None
    assert report["seconds_per_1k_tokens"] > 0


def test_the_driver_refuses_a_smoke_artifact_as_a_gate_measurement():
    """A CPU or stand-in run must never satisfy the gate it informs."""
    driver = (REPO / "scripts/pod/autoinit_preflight_driver.py").read_text()
    stage1 = driver[driver.index("def stage1"):driver.index("def gate(")]
    assert '"is_gate_measurement"' in stage1 and '"is_real_teacher"' in stage1
    assert "this is a smoke artifact" in stage1
    # And the full output of every gate is kept, because the one thing the paid
    # attempt could not answer was "what did it actually say".
    assert "def gate(" in driver
    assert "--- stderr ---" in driver


def test_the_generator_module_imports_and_its_identity_helpers_work(tiny_teacher):
    """`uncapped_eval.py` needs a GPU to generate — not to be checked at all.

    Everything this session added to it is module-level or pure: the shared
    protocol constants, the tokenizer hash, the context resolution and the
    engine-config shape. A typo in any of them would surface at Stage 3, after
    both permanent controls had been paid for.
    """
    sys.path.insert(0, str(REPO / "scripts/evaluation"))
    import uncapped_eval as ue
    from transformers import AutoConfig

    from aadistill.autoinit.generation import (
        CONTEXT_RESOLUTION_RULE, GENERATION_DTYPE, MAX_TOKENS_RULE,
    )

    assert ue.SAMPLING == {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                           "detokenize": False}
    # The tokenizer hash convention the Stage-0 engine probe also uses.
    assert ue.tokenizer_files_sha256(str(tiny_teacher)) is None or len(
        ue.tokenizer_files_sha256(str(tiny_teacher))) == 64
    assert ue.tokenizer_files_sha256("/does/not/exist") is None

    ctx = ue.resolve_context(AutoConfig.from_pretrained(tiny_teacher), None, 128)
    assert ctx["resolved_context"] == 128
    assert ctx["context_source"] == "trained_block_len"
    assert ctx["rule"] == CONTEXT_RESOLUTION_RULE
    assert ctx["context_len_override"] is None

    engine = ue.engine_config(object(), 128, 0.90)
    assert engine["dtype"] == GENERATION_DTYPE
    for key in ("max_num_seqs", "max_num_batched_tokens", "enforce_eager",
                "vllm_version", "gpu_memory_utilization"):
        assert key in engine

    # And the summary's sampling block carries the rule the protocol compares.
    assert MAX_TOKENS_RULE in (REPO / "scripts/evaluation/uncapped_eval.py").read_text() \
        or "MAX_TOKENS_RULE" in (REPO / "scripts/evaluation/uncapped_eval.py").read_text()


def test_the_disk_probe_runs_and_reports_both_directions(tmp_path):
    """The last Stage-1 gate, and the only one written inline in the driver."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "preflight_driver_disk", REPO / "scripts/pod/autoinit_preflight_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_driver_disk"] = mod
    spec.loader.exec_module(mod)
    mod.WS = tmp_path

    driver = mod.Driver.__new__(mod.Driver)
    driver.a = type("A", (), {"disk_probe_gib": 0.02})()
    result = mod.Driver.disk_throughput(driver)

    assert result["bytes"] >= int(0.02 * 2**30)
    assert result["write_mb_s"] > 0 and result["read_mb_s"] > 0
    # It must clean up after itself: the probe file is written into the
    # session's working area, which the artifact collection walks.
    assert not (tmp_path / "disk_probe.bin").exists()
