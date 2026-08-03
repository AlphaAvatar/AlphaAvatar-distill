"""Tests for the Phase 1 throughput gate.

The gate decides, from the first D0 endpoint's own telemetry, whether the
execution-path correction actually worked — before the second D0 endpoint runs
and before either D1 training run starts. A gate that cannot fail is not a gate,
so every stop condition is tested firing and not firing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "pod"))
from throughput_gate import (  # noqa: E402
    BASELINE_TOK_S,
    STEP_MS_LIMIT,
    THRESHOLD_TOK_S,
    evaluate,
    load_sets,
)


def wave(name="gsm8k.json", out_tok=80_000, secs=40.0, p50=700, batch=37.0,
         step_ms=12.0, gpu=88):
    return {
        "file": name,
        "throughput": {
            "input_tokens": 8_000, "output_tokens": out_tok,
            "output_tokens_p50": p50, "output_tokens_p95": 1280,
            "output_tokens_max": 2048,
            "generation_wall_seconds": secs,
            "output_tokens_per_second": round(out_tok / secs, 1),
            "prompts_per_second": round(100 / secs, 4),
            "scheduler_steps": 2048, "step_ms_p50": step_ms, "step_ms_p95": 30.0,
            "effective_batch_size_mean": batch, "concurrency_max": 100,
        },
        "gpu": {"utilization_p50": gpu},
        "natural_termination_rate": 0.38, "degeneration_rate": 0.62,
        "context_limit_rate": 0.0, "right_censored_rate": 0.62,
        "engine": {"max_num_seqs": 256, "max_model_len": 8192},
    }


def run(sets, min_gpu=40.0, require=True):
    return evaluate(sets, min_gpu, require)


# --------------------------------------------------------------------------
# the healthy case
# --------------------------------------------------------------------------
def test_a_fixed_path_passes():
    r = run([wave()])
    assert r["verdict"] == "pass" and not r["failures"]
    assert r["aggregate_output_tokens_per_second"] == 2000.0
    assert r["speedup_vs_baseline"] > 7


def test_aggregate_is_summed_across_sets_not_averaged():
    r = run([wave("a.json", 60_000, 30.0), wave("b.json", 40_000, 20.0)])
    assert r["total_output_tokens"] == 100_000
    assert r["total_generation_wall_seconds"] == 50.0
    assert r["aggregate_output_tokens_per_second"] == 2000.0


# --------------------------------------------------------------------------
# condition 1 — throughput unchanged
# --------------------------------------------------------------------------
def test_throughput_at_the_old_rate_fails():
    """Exactly the Experiment 1 result: 209,850 tok / 823.5 s."""
    r = run([wave(out_tok=209_850, secs=823.5, step_ms=167.0)])
    assert r["verdict"] == "fail"
    assert any("within 20%" in f for f in r["failures"])


def test_throughput_just_below_the_limit_fails():
    r = run([wave(out_tok=int(THRESHOLD_TOK_S * 100), secs=100.0, step_ms=12.0)])
    assert r["verdict"] == "fail"


def test_throughput_just_above_the_limit_passes_condition_1():
    r = run([wave(out_tok=int((THRESHOLD_TOK_S + 1) * 100), secs=100.0,
                  step_ms=12.0)])
    assert not any("within 20%" in f for f in r["failures"])


# --------------------------------------------------------------------------
# condition 2 — step time on a comparable wave
# --------------------------------------------------------------------------
def test_slow_steps_on_a_comparable_wave_fail_even_at_high_throughput():
    """Throughput can be carried by a big batch while each step is still slow."""
    r = run([wave(out_tok=500_000, secs=100.0, p50=700, batch=37.0,
                  step_ms=150.0)])
    assert r["verdict"] == "fail"
    assert any("median step" in f for f in r["failures"])


def test_slow_steps_outside_the_comparable_regime_do_not_fire():
    """A short-output wave at batch 4 is not the regime the baseline measured."""
    r = run([wave(out_tok=500_000, secs=100.0, p50=50, batch=4.0, step_ms=150.0)])
    assert r["verdict"] == "pass"
    assert any("no wave matched" in n for n in r["notes"])


@pytest.mark.parametrize("batch,fires", [(19.0, False), (20.0, True),
                                         (37.0, True), (60.0, True),
                                         (61.0, False)])
def test_the_comparable_batch_window_is_closed_at_both_ends(batch, fires):
    r = run([wave(p50=700, batch=batch, step_ms=150.0, out_tok=500_000,
                  secs=100.0)])
    assert any("median step" in f for f in r["failures"]) is fires


def test_step_time_exactly_at_the_limit_fails():
    r = run([wave(out_tok=500_000, secs=100.0, step_ms=STEP_MS_LIMIT)])
    assert any("median step" in f for f in r["failures"])


def test_a_comparable_wave_without_a_step_median_is_noted_not_silently_passed():
    w = wave(out_tok=500_000, secs=100.0)
    w["throughput"]["step_ms_p50"] = None
    r = run([w])
    assert any("no step-time median" in n for n in r["notes"])


# --------------------------------------------------------------------------
# condition 3 — GPU starvation and telemetry
# --------------------------------------------------------------------------
def test_starved_gpu_fails():
    r = run([wave(out_tok=500_000, secs=100.0, gpu=12)])
    assert r["verdict"] == "fail"
    assert any("starved" in f for f in r["failures"])


def test_missing_telemetry_fails_when_required():
    w = wave(out_tok=500_000, secs=100.0)
    w.pop("gpu")
    assert run([w], require=True)["verdict"] == "fail"


def test_missing_telemetry_is_only_a_note_when_explicitly_allowed():
    w = wave(out_tok=500_000, secs=100.0)
    w.pop("gpu")
    r = run([w], require=False)
    assert r["verdict"] == "pass"
    assert any("telemetry" in n for n in r["notes"])


def test_the_worst_set_decides_gpu_utilization():
    r = run([wave("a.json", 400_000, 80.0, gpu=90),
             wave("b.json", 100_000, 20.0, gpu=15)])
    assert r["verdict"] == "fail" and r["gpu_utilization_p50_min"] == 15


# --------------------------------------------------------------------------
# several conditions at once, and the file loader
# --------------------------------------------------------------------------
def test_every_failing_condition_is_reported_not_just_the_first():
    r = run([wave(out_tok=209_850, secs=823.5, step_ms=167.0, gpu=9)])
    assert len(r["failures"]) == 3


def test_loader_reads_instrumented_files_and_skips_the_gate_output(tmp_path):
    (tmp_path / "gsm8k.json").write_text(json.dumps(wave()))
    (tmp_path / "knowledge.json").write_text(json.dumps(wave("knowledge.json")))
    (tmp_path / "throughput_gate.json").write_text(json.dumps({"verdict": "pass"}))
    (tmp_path / "notes.json").write_text(json.dumps({"unrelated": True}))
    (tmp_path / "broken.json").write_text("{not json")
    got = load_sets(tmp_path)
    assert sorted(s["file"] for s in got) == ["gsm8k.json", "knowledge.json"]


def test_baseline_and_threshold_match_the_preregistered_numbers():
    assert BASELINE_TOK_S == 254.8
    assert THRESHOLD_TOK_S == 306.0
    assert STEP_MS_LIMIT == 100.0
