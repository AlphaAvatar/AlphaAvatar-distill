"""Research B's records agree with the code and the evidence they cite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aadistill.initialization.transforms.projected_residual import selection_problems
from aadistill.models.cost_model import parameter_counts

REPO = Path(__file__).resolve().parents[5]
DESIGN = REPO / "configs/stages/stage-1/projected_residual/design.json"
SELECTION = REPO / "configs/stages/stage-1/projected_residual/d4_top1_selection.json"
PREREG = REPO / "logs/stages/stage-1/projected_residual/plans/preregistration.json"
REPORT = REPO / "logs/stages/stage-1/projected_residual/analyses/architecture_report.json"
FORECAST = REPO / "logs/stages/stage-1/projected_residual/plans/cost_forecast.json"
PROFILES = REPO / "configs/stages/stage-0/calibration_volume/profiles_formal.json"


def load(p):
    return json.loads(p.read_text())


def test_the_selection_record_is_d4_top1s_and_describes_the_target():
    sel = load(SELECTION)
    body = {k: sel[k] for k in ("kept_layers", "kept_heads", "kept_neurons")}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == sel["content_sha256"]
    assert selection_problems(body, sel["parent_geometry"], sel["target_geometry"]) == []
    assert sel["depth_removed_teacher_layers"] == [8, 11, 12, 15, 18, 21, 25, 32]
    assert sel["source"]["state_id"] == "1301deb71685aa1ca5c7b6cf2c491ac4"
    assert [s["kind"] for s in sel["source"]["steps"]] == ["ATTENTION", "DEPTH", "RESIDUAL_WIDTH", "FFN"]


def test_the_design_binds_the_selection_the_profile_and_the_counted_parameters():
    d = load(DESIGN)
    assert d["selection"]["content_sha256"] == load(SELECTION)["content_sha256"]
    a = d["architecture"]
    geom = {k: a[k] for k in ("hidden_size", "internal_size", "intermediate_size", "num_hidden_layers",
                              "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings")}
    counts = parameter_counts({**geom, "vocab_size": 151936})
    assert a["num_parameters"] == counts["total"] and a["trainable_parameters"] == counts["non_embedding"]
    prof = next(p for p in load(PROFILES)["profiles"] if p["profile_id"] == "calib.s0v_base_d4")
    assert prof["items_file_sha256"] == d["calibration"]["items_file_sha256"]
    assert prof["content_sha256"] == d["calibration"]["content_sha256"]
    items = REPO / d["calibration"]["items_path"]
    if items.is_file():
        assert hashlib.sha256(items.read_bytes()).hexdigest() == d["calibration"]["items_file_sha256"]
    r6 = d["reserved_600m"]
    assert parameter_counts({**r6["geometry"], "vocab_size": 151936, "tie_word_embeddings": True})["total"] == r6["num_parameters"]


def test_the_preregistration_binds_the_rule_files_as_they_are():
    rule = load(PREREG)["decision_rule"]
    for path, sha in rule["bound_sha256"].items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == sha, path


def test_report_and_forecast_regenerate_identically():
    from stages.projected_residual import architecture_report, forecast_cost

    assert architecture_report.build() == load(REPORT)
    assert forecast_cost.build() == load(FORECAST)


def test_the_materialization_driver_runs_end_to_end_on_the_host(tmp_path):
    from stages.projected_residual import materialize

    rec = materialize.run({}, tmp_path, device="cpu", tiny=True)
    assert rec["num_parameters"] == rec["num_parameters_cost_model"]
    fc = rec["fold_check"]
    assert fc["max_abs_logit_diff_fp32"] < 1e-5
    assert fc["max_abs_logit_diff_bf16"] <= 2 * fc["bf16_vs_fp32_same_model"]
    assert (tmp_path / "model" / "config.json").is_file()
    assert json.loads((tmp_path / "model" / "config.json").read_text())["model_type"] == "aad_projected_residual"


def test_the_engineering_validation_driver_runs_every_check_on_the_host(tmp_path):
    """B-E's driver end to end on a random small teacher: the code path a paid
    session runs, short of CUDA and vLLM, which only the GPU can answer."""
    from stages.projected_residual import engineering_validation as EV

    rec = EV.run(tmp_path, device="cpu", tiny=True, steps=4)
    c = rec["checks"]
    assert rec["passed"] and c["vllm"]["status"] == "NOT_RUN"
    assert {k: v["status"] for k, v in c.items() if k != "vllm"} == dict.fromkeys(
        ("initialization", "fold", "trainer_smoke", "resume", "checkpoint_identity"), "PASS")
    smoke = c["trainer_smoke"]
    assert smoke["projections_with_a_gradient"] == smoke["projections"] > 0 and smoke["embedding_unchanged"]
    assert smoke["orthogonality_rel_fro_max"]["after"] > smoke["orthogonality_rel_fro_max"]["before"]
    assert c["resume"]["max_abs_param_diff"] == 0.0
    # the declared basis is measured against the outputs' subspaces too
    layer = rec["per_layer"][0]
    assert {"attn_out_energy_under_declared_basis", "ffn_out_energy_under_declared_basis",
            "residual_in_energy_under_declared_basis"} <= set(layer)
    assert json.loads((tmp_path / "engineering_validation.json").read_text())["passed"]


def test_export_reproduces_the_committed_selection_when_the_journal_is_present():
    from stages.projected_residual import export_selection

    if not export_selection.JOURNAL.is_file():
        pytest.skip("the d4 search journal is out of tree and absent here")
    assert export_selection.build(export_selection.JOURNAL)["content_sha256"] == load(SELECTION)["content_sha256"]
