"""The initialization-NLL gate: every way a stale NLL could slip through.

E8's rule is that an initialization checkpoint is incomplete until *its own* NLL
is measured, and that nothing may be inherited from a previous initialization
however closely related the recipe is. These tests are the enforcement: each one
is a plausible shortcut someone would take at the end of a long pod session.

The gate deliberately never looks at whether an NLL is *good*. A worse or better
initialization NLL must not cancel or promote E8 — so the only value-level check
is validity, and a test asserts the gate accepts a record whose NLL is terrible.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.init.nll_gate import (  # noqa: E402
    REQUIRED_MEASUREMENTS, InitNllGateError, checkpoint_fingerprint, gate_summary,
    require_init_nll,
)


def fake_checkpoint(path: Path, weights: bytes = b"weights", layers: int = 28) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(weights)
    (path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": layers,
        "hidden_size": 1024, "vocab_size": 151936}))
    return checkpoint_fingerprint(path)


def record_for(fingerprint: dict, *, nll: float = 11.7482, **overrides) -> dict:
    measurements = {
        name: {"nll": nll, "positions": 1000,
               "measured_checkpoint_sha256": fingerprint["model_sha256"]}
        for name in REQUIRED_MEASUREMENTS
    }
    rec = {
        "artifact": "initialization_nll_record",
        "checkpoint": {**fingerprint, "num_parameters": 596_049_920},
        "depth_map": {"source": "explicit_kept_layers",
                      "kept_teacher_layers": list(range(28)),
                      "removed_teacher_layers": list(range(28, 36))},
        "measurements": measurements,
    }
    rec.update(overrides)
    return rec


def write(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record, indent=2))
    return path


def test_a_complete_record_for_this_checkpoint_passes(tmp_path):
    fp = fake_checkpoint(tmp_path / "ckpt")
    rec = write(tmp_path / "init_nll.json", record_for(fp))
    out = require_init_nll(tmp_path / "ckpt", rec)
    assert sorted(out["measurements"]) == sorted(REQUIRED_MEASUREMENTS)


def test_a_missing_record_blocks_training(tmp_path):
    fake_checkpoint(tmp_path / "ckpt")
    with pytest.raises(InitNllGateError, match="not complete until its own NLL"):
        require_init_nll(tmp_path / "ckpt", tmp_path / "absent.json")


def test_a_record_copied_from_a_sibling_initialization_is_rejected(tmp_path):
    """The exact shortcut E8 forbids: two closely-related inits, one measurement."""
    first = fake_checkpoint(tmp_path / "positional", b"positional-weights")
    second = fake_checkpoint(tmp_path / "contribution", b"contribution-weights")
    rec = write(tmp_path / "init_nll.json", record_for(first))
    require_init_nll(tmp_path / "positional", rec)          # fine for its own
    with pytest.raises(InitNllGateError, match="belongs to a different initialization"):
        require_init_nll(tmp_path / "contribution", rec)
    assert first["model_sha256"] != second["model_sha256"]


def test_a_record_that_admits_it_is_inherited_is_rejected_outright(tmp_path):
    fp = fake_checkpoint(tmp_path / "ckpt")
    for flag, value in (("inherited", True), ("copied_from", "init_v0"),
                        ("source_record", "other.json"), ("interpolated", True)):
        rec = write(tmp_path / f"{flag}.json", record_for(fp, **{flag: value}))
        with pytest.raises(InitNllGateError, match=flag):
            require_init_nll(tmp_path / "ckpt", rec)


def test_a_single_spliced_measurement_is_caught(tmp_path):
    """Right envelope, wrong provenance on one series."""
    fp = fake_checkpoint(tmp_path / "ckpt")
    rec = record_for(fp)
    rec["measurements"]["fineweb_val_e7"]["measured_checkpoint_sha256"] = "deadbeef"
    path = write(tmp_path / "init_nll.json", rec)
    with pytest.raises(InitNllGateError, match="spliced in from another run"):
        require_init_nll(tmp_path / "ckpt", path)


def test_a_missing_series_is_a_failure_not_a_shorter_report(tmp_path):
    fp = fake_checkpoint(tmp_path / "ckpt")
    rec = record_for(fp)
    del rec["measurements"]["teacher_native_val"]
    path = write(tmp_path / "init_nll.json", rec)
    with pytest.raises(InitNllGateError, match="teacher_native_val"):
        require_init_nll(tmp_path / "ckpt", path)


def test_an_invalid_nll_value_is_refused(tmp_path):
    fp = fake_checkpoint(tmp_path / "ckpt")
    for bad in (0, -1.0, None, "11.7", float("nan"), float("inf")):
        rec = record_for(fp)
        rec["measurements"]["holdout_v1"]["nll"] = bad
        path = write(tmp_path / "bad.json", rec)
        with pytest.raises(InitNllGateError, match="not a valid negative log"):
            require_init_nll(tmp_path / "ckpt", path)


def test_the_gate_does_not_judge_whether_the_nll_is_good(tmp_path):
    """A catastrophic initialization still passes: the endpoint is behaviour."""
    fp = fake_checkpoint(tmp_path / "ckpt")
    rec = write(tmp_path / "init_nll.json", record_for(fp, nll=99.0))
    out = require_init_nll(tmp_path / "ckpt", rec)
    assert out["measurements"]["holdout_v1"]["nll"] == 99.0


def test_unreadable_or_incomplete_checkpoints_raise_clearly(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(InitNllGateError, match="model.safetensors is missing"):
        checkpoint_fingerprint(tmp_path / "empty")
    (tmp_path / "empty" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(InitNllGateError, match="config.json is missing"):
        checkpoint_fingerprint(tmp_path / "empty")
    fp = fake_checkpoint(tmp_path / "ckpt")
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(InitNllGateError, match="not readable JSON"):
        require_init_nll(tmp_path / "ckpt", tmp_path / "broken.json")
    assert fp["num_hidden_layers"] == 28


def test_the_fingerprint_separates_weights_from_geometry(tmp_path):
    a = fake_checkpoint(tmp_path / "a", b"one", layers=28)
    b = fake_checkpoint(tmp_path / "b", b"two", layers=28)
    c = fake_checkpoint(tmp_path / "c", b"one", layers=24)
    assert a["model_sha256"] != b["model_sha256"]
    assert a["config_sha256"] == b["config_sha256"]     # same geometry
    assert a["config_sha256"] != c["config_sha256"]     # different depth
    assert a["model_sha256"] == c["model_sha256"]       # same bytes


def test_the_summary_flattens_one_row_per_checkpoint(tmp_path):
    fp = fake_checkpoint(tmp_path / "ckpt")
    rec = record_for(fp)
    rec["measurements"]["fineweb_val_e7"].update({"kl": 7.35, "top1": 0.0319,
                                                  "mean_rank": 10177.6})
    row = gate_summary(rec)
    assert row["checkpoint_sha256"] == fp["model_sha256"]
    assert row["depth_map_source"] == "explicit_kept_layers"
    assert row["removed_teacher_layers"] == list(range(28, 36))
    assert row["fineweb_val_e7.kl"] == 7.35
    assert row["holdout_v1.nll"] == 11.7482
    assert "fineweb_val_e7.entropy" not in row
