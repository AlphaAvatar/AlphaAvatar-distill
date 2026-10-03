"""Tests for the Experiment 2 checkpoint retention policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "pod"))
from retain_checkpoints import (  # noqa: E402
    choose_keep,
    deterioration_onset,
    read_trajectory,
)


def points(*rows):
    return [{"step": s, "val_ce": c, "holdout_nll": n} for s, c, n in rows]


# --------------------------------------------------------------------------
# onset detection
# --------------------------------------------------------------------------
def test_onset_needs_sustained_rises_not_one_uptick():
    """A single up-tick is noise at this metric's spread and must not fire."""
    traj = points((0, 10.0, 6.0), (127, 5.0, 5.5), (254, 4.0, 5.9), (381, 3.0, 5.4))
    assert deterioration_onset(traj, sustained=2) is None


def test_onset_is_the_step_before_the_sustained_rise():
    traj = points((0, 10.0, 6.0), (127, 5.0, 5.5), (254, 4.0, 6.1),
                  (381, 3.0, 6.8), (508, 2.5, 7.4))
    assert deterioration_onset(traj, sustained=2) == 127


def test_a_monotonically_improving_run_has_no_onset():
    traj = points((0, 10.0, 9.0), (127, 5.0, 8.0), (254, 4.0, 7.0), (381, 3.0, 6.0))
    assert deterioration_onset(traj, sustained=2) is None


def test_missing_holdout_values_are_skipped_not_guessed():
    traj = [{"step": 0, "val_ce": 10.0, "holdout_nll": 6.0},
            {"step": 127, "val_ce": 5.0, "holdout_nll": None},
            {"step": 254, "val_ce": 4.0, "holdout_nll": 6.5},
            {"step": 381, "val_ce": 3.0, "holdout_nll": 7.0}]
    assert deterioration_onset(traj, sustained=2) == 0


# --------------------------------------------------------------------------
# keep set
# --------------------------------------------------------------------------
def test_final_is_always_kept():
    traj = points((0, 10.0, 6.0), (127, 5.0, 6.5), (254, 4.0, 7.0))
    keep = choose_keep(traj)["keep"]
    assert "final" in keep[254]


def test_best_val_ce_and_best_nll_are_both_kept():
    traj = points((0, 10.0, 6.0), (127, 5.0, 5.2), (254, 4.0, 7.0))
    keep = choose_keep(traj)["keep"]
    assert "best_val_ce" in keep[254]
    assert "best_holdout_nll" in keep[127]


def test_one_step_can_carry_several_reasons():
    """The common case: val CE is monotone, so final is also best-val-CE."""
    traj = points((0, 10.0, 6.0), (127, 5.0, 6.5), (254, 4.0, 7.0))
    keep = choose_keep(traj)["keep"]
    assert set(keep[254]) >= {"final", "best_val_ce"}
    assert len(keep) <= 3  # never one checkpoint per eval point


def test_the_deterioration_bracket_is_kept_on_both_sides():
    traj = points((0, 10.0, 6.0), (127, 5.0, 5.5), (254, 4.0, 6.1),
                  (381, 3.0, 6.8), (508, 2.5, 7.4))
    result = choose_keep(traj, sustained=2)
    assert result["onset"] == 127
    assert "deterioration_onset" in result["keep"][127]
    assert "after_deterioration_onset" in result["keep"][254]


def test_an_empty_trajectory_fails_loudly():
    with pytest.raises(SystemExit):
        choose_keep([])


# --------------------------------------------------------------------------
# log parsing
# --------------------------------------------------------------------------
def test_trajectory_reads_only_primary_val_eval_rows(tmp_path):
    log = tmp_path / "train_log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"event": "run_start"},
        {"event": "train_step", "step": 5, "loss": 1.0},
        {"event": "eval_result", "step": 127, "val_ce": 2.0, "holdout_nll": 6.0},
        {"event": "eval_result", "step": 127, "val_set": "extra", "val_ce": 9.9},
        {"event": "eval_result", "step": 254, "val_ce": 1.5, "holdout_nll": 6.4},
        "not json",
    ]) + "\n")
    traj = read_trajectory(log)
    assert [p["step"] for p in traj] == [127, 254]
    assert traj[0]["val_ce"] == 2.0


def test_a_later_eval_at_the_same_step_wins(tmp_path):
    """A resumed run re-evaluates its restart step; the newer row is the truth."""
    log = tmp_path / "train_log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"event": "eval_result", "step": 127, "val_ce": 2.0, "holdout_nll": 6.0},
        {"event": "eval_result", "step": 127, "val_ce": 1.9, "holdout_nll": 5.9},
    ]) + "\n")
    traj = read_trajectory(log)
    assert len(traj) == 1 and traj[0]["val_ce"] == 1.9


def test_holdout_trajectory_is_merged_from_the_orchestrator_file(tmp_path):
    """Held-out NLL is scored outside the trainer, so it arrives separately.

    Adding it to the training loop would change the trainer that produced the
    Experiment 1 control, which is the one thing an A/B against that control
    cannot afford.
    """
    (tmp_path / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"event": "eval_result", "step": 127, "val_ce": 2.0},
        {"event": "eval_result", "step": 254, "val_ce": 1.5},
        {"event": "eval_result", "step": 381, "val_ce": 1.2},
    ]) + "\n")
    (tmp_path / "holdout_trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 127, "holdout_nll": 6.0},
        {"step": 254, "holdout_nll": 6.6},
        {"step": 381, "holdout_nll": 7.1},
    ]) + "\n")
    traj = read_trajectory(tmp_path / "train_log.jsonl")
    assert [p["holdout_nll"] for p in traj] == [6.0, 6.6, 7.1]
    result = choose_keep(traj, sustained=2)
    assert result["onset"] == 127
    assert "best_holdout_nll" in result["keep"][127]
    assert "final" in result["keep"][381]


def test_a_step_present_only_in_the_holdout_file_still_appears(tmp_path):
    (tmp_path / "train_log.jsonl").write_text(
        json.dumps({"event": "eval_result", "step": 127, "val_ce": 2.0}) + "\n")
    (tmp_path / "holdout_trajectory.jsonl").write_text(
        json.dumps({"step": 254, "holdout_nll": 6.0}) + "\n")
    traj = read_trajectory(tmp_path / "train_log.jsonl")
    assert [p["step"] for p in traj] == [127, 254]
