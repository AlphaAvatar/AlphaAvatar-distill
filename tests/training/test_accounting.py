"""U, E and C stay apart when supervision density differs.

Two packs with the same unique supervised tokens but different context and
padding must report the same U and E and different C; truncating padding must
lower C and leave U and E alone; the trainer's step rule must be mirrored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.training.accounting import matched_steps, training_accounting  # noqa: E402

SPEC = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "intermediate_size": 128, "vocab_size": 1000, "num_hidden_layers": 2}


def blocks(n, real, sup):
    return [{"real_length": real, "supervised": sup, "reasoning": sup - 10, "answer": 10}
            for _ in range(n)]


def acc(bl, **kw):
    base = dict(epochs=3, blocks_per_step=2, block_len=512, truncate_padding=False,
                student_spec=SPEC, teacher_spec=SPEC)
    return training_accounting(bl, **{**base, **kw})


def test_same_supervision_different_context_gives_same_u_e_and_different_c():
    dense = acc(blocks(10, 500, 450))
    sparse = acc(blocks(10, 200, 450 // 1), truncate_padding=True)
    assert dense["U_supervised"] == sparse["U_supervised"] == 4500
    assert dense["E_supervised"] == sparse["E_supervised"] == 13500
    assert sparse["C_total_flops"] < dense["C_total_flops"]


def test_truncating_padding_lowers_c_only():
    full, cut = acc(blocks(4, 300, 200)), acc(blocks(4, 300, 200), truncate_padding=True)
    assert full["U_supervised"] == cut["U_supervised"] and full["E_supervised"] == cut["E_supervised"]
    assert full["padding_executed"] > 0 and cut["padding_executed"] == 0
    assert cut["C_total_flops"] < full["C_total_flops"]


def test_the_step_rule_wraps_the_last_step():
    a = acc(blocks(5, 400, 300))          # ceil(5 * 3 / 2) = 8 steps, 16 passes
    assert a["optimizer_steps"] == 8 and a["block_passes"] == 16
    assert a["epochs"] == pytest.approx(16 / 5)
    assert a["E_supervised"] == round(1500 * 16 / 5)


def test_no_teacher_means_no_teacher_compute():
    assert acc(blocks(2, 100, 50), teacher_spec=None)["C_teacher_flops"] == 0.0


def test_matched_steps_reports_its_own_mismatch():
    m = matched_steps(416, 1460, epochs=3, reference_blocks_per_step=2)
    assert m["reference_steps"] == 624 and m["other_blocks_per_step"] == 7
    assert m["other_steps"] == 626 and abs(m["step_ratio"] - 1) < 0.01
