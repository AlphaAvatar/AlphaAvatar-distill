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

from aadistill.training.accounting import (  # noqa: E402
    consumed_exposure, match_schedule, training_accounting)

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


def test_one_step_count_for_every_pack_with_its_passes_reported():
    m = match_schedule({"B": 416, "A": 1460}, epochs=3, reference_blocks_per_step=2)
    assert m["reference"] == "B" and m["T0"] == 624 and m["blocks_per_step"]["B"] == 2
    t, bps = m["steps"], m["blocks_per_step"]["A"]
    assert m["passes"]["A"] == pytest.approx(t * bps / 1460, abs=1e-6)
    assert m["max_pass_deviation"] <= abs(624 * 7 / 1460 - 3) / 3      # never worse than T0 itself
    lo, hi = m["searched_steps"]
    assert lo <= t <= hi and lo == 561 and hi == 687


def test_the_rule_is_deterministic_and_reference_independent_of_naming():
    a = match_schedule({"x": 500, "y": 1777}, epochs=3, reference_blocks_per_step=2)
    b = match_schedule({"y": 1777, "x": 500}, epochs=3, reference_blocks_per_step=2)
    assert a == b and a["reference"] == "x"


def test_a_fixed_step_count_overrides_the_ceil_rule():
    a = acc(blocks(5, 400, 300), steps=10)
    assert a["optimizer_steps"] == 10 and a["block_passes"] == 20 and a["epochs"] == pytest.approx(4)


def test_kd_positions_split_into_prompt_reasoning_and_answer():
    a = acc(blocks(2, 500, 300), kd_scope="all")
    per = a["KD_positions_per_pass"]
    assert per == {"prompt": 2 * (500 - 1 - 300), "reasoning": 2 * 290, "answer": 2 * 10,
                   "unsplit_supervised": 0}
    assert acc(blocks(2, 500, 300), kd_scope="assistant")["KD_positions_per_pass"]["prompt"] == 0
    with pytest.raises(ValueError):
        acc(blocks(1, 10, 5), kd_scope="everything")


def test_exact_exposure_follows_the_consumed_order():
    bl = [{"real_length": 100, "supervised": s, "reasoning": s, "answer": 0} for s in (10, 20, 30)]
    one = consumed_exposure(bl, [0, 1, 2, 0])
    two = consumed_exposure(bl, [0, 1, 2, 2])
    assert one["U_supervised_touched"] == two["U_supervised_touched"] == 60
    assert (one["E_supervised"], two["E_supervised"]) == (70, 90)
    with pytest.raises(ValueError):
        consumed_exposure(bl, [3])
