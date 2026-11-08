"""Stage 3's $0 failure analysis (maintainer 2026-11-08): its answer extraction balances braces and takes the last
boxed answer, a pre-closed reply is all answer, supervision runs are found exactly, and the usable/correct split of
the difference is exact."""
from __future__ import annotations

import numpy as np

from stages.teacher_modes import stage3_failure_analysis as FA


def test_the_last_boxed_answer_is_read_with_balanced_braces():
    assert FA.last_boxed(r"first \boxed{1}, then \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert FA.last_boxed("no answer here") is None
    assert FA.last_boxed(r"cut \boxed{\frac{1}{") is None


def test_a_reply_is_split_at_its_think_close_unless_the_template_closed_it():
    assert FA.split_turn({"raw": "<think>\nr\n</think>\n\nA<|im_end|>", "think_state": "self_opening"}) == \
        ("<think>\nr\n", "\n\nA", True)
    assert FA.split_turn({"raw": "still reasoning", "think_state": "preopened"}) == ("still reasoning", "", False)
    assert FA.split_turn({"raw": "the answer<|im_end|>", "think_state": "preclosed"}) == ("", "the answer", True)


def test_supervision_runs_are_found_exactly():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
    assert FA.segments(m) == [(1, 3), (5, 6), (7, 8)]


def test_the_decomposition_adds_up_to_the_difference():
    def unit(rate_u, rate_c):
        return {"static": {"formal": {"usable_rate": rate_u, "correct_among_usable": rate_c,
                                      "rate": rate_u * rate_c}}}
    per = {f"{arm}-s{s}": unit(u, c) for s in FA.SEEDS
           for arm, (u, c) in (("A1-6cat-sr", (0.64, 0.12)), ("A2-6cat-sr", (0.88, 0.124)))}
    d = FA.decomposition(per)[str(FA.SEEDS[0])]
    assert abs(d["from_usable_rate_pp"] + d["from_correct_among_usable_pp"] - d["A2_minus_A1_pp"]) < 0.02
