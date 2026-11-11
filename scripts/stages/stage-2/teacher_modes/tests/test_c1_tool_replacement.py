"""C1-6cat's replacement audit (maintainer 2026-11-03): only tool-only sequences are replaced, a sequence mixing tool
turns with other turns is refused, and the kept sequences and the training/validation crossing are counted exactly."""
from __future__ import annotations

import pytest

from stages.teacher_modes import c1_tool_replacement as R


def _row(block, part, sessions, used=1000):
    return {"block": block, "part": part, "unpadded_length": used,
            "sessions": [{"session_id": sid, "data_type": dt, "supervised_retained": sup, "truncated": False,
                          "candidate_sha256": f"{block}{sid}"} for sid, dt, sup in sessions]}


def test_tool_only_sequences_are_replaced_and_the_rest_kept():
    rows = [_row(0, "e1", [("squad_v2-1#t1", "rag_evidence", 700), ("gsm8k-2#t1", "gsm8k", 300)], 2000),
            _row(1, "e1", [("glaive-000941#t1", "tool_calling", 400), ("glaive-000941#t2", "tool_calling", 200)], 900),
            _row(2, "e1", [("xlam-000007#t1", "tool_calling", 500)], 600),
            _row(3, "fluid", [("grid-9#t1", "primary.x", 900)], 1200),
            _row(4, "e1_validation", [("glaive-000941#t3", "tool_calling", 100)], 300),
            _row(5, "e1_validation", [("hotpot-4#t1", "multihop_qa", 250)], 400)]
    a = R.audit(rows)
    assert a["training"]["replace"] == {"sequences": 2, "turns": 3, "supervised_tokens": 1100,
                                        "non_padding_tokens": 1500}
    assert a["training"]["keep_byte_for_byte"]["sequences"] == 2
    assert a["training"]["keep_byte_for_byte"]["supervised_tokens"] == 1900
    assert a["training"]["replace_by_source"]["glaive_fc_v2"]["conversations"] == 1
    assert a["training"]["replace_by_source"]["xlam_fc_60k"]["turns"] == 1
    assert a["validation"]["replace"]["sequences"] == 1
    assert a["validation"]["keep_byte_for_byte"]["supervised_tokens"] == 250
    assert a["conversations_on_both_sides"] == ["glaive-000941"]
    assert a["steps_for_a_new_tool_part"]["historical"] == {"tool_sequences": 2, "optimizer_steps": 6}


def test_a_sequence_mixing_tool_and_other_turns_is_refused():
    rows = [_row(0, "e1", [("glaive-1#t1", "tool_calling", 400), ("gsm8k-2#t1", "gsm8k", 300)])]
    with pytest.raises(SystemExit, match="mix tool-calling"):
        R.audit(rows)
