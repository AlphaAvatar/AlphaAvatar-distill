"""The teacher-target quality contract: every category kept apart, nothing by length."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.target_quality import (  # noqa: E402
    AcceptancePolicy, Check, assess, select_target, summarize,
)

ENDS = {"natural": ("eos",), "length": ("length",), "degeneration": ("degeneration",)}
STRICT = AcceptancePolicy()


def rec(i=0, *, stop="eos", proto=None, ans="correct", trace=None, policy=STRICT, unit="u",
        tokens=None):
    return assess(unit, i, stop_reason=stop, protocol_reason=proto, answer=Check(ans),
                  trace=trace, policy=policy, end_reasons=ENDS, generated_tokens=tokens)


def test_a_clean_correct_candidate_is_accepted_with_trace_not_assessed():
    r = rec()
    assert r.accepted_for_training and r.trace == "not_assessed" and r.reasons == []


@pytest.mark.parametrize("kw,reason", [
    ({"stop": "length"}, "ending:length"),
    ({"stop": "degeneration"}, "ending:degeneration"),
    ({"proto": "think_not_closed"}, "protocol:think_not_closed"),
    ({"ans": "incorrect"}, "answer:incorrect"),
    ({"ans": "unverifiable"}, "answer:unverifiable"),
    ({"trace": Check("invalid", "illegal_action")}, "trace:invalid:illegal_action"),
])
def test_each_failure_is_named_and_rejects(kw, reason):
    r = rec(**kw)
    assert not r.accepted_for_training and reason in r.reasons


def test_all_failures_are_recorded_not_only_the_first():
    r = rec(stop="length", proto="empty_answer", ans="incorrect")
    assert len(r.reasons) == 3
    assert r.length_limited and not r.naturally_terminated and not r.protocol_valid


def test_unverifiable_needs_an_explicit_switch():
    with pytest.raises(ValueError):
        AcceptancePolicy(require_answer=("correct", "unverifiable"))
    open_ = AcceptancePolicy(require_answer=("correct", "unverifiable"), admit_unverifiable=True)
    assert rec(ans="unverifiable", policy=open_).accepted_for_training


def test_unmapped_stop_reason_is_an_error():
    with pytest.raises(ValueError, match="not mapped"):
        rec(stop="mystery")


def test_selection_is_lowest_index_never_length():
    rows = [rec(2, tokens=100), rec(1, tokens=9000), rec(0, ans="incorrect", tokens=500)]
    assert select_target(rows).candidate_index == 1
    assert select_target([rec(0, ans="incorrect")]) is None


def test_summary_counts_units_and_candidates_separately():
    rows = [rec(0, unit="a"), rec(1, unit="a", ans="incorrect"), rec(0, unit="b", stop="length")]
    s = summarize(rows)["all"]
    assert s["candidates"] == 3 and s["units"] == 2
    assert s["accepted_candidates"] == 1 and s["accepted_units"] == 1
    assert s["rejection_reasons"] == {"answer:incorrect": 1, "ending:length": 1}
