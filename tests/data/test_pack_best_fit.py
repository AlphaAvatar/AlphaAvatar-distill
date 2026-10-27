"""Best-fit packing that never cuts a session, and its nesting by increment.

Synthetic `RenderedSession`s, no tokenizer: the properties are about placement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.data.sessions import RenderedSession, pack_best_fit, pack_sessions  # noqa: E402

PAD = 0
SYS = [9, 9, 9]


def sess(i, n, *, key="k", source=None, sup=None):
    sup = n // 2 if sup is None else sup
    return RenderedSession(
        session_id=f"s{i:03d}", data_type="t", system_text="x", system_key=key,
        body_ids=[100 + i] * n, body_mask=[False] * (n - sup) + [True] * sup,
        n_system_tokens=len(SYS), source_id=source or f"src{i}")


LENGTHS = [700, 300, 650, 120, 480, 900, 60, 333, 512, 41, 260, 777]


def test_nothing_is_cut_and_every_session_lands_once():
    sessions = [sess(i, n) for i, n in enumerate(LENGTHS)]
    blocks = pack_best_fit(sessions, {"k": SYS}, block_len=1024, pad_id=PAD)
    seen = [m["session_id"] for b in blocks for m in b.audit["sessions"]]
    assert sorted(seen) == sorted(s.session_id for s in sessions)
    assert not any(m["truncated"] for b in blocks for m in b.audit["sessions"])
    assert sum(b.n_supervised for b in blocks) == sum(s.n_supervised for s in sessions)
    for b in blocks:
        assert len(b.input_ids) == 1024 and b.input_ids[:3] == SYS
        assert b.audit["unpadded_length"] + b.audit["padding_length"] == 1024


def test_best_fit_uses_fewer_blocks_than_sequential_fill_without_cuts():
    sessions = [sess(i, n) for i, n in enumerate(LENGTHS)]
    bf = pack_best_fit(sessions, {"k": SYS}, block_len=1024, pad_id=PAD)
    seq = pack_sessions(sessions, {"k": SYS}, block_len=1024, pad_id=PAD,
                        allow_terminal_truncation=False)
    assert len(bf) < len(seq)
    lower_bound = -(-sum(LENGTHS) // (1024 - len(SYS)))
    assert len(bf) <= lower_bound + 1


def test_output_does_not_depend_on_input_order():
    sessions = [sess(i, n) for i, n in enumerate(LENGTHS)]
    a = pack_best_fit(sessions, {"k": SYS}, block_len=1024, pad_id=PAD)
    b = pack_best_fit(list(reversed(sessions)), {"k": SYS}, block_len=1024, pad_id=PAD)
    assert [x.input_ids for x in a] == [x.input_ids for x in b]


def test_same_source_never_shares_a_block():
    sessions = [sess(i, 100, source="conv") for i in range(5)]
    blocks = pack_best_fit(sessions, {"k": SYS}, block_len=1024, pad_id=PAD)
    assert len(blocks) == 5


def test_groups_never_mix():
    sessions = [sess(0, 100, key="a"), sess(1, 100, key="b")]
    blocks = pack_best_fit(sessions, {"a": SYS, "b": [7, 7]}, block_len=512, pad_id=PAD)
    assert len(blocks) == 2 and blocks[1].input_ids[:2] == [7, 7]


def test_a_session_too_long_for_an_empty_block_is_an_error():
    with pytest.raises(RuntimeError, match="cannot fit"):
        pack_best_fit([sess(0, 1022)], {"k": SYS}, block_len=1024, pad_id=PAD)


def test_increment_packing_keeps_blocks_nested():
    """rung k = blocks[:n_k] when each increment is packed alone."""
    s = [sess(i, n) for i, n in enumerate(LENGTHS)]
    inc1, inc2 = s[:5], s[5:]
    b1 = pack_best_fit(inc1, {"k": SYS}, block_len=1024, pad_id=PAD)
    b2 = pack_best_fit(inc2, {"k": SYS}, block_len=1024, pad_id=PAD)
    rung1 = {m["session_id"] for b in b1 for m in b.audit["sessions"]}
    rung2 = {m["session_id"] for b in b1 + b2 for m in b.audit["sessions"]}
    assert rung1 == {x.session_id for x in inc1} and rung1 < rung2
