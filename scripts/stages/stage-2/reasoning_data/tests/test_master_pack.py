"""The master-ladder pack: nested at block level, never cut, and readable by the trainer.

Synthetic rendered sessions (no tokenizer): the properties are about the
ladder, the packing order and the files `aadistill.data.ladder.ladder_blocks`
reads, which is exactly the consumer a real pack has.
"""
from __future__ import annotations

import pytest

from stages.reasoning_data import build_master_ladder as B

from aadistill.data.ladder import ladder_blocks
from aadistill.data.nested_ladder import build_ladder
from aadistill.data.sessions import RenderedSession

BLOCK = 512
SYS = [1, 2, 3]
THINK_CLOSE = 99


def sessions():
    out = {}
    for s in ("a", "b"):
        for f in range(4):
            for k in range(14):
                sid = f"{s}{f}-{k}"
                n = 60 + 7 * ((k * 5 + f) % 13)
                body = [10] * 8 + [11] * (n - 12) + [THINK_CLOSE] + [12, 12, 13]
                mask = [False] * 8 + [True] * (n - 8)
                r = RenderedSession(session_id=sid, data_type=s, system_text="x", system_key="k",
                                    body_ids=body, body_mask=mask, n_system_tokens=len(SYS),
                                    source_id=sid)
                r.meta.update(stratum=s, family=f"{s}.{f}", attributes={"gen": f"g{f % 2}"})
                out[sid] = r
    return out


@pytest.fixture()
def built(tmp_path):
    rendered = sessions()
    units = B.ladder_units(rendered, THINK_CLOSE)
    spec = {"block_len": BLOCK, "salt": "t", "shares": {"a": 0.5, "b": 0.5},
            "rung_targets": [2_000, 4_000], "reserve_tokens": 1_200}
    ladder = build_ladder(units, spec["shares"], spec["rung_targets"], salt="t",
                          reserve=spec["reserve_tokens"])
    blocks, cuts, val_start = B.pack(ladder, rendered, {"k": SYS}, block_len=BLOCK, pad_id=0)
    for row, cut in zip(ladder["rungs"], cuts):
        row["n_blocks"] = cut
        row["packed"] = B.block_accounting(blocks[:cut], BLOCK)
    built = {"ladder": ladder, "blocks": blocks, "val_start": val_start, "excluded": [],
             "pad_id": 0, "validation_tail": B.block_accounting(blocks[val_start:], BLOCK)}
    meta = B.write_pack(tmp_path, built, spec, {"test": True})
    return tmp_path, built, meta


def test_reasoning_and_answer_tokens_split_at_the_delimiter():
    r = next(iter(sessions().values()))
    c = B.split_counts(r, THINK_CLOSE)
    assert c["reasoning_tokens"] + c["answer_tokens"] == c["supervised_tokens"] == r.n_supervised
    assert c["answer_tokens"] == 4  # </think> + three answer tokens


def test_rungs_are_block_prefixes_and_nothing_is_cut(built):
    out, b, meta = built
    n1, n2 = (r["n_blocks"] for r in meta["rungs"])
    assert 0 < n1 < n2 <= meta["validation_tail"]["first_block"]
    ids_1 = {s["session_id"] for blk in b["blocks"][:n1] for s in blk.audit["sessions"]}
    ids_2 = {s["session_id"] for blk in b["blocks"][:n2] for s in blk.audit["sessions"]}
    assert ids_1 < ids_2
    assert not any(blk.audit["terminal_truncated"] for blk in b["blocks"])


def test_the_trainer_reads_every_rung_and_validates_on_the_reserved_tail(built):
    out, b, meta = built
    for rung in meta["rungs"]:
        train, val, stats = ladder_blocks(out, rung["target_supervised_tokens"], n_val=2)
        assert train[0].shape[0] == rung["n_blocks"]
        assert stats["val_disjoint_from_all_rungs"]
        assert stats["train_supervised_tokens"] == rung["actual_supervised_tokens"]
    reserved = set(b["ladder"]["reserved"])
    tail = {s["session_id"] for blk in b["blocks"][b["val_start"]:] for s in blk.audit["sessions"]}
    assert tail == reserved


def test_the_pack_records_its_own_hashes(built):
    out, _, meta = built
    from stages.reasoning_data import assets as A
    assert A.sha256_file(out / "blocks.npz") == meta["outputs"]["blocks.npz"]
