"""The ladder data source: rung prefixes train, the pack tail validates.

These properties are what make the scaling curve readable — a rung must be the
blocks the gate measured, and no rung may see its own validation data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aadistill.data.ladder import (
    ladder_blocks,
    load_ladder_meta,
    rung_n_blocks,
    select_val_blocks,
)

TYPES = ["alpha", "beta", "gamma"]


def _write_pack(tmp_path, n_blocks=60, block_len=16, rungs=(100, 200, 400)):
    """A miniature pack with the same file layout build_token_ladder writes."""
    rng = np.random.default_rng(0)
    input_ids = rng.integers(0, 1000, size=(n_blocks, block_len), dtype=np.int32)
    ce_mask = np.zeros((n_blocks, block_len), dtype=bool)
    ce_mask[:, block_len // 2:] = True
    content_mask = np.ones((n_blocks, block_len), dtype=bool)
    np.savez_compressed(tmp_path / "blocks.npz", input_ids=input_ids,
                        ce_mask=ce_mask, content_mask=content_mask)

    with open(tmp_path / "audit.jsonl", "w") as f:
        for i in range(n_blocks):
            row = {
                "unpadded_length": block_len,
                "padding_length": 0,
                "terminal_truncated": False,
                "sessions": [{
                    "session_id": f"s{i}",
                    "data_type": TYPES[i % len(TYPES)],
                    "supervised_retained": 10,
                }],
            }
            f.write(json.dumps(row) + "\n")

    # Cumulative supervision is 10 tokens/block, so rung r needs r/10 blocks.
    meta = {
        "block_len": block_len,
        "n_blocks": n_blocks,
        "declared_mixture": {t: 1 / 3 for t in TYPES},
        "rungs": [
            {"target_supervised_tokens": r, "reachable": True,
             "n_blocks": r // 10, "actual_supervised_tokens": r}
            for r in rungs
        ],
    }
    (tmp_path / "ladder.json").write_text(json.dumps(meta))
    return meta


def test_rung_is_a_prefix_and_rungs_nest(tmp_path):
    _write_pack(tmp_path)
    small, _, _ = ladder_blocks(tmp_path, 100, n_val=6)
    large, _, _ = ladder_blocks(tmp_path, 400, n_val=6)
    assert small[0].shape[0] == 10 and large[0].shape[0] == 40
    # Nesting is what makes the ladder a scaling series rather than six draws.
    assert torch.equal(large[0][:10], small[0])
    assert torch.equal(large[1][:10], small[1])


def test_validation_never_overlaps_any_rung(tmp_path):
    _write_pack(tmp_path)
    _, _, stats = ladder_blocks(tmp_path, 100, n_val=6)
    assert stats["val_disjoint_from_all_rungs"]
    # Disjoint from the *largest* rung, not just the one being trained.
    assert min(stats["val_block_indices"]) >= 40


def test_validation_is_identical_across_rungs(tmp_path):
    _write_pack(tmp_path)
    _, val_small, stats_small = ladder_blocks(tmp_path, 100, n_val=6)
    _, val_large, stats_large = ladder_blocks(tmp_path, 400, n_val=6)
    assert stats_small["val_block_indices"] == stats_large["val_block_indices"]
    assert torch.equal(val_small[0], val_large[0])


def test_val_selection_is_mixture_balanced(tmp_path):
    _write_pack(tmp_path)
    _, _, stats = ladder_blocks(tmp_path, 400, n_val=6)
    mix = stats["val_token_mix"]
    assert set(mix) == set(TYPES)
    assert all(abs(v - 1 / 3) < 0.02 for v in mix.values())


def test_stats_report_the_realized_train_mixture(tmp_path):
    _write_pack(tmp_path)
    _, _, stats = ladder_blocks(tmp_path, 200, n_val=6)
    assert stats["train_blocks"] == 20
    assert stats["train_supervised_tokens"] == 200
    assert pytest.approx(sum(stats["train_token_mix"].values()), abs=1e-6) == 1.0


def test_unknown_rung_is_rejected(tmp_path):
    meta = _write_pack(tmp_path)
    with pytest.raises(ValueError, match="not in this pack"):
        rung_n_blocks(meta, 123)


def test_unreachable_rung_is_rejected(tmp_path):
    _write_pack(tmp_path)
    meta = load_ladder_meta(tmp_path)
    meta["rungs"].append({"target_supervised_tokens": 9_999, "reachable": False,
                          "actual_supervised_tokens": 600})
    with pytest.raises(ValueError, match="unreachable"):
        rung_n_blocks(meta, 9_999)


def test_val_request_larger_than_the_tail_fails_loudly(tmp_path):
    _write_pack(tmp_path)
    with pytest.raises(ValueError, match="blocks past the largest rung"):
        ladder_blocks(tmp_path, 400, n_val=100)


def test_select_val_blocks_is_deterministic(tmp_path):
    _write_pack(tmp_path)
    audit = [json.loads(line) for line in open(tmp_path / "audit.jsonl")]
    first = select_val_blocks(audit, 40, 6)
    assert first == select_val_blocks(audit, 40, 6)
    assert first == sorted(first)


def test_audit_and_blocks_must_agree(tmp_path):
    _write_pack(tmp_path, n_blocks=60)
    rows = (tmp_path / "audit.jsonl").read_text().splitlines()
    (tmp_path / "audit.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="audit rows"):
        ladder_blocks(tmp_path, 100, n_val=6)
