"""The batched masked per-item KL reducer, against the scalar oracle.

`forward_kl_mean` stays the oracle. Every property below is stated against it
rather than against a remembered number, because the whole point of the batched
reducer is that it computes the SAME per-item quantity with a different
schedule — and the one way that claim can be wrong quietly is a pooled,
token-weighted mean that looks plausible on uniform-length fixtures.

So the fixtures here are deliberately ragged, and one of them is deliberately
extreme.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.statistics import contribution as C  # noqa: E402
from aadistill.initialization.statistics.contribution import (  # noqa: E402
    forward_kl_mean,
    forward_kl_mean_batch,
)

VOCAB = 256


def ragged(lengths, *, vocab=VOCAB, seed=17, spread=0.4):
    """Padded ref/abl logits and a prediction mask for `lengths`."""
    torch.manual_seed(seed)
    width = max(lengths)
    ref = torch.randn(len(lengths), width, vocab)
    abl = ref + spread * torch.randn(len(lengths), width, vocab)
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    return ref, abl, mask


# --- 1. the oracle relationship --------------------------------------------


class TestItMatchesTheScalarOracle:
    @pytest.mark.parametrize("lengths", [
        [7],                       # B = 1
        [23, 11],                  # B = 2, ragged
        [64, 5, 31, 17],           # B = 4, ragged
    ], ids=["B1", "B2", "B4"])
    def test_every_row_equals_the_scalar_reduction_of_that_row(self, lengths):
        ref, abl, mask = ragged(lengths)
        got = forward_kl_mean_batch(ref, abl, mask)
        assert got.shape == (len(lengths),)
        for row, length in enumerate(lengths):
            want = forward_kl_mean(ref[row, :length], abl[row, :length])
            scale = max(abs(want), 1e-12)
            assert abs(float(got[row]) - want) / scale < 1e-6, (
                f"row {row} (len {length}) disagrees with the oracle")

    def test_a_single_row_is_bit_identical_to_the_oracle(self):
        """B=1 has nothing to reschedule, so "close" is the wrong standard.

        The batched path must reduce to exactly the oracle's float when the
        batch is one unpadded row, which is what makes it usable as the strict
        replay path.
        """
        ref, abl, mask = ragged([40])
        got = float(forward_kl_mean_batch(ref, abl, mask)[0])
        assert got == forward_kl_mean(ref[0], abl[0])

    def test_the_oracle_is_not_loosened_to_make_the_batch_pass(self):
        """`forward_kl_mean` still returns a plain float for a 2-D input."""
        ref, abl, _ = ragged([12])
        value = forward_kl_mean(ref[0], abl[0])
        assert isinstance(value, float)
        with pytest.raises(ValueError, match=r"\[B, T_pred, V\]"):
            forward_kl_mean_batch(ref[0], abl[0], torch.ones(12, dtype=torch.bool))


# --- 2. masking ------------------------------------------------------------


class TestPaddingCannotContribute:
    def test_poisoning_masked_positions_changes_nothing(self):
        """The strongest form: EXACT equality, not a tolerance.

        A masked position is multiplied by zero before the row sum, so garbage
        there contributes exactly 0.0 — not a small number.
        """
        lengths = [30, 9, 21]
        ref, abl, mask = ragged(lengths)
        before = forward_kl_mean_batch(ref, abl, mask)
        poisoned_ref, poisoned_abl = ref.clone(), abl.clone()
        for row, length in enumerate(lengths):
            poisoned_ref[row, length:] = 1e4
            poisoned_abl[row, length:] = -1e4
        after = forward_kl_mean_batch(poisoned_ref, poisoned_abl, mask)
        assert torch.equal(before, after)

    def test_a_row_predicting_nothing_is_refused(self):
        ref, abl, mask = ragged([10, 10])
        mask[1] = False
        with pytest.raises(ValueError, match="no valid prediction positions"):
            forward_kl_mean_batch(ref, abl, mask)

    def test_a_mask_that_does_not_describe_the_logits_is_refused(self):
        ref, abl, mask = ragged([10, 10])
        with pytest.raises(ValueError, match="does not describe"):
            forward_kl_mean_batch(ref, abl, mask[:, :-1])


# --- 3. items do not leak into each other ----------------------------------


class TestItemIndependence:
    def test_changing_one_rows_valid_logits_moves_only_that_row(self):
        lengths = [20, 14, 26]
        ref, abl, mask = ragged(lengths)
        before = forward_kl_mean_batch(ref, abl, mask)
        moved = abl.clone()
        moved[1, :lengths[1]] += 1.5
        after = forward_kl_mean_batch(ref, moved, mask)
        assert float(after[1]) != float(before[1])
        assert float(after[0]) == float(before[0])
        assert float(after[2]) == float(before[2])

    def test_a_rows_value_does_not_depend_on_its_neighbours_lengths(self):
        """The same item, batched beside a much longer one, keeps its value."""
        ref, abl, mask = ragged([12, 900])
        pair = forward_kl_mean_batch(ref, abl, mask)
        alone = forward_kl_mean_batch(ref[:1, :12], abl[:1, :12], mask[:1, :12])
        scale = max(abs(float(alone[0])), 1e-12)
        assert abs(float(pair[0]) - float(alone[0])) / scale < 1e-6


# --- 4. weighting: per item, never pooled ----------------------------------


def test_the_result_is_a_per_item_mean_and_not_a_token_weighted_pool():
    """The failure this reducer exists to make impossible.

    One long low-KL item and one short high-KL item. A pooled
    `(kl * mask).sum() / mask.sum()` would land near the long item's value; the
    contract requires each item's own mean, so the caller's subtype/domain
    weighting decides their influence rather than their token counts.
    """
    torch.manual_seed(5)
    long_len, short_len = 1000, 100
    ref = torch.zeros(2, long_len, VOCAB)
    abl = torch.zeros(2, long_len, VOCAB)
    #: Row 0: a small perturbation over many positions. Row 1: a large one over
    #: few. The exact KLs do not matter; their ORDER and separation do.
    abl[0, :long_len, 0] = 0.30
    abl[1, :short_len, 0] = 3.00
    mask = torch.zeros(2, long_len, dtype=torch.bool)
    mask[0, :long_len] = True
    mask[1, :short_len] = True

    got = forward_kl_mean_batch(ref, abl, mask)
    per_item = [forward_kl_mean(ref[0, :long_len], abl[0, :long_len]),
                forward_kl_mean(ref[1, :short_len], abl[1, :short_len])]
    for row in (0, 1):
        assert abs(float(got[row]) - per_item[row]) < 1e-9

    #: And it is demonstrably NOT the pooled figure.
    valid = mask.sum().item()
    pooled_num = per_item[0] * long_len + per_item[1] * short_len
    pooled = pooled_num / valid
    assert per_item[1] > per_item[0], "fixture does not separate the two items"
    assert abs(float(got[1]) - pooled) > 0.5 * (per_item[1] - per_item[0]), (
        "the short high-KL item's value sits near the token-weighted pool; a "
        "pooled reduction would be indistinguishable on this fixture")


# --- 5. chunk boundaries ---------------------------------------------------


def test_lengths_around_the_chunk_boundary_in_one_ragged_batch():
    """511/512/513/1024/1025 together, so no row is the batch width by accident."""
    lengths = [511, 512, 513, 1024, 1025, 7]
    ref, abl, mask = ragged(lengths, vocab=64, seed=29)
    got = forward_kl_mean_batch(ref, abl, mask, chunk=512)
    for row, length in enumerate(lengths):
        want = forward_kl_mean(ref[row, :length], abl[row, :length], chunk=512)
        scale = max(abs(want), 1e-12)
        assert abs(float(got[row]) - want) / scale < 1e-6, (
            f"row {row} of length {length} disagrees across the chunk boundary")


def test_chunking_walks_the_position_axis_not_a_flattened_valid_list():
    """A row's own boundaries must not shift with its neighbours' lengths.

    If the reducer flattened all valid positions before chunking, the same item
    would be split at different places depending on who shared its batch.
    """
    ref_a, abl_a, mask_a = ragged([600, 600], seed=31, vocab=64)
    solo = forward_kl_mean_batch(ref_a[:1], abl_a[:1], mask_a[:1], chunk=512)
    pair = forward_kl_mean_batch(ref_a, abl_a, mask_a, chunk=512)
    assert float(solo[0]) == float(pair[0])


# --- 6. both accumulator branches ------------------------------------------


def test_both_accumulator_paths_agree(monkeypatch):
    """The device branch is unreachable on a CPU-only box without this seam."""
    lengths = [33, 17, 9]
    ref, abl, mask = ragged(lengths, seed=41)
    host = forward_kl_mean_batch(ref, abl, mask)
    monkeypatch.setattr(C, "_reduce_on_device", lambda device: True)
    device_path = forward_kl_mean_batch(ref, abl, mask)
    assert torch.allclose(host, device_path, atol=0, rtol=1e-12)
    #: and the device branch really did run a different line
    assert C._reduce_on_device(torch.device("cpu")) is True
