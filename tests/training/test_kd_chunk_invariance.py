"""`kd_forward_kl`'s chunk size is a memory knob, not a numerical one.

This matters because of how E8b-S2's registered throughput gate died: CUDA OOM at

    torch.log_softmax(tp[i : i + chunk].float() / temperature, dim=-1)

on an 80 GB A100, missing by 298 MiB. At vocab 151,936 each fp32 buffer in that loop
is 512 x 151936 x 4 B = 311 MB, which is the allocation that failed. Lowering `chunk`
is therefore the obvious lever — but only if it is genuinely free of numerical
consequence, because E8b's whole design rests on the arms being comparable.

It is very nearly free, and the precise sense matters. The loop sums row-independent
terms, so any partition computes **the same quantity** — but it accumulates one
float32 scalar per chunk, so changing the number of chunks changes the summation
order and therefore the rounding. Measured here: ~4e-6 absolute on a loss of ~54,
i.e. ~7e-8 relative. Mathematically identical, **not** bitwise identical.

That distinction is the reason this file exists rather than a one-line comment. A
first check appeared to show bit-identity, but with only 54 masked positions the
chunk sizes 512/256/128/64 all produced a *single* chunk and were trivially equal.
Any claim of bit-identity has to vary the chunk COUNT, which these tests do.

Consequence for E8b: applying the change uniformly to both arms of a pair keeps that
pair's comparison exact in the only sense that matters. It must NOT be applied to
only one arm, nor to S4, whose FP control was trained at chunk 512 and has no memory
problem to solve (23-27 GB on an L40S).
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.training.train import kd_forward_kl  # noqa: E402

CHUNKS = (512, 256, 128, 64, 17, 1)


def _inputs(seed: int, vocab: int, batch: int = 2, time: int = 40):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(batch, time, vocab, generator=g)
    t = torch.randn(batch, time, vocab, generator=g)
    mask = torch.rand(batch, time - 1, generator=g) > 0.3
    return s, t, mask


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_the_loss_agrees_across_chunk_sizes_to_float32_rounding(temperature):
    s, t, mask = _inputs(0, vocab=1024)
    base, base_n = kd_forward_kl(s, t, mask, temperature=temperature, chunk=512)
    for chunk in CHUNKS:
        loss, n = kd_forward_kl(s, t, mask, temperature=temperature, chunk=chunk)
        assert n == base_n, f"chunk {chunk} changed the position count"
        rel = abs(float(loss) - float(base)) / abs(float(base))
        assert rel < 1e-6, f"chunk {chunk} moved the loss by {rel:.3e} relative"


def test_changing_the_chunk_count_does_perturb_the_last_bits():
    """The claim is mathematical identity, NOT bit-identity — pin the difference.

    If this ever starts passing as an exact equality, the summation has changed and
    the re-pricing note in logs/e8b_reprice_after_gate.json needs rewording.
    """
    s, t, mask = _inputs(0, vocab=1024)
    one_chunk, _ = kd_forward_kl(s, t, mask, chunk=10_000)
    many_chunks, _ = kd_forward_kl(s, t, mask, chunk=1)
    assert float(one_chunk) != float(many_chunks), (
        "unexpected exact equality — float32 accumulation order should differ")
    assert abs(float(one_chunk) - float(many_chunks)) < 1e-4


def test_it_holds_at_the_real_vocabulary_size():
    # The teacher's vocab is what makes the buffers large enough to OOM; a test at
    # vocab 1024 would not exercise the same code path shape.
    s, t, mask = _inputs(1, vocab=151_936, batch=1, time=6)
    base, base_n = kd_forward_kl(s, t, mask, chunk=512)
    for chunk in (128, 32):
        loss, n = kd_forward_kl(s, t, mask, chunk=chunk)
        assert n == base_n
        assert abs(float(loss) - float(base)) / abs(float(base)) < 1e-6


def test_a_chunk_at_or_above_the_row_count_is_one_chunk():
    s, t, mask = _inputs(2, vocab=512)
    a, na = kd_forward_kl(s, t, mask, chunk=10_000)
    b, nb = kd_forward_kl(s, t, mask, chunk=100_000)
    # Both take the loop once, so these ARE bit-identical.
    assert (float(a), na) == (float(b), nb)


def test_an_empty_mask_returns_zero_without_touching_the_loop():
    s, t, _ = _inputs(3, vocab=256)
    mask = torch.zeros(s.shape[0], s.shape[1] - 1, dtype=torch.bool)
    for chunk in (512, 1):
        loss, n = kd_forward_kl(s, t, mask, chunk=chunk)
        assert n == 0 and float(loss) == 0.0


def test_the_default_chunk_is_the_value_the_gate_ran_with():
    # If the default moves, the OOM arithmetic recorded in
    # logs/e8b_reprice_after_gate.json no longer describes what runs.
    import inspect
    sig = inspect.signature(kd_forward_kl)
    assert sig.parameters["chunk"].default == 512
    assert round(512 * 151_936 * 4 / 1e6) == 311, (
        "the recorded 311 MB buffer size depends on chunk x vocab x 4 bytes")
