"""INT8 weight fake-quant tests on tensors and a tiny tied-embedding Qwen3.

Load-bearing checks: dequantized weights sit exactly on the per-row INT8
grid with the guaranteed error bound, scope selection matches the intended
module sets (decoder-only vs decoder+lm_head), and the tied embedding is
quantized once through the shared lm_head tensor.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aadistill.quant import QMAX, int8_fake_quant_tensor, int8_fake_quantize_

VOCAB = 64


def tiny_model(seed: int):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        intermediate_size=64, tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg)


def test_grid_and_error_bound():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    dq, stats = int8_fake_quant_tensor(w)
    scale = w.abs().amax(dim=1, keepdim=True) / QMAX
    q = dq / scale
    assert torch.allclose(q, q.round(), atol=1e-3)          # on the INT8 grid
    assert q.abs().max() <= QMAX + 1e-3
    assert (dq - w).abs().max() <= (scale / 2 + 1e-6).max()  # round-to-nearest bound
    assert dq.dtype == w.dtype
    assert 0 < stats["rel_fro_err"] < 0.02


def test_zero_row_and_dtype():
    w = torch.zeros(4, 8, dtype=torch.bfloat16)
    w[1] = torch.linspace(-1, 1, 8, dtype=torch.bfloat16)
    dq, _ = int8_fake_quant_tensor(w)
    assert dq.dtype == torch.bfloat16
    assert torch.isfinite(dq.float()).all()
    assert (dq[0] == 0).all() and (dq[2] == 0).all()


def test_deterministic():
    torch.manual_seed(1)
    w = torch.randn(8, 8)
    a, _ = int8_fake_quant_tensor(w.clone())
    b, _ = int8_fake_quant_tensor(w.clone())
    assert torch.equal(a, b)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        int8_fake_quant_tensor(torch.zeros(3))


def test_scope_decoder_leaves_head_untouched():
    model = tiny_model(2)
    head_before = model.lm_head.weight.detach().clone()
    summary = int8_fake_quantize_(model, scope="decoder")
    # 2 layers x (q,k,v,o + gate,up,down) = 14 linears, no lm_head.
    assert summary["n_linear_quantized"] == 14
    assert torch.equal(model.lm_head.weight, head_before)
    assert summary["tied_embeddings"] is True


def test_scope_all_quantizes_tied_head_once():
    model = tiny_model(2)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    head_before = model.lm_head.weight.detach().clone()
    summary = int8_fake_quantize_(model, scope="all")
    assert summary["n_linear_quantized"] == 15               # 14 decoder + lm_head
    assert not torch.equal(model.lm_head.weight, head_before)
    assert model.lm_head.weight is model.model.embed_tokens.weight  # tie preserved
    ids = torch.arange(8)[None]
    with torch.no_grad():
        logits = model(ids).logits
    assert torch.isfinite(logits).all()


def test_quant_changes_outputs_but_stays_close():
    model = tiny_model(3).eval()
    ids = torch.arange(12)[None]
    with torch.no_grad():
        base = model(ids).logits
    int8_fake_quantize_(model, scope="all")
    with torch.no_grad():
        quant = model(ids).logits
    assert not torch.equal(base, quant)
    rel = (quant - base).norm() / base.norm()
    assert rel < 0.05  # INT8 weight rounding is a small perturbation


def test_unknown_scope_fails_loudly():
    with pytest.raises(ValueError):
        int8_fake_quantize_(tiny_model(4), scope="w4a16")
