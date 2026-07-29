"""CPU toy-model tests for the Stage 0 ActivationStatsCollector.

Uses a tiny random Qwen3 (2 layers, hidden 32) so the suite runs in seconds
without any download. Covers the AGENTS.md 2.3 test requirements available at
this stage: shape/statistics correctness against direct computation,
deterministic behavior across runs, save/load roundtrip, PSD covariance,
and rejection of batched input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aadistill.collect import ActivationStatsCollector, residual_covariance

VOCAB = 128


def make_tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def make_sequences():
    g = torch.Generator().manual_seed(42)
    return [
        torch.randint(0, VOCAB, (1, t), generator=g) for t in (7, 19, 33)
    ]


@pytest.fixture(scope="module")
def model():
    return make_tiny_model()


@pytest.fixture(scope="module")
def sequences():
    return make_sequences()


def run_collector(model, sequences):
    collector = ActivationStatsCollector(model)
    for ids in sequences:
        collector.process(ids)
    collector.close()
    return collector


def test_residual_stats_match_direct(model, sequences):
    collector = run_collector(model, sequences)
    n_points = model.config.num_hidden_layers + 1
    d = model.config.hidden_size

    ref_sum = torch.zeros(n_points, d, dtype=torch.float64)
    ref_sqsum = torch.zeros(n_points, d, d, dtype=torch.float64)
    ref_count = 0
    with torch.no_grad():
        for ids in sequences:
            hs = model(ids, output_hidden_states=True).hidden_states
            assert len(hs) == n_points
            for point, h in enumerate(hs):
                x = h[0].to(torch.float64)
                ref_sum[point] += x.sum(0)
                ref_sqsum[point] += x.T @ x
            ref_count += ids.shape[1]

    assert collector.res_count == ref_count
    torch.testing.assert_close(collector.res_sum, ref_sum)
    torch.testing.assert_close(collector.res_sqsum, ref_sqsum)


def test_ffn_stats_match_direct(model, sequences):
    collector = run_collector(model, sequences)
    i = model.config.intermediate_size

    captured: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        def grab(_m, args, idx=idx):
            captured[idx].append(args[0].detach().reshape(-1, i).to(torch.float64))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(grab))
    with torch.no_grad():
        for ids in sequences:
            model(ids)
    for h in hooks:
        h.remove()

    for idx in captured:
        a = torch.cat(captured[idx])
        torch.testing.assert_close(collector.ffn_abs_sum[idx], a.abs().sum(0))
        torch.testing.assert_close(collector.ffn_sq_sum[idx], (a * a).sum(0))


def test_token_counts(model, sequences):
    collector = run_collector(model, sequences)
    ref = torch.zeros(VOCAB, dtype=torch.int64)
    for ids in sequences:
        ref += torch.bincount(ids[0], minlength=VOCAB)
    assert torch.equal(collector.token_counts, ref)
    assert collector.token_counts.sum() == sum(ids.shape[1] for ids in sequences)


def test_deterministic_across_runs(model, sequences):
    a = run_collector(model, sequences).state()
    b = run_collector(model, sequences).state()
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs between identical runs"


def test_save_load_roundtrip(model, sequences, tmp_path):
    from safetensors.torch import load_file

    collector = run_collector(model, sequences)
    path = tmp_path / "stats.safetensors"
    meta = collector.save(str(path))
    assert meta["tokens_processed"] == collector.res_count

    loaded = load_file(str(path))
    state = collector.state()
    assert set(loaded) == set(state)
    for key in state:
        assert torch.equal(loaded[key], state[key]), f"{key} changed in roundtrip"


def test_covariance_psd_and_shapes(model, sequences):
    state = run_collector(model, sequences).state()
    d = model.config.hidden_size
    for point in range(model.config.num_hidden_layers + 1):
        mean, cov = residual_covariance(state, point)
        assert mean.shape == (d,)
        assert cov.shape == (d, d)
        torch.testing.assert_close(cov, cov.T)
        eigvals = torch.linalg.eigvalsh(cov)
        top = eigvals[-1].item()
        assert top > 0
        assert eigvals[0].item() >= -1e-10 * top, (
            f"Point {point}: covariance not PSD within tolerance, min eig {eigvals[0]}"
        )


def test_rejects_batched_and_unbatched_input(model):
    collector = ActivationStatsCollector(model)
    try:
        with pytest.raises(ValueError):
            collector.process(torch.randint(0, VOCAB, (2, 5)))
        with pytest.raises(ValueError):
            collector.process(torch.randint(0, VOCAB, (5,)))
    finally:
        collector.close()


def test_covariance_requires_tokens(model):
    collector = ActivationStatsCollector(model)
    collector.close()
    with pytest.raises(ValueError):
        residual_covariance(collector.state(), point=0)
