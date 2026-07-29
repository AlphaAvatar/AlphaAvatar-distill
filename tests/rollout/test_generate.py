"""Tests for the in-stack batched generator (src/aadistill/generate.py).

The load-bearing test is `test_batching_does_not_change_tokens`: it is the
property the whole module exists for. If batching changes what a prompt
generates, then a corpus built in batches is not the corpus the model would
have produced one prompt at a time, and Stage 4/5's on-policy assumption is
already wrong before any training happens.

Everything runs on a tiny random model on CPU in fp32, which is the *friendly*
case — fp32 has enough mantissa that padding-induced reduction differences
rarely flip an argmax. Passing here does not prove the property for a 4B model
in bf16, which is why `assert_batch_invariant` exists as a runtime check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.rollout.generate import _left_pad, assert_batch_invariant, generate_ids

VOCAB = 64
EOS = 2


def tiny_model(seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2,
        intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, tie_word_embeddings=True, max_position_embeddings=256,
        eos_token_id=EOS, pad_token_id=EOS,
    )
    return Qwen3ForCausalLM(cfg).float().eval()


def prompts():
    # Deliberately uneven lengths: equal lengths would not exercise padding,
    # which is the thing most likely to break batch invariance.
    return [
        [5, 9, 13],
        [7, 11, 17, 19, 23, 29],
        [3],
        [31, 37, 41, 43],
    ]


def test_left_pad_shapes_and_mask():
    ids, mask = _left_pad([[1, 2], [3, 4, 5], [6]], pad_id=0)
    assert ids.shape == (3, 3) and mask.shape == (3, 3)
    assert ids.tolist() == [[0, 1, 2], [3, 4, 5], [0, 0, 6]]
    assert mask.tolist() == [[0, 1, 1], [1, 1, 1], [0, 0, 1]]
    # Real tokens are right-aligned so every row's last position is real.
    assert all(row[-1] == 1 for row in mask.tolist())


def test_returns_completions_in_input_order_without_the_prompt():
    model = tiny_model()
    ps = prompts()
    out = generate_ids(model, ps, max_new_tokens=6, eos_token_id=EOS,
                       batch_size=2, greedy=True, device="cpu")
    assert len(out) == len(ps)
    for r in out:
        assert 0 < len(r["tokens"]) <= 6
        assert r["n_new"] == len(r["tokens"])
        # The prompt must not be echoed back.
        assert r["tokens"] != ps[0][: len(r["tokens"])] or len(r["tokens"]) != len(ps[0])
    # Internally the module sorts by length; results must come back in input
    # order, which a length-sorted implementation gets wrong if it forgets to
    # restore the mapping.
    single = generate_ids(model, [ps[2]], max_new_tokens=6, eos_token_id=EOS,
                          batch_size=1, greedy=True, device="cpu")
    assert out[2]["tokens"] == single[0]["tokens"]


def test_batching_does_not_change_tokens():
    """The property the module exists for (fp32 CPU, the friendly case)."""
    model = tiny_model()
    ps = prompts()
    alone = generate_ids(model, ps, max_new_tokens=8, eos_token_id=EOS,
                         batch_size=1, greedy=True, device="cpu")
    batched = generate_ids(model, ps, max_new_tokens=8, eos_token_id=EOS,
                           batch_size=4, greedy=True, device="cpu")
    for i, (a, b) in enumerate(zip(alone, batched)):
        assert a["tokens"] == b["tokens"], (
            f"prompt {i}: batching changed the generated tokens\n"
            f"  alone  : {a['tokens']}\n  batched: {b['tokens']}"
        )


def test_batch_position_does_not_change_tokens():
    """Same prompt, different position in the batch, same tokens."""
    model = tiny_model()
    target = [7, 11, 17, 19, 23, 29]
    filler = [[3], [5, 9, 13], [31, 37, 41, 43]]
    first = generate_ids(model, [target] + filler, max_new_tokens=8,
                         eos_token_id=EOS, batch_size=4, greedy=True, device="cpu")
    last = generate_ids(model, filler + [target], max_new_tokens=8,
                        eos_token_id=EOS, batch_size=4, greedy=True, device="cpu")
    assert first[0]["tokens"] == last[-1]["tokens"]


def test_assert_batch_invariant_reports_cleanly():
    model = tiny_model()
    report = assert_batch_invariant(model, prompts(), eos_token_id=EOS,
                                    max_new_tokens=8, device="cpu")
    assert report["n"] == 4
    assert report["identical"] is True
    assert report["n_diverged"] == 0
    assert all(d is None for d in report["first_divergence"])


def test_eos_truncates_and_flags_termination():
    """A completion's length must not depend on its batch-mates."""
    model = tiny_model(seed=3)
    ps = prompts()
    out = generate_ids(model, ps, max_new_tokens=8, eos_token_id=EOS,
                       batch_size=4, greedy=True, device="cpu")
    for r in out:
        if r["finished"]:
            # eos is kept, and only as the final token.
            assert r["tokens"][-1] == EOS
            assert EOS not in r["tokens"][:-1]
            assert r["hit_cap"] is False
        else:
            assert EOS not in r["tokens"]


def test_greedy_is_reproducible_and_sampling_is_seeded():
    model = tiny_model()
    ps = prompts()
    g1 = generate_ids(model, ps, max_new_tokens=6, eos_token_id=EOS,
                      batch_size=2, greedy=True, device="cpu")
    g2 = generate_ids(model, ps, max_new_tokens=6, eos_token_id=EOS,
                      batch_size=2, greedy=True, device="cpu")
    assert [r["tokens"] for r in g1] == [r["tokens"] for r in g2]

    s1 = generate_ids(model, ps, max_new_tokens=6, eos_token_id=EOS,
                      batch_size=2, greedy=False, seed=1234, device="cpu")
    s2 = generate_ids(model, ps, max_new_tokens=6, eos_token_id=EOS,
                      batch_size=2, greedy=False, seed=1234, device="cpu")
    assert [r["tokens"] for r in s1] == [r["tokens"] for r in s2], "seed must reproduce"


def test_model_left_in_its_original_mode():
    model = tiny_model()
    model.train()
    generate_ids(model, [[5, 9]], max_new_tokens=2, eos_token_id=EOS,
                 batch_size=1, greedy=True, device="cpu")
    assert model.training is True, "generation must not silently leave eval() on"
