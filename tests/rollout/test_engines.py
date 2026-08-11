"""Tests for the engine adapter layer (src/aadistill/engines.py).

vLLM and SGLang are CUDA-only and cannot run on the dev box, so what is tested
here is everything that is *not* engine-specific: the shared post-processing
that makes the comparison fair, the defenses against the two known engine
quirks, and the measurement functions that produce the benchmark's verdict.

That split is deliberate. The adapters are thin by design — if the shared layer
is right, an adapter can only be wrong in ways the pod-side smoke test catches
in seconds. The logic that would silently corrupt a *result* rather than crash
lives here and is tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.rollout.engines import (
    Engine, HFEngine, SGLangEngine, SGLangServerEngine, VLLMServerEngine, _finalize,
    _strip_prefix, agreement,
    batch_invariance, timed,
)

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
    return Qwen3ForCausalLM(cfg).eval()


# --- _strip_prefix: the SGLang output_ids overlap defense ------------------


def test_strip_prefix_removes_whole_prompt_echo():
    prompt = [10, 11, 12]
    assert _strip_prefix([10, 11, 12, 20, 21], prompt) == [20, 21]


def test_strip_prefix_leaves_clean_output_alone():
    assert _strip_prefix([20, 21], [10, 11, 12]) == [20, 21]
    assert _strip_prefix([], [10, 11]) == []


def test_strip_prefix_does_not_guess_at_partial_overlaps():
    """The regression this file already caught once — see `_strip_prefix`.

    A completion may legitimately begin with the token the prompt ended on.
    Stripping "the longest prompt-suffix that is also an output-prefix" silently
    shortens such completions, which corrupts a corpus rather than crashing on
    it. sglang#10896 is handled exactly, in the SGLang adapter, from that
    engine's reported `completion_tokens`.
    """
    assert _strip_prefix([7, 7, 7, 30], [5, 6, 7]) == [7, 7, 7, 30]
    assert _strip_prefix([12, 20, 21], [10, 11, 12]) == [12, 20, 21]


# --- _finalize: one post-processing path for every engine ------------------


def test_finalize_cuts_at_first_stop_and_keeps_it():
    out = _finalize([5, 6, EOS, 9, 9], "stop", {EOS}, cap=100)
    assert out == {"tokens": [5, 6, EOS], "n_new": 3, "hit_cap": False, "finished": True}


def test_finalize_reappends_stop_when_engine_strips_it():
    """vLLM omits the stop token by default; lengths must still be comparable."""
    out = _finalize([5, 6], "stop", {EOS}, cap=100)
    assert out["tokens"] == [5, 6, EOS]
    assert out["finished"] is True and out["hit_cap"] is False


def test_finalize_marks_cap_hit():
    out = _finalize([5, 6, 7], "length", {EOS}, cap=3)
    assert out["hit_cap"] is True and out["finished"] is False


def test_finalize_uses_lowest_stop_id_canonically():
    """With several stop ids the re-appended one must be deterministic."""
    assert _finalize([5], "stop", {9, 2, 7}, cap=100)["tokens"] == [5, 2]


# --- agreement: the metric the engine choice turns on ----------------------


def test_agreement_all_match():
    a = [{"tokens": [1, 2, 3]}, {"tokens": [4, 5]}]
    result = agreement(a, [dict(x) for x in a])
    assert result["exact_match_rate"] == 1.0
    assert result["first_divergence"] == [None, None]
    assert result["median_divergence_token"] is None


def test_agreement_reports_where_it_diverges():
    a = [{"tokens": [1, 2, 3]}, {"tokens": [4, 5, 6]}]
    b = [{"tokens": [1, 2, 9]}, {"tokens": [4, 5, 6]}]
    result = agreement(a, b)
    assert result["exact_match"] == 1
    assert result["first_divergence"] == [2, None]


def test_agreement_counts_a_length_difference_as_divergence():
    a = [{"tokens": [1, 2, 3]}]
    b = [{"tokens": [1, 2]}]
    assert agreement(a, b)["first_divergence"] == [2]


def test_agreement_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        agreement([{"tokens": [1]}], [])


# --- the Engine contract ---------------------------------------------------


class _DropsARow(Engine):
    name = "drops"

    def _raw_generate(self, prompts, **kw):
        return [([1], "length")] * (len(prompts) - 1)


class _Echoes(Engine):
    """Returns the prompt plus one token — the whole-sequence-return convention."""

    name = "echo"

    def _raw_generate(self, prompts, **kw):
        return [(list(p) + [42], "length") for p in prompts]


def test_engine_rejects_a_row_count_mismatch():
    """Silently dropping a row would misalign every target after it."""
    with pytest.raises(RuntimeError, match="cannot build a corpus"):
        _DropsARow().generate([[1], [2]], max_new_tokens=8, stop_ids={EOS})


def test_engine_strips_echoed_prompts_through_the_shared_path():
    out = _Echoes().generate([[10, 11], [20]], max_new_tokens=8, stop_ids={EOS})
    assert [o["tokens"] for o in out] == [[42], [42]]
    assert [o["n_new"] for o in out] == [1, 1]


# --- the in-stack adapter on a real (tiny) model ---------------------------


def test_hf_engine_returns_completions_in_input_order():
    """Length-sorted batching reorders internally; results must not be."""
    model = tiny_model()
    engine = HFEngine(model, pad_token_id=EOS, batch_size=2)
    prompts = [[5, 6, 7, 8, 9], [11], [12, 13], [14, 15, 16]]
    out = engine.generate(prompts, max_new_tokens=6, stop_ids={EOS})
    assert len(out) == len(prompts)

    alone = [engine.generate([p], max_new_tokens=6, stop_ids={EOS})[0] for p in prompts]
    # Same prompt, same tokens, whatever position it was batched at. This is the
    # toy-model version of the property the pod run checks on the real 4B.
    assert [o["tokens"] for o in out] == [a["tokens"] for a in alone]


def test_hf_engine_is_batch_invariant_on_a_toy_model():
    model = tiny_model(seed=1)
    engine = HFEngine(model, pad_token_id=EOS, batch_size=4)
    prompts = [[5, 6, 7], [8, 9], [10, 11, 12, 13], [14]]
    result = batch_invariance(engine, prompts, stop_ids={EOS}, max_new_tokens=8)
    assert result["n"] == 4
    assert result["identical"], result["first_divergence"]


def test_hf_engine_respects_the_cap():
    model = tiny_model(seed=2)
    engine = HFEngine(model, pad_token_id=EOS, batch_size=2)
    out = engine.generate([[5, 6, 7]], max_new_tokens=4, stop_ids=set())
    assert out[0]["n_new"] <= 4
    assert out[0]["hit_cap"] is True


# --- the SGLang overlap fix, without SGLang --------------------------------


def _sglang_stub(outputs):
    """An SGLangEngine whose `__init__` is bypassed, so the CUDA-only import
    is never reached and only the response-parsing contract is exercised."""
    engine = object.__new__(SGLangEngine)
    engine.engine = type("E", (), {"generate": staticmethod(lambda **kw: outputs)})()
    engine.deterministic = False
    return engine


def _raw(engine, prompts):
    return engine._raw_generate(
        prompts, max_new_tokens=8, stop_ids={EOS}, greedy=True,
        temperature=1.0, top_p=1.0, top_k=0, seed=None)


def test_sglang_trims_the_echo_using_completion_tokens():
    """sglang#10896, resolved from the engine's own count rather than guessed."""
    prompt = [10, 11, 12]
    outputs = [{"output_ids": [11, 12, 20, 21],
                "meta_info": {"completion_tokens": 2, "finish_reason": {"type": "stop"}}}]
    assert _raw(_sglang_stub(outputs), [prompt]) == [([20, 21], "stop")]


def test_sglang_keeps_a_completion_that_repeats_the_prompt_tail():
    """The case the old heuristic corrupted: real tokens equal to the prompt tail."""
    prompt = [5, 6, 7]
    outputs = [{"output_ids": [7, 7, 7], "meta_info": {"completion_tokens": 3}}]
    assert _raw(_sglang_stub(outputs), [prompt]) == [([7, 7, 7], "length")]


def test_sglang_falls_back_to_whole_prompt_echo_without_a_count():
    prompt = [10, 11]
    outputs = [{"output_ids": [10, 11, 30], "meta_info": {}}]
    assert _raw(_sglang_stub(outputs), [prompt]) == [([30], "length")]


def test_sglang_refuses_a_build_without_output_ids():
    """Re-encoding `text` would reintroduce retokenization drift silently."""
    with pytest.raises(RuntimeError, match="token-in/token-out"):
        _raw(_sglang_stub([{"text": "hello", "meta_info": {}}]), [[1, 2]])


def test_timed_returns_result_and_a_duration(monkeypatch):
    """The return contract, held apart from GPU health on purpose.

    `timed` calls `torch.cuda.synchronize()` when CUDA is present, and that call
    re-raises any *earlier* async CUDA fault in the process as a sticky error. So
    this assertion — which is about returning `(value, seconds)` for a pure-CPU
    callable — used to fail on a GPU host for reasons that had nothing to do with
    it, and passed on the dev box only because CUDA is absent there. It aborted an
    E8b-S2 pod at $0.41 with `AcceleratorError` after 1,219 other tests had run.

    The synchronize is correct and stays; it is covered by the test below.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    value, seconds = timed(lambda: 21 * 2)
    assert value == 42
    assert seconds >= 0.0


def test_timed_drains_the_gpu_before_and_after(monkeypatch):
    """The reason `timed` exists: without the drain a CUDA-async engine looks fast.

    Asserted through a fake `torch.cuda`, so it holds on a CPU-only box and does
    not depend on a healthy accelerator.
    """
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize",
                        lambda *a, **k: calls.append("sync"))
    value, seconds = timed(lambda: (calls.append("work"), 42)[1])
    assert value == 42
    assert seconds >= 0.0
    # Drain, run, drain — in that order, so the timing brackets the real work.
    assert calls == ["sync", "work", "sync"], calls


# --- the HTTP adapter, against a stub server -------------------------------


def _server_stub(response):
    """A VLLMServerEngine whose `_post` is replaced, so the contract is tested
    without a GPU, a vLLM install, or a socket."""
    engine = object.__new__(VLLMServerEngine)
    engine.base_url, engine.model, engine.timeout = "http://x", "m", 1.0
    engine.sent = {}

    def _post(path, payload):
        engine.sent = payload
        return response
    engine._post = _post
    return engine


def _sraw(engine, prompts, **kw):
    params = dict(max_new_tokens=8, stop_ids={EOS}, greedy=True,
                  temperature=1.0, top_p=1.0, top_k=0, seed=None)
    params.update(kw)
    return engine._raw_generate(prompts, **params)


def test_server_engine_sends_token_ids_not_text():
    """Text in the request would reintroduce the retokenization drift the whole
    token-in/token-out contract exists to remove."""
    engine = _server_stub({"choices": [{"index": 0, "token_ids": [5, 6],
                                        "finish_reason": "stop"}]})
    _sraw(engine, [[1, 2, 3]])
    assert engine.sent["prompt"] == [[1, 2, 3]]
    assert engine.sent["return_token_ids"] is True
    assert engine.sent["stop_token_ids"] == [EOS]


def test_server_engine_orders_by_index_not_arrival():
    """The OpenAI schema carries an index because order is not contractual.
    Pairing completions with the wrong prompts is silently wrong."""
    engine = _server_stub({"choices": [
        {"index": 2, "token_ids": [30], "finish_reason": "length"},
        {"index": 0, "token_ids": [10], "finish_reason": "length"},
        {"index": 1, "token_ids": [20], "finish_reason": "length"},
    ]})
    rows = _sraw(engine, [[1], [2], [3]])
    assert [ids for ids, _ in rows] == [[10], [20], [30]]


def test_server_engine_rejects_a_response_without_token_ids():
    engine = _server_stub({"choices": [{"index": 0, "text": "hello"}]})
    with pytest.raises(RuntimeError, match="token-in/token-out"):
        _sraw(engine, [[1]])


def test_server_engine_rejects_a_choice_count_mismatch():
    engine = _server_stub({"choices": [{"index": 0, "token_ids": [1]}]})
    with pytest.raises(RuntimeError, match="choices for"):
        _sraw(engine, [[1], [2]])


def test_server_engine_rejects_an_out_of_range_index():
    engine = _server_stub({"choices": [{"index": 7, "token_ids": [1]}]})
    with pytest.raises(RuntimeError, match="out of range"):
        _sraw(engine, [[1]])


def test_server_engine_greedy_ignores_sampling_params():
    engine = _server_stub({"choices": [{"index": 0, "token_ids": [1]}]})
    _sraw(engine, [[1]], greedy=True, temperature=0.7, top_p=0.9, top_k=20, seed=5)
    assert engine.sent["temperature"] == 0.0
    assert engine.sent["top_p"] == 1.0
    assert engine.sent["top_k"] == -1
    assert "seed" not in engine.sent


def test_server_engine_passes_sampling_params_when_sampling():
    engine = _server_stub({"choices": [{"index": 0, "token_ids": [1]}]})
    _sraw(engine, [[1]], greedy=False, temperature=1.0, top_p=1.0, top_k=0, seed=7)
    assert engine.sent["temperature"] == 1.0
    assert engine.sent["top_k"] == -1  # 0 means disabled, which vLLM spells -1
    assert engine.sent["seed"] == 7


# --- SGLang native HTTP adapter, against a stub server ---------------------


def _sglang_server_stub(response):
    engine = object.__new__(SGLangServerEngine)
    engine.base_url, engine.timeout = "http://x", 1.0
    engine.sent = {}

    def _post(path, payload):
        engine.sent = payload
        return response
    engine._post = _post
    return engine


def _ssraw(engine, prompts, **kw):
    params = dict(max_new_tokens=8, stop_ids={EOS}, greedy=True,
                  temperature=1.0, top_p=1.0, top_k=0, seed=None)
    params.update(kw)
    return engine._raw_generate(prompts, **params)


def test_sglang_server_sends_input_ids_and_takes_output_ids():
    engine = _sglang_server_stub([{"output_ids": [20, 21],
                                   "meta_info": {"completion_tokens": 2,
                                                 "finish_reason": {"type": "stop"}}}])
    assert _ssraw(engine, [[10, 11]]) == [([20, 21], "stop")]
    assert engine.sent["input_ids"] == [[10, 11]]


def test_sglang_server_trims_echo_by_completion_tokens():
    """sgl-project/sglang#10896, resolved exactly rather than guessed."""
    engine = _sglang_server_stub([{"output_ids": [11, 20, 21],
                                   "meta_info": {"completion_tokens": 2}}])
    assert _ssraw(engine, [[10, 11]]) == [([20, 21], "length")]


def test_sglang_server_verifies_logprob_token_alignment():
    """A positional zip would attach each probability to its neighbour's token."""
    good = [{"output_ids": [20, 21],
             "meta_info": {"completion_tokens": 2,
                           "output_token_logprobs": [[-0.5, 20, "a"], [-1.5, 21, "b"]]}}]
    assert _ssraw(_sglang_server_stub(good), [[10]], logprobs=True) == \
        [([20, 21], "length", [-0.5, -1.5])]

    bad = [{"output_ids": [20, 21],
            "meta_info": {"completion_tokens": 2,
                          "output_token_logprobs": [[-0.5, 99, "a"], [-1.5, 21, "b"]]}}]
    with pytest.raises(RuntimeError, match="do not match output_ids"):
        _ssraw(_sglang_server_stub(bad), [[10]], logprobs=True)


def test_sglang_server_refuses_a_build_without_output_ids():
    with pytest.raises(RuntimeError, match="token-in/token-out"):
        _ssraw(_sglang_server_stub([{"text": "hi", "meta_info": {}}]), [[1]])
