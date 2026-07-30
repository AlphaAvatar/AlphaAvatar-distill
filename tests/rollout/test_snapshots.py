"""Tests for rollout snapshots and off-policy correction diagnostics.

These are the pieces a Stage 4/5 correction term is computed from, and every one
of them fails *silently* when it fails: a one-position alignment slip, a masked
value quietly imputed, or a snapshot read back after the file changed all produce
plausible numbers rather than errors. So the tests here are mostly about
alignment, masking and integrity rather than happy-path shapes.

The load-bearing one is `test_scorer_recovers_the_generation_logprobs`: it checks
the trainer-side scorer against the *generation-time* log-probs of the same
model, which is the only end-to-end evidence that the two halves of an importance
ratio refer to the same token.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.rollout.engines import HFEngine
from aadistill.rollout.snapshots import (
    aggregate_stats, importance_stats, read_snapshot, score_tokens, write_snapshot,
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


# --- the scorer's alignment ------------------------------------------------


def test_scorer_recovers_the_generation_logprobs():
    """The end-to-end alignment check.

    Generate with log-probs, then recompute them teacher-forced with the scorer.
    Same model, same tokens, so the two must agree. If the scorer were off by one
    position, the values would differ while still looking like reasonable
    log-probs — which is exactly the failure this guards.
    """
    model = tiny_model(seed=3)
    engine = HFEngine(model, pad_token_id=EOS, batch_size=1)
    prompt = [5, 6, 7, 8]
    out = engine.generate([prompt], max_new_tokens=6, stop_ids=set(), logprobs=True)[0]

    recomputed = score_tokens(model, prompt, out["tokens"])
    assert len(recomputed) == len(out["tokens"])
    for generated, scored in zip(out["logprobs"], recomputed):
        assert generated == pytest.approx(scored, abs=1e-4)


def test_scorer_returns_one_value_per_completion_token():
    model = tiny_model(seed=4)
    assert len(score_tokens(model, [5, 6, 7], [10, 11, 12, 13])) == 4
    assert score_tokens(model, [5, 6], []) == []


def test_scorer_values_are_valid_logprobs():
    model = tiny_model(seed=5)
    for value in score_tokens(model, [5, 6, 7], [10, 11]):
        assert value <= 0.0 and value > -50.0


# --- log-probs travel with the tokens through the shared trim path ---------


def test_generate_logprobs_align_with_tokens_after_stop_trim():
    """`_finalize` cuts at the stop token; log-probs must be cut identically."""
    model = tiny_model(seed=6)
    engine = HFEngine(model, pad_token_id=EOS, batch_size=2)
    prompts = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    for out in engine.generate(prompts, max_new_tokens=8, stop_ids={EOS}, logprobs=True):
        assert len(out["logprobs"]) == len(out["tokens"]) == out["n_new"]


def test_generate_without_logprobs_omits_the_key():
    model = tiny_model(seed=7)
    engine = HFEngine(model, pad_token_id=EOS, batch_size=1)
    out = engine.generate([[5, 6]], max_new_tokens=4, stop_ids=set())[0]
    assert "logprobs" not in out


def test_engine_rejects_a_backend_that_cannot_supply_logprobs():
    """Silently returning no log-probs would leave Stage 4/5 uncorrectable."""
    from aadistill.rollout.engines import Engine

    class _NoLogprobs(Engine):
        name = "mute"

        def _raw_generate(self, prompts, **kw):
            return [([1, 2], "length") for _ in prompts]

    with pytest.raises(RuntimeError, match="cannot supply rollout log-prob"):
        _NoLogprobs().generate([[9]], max_new_tokens=4, stop_ids=set(), logprobs=True)


# --- importance-ratio diagnostics ------------------------------------------


def test_identical_policies_give_unit_ratios_and_zero_kl():
    lp = [-1.0, -2.0, -0.5]
    stats = importance_stats(lp, list(lp))
    assert stats["ratio_median"] == pytest.approx(1.0)
    assert stats["kl"] == pytest.approx(0.0, abs=1e-9)
    assert stats["off_policy_rate"] == 0.0
    assert stats["n"] == 3 and stats["n_masked"] == 0


def test_kl_is_non_negative_for_diverging_policies():
    """The k3 estimator is non-negative by construction; a negative value would
    mean the ratio maths is wrong."""
    stats = importance_stats([-1.0, -3.0, -0.2], [-2.0, -1.0, -4.0])
    assert stats["kl"] >= 0.0


def test_off_policy_rate_counts_both_tails():
    # ratios: e^1 = 2.72 (> band 2), e^-1 = 0.37 (< 1/2), e^0 = 1 (inside)
    stats = importance_stats([-2.0, -1.0, -1.0], [-1.0, -2.0, -1.0], band=2.0)
    assert stats["off_policy_rate"] == pytest.approx(2 / 3)


def test_masked_positions_are_dropped_not_imputed():
    """A re-appended stop token has no rollout log-prob. Imputing one would bias
    every statistic; it must be excluded and counted instead."""
    stats = importance_stats([-1.0, None, -1.0], [-1.0, -5.0, -1.0])
    assert stats["n"] == 2 and stats["n_masked"] == 1
    assert stats["ratio_median"] == pytest.approx(1.0)


def test_all_masked_reports_nothing_rather_than_zero():
    stats = importance_stats([None, None], [-1.0, -2.0])
    assert stats["n"] == 0 and stats["n_masked"] == 2
    assert stats["kl"] is None and stats["off_policy_rate"] is None


def test_extreme_ratios_do_not_overflow():
    stats = importance_stats([-500.0], [0.0])
    assert stats["ratio_max"] < float("inf")


def test_importance_stats_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        importance_stats([-1.0, -2.0], [-1.0])


def test_aggregate_pools_by_token_not_by_sequence():
    """A 1-token sequence must not weigh as much as a 99-token one."""
    long_seq = importance_stats([-1.0] * 99, [-1.0] * 99)          # 0% off-policy
    short = importance_stats([-1.0], [-10.0])                       # 100% off-policy
    pooled = aggregate_stats([long_seq, short])
    assert pooled["tokens"] == 100
    assert pooled["off_policy_rate"] == pytest.approx(0.01)


def test_aggregate_handles_all_empty():
    assert aggregate_stats([importance_stats([None], [-1.0])])["tokens"] == 0


# --- snapshot integrity -----------------------------------------------------


def records():
    return [{"prompt_id": "p0", "prompt_tokens": [5, 6], "tokens": [7, 8],
             "logprobs": [-0.5, -1.5]}]


def identity():
    return (dict(checkpoint="stage3/x/step_001", step=1, sha256="abc"),
            dict(name="vllm_server", version="0.11.0"),
            dict(temperature=1.0, top_p=1.0, top_k=0))


def test_snapshot_round_trips_and_verifies(tmp_path):
    policy, engine, sampling = identity()
    manifest = write_snapshot(tmp_path, records(), policy=policy, engine=engine,
                              sampling=sampling)
    assert manifest["n_records"] == 1 and manifest["n_tokens"] == 2
    back, read_manifest = read_snapshot(tmp_path)
    assert back == records()
    assert read_manifest["policy"]["checkpoint"] == "stage3/x/step_001"
    assert read_manifest["engine"]["name"] == "vllm_server"


def test_snapshot_detects_tampering(tmp_path):
    """The hash is the point: a correction computed from edited rollouts is
    against different data than the record claims."""
    policy, engine, sampling = identity()
    write_snapshot(tmp_path, records(), policy=policy, engine=engine, sampling=sampling)
    path = tmp_path / "rollouts.jsonl"
    row = json.loads(path.read_text())
    row["tokens"] = [7, 9]
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_snapshot(tmp_path)


def test_snapshot_requires_policy_and_engine_identity(tmp_path):
    _, engine, sampling = identity()
    with pytest.raises(ValueError, match="checkpoint"):
        write_snapshot(tmp_path, records(), policy={"step": 1}, engine=engine,
                       sampling=sampling)
    with pytest.raises(ValueError, match="engine identity"):
        write_snapshot(tmp_path, records(), policy=dict(checkpoint="c", step=1),
                       engine={}, sampling=sampling)


def test_snapshot_rejects_misaligned_logprobs(tmp_path):
    policy, engine, sampling = identity()
    bad = records()
    bad[0]["logprobs"] = [-0.5]
    with pytest.raises(ValueError, match="logprobs for"):
        write_snapshot(tmp_path, bad, policy=policy, engine=engine, sampling=sampling)


def test_kl_uses_the_same_clamped_ratio_as_the_quantiles():
    """Regression: an earlier version clamped the ratio but fed the *unclamped*
    log-ratio into the KL term, so an extreme token made `kl` disagree with the
    distribution every other statistic described."""
    stats = importance_stats([-500.0, -1.0], [0.0, -1.0])
    # r - 1 - log r for the clamped pair, averaged with the identical pair's 0.
    expected = ((math.exp(80.0) - 1.0 - 80.0) + 0.0) / 2
    assert stats["kl"] == pytest.approx(expected, rel=1e-9)


def test_corpus_build_snapshot_shape_round_trips(tmp_path):
    """The corpus builder writes its snapshot in the *last* step of a paid
    generation run, so a schema mismatch there costs the whole run. This pins
    the record shape generate_teacher_answers.py actually emits, including the
    `id#candidate_index` prompt key that keeps n candidates distinct."""
    records_out = [
        {"prompt_id": "squad_v2-000000#0", "slice": "rag_evidence",
         "prompt_tokens": [5, 6, 7], "tokens": [11, 12, EOS],
         "logprobs": [-0.1, -0.2, -0.3], "accepted": True, "reason": "ok"},
        {"prompt_id": "squad_v2-000000#1", "slice": "rag_evidence",
         "prompt_tokens": [5, 6, 7], "tokens": [11, 13],
         "logprobs": [-0.1, -0.9], "accepted": False, "reason": "wrong_answer"},
    ]
    manifest = write_snapshot(
        tmp_path, records_out,
        policy={"checkpoint": "Qwen/Qwen3-4B-Thinking-2507@768f209d",
                "step": None, "role": "teacher", "revision": "768f209d"},
        engine={"name": "vllm_server", "server_url": "http://127.0.0.1:8000",
                "version": None},
        sampling={"n": 2, "greedy": False, "temperature": 1.0, "top_p": 1.0,
                  "top_k": 0, "max_new_tokens": 4096, "seed_base": 20260728,
                  "seed_per_batch": "seed + batch_index"},
    )
    assert manifest["n_records"] == 2
    assert manifest["n_tokens"] == 5

    back, read_manifest = read_snapshot(tmp_path)
    assert read_manifest["rollouts_sha256"] == manifest["rollouts_sha256"]
    assert [r["prompt_id"] for r in back] == [r["prompt_id"] for r in records_out]
    # Two candidates of the same prompt must not collide into one record.
    assert len({r["prompt_id"] for r in back}) == 2
    assert back[0]["logprobs"] == [-0.1, -0.2, -0.3]


def test_snapshot_tolerates_an_engine_that_reports_no_logprobs(tmp_path):
    """`--snapshot` off, or a backend that cannot supply them: the record still
    has to be writable, with None rather than a fabricated value."""
    manifest = write_snapshot(
        tmp_path,
        [{"prompt_id": "gsm8k-000001#0", "prompt_tokens": [1],
          "tokens": [2, 3], "logprobs": None}],
        policy={"checkpoint": "c", "step": None},
        engine={"name": "hf"}, sampling={"n": 1},
    )
    assert manifest["n_tokens"] == 2
    back, _ = read_snapshot(tmp_path)
    assert back[0]["logprobs"] is None
