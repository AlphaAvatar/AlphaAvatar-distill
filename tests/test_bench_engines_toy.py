"""Tests for the benchmark's verdict logic (scripts/bench_engines.py) and the
engine-driven candidate generation (scripts/generate_teacher_answers.py).

Neither of these can be checked by the pod-side smoke test: a wrong adapter
crashes, but a wrong *selection rule* or a wrong candidate-to-prompt mapping
produces a plausible-looking artifact with the wrong contents. The session is
meant to run unattended and chain straight into a corpus build, so these are
exactly the failures that would otherwise be discovered after the spend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench_engines import cost_per_1k, decide
from generate_teacher_answers import generate_candidates

THINK_CLOSE_ID = 99


def arm(name, cost, agreement=None, ok=True):
    a = {"engine": name, "ok": ok, "best": {"cost_usd_per_1k_prompts": cost}}
    if agreement is not None:
        a["agreement_vs_hf"] = {"exact_match_rate": agreement}
    return a


# --- R1: agreement with the training stack is a gate, not a tiebreak -------


def test_a_faster_engine_that_disagrees_is_rejected():
    """Cheap wrong data is not a bargain: a rollout engine that does not match
    the trainer makes 'on-policy' quietly off-policy."""
    arms = [arm("hf", 10.0), arm("vllm", 1.0, agreement=0.10)]
    result = decide(arms, min_agreement=0.90, max_cost_ratio=1.5)
    assert result["winner"] == "hf"
    assert result["rejected"][0]["engine"] == "vllm"
    assert result["rejected"][0]["rule"] == "R1"


def test_an_agreeing_and_much_cheaper_engine_wins():
    arms = [arm("hf", 10.0), arm("vllm", 2.0, agreement=1.0)]
    result = decide(arms, min_agreement=0.90, max_cost_ratio=1.5)
    assert result["winner"] == "vllm"
    assert result["rule"] == "R2"
    assert result["speedup_vs_hf"] == 5.0


# --- R3: a second stack has to earn its keep (P1) --------------------------


def test_a_marginally_faster_engine_does_not_displace_the_incumbent():
    arms = [arm("hf", 10.0), arm("vllm", 8.0, agreement=1.0)]
    result = decide(arms, min_agreement=0.90, max_cost_ratio=1.5)
    assert result["winner"] == "hf"
    assert result["rule"] == "R3"


def test_the_threshold_is_applied_at_its_boundary():
    """Exactly `min_speedup` clears R3; a hair under does not."""
    assert decide([arm("hf", 15.0), arm("vllm", 10.0, agreement=1.0)],
                  min_agreement=0.9, max_cost_ratio=1.5)["winner"] == "vllm"
    assert decide([arm("hf", 14.9), arm("vllm", 10.0, agreement=1.0)],
                  min_agreement=0.9, max_cost_ratio=1.5)["winner"] == "hf"


def test_the_cheapest_eligible_arm_is_chosen_among_several():
    arms = [arm("hf", 10.0), arm("vllm", 4.0, agreement=1.0),
            arm("sglang", 2.0, agreement=0.99)]
    assert decide(arms, min_agreement=0.9, max_cost_ratio=1.5)["winner"] == "sglang"


# --- failure modes: the session must not proceed on a broken measurement ---


def test_no_winner_when_the_reference_arm_failed():
    """Without the in-stack reference there is nothing to judge agreement
    against, so the run must refuse to pick rather than trust throughput alone."""
    arms = [arm("hf", None, ok=False), arm("vllm", 1.0, agreement=1.0)]
    result = decide(arms, min_agreement=0.9, max_cost_ratio=1.5)
    assert result["winner"] is None
    assert "hf" in result["reason"]


def test_a_crashed_arm_is_ignored_not_ranked():
    arms = [arm("hf", 10.0), arm("vllm", 0.1, agreement=1.0, ok=False)]
    result = decide(arms, min_agreement=0.9, max_cost_ratio=1.5)
    assert result["winner"] == "hf"
    assert "vllm" not in result["eligible"]


def test_cost_per_1k_is_none_at_zero_throughput():
    assert cost_per_1k(0.0, 0.86) is None
    assert cost_per_1k(1.0, 3600.0) == 1000.0


# --- candidate mapping: which completion belongs to which prompt ----------


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


class _StubEngine:
    """Returns one completion per submitted prompt, tagged with its own id list,
    so the mapping back to prompts is observable."""

    name = "stub"

    def __init__(self):
        self.calls = []

    def generate(self, prompts, *, max_new_tokens, stop_ids, greedy=True, **kw):
        self.calls.append({"n": len(prompts), "greedy": greedy})
        out = []
        for p in prompts:
            tokens = [p[0], THINK_CLOSE_ID, 7]
            out.append({"tokens": tokens, "n_new": len(tokens),
                        "hit_cap": False, "finished": True})
        return out


def _run(n, prompts):
    engine = _StubEngine()
    result = generate_candidates(
        engine, _StubTokenizer(), prompts, n=n, max_new_tokens=32,
        temperature=1.0, top_p=1.0, top_k=0, seed=1, stops={2},
        think_close=THINK_CLOSE_ID)
    return engine, result


def test_no_candidate_is_generated_greedily():
    """Maintainer decision 2026-07-29: every candidate is an equal sampled draw.
    A greedy candidate would be mode-collapsed by construction, and the
    determinism that once justified it does not survive bf16 batching."""
    engine, per_prompt = _run(1, [[10], [20]])
    assert [len(c) for c in per_prompt] == [1, 1]
    assert engine.calls == [{"n": 2, "greedy": False}]
    assert all(call["greedy"] is False for call in engine.calls)


def test_all_n_candidates_come_from_one_batched_call():
    """One call with prompts replicated n times, not n calls: it is what keeps a
    continuous-batching engine saturated."""
    engine, per_prompt = _run(4, [[10], [20], [30]])
    assert [len(c) for c in per_prompt] == [4, 4, 4]
    assert engine.calls == [{"n": 12, "greedy": False}]


def test_candidates_map_back_to_their_own_prompt():
    """The replication is contiguous per prompt; an off-by-one here would
    attach one prompt's candidates to another prompt's gold key, and every
    verifier verdict downstream would be silently wrong."""
    prompts = [[10], [20], [30]]
    _engine, per_prompt = _run(4, prompts)
    for prompt, candidates in zip(prompts, per_prompt):
        for raw, _n_new, _cap, _think in candidates:
            assert raw.split()[0] == str(prompt[0])


def test_think_tokens_is_the_position_of_the_close_tag():
    _engine, per_prompt = _run(1, [[10]])
    _raw, _n_new, _hit_cap, think_tokens = per_prompt[0][0]
    assert think_tokens == 1  # [prompt0, </think>, 7] -> close tag at index 1
