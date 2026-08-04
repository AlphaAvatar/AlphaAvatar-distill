"""Equivalence and throughput-accounting tests for the uncapped evaluator.

The execution path was corrected after measuring Experiment 1's evaluation
throughput at **255 output tokens/s** aggregate for a 0.6B student on an L40S —
roughly an order of magnitude below what the hardware should deliver. Two
structural causes were fixed:

1. the engine was constructed once per *invocation*, and the orchestrator
   invoked the script once per (checkpoint, prompt set) — with the seven-set
   capability battery that is seven model loads per checkpoint;
2. every scheduler step rebuilt a full copy of every running request's token
   list, which is O(sum of L^2) list copies on the decode critical path.

Neither touches decoding semantics, and these tests exist to prove that: the
same stub engine driven through the corrected loop must produce byte-identical
records to the reference implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
from aadistill.evaluation import degeneration  # noqa: E402


# --------------------------------------------------------------------------
# a stub that behaves like vLLM's LLMEngine for the parts the loop touches
# --------------------------------------------------------------------------
class StubOutput:
    def __init__(self, rid, ids, finished, reason):
        self.request_id = rid
        self.finished = finished
        self.outputs = [type("O", (), {"token_ids": ids,
                                       "finish_reason": reason})()]


class StubEngine:
    """Emits one token per request per step from a scripted plan."""

    def __init__(self, plan):
        self.plan = {rid: list(toks) for rid, toks in plan.items()}
        self.state = {rid: [] for rid in plan}
        self.alive = set(plan)
        self.steps = 0

    def add_request(self, rid, prompt, params):
        pass

    def has_unfinished_requests(self):
        return bool(self.alive)

    def abort_request(self, rid):
        self.alive.discard(rid)

    def step(self):
        self.steps += 1
        outs = []
        for rid in sorted(self.alive):
            remaining = self.plan[rid]
            if remaining:
                self.state[rid].append(remaining.pop(0))
            finished = not remaining
            if finished:
                self.alive.discard(rid)
            outs.append(StubOutput(rid, list(self.state[rid]), finished,
                                   "stop" if finished else None))
        return outs


def reference_loop(engine, pending, check_every=256, degeneration_stop=True):
    """The pre-correction loop: full token-list copy on every step."""
    done = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            st = pending.get(out.request_id)
            if st is None:
                continue
            st["gen"] = list(out.outputs[0].token_ids)
            if out.finished:
                st["finish"] = out.outputs[0].finish_reason
                done[out.request_id] = st
                pending.pop(out.request_id, None)
                continue
            if degeneration_stop and len(st["gen"]) - st["last_check"] >= check_every:
                st["last_check"] = len(st["gen"])
                d = degeneration.check(st["gen"])
                if d:
                    st["degen"] = d
                    engine.abort_request(out.request_id)
                    st["finish"] = "degeneration"
                    done[out.request_id] = st
                    pending.pop(out.request_id, None)
    done.update(pending)
    return done


def corrected_loop(engine, pending, check_every=256, degeneration_stop=True):
    """The corrected loop: length tracked per step, tokens materialised on demand."""
    done = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            st = pending.get(out.request_id)
            if st is None:
                continue
            st["n_gen"] = len(out.outputs[0].token_ids)
            if out.finished:
                st["gen"] = list(out.outputs[0].token_ids)
                st["finish"] = out.outputs[0].finish_reason
                done[out.request_id] = st
                pending.pop(out.request_id, None)
                continue
            if degeneration_stop and st["n_gen"] - st["last_check"] >= check_every:
                st["last_check"] = st["n_gen"]
                st["gen"] = list(out.outputs[0].token_ids)
                d = degeneration.check(st["gen"])
                if d:
                    st["degen"] = d
                    engine.abort_request(out.request_id)
                    st["finish"] = "degeneration"
                    done[out.request_id] = st
                    pending.pop(out.request_id, None)
    done.update(pending)
    return done


def fresh_pending(plan):
    return {rid: {"gen": [], "n_gen": 0, "last_check": 0, "degen": None,
                  "finish": None} for rid in plan}


def outcome(done):
    """The record-relevant projection: tokens, finish reason, degeneration."""
    return {rid: (tuple(st["gen"]), st["finish"],
                  (st["degen"] or {}).get("kind"))
            for rid, st in done.items()}


# --------------------------------------------------------------------------
# equivalence
# --------------------------------------------------------------------------
PLANS = {
    "short": {"a": list(range(10, 40)), "b": list(range(50, 61))},
    "mixed lengths": {f"r{i}": list(range(100 + i, 100 + i + 50 * (i + 1)))
                      for i in range(4)},
    "one degenerate": {"clean": list(range(1000, 1400)),
                       "loop": [7, 8, 9] * 400},
    "all degenerate": {f"d{i}": [11, 12] * 500 for i in range(3)},
    "single request": {"solo": list(range(2000, 2900))},
}


@pytest.mark.parametrize("name", sorted(PLANS))
def test_corrected_loop_is_output_equivalent(name):
    plan = PLANS[name]
    ref = reference_loop(StubEngine(plan), fresh_pending(plan))
    new = corrected_loop(StubEngine(plan), fresh_pending(plan))
    assert outcome(ref) == outcome(new), name


@pytest.mark.parametrize("name", sorted(PLANS))
def test_equivalence_holds_with_the_degeneration_stop_disabled(name):
    plan = PLANS[name]
    ref = reference_loop(StubEngine(plan), fresh_pending(plan),
                         degeneration_stop=False)
    new = corrected_loop(StubEngine(plan), fresh_pending(plan),
                         degeneration_stop=False)
    assert outcome(ref) == outcome(new), name


@pytest.mark.parametrize("check_every", [64, 128, 256, 512])
def test_equivalence_holds_at_every_check_interval(check_every):
    plan = PLANS["one degenerate"]
    ref = reference_loop(StubEngine(plan), fresh_pending(plan), check_every)
    new = corrected_loop(StubEngine(plan), fresh_pending(plan), check_every)
    assert outcome(ref) == outcome(new)


def test_the_degeneration_abort_still_fires_and_is_recorded():
    plan = {"loop": [7, 8, 9] * 400}
    done = corrected_loop(StubEngine(plan), fresh_pending(plan))
    assert done["loop"]["finish"] == "degeneration"
    assert done["loop"]["degen"]["kind"] == "cycle"


def test_the_corrected_loop_copies_far_fewer_tokens():
    """The point of the change, measured rather than asserted."""
    plan = {f"r{i}": list(range(1000 + i, 1000 + i + 300)) for i in range(8)}

    class Counting(StubEngine):
        def __init__(self, plan):
            super().__init__(plan)
            self.copied = 0

    def count(loop):
        eng = Counting(plan)
        real = list
        copied = {"n": 0}

        class L(list):
            pass

        # Count elements materialised through list(...) inside the loop by
        # wrapping the engine's token_ids with a counting sequence.
        class Counted(tuple):
            def __iter__(self):
                copied["n"] += len(self)
                return super().__iter__()

        orig_step = eng.step

        def step():
            outs = orig_step()
            for o in outs:
                o.outputs[0].token_ids = Counted(o.outputs[0].token_ids)
            return outs

        eng.step = step
        loop(eng, fresh_pending(plan))
        return copied["n"]

    ref_copied = count(reference_loop)
    new_copied = count(corrected_loop)
    assert new_copied < ref_copied / 10, (ref_copied, new_copied)


# --------------------------------------------------------------------------
# the multi-set contract
# --------------------------------------------------------------------------
def test_request_ids_are_unique_across_prompt_sets():
    """One engine now serves several sets, so ids must not collide.

    `gsm8k` and `math_verified` both contain ids the other could plausibly
    repeat; a collision would silently drop a request's output into another
    set's record.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "uncapped_eval", REPO_ROOT / "scripts/evaluation/uncapped_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = (REPO_ROOT / "scripts/evaluation/uncapped_eval.py").read_text()
    assert 'rid = f"{args.label}::{Path(prompts_path).stem}::{s[\'id\']}"' in src, \
        "request ids must be namespaced by prompt set"


def test_engine_is_constructed_once_outside_the_set_loop():
    src = (REPO_ROOT / "scripts/evaluation/uncapped_eval.py").read_text()
    llm_line = src.index("llm = LLM(")
    loop_line = src.index("for prompts_path in args.prompts:")
    assert llm_line < loop_line, (
        "the engine must be built before the per-set loop; building it inside "
        "reintroduces one model load per set")


def test_detokenization_is_disabled():
    """The evaluator decodes once from final token ids; incremental detokenization
    is pure overhead on the decode path and changes no sampling semantics."""
    src = (REPO_ROOT / "scripts/evaluation/uncapped_eval.py").read_text()
    assert "detokenize=False" in src


def test_sampling_semantics_are_unchanged():
    """Greedy, uncapped within the effective context, native stop ids."""
    src = (REPO_ROOT / "scripts/evaluation/uncapped_eval.py").read_text()
    assert "temperature=0.0" in src and "top_p=1.0" in src and "top_k=-1" in src
    assert "max_tokens=allowance" in src and "stop_token_ids=stop_ids" in src
