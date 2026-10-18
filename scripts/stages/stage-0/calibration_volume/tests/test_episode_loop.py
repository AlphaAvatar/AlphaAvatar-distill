"""`run_battery.run_episodes` driven by a scripted engine, at $0.

The engine is fake; the episode logic is not. What this protects is the part a
paid pod would otherwise be the first to exercise: the history each turn is
rendered from, the environment step, and how an episode ends. vLLM's own calls
(add_request / step / abort_request) are the ones `uncapped_eval` already runs
on every D-series pod.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aadistill.evaluation.behavior import split_generation, template_opens_think
from stages.calibration_volume import score_battery as S
from stages.calibration_volume.run_battery import run_episodes
from stages.calibration_volume.tests.test_tasks import DEV, _oracle_replies

EOS = 0


class Tok:
    """Characters as token ids; id 0 is <|im_end|>."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return body + "<|im_start|>assistant\n<think>\n"

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join("<|im_end|>" if i == EOS else chr(i) for i in ids)


class Engine:
    def __init__(self, reply):
        self.reply, self.queue, self.prompts = reply, [], []

    def add_request(self, rid, prompt, params):
        text = "".join(chr(i) for i in prompt["prompt_token_ids"])
        self.prompts.append(text)
        out = self.reply(rid, text)
        if out.endswith("~"):  # never emits EOS: runs to its allowance, as vLLM would
            ids = ([ord(c) for c in out] * (params // len(out) + 1))[:params]
            finish = "length"
        else:
            ids, finish = [ord(c) for c in out] + [EOS], "stop"
        self.queue.append(SimpleNamespace(request_id=rid, finished=True, outputs=[
            SimpleNamespace(token_ids=ids, finish_reason=finish)]))

    def has_unfinished_requests(self):
        return bool(self.queue)

    def step(self):
        out, self.queue = self.queue, []
        return out

    def abort_request(self, rid):
        pass


def _run(tasks, reply, limit=10 ** 6):
    eng = Engine(reply)
    eps, rounds, _ = run_episodes(
        tasks, tok=Tok(), eng=eng, make_sampling=lambda n, seed: n, limit=limit,
        ctx=limit, stop_ids=[EOS], mode="eval", seed_base=1, system="SYS",
        check_every=10 ** 9, degeneration_check=lambda ids: None,
        split_generation=split_generation, template_opens_think=template_opens_think,
        label="t", log=lambda *a, **k: None)
    return eps, rounds, eng


def _records(eps):
    return [{"task_id": tid, "label": "t", "end": e["end"], "turns": e["turns"],
             "env_result": e["env"].result() if e["env"] else None} for tid, e in eps.items()]


def test_static_tasks_end_answered_and_score():
    tasks = [json.loads(l) for l in Path(DEV).read_text().splitlines()][:40]
    tasks = [t for t in tasks if not t["interactive"]]
    by_id = {t["task_id"]: t for t in tasks}
    eps, rounds, _ = _run(tasks, lambda rid, p: f"think</think>\\boxed{{{by_id[rid.split('::')[0]]['answer']}}}")
    assert rounds == 1
    assert all(e["end"] == "answered" for e in eps.values())
    rows = S.score({t["task_id"]: t for t in tasks}, _records(eps))
    assert all(r["correct"] for r in rows)


def test_oracle_episodes_succeed_through_the_real_environment():
    tasks = [json.loads(l) for l in Path(DEV).read_text().splitlines()]
    tasks = [t for t in tasks if t["interactive"]]
    plans = {t["task_id"]: _oracle_replies(t) for t in tasks}

    def reply(rid, prompt):
        tid, turn = rid.split("::")
        return "think</think>" + plans[tid][int(turn)]

    eps, rounds, eng = _run(tasks, reply)
    assert all(e["end"] == "episode_done" for e in eps.values())
    rows = S.score({t["task_id"]: t for t in tasks}, _records(eps))
    assert all(r["correct"] and r["replay_matches_pod"] for r in rows)
    # every later turn is rendered with the previous observation as a user turn
    multi = [e for e in eps.values() if len(e["turns"]) > 1]
    assert multi
    for e in multi:
        obs = e["turns"][0]["observation"]
        assert any(obs in p for p in eng.prompts)


def test_a_reply_that_never_terminates_ends_the_episode_censored():
    tasks = [t for t in (json.loads(l) for l in Path(DEV).read_text().splitlines())
             if t["interactive"]][:5]
    eps, _, _ = _run(tasks, lambda rid, p: "thinking forever~", limit=4000)
    assert all(e["end"] == "context_limit" and len(e["turns"]) == 1 for e in eps.values())
    assert all(e["turns"][0]["right_censored"] for e in eps.values())
    rows = S.score({t["task_id"]: t for t in tasks}, _records(eps))
    assert not any(r["correct"] for r in rows)


def test_context_exhaustion_is_recorded_not_raised():
    tasks = [t for t in (json.loads(l) for l in Path(DEV).read_text().splitlines())
             if not t["interactive"]][:3]
    eps, _, _ = _run(tasks, lambda rid, p: "x</think>\\boxed{1}", limit=10)
    assert all(e["end"] == "context_exhausted" and not e["turns"] for e in eps.values())
