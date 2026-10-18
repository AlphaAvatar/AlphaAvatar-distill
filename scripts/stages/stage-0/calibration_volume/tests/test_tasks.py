"""The Stage-0 task generators, scorer and isolation checks, at $0.

Each test protects a measurement: an instance that is not deterministic is not
reproducible; a scorer that cannot credit a perfect reply or refuse a wrong one
makes every rate on the battery meaningless; an environment an oracle cannot
solve has no achievable success; an isolation check that passes a planted
overlap proves nothing about the real sets.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from aadistill.evaluation.isolation import (IsolationViolation, key_overlaps,
                                            require_disjoint)
from stages.calibration_volume import score_battery as S
from stages.calibration_volume.tasks import registry as R
from stages.calibration_volume.tasks.common import (CANON, DIFFICULTIES,
                                                    build_instance, gold_key)
from stages.calibration_volume.tasks.interactive import (ButtonsEnv, ProbeFunctionEnv,
                                                       max_turns)

REPO = Path(__file__).resolve().parents[5]
DEV = REPO / "data/stages/stage-0/calibration_volume/tasks/dev_v1.jsonl"


def _tasks(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _turn(answer_text, *, eos=True, think="reasoning"):
    raw = f"{think}\n</think>\n\n{answer_text}" + ("<|im_end|>" if eos else "")
    return {"raw": raw, "answer": answer_text, "natural_termination": eos,
            "degeneration_triggered": False, "context_limit_reached": False,
            "think_preopened": True, "stop_reason": "eos" if eos else "length_limited",
            "generated_tokens": 10}


def test_every_family_is_deterministic():
    for gen in R.GENERATORS:
        for fam in gen.families:
            for d in DIFFICULTIES:
                a = build_instance(gen, fam, d, 7)
                b = build_instance(gen, fam, d, 7)
                assert a == b, (gen.generator_id, fam.family_id, d)


def test_split_is_disjoint_and_complete():
    for gen in R.GENERATORS:
        sp = R.family_split(gen)
        assert not set(sp["calibration"]) & set(sp["evaluation"])
        assert set(sp["calibration"]) | set(sp["evaluation"]) == \
            {f.family_id for f in gen.families}
        assert sp["evaluation"], gen.generator_id
        if gen.generator_id in R.EVAL_ONLY_GENERATORS:
            assert not sp["calibration"]


@pytest.mark.parametrize("variant,atype,gold", [
    ("[1, 2, 3]", "int_list", "[1, 2, 3]"), ("1, 2, 3", "int_list", "[1, 2, 3]"),
    ("\\left[1,2,3\\right]", "int_list", "[1, 2, 3]"), ("[]", "int_list", "[]"),
    ("\\text{abc}", "word", "abc"), (" ABC ", "word", "abc"),
    ("120 / 003", "grid", "120 / 003"), ("1 2 0 \\\\ 0 0 3", "grid", "120 / 003"),
    ("\\begin{matrix}1&2&0\\\\0&0&3\\end{matrix}", "grid", "120 / 003"),
    ("Ava", "name", "Ava"), ("\\text{Ava}", "name", "ava"),
    ("Ava and Ben", "name_set", "Ava, Ben"), ("the apple, the key", "name_set", "apple, key"),
    ("none", "name_set", "none"), ("42", "int", "42"), ("-7", "int", "-7"),
    ("JUMP JUMP WALK", "token_seq", "JUMP JUMP WALK"),
    ("jump, jump, walk", "token_seq", "JUMP JUMP WALK"),
])
def test_canonical_variants_match(variant, atype, gold):
    assert CANON[atype](variant) == CANON[atype](gold)


@pytest.mark.parametrize("wrong,atype,gold", [
    ("[1, 2]", "int_list", "[1, 2, 3]"), ("abd", "word", "abc"),
    ("120 / 004", "grid", "120 / 003"), ("Ben", "name", "Ava"),
    ("Ava", "name_set", "Ava, Ben"), ("43", "int", "42"), ("1 2", "int", "12"),
    ("JUMP WALK JUMP", "token_seq", "JUMP JUMP WALK"),
])
def test_canonical_wrong_answers_differ(wrong, atype, gold):
    assert CANON[atype](wrong) != CANON[atype](gold)


def test_perfect_static_replies_score_correct_and_wrong_ones_do_not():
    tasks = [t for t in _tasks(DEV) if not t["interactive"]]
    assert tasks
    for t in tasks:
        good = {"task_id": t["task_id"], "label": "oracle", "end": "answered",
                "turns": [_turn(f"The answer is \\boxed{{{t['answer']}}}.")]}
        row = S.score_static(t, good)
        assert row["correct"], (t["generator_id"], t["family_id"], t["answer"], row)
        bad = {**good, "turns": [_turn("The answer is \\boxed{zzz 999 / 1}.")]}
        assert not S.score_static(t, bad)["correct"]
        cut = {**good, "turns": [_turn(f"\\boxed{{{t['answer']}}}", eos=False)]}
        assert not S.score_static(t, cut)["correct"], "credited outside a usable rollout"
        assert S.score_static(t, {**good, "turns": [_turn(t["answer"])]})["reason"] == "no_boxed"


def _oracle_replies(t):
    spec = t["env"]
    if spec["kind"] == "probe_function":
        return [f"ANSWER: {ProbeFunctionEnv.f(spec, spec['target'])}"]
    if spec["kind"] == "code":
        return [f"GUESS: {spec['code']}"]
    # buttons: shortest press sequence by BFS over the hidden effects
    prev, q = {spec["start"]: None}, deque([spec["start"]])
    while q:
        v = q.popleft()
        if v == spec["goal"]:
            break
        for b in sorted(spec["buttons"]):
            w = ButtonsEnv.apply(spec, b, v)
            if w not in prev:
                prev[w] = (v, b)
                q.append(w)
    path, v = [], spec["goal"]
    while prev[v] is not None:
        v, b = prev[v]
        path.append(b)
    return [f"I will press it.\nPRESS: {b}" for b in reversed(path)]


def test_interactive_oracles_succeed_and_noise_fails():
    tasks = [t for t in _tasks(DEV) if t["interactive"]]
    assert tasks
    for t in tasks:
        rec = {"task_id": t["task_id"], "label": "oracle", "end": "episode_done",
               "turns": [_turn(r) for r in _oracle_replies(t)], "env_result": None}
        row = S.score_episode(t, rec)
        assert row["correct"], (t["family_id"], row)
        assert len(rec["turns"]) <= max_turns(t["env"]), "oracle exceeds the episode bound"
        junk = {**rec, "turns": [_turn("I am not sure.")] * 3}
        assert not S.score_episode(t, junk)["correct"]


def test_isolation_refuses_a_planted_overlap():
    a = [{"family_id": "x", "content_sha256": "1"}]
    b = [{"family_id": "x", "content_sha256": "2"}]
    rep = key_overlaps({"cal": a, "dev": b}, {"family": lambda r: r["family_id"],
                                               "content": lambda r: r["content_sha256"]})
    require_disjoint(rep, [("cal", "dev", "content")])
    with pytest.raises(IsolationViolation):
        require_disjoint(rep, [("cal", "dev", "family")])


def test_built_sets_satisfy_their_declared_isolation():
    d = Path(DEV).parent
    for m in d.glob("*.manifest.json"):
        man = json.loads(m.read_text())
        assert all(c["overlap"] == 0 for c in man["isolation"]["checked"]), m.name
        body = (REPO / man["file"]).read_bytes()
        import hashlib
        assert hashlib.sha256(body).hexdigest() == man["sha256"], m.name
    for t in _tasks(DEV):
        assert t["answer"] is None or gold_key(t) is not None
