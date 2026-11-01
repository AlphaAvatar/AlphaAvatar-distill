"""Research A v2's code-stability recheck (maintainer review 2026-11-01): an MBPP candidate that passes only at some
string-hash seeds is refused and the task's next candidate considered; a program task whose gold moves with the
seed is refused in both modes."""
from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from stages.teacher_modes import code_stability as CS

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed here")


def _seed0(expr: str) -> str:
    return subprocess.run([sys.executable, "-c", f"print(repr({expr}))"], env={"PYTHONHASHSEED": "0"},
                          capture_output=True, text=True, check=True).stdout.strip()


#: unpacks set(s) into two names, as the refused MBPP candidate d934be3a... did; its asserts hold at seed 0 only
HASHY = "def first(s):\n    a, b = set(s)\n    return a\n"
STABLE = "def first(s):\n    a, b = sorted(set(s))\n    return a\n"


FIRST = "next(iter(set('ab')))"


def _mbpp_task():
    return {"task_id": "mbpp-x", "verifier": "code_asserts", "source": "mbpp_train",
            "test_list": ["assert first('ab') == " + _seed0(FIRST)]}


def _rec(code: str, k: int) -> dict:
    return {"candidate_index": k, "turns": [{"raw": f"<think>\nok\n</think>\n\n```python\n{code}```\n"}]}


def _admit(task, mode, recs, proto):
    """A stand-in for the frozen verdict: the lowest candidate is admitted."""
    if not recs:
        return {"admitted": False, "docs": [], "candidates": []}
    return {"admitted": True, "candidate": recs[0]["candidate_index"], "record": recs[0], "docs": ["d"],
            "candidates": [{"candidate": r["candidate_index"]} for r in recs]}


def test_a_hash_dependent_mbpp_candidate_is_unstable_and_a_stable_one_passes():
    t = _mbpp_task()
    if _seed0(FIRST) != "'a'":
        pytest.skip("seed 0 does not put 'a' first here")
    bad = CS.mbpp(t, _rec(HASHY, 0))
    assert not bad["passed"] and not bad["stable"] and bad["status"] == "unstable"
    assert {r["passed"] for r in bad["by_seed"]} == {True, False}
    assert CS.mbpp(t, _rec(STABLE, 1))["passed"]


def test_the_recheck_refuses_the_unstable_candidate_and_admits_the_next_one():
    if _seed0(FIRST) != "'a'":
        pytest.skip("seed 0 does not put 'a' first here")
    t = _mbpp_task()
    recs = {("mbpp-x", "H-N"): [_rec(HASHY, 0), _rec(STABLE, 1)], ("mbpp-x", "H-T"): [_rec(STABLE, 0)]}
    verdicts = {k: _admit(t, k[1], v, None) for k, v in recs.items()}
    res = CS.recheck(verdicts, {"mbpp-x": t}, recs, None, _admit)
    assert verdicts[("mbpp-x", "H-N")]["candidate"] == 1 and verdicts[("mbpp-x", "H-N")]["excluded_candidates"] == [0]
    assert verdicts[("mbpp-x", "H-T")]["candidate"] == 0 and "unstable_verification" not in verdicts[("mbpp-x", "H-T")]
    assert list(res["changed"]) == ["mbpp-x|H-N"]
    # with no stable candidate left the task is refused in that mode
    recs = {("mbpp-x", "H-N"): [_rec(HASHY, 0)]}
    verdicts = {k: _admit(t, k[1], v, None) for k, v in recs.items()}
    CS.recheck(verdicts, {"mbpp-x": t}, recs, None, _admit)
    assert not verdicts[("mbpp-x", "H-N")]["admitted"] and verdicts[("mbpp-x", "H-N")]["docs"] == []


def _program(body: str, answer: str) -> dict:
    return {"task_id": "rc-x", "source": "reasoning_core", "module": "controlled_code_execution", "answer": answer,
            "messages": [{"role": "user", "content": "Predict the value returned by this Python call.\n```python\n"
                                                     f"def endpoint():\n    {body}\n```\nCall: `endpoint()`\n"}]}


def test_a_program_task_whose_gold_moves_with_the_seed_is_refused_in_both_modes():
    hashy = _program("return list(set('abcdef'))", _seed0("list(set('abcdef'))"))
    fine = _program("return sorted(set('abcdef'))", repr(sorted(set("abcdef"))))
    assert CS.program_parts(fine) == ("def endpoint():\n    return sorted(set('abcdef'))\n", "endpoint()")
    assert CS.program(fine)["passed"]
    r = CS.program(hashy)
    assert not r["passed"] and r["status"] == "unstable"
    tasks = {"rc-x": hashy}
    recs = {("rc-x", "H-T"): [_rec("x", 0)], ("rc-x", "H-N"): [_rec("x", 0)]}
    verdicts = {k: _admit(hashy, k[1], v, None) for k, v in recs.items()}
    CS.recheck(verdicts, tasks, recs, None, _admit)
    assert not any(v["admitted"] for v in verdicts.values())
    assert all(v["unstable_verification"] == "gold" for v in verdicts.values())
