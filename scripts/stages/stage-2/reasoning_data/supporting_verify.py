"""Answer checks for the SUPPORTING sources of stage2.reasoning_master@v1. $0.

    VERIFIERS[task["verifier"]](task, reasoning, final) -> Check

One entry per supporting source kind, each reading the gold fields its pool
task carries (`build_supporting_tasks.py`) and judging the FINAL answer -- the
text after ``</think>`` -- alone. Each Check's `scope` says what a pass shows
and no more; none of them inspects a reasoning step (steps stay
``not_assessed``, `process_check.NO_STEP_CHECKER`).

* ``boxed_math`` (MATH train): the final answer's last ``\\boxed{}`` against
  the reference solution's last ``\\boxed{}``, as a normalized STRING
  (`aadistill.data.verify.normalize_math`: LaTeX spacing, ``\\left``/``\\right``,
  ``\\dfrac``). No symbolic equivalence: ``0.5`` against ``\\frac{1}{2}`` is
  incorrect, a false rejection rather than a false acceptance.
* ``code_asserts`` (MBPP): the final answer's last fenced Python block, run
  against EVERY declared assert of the task by `aadistill.data.code_exec.run_asserts`
  inside the probed bubblewrap sandbox (`code_sandbox.verified_sandbox`, one
  probe per process). A machine whose probe fails raises: no teacher program
  runs on a bare host, and code is never silently judged unverifiable there.
  A pass means the declared tests passed (`code_exec.ESTABLISHES`).
* ``abstention`` (SQuAD v2 unanswerable): the final answer LEADS with an
  explicit statement that the supplied context does not contain the answer,
  and asserts no answer anywhere (`is_abstention`). Leading, as a yes/no gold
  must lead its answer in `verify.py`: an answer followed by "the context does
  not specify X" is an answer with a caveat, not an abstention. Phrase patterns
  and position only -- there is no word-count rule (AGENTS.md P3), and the core
  `verify.py` refusal length limit is not used.
* ``final_number`` / ``span``: the corpus_v2 reuse checks, `aadistill.data.verify`'s
  own key rules (``verify_answer_key``: no hygiene length rule), used when a
  reused session is selected.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from aadistill.data.code_exec import ESTABLISHES, Limits, run_asserts
from aadistill.data.target_quality import Check
from aadistill.data.verify import boxed_answer, normalize_math, verify_answer_key

MATH_SCOPE = ("the final answer's last \\boxed{} against the reference solution's last \\boxed{}; "
              "normalized string match, no symbolic equivalence")
CODE_SCOPE = f"the final answer's last ```python block against every declared assert, sandboxed: {ESTABLISHES}"
ABSTAIN_SCOPE = ("an explicit statement in the final answer that the supplied context does not contain the "
                 "answer, with no answer asserted beside it; phrase patterns, no length rule")
NUMBER_SCOPE = "the final answer's last number against the human gold's (aadistill.data.verify final_number)"
SPAN_SCOPE = ("the normalized gold span contained in the final answer, a yes/no gold leading it "
              "(aadistill.data.verify span)")
CODE_LIMITS = Limits(wall_seconds=10, cpu_seconds=5, memory_mb=512)


def boxed_math(task: dict, reasoning: str, final: str) -> Check:
    got = boxed_answer(final or "")
    if got is None:
        return Check("incorrect", "no boxed answer in the final answer", MATH_SCOPE)
    ok = normalize_math(got) == normalize_math(str(task["answer"]))
    return Check("correct" if ok else "incorrect", "" if ok else "wrong answer", MATH_SCOPE)


_FENCE = re.compile(r"```[ \t]*(python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)
_sandbox = None


def code_block(final: str) -> tuple[str | None, int]:
    """The last fenced block tagged python (or untagged), and how many there were."""
    blocks = [code for _, code in _FENCE.findall(final or "")]
    return (blocks[-1] if blocks else None), len(blocks)


def _the_sandbox():
    global _sandbox
    if _sandbox is None:
        from stages.reasoning_data import code_sandbox as CS
        _sandbox = CS.verified_sandbox()[0]        # SystemExit when not isolated
    return _sandbox


def code_asserts(task: dict, reasoning: str, final: str) -> Check:
    code, n = code_block(final)
    if code is None:
        return Check("incorrect", "no fenced code block in the final answer", CODE_SCOPE)
    res = run_asserts(code, task["test_list"], setup=task.get("test_setup_code") or "",
                      limits=CODE_LIMITS, sandbox=_the_sandbox())
    note = f" (last of {n} blocks)" if n > 1 else ""
    if res["passed"]:
        return Check("correct", f"{res['tests_passed']}/{res['tests_total']} asserts passed{note}", CODE_SCOPE)
    return Check("incorrect", f"{res['status']}: {res['tests_passed']}/{res['tests_total']} passed{note}",
                 CODE_SCOPE)


#: What a statement that the context lacks the answer looks like. Each pattern
#: names the CONTEXT (context / passage / text / paragraph / provided or given
#: information) as what lacks it, so a bare "I'm not sure" is not an abstention.
_SRC = r"(?:(?:provided|given|supplied|above)\s+)?(?:context|passage|text|paragraph|information|excerpt|document)"
_ABSTAIN = tuple(re.compile(p, re.I) for p in (
    _SRC + r"\s+(?:does not|doesn't|did not|do not|fails to)\s+(?:explicitly\s+|directly\s+|specifically\s+)?"
           r"(?:contain|mention|provide|include|specify|state|say|give|address|cover|answer|discuss|"
           r"indicate|identify|describe|offer|reference|name|tell)",
    r"(?:cannot|can't|can not|unable to|not possible to|impossible to)\s+(?:be\s+)?"
    r"(?:answer(?:ed)?|determine[d]?|find|found|tell|say|identif(?:y|ied)|infer(?:red)?|deduce[d]?)"
    r"\b[^.\n]{0,120}?\b(?:from|based on|using|with|in|within|by)\s+(?:the\s+|this\s+)?" + _SRC,
    r"(?:\bnot|n't|\bnever|\bnowhere)\s+(?:explicitly\s+|directly\s+)?(?:mentioned|stated|provided|specified|given|"
    r"contained|found|addressed|included|discussed|described|covered|answered|indicated|referenced)"
    r"\s+(?:anywhere\s+)?(?:in|by|within)\s+(?:the\s+|this\s+)?" + _SRC,
    r"(?:no|not any|not enough|insufficient)\s+(?:information|mention|details?|data|evidence|indication)\b"
    r"[^.\n]{0,120}?\b(?:in|within|from)\s+(?:the\s+|this\s+)?" + _SRC,
    _SRC + r"\s+(?:is|are)\s+(?:silent|insufficient)\b",
))
#: An answer asserted beside the abstention ("... but the answer is X", a
#: boxed value, an outside-knowledge answer): then it is not an abstention.
_ASSERTS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:the\s+)?(?:correct\s+|final\s+|likely\s+|probable\s+)?answer\s+(?:is|would be|might be|may be|"
    r"could be|seems to be|appears to be)\b(?!\s+(?:not|unknown|unavailable|absent|missing)\b)",
    r"\b(?:based on|from|using|according to)\s+(?:my\s+|general\s+|common\s+|outside\s+|external\s+|prior\s+|"
    r"background\s+|world\s+)knowledge\b",
    r"\b(?:however|but|although|though),?\s+(?:it\s+is|it's|this\s+is)\s+(?:likely|probably|possibly|"
    r"presumably|known)\b",
    r"\\boxed\{",
))


_LABEL = re.compile(r"^(?:[#*_\s]*(?:final\s+)?answer[*_\s]*:[*_\s]*)", re.I)
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s")


def lead_sentence(final: str) -> str:
    """The final answer's first sentence, an ``Answer:`` label removed."""
    text = _LABEL.sub("", " ".join((final or "").split()))
    m = _SENTENCE_END.search(text)
    return text[:m.end()].strip() if m else text


def is_abstention(final: str) -> tuple[bool, str]:
    text = " ".join((final or "").split())
    if not text:
        return False, "empty final answer"
    if not any(p.search(lead_sentence(text)) for p in _ABSTAIN):
        return False, "the answer does not lead with a statement that the context lacks the answer"
    also = next((m.group(0) for p in _ASSERTS for m in [p.search(text)] if m), None)
    if also:
        return False, f"an answer is asserted beside the abstention: {also[:60]!r}"
    return True, "explicit abstention"


def abstention(task: dict, reasoning: str, final: str) -> Check:
    ok, why = is_abstention(final)
    return Check("correct" if ok else "incorrect", "" if ok else why, ABSTAIN_SCOPE)


def _key(rule: str, scope: str) -> Callable[[dict, str, str], Check]:
    group = {"final_number": ("code_math", "gsm8k"), "span": None}[rule]

    def check(task: dict, reasoning: str, final: str) -> Check:
        g, s = group or (task["group"], task["corpus_source"])
        ok, why = verify_answer_key({"group": g, "source": s,
                                     "messages": [{"role": "assistant", "content": task["answer"]}]}, final or "")
        return Check("correct" if ok else "incorrect", "" if ok else why, scope)
    return check


VERIFIERS: dict[str, Callable[[dict, str, str], Check]] = {
    "boxed_math": boxed_math,
    "code_asserts": code_asserts,
    "abstention": abstention,
    "final_number": _key("final_number", NUMBER_SCOPE),
    "span": _key("span", SPAN_SCOPE),
}
