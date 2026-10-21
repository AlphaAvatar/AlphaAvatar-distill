"""The process check for STATIC procedural targets: what can be shown, and no more.

The answer check (the generator's gold) says whether the final answer is right.
This says something narrower about the reasoning that produced it, in the
contract's vocabulary (`aadistill.data.target_quality`):

* ``contradicted`` -- the reasoning boxes an answer and its LAST boxed answer
  canonicalizes to something other than the final answer: the trace concluded
  one thing and answered another;
* ``partial`` -- the final answer is derived inside the reasoning: its canonical
  form is boxed there, or its text appears in the reasoning's closing quarter;
* ``not_assessed`` -- neither can be shown. This is the honest default and it
  does not reject a target.

It is NOT a proof that each step is valid. Episodes are checked by environment
replay instead (every action legal), code by execution; a sampled independent
review per family is part of the protocol, not of this function.
"""
from __future__ import annotations

import re

from aadistill.data.target_quality import Check
from stages.calibration_volume.tasks.common import CANON_BY_PROTOCOL

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
PROTOCOL = "v2"


def static_process_check(reasoning: str, final_canon: str | None, answer_type: str,
                         *, prompt: str = "", protocol: str = PROTOCOL) -> Check:
    """`prompt` is the task's own text: a boxed value it shows (a format example
    such as ``\\boxed{120 / 003 / 450}``) is quoted, not concluded, when the
    reasoning repeats it, and is ignored."""
    canon = CANON_BY_PROTOCOL[protocol][answer_type]
    if final_canon is None:
        return Check("not_assessed", "no final answer to compare")
    shown = {canon(b) for b in _BOXED.findall(prompt or "")}
    boxed = [b for b in _BOXED.findall(reasoning or "") if canon(b) not in shown]
    if boxed:
        last = canon(boxed[-1])
        if last is not None and last != final_canon:
            return Check("contradicted", "last boxed answer in the reasoning differs from the final")
        if last == final_canon:
            return Check("partial", "final answer boxed inside the reasoning")
    tail = " ".join((reasoning or "")[-max(1, len(reasoning or "") // 4):].split()).lower()
    if final_canon and str(final_canon).lower() in tail:
        return Check("partial", "final answer stated in the reasoning's closing quarter")
    return Check("not_assessed", "the reasoning does not show the final answer")
