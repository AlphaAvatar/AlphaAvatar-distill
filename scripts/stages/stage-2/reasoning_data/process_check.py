"""What can be shown about a STATIC target's reasoning, kept in separate records.

Three different questions, three different records
(`aadistill.data.target_quality`):

* the **answer** -- the final answer against the generator's gold (the
  battery's scorer; not this module);
* **text consistency** (an observation, `text_consistency`) -- does the
  reasoning restate the final answer? ``restated`` when the answer's canonical
  form is boxed in the reasoning or its text is in the reasoning's closing
  quarter; ``not_restated`` otherwise; ``not_assessed`` with no final answer.
  This is textual agreement and nothing more. A DIFFERENT value boxed in the
  reasoning is noted in the reason and never treated as a contradiction: a
  boxed value inside a derivation may be an intermediate quantity. And an
  answer restated at the end does not show that the derivation is right;
* the **steps** -- intermediate-step verification. Only a check that reads the
  steps against the task sets it. For the project's static generators there is
  none, so their steps are ``not_assessed``, and are reported so.
  `reference_steps` is the one step check available today: for a task that
  ships a reference proof (PrOntoQA), every intermediate conclusion must be
  stated in the reasoning and none negated. Stating a conclusion is not
  deriving it validly, so a pass is ``partial``, never ``verified``.

Episodes are covered elsewhere: environment replay checks the ACTIONS
(observation `action_legality`), which says nothing about the natural-language
reasoning, so an episode's steps stay ``not_assessed`` too. Code is checked by
execution (`aadistill.data.code_exec`), whose pass means its declared tests
passed.
"""
from __future__ import annotations

import re

from aadistill.data.target_quality import Check
from stages.calibration_volume.tasks.common import CANON_BY_PROTOCOL

_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
PROTOCOL = "v2"
NO_STEP_CHECKER = Check("not_assessed", "no intermediate-step checker for this family")
TEXT_SCOPE = "textual agreement between the reasoning and the final answer; not a check of any step"
STEP_SCOPE = "each reference-proof conclusion stated in the reasoning and none negated; presence, not validity"


def text_consistency(reasoning: str, final_canon: str | None, answer_type: str,
                     *, prompt: str = "", protocol: str = PROTOCOL) -> Check:
    """`prompt` is the task's own text: a boxed value it shows (a format example
    such as ``\\boxed{120 / 003 / 450}``) is quoted, not concluded, when the
    reasoning repeats it, and is ignored."""
    if final_canon is None:
        return Check("not_assessed", "no final answer to compare", TEXT_SCOPE)
    canon = CANON_BY_PROTOCOL[protocol][answer_type]
    shown = {canon(b) for b in _BOXED.findall(prompt or "")}
    boxed = [c for c in (canon(b) for b in _BOXED.findall(reasoning or "")) if c is not None and c not in shown]
    later = ""
    if final_canon in boxed:
        after = boxed[len(boxed) - 1 - boxed[::-1].index(final_canon) + 1:]
        later = "; a different value is boxed after it (possibly an intermediate)" if after else ""
        return Check("restated", "final answer boxed in the reasoning" + later, TEXT_SCOPE)
    tail = " ".join((reasoning or "")[-max(1, len(reasoning or "") // 4):].split()).lower()
    if final_canon and str(final_canon).lower() in tail:
        return Check("restated", "final answer stated in the reasoning's closing quarter", TEXT_SCOPE)
    other = "; other values are boxed (possibly intermediates)" if boxed else ""
    return Check("not_restated", "the reasoning does not restate the final answer" + other, TEXT_SCOPE)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _negated(sentence: str) -> str:
    n = _norm(sentence)
    return n.replace(" is not ", " is ", 1) if " is not " in n else n.replace(" is ", " is not ", 1)


def reference_steps(reasoning: str, conclusions: list[str]) -> Check:
    """Every reference-proof conclusion present in the reasoning and none of
    them negated there: ``partial``. A negated one: ``contradicted``. A missing
    one: ``not_assessed`` (the steps are not shown)."""
    if not conclusions:
        return Check("not_assessed", "the task ships no reference conclusions", STEP_SCOPE)
    text = f" {_norm(reasoning or '')} "
    negated = [c for c in conclusions if f" {_negated(c)} " in text]
    if negated:
        return Check("contradicted", f"negates a reference conclusion: {negated[0][:80]}", STEP_SCOPE)
    missing = [c for c in conclusions if f" {_norm(c)} " not in text]
    if missing:
        return Check("not_assessed", f"{len(missing)} of {len(conclusions)} reference conclusions not stated",
                     STEP_SCOPE)
    return Check("partial", f"all {len(conclusions)} reference conclusions stated", STEP_SCOPE)
