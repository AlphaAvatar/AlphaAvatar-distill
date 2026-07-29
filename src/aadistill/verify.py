"""Correctness verification for teacher-generated targets.

A teacher answer replaces a public target **only** if it verifies against a gold
key (decision record 2026-07-28). The gold key needs no new data: the mixture-v1
builders wrote the reference answer as the assistant message, so the target a
candidate would replace *is* the key it is checked against.

One rule per slice, all mechanical, all reusing the scorers `eval_behavior_v0`
already ships:

| slice | key | rule |
| --- | --- | --- |
| `rag_evidence` (squad_v2) | extractive span | the normalized span appears in the answer |
| `multihop_qa` (hotpot_qa) | short answer | same, plus yes/no golds must *lead* the answer |
| `refusal_uncertainty` (squad_v2 unanswerable) | "not answerable here" | the answer is a refusal and stays short |
| `code_math` (gsm8k) | final number | exact match on the final number |
| `code_math` (openmath_instruct_2) | `\\boxed{…}` | normalized boxed-answer match |

Everything else is out of scope by design: mbpp/magicoder would need sandboxed
test execution, `tool_calling` gold is already schema-valid, and `instruction` /
`short_realtime` have no mechanical key at all.

The yes/no rule is not decoration. HotpotQA yes/no items are ~10% of the slice,
and plain containment passes for any answer that happens to contain the word
"no" — including one that says the opposite.
"""

from __future__ import annotations

import re

from .behavior import (
    IM_END,
    STRAY_MARKERS,
    contains_gold,
    final_number,
    is_refusal,
    normalize_text,
)

# Slices this module can verify, keyed by (group, source).
VERIFIABLE = {
    ("rag_evidence", "squad_v2"): "span",
    ("multihop_qa", "hotpot_qa"): "span",
    ("refusal_uncertainty", "squad_v2"): "refusal",
    ("code_math", "gsm8k"): "final_number",
    ("code_math", "openmath_instruct_2"): "boxed",
}

# A refusal that runs on is not a refusal worth training on.
REFUSAL_MAX_WORDS = 60
# Answers far longer than the target they replace are rejected as runaway.
MAX_ANSWER_WORDS = 600

_BOXED = r"\boxed{"
_MATH_NOISE = (r"\left", r"\right", r"\!", r"\,", r"\;", r"\ ", "$", " ", "\n")


def boxed_answer(text: str) -> str | None:
    """Contents of the last `\\boxed{…}`, brace-balanced (nested `\\frac{}{}`)."""
    start = text.rfind(_BOXED)
    if start == -1:
        return None
    i, depth, out = start + len(_BOXED), 1, []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(char)
        i += 1
    return None  # unbalanced: treat as absent rather than guessing


def normalize_math(value: str) -> str:
    """Spacing and LaTeX-noise-insensitive comparison key for a math answer."""
    out = value.strip().rstrip(".")
    for noise in _MATH_NOISE:
        out = out.replace(noise, "")
    out = out.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return out.lower()


def leading_token(text: str) -> str:
    match = re.match(r"[a-z0-9]+", normalize_text(text))
    return match.group(0) if match else ""


def hygiene_reason(answer: str, raw: str) -> str | None:
    """Rejections that apply to every slice. None means the answer is usable."""
    if not answer.strip():
        return "empty"
    if IM_END not in raw:
        return "not_terminated"
    if any(marker in answer for marker in STRAY_MARKERS):
        return "stray_marker"
    if len(normalize_text(answer).split()) > MAX_ANSWER_WORDS:
        return "too_long"
    return None


def verify(sample: dict, answer: str, raw: str) -> tuple[bool, str]:
    """Is `answer` a correct replacement for `sample`'s public target?

    `raw` is the full generation (used for the termination check); `answer` is
    the text after `</think>`. Returns (accepted, reason) — the reason is
    recorded for accepted and rejected candidates alike, so accept rates can be
    broken down by failure mode.
    """
    rule = VERIFIABLE.get((sample["group"], sample["source"]))
    if rule is None:
        return False, "unverifiable_slice"

    reason = hygiene_reason(answer, raw)
    if reason:
        return False, reason

    gold = sample["messages"][-1]["content"]

    if rule == "span":
        if not contains_gold(answer, gold):
            return False, "gold_span_missing"
        # Containment is trivially satisfiable for yes/no golds.
        if normalize_text(gold) in ("yes", "no"):
            if leading_token(answer) != normalize_text(gold):
                return False, "yesno_not_leading"
        return True, "ok"

    if rule == "refusal":
        if not is_refusal(answer):
            return False, "not_a_refusal"
        if len(normalize_text(answer).split()) > REFUSAL_MAX_WORDS:
            return False, "refusal_too_long"
        return True, "ok"

    if rule == "final_number":
        got, want = final_number(answer), final_number(gold)
        if want is None:
            return False, "gold_has_no_final_number"
        if got is None:
            return False, "no_final_number"
        return (True, "ok") if got == want else (False, "answer_mismatch")

    if rule == "boxed":
        got, want = boxed_answer(answer), boxed_answer(gold)
        if want is None:
            return False, "gold_has_no_boxed"
        if got is None:
            return False, "no_boxed"
        return ((True, "ok") if normalize_math(got) == normalize_math(want)
                else (False, "answer_mismatch"))

    raise AssertionError(f"unhandled rule {rule!r}")  # pragma: no cover


def select(candidates: list[dict]) -> dict | None:
    """Pick the target from n verified candidates, or None if none verified.

    The **median-length** accepted candidate, tie-broken by candidate index.

    Not "shortest correct": on the math slices that systematically selects
    answers that skip the derivation (`The answer is 42.`), which would train the
    student to state answers without working them out.

    Candidate 0 used to win outright whenever it verified, on the grounds that
    it was greedy — "the teacher's modal answer and the only deterministic one".
    Both halves of that are gone (2026-07-29). Answer generation no longer
    decodes a greedy candidate at all, and the determinism claim did not survive
    measurement: bf16 greedy decoding is not batch-invariant, so candidate 0 was
    never reproducible across batch compositions the way the rule assumed
    ([log](../../logs/experiments/2026-07-29_engine_adapter_and_bf16_invariance.md)).
    With every candidate an equal draw, privileging index 0 would just be
    selecting on batch position.
    """
    accepted = [c for c in candidates if c["accepted"]]
    if not accepted:
        return None
    by_length = sorted(accepted, key=lambda c: (len(c["answer"].split()), c["index"]))
    return by_length[len(by_length) // 2]
