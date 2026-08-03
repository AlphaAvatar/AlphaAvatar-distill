"""Strict final-answer extraction for verifiable reasoning benchmarks.

The evaluator this replaces scored GSM8K with `behavior.final_number`, i.e. the
**last number anywhere in the answer**. That is too permissive in three ways
that all fire on this project's checkpoints:

* a number inside a `<tool_call>` payload counted as the model's answer, even
  though a GSM8K prompt carries no tool schema and a tool call there is a
  protocol violation;
* an answer that never states a conclusion still scored, as long as some
  arithmetic in the prose happened to end on the gold value;
* a cyclic or rambling generation scored whenever the loop contained the gold
  number, so degeneration could *raise* exact match.

The rule here, pre-registered before the Experiment 2 comparison:

1. prefer the last valid `\\boxed{…}` (brace-balanced);
2. otherwise accept only an explicit standalone `Final Answer:` / `Answer:`
   marker and read the number that follows it;
3. never read anything inside a `<tool_call>` block;
4. no valid final answer means **incorrect**, never "fall back to the last
   number";
5. a protocol-invalid, degenerate or unterminated generation is incorrect even
   when it contains the gold value.

Point 5 is the one that matters for reading a result: it keeps "the model
learned to terminate" from being reported as "the model learned to reason".
Termination and correctness are returned as separate fields so they are never
conflated.
"""

from __future__ import annotations

import re

from .behavior import IM_END, STRAY_MARKERS, split_generation
from ..data.verify import boxed_answer, normalize_math

# `Final Answer:` / `Answer:` at a line start (optionally decorated with markup
# or a leading bullet), which is what "explicit standalone marker" means.
_ANSWER_MARKER = re.compile(
    r"(?im)^[\s>*\-#]*(?:\*\*|__)?\s*(?:final\s+answer|answer)\s*(?:\*\*|__)?\s*[:：]",
)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def strip_tool_calls(text: str) -> str:
    """Remove `<tool_call>` payloads so no number can be read out of one."""
    return _TOOL_CALL_BLOCK.sub(" ", text)


def normalize_number(value: str) -> str | None:
    """Comma- and trailing-zero-normalized numeric key, or None if not numeric."""
    cleaned = value.replace(",", "").replace("$", "").strip().rstrip(".")
    if not cleaned:
        return None
    try:
        f = float(cleaned)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract_final_answer(answer: str) -> tuple[str | None, str]:
    """Return (answer_string, how) under the strict rule. `how` records the path.

    `how` is one of `boxed`, `marker`, `no_final_answer` — recorded per sample so
    a change in extraction path is visible in the audit rather than hidden in an
    aggregate.
    """
    body = strip_tool_calls(answer)

    boxed = boxed_answer(body)
    if boxed is not None:
        return boxed, "boxed"

    matches = list(_ANSWER_MARKER.finditer(body))
    if matches:
        tail = body[matches[-1].end():]
        # Only the first line after the marker: a marker followed by another
        # paragraph of reasoning is not a stated conclusion.
        line = tail.split("\n\n")[0]
        numbers = _NUMBER.findall(line)
        if numbers:
            return numbers[-1], "marker"
        stripped = line.strip()
        if stripped:
            return stripped.split("\n")[0].strip(), "marker"
    return None, "no_final_answer"


def protocol_valid(raw: str) -> tuple[bool, str]:
    """Structural validity of the generation, independent of its content."""
    if IM_END not in raw:
        return False, "not_terminated"
    parts = split_generation(raw)
    if not parts["think_closed"]:
        return False, "think_delimiters_invalid"
    if any(marker in parts["answer"] for marker in STRAY_MARKERS):
        return False, "stray_marker"
    if not parts["answer"].strip():
        return False, "empty_answer"
    if "<tool_call>" in parts["answer"]:
        return False, "unexpected_tool_call"
    return True, "ok"


def score_numeric(record: dict, gold: str) -> dict:
    """Strict exact-match verdict for one stored generation.

    `record` is a row of an `*.generations.jsonl` file: it must carry `raw`, and
    optionally `degeneration_triggered` / `natural_termination`.
    """
    raw = record.get("raw", "")
    parts = split_generation(raw)
    valid, protocol_reason = protocol_valid(raw)
    degenerate = bool(record.get("degeneration_triggered"))

    extracted, how = extract_final_answer(parts["answer"])
    want = normalize_number(gold)
    got = normalize_number(extracted) if extracted is not None else None
    if got is None and extracted is not None:
        got = normalize_math(extracted)
        want = normalize_math(gold) if want is None else want

    answer_matches = got is not None and want is not None and got == want
    correct = bool(answer_matches and valid and not degenerate)

    if correct:
        reason = "ok"
    elif not valid:
        reason = f"protocol:{protocol_reason}"
    elif degenerate:
        reason = f"degenerate:{record.get('degeneration_kind') or 'unknown'}"
    elif how == "no_final_answer":
        reason = "no_final_answer"
    else:
        reason = "answer_mismatch"

    return {
        "correct": correct,
        "reason": reason,
        "extraction": how,
        "extracted": extracted,
        "gold": gold,
        # Reported separately and never folded into `correct`: a naturally
        # terminated wrong answer is better termination, not better reasoning.
        "protocol_valid": valid,
        "protocol_reason": protocol_reason,
        "natural_termination": bool(record.get("natural_termination")),
        "degenerate": degenerate,
        "answer_matches_ignoring_protocol": bool(answer_matches),
    }
