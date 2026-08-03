"""Deterministic, versioned target cleaning over retained teacher candidates.

Corpus v2 kept all `n=4` sampled candidates per prompt and accepted the
**lowest-index candidate that passed structural hygiene**; correctness was
computed and stored but deliberately not enforced (`build_recovery_corpus.py`,
2026-08-01 manifest). That leaves measurable defects in the training targets —
most visibly `openmath`, where only 0.380 of candidates carry the right final
answer.

This module re-selects a target for each prompt from the *same* retained
candidates. It generates nothing, so it costs no GPU time, and it can only ever
choose a completion the teacher actually produced under the recorded preset.

Five ordered stages, applied per candidate; the first failure is the recorded
reason, so rejections break down by cause:

1. `serialization` — think/answer split, closed think block, no stray template
   markers, non-empty answer.
2. `correctness`   — the slice's mechanical key (`aadistill.data.verify`).
   Slices with no key are **not** guessed at; they are recorded as
   `unverifiable_slice` and pass this stage, which is a stated limitation
   rather than a claim of correctness.
3. `tool_protocol` — a tool call is forbidden when the prompt carries no tool
   schema; where a schema exists, every call must name a declared tool and
   supply a JSON object of arguments whose required keys are present.
4. `completion`    — natural termination, no context-limit / budget hit, and no
   degeneration under the **current** detector, which is stricter than the one
   the corpus was built with (`rambling` did not exist on 2026-08-01).
5. `selection`     — keep the corpus's own candidate when it survives all four;
   otherwise the shortest surviving candidate by supervised-token length, tie
   broken by candidate index.

The stage order is load-bearing and matches the pre-registered specification:
"shortest" is a tie-break *after* correctness, evidence, protocol validity and
completeness, never a quality criterion of its own.

`RULES_VERSION` is recorded in every audit and every emitted manifest. Changing
a rule requires bumping it, because two corpora cleaned under different rule
sets are not comparable.
"""

from __future__ import annotations

import json
import re

from ..evaluation.behavior import IM_END, STRAY_MARKERS, THINK_CLOSE, split_generation
from .verify import VERIFIABLE, verify_answer_key

RULES_VERSION = "clean-v1"

STAGES = ("serialization", "correctness", "tool_protocol", "completion")

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


# --------------------------------------------------------------------------
# stage 1 — serialization
# --------------------------------------------------------------------------
def check_serialization(candidate: dict) -> str | None:
    """Structural validity of the teacher's own output protocol."""
    raw, answer = candidate["raw"], candidate["answer"]
    if not answer.strip():
        return "empty_answer"
    if THINK_CLOSE not in raw:
        return "think_not_closed"
    parts = split_generation(raw)
    if not parts["think_closed"]:
        return "think_delimiters_invalid"
    if any(marker in answer for marker in STRAY_MARKERS):
        return "stray_marker"
    return None


# --------------------------------------------------------------------------
# stage 2 — correctness / evidence
# --------------------------------------------------------------------------
def check_correctness(candidate: dict, example: dict) -> str | None:
    """Mechanical answer key for the slices that have one.

    Deliberately `verify_answer_key`, not `verify`: the latter also applies
    `hygiene_reason`, whose `MAX_ANSWER_WORDS` cut is a generic length gate that
    AGENTS.md P3/P10 forbid and that the corpus build already refused to apply.
    Structure and completion are this pipeline's stages 1 and 4; stage 2 asks
    only whether the answer is right.
    """
    key = (example["group"], example["source"])
    if key not in VERIFIABLE:
        return None  # recorded as unverifiable_slice by the caller; not a pass claim
    shim = {"group": example["group"], "source": example["source"],
            "messages": [{"role": "assistant", "content": example["gold"]}]}
    ok, verdict = verify_answer_key(shim, candidate["answer"])
    return None if ok else f"correctness:{verdict}"


def is_verifiable(example: dict) -> bool:
    return (example["group"], example["source"]) in VERIFIABLE


# --------------------------------------------------------------------------
# stage 3 — tool protocol
# --------------------------------------------------------------------------
def _declared_tools(tools: list | None) -> dict:
    """Map declared tool name -> (properties, required) from the schema list."""
    out = {}
    for entry in tools or []:
        fn = entry.get("function", entry) if isinstance(entry, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        required = params.get("required") if isinstance(params, dict) else None
        out[name] = (props if isinstance(props, dict) else {},
                     required if isinstance(required, list) else [])
    return out


def check_tool_protocol(candidate: dict, example: dict) -> str | None:
    """Tool calls must be licensed by a schema, and must conform to it."""
    calls = _TOOL_CALL.findall(candidate["answer"])
    declared = _declared_tools(example.get("tools"))

    if not declared:
        # No schema was offered, so any tool call is an invented capability.
        return "unexpected_tool_call" if calls else None

    if not calls:
        return None  # a schema does not oblige a call; the task decides
    for payload in calls:
        try:
            call = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return "tool_call_not_json"
        if not isinstance(call, dict):
            return "tool_call_not_object"
        name = call.get("name")
        if name not in declared:
            return "tool_name_undeclared"
        args = call.get("arguments")
        if not isinstance(args, dict):
            return "tool_arguments_not_object"
        _props, required = declared[name]
        missing = [k for k in required if k not in args]
        if missing:
            return "tool_required_argument_missing"
    return None


# --------------------------------------------------------------------------
# stage 4 — completion / termination / degeneration
# --------------------------------------------------------------------------
def check_completion(candidate: dict, degeneration) -> str | None:
    """Complete, naturally terminated and not degenerate.

    `degeneration` is injected so the caller pins which detector version ran;
    the corpus was built before the `rambling` signal existed.
    """
    if candidate.get("hit_cap"):
        return "context_limit_reached"
    if candidate.get("over_budget"):
        return "over_budget"
    if not candidate.get("finished"):
        return "not_finished"
    if candidate.get("length_limited"):
        return "length_limited"
    if IM_END not in candidate["raw"]:
        return "not_terminated"
    verdict = degeneration.check(candidate["tokens"])
    if verdict:
        return f"degenerate:{verdict['kind']}"
    return None


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def screen_candidate(candidate: dict, example: dict, degeneration) -> str | None:
    """Run the four gates in order; return the first failure reason, or None."""
    for check in (
        lambda: check_serialization(candidate),
        lambda: check_correctness(candidate, example),
        lambda: check_tool_protocol(candidate, example),
        lambda: check_completion(candidate, degeneration),
    ):
        reason = check()
        if reason is not None:
            return reason
    return None


def select_clean(
    example: dict,
    degeneration,
    length_of,
    original_index: int | None,
) -> dict:
    """Screen every candidate and choose the replacement deterministically.

    `length_of(candidate) -> int` supplies the length used for the tie-break;
    the driver passes the *supervised-token* length of the rendered session, so
    "shortest" is measured in the unit the training budget is denominated in
    rather than in words.

    Returns a verdict dict: the chosen candidate index (or None), whether the
    corpus's original candidate was retained, and the per-candidate reasons.
    """
    reasons = {}
    survivors = []
    for candidate in example["candidates"]:
        reason = screen_candidate(candidate, example, degeneration)
        reasons[candidate["index"]] = reason or "ok"
        if reason is None:
            survivors.append(candidate)

    if not survivors:
        return {"chosen": None, "retained_original": False, "reasons": reasons,
                "n_survivors": 0}

    # Rule 1: minimise change — keep the corpus's own target when it survives.
    for candidate in survivors:
        if candidate["index"] == original_index:
            return {"chosen": candidate, "retained_original": True,
                    "reasons": reasons, "n_survivors": len(survivors)}

    # Rule 2: shortest complete-and-correct survivor, tie-broken by index.
    chosen = min(survivors, key=lambda c: (length_of(c), c["index"]))
    return {"chosen": chosen, "retained_original": False, "reasons": reasons,
            "n_survivors": len(survivors)}
