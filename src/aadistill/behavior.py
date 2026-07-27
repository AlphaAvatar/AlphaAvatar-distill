"""Mechanical behavior scorers for `eval_behavior_v0`.

Why this module exists: `holdout_v1` (fineweb-edu NLL) is a language-modeling
metric and is nearly blind to the defects actually observed in generation —
chat-format breakage, question restatement instead of answering, ungrounded
answers, and invalid tool calls. These scorers turn those defects into numbers.

Everything here is **mechanical**: string/regex/JSON checks and exact match. No
LLM judge, so a scorecard is free to compute, deterministic, and reproducible
from the raw generations alone. The tradeoff is deliberate — these scorers
measure *form and grounding*, not answer quality. A model can score well here
and still be unhelpful; a low score, however, is real evidence of a defect.

## Generation contract assumed

The Qwen3-Thinking-2507 chat template ends a generation prompt with
``<|im_start|>assistant\\n<think>\\n`` — the think block is opened *by the
prompt*. Stage 2 training data injects an empty ``<think>\\n\\n</think>\\n\\n``
into the final assistant turn on purpose (`aadistill.data` docstring), so the
student is trained to close the think block immediately and answer.

Therefore, for a well-formed generation:

* it contains **exactly one** ``</think>`` and **no** ``<think>``
  (the opener came from the prompt);
* the answer is what follows that ``</think>``;
* it terminates with ``<|im_end|>``.

A generation with no ``</think>`` never left the think block — the observed
`s2_blocks_v1` failure on "What is 2+2?" — and its "answer" is scored as empty.

## Echo credit

Several prompts contain their own answer material: the rag/refusal prompts embed
the gold span *and* the instruction "say you cannot answer from the context". A
model that parrots the prompt would therefore score `evidence_hit` and `refusal`
without answering anything — this is not hypothetical, it is what s1@660 does.
Every content metric consequently ships in two forms: the raw check, and a
``_credited`` variant that additionally requires a non-empty, non-echoed answer
(see ``ECHO_THRESHOLD``). **Compare checkpoints on the credited variants.**
"""

from __future__ import annotations

import json
import re
from collections import Counter

# Groups that carry a behavioral prompt. `long_context` is excluded: it is
# format=="text" fineweb-edu continuation data with no conversation to prompt
# from, so none of the scorers below apply to it.
BEHAVIOR_GROUPS = (
    "instruction",
    "rag_evidence",
    "multihop_qa",
    "tool_calling",
    "refusal_uncertainty",
    "code_math",
    "short_realtime",
)

IM_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Markers that must never appear in a generation body. `<|im_end|>` is handled
# separately (one trailing occurrence is the correct terminator).
STRAY_MARKERS = ("<|im_start|>", "<|endoftext|>", "<tool_response>", THINK_OPEN)

# A generation whose 4-gram overlap with the prompt reaches this is treated as
# copying rather than answering. Load-bearing: the rag/refusal prompts contain
# both the gold span and the string "say you cannot answer from the context",
# so a parroting model scores `evidence_hit` and `refusal` for free. Every
# content metric therefore has a `_credited` variant that requires a non-empty,
# non-echoed answer. Measured on s1@660, where raw refusal was 1.000 purely
# because the model re-emitted the prompt.
ECHO_THRESHOLD = 0.5

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Refusal / "I don't know" detector. Calibrated against the gold answers of the
# `refusal_uncertainty` val split (squad_v2 unanswerable questions): the
# build script asserts recall on those golds, so this pattern is not guesswork.
_REFUSAL = re.compile(
    r"can(?:not|'t| not)\s+(?:answer|be answered|determine|find|tell|say)"
    r"|(?:don't|do not|doesn't|does not|didn't)\s+(?:know|see|contain|say|mention|cover|provide|specify|include|appear)"
    r"|(?:isn't|is not|aren't|are not|wasn't|was not)\s+(?:in|contained|provided|mentioned|stated|specified|available|given|covered|included|answered|there)"
    r"|no (?:information|answer|mention|indication|details?|evidence)"
    r"|not (?:provided|mentioned|stated|specified|available|given|covered|included|possible to)"
    r"|unable to|won't guess|not enough (?:information|context|detail)"
    r"|unanswerable|i'm not sure|i am not sure|there'?s no",
    re.I,
)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace."""
    return " ".join(_WORD.findall(s.lower()))


def split_generation(raw: str) -> dict:
    """Split a raw decoded continuation into its structural parts.

    Returns a dict with:
      ``terminated``      — the generation emitted ``<|im_end|>``;
      ``think_closed``    — exactly one ``</think>`` and no ``<think>``;
      ``think_immediate`` — the generation closes the think block right away
                            (only whitespace before ``</think>``), the behavior
                            the training data teaches;
      ``no_stray_markers``— no template control markers in the body;
      ``think``           — text before ``</think>`` (the model's thinking);
      ``answer``          — text after ``</think>``, terminator stripped;
                            empty when the block was never closed.
    """
    body = raw.split(IM_END)[0]
    terminated = IM_END in raw
    n_close = body.count(THINK_CLOSE)
    n_open = body.count(THINK_OPEN)
    think_closed = n_close == 1 and n_open == 0

    if n_close >= 1:
        think, answer = body.split(THINK_CLOSE, 1)
    else:
        think, answer = body, ""
    think_immediate = think_closed and think.strip() == ""

    stray = [m for m in STRAY_MARKERS if m in body]
    # A second <|im_end|> would mean the model kept talking past its terminator.
    if raw.count(IM_END) > 1:
        stray.append(IM_END)

    return {
        "terminated": terminated,
        "think_closed": think_closed,
        "think_immediate": think_immediate,
        "no_stray_markers": not stray,
        "stray_markers": stray,
        "n_think_close": n_close,
        "think": think,
        "answer": answer.strip(),
    }


def format_ok(parts: dict) -> bool:
    """Chat-format validity: terminated, one closed think block, no stray markers."""
    return bool(
        parts["terminated"] and parts["think_closed"] and parts["no_stray_markers"]
    )


def ngrams(words: list[str], n: int) -> list[tuple]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def echo_rate(answer: str, prompt_text: str, n: int = 4) -> float:
    """Fraction of the answer's word n-grams that also occur in the prompt.

    1.0 means the answer is entirely copied from the prompt — the "restates the
    question instead of answering it" failure. Returns 0.0 for answers shorter
    than n words (too short to judge; `empty_answer` covers those).
    """
    a = ngrams(normalize_text(answer).split(), n)
    if not a:
        return 0.0
    p = set(ngrams(normalize_text(prompt_text).split(), n))
    return sum(g in p for g in a) / len(a)


def repetition_rate(answer: str, n: int = 3) -> float:
    """1 - distinct-n. High values mean degenerate looping text."""
    g = ngrams(normalize_text(answer).split(), n)
    if not g:
        return 0.0
    return 1.0 - len(set(g)) / len(g)


def contains_gold(answer: str, gold: str) -> bool:
    """Normalized containment — the standard extractive-QA hit criterion."""
    g = normalize_text(gold)
    return bool(g) and g in normalize_text(answer)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))


def final_number(text: str) -> str | None:
    """Last number in the text, comma- and trailing-zero-normalized.

    gsm8k targets in mixture v1 end with "The answer is N." (the `####` form was
    normalized away during the v1 rebuild), so the last number is the answer.
    """
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    value = matches[-1].replace(",", "").rstrip(".")
    try:
        f = float(value)
    except ValueError:
        return value
    return str(int(f)) if f == int(f) else str(f)


def parse_tool_calls(answer: str) -> list[dict]:
    """Extract and JSON-parse ``<tool_call>`` blocks. Unparseable → None entry."""
    calls = []
    for blob in _TOOL_CALL.findall(answer):
        try:
            calls.append(json.loads(blob))
        except json.JSONDecodeError:
            calls.append(None)
    return calls


def score_tool_call(answer: str, tools: list[dict], gold_calls: list[dict]) -> dict:
    """Validity of a tool-calling generation against the declared tool schemas.

    Checks, in increasing strictness: a call was emitted, it is valid JSON, the
    function name is one the prompt declared, every ``required`` parameter is
    present, and (strictest) the name and arguments match the gold call exactly.
    """
    calls = parse_tool_calls(answer)
    emitted = bool(calls)
    parsed = emitted and all(c is not None for c in calls)
    by_name = {
        t.get("function", {}).get("name"): t.get("function", {}) for t in tools or []
    }

    name_valid = args_ok = False
    if parsed:
        name_valid = all(c.get("name") in by_name for c in calls)
    if name_valid:
        args_ok = True
        for c in calls:
            args = c.get("arguments")
            if not isinstance(args, dict):
                args_ok = False
                break
            required = (
                by_name[c["name"]].get("parameters", {}).get("required", []) or []
            )
            if any(r not in args for r in required):
                args_ok = False
                break

    exact = False
    if parsed and gold_calls:
        got = [(c.get("name"), json.dumps(c.get("arguments"), sort_keys=True)) for c in calls]
        want = [
            (g["function"]["name"], json.dumps(g["function"]["arguments"], sort_keys=True))
            for g in gold_calls
        ]
        exact = got == want

    return {
        "tool_call_emitted": emitted,
        "tool_call_parsed": parsed,
        "tool_name_valid": name_valid,
        "tool_args_schema_ok": args_ok,
        "tool_call_exact_match": exact,
    }


def score_sample(sample: dict, raw: str, hit_cap: bool = False) -> dict:
    """Score one generation. `sample` is an eval_behavior_v0 prompt-set entry.

    `hit_cap` says the generation was cut off at ``max_new_tokens`` rather than
    stopping on its own. It is recorded separately from `terminated` because the
    two are otherwise indistinguishable in the text and mean different things: a
    cap hit at a *low* cap says the answer was long, while a cap hit at a
    generous cap says the model does not know how to stop. Measured at the
    200-token cap, 100% of `s2_blocks_v1`'s non-terminations were cap hits —
    which is why the cap is now 512 and this metric exists.
    """
    parts = split_generation(raw)
    answer = parts["answer"]
    group = sample["group"]

    scores = {
        "id": sample["id"],
        "group": group,
        "terminated": parts["terminated"],
        "truncated_at_cap": bool(hit_cap),
        "think_closed": parts["think_closed"],
        "think_immediate": parts["think_immediate"],
        "no_stray_markers": parts["no_stray_markers"],
        "stray_markers": parts["stray_markers"],
        "format_ok": format_ok(parts),
        "empty_answer": answer == "",
        "echo_4gram": round(echo_rate(answer, sample["prompt_text"]), 4),
        "rep_3gram": round(repetition_rate(answer), 4),
        "answer_words": len(normalize_text(answer).split()),
    }

    # A copied or empty answer cannot be credited with content it merely
    # inherited from the prompt (see ECHO_THRESHOLD).
    echoed = scores["empty_answer"] or scores["echo_4gram"] >= ECHO_THRESHOLD
    scores["answer_is_echo"] = not scores["empty_answer"] and scores["echo_4gram"] >= ECHO_THRESHOLD

    def credit(hit: bool) -> bool:
        return bool(hit and not echoed)

    if group in ("rag_evidence", "multihop_qa"):
        gold = sample.get("gold_answer", "")
        hit = contains_gold(answer, gold)
        scores["evidence_hit"] = hit
        scores["evidence_hit_credited"] = credit(hit)
    if group == "refusal_uncertainty":
        refused = is_refusal(answer)
        scores["refusal"] = refused
        scores["refusal_credited"] = credit(refused)
    if group == "tool_calling":
        # Structural checks — a JSON call inside <tool_call> tags cannot be
        # produced by copying these prompts, so no echo adjustment is needed.
        scores.update(score_tool_call(answer, sample.get("tools"), sample.get("gold_tool_calls")))
    if sample.get("gsm8k_answer") is not None:
        got = final_number(answer)
        em = got is not None and got == sample["gsm8k_answer"]
        scores["answer_em"] = em
        scores["answer_em_credited"] = credit(em)

    return scores


# Metrics where a HIGHER value is worse — recorded so the scorecard and any
# later comparison can orient deltas without a hardcoded per-metric table.
LOWER_IS_BETTER = ("echo_4gram", "rep_3gram", "empty_answer", "answer_is_echo")


# The six axes of `behavior_score` — the project's headline number while real
# test suites are still out of reach for a student this damaged. Each is a rate
# in [0, 1], higher is better, and each is credited (a copied or empty answer
# earns nothing), so the score cannot be gamed by saying less.
#
# `fluency` deliberately replaces a naive `1 - rep_3gram`: on its own that term
# *rewards* silence, and s1@660 — which emits nothing on 61% of prompts — would
# have ranked second on it. Scoring an empty or echoed answer as 0 removes that.
BEHAVIOR_SCORE_AXES = ("format_ok", "fluency", "grounding", "refusal",
                       "tool_call", "math")


def behavior_score(scored: list[dict]) -> dict:
    """Composite behavior score in [0, 1] from `score_sample` rows.

    Returns the per-axis rates, the sample count behind each, and their
    unweighted mean. Unweighted because there is no evidence yet for any other
    weighting; the axes are listed in the README so the number can be taken
    apart. An axis with no samples in this prompt set is skipped and reported
    with n=0 rather than counted as a zero.
    """
    n = len(scored)
    if not n:
        raise ValueError("behavior_score needs at least one scored sample")

    def credited(groups: tuple[str, ...], key: str) -> tuple[float | None, int]:
        rows = [r for r in scored if r["group"] in groups and key in r]
        if not rows:
            return None, 0
        return sum(bool(r[key]) for r in rows) / len(rows), len(rows)

    # An answer that is empty or echoed scores 0; otherwise its non-repetition.
    fluency = sum(0.0 if (r["empty_answer"] or r["answer_is_echo"])
                  else 1.0 - r["rep_3gram"] for r in scored) / n

    axes = {
        "format_ok": (sum(float(r["format_ok"]) for r in scored) / n, n),
        "fluency": (fluency, n),
        "grounding": credited(("rag_evidence", "multihop_qa"), "evidence_hit_credited"),
        "refusal": credited(("refusal_uncertainty",), "refusal_credited"),
        "tool_call": credited(("tool_calling",), "tool_call_parsed"),
        "math": credited(("code_math",), "answer_em_credited"),
    }
    present = [v for v, _ in axes.values() if v is not None]
    return {
        "score": round(sum(present) / len(present), 4),
        "axes": {k: (None if v is None else round(v, 4)) for k, (v, _) in axes.items()},
        "n": {k: count for k, (_, count) in axes.items()},
    }


def aggregate(scored: list[dict]) -> dict:
    """Per-group and overall means of every boolean/numeric metric."""

    def means(rows: list[dict]) -> dict:
        keys = [
            k
            for k in rows[0]
            if k not in ("id", "group", "stray_markers")
        ]
        out = {"n": len(rows)}
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if not vals:
                continue
            out[k] = round(sum(float(v) for v in vals) / len(vals), 4)
        return out

    by_group = {}
    for g in sorted({r["group"] for r in scored}):
        rows = [r for r in scored if r["group"] == g]
        by_group[g] = means(rows)

    # Overall: only the metrics defined for every sample, so the number is not
    # skewed by which groups happen to define a group-specific metric.
    universal = [
        "terminated",
        "truncated_at_cap",
        "think_closed",
        "think_immediate",
        "no_stray_markers",
        "format_ok",
        "empty_answer",
        "answer_is_echo",
        "echo_4gram",
        "rep_3gram",
        "answer_words",
    ]
    overall = {"n": len(scored)}
    for k in universal:
        overall[k] = round(sum(float(r[k]) for r in scored) / len(scored), 4)

    return {
        "overall": overall,
        "by_group": by_group,
        "stray_marker_counts": dict(
            Counter(m for r in scored for m in r["stray_markers"])
        ),
    }
