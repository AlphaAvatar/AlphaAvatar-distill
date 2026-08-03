"""Deterministic scorers for the frozen Experiment 2 capability battery.

Every scorer here is mechanical and re-derivable from a stored generation: exact
match against an alias set, symbolic equivalence, span containment, supporting-
title recall, protocol validation. **No LLM judge is used as a primary scorer**,
because each of these capabilities has a deterministic key and a judge would make
the battery expensive, non-reproducible and impossible to re-run offline against
a historical control.

Every scorer shares two rules with `strict_answer`:

* a **protocol-invalid or degenerate generation is incorrect**, whatever it
  contains — otherwise degeneration can raise a score;
* **termination and correctness are reported separately**, so "the model learned
  to stop" is never read as "the model learned the task".

`BATTERY_VERSION` is recorded in the manifest and in every result file. Changing
a rule requires bumping it: two runs scored under different rules are not
comparable, and the whole point of freezing the battery before D1 trains is that
nothing here moves after results are seen.
"""

from __future__ import annotations

import re

from ..data.verify import boxed_answer, normalize_math
from .behavior import ECHO_THRESHOLD, is_refusal, normalize_text, split_generation
from .strict_answer import extract_final_answer, normalize_number, protocol_valid

BATTERY_VERSION = "capability-v1"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles."""
    out = _PUNCT.sub(" ", (text or "").lower())
    out = _ARTICLES.sub(" ", out)
    return " ".join(out.split())


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1, the standard extractive-QA partial-credit measure."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = 0
    pool = list(g)
    for tok in p:
        if tok in pool:
            pool.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def _shared(record: dict) -> dict:
    """Protocol, termination and degeneration facts every scorer reports."""
    raw = record.get("raw", "")
    parts = split_generation(raw)
    valid, reason = protocol_valid(raw)
    return {
        "answer": parts["answer"],
        "protocol_valid": valid,
        "protocol_reason": reason,
        "natural_termination": bool(record.get("natural_termination")),
        "degenerate": bool(record.get("degeneration_triggered")),
        "degeneration_kind": record.get("degeneration_kind"),
    }


def _blocked(base: dict) -> str | None:
    """Reason this generation cannot score, independent of its content."""
    if not base["protocol_valid"]:
        return f"protocol:{base['protocol_reason']}"
    if base["degenerate"]:
        return f"degenerate:{base['degeneration_kind'] or 'unknown'}"
    return None


# --------------------------------------------------------------------------
# knowledge — closed-book factual QA against an alias set
# --------------------------------------------------------------------------
def score_knowledge(record: dict, sample: dict) -> dict:
    """Exact match against the gold answer's normalized alias set.

    Alias matching, not free-text similarity: TriviaQA ships the aliases, so
    "David Seville" and "Seville, David" are one answer and neither needs a
    judge. Containment is allowed because a thinking model states the answer in
    a sentence, but the alias must appear as a whole token span.
    """
    base = _shared(record)
    aliases = {normalize_answer(a) for a in sample["aliases"] if a and a.strip()}
    aliases.discard("")
    got = normalize_answer(base["answer"])
    hit = any(a == got or f" {a} " in f" {got} " for a in aliases)
    blocked = _blocked(base)
    return {**base, "correct": bool(hit and not blocked),
            "answer_matches_ignoring_protocol": bool(hit),
            "reason": blocked or ("ok" if hit else "answer_mismatch")}


# --------------------------------------------------------------------------
# math — symbolic verification of a boxed answer
# --------------------------------------------------------------------------
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def _delatex(text: str) -> str:
    """Enough LaTeX to hand a math answer to `sympify`.

    Deliberately not `sympy.parsing.latex.parse_latex`: that requires the
    optional `antlr4` runtime, which is not installed here, and a scorer that
    silently degrades when an optional dependency is missing is worse than one
    that never depended on it. This covers the forms OpenMathInstruct-2 actually
    emits — fractions, roots, pi, explicit multiplication — and anything it
    cannot convert falls through to normalized-string comparison rather than
    being guessed at.
    """
    out = text.strip().strip("$")
    for _ in range(3):  # nested \frac{\frac{}{}}{}
        out = _FRAC.sub(r"((\1)/(\2))", out)
    out = out.replace(r"\left", "").replace(r"\right", "")
    out = out.replace(r"\cdot", "*").replace(r"\times", "*")
    out = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", out)
    out = out.replace(r"\pi", "pi").replace("^", "**")
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    return out.replace("{", "(").replace("}", ")").replace(",", "")


def _rational_equal(a: str, b: str) -> bool | None:
    """Fraction comparison, so `0.5` and `1/2` are one answer."""
    from fractions import Fraction

    def parse(v):
        v = _delatex(v).replace("(", "").replace(")", "").replace(" ", "")
        try:
            return Fraction(v)
        except (ValueError, ZeroDivisionError):
            return None

    fa, fb = parse(a), parse(b)
    if fa is None or fb is None:
        return None
    return fa == fb


def _sympy_equal(a: str, b: str) -> bool | None:
    """True/False if sympy can decide, None if it cannot parse either side."""
    try:
        from sympy import simplify, sympify
    except Exception:
        return None
    try:
        pa, pb = sympify(_delatex(a)), sympify(_delatex(b))
    except Exception:
        return None
    try:
        return bool(simplify(pa - pb) == 0)
    except Exception:
        return None


def score_math_verified(record: dict, sample: dict) -> dict:
    """Boxed answer verified numerically, then symbolically, then normalized.

    Three ladders rather than string equality, because `\\frac{1}{2}`, `0.5` and
    `\\dfrac{1}{2}` are the same answer and a string comparison would score two
    of them wrong. The path that decided is recorded so a disagreement between
    ladders is visible instead of silent.
    """
    base = _shared(record)
    got = boxed_answer(base["answer"])
    want = sample["boxed"]
    path, hit = "no_boxed", False
    if got is not None:
        gn, wn = normalize_number(got), normalize_number(want)
        rational = _rational_equal(got, want)
        if gn is not None and wn is not None:
            path, hit = "numeric", gn == wn
        elif rational is not None:
            path, hit = "rational", rational
        else:
            sym = _sympy_equal(got, want)
            if sym is not None:
                path, hit = "symbolic", sym
            else:
                path = "normalized_string"
                hit = normalize_math(got) == normalize_math(want)
    blocked = _blocked(base)
    return {**base, "correct": bool(hit and not blocked),
            "answer_matches_ignoring_protocol": bool(hit),
            "extracted": got, "gold": want, "verification_path": path,
            "reason": blocked or ("ok" if hit else
                                  "no_boxed" if got is None else "answer_mismatch")}


# --------------------------------------------------------------------------
# multihop — final answer AND supporting evidence
# --------------------------------------------------------------------------
def score_multihop(record: dict, sample: dict) -> dict:
    """Answer correctness and supporting-title recall, scored separately.

    Both are needed: a multihop answer that is right without naming the
    documents it came from has not demonstrated the hop, and naming them without
    the right answer has not demonstrated the reasoning. They are never averaged
    into one number here.
    """
    base = _shared(record)
    answer, gold = base["answer"], sample["answer"]
    norm = normalize_answer(answer)
    gnorm = normalize_answer(gold)
    if gnorm in ("yes", "no"):
        # Containment is trivially satisfiable for yes/no, so require it to lead.
        first = norm.split()[0] if norm.split() else ""
        hit = first == gnorm
    else:
        hit = bool(gnorm) and (gnorm == norm or f" {gnorm} " in f" {norm} ")
    titles = [normalize_answer(t) for t in sample["supporting_titles"]]
    cited = [t for t in titles if t and f" {t} " in f" {norm} "]
    blocked = _blocked(base)
    return {**base, "correct": bool(hit and not blocked),
            "answer_matches_ignoring_protocol": bool(hit),
            "f1": token_f1(answer, gold),
            "supporting_titles": len(titles),
            "supporting_titles_cited": len(cited),
            "evidence_recall": (len(cited) / len(titles)) if titles else None,
            "reason": blocked or ("ok" if hit else "answer_mismatch")}


# --------------------------------------------------------------------------
# rag — answer, attribution, unsupported claims, protocol
# --------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?])\s+")


def score_rag(record: dict, sample: dict) -> dict:
    """Four separate axes; none of them is allowed to stand in for another.

    * `correct` — the gold span appears in the answer;
    * `evidence_attributed` — the answer quotes a span that is really in the
      supplied context, i.e. it points at its evidence rather than asserting;
    * `unsupported_claim_rate` — the fraction of the answer's content sentences
      whose content words are not all present in the context, which is what
      "made something up" looks like without a judge;
    * `echoed` — the answer is a copy of the prompt, which would otherwise score
      well on all three.
    """
    base = _shared(record)
    answer, gold, context = base["answer"], sample["gold"], sample["context"]
    ctx = normalize_answer(context)
    ctx_words = set(ctx.split())
    norm = normalize_answer(answer)

    hit = bool(normalize_answer(gold)) and normalize_answer(gold) in norm
    attributed = bool(normalize_answer(gold)) and normalize_answer(gold) in ctx \
        and normalize_answer(gold) in norm

    sentences = [s for s in _SENT.split(answer) if normalize_answer(s).split()]
    unsupported = 0
    for s in sentences:
        words = [w for w in normalize_answer(s).split() if len(w) > 3]
        if words and sum(w in ctx_words for w in words) / len(words) < 0.5:
            unsupported += 1
    # Echo check: an answer that is mostly the prompt scores nothing. The
    # reference is everything the model was shown — instruction *and* context —
    # not the instruction alone, or copying the passage back would pass.
    prompt_words = set(
        normalize_answer(f"{sample.get('prompt_text', '')} {context}").split())
    ans_words = norm.split()
    echo = (sum(w in prompt_words for w in ans_words) / len(ans_words)
            if ans_words else 0.0)
    echoed = echo >= ECHO_THRESHOLD and len(ans_words) > 30

    blocked = _blocked(base)
    return {**base, "correct": bool(hit and not blocked and not echoed),
            "answer_matches_ignoring_protocol": bool(hit),
            "evidence_attributed": bool(attributed and not blocked),
            "unsupported_claims": unsupported,
            "n_sentences": len(sentences),
            "unsupported_claim_rate": (unsupported / len(sentences)
                                       if sentences else None),
            "echoed": bool(echoed), "echo_overlap": round(echo, 4),
            "f1": token_f1(answer, gold),
            "reason": blocked or ("echoed" if echoed else
                                  "ok" if hit else "answer_mismatch")}


# --------------------------------------------------------------------------
# refusal — paired safe / unsafe
# --------------------------------------------------------------------------
def score_refusal_paired(record: dict, sample: dict) -> dict:
    """Score one half of a pair: answerable must answer, unanswerable must refuse.

    The pairing is the point. A model that refuses everything scores 1.0 on the
    unanswerable half and 0.0 on the answerable half, so `pair_correct` — both
    halves right — cannot be won by unconditional refusal. Aggregate on
    `pair_correct`, never on the unanswerable half alone.
    """
    base = _shared(record)
    answer = base["answer"]
    refused = is_refusal(answer)
    blocked = _blocked(base)
    if sample["answerable"]:
        gold = normalize_answer(sample["gold"])
        hit = bool(gold) and gold in normalize_answer(answer)
        correct = bool(hit and not refused and not blocked)
        reason = blocked or ("refused_answerable" if refused else
                             "ok" if hit else "answer_mismatch")
    else:
        correct = bool(refused and not blocked)
        reason = blocked or ("ok" if refused else "answered_unanswerable")
    return {**base, "correct": correct, "refused": bool(refused),
            "answerable": bool(sample["answerable"]),
            "reason": reason}


SCORERS = {
    "knowledge": score_knowledge,
    "math_verified": score_math_verified,
    "multihop": score_multihop,
    "rag": score_rag,
    "refusal_paired": score_refusal_paired,
}
