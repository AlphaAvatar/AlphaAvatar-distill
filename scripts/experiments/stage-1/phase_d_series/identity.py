#!/usr/bin/env python3
"""The D-series identity layer: ONE new concept, plus the ids that carry it.

**The concept is canonical problem-content identity.** A source row's *problem*,
normalized and hashed, independent of how it is rendered, which split it came
from, and what native key the upstream release gave it.

It exists because the historical chain cannot see a restated problem, and the
parity round proved that twice over:

* MBPP `full/train` task 602 is the same problem as the consumed `full/test` task
  217 — different `task_id`, different test asserts, so both of the historical
  chain's keys differ and it passes;
* the chain hashes each recovery-training session's non-assistant messages
  JOINED, and every session carries a system message, so that hash can never
  equal a bare question. It catches **0 of the 1,708** GSM8K train rows that are
  literally in the training corpus.

**This SUPPLEMENTS and does not replace anything.** Three coordinates, kept
separate on purpose:

```text
native provenance   the upstream key, preserved verbatim   task_id, config, row index
item identity       split-aware, D-series-owned            mbpp-full-train-602
problem content     sha256(norm(problem payload))          the semantic coordinate
```

Conflating the last two is the mistake that made `gsm8k-test-00000` name a
different problem in two files. A provenance coordinate answers *where this came
from*; a content coordinate answers *whether we have seen this problem*. One
cannot do both jobs.

**It is NOT a semantic-dedup framework.** The comparison is exact: one hash,
equal or not. Fuzzy similarity stays a pre-freeze review surface with no
automatic cutoff, and the decisions from that review are frozen here as data
(:data:`SEMANTIC_DUPLICATE_EXCLUSIONS`) rather than recomputed from a threshold.

**The historical chain is not modified.** `build_c1_confirmation_battery`'s
`excluded_identities` keeps its semantics exactly, because the frozen C1/C2/C3
batteries' membership is reproducible only from it. This layer runs on top.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO_ROOT / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from battery_render import norm  # noqa: E402

#: The problem payload, per source. The ONE place a source-specific field name
#: appears: everything downstream takes the hash and never the field.
PROBLEM_FIELD: dict[str, str] = {
    "code": "text",            # MBPP's bare problem statement
    "gsm8k": "question",       # the word problem, without the worked answer
    "math_verified": "problem",  # the problem statement, without the solution
}


def problem_content_id(group: str, row: dict[str, Any]) -> str:
    """`sha256(norm(<the problem>))` — the semantic coordinate.

    Normalization is `battery_render.norm` (whitespace-collapse, lower-case), the
    same function the historical chain normalizes with. Using a second
    normalizer would make two rows "the same problem" under one comparison and
    not the other.
    """
    field = PROBLEM_FIELD.get(group)
    if field is None:
        raise KeyError(
            f"no problem payload declared for {group!r}; add it to "
            "PROBLEM_FIELD rather than hashing a guessed field")
    payload = row.get(field)
    if payload is None or not str(payload).strip():
        raise ValueError(
            f"{group}: row carries no {field!r} to identify. A row with no "
            "problem text cannot be isolated by content and must not be "
            "admitted as though it had been")
    return hashlib.sha256(norm(str(payload)).encode()).hexdigest()


# --- split-aware provenance ids ---------------------------------------------

#: How a D-series item id is built, per source. Deliberately NOT the historical
#: renderer's id: `mbpp-test-<task_id>` names the wrong split for three of the
#: four MBPP files, and `gsm8k-test-<row index>` names a different problem in
#: every file it is applied to.
#: The id prefix and native-key field per stratum. ALL SEVEN are declared, not
#: only the three the source decision extended: the builder draws every stratum
#: and a scheme missing for one of them is a crash at best and an inconsistent
#: ranking at worst. `None` means the source has no native key and position is
#: the only provenance available.
ID_SCHEME: dict[str, tuple[str, str | None]] = {
    "code": ("mbpp", "task_id"),        # MBPP's own key, partitions across splits
    "gsm8k": ("gsm8k", None),           # no native key; position only
    "math_verified": ("math", None),    # no native key; position only
    "knowledge": ("trivia", "question_id"),
    "multihop": ("hotpot", "id"),
    "rag": ("squad", "id"),
    "tool": ("xlam", "id"),
}


def d_series_item_id(group: str, config: str, split: str,
                     row: dict[str, Any], *, index: int | None = None) -> str:
    """A provenance id that says which FILE the row came from.

    `<prefix>-<config>-<split>-<native key or row index>`. The config and split
    are always present, which is the whole point: a historical id that omits them
    cannot distinguish two files of one repository, and `make_gsm8k`'s positional
    id actively names a different problem in each.

    Historical ids are untouched. This builds ids for D-series rows, and nothing
    here renames an item in a frozen battery.
    """
    scheme = ID_SCHEME.get(group)
    if scheme is None:
        raise KeyError(
            f"no D-series id scheme declared for {group!r}; add it to ID_SCHEME. "
            "Every stratum the builder draws needs one -- a missing scheme is an "
            "inconsistent ranking, not a missing convenience.")
    prefix, key_field = scheme
    if key_field is not None:
        native = row.get(key_field)
        if native is None or str(native) == "":
            raise ValueError(
                f"{group}: row carries no {key_field!r}, but the declared scheme "
                "says it has a native key. Falling back to position would make "
                "two rows' ids depend on file order.")
        return f"{prefix}-{config}-{split}-{native}"
    if index is None:
        raise ValueError(
            f"{group} rows have no native key, so a row index is required; "
            "without it the id cannot distinguish two files' row 0")
    return f"{prefix}-{config}-{split}-{index:05d}"


# --- the frozen duplicate-review decision ------------------------------------

#: MAINTAINER DECISION, 2026-10-03, FROZEN BEFORE ANY D1 OUTCOME EXISTS.
#:
#: All 44 pairs of the committed `>= 0.8` bare-problem review list were reviewed
#: individually. One is an exact duplicate and is excluded; the other 43 are
#: distinct task semantics despite high lexical overlap -- min vs max, even vs
#: odd, area vs perimeter, first vs last, sum vs product, and similar.
#:
#: Recorded as DATA, not as a threshold. A Jaccard cutoff would have excluded all
#: 44 and thrown away 43 usable problems; the review is what decides, and freezing
#: the decision here before any outcome exists is what keeps it from being
#: informed by a result.
SEMANTIC_DUPLICATE_EXCLUSIONS: dict[str, tuple[dict[str, Any], ...]] = {
    "code": (
        {"d_series_item_id": "mbpp-full-train-602",
         "native_task_id": 602,
         "duplicate_of_consumed_task_id": 217,
         "jaccard": 1.0,
         "reason": "exact same problem text as a consumed full/test row"},
    ),
    "gsm8k": (),
    "math_verified": (),
}

#: What the review covered, so a later reader knows the 43 retentions were
#: decided rather than defaulted.
REVIEW_PROVENANCE = {
    "decided_utc": "2026-10-03",
    "reviewed_pairs": 44,
    "excluded": 1,
    "retained": 43,
    "retained_reason": ("distinct task semantics despite high lexical overlap: "
                        "min vs max, even vs odd, area vs perimeter, first vs "
                        "last, sum vs product and similar"),
    "trigger_threshold": 0.8,
    "_the_threshold_decided_nothing": (
        "it selected which 44 pairs a human read. Excluding on it would have "
        "dropped 43 usable problems."),
    "frozen_before_any_d1_outcome": True,
}


def excluded_by_review(group: str) -> set[str]:
    """D-series item ids the frozen review rules out."""
    return {str(d["d_series_item_id"])
            for d in SEMANTIC_DUPLICATE_EXCLUSIONS.get(group, ())}


def review_decision(group: str) -> dict[str, Any]:
    """The frozen decision for one stratum, for the evidence record."""
    return {
        "provenance": REVIEW_PROVENANCE,
        "excluded": [dict(d) for d in SEMANTIC_DUPLICATE_EXCLUSIONS.get(group, ())],
        "n_excluded": len(SEMANTIC_DUPLICATE_EXCLUSIONS.get(group, ())),
    }

#: Whether a source's RENDERED prompt is the bare problem payload itself.
#:
#: Where it is, a reserved item's problem content is recoverable straight from a
#: committed pool's `prompt_text` -- no native key needed. Where it is not, the
#: rendering wraps the problem (MBPP prepends a fixed instruction and appends the
#: test list) and the problem must be recovered through the source file.
#:
#: Declared rather than sniffed, and VERIFIED by a test that renders real rows
#: and compares. The first version of the strengthened contract recovered
#: problem content only through a native key, so for GSM8K -- which has no native
#: key -- it reported 430 reserved problems as unrecoverable while their text was
#: sitting in `prompt_text` all along. An unrecoverable reserved problem is a
#: hole in the isolation, so over-reporting one is not a safe error.
RENDERED_PROMPT_IS_THE_PROBLEM: dict[str, bool] = {
    "code": False,          # instruction + problem + test list
    "gsm8k": True,          # prompt_text IS the question
    "math_verified": True,  # prompt_text IS the problem
}


def problem_content_id_from_prompt(group: str, prompt_text: str) -> str:
    """The problem-content id of a reserved item, from its rendered prompt.

    Only valid where :data:`RENDERED_PROMPT_IS_THE_PROBLEM` is true -- refused
    otherwise rather than returning a hash of the wrapper, which would silently
    make every wrapped item look like a distinct problem.
    """
    if not RENDERED_PROMPT_IS_THE_PROBLEM.get(group):
        raise ValueError(
            f"{group}: the rendered prompt wraps the problem, so hashing it "
            "would not be a problem-content id. Recover the problem through "
            "the source file instead.")
    if not str(prompt_text).strip():
        raise ValueError(f"{group}: empty prompt text has no problem content")
    return hashlib.sha256(norm(str(prompt_text)).encode()).hexdigest()
