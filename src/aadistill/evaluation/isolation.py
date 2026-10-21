"""Isolation-key checks between data roles (calibration, training, evaluation).

A claim that an evaluation measures something UNSEEN is a claim about keys: no
evaluation item may share a declared key with any item of a role it must be
isolated from. This module computes those overlaps for arbitrary roles and key
extractors and refuses a declared requirement that the data violates. It names
no experiment, no dataset and no key: callers supply all three.

    report = key_overlaps({"cal": cal_items, "dev": dev_items},
                          {"family": lambda r: r["family_id"]})
    require_disjoint(report, [("cal", "dev", "family")])

Exact keys miss a reworded copy, so ``ngram_containment`` screens free text
against an n-gram index of the protected items; ``n`` and the threshold are the
caller's protocol.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

KeyFn = Callable[[Mapping], Hashable | None]


class IsolationViolation(RuntimeError):
    """A role pair shares a key it was required not to share."""


def key_sets(items: Iterable[Mapping], keys: Mapping[str, KeyFn]) -> dict[str, set]:
    out: dict[str, set] = {k: set() for k in keys}
    for item in items:
        for name, fn in keys.items():
            value = fn(item)
            if value is not None:
                out[name].add(value)
    return out


def key_overlaps(roles: Mapping[str, Iterable[Mapping]],
                 keys: Mapping[str, KeyFn], *, examples: int = 5) -> dict:
    """Per role pair and key: how many key values the two roles share."""
    sets = {role: key_sets(items, keys) for role, items in roles.items()}
    pairs = []
    for a, b in itertools.combinations(sorted(sets), 2):
        row = {"a": a, "b": b, "overlap": {}, "examples": {}}
        for k in keys:
            shared = sets[a][k] & sets[b][k]
            row["overlap"][k] = len(shared)
            if shared:
                row["examples"][k] = sorted(map(str, shared))[:examples]
        pairs.append(row)
    return {"roles": {r: {k: len(v) for k, v in s.items()} for r, s in sets.items()},
            "keys": list(keys), "pairs": pairs}


def ngrams(text: str, n: int) -> set[str]:
    """Word n-grams of case- and whitespace-normalized text (hashed for size).

    A text shorter than ``n`` words contributes its whole normalized form, so a
    short prompt can still collide with an identical one.
    """
    words = text.casefold().split()
    if not words:
        return set()
    if len(words) < n:
        grams = [" ".join(words)]
    else:
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return {hashlib.blake2b(g.encode(), digest_size=8).hexdigest() for g in grams}


def ngram_index(texts: Iterable[str], n: int, *,
                max_document_frequency: int | None = None) -> set[str]:
    """The n-grams of a reference set.

    ``max_document_frequency`` drops every n-gram found in more than that many
    reference texts: an instruction template repeated across a whole unit is
    boilerplate, and counting it would report every prompt that uses the same
    template as "sharing" text with the evaluation set.
    """
    if max_document_frequency is None:
        out: set[str] = set()
        for t in texts:
            out |= ngrams(t, n)
        return out
    df: Counter = Counter()
    for t in texts:
        df.update(ngrams(t, n))
    return {g for g, c in df.items() if c <= max_document_frequency}


def ngram_containment(candidates: Iterable[tuple[str, str]], index: set[str], n: int,
                      *, threshold: float) -> dict:
    """Near-duplicate screen of candidate texts against a reference n-gram index.

    Exact key checks miss a reworded or re-formatted copy of an evaluation item;
    n-gram containment -- the share of a candidate's n-grams found in the
    reference -- catches it. ``candidates`` are ``(id, text)`` pairs; a
    candidate at or above ``threshold`` is flagged. ``n`` and ``threshold`` are
    the caller's declared protocol, recorded with the result.
    """
    flagged, any_shared, checked = [], 0, 0
    for cid, text in candidates:
        grams = ngrams(text, n)
        if not grams:
            continue
        checked += 1
        shared = len(grams & index)
        if shared:
            any_shared += 1
        frac = shared / len(grams)
        if frac >= threshold:
            flagged.append({"id": cid, "containment": round(frac, 4), "shared": shared})
    return {"n": n, "threshold": threshold, "checked": checked,
            "with_any_shared_ngram": any_shared, "flagged": len(flagged),
            "flagged_items": sorted(flagged, key=lambda r: (-r["containment"], r["id"]))}


def require_disjoint(report: Mapping, required: Sequence[tuple[str, str, str]]) -> list[dict]:
    """Raise unless every (role_a, role_b, key) in `required` has zero overlap."""
    index = {}
    for row in report["pairs"]:
        index[(row["a"], row["b"])] = row
        index[(row["b"], row["a"])] = row
    checked, bad = [], []
    for a, b, k in required:
        row = index.get((a, b))
        if row is None or k not in row["overlap"]:
            raise IsolationViolation(f"no overlap computed for {a} x {b} on {k!r}")
        entry = {"a": a, "b": b, "key": k, "overlap": row["overlap"][k]}
        checked.append(entry)
        if row["overlap"][k]:
            bad.append({**entry, "examples": row["examples"].get(k, [])})
    if bad:
        raise IsolationViolation(f"isolation requirements violated: {bad}")
    return checked
