"""Isolation-key checks between data roles (calibration, training, evaluation).

A claim that an evaluation measures something UNSEEN is a claim about keys: no
evaluation item may share a declared key with any item of a role it must be
isolated from. This module computes those overlaps for arbitrary roles and key
extractors and refuses a declared requirement that the data violates. It names
no experiment, no dataset and no key: callers supply all three.

    report = key_overlaps({"cal": cal_items, "dev": dev_items},
                          {"family": lambda r: r["family_id"]})
    require_disjoint(report, [("cal", "dev", "family")])
"""
from __future__ import annotations

import itertools
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
