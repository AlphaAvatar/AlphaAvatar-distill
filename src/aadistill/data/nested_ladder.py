"""Nested data ladders cut from one master corpus, composition held at every rung.

A data-scaling claim compares rungs that differ in QUANTITY and nothing else.
Two things make that hard:

* the rungs must be nested -- ``D_1 ⊂ D_2 ⊂ … ⊂ D_k`` -- so a larger rung adds
  data rather than swapping it, and
* every rung must carry the same composition (the declared share of each
  stratum) and as much of the corpus's family structure as its size allows, so
  quantity is not confounded with mixture or with family coverage.

Both follow from cutting every rung as a PREFIX of one deterministic order:

1. inside a stratum, units are ordered family by family in a salted
   round-robin, so a small prefix already spans as many families as it can
   (the coverage a rung reaches is reported, never assumed);
2. strata are interleaved by largest remainder on the ladder's size measure, so
   every prefix tracks the declared shares (``mixture.interleave``'s rule, on
   opaque units);
3. a rung is the shortest prefix whose size reaches its target.

The size measure is supplied by the caller (for a recovery corpus: unique
supervised assistant tokens). Every other quantity a rung should report --
reasoning tokens, answer tokens, rendered tokens -- rides along in
``LadderUnit.counts`` and is summed per rung; it is never used to cut.

Packing is a separate step (``sessions.pack_best_fit``). Packing each rung's
INCREMENT ``D_k \\ D_{k-1}`` on its own and concatenating keeps the packed
blocks nested too, so a rung is still ``blocks[:n_k]`` for the trainer.

Nothing here names a dataset, a stratum, a family scheme or a rung size; the
caller declares all of them.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

LADDER_SCHEMA = "aadistill.data.nested_ladder/v1"


class LadderError(ValueError):
    """A ladder request the corpus cannot satisfy as declared."""


@dataclass(frozen=True)
class LadderUnit:
    """One indivisible piece of training data (a session, an episode, a task).

    ``size`` is the quantity rungs are cut on. ``stratum`` is the composition
    cell the declared shares refer to; ``family`` the coverage key that the
    within-stratum round-robin spreads over (a rule family, a source family).
    ``counts`` are extra additive quantities reported per rung, and
    ``attributes`` categorical ones whose distinct values a rung reports
    (generator, difficulty, ...).
    """

    unit_id: str
    stratum: str
    size: int
    family: str | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    attributes: Mapping[str, str] = field(default_factory=dict)


def _rank(salt: str, *parts: str) -> str:
    return hashlib.sha256("|".join((salt, *parts)).encode()).hexdigest()


def family_round_robin(units: Sequence[LadderUnit], *, salt: str) -> list[LadderUnit]:
    """Order one stratum's units so every prefix spans the most families.

    Families are visited in salted-hash order, one unit per family per round;
    inside a family units are in salted-hash order of their id. A unit with no
    family is its own family. Deterministic and independent of input order.
    """
    by_family: dict[str, list[LadderUnit]] = defaultdict(list)
    for u in units:
        by_family[u.family if u.family is not None else f"\0{u.unit_id}"].append(u)
    queues = [sorted(v, key=lambda u: _rank(salt, "unit", u.unit_id))
              for _, v in sorted(by_family.items(), key=lambda kv: _rank(salt, "family", kv[0]))]
    out: list[LadderUnit] = []
    depth = 0
    while True:
        layer = [q[depth] for q in queues if depth < len(q)]
        if not layer:
            return out
        out.extend(layer)
        depth += 1


def nested_order(units: Iterable[LadderUnit], shares: Mapping[str, float], *,
                 salt: str) -> list[LadderUnit]:
    """The master order every rung is a prefix of.

    ``shares`` declares each stratum's target share of the size measure; a
    stratum absent from ``shares`` (or at share 0) is excluded. Interleaving is
    largest remainder: the next unit comes from the stratum furthest behind its
    share so far. A stratum that runs out stops contributing, and from that
    point the realized shares drift; ``composition_capacity`` says where.
    """
    units = list(units)
    seen: set[str] = set()
    for u in units:
        if u.unit_id in seen:
            raise LadderError(f"duplicate unit id {u.unit_id!r}")
        seen.add(u.unit_id)
        if u.size <= 0:
            raise LadderError(f"unit {u.unit_id!r} has non-positive size {u.size}")
    weights = {s: float(w) for s, w in shares.items() if float(w) > 0}
    if not weights:
        raise LadderError("declared shares are empty or all zero")
    scale = sum(weights.values())
    weights = {s: w / scale for s, w in weights.items()}
    by_stratum: dict[str, list[LadderUnit]] = defaultdict(list)
    for u in units:
        if u.stratum in weights:
            by_stratum[u.stratum].append(u)
    missing = sorted(s for s in weights if not by_stratum.get(s))
    if missing:
        raise LadderError(f"declared strata with no units: {missing}")
    queues = {s: family_round_robin(v, salt=salt) for s, v in by_stratum.items()}
    cursor = {s: 0 for s in queues}
    emitted = {s: 0 for s in queues}
    total = 0
    order: list[LadderUnit] = []
    while True:
        best, best_deficit = None, None
        for s in sorted(queues):
            if cursor[s] >= len(queues[s]):
                continue
            deficit = weights[s] * total - emitted[s]
            if best is None or deficit > best_deficit:
                best, best_deficit = s, deficit
        if best is None:
            return order
        u = queues[best][cursor[best]]
        cursor[best] += 1
        emitted[best] += u.size
        total += u.size
        order.append(u)


def composition_capacity(order: Sequence[LadderUnit], shares: Mapping[str, float]) -> int:
    """Largest prefix size at which no declared stratum has yet run dry.

    Past this point the interleave can no longer serve the exhausted stratum,
    so composition is no longer held; a rung above it is refused by
    ``cut_rungs`` unless the caller explicitly allows drift.
    """
    remaining = Counter()
    for u in order:
        remaining[u.stratum] += 1
    declared = {s for s, w in shares.items() if float(w) > 0}
    cum = 0
    for u in order:
        remaining[u.stratum] -= 1
        cum += u.size
        if remaining[u.stratum] == 0 and u.stratum in declared:
            return cum
    return cum


def cut_rungs(order: Sequence[LadderUnit], targets: Sequence[int], *,
              shares: Mapping[str, float] | None = None,
              allow_composition_drift: bool = False) -> list[dict]:
    """Prefix length for each target: the shortest prefix whose size reaches it.

    Targets must be strictly increasing. A target beyond the corpus is reported
    unreachable; one beyond ``composition_capacity`` (when ``shares`` is given)
    is refused unless ``allow_composition_drift``.
    """
    targets = [int(t) for t in targets]
    if any(b <= a for a, b in zip(targets, targets[1:])):
        raise LadderError(f"rung targets must be strictly increasing: {targets}")
    capacity = composition_capacity(order, shares) if shares else None
    out, cum, n = [], 0, 0
    for target in targets:
        while n < len(order) and cum < target:
            cum += order[n].size
            n += 1
        reachable = cum >= target
        row = {"target": target, "n_units": n if reachable else None,
               "size": cum if reachable else None, "reachable": reachable}
        if reachable and capacity is not None and cum > capacity:
            if not allow_composition_drift:
                raise LadderError(
                    f"rung {target}: prefix size {cum} exceeds the composition "
                    f"capacity {capacity}; a declared stratum is exhausted below it")
            row["composition_drift"] = True
        out.append(row)
    return out


def rung_accounting(prefix: Sequence[LadderUnit], shares: Mapping[str, float] | None = None
                    ) -> dict:
    """What one rung holds: size, counts, per-stratum shares, coverage."""
    size = sum(u.size for u in prefix)
    counts: Counter = Counter()
    by_stratum: Counter = Counter()
    families: dict[str, set] = defaultdict(set)
    attrs: dict[str, set] = defaultdict(set)
    for u in prefix:
        counts.update(u.counts)
        by_stratum[u.stratum] += u.size
        if u.family is not None:
            families[u.stratum].add(u.family)
        for k, v in u.attributes.items():
            attrs[k].add(v)
    realized = {s: round(v / size, 6) if size else 0.0 for s, v in sorted(by_stratum.items())}
    row = {
        "n_units": len(prefix), "size": size, "counts": dict(sorted(counts.items())),
        "size_by_stratum": dict(sorted(by_stratum.items())),
        "realized_shares": realized,
        "distinct_families_by_stratum": {s: len(v) for s, v in sorted(families.items())},
        "distinct_families": len({f for v in families.values() for f in v}),
        "distinct_attribute_values": {k: len(v) for k, v in sorted(attrs.items())},
    }
    if shares:
        scale = sum(float(w) for w in shares.values() if float(w) > 0)
        declared = {s: float(w) / scale for s, w in shares.items() if float(w) > 0}
        row["max_share_deviation"] = round(max(
            abs(declared.get(s, 0.0) - realized.get(s, 0.0))
            for s in set(declared) | set(realized)), 6)
    return row


def require_nested(rung_unit_ids: Sequence[Sequence[str]]) -> None:
    """Refuse a ladder whose rungs are not subsets of the next one."""
    for k in range(1, len(rung_unit_ids)):
        lo, hi = set(rung_unit_ids[k - 1]), set(rung_unit_ids[k])
        if not lo <= hi:
            raise LadderError(f"rung {k - 1} is not contained in rung {k}: "
                              f"{len(lo - hi)} units missing")


def build_ladder(units: Iterable[LadderUnit], shares: Mapping[str, float],
                 targets: Sequence[int], *, salt: str,
                 reserve: int = 0, allow_composition_drift: bool = False) -> dict:
    """Order, optionally reserve a held-out tail, and cut nested rungs.

    ``reserve`` takes a composition-matched set of units of at least that size
    OUT of the corpus before any rung is cut -- the in-training validation set,
    which no rung may contain. It is drawn by the same interleave under its own
    salt, so its composition follows the declared shares, and it never takes a
    family's last remaining unit, so reserving cannot remove a family from the
    rungs.

    Returns ``{"schema", "order": [unit ids], "rungs": [...], "reserved":
    [unit ids], "reserved_accounting", "composition_capacity"}``, where each rung
    row also carries ``rung_accounting`` of its prefix.
    """
    units = list(units)
    reserved: list[LadderUnit] = []
    if reserve > 0:
        left = Counter((u.stratum, u.family) for u in units if u.family is not None)
        acc = 0
        for u in nested_order(units, shares, salt=salt + "|reserve"):
            if acc >= reserve:
                break
            if u.family is not None:
                if left[(u.stratum, u.family)] <= 1:
                    continue
                left[(u.stratum, u.family)] -= 1
            reserved.append(u)
            acc += u.size
        if acc < reserve:
            raise LadderError(f"corpus cannot reserve {reserve}: only {acc}")
        held = {u.unit_id for u in reserved}
        units = [u for u in units if u.unit_id not in held]
    order = nested_order(units, shares, salt=salt)
    rungs = cut_rungs(order, targets, shares=shares,
                      allow_composition_drift=allow_composition_drift)
    for row in rungs:
        if row["reachable"]:
            row["accounting"] = rung_accounting(order[:row["n_units"]], shares)
    reachable = [order[:r["n_units"]] for r in rungs if r["reachable"]]
    require_nested([[u.unit_id for u in p] for p in reachable])
    return {
        "schema": LADDER_SCHEMA,
        "salt": salt,
        "declared_shares": dict(sorted(shares.items())),
        "order": [u.unit_id for u in order],
        "composition_capacity": composition_capacity(order, shares),
        "rungs": rungs,
        "reserved": [u.unit_id for u in reserved],
        "reserved_accounting": rung_accounting(reserved, shares) if reserved else None,
    }
