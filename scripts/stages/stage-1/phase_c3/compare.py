"""What changed between the B1 and B4 head maps, and whether it was close.

"N layers changed" is the number that overstates. The FFN evidence already
showed this: 31 of 36 layers moved, which sounds total, while the fraction of
actual neurons that moved was small. A layer counts as changed if ONE slot in
it differs, so the layer count measures how WIDELY a difference is spread, not
how MUCH changed.

So this reports the denominator at every level — layers, GQA groups, retained
slots — and then the thing the counts cannot say: how close each changed
selection was. A slot that flipped across a cutoff margin of 1e-9 and one that
flipped across a margin of 0.3 are the same integer and completely different
findings.

Nothing here decides anything. The gate is in the pilot record; this produces
the numbers it reads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: A changed selection whose losing margin is below this is "within noise" in
#: the descriptive sense only — it is a REPORTING split, not a threshold that
#: decides anything, and it is stated so the two columns have a definition.
NARROW_MARGIN = 1e-6


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Rank correlation, or None when it would not mean anything.

    Undefined for fewer than three points and for a constant vector; returning
    a number in those cases would put a confident 0.0 or nan next to the real
    ones.
    """
    n = len(a)
    if n < 3 or len(b) != n:
        return None

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def compare_head_maps(b1: Mapping[str, Any], b4: Mapping[str, Any]) -> dict:
    """The structural delta between two arms' `causal_head_evidence`.

    Both arguments are the artifact the operator produces, so this reads the
    same object a record holds rather than a summary of it.
    """
    for name, ev in (("B1", b1), ("B4", b4)):
        for key in ("head_scores", "gqa_decisions"):
            if key not in ev:
                raise ValueError(f"{name} evidence has no {key!r}")
    if len(b1["head_scores"]) != len(b4["head_scores"]):
        raise ValueError("the two arms scored different layer counts")

    n_layers = len(b1["head_scores"])
    layers_changed, groups_total, groups_changed = [], 0, 0
    slots_total = slots_changed = 0
    symmetric_difference = 0
    per_group: list[dict[str, Any]] = []
    narrow = wide = 0

    for layer in range(n_layers):
        g1, g4 = b1["gqa_decisions"][layer], b4["gqa_decisions"][layer]
        if len(g1) != len(g4):
            raise ValueError(f"layer {layer}: different GQA group counts")
        layer_changed = False
        for gi, (d1, d4) in enumerate(zip(g1, g4)):
            groups_total += 1
            s1, s4 = set(d1["selected_heads"]), set(d4["selected_heads"])
            slots_total += len(s1)
            if s1 == s4:
                continue
            groups_changed += 1
            layer_changed = True
            gained, lost = sorted(s4 - s1), sorted(s1 - s4)
            slots_changed += len(lost)
            symmetric_difference += len(gained) + len(lost)
            #: The margin the LOSING side had to cross, from each arm's own
            #: cutoffs. A flip across 1e-9 and one across 0.3 are the same
            #: integer in every count above and are not the same finding.
            margins = [m for m in (d1.get("cutoff_margin"),
                                   d4.get("cutoff_margin")) if m is not None]
            smallest = min(margins) if margins else None
            if smallest is not None and smallest < NARROW_MARGIN:
                narrow += 1
            else:
                wide += 1
            per_group.append({
                "layer": layer, "group": gi,
                "b1_selected": sorted(s1), "b4_selected": sorted(s4),
                "gained_by_b4": gained, "lost_by_b4": lost,
                "b1_cutoff_margin": d1.get("cutoff_margin"),
                "b4_cutoff_margin": d4.get("cutoff_margin"),
                "smallest_margin": smallest,
                "crossing": ("narrow" if smallest is not None
                             and smallest < NARROW_MARGIN else "wide"),
            })
        if layer_changed:
            layers_changed.append(layer)

    flat1 = [s for layer in b1["head_scores"] for s in layer]
    flat4 = [s for layer in b4["head_scores"] for s in layer]
    drift = [y - x for x, y in zip(flat1, flat4)]
    rel = [abs(d) / abs(x) for d, x in zip(drift, flat1) if x != 0]

    return {
        "schema": "aadistill.phase_c3.head_map_comparison/v1",
        "identical": symmetric_difference == 0,
        "layers": {"changed": len(layers_changed), "total": n_layers,
                   "changed_indices": layers_changed,
                   "_caveat": ("a layer counts as changed if ONE slot in it "
                               "differs; this measures how widely a difference "
                               "is spread, not how much changed")},
        "gqa_groups": {"changed": groups_changed, "total": groups_total},
        "retained_slots": {"changed": slots_changed, "total": slots_total,
                           "fraction": (slots_changed / slots_total
                                        if slots_total else 0.0)},
        "symmetric_difference_heads": symmetric_difference,
        "changed_selections": per_group,
        "crossings": {"narrow": narrow, "wide": wide,
                      "narrow_margin_threshold": NARROW_MARGIN,
                      "_threshold_decides_nothing": (
                          "a reporting split so the two columns have a "
                          "definition; no gate reads it")},
        "score_drift": _distribution(drift),
        "score_relative_drift": _distribution(rel),
        "rank_correlation": {
            "overall": _spearman(flat1, flat4),
            "per_layer": [_spearman(b1["head_scores"][i], b4["head_scores"][i])
                          for i in range(n_layers)],
            "_none_means": "fewer than 3 heads, or a constant vector",
        },
        "cutoff_margins": {
            "b1": _distribution([d["cutoff_margin"] for layer in b1["gqa_decisions"]
                                 for d in layer if d["cutoff_margin"] is not None]),
            "b4": _distribution([d["cutoff_margin"] for layer in b4["gqa_decisions"]
                                 for d in layer if d["cutoff_margin"] is not None]),
        },
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    """Enough to see the shape, and the n it was computed over."""
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    mean = sum(s) / n
    return {
        "n": n, "min": s[0], "max": s[-1], "mean": mean,
        "median": q(0.5), "p05": q(0.05), "p95": q(0.95),
        "abs_max": max(abs(v) for v in s),
        "std": math.sqrt(sum((v - mean) ** 2 for v in s) / n) if n > 1 else 0.0,
    }


def speedup(b1_seconds: float, b4_seconds: float) -> dict:
    """The gate's statistic, and nothing inferred.

    MEASURED wall clock only. Not FLOPs, not the physical invocation ratio,
    and emphatically not `B1/B4 = 4` — B4 computes ~37% MORE token-positions
    than B1 on this mixture, so the arithmetic prediction is not even the
    right direction of approximation.
    """
    if b4_seconds <= 0:
        raise ValueError(f"B4 scorer wall clock must be positive, got {b4_seconds}")
    return {
        "b1_scorer_seconds": round(float(b1_seconds), 4),
        "b4_scorer_seconds": round(float(b4_seconds), 4),
        "speedup": round(float(b1_seconds) / float(b4_seconds), 4),
        "_basis": "measured scorer wall clock, CUDA-synchronized at both ends",
    }


def verdict(speedup_value: float, identical: bool, *, threshold: float) -> str:
    """The three predeclared outcomes, and no fourth."""
    if speedup_value < threshold:
        return "B4_NOT_WORTH_ADOPTION_PILOT"
    if identical:
        return "B4_STRUCTURALLY_EQUIVALENT_AND_FASTER"
    return "B4_FASTER_AND_STRUCTURALLY_DIFFERENT_RECOVERY_TRIGGERED"
