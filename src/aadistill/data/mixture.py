"""Type-mixture construction for the token ladder.

Both functions here answer the same question at two levels: *in what order do we
emit items so that every prefix carries the declared type mixture?* That matters
because the ladder's rungs are prefixes — if the mixture drifted along the pack,
data size and data mixture would be confounded and a scaling curve would be
measuring both at once.

* `interleave` orders **sessions**, before packing.
* `order_blocks` orders **packed blocks**, after packing — needed because the
  system prompt is a hard packing boundary, so sessions cannot be interleaved
  across system-prompt groups and the mixture has to be repaired one level up.

A third largest-remainder rule lives in `ladder.select_val_blocks`. **The three
are deliberately not unified.** They optimize different objectives on different
units: `interleave` tracks a token-share target per type, `order_blocks`
minimizes squared error against declared shares, and `select_val_blocks`
maximizes the dominant type's deficit against an equal share. Collapsing them
into one parameterized helper would hide those differences behind flags and make
each call site harder to reason about, not easier.

Everything here is deterministic and seed-free: the same inputs give the same
order on any machine, which is what lets a rung be reproduced from its manifest.
"""

from __future__ import annotations

from collections import Counter


def interleave(by_type: dict[str, list], shares: dict[str, float] | None = None) -> list:
    """Deterministic stratified interleave: every prefix keeps the type mixture.

    Without `shares`, types are emitted in proportion to their session counts.
    With `shares`, the target is a **supervised-token** share per type — the
    quantity that actually trains the model, and the one a difficulty-aware
    mixture is declared in. Session lengths differ by ~6x across these types, so
    an equal session count is not an equal token contribution.

    Either way the rule is largest-remainder: emit from whichever type is
    furthest behind its target so far. It needs no seed and no shuffling, so the
    order is a pure function of the per-type session lists and the declared
    shares — which is what keeps every ladder rung and both training seeds on
    one fixed mixture.

    A type that runs out early stops contributing; the caller compares the
    realized shares against the declared ones and reports the drift.
    """
    live = {t: v for t, v in by_type.items() if v}
    if not live:
        return []
    cursors = {t: 0 for t in live}
    order = []

    if shares is None:
        totals = {t: len(v) for t, v in live.items()}
        grand = sum(totals.values())
        for step in range(grand):
            best, best_deficit = None, None
            for t in sorted(totals):
                if cursors[t] >= totals[t]:
                    continue
                deficit = (totals[t] / grand) * step - cursors[t]
                if best is None or deficit > best_deficit:
                    best, best_deficit = t, deficit
            order.append(live[best][cursors[best]])
            cursors[best] += 1
        return order

    weights = {t: shares.get(t, 0.0) for t in live}
    if sum(weights.values()) <= 0:
        raise ValueError("declared mixture shares sum to zero")
    scale = sum(weights.values())
    weights = {t: w / scale for t, w in weights.items()}
    emitted = {t: 0 for t in live}  # supervised tokens emitted per type
    total_emitted = 0
    remaining = sum(len(v) for v in live.values())

    for _ in range(remaining):
        best, best_deficit = None, None
        for t in sorted(live):
            if cursors[t] >= len(live[t]) or weights[t] <= 0:
                continue
            deficit = weights[t] * total_emitted - emitted[t]
            if best is None or deficit > best_deficit:
                best, best_deficit = t, deficit
        if best is None:
            break
        session = live[best][cursors[best]]
        order.append(session)
        cursors[best] += 1
        emitted[best] += session.n_supervised
        total_emitted += session.n_supervised
    return order


def block_token_mix(block) -> Counter:
    """Supervised tokens contributed by each data type inside one packed block."""
    mix = Counter()
    for m in block.audit["sessions"]:
        mix[m["data_type"]] += m["supervised_retained"]
    return mix


def order_blocks(blocks, shares):
    """Order packed blocks so every prefix carries the declared token mixture.

    Sessions cannot be interleaved across system-prompt groups — the system
    prompt is a hard packing boundary — so the mixture is restored one level up,
    on blocks. Largest-remainder again: emit whichever block most reduces the
    gap between the emitted per-type token shares and the declared ones.

    Without declared shares the input order is kept, which is the single-group
    behaviour and leaves the session-level interleave in charge.
    """
    if not shares or len(blocks) <= 1:
        return list(blocks)
    scale = sum(shares.values())
    weights = {t: w / scale for t, w in shares.items()}
    pending = list(range(len(blocks)))
    mixes = [block_token_mix(b) for b in blocks]
    emitted = Counter()
    total = 0
    out = []
    while pending:
        best, best_score = None, None
        for i in pending:
            mix = mixes[i]
            n = sum(mix.values())
            # Squared error of the resulting shares against the declared ones;
            # lower is better, so the block that best repairs the mix wins.
            new_total = total + n
            if new_total == 0:
                score = 0.0
            else:
                score = sum(
                    (weights.get(t, 0.0) - (emitted[t] + mix[t]) / new_total) ** 2
                    for t in set(weights) | set(emitted) | set(mix))
            if best is None or score < best_score:
                best, best_score = i, score
        out.append(blocks[best])
        emitted += mixes[best]
        total += sum(mixes[best].values())
        pending.remove(best)
    return out
