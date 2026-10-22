"""Mechanism isolation: does a training rule COMPUTE what an evaluation rule computes?

Keys and 13-grams (R1, R2) show that a training prompt is not a copy of a
held-out prompt; R3-R5 show that it does not come from a held-out family,
generator or task type. None of them shows that the training data does not
TEACH the mechanism an evaluation item asks the model to discover. A rule
stated or exemplified in one format (a word rewrite) can be the same function
as a held-out rule in another format (a letter analogy), and training on it
turns an "unseen" evaluation family into a seen one.

For the project's procedural families the rules are tabulated functions, so
this is checked by execution, not by reading names: every training-side
family's rule is applied to the same seeded random inputs as every
evaluation-side rule of the same domain, and the agreement rate is measured.

* ``equivalent`` -- some variant of the training rule agrees with some variant
  of the evaluation rule on every probe input: the same function. Refused for
  training (isolation rule R9).
* ``overlapping`` -- agreement on at least ``OVERLAP`` of the inputs: a shared
  primitive, reported.

The evaluation side is every evaluation-split family of a shared generator and
every family of an evaluation-only generator; the training side is every
calibration-split family. Domains with no tabulated rule (deduction, ordering,
tracking, defined operations, episodes) and external modules are covered by
the declared review in ``external_modules.json :: mechanism``, which this
module does not replace.
"""
from __future__ import annotations

import random
from functools import lru_cache

from stages.calibration_volume.tasks import analogy, induction
from stages.calibration_volume.tasks import registry as R

N_INPUTS = 400
OVERLAP = 0.5
SEED = 20261022


def _variants(table: dict, fid: str) -> list:
    v = table[fid][1]
    return [fn for _label, fn in v] if v and isinstance(v[0], tuple) else list(v if isinstance(v, list) else [v])


def _tables() -> dict[str, dict[str, tuple[str, list]]]:
    """domain -> {generator/family -> (generator id, rule variants)}."""
    letter = {fid: ("letter_analogy@v2", [fn]) for fid, (_c, fn) in analogy._LETTER_RULES.items()}
    return {
        "list": {f: ("list_induction@v1", _variants(induction.LIST_RULES, f)) for f in induction.LIST_RULES},
        "string": {**{f: ("string_induction@v2", _variants(induction.STRING_RULES, f))
                      for f in induction.STRING_RULES}, **letter},
        "grid": {f: ("grid_transform@v1", list(analogy._GRID_RULES[f][1])) for f in analogy._GRID_RULES},
    }


def _inputs(domain: str, rng: random.Random) -> list:
    if domain == "list":
        return [[rng.randint(0, 20) for _ in range(rng.randint(3, 8))] for _ in range(N_INPUTS)]
    if domain == "string":
        out = []
        for _ in range(N_INPUTS):
            start, n = rng.randrange(0, 18), rng.randint(3, 7)
            # half runs (letter analogies are defined on them), half random words
            out.append("".join(chr(97 + start + k) for k in range(n)) if rng.random() < 0.5
                       else "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)))
        return out
    h_w = [(rng.randint(3, 5), rng.randint(3, 5)) for _ in range(N_INPUTS)]
    return [[[rng.choice([0, 0, 0, rng.randint(1, 9)]) for _ in range(w)] for _ in range(h)] for h, w in h_w]


def _apply(fn, x):
    try:
        return fn(list(x) if isinstance(x, list) and x and not isinstance(x[0], list) else
                  [list(r) for r in x] if isinstance(x, list) else x)
    except (IndexError, ValueError, ZeroDivisionError, KeyError):
        return None


def _agreement(fs: list, gs: list, inputs: list) -> float:
    best = 0.0
    for f in fs:
        for g in gs:
            same = n = 0
            for x in inputs:
                a, b = _apply(f, x), _apply(g, x)
                if a is None or b is None:
                    continue
                n += 1
                same += a == b
            if n >= N_INPUTS // 4:
                best = max(best, same / n)
    return best


def sides() -> tuple[set[str], set[str]]:
    """(training-side, evaluation-side) as 'generator/family' keys, v2 roster."""
    train, held = set(), set()
    for g in R.GENERATORS_V2:
        split = R.family_split(g)
        if g.generator_id in R.EVAL_ONLY_GENERATORS:
            held |= {f"{g.generator_id}/{f.family_id}" for f in g.families}
        else:
            train |= {f"{g.generator_id}/{f}" for f in split["calibration"]}
            held |= {f"{g.generator_id}/{f}" for f in split["evaluation"]}
    return train, held


@lru_cache(maxsize=1)
def review() -> dict:
    train, held = sides()
    rng = random.Random(SEED)
    pairs = []
    covered_train, covered_held = set(), set()
    for domain, table in _tables().items():
        inputs = _inputs(domain, rng)
        rows = {f"{gid}/{fid}": fns for fid, (gid, fns) in table.items()}
        tr = sorted(k for k in rows if k in train)
        ev = sorted(k for k in rows if k in held)
        covered_train |= set(tr)
        covered_held |= set(ev)
        for t in tr:
            for e in ev:
                a = _agreement(rows[t], rows[e], inputs)
                if a >= OVERLAP:
                    pairs.append({"domain": domain, "training_family": t, "evaluation_family": e,
                                  "agreement": round(a, 4),
                                  "verdict": "equivalent" if a == 1.0 else "overlapping"})
    pairs.sort(key=lambda p: (-p["agreement"], p["training_family"], p["evaluation_family"]))
    return {"rule": "stage2.reasoning_data.mechanism_check/v1", "inputs_per_domain": N_INPUTS,
            "seed": SEED, "overlap_threshold": OVERLAP,
            "training_families_checked": len(covered_train), "evaluation_families_checked": len(covered_held),
            "training_families_unchecked": sorted(train - covered_train),
            "pairs": pairs,
            "equivalent_training_families": sorted({p["training_family"] for p in pairs
                                                    if p["verdict"] == "equivalent"})}
