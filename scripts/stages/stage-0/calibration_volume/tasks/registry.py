"""Every generator, and the frozen family split between calibration and evaluation.

Two isolation tiers, both checked by key rather than assumed:

* **generator tier** -- `EVAL_ONLY_GENERATORS` never contribute a calibration
  item, so evaluation on them measures transfer to an unseen task FORMAT;
* **family tier** -- inside every other generator, the families are split by a
  salted hash rank (not by hand), the first `CAL_FRACTION` going to calibration
  and the rest to evaluation, so evaluation on them measures an unseen RULE.

Dev and final evaluation share the evaluation families and differ by seed
range; calibration seeds are a third, disjoint range. The split is a pure
function of the generator code and `SPLIT_SALT`, so it is reproducible without
a stored table, and `split_table()` prints it for the record.

Two rosters. `GENERATORS` is the v1 roster that built every historical set and
is what the historical builders and audits iterate; it never changes.
`GENERATORS_V2` (maintainer-authorized 2026-10-19) is the same roster with the
four generators the eval-set-1 audit found defective replaced by their v2
versions. A v2 generator keeps its v1 family ids, so the family split -- a
function of family ids only -- is the same in both rosters.
"""
from __future__ import annotations

import hashlib
import math

from . import analogy, composition, deduction, induction, interactive
from .common import Generator

GENERATORS: tuple[Generator, ...] = (
    *induction.GENERATORS, *composition.GENERATORS, *deduction.GENERATORS,
    *analogy.GENERATORS, *interactive.GENERATORS)

BY_ID = {g.generator_id: g for g in GENERATORS}

GENERATORS_V2: tuple[Generator, ...] = (
    *induction.GENERATORS_V2, *composition.GENERATORS, *deduction.GENERATORS_V2,
    *analogy.GENERATORS_V2, *interactive.GENERATORS)

BY_ID_V2 = {g.generator_id: g for g in GENERATORS_V2}

EVAL_ONLY_GENERATORS = frozenset({
    "sequence_induction@v1",
    "counterfactual_arith@v1",
    "graph_paths@v1",
    "graph_paths@v2",
    "letter_analogy@v1",
    "letter_analogy@v2",
    "code_breaker@v1",
})

SPLIT_SALT = "stage0.calibration_volume.family_split/v1"
CAL_FRACTION = 0.45

#: Seed ranges per role: disjoint, so no two roles can share an instance.
SEED_RANGES = {"dev": (0, 1_000_000), "final": (1_000_000, 2_000_000),
               "calibration": (2_000_000, 3_000_000)}


def _rank(family_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}|{family_id}".encode()).hexdigest()


def family_split(gen: Generator) -> dict[str, list[str]]:
    fams = sorted((f.family_id for f in gen.families), key=_rank)
    if gen.generator_id in EVAL_ONLY_GENERATORS:
        return {"calibration": [], "evaluation": sorted(fams)}
    n_cal = max(1, math.floor(len(fams) * CAL_FRACTION))
    return {"calibration": sorted(fams[:n_cal]), "evaluation": sorted(fams[n_cal:])}


def split_table() -> dict:
    return {g.generator_id: {"axis": g.axis, "interactive": g.interactive,
                             "eval_only_generator": g.generator_id in EVAL_ONLY_GENERATORS,
                             **family_split(g)} for g in GENERATORS}
