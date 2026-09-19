"""The 12-probe schedule as executable control flow. Decides nothing by itself.

This is the part of the behavioural session that carries the science: which
probes exist, which candidate advances, and when confirmation may begin. The
training and scoring primitives are C1's and are already proven on real
hardware; what has never been executed is the shape around them, and that shape
is where a selection experiment goes wrong.

Everything here is pure. It takes descriptors and scores and returns decisions,
so the rules can be exercised exhaustively without a GPU — including the cases
that must never happen, which is the only way to know a guard exists.

Three properties it enforces, each because the frozen protocol says so:

* **B never advances.** It is the anchor and the estimand is a delta against it.
  A schedule that could advance its own anchor would be comparing B with B.
* **Screening produces no verdict.** It ranks. `GO`/`NO_GO`/`INCONCLUSIVE` are
  confirmation's alone, and a screening result that carried one would let a
  single seed name an incumbent.
* **Confirmation cannot start early.** All six screening probes must be trained
  AND scored and the winner mechanically determined first — otherwise the
  candidate being confirmed was chosen on partial data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: The anchor's arm name. One spelling, because a typo'd arm silently becomes a
#: sixth candidate and the anchor disappears from the comparison.
ANCHOR = "B"


class ScheduleError(RuntimeError):
    """The schedule was asked for something the frozen protocol forbids."""


@dataclass(frozen=True)
class Probe:
    """One probe: an arm trained from one initialization under one seed."""

    rung: str                 # "screening" | "confirmation"
    arm: str                  # a candidate state id, or ANCHOR
    seed: int
    initialization_artifact_digest: str
    initialization_path: str

    @property
    def probe_id(self) -> str:
        return f"{self.rung}.{self.arm}.s{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return {"probe_id": self.probe_id, "rung": self.rung, "arm": self.arm,
                "seed": self.seed,
                "initialization_artifact_digest":
                    self.initialization_artifact_digest,
                "initialization_path": self.initialization_path}


def screening_probes(candidates: Sequence[Mapping[str, Any]],
                     anchor: Mapping[str, Any],
                     seeds: Sequence[int]) -> list[Probe]:
    """Six probes: five candidates and the anchor, all on the screening seed.

    One seed, six arms. The frozen schedule says `seeds: 1, arms: 6`, and the
    arms are the five reconstructed candidates plus B — which is why B must be
    materialized before screening starts rather than being a staged input.
    """
    if len(seeds) != 1:
        raise ScheduleError(
            f"screening takes exactly one seed, got {len(seeds)}: {list(seeds)}")
    if len(candidates) != 5:
        raise ScheduleError(f"screening takes five candidates, got {len(candidates)}")
    seed = int(seeds[0])
    probes = [Probe("screening", c["state_id"], seed, c["artifact_digest"],
                    c["durable_path"]) for c in candidates]
    probes.append(Probe("screening", ANCHOR, seed, anchor["artifact_digest"],
                        anchor["durable_path"]))
    if len({p.arm for p in probes}) != 6:
        raise ScheduleError("two screening arms share a name; one would "
                            "overwrite the other's probe")
    return probes


def confirmation_probes(advanced: Mapping[str, Any],
                        anchor: Mapping[str, Any],
                        seeds: Sequence[int]) -> list[Probe]:
    """Six probes: the one advanced candidate and the anchor, paired on three seeds.

    Paired means both arms on every seed: a delta per seed, not two independent
    samples. Three seeds times two arms is six.
    """
    if len(seeds) != 3:
        raise ScheduleError(
            f"confirmation takes exactly three seeds, got {len(seeds)}")
    if len(set(seeds)) != 3:
        raise ScheduleError(f"the confirmation seeds are not distinct: {list(seeds)}")
    if advanced["state_id"] == ANCHOR:
        raise ScheduleError(
            "the anchor cannot be the advanced candidate; it is what the "
            "advanced candidate is measured against")
    probes: list[Probe] = []
    for seed in seeds:
        probes.append(Probe("confirmation", advanced["state_id"], int(seed),
                            advanced["artifact_digest"], advanced["durable_path"]))
        probes.append(Probe("confirmation", ANCHOR, int(seed),
                            anchor["artifact_digest"], anchor["durable_path"]))
    return probes


def rank_screening(scores: Mapping[str, float],
                   candidates: Sequence[Mapping[str, Any]],
                   ) -> list[dict[str, Any]]:
    """Order the candidates by paired delta against B. RANKING ONLY.

    `scores` maps arm -> `correct_overall` on the screening battery, and must
    contain the anchor: the estimand is `correct_overall(candidate) -
    correct_overall(B)` on the same seed, so a ranking without B is not a
    ranking of anything.

    Ties break by the frozen full-search ordering and then by state id — both
    fixed before any behavioural datum existed, which is what makes the
    tie-break a rule rather than a choice.
    """
    if ANCHOR not in scores:
        raise ScheduleError(
            "the anchor has no screening score; the delta every candidate is "
            "ranked by is against B, so there is nothing to rank against")
    anchor_score = float(scores[ANCHOR])
    missing = [c["state_id"] for c in candidates if c["state_id"] not in scores]
    if missing:
        raise ScheduleError(
            f"{len(missing)} candidate(s) have no screening score: {missing}. "
            "Ranking a partial field would select on who happened to finish.")
    ranked = [
        {"state_id": c["state_id"],
         "correct_overall": float(scores[c["state_id"]]),
         "anchor_correct_overall": anchor_score,
         "delta_vs_b": round(float(scores[c["state_id"]]) - anchor_score, 10),
         "frozen_search_rank": int(c["rank_in_frozen_selection"])}
        for c in candidates
    ]
    ranked.sort(key=lambda r: (-r["delta_vs_b"], r["frozen_search_rank"],
                               r["state_id"]))
    for position, row in enumerate(ranked):
        row["screening_position"] = position
    return ranked


def advance_one(ranked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exactly one candidate advances, and it is never the anchor.

    Advancing one is what makes the confirmation a single hypothesis needing no
    multiplicity correction. The protocol records that as a deliberate trade of
    breadth for a clean bound, and this is where it is enforced.
    """
    if not ranked:
        raise ScheduleError("nothing was ranked, so nothing can advance")
    winner = dict(ranked[0])
    if winner["state_id"] == ANCHOR:
        raise ScheduleError(
            "the anchor came first in a candidate ranking, which means B was "
            "ranked as a candidate. B is the comparator and never advances.")
    runner_up = ranked[1] if len(ranked) > 1 else None
    winner["advanced"] = True
    winner["margin_over_runner_up"] = (
        None if runner_up is None
        else round(winner["delta_vs_b"] - runner_up["delta_vs_b"], 10))
    winner["tie_broken"] = bool(
        runner_up is not None
        and winner["delta_vs_b"] == runner_up["delta_vs_b"])
    winner["_tie_break_rule"] = (
        "the frozen full-search ordering, then the state id. Both were fixed "
        "before any behavioural datum existed.")
    return winner


def screening_is_complete(probes: Sequence[Probe],
                          trained: Mapping[str, Any],
                          scored: Mapping[str, Any]) -> tuple[bool, str]:
    """May confirmation begin?

    Every screening probe must be trained AND scored. Starting confirmation on a
    partial field would confirm a candidate selected from whoever finished
    first, which is a different experiment from the one that was preregistered.
    """
    want = {p.probe_id for p in probes if p.rung == "screening"}
    untrained = sorted(want - set(trained))
    unscored = sorted(want - set(scored))
    if untrained:
        return False, (f"{len(untrained)} screening probe(s) are not trained: "
                       f"{untrained}. Confirmation may not begin.")
    if unscored:
        return False, (f"{len(unscored)} screening probe(s) are not scored: "
                       f"{unscored}. Confirmation may not begin.")
    return True, f"all {len(want)} screening probes are trained and scored"


#: What screening may NEVER return. Checked rather than trusted: a rung that
#: emitted one of these would let a single seed name an incumbent.
FORBIDDEN_IN_SCREENING = ("GO", "NO_GO", "INCONCLUSIVE")


def assert_screening_emits_no_verdict(result: Mapping[str, Any]) -> None:
    """Screening ranks. It does not decide, promote or veto."""
    for key in ("verdict", "decision", "terminal", "promotion", "incumbent"):
        if key in result:
            raise ScheduleError(
                f"the screening result carries {key!r}={result[key]!r}. "
                "Screening ranks and advances one candidate; it may not "
                "produce a verdict, name an incumbent or promote anything — "
                "that is confirmation's alone, on three disjoint seeds.")
    flat = str(result)
    for word in FORBIDDEN_IN_SCREENING:
        if word in flat:
            raise ScheduleError(
                f"the screening result mentions {word!r}. Screening produces no "
                "terminal verdict.")
