"""The 12-probe control flow, executed end to end and at its boundaries.

The training and scoring primitives are C1's and are proven on real hardware.
What has never run is the shape around them — descriptors, ranking, tie-break,
advancement, the gate before confirmation — and that shape is where a selection
experiment goes wrong. So it is exercised here exhaustively, including every
case that must never happen, because a guard nobody has seen refuse is not a
guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

from experiments.phase_c2 import behavioural as B  # noqa: E402
from experiments.phase_c2 import behavioural_schedule as S  # noqa: E402

SCREENING_SEED = 616738081
CONFIRMATION_SEEDS = [1936324010, 1916380711, 1523147638]


def anchor():
    return {"state_id": S.ANCHOR, "artifact_digest": "b" * 64,
            "durable_path": "/store/incumbent_b"}


def candidates(n: int = 5):
    return [{"state_id": f"cand{i}", "artifact_digest": f"{i}" * 64,
             "durable_path": f"/store/cand{i}", "rank_in_frozen_selection": i}
            for i in range(n)]


# -- the probes themselves ---------------------------------------------------
def test_screening_is_six_probes_over_six_arms_on_one_seed():
    probes = S.screening_probes(candidates(), anchor(), [SCREENING_SEED])
    assert len(probes) == 6
    assert {p.seed for p in probes} == {SCREENING_SEED}
    assert sum(p.arm == S.ANCHOR for p in probes) == 1, "the anchor runs once"
    assert len({p.probe_id for p in probes}) == 6


def test_confirmation_is_six_probes_paired_across_three_seeds():
    probes = S.confirmation_probes(candidates()[0], anchor(), CONFIRMATION_SEEDS)
    assert len(probes) == 6
    #: PAIRED: both arms on every seed, so each seed yields a delta.
    for seed in CONFIRMATION_SEEDS:
        arms = {p.arm for p in probes if p.seed == seed}
        assert arms == {"cand0", S.ANCHOR}, (seed, arms)


def test_the_two_rungs_together_are_exactly_twelve():
    total = (len(S.screening_probes(candidates(), anchor(), [SCREENING_SEED]))
             + len(S.confirmation_probes(candidates()[0], anchor(),
                                         CONFIRMATION_SEEDS)))
    assert total == 12


@pytest.mark.parametrize("seeds", [[], [1, 2], CONFIRMATION_SEEDS])
def test_screening_refuses_anything_but_one_seed(seeds):
    with pytest.raises(S.ScheduleError, match="exactly one seed"):
        S.screening_probes(candidates(), anchor(), seeds)


@pytest.mark.parametrize("n", [4, 6])
def test_screening_refuses_a_field_that_is_not_five(n):
    with pytest.raises(S.ScheduleError, match="five candidates"):
        S.screening_probes(candidates(n), anchor(), [SCREENING_SEED])


def test_confirmation_refuses_repeated_seeds():
    """Three draws of one seed is one seed, not three."""
    with pytest.raises(S.ScheduleError, match="not distinct"):
        S.confirmation_probes(candidates()[0], anchor(), [7, 7, 9])


def test_the_anchor_can_never_be_the_advanced_candidate():
    with pytest.raises(S.ScheduleError, match="cannot be the advanced"):
        S.confirmation_probes(anchor(), anchor(), CONFIRMATION_SEEDS)


# -- ranking and advancement -------------------------------------------------
def test_ranking_is_by_paired_delta_against_the_anchor():
    cands = candidates()
    scores = {"cand0": 0.30, "cand1": 0.42, "cand2": 0.35,
              "cand3": 0.11, "cand4": 0.38, S.ANCHOR: 0.33}
    ranked = S.rank_screening(scores, cands)
    assert [r["state_id"] for r in ranked] == [
        "cand1", "cand4", "cand2", "cand0", "cand3"]
    assert ranked[0]["delta_vs_b"] == pytest.approx(0.09)
    #: A candidate BELOW the anchor still ranks; screening does not veto.
    assert ranked[-1]["delta_vs_b"] < 0


def test_ranking_without_the_anchor_is_refused():
    """The estimand is a delta against B. Without B there is no delta."""
    with pytest.raises(S.ScheduleError, match="anchor has no screening score"):
        S.rank_screening({f"cand{i}": 0.3 for i in range(5)}, candidates())


def test_ranking_a_partial_field_is_refused():
    """Selecting from whoever finished is a different experiment."""
    scores = {"cand0": 0.4, "cand1": 0.5, S.ANCHOR: 0.3}
    with pytest.raises(S.ScheduleError, match="no screening score"):
        S.rank_screening(scores, candidates())


def test_a_tie_breaks_on_the_frozen_search_order_then_state_id():
    """Both were fixed before any behavioural datum existed."""
    cands = candidates()
    scores = {c["state_id"]: 0.40 for c in cands}
    scores[S.ANCHOR] = 0.30
    ranked = S.rank_screening(scores, cands)
    assert [r["state_id"] for r in ranked] == [f"cand{i}" for i in range(5)]
    winner = S.advance_one(ranked)
    assert winner["state_id"] == "cand0"
    assert winner["tie_broken"] is True
    assert winner["margin_over_runner_up"] == 0


def test_the_tie_break_prefers_search_rank_over_state_id():
    """Two orderings could decide a tie; the frozen one must win."""
    cands = [{"state_id": "zzz", "artifact_digest": "a" * 64,
              "durable_path": "/s/z", "rank_in_frozen_selection": 0},
             {"state_id": "aaa", "artifact_digest": "b" * 64,
              "durable_path": "/s/a", "rank_in_frozen_selection": 1}]
    ranked = S.rank_screening({"zzz": 0.4, "aaa": 0.4, S.ANCHOR: 0.1}, cands)
    assert ranked[0]["state_id"] == "zzz", (
        "the tie broke on state id; the frozen search order comes first")


def test_exactly_one_candidate_advances():
    cands = candidates()
    scores = {c["state_id"]: 0.3 + i / 100 for i, c in enumerate(cands)}
    scores[S.ANCHOR] = 0.3
    winner = S.advance_one(S.rank_screening(scores, cands))
    assert winner["advanced"] is True
    assert winner["state_id"] == "cand4"
    assert winner["margin_over_runner_up"] > 0


def test_advancing_the_anchor_is_refused():
    ranked = [{"state_id": S.ANCHOR, "delta_vs_b": 0.0,
               "frozen_search_rank": 0}]
    with pytest.raises(S.ScheduleError, match="never advances"):
        S.advance_one(ranked)


# -- the gate before confirmation --------------------------------------------
def test_confirmation_is_blocked_until_every_screening_probe_is_scored():
    probes = S.screening_probes(candidates(), anchor(), [SCREENING_SEED])
    ids = [p.probe_id for p in probes]

    ok, why = S.screening_is_complete(probes, {}, {})
    assert not ok and "not trained" in why

    ok, why = S.screening_is_complete(probes, dict.fromkeys(ids, 1), {})
    assert not ok and "not scored" in why

    ok, why = S.screening_is_complete(
        probes, dict.fromkeys(ids, 1), dict.fromkeys(ids[:-1], 1))
    assert not ok and ids[-1] in why

    ok, why = S.screening_is_complete(
        probes, dict.fromkeys(ids, 1), dict.fromkeys(ids, 1))
    assert ok and "all 6" in why


# -- screening may not decide ------------------------------------------------
@pytest.mark.parametrize("payload", [
    {"verdict": "GO"}, {"decision": "NO_GO"}, {"terminal": "INCONCLUSIVE"},
    {"incumbent": "cand1"}, {"promotion": True},
    {"note": "this rung reached INCONCLUSIVE"},
])
def test_a_screening_result_carrying_a_verdict_is_refused(payload):
    with pytest.raises(S.ScheduleError):
        S.assert_screening_emits_no_verdict(payload)


def test_an_ordinary_screening_result_passes_that_guard():
    """Guards the guard: if everything raised, the check would prove nothing."""
    S.assert_screening_emits_no_verdict(
        {"ranked": [{"state_id": "cand1", "delta_vs_b": 0.09}],
         "advanced": "cand1"})


# -- end to end, against the real frozen inputs ------------------------------
def test_the_whole_flow_runs_on_the_real_candidates_and_seeds():
    """Screening -> scoring -> ranking -> advancement -> confirmation, using the
    ACTUAL five reconstructed candidates and the ACTUAL frozen seeds.

    Only the scores are synthetic: producing real ones needs twelve GPU-hours.
    Everything the scores flow through is the code the session will run.
    """
    proto = B.protocol(ROOT)["behavioural_selection"]
    cands = B.candidate_manifest(ROOT)
    b = B.b_binding(ROOT)
    anchor_arm = {"state_id": S.ANCHOR,
                  "artifact_digest": b["required_identity"]["artifact_digest"],
                  "durable_path": b["availability"]["durable_path"]}

    screening = S.screening_probes(cands, anchor_arm, proto["seeds"]["screening"])
    assert len(screening) == 6

    trained = {p.probe_id: {"ok": True} for p in screening}
    scored = {p.probe_id: {"ok": True} for p in screening}
    ok, _ = S.screening_is_complete(screening, trained, scored)
    assert ok

    scores = {c["state_id"]: 0.30 + i / 100 for i, c in enumerate(cands)}
    scores[S.ANCHOR] = 0.31
    ranked = S.rank_screening(scores, cands)
    result = {"ranked": ranked}
    S.assert_screening_emits_no_verdict(result)

    winner = S.advance_one(ranked)
    advanced = next(c for c in cands if c["state_id"] == winner["state_id"])
    confirmation = S.confirmation_probes(advanced, anchor_arm,
                                         proto["seeds"]["confirmation"])
    assert len(confirmation) == 6
    assert len(screening) + len(confirmation) == 12

    #: Every probe names a real initialization, and the anchor's is B's frozen
    #: identity rather than a candidate's.
    digests = {c["artifact_digest"] for c in cands}
    for p in screening + confirmation:
        if p.arm == S.ANCHOR:
            assert p.initialization_artifact_digest == \
                b["required_identity"]["artifact_digest"]
        else:
            assert p.initialization_artifact_digest in digests
