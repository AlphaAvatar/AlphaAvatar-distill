"""The shared evaluation protocol, from committed records only, at $0.

The audit that re-hashes the durable bytes and replays D1's two decisions runs
where the store is (`shared_evaluation.py`, recorded in
`analyses/d1_shared_evidence_audit.json`). These tests check what must hold
everywhere: the committed protocol still hashes to its identity, this tree
still trains, generates and scores the way the imported units were, and the
rules that make the protocol weaker than a fresh confirmation are the ones
recorded.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[6]

from stages.d_series import shared_evaluation as S  # noqa: E402


@pytest.fixture(scope="module")
def record():
    return json.loads((REPO / S.RECORD).read_text())


def test_the_committed_protocol_hashes_to_its_identity(record):
    assert S.verify_record(record)


def test_an_edited_protocol_no_longer_verifies(record):
    edited = copy.deepcopy(record)
    edited["statistics"]["sesoi"] = 0.005
    assert not S.verify_record(edited)


def test_this_tree_reproduces_the_pinned_implementation(record):
    """A candidate is paired with the imported units only if it is trained,
    generated and scored by the same code. Changing the trainer or a scorer is
    allowed -- under a new protocol version, not silently under this one."""
    assert S.tree_matches(record, REPO) == {"trainer": [], "generation": [],
                                            "scoring": []}


def test_each_rung_reuses_a_d1_battery_and_d1s_own_seeds(record):
    prereg = json.loads((REPO / S.D1_PREREGISTRATION).read_text())
    screening = record["rungs"]["shared_screening"]
    confirmation = record["rungs"]["shared_confirmation"]
    assert screening["reuses_battery"]["role"] == "d1_screening"
    assert confirmation["reuses_battery"]["role"] == "d1_confirmation"
    assert screening["seeds"] == sorted(prereg["seeds"]["screening"])
    assert confirmation["seeds"] == sorted(prereg["seeds"]["confirmation"])
    for rung in (screening, confirmation):
        arms = sorted({u["arm"] for u in rung["reference_units"]})
        assert arms == ["B", "q2"]
        assert len(rung["reference_units"]) == 2 * len(rung["seeds"])


def test_d1_is_cited_and_not_rewritten(record):
    prereg = json.loads((REPO / S.D1_PREREGISTRATION).read_text())
    verdict = json.loads((REPO / S.D1_RUNS / S.RUNGS["shared_confirmation"][
        "source_run"] / "evidence" / "confirmation_verdict.json").read_text())
    cited = record["d1_records_unchanged"]
    assert cited["preregistration_sha256"] == prereg["preregistration_sha256"]
    assert cited["confirmation_verdict_sha256"] == verdict["verdict_sha256"]
    assert cited["verdict"] == "NO-GO"


def test_the_reserved_batteries_are_untouched(record):
    family = json.loads((REPO / S.FAMILY_LOGS / "current.json").read_text())
    for role in S.RESERVED_ROLES:
        assert record["reserved_independent_batteries"][role] == \
            family["roles"][role]
    used = {r["reuses_battery"]["role"] for r in record["rungs"].values()}
    assert not used & set(S.RESERVED_ROLES)


def test_the_family_alpha_is_split_before_any_result(record):
    fam = record["statistics"]["family"]
    assert fam["members"] == ["phase_d2", "phase_d3"]
    assert fam["per_experiment_alpha_one_sided"] == pytest.approx(0.025)
    assert fam["d1_already_spent"]["union_bound_over_d1_d2_d3"] == \
        pytest.approx(0.10)


@pytest.mark.parametrize("verdict,outcome", [("GO", "SHARED_GO"),
                                             ("NO-GO", "NO_GO"),
                                             ("INCONCLUSIVE", "INCONCLUSIVE")])
def test_a_pass_on_shared_rows_nominates_and_never_promotes(verdict, outcome):
    assert S.shared_outcome({"verdict": verdict}) == outcome


def test_bootstrap_seeds_are_separated_per_experiment(record):
    h = record["protocol_hash"]
    seeds = {S.bootstrap_seed(h, e, "shared_confirmation")
             for e in ("phase_d2", "phase_d3")}
    assert len(seeds) == 2


# --- the screening rule over a new field and imported comparator rows ------

SEEDS = (11, 22)


def _rows(arm, correct, usable):
    return [{"arm": arm, "seed": s, "correct_overall": c,
             "usable_rollout_rate": u}
            for s, c, u in zip(SEEDS, correct, usable)]


FIELD = [S.FieldArm("c1", 1, "s1"), S.FieldArm("c2", 2, "s2")]
B_ROWS = _rows("B", (0.02, 0.02), (0.60, 0.60))


def test_the_better_candidate_advances():
    scored = _rows("c1", (0.03, 0.03), (0.6, 0.6)) + \
        _rows("c2", (0.02, 0.025), (0.6, 0.6))
    got = S.rank_screening(FIELD, scored, B_ROWS, seeds=SEEDS)
    assert got["selection"]["arm"] == "c1"


def test_a_vetoed_candidate_sorts_last_whatever_its_delta():
    scored = _rows("c1", (0.05, 0.05), (0.50, 0.50)) + \
        _rows("c2", (0.021, 0.021), (0.6, 0.6))
    got = S.rank_screening(FIELD, scored, B_ROWS, seeds=SEEDS)
    assert [r["arm"] for r in got["ranking"]["ranked"]] == ["c2", "c1"]
    assert got["selection"]["arm"] == "c2"


def test_no_winner_is_forced():
    scored = _rows("c1", (0.05, 0.05), (0.4, 0.4)) + \
        _rows("c2", (0.05, 0.05), (0.4, 0.4))
    got = S.rank_screening(FIELD, scored, B_ROWS, seeds=SEEDS)
    assert got["selection"]["outcome"] == "NO_CANDIDATE_ADVANCES"


def test_a_partial_field_is_refused():
    scored = _rows("c1", (0.03, 0.03), (0.6, 0.6)) + \
        _rows("c2", (0.02,), (0.6,))
    with pytest.raises(S.SharedEvaluationError, match="partial"):
        S.rank_screening(FIELD, scored, B_ROWS, seeds=SEEDS)
