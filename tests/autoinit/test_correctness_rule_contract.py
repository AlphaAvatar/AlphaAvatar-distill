"""`CorrectnessRule` must mean the same thing to every component that reads it.

The type advertised `correct_implies_usable` and one component honoured it. The
scorer applied the caller's rule, produced `correct=True, usable=False` under a
rule that declares correctness independent -- and then `validate_scored_rows`
rejected that row unconditionally, quoting the rule's OWN note as the reason it
was illegal. The note said correctness is independent of usability.

An option that only the first conditional respects is worse than no option: it
produces apparently valid rows and then refuses them, and the same assumption
was hidden twice more downstream, in both pooled aggregations.

What this pins:

* the strict policy, unchanged in definition, row shape and counts -- because
  the C1 battery's numbers must not move;
* an independent policy, which must now survive scoring AND validation;
* the two aggregations that cannot support it declaring so and refusing BEFORE
  aggregating, rather than emitting numbers nobody can interpret;
* the conditional metric using the JOINT count where it is supported, since
  `correct / usable` is not a conditional probability when the two are not
  nested -- it can exceed 1, and does in `test_the_old_formula_would_exceed_one`.

A restored unconditional `correct => usable` check must turn
`test_an_independent_rule_survives_validation` red.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.planning.recovery import (  # noqa: E402
    CorrectnessRule,
    JointCountSeedAggregation,
    ScorableAwareSeedAggregation,
    ScoringContractError,
    SeedAggregation,
    score_recovery_row,
    validate_scored_rows,
)
from experiments.recovery_policy import CORRECT_IN_USABLE_ROLLOUT as STRICT  # noqa: E402

#: A caller-supplied rule of the exact shape the maintainer named.
INDEPENDENT = CorrectnessRule(
    rule_id="independent_correctness@test",
    decide=lambda scorable, usable, scorer_correct: scorable and scorer_correct,
    note="correctness is scored independently of usability",
    correct_implies_usable=False,
)

#: The disputed input: scorable, scored correct, and NOT usable.
DISPUTED = dict(scorable=True, usable=False, scorer_correct=True)


# --- the strict policy is untouched -----------------------------------------

class TestTheCurrentC1PolicyIsUnchanged:
    def test_its_definition_still_requires_usability(self):
        row = score_recovery_row(**DISPUTED, rule=STRICT)
        assert row["correct"] is False
        assert row["correct_but_unusable"] is True

    def test_its_row_shape_is_byte_for_byte_what_it_was(self):
        """The joint field must NOT appear under a strict rule: it equals
        `correct` by construction there, and adding it would change every
        record this project has already written."""
        row = score_recovery_row(scorable=True, usable=True, scorer_correct=True,
                                 rule=STRICT)
        assert sorted(row) == ["correct", "correct_but_unusable", "scorable",
                               "scorer_correct", "usable"]
        assert "correct_and_usable" not in row

    def test_its_validator_counts_are_unchanged(self):
        rows = [score_recovery_row(scorable=True, usable=True, scorer_correct=c,
                                   rule=STRICT) for c in (True, False)]
        counts = validate_scored_rows(rows, rule=STRICT)
        assert sorted(counts) == ["correct", "correct_but_unusable", "n",
                                  "scorable", "usable"]
        assert counts == {"n": 2, "usable": 2, "correct": 1, "scorable": 2,
                          "correct_but_unusable": 0}

    def test_it_still_refuses_a_row_that_breaks_its_own_invariant(self):
        """A hand-made row claiming correct-without-usable is still illegal
        under the strict rule. Honouring the option must not disable the
        invariant for the policy that declares it."""
        with pytest.raises(ScoringContractError, match="correct => usable"):
            validate_scored_rows(
                [{"correct": True, "usable": False, "scorable": True}],
                rule=STRICT)

    def test_no_rule_at_all_still_means_strict(self):
        """Every existing caller passes no rule to the validator. Silence must
        keep meaning what it has always meant."""
        with pytest.raises(ScoringContractError, match="correct => usable"):
            validate_scored_rows([{"correct": True, "usable": False}])


# --- an independent policy now works end to end -----------------------------

class TestAnIndependentPolicyIsHonoured:
    def test_the_scorer_and_the_validator_agree(self):
        """THE DEFECT. The scorer produced this row and the validator refused
        it, citing the rule's own note as the reason."""
        row = score_recovery_row(**DISPUTED, rule=INDEPENDENT)
        assert row["correct"] is True
        counts = validate_scored_rows([row], rule=INDEPENDENT)   # must not raise
        assert counts["correct"] == 1
        assert counts["usable"] == 0

    def test_an_independent_rule_survives_validation(self):
        """The regression a restored unconditional check must turn red."""
        rows = [score_recovery_row(**DISPUTED, rule=INDEPENDENT)]
        assert validate_scored_rows(rows, rule=INDEPENDENT)["n"] == 1

    def test_the_row_carries_the_joint_fact_it_cannot_otherwise_recover(self):
        assert score_recovery_row(**DISPUTED,
                                  rule=INDEPENDENT)["correct_and_usable"] is False
        assert score_recovery_row(scorable=True, usable=True, scorer_correct=True,
                                  rule=INDEPENDENT)["correct_and_usable"] is True

    def test_the_validator_reports_the_joint_count(self):
        rows = [score_recovery_row(scorable=True, usable=u, scorer_correct=True,
                                   rule=INDEPENDENT) for u in (True, False, True)]
        counts = validate_scored_rows(rows, rule=INDEPENDENT)
        assert counts["correct"] == 3
        assert counts["usable"] == 2
        assert counts["correct_and_usable"] == 2


# --- an incompatible aggregation declares itself and refuses first ----------

class TestAnIncompatibleAggregationRefusesBeforeExecuting:
    @pytest.mark.parametrize("agg", [SeedAggregation(),
                                     ScorableAwareSeedAggregation()])
    def test_it_declares_that_it_cannot(self, agg):
        assert agg.supports_independent_correctness is False

    @pytest.mark.parametrize("agg", [SeedAggregation(),
                                     ScorableAwareSeedAggregation()])
    def test_it_refuses_before_producing_a_number(self, agg):
        """Not after. A pooled result that nobody can interpret, emitted and
        then questioned, is how an unsupported combination becomes a finding."""
        with pytest.raises(ScoringContractError,
                           match="assumes correct => usable"):
            agg.pool([{"seed": 1, "n": 10, "usable": 5, "correct": 7,
                       "n_scorable": 10, "usable_scorable": 5}], rule=INDEPENDENT)

    @pytest.mark.parametrize("agg", [SeedAggregation(),
                                     ScorableAwareSeedAggregation()])
    def test_it_still_pools_normally_under_a_strict_rule_and_with_none(self, agg):
        rows = [{"seed": 1, "n": 10, "usable": 6, "correct": 3,
                 "n_scorable": 10, "usable_scorable": 6}]
        assert agg.pool(rows, rule=STRICT)["correct"] == 3
        assert agg.pool(rows)["correct"] == 3


# --- the conditional metric, where it IS supported --------------------------

class TestConditionalCorrectnessUsesTheJointCount:
    SEEDS = [{"seed": 1, "n": 10, "usable": 5, "correct": 7, "correct_and_usable": 4},
             {"seed": 2, "n": 10, "usable": 6, "correct": 8, "correct_and_usable": 5}]

    def test_the_numerator_is_correct_AND_usable(self):
        out = JointCountSeedAggregation().pool(self.SEEDS, rule=INDEPENDENT)
        assert out["correct_and_usable"] == 9
        assert out["usable"] == 11
        assert out["correct_given_usable"] == pytest.approx(9 / 11)

    def test_the_old_formula_would_exceed_one(self):
        """Why the joint count is required rather than nice to have: dividing
        ALL correct answers by usable ones gives 15/11 here -- a conditional
        'probability' of 1.36."""
        out = JointCountSeedAggregation().pool(self.SEEDS, rule=INDEPENDENT)
        assert out["correct"] / out["usable"] > 1.0
        assert out["correct_given_usable"] <= 1.0

    def test_a_missing_joint_count_is_refused_not_invented(self):
        """It cannot be recovered from the marginals, so a caller that does not
        have it must be told, not given a plausible number."""
        with pytest.raises(ScoringContractError, match="correct_and_usable"):
            JointCountSeedAggregation().pool(
                [{"seed": 1, "n": 10, "usable": 5, "correct": 7}], rule=INDEPENDENT)

    def test_a_joint_count_above_either_marginal_is_refused(self):
        with pytest.raises(ValueError, match="cannot exceed either marginal"):
            JointCountSeedAggregation().pool(
                [{"seed": 1, "n": 10, "usable": 5, "correct": 7,
                  "correct_and_usable": 6}], rule=INDEPENDENT)

    def test_it_accepts_a_strict_rule_too(self):
        """The joint count is correct under either policy; it is the marginal
        shortcut that is only valid under one."""
        out = JointCountSeedAggregation().pool(
            [{"seed": 1, "n": 10, "usable": 6, "correct": 3,
              "correct_and_usable": 3}], rule=STRICT)
        assert out["correct_given_usable"] == pytest.approx(0.5)

    def test_nothing_usable_leaves_it_undefined_rather_than_zero(self):
        out = JointCountSeedAggregation().pool(
            [{"seed": 1, "n": 10, "usable": 0, "correct": 4,
              "correct_and_usable": 0}], rule=INDEPENDENT)
        assert out["correct_given_usable"] is None


# --- the historical numbers did not move ------------------------------------

def test_the_covered_behavior_v0_equivalence_still_holds():
    """Read from the record rather than asserted: 570 frozen Phase-A samples
    re-scored through the pre- and post-migration trees."""
    import json

    eq = json.loads((REPO / "logs/maintenance/inventories/architecture_scoring_equivalence.json").read_text())
    assert eq["all_scores_identical"] is True
    assert eq["total_samples"] == 570
