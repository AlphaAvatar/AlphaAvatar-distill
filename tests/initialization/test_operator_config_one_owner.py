"""A replay must reconstruct an operator's INPUTS, not only check its output.

THE FAILURE, and it cost two paid subruns and $0.98. `BeamSearch` built its
hashed operator config from two private methods -- the scoring-position policy
and its content, and the distribution support -- and `materialize_fixed_path`,
which exists to REPLAY a path the beam produced, built its own:

    operator_config = {"n_calibration_items": n, **step.config}

That is not a smaller config. It is a different one. D1's search ran under a
Top-200 reference support and a supervised-target position policy, hashing to
`464cb782ea8095`; the replay's config hashed to `44136fa355b367`, which is
sha256 of `{}`. Every operator therefore reduced over the full vocabulary under
the incumbent policy and computed something the pinned digests were not
produced by.

Two things hid it:

* **The executor's own agreement check passed.** `apply_checked` compares the
  config's support declaration against the context's support object, and
  neither was set -- so both consistently said "full vocabulary", agreeing with
  each other and with nothing else.
* **The only symptom was a digest mismatch**, whose own message says "this is a
  replay mismatch, not a recoverable condition". That reading was wrong both
  times, and the second time it took 29 minutes of DEPTH to produce.

Two repairs, and these test both:

1. ONE OWNER. `planning/operator_config.py` builds the config and the hash, and
   both callers use it. The beam's behaviour is unchanged -- proven here by
   reproducing a historical hash -- and the duplication that could diverge is
   gone.
2. PINNED INPUTS. `FixedPathStep.expected_config_hash` is checked BEFORE the
   operator runs, and a disagreement raises `FixedPathConfigMismatch`, which is
   deliberately a different exception from `FixedPathDigestMismatch`: one says
   the mechanism did not reproduce the inputs (ordinary, repair it), the other
   says it did and the output still differs (material, stop).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aadistill.initialization.planning.operator_config import (
    EXECUTOR_DERIVED_CONFIG_KEYS,
    distribution_support_config,
    hashed_operator_config,
    operator_config_hash,
    position_policy_config,
)
from aadistill.initialization.scoring.positions import ALL_POSITIONS_V1
from aadistill.initialization.scoring.support import (
    FULL_VOCAB_V1, reference_topk_tail,
)

#: sha256 of `{}` -- what an empty hashed config hashes to, and what two paid
#: subruns ran under.
EMPTY_CONFIG_HASH = operator_config_hash({})


def _toy(need_items: bool):
    """A stand-in operator. `consumes_calibration` reads `.calibration`, so
    that is the attribute these set -- asked of the real predicate rather than
    of a remembered signature."""
    from aadistill.initialization.calibration.profiles import CalibrationNeed

    class _Toy:
        impl_id = "toy.calibrated_v0" if need_items else "toy.weight_only_v0"
        calibration = (CalibrationNeed.FORWARD_LOGITS if need_items
                       else CalibrationNeed.NONE)

    return _Toy()


def _Calibrated():
    return _toy(True)


def _WeightOnly():
    return _toy(False)


@pytest.fixture
def support():
    return reference_topk_tail(top_k=200)


class TestTheSupportReachesTheHashedConfig:

    def test_a_nonincumbent_support_is_declared(self, support):
        assert distribution_support_config(support) == {
            "distribution_support": support.as_dict()}

    def test_the_full_vocabulary_is_OMITTED_not_stated(self):
        """Omission is the historical contract, not tidiness: emitting an
        explicit default would move every committed `config_hash` and the
        `measurement_protocol_id`s beside them."""
        assert distribution_support_config(FULL_VOCAB_V1) == {}
        assert distribution_support_config(None) == {}

    def test_omitting_it_is_not_the_same_computation(self, support):
        """The claim the replay implicitly made, falsified."""
        with_support = operator_config_hash(hashed_operator_config(
            implementation=_WeightOnly(), policy=None, support=support,
            items=[]))
        without = operator_config_hash(hashed_operator_config(
            implementation=_WeightOnly(), policy=None, support=None, items=[]))
        assert with_support != without
        assert without == EMPTY_CONFIG_HASH


class TestThePositionPolicyReachesItToo:

    def test_a_weight_only_operator_declares_no_policy(self, support):
        """It has no mechanism by which positions could change its output, so
        branching it would manufacture byte-identical states."""
        assert position_policy_config(_WeightOnly(), ALL_POSITIONS_V1, []) == {}

    def test_the_incumbent_policy_is_omitted(self):
        assert position_policy_config(_Calibrated(), ALL_POSITIONS_V1, []) == {}

    def test_no_declaration_is_treated_as_the_incumbent(self):
        """`policy_config(None)` raises on `.policy_hash`, and a fixed path
        legitimately declares no policy."""
        assert position_policy_config(_Calibrated(), None, []) == {}


class TestNCalibrationItemsIsTheExecutorsAndNotACallersToDeclare:

    def test_it_is_derived_from_the_items(self):
        cfg = hashed_operator_config(implementation=_WeightOnly(), policy=None,
                                     support=None, items=[1, 2, 3])
        assert cfg["n_calibration_items"] == 3

    def test_a_caller_that_declares_it_is_refused(self):
        """"Last writer wins" let a step lie about the corpus it was running
        against, which is the one thing a plan must not be able to do."""
        with pytest.raises(ValueError) as exc:
            hashed_operator_config(
                implementation=_WeightOnly(), policy=None, support=None,
                items=[1, 2, 3], declared={"n_calibration_items": 99})
        assert "executor-owned" in str(exc.value)
        assert "n_calibration_items" in EXECUTOR_DERIVED_CONFIG_KEYS

    def test_the_hash_EXCLUDES_it(self):
        """The historical contract: it is a property of the corpus a run was
        given, so including it would make two otherwise identical expansions
        over different-sized mixtures incomparable."""
        a = operator_config_hash({"n_calibration_items": 3, "x": 1})
        b = operator_config_hash({"n_calibration_items": 999, "x": 1})
        assert a == b
        assert a != operator_config_hash({"x": 2})


class TestTheBeamStillComputesWhatItAlwaysComputed:
    """The rewiring must not move a single recorded hash. The beam is the
    authority for every committed `config_hash` in this repository."""

    def test_the_beam_uses_the_shared_owner(self):
        import ast
        from pathlib import Path

        from aadistill.initialization.planning import search

        src = Path(search.__file__).read_text()
        assert "hashed_operator_config(" in src
        assert "operator_config_hash(operator_config)" in src
        #: And no longer builds it itself.
        tree = ast.parse(src)
        methods = {n.name for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)}
        assert "_position_policy_config" not in methods, (
            "the beam still has its own position-policy contributor; two "
            "implementations of one identity disagree immediately")
        assert "_distribution_support_config" not in methods

    def test_the_fixed_path_uses_it_too(self):
        from pathlib import Path

        from aadistill.initialization.planning import fixed_path

        src = Path(fixed_path.__file__).read_text()
        body = ast.parse(src)
        #: `materialize_fixed_path` delegates to `_run_steps`, which is where
        #: the config is built, so the walk covers both rather than assuming
        #: which one holds it.
        executors = [n for n in ast.walk(body)
                     if isinstance(n, ast.FunctionDef)
                     and n.name in ("materialize_fixed_path", "_run_steps")]
        assert executors, "neither executor function is present"
        called = {getattr(c.func, "id", getattr(c.func, "attr", ""))
                  for fn in executors
                  for c in ast.walk(fn) if isinstance(c, ast.Call)}
        assert "hashed_operator_config" in called
        #: AST, not a substring: an earlier version of this assertion matched
        #: the explanatory COMMENT that quotes the old call.
        assert "step_operator_config" not in called, (
            "the executor still derives its own config, which is the defect")


class TestTheInputPinIsCheckedBeforeTheOperatorRuns:

    def test_the_step_carries_it_and_serializes_it_only_when_set(self):
        from aadistill.initialization.planning.fixed_path import FixedPathStep

        bare = FixedPathStep(impl_id="i", profile_id="p")
        assert bare.expected_config_hash is None
        assert "expected_config_hash" not in bare.as_dict(), (
            "a step that pins nothing must serialize as it did before this "
            "field existed, or every historical spec_hash moves")

        pinned = FixedPathStep(impl_id="i", profile_id="p",
                               expected_config_hash="a" * 64)
        assert pinned.as_dict()["expected_config_hash"] == "a" * 64

    def test_the_two_mismatches_are_different_exceptions(self):
        """One is ordinary and gets repaired; the other is material and stops.
        Conflating them is what made two paid subruns read as findings."""
        from aadistill.initialization.planning.fixed_path import (
            FixedPathConfigMismatch, FixedPathDigestMismatch, FixedPathError,
        )

        assert FixedPathConfigMismatch is not FixedPathDigestMismatch
        assert issubclass(FixedPathConfigMismatch, FixedPathError)
        assert not issubclass(FixedPathConfigMismatch, FixedPathDigestMismatch)

    def test_it_carries_the_evidence_a_reader_needs(self):
        from aadistill.initialization.planning.fixed_path import (
            FixedPathConfigMismatch,
        )

        exc = FixedPathConfigMismatch("msg", step_index=2, label="DEPTH(p)",
                                      expected="a" * 64, actual="b" * 64)
        assert exc.step_index == 2 and exc.label == "DEPTH(p)"
        assert exc.expected == "a" * 64 and exc.actual == "b" * 64

    def test_the_executor_checks_it_before_planning_the_operator(self):
        """Order matters for money: the comparison is a dict lookup and the
        operator it precedes took 1732 s."""
        from pathlib import Path

        from aadistill.initialization.planning import fixed_path

        src = Path(fixed_path.__file__).read_text()
        assert src.index("FixedPathConfigMismatch(") < src.index(
            "outcome = impl.execute(ctx)")


class TestTheSpecCarriesTheProtocolWithoutMovingHistoricalHashes:

    def test_the_incumbent_is_omitted_from_the_serialization(self):
        from pathlib import Path

        from aadistill.initialization.planning import fixed_path

        src = Path(fixed_path.__file__).read_text()
        #: Both guarded by an incumbent check inside `as_dict`.
        assert "self.distribution_support.is_full_vocab" in src
        assert "self.position_policy is None" in src
