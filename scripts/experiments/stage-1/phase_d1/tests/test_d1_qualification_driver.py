"""Regression for the three defects paid subrun s2 exposed.

s2 cost $0.9006, passed its hard gate, and then failed arm B after 1,384 paid
seconds. Each defect below is the specific thing that went wrong, not a general
sweep: the point is that these three cannot recur, at $0.

1. A3's fixed path pins an intermediate to the INCUMBENT's artifact. Arm B
   changes the position policy, so it builds something else by construction, and
   the pin stopped it -- correctly. Reusing a pinned spec to ask an unpinned
   question was the bug.
2. Arm B completed three steps and recorded their digests, and the record kept
   `steps: []` because the raise preceded the return. Expensive completed work
   must survive a later failure.
3. Arms A and B differ in TWO knobs, so the moved FFN selection s2 found could
   be attributed to neither. A third arm identifies it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/pod"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

import d1_qualification_driver as drv  # noqa: E402


def _a3_spec():
    """A3's real fixed path. Four process-global registries must be filled first.

    All four are empty in a fresh interpreter and registering is explicit in this
    project -- the missing fourth cost paid subrun s1 $0.1098.
    """
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from experiments.calibration import register_builtin_profiles
    from experiments.phase_c2.search_space import register_c2_operators

    register_builtin_adapters()
    register_builtin_profiles()
    register_c2_operators()
    activation_importance.register()

    from experiments.phase_a3 import a3_session

    return a3_session.path_spec(workdir_device="cpu")


class TestOnlyTheArmThatClaimsTheIncumbentKeepsThePins:
    """`unpinned` clears every expected-artifact pin and renames the path."""

    def test_it_clears_pins_and_carries_every_other_field(self):
        """Against A3's REAL spec, not a spec I invented.

        The first version of this test built its own `ArchSpec` from keyword
        arguments that do not exist, which is a test of a shape rather than of
        the code. A3's spec is the actual input the driver hands `unpinned`, and
        it carries exactly one real pin -- the pre-ATTENTION parent that stopped
        s2.
        """
        pytest.importorskip("torch")
        spec = _a3_spec()

        pins = [s.expected_artifact_digest for s in spec.steps]
        assert sum(p is not None for p in pins) == 1, (
            f"A3 pins exactly the pre-ATTENTION parent; found {pins}")

        out = drv.unpinned(spec)

        assert all(s.expected_artifact_digest is None for s in out.steps)
        assert out.path_id == f"{spec.path_id}.unpinned", (
            "an unpinned path is not the pinned one and must not be recorded "
            "under its id")
        #: `dataclasses.replace`, so a field added later is carried rather than
        #: dropped by a hand-written reconstruction.
        assert (out.family, out.root_repo_id, out.root_revision, out.seed,
                out.target_spec) == \
            (spec.family, spec.root_repo_id, spec.root_revision, spec.seed,
             spec.target_spec)
        assert [s.impl_id for s in out.steps] == [s.impl_id for s in spec.steps]
        assert out.spec_hash != spec.spec_hash

    def test_the_driver_unpins_exactly_the_non_incumbent_arms(self):
        """The decision is `expected_final is None`, read off the source."""
        source = (REPO / "scripts/pod/d1_qualification_driver.py").read_text()
        assert "    if expected_final is None:\n" in source
        assert "        spec = unpinned(spec)" in source


class TestCompletedStepsSurviveALaterFailure:
    """`PartialPath` carries what the arm finished before it failed."""

    def test_it_carries_the_steps_and_the_failure(self):
        exc = drv.PartialPath("B: boom", partial={
            "arm": "B", "n_steps": 3, "steps": [{"index": 0}, {"index": 1},
                                                {"index": 2}],
            "failed_at_step": 3, "failure": "FixedPathDigestMismatch: ..."})
        assert exc.partial["n_steps"] == 3
        assert len(exc.partial["steps"]) == 3
        assert "FixedPathDigestMismatch" in exc.partial["failure"]
        #: It is still a QualificationError, so the driver's outer handler
        #: records a FAILED status rather than treating it as a new kind of
        #: control flow.
        assert isinstance(exc, drv.QualificationError)

    def test_the_arm_helper_files_the_partial_before_re_raising(self):
        source = (REPO / "scripts/pod/d1_qualification_driver.py").read_text()
        body = source[source.index("        def arm("):]
        body = body[:body.index("\n        #: A -- the hard gate")]
        assert "except PartialPath as exc:" in body
        assert "record[key] = exc.partial" in body
        i, j = body.index("record[key] = exc.partial"), body.index("raise", body.index("except PartialPath"))
        assert i < j, "the partial must be filed BEFORE the failure propagates"


class TestAMovedSelectionIsAttributedToAKnob:
    """A difference is evidence; an UNEXPLAINED difference is the failure."""

    @staticmethod
    def _step(selection):
        return {"index": 0, "impl_id": "ffn.activation_importance_v0",
                "artifact_digest": "d" * 64, "selection": selection}

    def test_no_difference(self):
        got = drv._attribute(self._step([1]), self._step([1]), self._step([1]))
        assert "no difference" in got

    def test_the_policy_moved_it(self):
        """Batch size alone reproduced the incumbent, so the policy did it."""
        got = drv._attribute(self._step([1]), self._step([2]), self._step([1]))
        assert "THE POSITION POLICY moved it" in got

    def test_the_batch_size_moved_it(self):
        """Batch size alone already reproduced B, so the policy is not implicated."""
        got = drv._attribute(self._step([1]), self._step([2]), self._step([2]))
        assert "THE CALIBRATION BATCH SIZE moved it" in got

    def test_both_or_an_interaction(self):
        got = drv._attribute(self._step([1]), self._step([2]), self._step([3]))
        assert "BOTH KNOBS, or an interaction" in got

    def test_a_missing_third_arm_says_unattributed_rather_than_guessing(self):
        got = drv._attribute(self._step([1]), self._step([2]), None)
        assert got.startswith("UNATTRIBUTED")
        assert "confounded" in got

    def test_the_comparison_reports_attributions_per_moved_operator(self):
        a = {"steps": [self._step([1])], "final_artifact_digest": "x" * 64}
        b = {"steps": [self._step([2])], "final_artifact_digest": "y" * 64}
        c = {"steps": [self._step([1])]}
        out = drv.compare_selections(a, b, c)

        assert out["n_moved"] == 1
        assert out["final_artifacts_differ"] is True
        assert "THE POSITION POLICY moved it" in \
            out["attributions"]["ffn.activation_importance_v0"]
        #: And without the third arm it degrades to UNATTRIBUTED rather than
        #: silently reporting a cause it cannot know.
        assert drv.compare_selections(a, b)["attributions"][
            "ffn.activation_importance_v0"].startswith("UNATTRIBUTED")
