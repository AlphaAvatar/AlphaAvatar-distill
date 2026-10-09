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

from stages.phase_d1 import d1_qualification_driver as drv  # noqa: E402


def _a3_spec():
    """A3's real fixed path. Four process-global registries must be filled first.

    All four are empty in a fresh interpreter and registering is explicit in this
    project -- the missing fourth cost paid subrun s1 $0.1098.
    """
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    from shared.calibration import register_builtin_profiles
    from stages.phase_c2.search_space import register_c2_operators

    register_builtin_adapters()
    register_builtin_profiles()
    register_c2_operators()
    activation_importance.register()

    from stages.phase_a3 import a3_session

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
        source = (REPO / "scripts/stages/stage-1/phase_d1/d1_qualification_driver.py").read_text()
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
        source = (REPO / "scripts/stages/stage-1/phase_d1/d1_qualification_driver.py").read_text()
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
        #: Fingerprints, not copies: the arms own their selection lists, and a
        #: per-process-salted `hash()` would make the record incomparable.
        row = out["steps"][0]
        assert len(row["incumbent_selection_sha256"]) == 64
        assert row["incumbent_selection_sha256"] != \
            row["target_aware_selection_sha256"]
        assert row["incumbent_selection_sha256"] == \
            row["batch_only_selection_sha256"]
        assert "incumbent_selection" not in row
        #: And without the third arm it degrades to UNATTRIBUTED rather than
        #: silently reporting a cause it cannot know.
        assert drv.compare_selections(a, b)["attributions"][
            "ffn.activation_importance_v0"].startswith("UNATTRIBUTED")


class TestACompletedArmReleasesWhatNothingReads:
    """Intermediates go; the final and every digest stay.

    Three arms retain ~11.75 GiB each, ~10.6 GiB of it intermediate, so a
    three-arm run would want ~65 GiB of a 60 GiB disk -- and the arm that failed
    would be the LAST one, after the first two had already been paid for.
    """

    @staticmethod
    def _arm(tmp_path):
        steps = []
        for i, name in enumerate(("00_depth", "01_ffn", "02_width", "03_attn")):
            d = tmp_path / name
            d.mkdir()
            (d / "model.safetensors").write_bytes(b"x" * (100 * (i + 1)))
            steps.append({"index": i, "impl_id": f"op{i}",
                          "artifact_digest": f"{i}" * 64,
                          "checkpoint_path": str(d)})
        return {"arm": "A", "steps": steps,
                "final_checkpoint_path": str(tmp_path / "03_attn")}

    def test_it_keeps_the_final_and_frees_the_rest(self, tmp_path):
        import types

        arm = self._arm(tmp_path)
        journal = types.SimpleNamespace(event=lambda **k: None)
        out = drv.release_intermediates(arm, journal=journal)

        assert out["released"] == ["00_depth", "01_ffn", "02_width"]
        assert out["freed_bytes"] == 100 + 200 + 300
        assert not (tmp_path / "00_depth").exists()
        #: The one stage D loads must survive.
        assert (tmp_path / "03_attn" / "model.safetensors").is_file()

    def test_every_digest_survives_the_release(self, tmp_path):
        """Identities are the evidence; the bytes nobody reads are not."""
        import types

        arm = self._arm(tmp_path)
        before = [s["artifact_digest"] for s in arm["steps"]]
        drv.release_intermediates(arm, journal=types.SimpleNamespace(
            event=lambda **k: None))

        assert [s["artifact_digest"] for s in arm["steps"]] == before
        assert [s.get("checkpoint_released") for s in arm["steps"]] == \
            [True, True, True, None], (
                "each released step says so, and the kept one does not claim to")
        #: And the path it USED is still recorded -- a released artifact is
        #: traceable, not erased from the record.
        assert all(s["checkpoint_path"] for s in arm["steps"])


class TestPresenceIsNotSuccess:
    """A gate that certified a failed run, and the two beside it that could.

    s3's `D_state_eval` was a non-empty list of two FAILURE entries. The check
    tested `not d` for absence and `within_derived_budget is False` for a
    violation. A measurement that was never taken trips neither: the field is
    simply absent. So the closeout called s3 PASSED and wrote the record that
    tells the rest of the repository the owed GPU validation had run, while
    stage D had produced nothing at all.
    """

    @staticmethod
    def _complete():
        """A record that legitimately passes, as the baseline to break."""
        return {
            "status": "COMPLETE",
            "bound_protocol_incumbent": {"suite_content_sha256": "a" * 64,
                                         "measurement_protocol_id": "p1"},
            "bound_protocol_target_aware": {"suite_content_sha256": "a" * 64,
                                            "measurement_protocol_id": "p2"},
            "A_incumbent": {"reconstructed": True, "seconds": 1.0,
                            "final_artifact_digest": "f" * 64,
                            "expected_final_artifact_digest": "f" * 64},
            "B_target_aware": {"final_artifact_digest": "b" * 64,
                               "seconds": 2.0, "batch_size": 3},
            "C_selection_comparison": {"steps": [{"impl_id": "op",
                                                  "selection_differs": False}],
                                       "n_moved": 0},
            "D_state_eval": [{"label": "incumbent", "peak_memory_bytes": 1,
                              "within_derived_budget": True, "seconds": 1.0}],
        }

    def _verdict(self, record):
        from stages.phase_d1 import d1_qualification_closeout as co
        return co.verdict_of(record)

    def test_the_baseline_passes(self):
        """Otherwise the breakages below prove nothing."""
        verdict, _ = self._verdict(self._complete())
        assert verdict == "PASSED"

    def test_a_list_of_D_failures_does_not_pass(self):
        """THE REGRESSION. This exact shape was booked as PASSED."""
        record = self._complete()
        record["D_state_eval"] = [
            {"label": "incumbent", "failed": "TypeError: evaluate() missing 1 "
                                             "required positional argument"},
            {"label": "target_aware", "failed": "TypeError: evaluate() missing "
                                                "1 required positional argument"}]
        verdict, detail = self._verdict(record)

        assert verdict == "INCOMPLETE"
        assert any("D failed for 'incumbent'" in u for u in detail["unmet"])
        assert any("no state-eval memory measurement at all" in u
                   for u in detail["unmet"])

    def test_a_partial_B_arm_does_not_pass(self):
        """`PartialPath` makes the key present and truthy without completing."""
        record = self._complete()
        record["B_target_aware"] = {"arm": "B", "n_steps": 3, "failed_at_step": 3,
                                    "steps": [{}, {}, {}],
                                    "failure": "FixedPathDigestMismatch: ..."}
        verdict, detail = self._verdict(record)

        assert verdict == "INCOMPLETE"
        assert any("B did not complete" in u for u in detail["unmet"])

    def test_a_comparison_with_no_steps_does_not_pass(self):
        record = self._complete()
        record["C_selection_comparison"] = {"n_moved": 0, "steps": []}
        verdict, detail = self._verdict(record)

        assert verdict == "INCOMPLETE"
        assert any("C compared nothing" in u for u in detail["unmet"])

    def test_a_failed_D_beside_a_good_one_still_fails(self):
        """One measurement does not excuse the arm that produced none."""
        record = self._complete()
        record["D_state_eval"] = record["D_state_eval"] + [
            {"label": "target_aware", "failed": "CUDA out of memory"}]
        verdict, detail = self._verdict(record)

        assert verdict == "INCOMPLETE"
        assert any("target_aware" in u and "CUDA out of memory" in u
                   for u in detail["unmet"])


class TestStageDMeasuresWithoutRepeatingTheArms:
    """Which protocol is state-evaluated, on whose artifact.

    s3 lost stage D to two contract defects after paying for all three arms. The
    repair must be able to measure D alone: repeating 85 minutes of arms whose
    digests are already recorded is paying twice for one measurement. What makes
    that sound is that state-eval memory is a property of the geometry and the
    protocol's batch plan, not of the weights -- and the fallback is recorded.
    """

    PROTOCOLS = (("incumbent", {"evaluator": "E1"}),
                 ("target_aware", {"evaluator": "E2"}))

    def test_each_protocol_prefers_its_own_arm(self):
        record = {"A_incumbent": {"final_artifact_digest": "a" * 64},
                  "B_target_aware": {"final_artifact_digest": "b" * 64}}
        plan, failures = drv.state_eval_plan(record, self.PROTOCOLS)

        assert not failures
        assert [(p[0], p[3], p[4]) for p in plan] == [
            ("incumbent", "A_incumbent", True),
            ("target_aware", "B_target_aware", True)]

    def test_with_only_arm_A_both_protocols_still_measure(self):
        """The repair subrun's shape: one materialization, two protocols."""
        record = {"A_incumbent": {"final_artifact_digest": "a" * 64}}
        plan, failures = drv.state_eval_plan(record, self.PROTOCOLS)

        assert not failures
        assert [(p[0], p[3], p[4]) for p in plan] == [
            ("incumbent", "A_incumbent", True),
            ("target_aware", "A_incumbent", False)]
        #: The evaluator is still the TARGET-AWARE protocol's own -- only the
        #: artifact is borrowed. Measuring at the incumbent's batch plan and
        #: calling it the target-aware figure is the mistake this guards.
        assert plan[1][2]["evaluator"] == "E2"

    def test_a_partial_arm_is_not_a_materialized_artifact(self):
        """A `PartialPath` record is truthy and has no final digest."""
        record = {"A_incumbent": {"arm": "A", "steps": [{}], "n_steps": 1,
                                  "failure": "boom"}}
        plan, failures = drv.state_eval_plan(record, self.PROTOCOLS)

        assert plan == []
        assert [f["label"] for f in failures] == ["incumbent", "target_aware"]
        assert all("nothing to state-evaluate" in f["failed"] for f in failures)

    def test_no_arms_at_all_fails_rather_than_raising(self):
        """An empty record must not KeyError in the middle of a paid run."""
        plan, failures = drv.state_eval_plan({}, self.PROTOCOLS)
        assert plan == []
        assert len(failures) == 2


class TestTheArmsSelectorResolvesBeforeAnyStage:
    """`$0.0632` for a variable defined between the arm that used it and the next.

    The selector was computed between arm A and arm B, so arm A's
    `if "A" in wanted` raised UnboundLocalError 15 seconds into a paid pod. The
    `--check-only` preflight could not catch it because it returns before the
    arms block -- so the fix is not a lint for one name, it is resolving the
    configuration BEFORE the early return, where the free preflight executes it.
    """

    def test_the_selector_is_resolved_before_the_first_arm_call(self):
        source = (REPO / "scripts/stages/stage-1/phase_d1/d1_qualification_driver.py").read_text()
        resolved = source.index('    wanted = {a.strip() for a in')
        first_use = source.index('if "A" in wanted:')
        early_return = source.index('record["status"] = "CHECK_ONLY_OK"')

        assert resolved < first_use, (
            "the arms selector must be resolved before any arm reads it")
        assert resolved < early_return, (
            "and before `--check-only` returns, so the $0 preflight executes "
            "it -- otherwise this whole class of defect is invisible until a "
            "pod is already billing")

    def test_an_unknown_arm_name_is_refused(self):
        """A typo would silently run fewer arms and look like a clean result."""
        import io
        import contextlib

        buf = io.StringIO()
        with pytest.raises(SystemExit) as caught, \
                contextlib.redirect_stderr(buf):
            drv.main(["--out", "/tmp/unused-d1-qual-argcheck",
                      "--arms", "A,Battr"])
        assert "Battr" in str(caught.value) or "Battr" in buf.getvalue()
