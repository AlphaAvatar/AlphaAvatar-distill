"""The control arm every D-series round measures against is DERIVED, not typed.

The defect these tests exist for: D1's design declared its incumbent as
`fe9683e6` / `c313d1b4`, which is C1's `attention.weight_proxy_v0` arm -- the
arm C1 measured and BEAT. C1 returned GO at +0.013725 against a SESOI of 0.010,
so the checkpoint that stands is C1's treatment at `53e30566`.

The error is bigger than its subject. C1's measured delta between those two
arms (+0.013725) EXCEEDS the SESOI (0.010) that D1's decision rule tests
against, so a candidate compared against the beaten arm inherits an effect
larger than the amount that decides -- it does not add noise, it manufactures a
GO. Four candidates would all have cleared a bar that was lowered by a
transcription.

Nothing caught it because both sides were prose and constants: the design typed
the identity, and every statement that B is "the frozen C1 treatment" was a
sentence rather than a comparison. These tests make it a comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]

from experiments.phase_d_series import incumbent as I  # noqa: E402

D1_DESIGN = "logs/stages/stage-1/phase_d1/plans/d1_design.json"


def _design() -> dict:
    return json.loads((REPO / D1_DESIGN).read_text())


class TestTheStandingIncumbentIsDerivedFromTheVerdict:
    """It follows from what C1 measured, so it moves when the record moves."""

    def test_c1_said_go_so_the_treatment_stands(self):
        verdict = I.c1_verdict(REPO)
        assert verdict["verdict"] == "GO"
        assert verdict["selects_arm"] == "treatment"
        standing = I.standing_incumbent(REPO)
        assert standing["c1_arm"] == "treatment"
        assert standing["impl_id"] == "attention.activation_importance_v1"

    def test_the_identity_is_the_measured_one_not_a_pinned_expectation(self):
        """Read from `c1_arm_identities.json`, which records observations.

        A pinned expectation and an observation agree until they do not, and
        what the bytes WERE is the observation.
        """
        measured = json.loads((REPO / I.C1_ARM_IDENTITIES_REL).read_text())
        standing = I.standing_incumbent(REPO)
        for field in I.IDENTITY_FIELDS:
            assert standing[field] == measured["treatment"][field]

    def test_all_four_identities_are_present(self):
        standing = I.standing_incumbent(REPO)
        for field in I.IDENTITY_FIELDS:
            assert len(standing[field]) == 64, field

    def test_an_unrecognised_verdict_selects_nothing(self, tmp_path):
        """It raises rather than defaulting to one arm.

        A default here is how an INCONCLUSIVE or a malformed verdict silently
        promotes a challenger: the mapping has no fallthrough.
        """
        run = tmp_path / I.C1_RUN_REL / "evidence"
        run.mkdir(parents=True)
        (run / "c1_decision.json").write_text(json.dumps({"verdict": "MAYBE"}))
        with pytest.raises(I.IncumbentUndetermined, match="not one of"):
            I.c1_verdict(tmp_path)

    def test_a_missing_record_raises_rather_than_falling_back(self, tmp_path):
        with pytest.raises(I.IncumbentUndetermined, match="is missing"):
            I.standing_incumbent(tmp_path)

    def test_a_no_go_would_leave_the_incumbent_arm_standing(self, tmp_path):
        """The derivation is a function of the verdict, exercised both ways.

        Without this the mapping is only ever evaluated at one key, so a rule
        that ignored the verdict entirely would pass every other test here.
        """
        src = json.loads((REPO / I.C1_ARM_IDENTITIES_REL).read_text())
        run = tmp_path / I.C1_RUN_REL / "evidence"
        run.mkdir(parents=True)
        (run / "c1_decision.json").write_text(json.dumps(
            {"verdict": "NO_GO", "delta": -0.001, "sesoi": 0.01}))
        (run / "c1_arm_identities.json").write_text(json.dumps(src))
        standing = I.standing_incumbent(tmp_path)
        assert standing["c1_arm"] == "incumbent"
        assert standing["impl_id"] == "attention.weight_proxy_v0"
        assert standing["artifact_digest"] == src["incumbent"]["artifact_digest"]


class TestEveryRoundsConstantsAgreeWithTheDerivation:
    """Closed rounds keep their own constants; they must not disagree.

    C2 and C3 are closed and their frozen constants correctly describe what they
    ran -- they are not rewritten to import this. What they may not do is
    DISAGREE with it, because then "which checkpoint is B" has two answers and
    the next round picks one by accident.
    """

    def test_c2s_four_frozen_b_constants_are_the_standing_identity(self):
        from experiments.phase_c2 import baseline as BL

        standing = I.standing_incumbent(REPO)
        assert BL.B_ARTIFACT_DIGEST == standing["artifact_digest"]
        assert BL.B_WEIGHTS_DIGEST == standing["weights_digest"]
        assert BL.B_SINGLE_SHARD_SHA256 == standing["single_shard_sha256"]
        assert BL.B_ARCH_SIGNATURE == standing["arch_signature"]

    def test_c2s_frozen_path_ends_in_the_operator_c1_promoted(self):
        from experiments.phase_c2 import baseline as BL

        standing = I.standing_incumbent(REPO)
        last_kind, last_impl, last_profile = BL.B_PATH[-1]
        assert last_kind == standing["kind"]
        assert last_impl == standing["impl_id"]
        assert last_profile == standing["profile_id"]

    def test_the_c3_gate_names_the_standing_identity(self):
        """Through the function C3's driver calls, not through its docstring.

        `expected_incumbent_digest()` is what stage E compares against; the
        module docstring's `53e30566...` is a label beside it. Asserting on the
        label would pass while the executed value drifted, which is the
        prose-instead-of-comparison failure this file exists for.
        """
        from experiments.phase_c3 import session as C3

        assert C3.expected_incumbent_digest() == \
            I.standing_incumbent(REPO)["artifact_digest"]

    def test_the_live_state_calls_it_the_c1_treatment(self):
        live = json.loads((REPO / "logs/state/current.json").read_text())
        flat = json.dumps(live)
        assert "B = frozen C1 treatment" in flat
        assert I.standing_incumbent(REPO)["c1_arm"] == "treatment"


class TestTheD1DesignDeclaresTheRightControl:
    """The test that would have caught it, and the one that keeps it caught."""

    def test_the_design_names_the_standing_incumbent(self):
        declared = _design()["incumbent"]
        differ = I.disagreements(declared, REPO)
        assert not differ, (
            "the D1 design declares a control arm that is not the standing "
            f"incumbent: {differ}. C1's measured delta between its two arms is "
            "+0.013725 and the D-series SESOI is 0.010, so a candidate "
            "measured against the wrong arm is credited with more than the "
            "amount the decision rule tests for.")

    def test_the_blocker_is_open_exactly_when_the_identity_disagrees(self):
        """Derived both ways, so neither can be satisfied by prose.

        A design that disagreed and reported no blocker would read as
        launchable; one that agreed and reported a blocker would refuse a
        correct launch forever.
        """
        import sys

        sys.path.insert(0, str(REPO / "scripts/autoinit"))
        import write_d1_design as w

        check = w.incumbent_identity_check()
        design = _design()
        assert check["status"] in ("AGREES", "DISAGREES", "UNRESOLVED")
        open_ = list(design["open_blockers"])
        if check["status"] == "AGREES":
            assert "incumbent identity" not in open_
        else:
            assert "incumbent identity" in open_

    def test_the_declared_identity_matches_what_the_writer_emits(self):
        """The constants and the document cannot drift apart silently."""
        import sys

        sys.path.insert(0, str(REPO / "scripts/autoinit"))
        import write_d1_design as w

        declared = _design()["incumbent"]
        assert declared["artifact_digest"] == w.INCUMBENT_DIGEST
        assert declared.get("state_id") == w.INCUMBENT_STATE_ID


class TestTheDesignHashCarriesNoRunState:
    """Opening a blocker must not redraw the behavioural seeds.

    `design_hash` is the preimage the screening and confirmation seeds are
    derived from. The launch-readiness claim used to interpolate the live
    blocker list into a HASHED field, so opening a FUNDING blocker moved the
    scientific identity and with it the seeds -- the same defect
    `_design_hash_covers` says it fixed for `budget`, surviving through a
    derived sentence.
    """

    def test_no_blocker_rendering_reaches_the_hash_preimage(self):
        """The rendered PHRASE, not the bare names.

        `evidence` is both a blocker name and an ordinary scientific key of the
        design, so searching for the names alone fails on a document that is
        entirely correct. What must not appear is the sentence the blocker list
        is rendered into.
        """
        import sys

        sys.path.insert(0, str(REPO / "scripts/autoinit"))
        import write_d1_design as w

        design = _design()
        preimage = json.dumps(w.scientific_preimage(design))
        rendered = [w._blocker_phrase(()), w._blocker_phrase(("evidence",)),
                    w._blocker_phrase(tuple(n for n, _ in w.BLOCKER_SPECS)),
                    w._blocker_phrase(tuple(design["open_blockers"]))]
        for phrase in rendered:
            assert phrase not in preimage, (
                f"{phrase!r} is run state and it reached the hashed scientific "
                "preimage; the behavioural seeds are derived from that hash, "
                "so opening a funding blocker would redraw them")

    def test_the_hash_does_not_move_when_a_blocker_opens(self):
        """Proved by CHANGING the blocker list, not by inspecting the document.

        The preimage search above is a necessary condition and not a
        sufficient one: a future field could encode the blocker state some
        other way. This builds the preimage twice over documents whose blocker
        lists differ and requires one hash.
        """
        import sys

        sys.path.insert(0, str(REPO / "scripts/autoinit"))
        import write_d1_design as w
        from aadistill.infrastructure.manifest import sha256_json

        design = _design()
        with_blocker = dict(design, open_blockers=["evidence", "funding "
                                                   "authorization"])
        without = dict(design, open_blockers=[])
        assert sha256_json(w.scientific_preimage(with_blocker)) == \
            sha256_json(w.scientific_preimage(without))

    def test_the_hash_is_the_preimages_and_only_the_preimages(self):
        import sys

        sys.path.insert(0, str(REPO / "scripts/autoinit"))
        import write_d1_design as w
        from aadistill.infrastructure.manifest import sha256_json

        design = _design()
        assert design["design_hash"] == sha256_json(
            w.scientific_preimage(design))
