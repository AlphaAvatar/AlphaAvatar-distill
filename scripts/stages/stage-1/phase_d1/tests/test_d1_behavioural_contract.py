"""D1's behavioural rungs: the arms, the seeds, the battery, the permissions.

Everything here is derived from a frozen record and checked against the bytes
it names, because the one thing this session must not do is measure a field
nobody authorized. Three of these tests exist because of specific failures:

* **The battery was reported as a blocker it is not.** A reader followed
  `inputs.evidence_capacity`, found `batteries_remaining: 0` binding on
  MATH-500, and stopped -- while `evidence.status` said CLOSED and
  `open_blockers` was empty in the same document. So `battery_role` checks the
  REALIZED BYTES against the manifest rather than any claim about them.
* **The seeds are not in the preregistration.** C0 fixes a count and requires
  freshness and is explicit that the values are "DELIBERATELY NOT SET HERE".
  They must be materialized before any candidate result exists, which is what
  `derive_seeds` does; a test that assumed a stored list would have passed on
  a document that has none.
* **A class-level `POLICY` is decoration.** `action_policy` is the field the
  base reads, and an artifact that set only the class attribute serialized
  under `deny_all` and refused itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from stages.phase_d1 import behavioural as B  # noqa: E402
from stages.phase_d1.behavioural_governance import (  # noqa: E402
    D1_BEHAVIOURAL_POLICY,
    D1BehaviouralAuthorization,
    D1BehaviouralRefused,
)


def _auth(**over):
    base = dict(
        authorization_id="d1.behavioural.test", granted_utc="t",
        granted_by="test", plan_id="p", plan_hash="h", expected_usd=1.0,
        hard_cap_usd=2.0, authorized_stages=(), stage_conditions={},
        scope_note="n", rung="screening", contract_hash="c" * 64)
    base.update(over)
    return D1BehaviouralAuthorization(**base)


class TestTheControlArmMustBeTheOneThatActuallyStands:
    """The refusal that found a real defect, kept armed after it was fixed.

    Until 2026-10-08 the design stated its control as `fe9683e6` / `c313d1b4`,
    which is C1's `attention.weight_proxy_v0` arm -- the arm C1 measured and
    BEAT. C1 returned GO at +0.013725 against a SESOI of 0.010, so the
    checkpoint that stands is C1's treatment at `53e30566`, and the delta
    EXCEEDS the amount the decision rule tests for: the error could manufacture
    a GO rather than merely add noise. Corrected on the maintainer decision of
    2026-10-08; owner of the finding
    `logs/stages/stage-1/phase_d1/analyses/d1_control_arm_identity.json`.

    `arms()` still refuses a design that disagrees with the derivation, and
    these tests still drive that refusal, because the design is a FILE and the
    derivation is code: an edited design, or a later round that promoted
    something else without this document following, must still be refused.
    """

    def test_the_live_design_agrees_with_the_derivation(self):
        from stages.d_series.incumbent import disagreements

        declared = json.loads((REPO / B.DESIGN_REL).read_text())["incumbent"]
        assert disagreements(declared, REPO) == [], (
            "the design's control arm is not the standing incumbent; "
            "see d1_control_arm_identity.json")
        #: And the field assembles, which is the other half: a guard that kept
        #: refusing after its cause was removed would block every launch.
        assert len(B.arms(REPO)) == 5

    def test_a_design_naming_the_beaten_arm_is_refused(self, tmp_path,
                                                        monkeypatch):
        """Non-vacuity, driven through the EXACT historical defect.

        The design is copied with `c313d1b4...` -- C1's beaten arm, the value
        that was really there -- put back, and the field must refuse to
        assemble. Without this the refusal is unexercised the moment the design
        is correct, which is precisely when it stops being tested and starts
        being decoration.
        """
        beaten = ("c313d1b4081b9a3b410dddf7a29ebcaad8dd0759179d51e1d761238c"
                  "1743a2a6")
        doc = json.loads((REPO / B.DESIGN_REL).read_text())
        doc["incumbent"]["artifact_digest"] = beaten
        fake = tmp_path / "d1_design.json"
        fake.write_text(json.dumps(doc))
        monkeypatch.setattr(B, "DESIGN_REL", str(fake))
        with pytest.raises(B.D1BehaviouralError,
                           match="not the standing incumbent") as exc:
            B.arms(REPO)
        #: The refusal names the DECISION, not just the mismatch: which
        #: checkpoint B is moves the arms and the seeds.
        assert "maintainer decision" in str(exc.value)


class TestTheArmsAreTheFrozenFieldAndNothingElse:

    def test_five_arms_four_candidates_one_incumbent(self):
        arms = B.arms(REPO)
        assert len(arms) == 5
        assert [a.arm_id for a in arms] == ["q1", "q2", "q3", "q4", "B"]
        assert sum(a.is_incumbent for a in arms) == 1

    def test_the_candidates_come_from_the_retention_decision(self):
        """Not recomputed. A session that re-derived its own field would be
        choosing its own arms."""
        decision = json.loads(
            (REPO / B.RETENTION_REL).read_text())[
            "the_frozen_behavioural_finalists"]
        assert decision["rule_applied"] == "quality_only"
        recorded = {int(m["quality_position"]): m["state_id"]
                    for m in decision["members"]}
        for arm in B.arms(REPO):
            if arm.is_incumbent:
                continue
            assert recorded[arm.quality_position] == arm.state_id

    def test_the_incumbent_has_no_quality_position_and_no_state_id(self):
        """B is not a candidate of the D1 search, and it has no search state id.

        Both are properties of the arm rather than gaps. Giving it a position
        would make it look like a fifth candidate; giving it a state id would
        claim it came out of a search, when it was built as a FIXED PATH and
        `c1_arm_identities.json` records `state_id: null` for it.

        The identity is asserted on the four CONTENT fields instead -- which is
        the only thing the promoted arm actually has, and the reason the design
        block carries four digests rather than a directory name.
        """
        b = next(a for a in B.arms(REPO) if a.is_incumbent)
        declared = json.loads(
            (REPO / B.DESIGN_REL).read_text())["incumbent"]
        assert b.quality_position is None
        assert b.state_id is None and declared["state_id"] is None
        assert b.checkpoint_dir is None
        assert b.is_materialized_from_a_spec
        assert b.construction == B.INCUMBENT_CONSTRUCTION
        for field in ("artifact_digest", "weights_digest",
                      "single_shard_sha256", "arch_signature"):
            assert b.identities[field] == declared[field]
            assert len(b.identities[field]) == 64

    def test_the_incumbent_is_the_arm_c1_promoted(self):
        """Derived from C1's verdict, not read from a constant beside it.

        The design stated C1's BEATEN arm until 2026-10-08 -- and C1's delta
        between the two arms exceeds the SESOI the decision rule tests against,
        so the error could manufacture a GO. Owner of the finding:
        `logs/stages/stage-1/phase_d1/analyses/d1_control_arm_identity.json`.
        """
        from stages.d_series.incumbent import (
            disagreements, standing_incumbent,
        )

        declared = json.loads(
            (REPO / B.DESIGN_REL).read_text())["incumbent"]
        assert disagreements(declared, REPO) == []
        standing = standing_incumbent(REPO)
        assert standing["c1_arm"] == "treatment"
        assert declared["impl_id"] == "attention.activation_importance_v1"
        b = next(a for a in B.arms(REPO) if a.is_incumbent)
        assert b.artifact_digest == standing["artifact_digest"]

    def test_every_arm_is_obtainable_at_the_recorded_identity(self):
        """The `$0` check that stops a probe training from the wrong checkpoint.

        TWO KINDS OF ARM, and the rows differ because the questions do: a
        candidate has secured bytes on this host and they are hashed; the
        incumbent has none and its construction spec's pins are compared
        against the design instead. Asserting `checkpoint_dir` on every row
        would have required the incumbent to have a directory it cannot have.
        """
        out = B.require_arms_present(REPO)
        assert out["n_arms"] == 5
        by_arm = {r["arm"]: r for r in out["arms"]}
        assert sorted(by_arm) == ["B", "q1", "q2", "q3", "q4"]
        for arm_id in ("q1", "q2", "q3", "q4"):
            row = by_arm[arm_id]
            assert Path(row["checkpoint_dir"]).is_dir()
            assert len(row["identities_checked"]) == 4
            assert "hashed here" in row["obtained_by"]
        b = by_arm["B"]
        assert b["checkpoint_dir"] is None
        assert b["construction"] == B.INCUMBENT_CONSTRUCTION
        assert b["construction_is_callable"]
        assert len(b["identities_checked"]) == 4
        assert "materialized on the pod" in b["obtained_by"]
        #: And it says what it does NOT prove, because a CPU box cannot prove
        #: the bytes match and a check that implied it would be worse than none.
        assert "can claim that" in b["_what_this_does_not_claim"]
        assert "digest gate proves it" in b["_what_this_does_not_claim"]

    def test_a_wrong_checkpoint_is_refused(self, monkeypatch):
        """Non-vacuity: point an arm at another arm's bytes and be refused.

        The stub carries q1's `identities`, which is the field the check
        actually reads. An earlier version omitted it, and because every check
        is "all declared identities agree", the stub passed -- a stub that
        omits the field under test proves the opposite of what it looks like.
        `test_an_arm_that_declares_no_identities_is_refused` closes that hole
        in the production code rather than only in this fixture.
        """
        arms = B.arms(REPO)
        q1, q2 = arms[0], arms[1]
        monkeypatch.setattr(B, "arms",
                            lambda repo_root=REPO, arm_root=None: (
            B.Arm(arm_id="q1", state_id=q1.state_id,
                  artifact_digest=q1.artifact_digest, quality_position=1,
                  checkpoint_dir=q2.checkpoint_dir, role="candidate",
                  identities=dict(q1.identities)),))
        with pytest.raises(B.D1BehaviouralError) as exc:
            B.require_arms_present(REPO)
        assert "not the checkpoint that was selected" in str(exc.value)

    def test_an_arm_that_declares_no_identities_is_refused(self, monkeypatch):
        """`all([])` is True, so an arm binding nothing would pass everything.

        Both branches of the check are "every declared identity agrees", which
        is vacuously satisfied by an empty set -- and silently. Refused in
        `require_arms_present` for both kinds of arm.
        """
        q1 = B.arms(REPO)[0]
        monkeypatch.setattr(B, "arms",
                            lambda repo_root=REPO, arm_root=None: (
            B.Arm(arm_id="q1", state_id=q1.state_id,
                  artifact_digest=q1.artifact_digest, quality_position=1,
                  checkpoint_dir=q1.checkpoint_dir, role="candidate",
                  identities={}),))
        with pytest.raises(B.D1BehaviouralError, match="binds nothing"):
            B.require_arms_present(REPO)

    def test_the_incumbents_construction_spec_must_agree_with_the_design(
            self, monkeypatch):
        """Non-vacuity for the other branch: move the design's digest.

        The spec's pins and the design's identities are two records of one
        fact. If they disagree, the arm the pod builds is not the arm the
        design binds, and that is the whole failure this round found -- caught
        on the dev box rather than by a digest gate after the pod has paid for
        four candidate materializations.
        """
        q1 = B.arms(REPO)[0]
        b = next(a for a in B.arms(REPO) if a.is_incumbent)
        monkeypatch.setattr(B, "arms",
                            lambda repo_root=REPO, arm_root=None: (
            B.Arm(arm_id="B", state_id=None,
                  artifact_digest="f" * 64, quality_position=None,
                  checkpoint_dir=None, role="incumbent",
                  identities={**b.identities, "artifact_digest": "f" * 64},
                  construction=b.construction),))
        with pytest.raises(B.D1BehaviouralError,
                           match="name different checkpoints"):
            B.require_arms_present(REPO)
        assert q1.identities  # the real field is non-empty, so the stub is apt


class TestTheSeedsAreMaterializedNotChosen:

    def test_the_counts_come_from_the_design(self):
        bd = B.behavioural_design(REPO)
        assert len(B.screening_seeds(REPO)) == bd["screening_seeds"] == 2
        assert len(B.confirmation_seeds(REPO)) == bd["confirmation_seeds"] == 3

    def test_the_two_rungs_use_disjoint_seeds(self):
        """A candidate selected on a seed is not independently confirmed on
        the same one."""
        assert not (set(B.screening_seeds(REPO))
                    & set(B.confirmation_seeds(REPO)))

    def test_no_draw_collides_with_a_selection_seed(self):
        """`sa/sb/sc` because B was selected under them; C1's three because
        B's standing result was measured on them."""
        excluded = set(B.excluded_seeds(REPO))
        assert len(excluded) == 6
        assert not (set(B.screening_seeds(REPO)) & excluded)
        assert not (set(B.confirmation_seeds(REPO)) & excluded)

    def test_the_exclusion_set_is_read_from_both_owners(self):
        prereg = json.loads(
            (REPO / "logs/stages/stage-1/phase_c1/plans/"
                    "phase_c0_preregistration.json").read_text())
        historical = prereg["confirmation_seeds"]["historical_seeds_excluded"]
        c1 = json.loads((REPO / B.C1_EXECUTION_PREREG_REL).read_text())
        excluded = set(B.excluded_seeds(REPO))
        for key in ("sa", "sb", "sc"):
            assert int(historical[key]) in excluded
        for value in c1["seeds"]["values"]:
            assert int(value) in excluded

    def test_the_derivation_is_deterministic(self):
        assert B.derive_seeds("screening", REPO) == \
            B.derive_seeds("screening", REPO)

    def test_it_is_based_on_an_identity_that_predates_every_d1_result(self):
        """`design_hash` is the scientific preimage of the frozen design. It
        did not move across two spend bookings or a ceiling amendment, which is
        what makes the seeds independent of any outcome."""
        import hashlib

        base = json.loads((REPO / B.DESIGN_REL).read_text())["design_hash"]
        digest = hashlib.sha256(
            f"{base}:phase-d1:screening-seed:0".encode()).digest()
        first = int.from_bytes(digest[:4], "big") % (2 ** 31)
        #: The first draw is the first non-colliding one, so it is either this
        #: value or a later index -- but it must be reachable from this rule.
        seeds = B.derive_seeds("screening", REPO)
        assert first in seeds or first in set(B.excluded_seeds(REPO))

    def test_a_missing_c1_preregistration_is_refused_not_ignored(self,
                                                                 monkeypatch):
        """An unknown exclusion set is how a draw collides with a selection
        seed without anyone noticing."""
        monkeypatch.setattr(B, "C1_EXECUTION_PREREG_REL", "logs/nope.json")
        with pytest.raises(B.D1BehaviouralError) as exc:
            B.excluded_seeds(REPO)
        assert "cannot be excluded" in str(exc.value)


class TestTheProbeScheduleAgreesWithThePricedDesign:

    def test_screening_is_ten_probes(self):
        probes = B.probes("screening", REPO)
        assert len(probes) == 10 == B.behavioural_design(REPO)["screening_probes"]
        assert len({p.probe_id for p in probes}) == 10

    def test_every_arm_appears_at_every_seed(self):
        probes = B.probes("screening", REPO)
        seeds = set(B.screening_seeds(REPO))
        for arm in B.arms(REPO):
            got = {p.seed for p in probes if p.arm_id == arm.arm_id}
            assert got == seeds, arm.arm_id

    def test_a_schedule_that_disagreed_with_the_design_is_refused(self,
                                                                  monkeypatch):
        """It would run a different experiment at the same price."""
        real = B.behavioural_design

        def fewer(repo_root=REPO):
            doc = dict(real(repo_root))
            doc["screening_probes"] = 6
            return doc

        monkeypatch.setattr(B, "behavioural_design", fewer)
        with pytest.raises(B.D1BehaviouralError) as exc:
            B.probes("screening", REPO)
        assert "the design declares" in str(exc.value)


class TestTheBatteryIsTheRealizedFamilyAndItsBytes:

    def test_the_role_is_realized_and_matches_the_manifest(self):
        role = B.battery_role("d1_screening", REPO)
        assert role["n_prompts"] == 950 and role["n_scorable"] == 850
        assert len(role["files"]) == len(B.FROZEN_STRATA) == 7
        assert role["_capacity_blocker"] == "CLOSED"

    def test_the_frozen_stratum_balance_is_preserved(self):
        """`correct_overall` is a mean over the mixture, so a different
        balance measures a different quantity."""
        for name in ("d1_screening", "d1_confirmation"):
            assert B.battery_role(name, REPO)["per_stratum"] == B.FROZEN_STRATA

    def test_the_two_roles_are_disjoint_by_item_id(self):
        """Read from the realized ids, not inferred from two hashes differing."""
        assert B.roles_are_disjoint(REPO)["disjoint"] is True

    def test_a_tampered_battery_file_is_refused(self, tmp_path, monkeypatch):
        """The check is on BYTES. A claim about evidence must not substitute
        for the evidence -- which is how the closed capacity blocker came to be
        reported as live."""
        import shutil

        src = REPO / B.BATTERY_ROOT_REL
        dst = tmp_path / B.BATTERY_ROOT_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        for rel in (B.DESIGN_REL, B.RETENTION_REL, B.FAMILY_REL,
                    B.MANIFEST_REL):
            out = tmp_path / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / rel, out)
        victim = dst / "d1_screening" / "math_verified.jsonl"
        victim.write_text(victim.read_text() + "\n")
        with pytest.raises(B.D1BehaviouralError) as exc:
            B.battery_role("d1_screening", tmp_path)
        assert "the manifest declares" in str(exc.value)

    def test_the_exhausted_pool_is_named_as_not_this_battery(self):
        """The record carries its own disambiguation, because the next reader
        will meet `batteries_remaining: 0` before they meet this."""
        role = B.battery_role("d1_screening", REPO)
        assert "batteries_remaining 0" in role["_not_the_exhausted_pool"]
        assert "CLOSED" in role["_not_the_exhausted_pool"]


class TestTheContractBindsEverythingTheRungMeasures:

    def test_it_derives(self):
        c = B.session_contract("screening", REPO)
        assert c["rung"] == "screening"
        assert c["n_probes"] == c["n_probes_declared"] == 10
        assert c["battery"]["role"] == "d1_screening"
        assert c["arms"]["n_arms"] == 5
        assert c["roles_disjoint"]["disjoint"] is True

    def test_the_endpoint_and_sesoi_are_c0s(self):
        c = B.session_contract("screening", REPO)
        assert c["endpoint"] == "correct_overall"
        assert c["sesoi"] == 0.01

    def test_the_sesoi_is_labelled_as_carried_forward(self):
        """D-series absolute scores are not interchangeable with C1's, because
        three strata draw from wider populations. The SESOI is an assumption
        about the within-family PAIRED difference, and saying so is the
        difference between an assumption and a silent reinterpretation."""
        c = B.session_contract("screening", REPO)
        note = c["_sesoi_is_carried_forward_not_remeasured"]
        assert "not directly interchangeable" in note
        assert "PAIRED difference" in note

    def test_the_recipe_is_the_frozen_one(self):
        from shared.recipes import E1_KD_HEAVY_0860K as recipe

        c = B.session_contract("screening", REPO)
        assert c["recovery_recipe"]["recipe_id"] == recipe.recipe_id
        assert c["recovery_recipe"]["tokens"] == recipe.tokens == 860000

    def test_the_design_declares_the_same_recipe(self):
        """Two owners naming one recipe is two things that can disagree."""
        c = B.session_contract("screening", REPO)
        assert c["recovery_recipe"]["declared_in_design"] == "E1_KD_HEAVY_0860K"


class TestThePermissionsCannotBeSubstituted:

    def test_it_allows_training_and_denies_a_search(self):
        a = _auth()
        assert a.allows("recovery_training")
        assert a.allows("behavioural_selection")
        assert not a.allows("beam_search")
        assert not a.allows("d1_search")

    def test_it_denies_reselection_and_incumbent_remeasurement(self):
        a = _auth()
        assert not a.authorizes_candidate_reselection
        assert not a.authorizes_incumbent_remeasurement
        assert not a.automatic_followon_start

    def test_the_policy_is_the_instance_field_the_base_reads(self):
        """A class-level `POLICY` is decoration: the base consults
        `self.action_policy`, and an artifact that set only the class attribute
        serialized under `deny_all` and refused itself with a message naming
        neither the policy nor the type."""
        a = _auth()
        assert a.action_policy is D1_BEHAVIOURAL_POLICY
        assert a.as_dict()["schema"].endswith("behavioural_authorization/v1")

    def test_it_serializes_every_claim(self):
        payload = _auth().as_dict()
        for wire in D1_BEHAVIOURAL_POLICY.wire_claims:
            assert wire in payload, wire

    def test_the_schema_differs_from_the_searchs(self):
        from stages.phase_d1.d1_authorization import SCHEMA as SEARCH

        assert _auth().as_dict()["schema"] != SEARCH

    def test_the_rung_is_enforced(self):
        a = _auth(rung="screening")
        a.require_rung("screening")
        with pytest.raises(D1BehaviouralRefused):
            a.require_rung("confirmation")

    def test_the_contract_hash_is_enforced(self):
        a = _auth(contract_hash="a" * 64)
        a.require_contract("a" * 64)
        with pytest.raises(D1BehaviouralRefused) as exc:
            a.require_contract("b" * 64)
        assert "not what was authorized" in str(exc.value)

    def test_an_artifact_binding_no_contract_is_refused(self):
        with pytest.raises(D1BehaviouralRefused) as exc:
            _auth(contract_hash="").require_contract("a" * 64)
        assert "binds no contract hash" in str(exc.value)

    def test_screening_has_no_advancing_candidate(self):
        with pytest.raises(D1BehaviouralRefused):
            _auth(rung="screening").require_advancing_candidate()

    def test_a_confirmation_artifact_must_name_the_candidate(self):
        """One that left it open would permit confirming whichever candidate
        the session chose, which is the selection it must be independent of."""
        with pytest.raises(D1BehaviouralRefused) as exc:
            _auth(rung="confirmation").require_advancing_candidate()
        assert "must NAME the advancing candidate" in str(exc.value)
        named = _auth(rung="confirmation", advancing_candidate="q2")
        assert named.require_advancing_candidate() == "q2"


class TestTheScorerIsPinnedToTheRealizedRole:
    """`score_d1_screening.py` exists rather than a `--battery` flag on an
    existing scorer, and that is the point: C1's `main()` pins its battery by
    equality on purpose -- "the production path cannot be aimed anywhere else"
    -- and the rung it guards is the one that may name an incumbent. C2's
    screening scorer pins its own for the same reason. A flag is exactly how a
    rung comes to be scored on the wrong prompts.
    """

    @staticmethod
    def _mod():
        import importlib.util

        for extra in ("scripts/autoinit",):
            if str(REPO / extra) not in sys.path:
                sys.path.insert(0, str(REPO / extra))
        path = REPO / "scripts/stages/stage-1/phase_d1/score_d1_screening.py"
        spec = importlib.util.spec_from_file_location("score_d1_screening",
                                                      path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_it_takes_no_battery_argument(self):
        mod = self._mod()
        flags = set()
        import argparse

        parser = argparse.ArgumentParser()
        #: Build the real parser by calling main's construction indirectly --
        #: simplest reliable probe is the module source, since main() parses
        #: and then runs.
        src = (REPO / "scripts/stages/stage-1/phase_d1/score_d1_screening.py").read_text()
        assert '"--battery"' not in src, (
            "the scorer accepts a battery path; that is how a rung gets "
            "scored on the wrong prompts")
        assert "THE BATTERY IS NOT AN ARGUMENT" in src
        assert flags == set() and parser is not None   # probe is the source

    def test_the_metric_is_c1s_imported_unchanged(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/score_d1_screening.py").read_text()
        assert ("from stages.phase_c1.score_c1_confirmation import "
                "build_result, score_battery") \
            in src, ("the metric must be C1's, imported: a reimplementation "
                     "would make a screening delta uninformative about a "
                     "confirmation delta")

    def test_the_scorable_split_is_derived_and_matches_the_manifest(self):
        """The D-series roles have no per-role `manifest.json`, so a stale
        sidecar is impossible by construction -- but only if the split really
        is derived."""
        mod = self._mod()
        layout = mod.battery_sets("d1_screening")
        role = B.battery_role("d1_screening", REPO)
        assert layout["behaviour_only_sets"] == {"code"}
        derived = sum(role["per_stratum"][s] for s in layout["scorable_sets"])
        assert derived == role["n_scorable"] == 850
        assert set(layout["sets"]) == set(B.FROZEN_STRATA)

    def test_the_result_schema_is_not_c1s(self):
        """A screening result carrying C1's confirmation schema would be
        indistinguishable from evidence that may promote."""
        mod = self._mod()
        assert mod.SCHEMA == "aadistill.phase_d1.screening_result/v1"
        assert mod.ROLE == "d1_screening"

    def test_it_carries_what_it_may_not_be_read_as(self):
        mod = self._mod()
        text = " ".join(mod.MAY_NOT)
        assert "winner's curse" in text
        assert "confirmation" in text

    def test_it_states_the_absolute_score_limitation(self):
        src = (REPO / "scripts/stages/stage-1/phase_d1/score_d1_screening.py").read_text()
        assert "_absolute_scores_are_not_interchangeable_with_c1s" in src


class TestTheRankingIsMechanicalAndNeverForcesAWinner:
    """Five decision cases, because the one that matters is the one where the
    leader on correctness is removed by the guardrail."""

    @staticmethod
    def _rows(table):
        seeds = B.screening_seeds(REPO)
        out = []
        for arm, (correct, usable) in table.items():
            for i, seed in enumerate(seeds):
                out.append({"arm": arm, "seed": seed,
                            "correct_overall": correct[i],
                            "usable_rollout_rate": usable[i]})
        return out

    CLEAN = {"q1": ([0.40, 0.41], [0.90, 0.91]),
             "q2": ([0.45, 0.46], [0.90, 0.90]),
             "q3": ([0.42, 0.42], [0.89, 0.90]),
             "q4": ([0.38, 0.39], [0.91, 0.90]),
             "B": ([0.40, 0.40], [0.90, 0.90])}

    def test_the_best_pooled_delta_advances(self):
        ranking = B.rank_screening(self._rows(self.CLEAN), REPO)
        assert [r["arm"] for r in ranking["ranked"]] == ["q2", "q3", "q1", "q4"]
        winner = B.advance_one(ranking)
        assert winner["arm"] == "q2" and winner["advanced"] is True
        assert winner["margin_over_runner_up"] == pytest.approx(0.035)

    def test_the_delta_is_against_the_incumbent_on_the_same_rung(self):
        ranking = B.rank_screening(self._rows(self.CLEAN), REPO)
        q2 = next(r for r in ranking["ranked"] if r["arm"] == "q2")
        assert q2["delta_vs_b"] == pytest.approx(0.455 - 0.400)
        assert q2["incumbent_correct_overall"] == pytest.approx(0.400)

    def test_pooling_is_the_seed_mean(self):
        """Because the confirmation estimand is the prompt-mean of the
        SEED-MEAN paired difference. A screening statistic computed another way
        would rank on a quantity the confirmation does not estimate."""
        pooled = B.pool_arm([{"seed": 1, "correct_overall": 0.4,
                              "usable_rollout_rate": 0.9},
                             {"seed": 2, "correct_overall": 0.5,
                              "usable_rollout_rate": 0.8}])
        assert pooled["correct_overall"] == pytest.approx(0.45)
        assert pooled["usable_rollout_rate"] == pytest.approx(0.85)
        assert pooled["n_seeds"] == 2

    def test_a_vetoed_leader_sorts_last_despite_the_best_delta(self):
        """THE CASE THAT MATTERS. A veto is not a penalty to be outweighed:
        usable_rollout never earns positive credit and a candidate removed by
        the guardrail cannot win on correctness."""
        table = dict(self.CLEAN)
        table["q2"] = ([0.50, 0.50], [0.80, 0.80])
        ranking = B.rank_screening(self._rows(table), REPO)
        q2 = next(r for r in ranking["ranked"] if r["arm"] == "q2")
        assert q2["vetoed"] is True
        assert q2["delta_vs_b"] == pytest.approx(0.10)
        assert q2["screening_position"] == len(ranking["ranked"]) - 1
        assert B.advance_one(ranking)["arm"] == "q3"

    def test_the_per_seed_guardrail_bites_independently(self):
        """A candidate can pass pooled and fail a single seed."""
        table = dict(self.CLEAN)
        table["q2"] = ([0.50, 0.50], [0.98, 0.78])
        ranking = B.rank_screening(self._rows(table), REPO)
        q2 = next(r for r in ranking["ranked"] if r["arm"] == "q2")
        assert q2["delta_usable_pooled"] > B.GUARDRAIL_POOLED_MIN_DELTA
        assert q2["vetoed"] is True
        assert any("seed" in v for v in q2["vetoes"])

    def test_an_all_vetoed_field_advances_nobody(self):
        """No forced winner. The decision rule is three-way, and advancing a
        vetoed candidate would spend six confirmation probes on an arm the
        guardrail already removed."""
        table = {k: (v[0], [0.70, 0.70]) for k, v in self.CLEAN.items()}
        table["B"] = ([0.40, 0.40], [0.90, 0.90])
        out = B.advance_one(B.rank_screening(self._rows(table), REPO))
        assert out["advanced"] is None
        assert out["outcome"] == "NO_CANDIDATE_ADVANCES"
        assert len(out["vetoed"]) == 4

    def test_a_tie_breaks_on_the_frozen_quality_position(self):
        """Fixed before any behavioural datum existed, which is what makes the
        tie-break a rule rather than a choice."""
        table = dict(self.CLEAN)
        table["q1"] = ([0.45, 0.46], [0.90, 0.90])
        winner = B.advance_one(B.rank_screening(self._rows(table), REPO))
        assert winner["arm"] == "q1"
        assert winner["tie_broken"] is True

    def test_an_incomplete_field_is_refused(self):
        """Ranking it would select on who happened to finish."""
        rows = self._rows(self.CLEAN)
        partial = [r for r in rows
                   if not (r["arm"] == "q3" and r["seed"] == B.screening_seeds(REPO)[1])]
        with pytest.raises(B.D1RankingError) as exc:
            B.rank_screening(partial, REPO)
        assert "incomplete" in str(exc.value)

    def test_the_incumbent_is_never_ranked_as_a_candidate(self):
        ranking = B.rank_screening(self._rows(self.CLEAN), REPO)
        assert "B" not in [r["arm"] for r in ranking["ranked"]]
        assert ranking["incumbent"]["arm"] == "B"

    def test_the_record_says_it_is_not_a_verdict(self):
        ranking = B.rank_screening(self._rows(self.CLEAN), REPO)
        assert "winner's curse" in ranking["_this_is_not_a_verdict"]
        assert "removed" in ranking["_guardrail_is_a_veto_only"]

    def test_the_guardrail_thresholds_are_c0s(self):
        prereg = json.loads(
            (REPO / "logs/stages/stage-1/phase_c1/plans/"
                    "phase_c0_preregistration.json").read_text())
        veto = prereg["behavioural_guardrails"]["usable_rollout_veto"]
        assert "-0.05" in veto["pooled"]
        assert "-0.10" in veto["per_seed"]
        assert B.GUARDRAIL_POOLED_MIN_DELTA == -0.05
        assert B.GUARDRAIL_PER_SEED_MIN_DELTA == -0.10


class TestTheDriverAssertsBeforeItTrains:

    @staticmethod
    def _driver():
        import importlib.util

        for extra in ("scripts/pod", "scripts/autoinit"):
            if str(REPO / extra) not in sys.path:
                sys.path.insert(0, str(REPO / extra))
        path = REPO / "scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_driver.py"
        spec = importlib.util.spec_from_file_location("d1b_driver", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_the_parser_builds_and_every_required_option_is_a_string(self):
        """So the shared dispatch probe can build it. A launcher invisible to
        that probe is one whose missing branch falls through."""
        parser = self._driver().build_parser()
        for action in parser._actions:
            if action.required:
                assert action.type in (None, str), action.option_strings

    def test_the_contract_is_asserted_before_the_trainer_is_reached(self):
        import ast

        src = (REPO / "scripts/stages/stage-1/phase_d1/autoinit_d1_behavioural_driver.py").read_text()
        assert src.index("session_contract(") < src.index("str(TRAINER)")
        tree = ast.parse(src)
        assert tree is not None

    def test_the_probe_config_refuses_an_override_outside_the_allowed_set(self):
        """The mechanical form of "identical recovery". A config free to differ
        elsewhere would make a paired difference uninterpretable."""
        mod = self._driver()
        import tempfile
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "recipe.json"
            recipe.write_text(json.dumps({"lr": 1e-4, "seed": 0,
                                          "run_name": "x", "out_dir": "o",
                                          "data_dir": "d", "student_path": "s"}))
            probe = SimpleNamespace(probe_id="p", arm_id="q1", seed=7,
                                    rung="screening", checkpoint_dir="/ckpt")
            #: The allowed set omits `lr`, so a run that changed it is refused.
            with pytest.raises(mod.D1BehaviouralDriverError) as exc:
                mod.probe_config(probe, audit=Path(tmp),
                                 frozen_recipe=recipe, pack_dir="pack",
                                 overrides=frozenset({"run_name"}))
            assert "outside the allowed override set" in str(exc.value)

    def test_the_terminal_marker_is_one_predicate(self):
        mod = self._driver()
        assert mod.terminal_marker(check_only=False, status="COMPLETE",
                                   evidence_written=True) == "ALL_DONE"
        assert mod.terminal_marker(check_only=False, status="COMPLETE",
                                   evidence_written=False) == "RUN_FAILED"
        assert mod.terminal_marker(check_only=True, status="CHECK_ONLY_OK",
                                   evidence_written=True) == "CHECK_ONLY_OK"
        assert mod.terminal_marker(check_only=True, status="FAILED",
                                   evidence_written=True) == "RUN_FAILED"

    def test_the_scorer_it_calls_is_the_pinned_one(self):
        mod = self._driver()
        assert mod.SCORER.name == "score_d1_screening.py"
        assert mod.SCORER.is_file()
