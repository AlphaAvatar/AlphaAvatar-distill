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
for _extra in ("src", "scripts", "scripts/experiments/stage-1"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from experiments.phase_d1 import behavioural as B  # noqa: E402
from experiments.phase_d1.behavioural_governance import (  # noqa: E402
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

    def test_the_incumbent_has_no_quality_position(self):
        """B is not a candidate of the D1 search. Giving it a position would
        make it look like a fifth one."""
        b = next(a for a in B.arms(REPO) if a.is_incumbent)
        assert b.quality_position is None
        assert b.state_id == json.loads(
            (REPO / B.DESIGN_REL).read_text())["incumbent"]["state_id"]

    def test_every_arm_has_verified_bytes_at_the_recorded_identity(self):
        """The $0 check that stops a probe being trained from the wrong
        checkpoint -- which costs its full training time to discover."""
        out = B.require_arms_present(REPO)
        assert out["n_arms"] == 5
        for row in out["arms"]:
            assert row["artifact_digest"].startswith(row["recorded_prefix"])
            assert Path(row["checkpoint_dir"]).is_dir()

    def test_a_wrong_checkpoint_is_refused(self, tmp_path, monkeypatch):
        """Non-vacuity: point an arm at another arm's bytes and the identity
        check must catch it."""
        arms = B.arms(REPO)
        q1, q2 = arms[0], arms[1]
        monkeypatch.setitem(B.ARM_SOURCES, "q1",
                            str(Path(q2.checkpoint_dir).parent))
        #: q1 now resolves to <q2's parent>/<q1's state id>, which does not
        #: exist -- so the refusal is the missing-checkpoint one. Point it at
        #: q2's actual directory to get the identity refusal instead.
        monkeypatch.setattr(B, "arms", lambda repo_root=REPO: (
            B.Arm(arm_id="q1", state_id=q1.state_id,
                  artifact_digest=q1.artifact_digest, quality_position=1,
                  checkpoint_dir=q2.checkpoint_dir, role="candidate"),))
        with pytest.raises(B.D1BehaviouralError) as exc:
            B.require_arms_present(REPO)
        assert "not the checkpoint that was selected" in str(exc.value)


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
        from experiments.recipes import E1_KD_HEAVY_0860K as recipe

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
        from experiments.phase_d1.d1_authorization import SCHEMA as SEARCH

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
        path = REPO / "scripts/autoinit/score_d1_screening.py"
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
        src = (REPO / "scripts/autoinit/score_d1_screening.py").read_text()
        assert '"--battery"' not in src, (
            "the scorer accepts a battery path; that is how a rung gets "
            "scored on the wrong prompts")
        assert "THE BATTERY IS NOT AN ARGUMENT" in src
        assert flags == set() and parser is not None   # probe is the source

    def test_the_metric_is_c1s_imported_unchanged(self):
        src = (REPO / "scripts/autoinit/score_d1_screening.py").read_text()
        assert "from score_c1_confirmation import build_result, score_battery" \
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
        src = (REPO / "scripts/autoinit/score_d1_screening.py").read_text()
        assert "_absolute_scores_are_not_interchangeable_with_c1s" in src
