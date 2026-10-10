"""The D1 BEHAVIOURAL pod's blocking CPU gate. Pod-faithful by construction.

`SetupManifest.test_paths` points HERE, not at the whole `phase_d1/tests`
directory, because the directory carries the SEARCH and REPLAY chains' tests —
which consume the search's staged assets (`state_eval_v1`), the teacher's HF
snapshot and this dev box's durable checkpoint store, none of which a
behavioural pod receives. The first launch-bound sweep of the full directory
failed 77 of them under the pod environment; every one was a search-chain test
asserting a staging this session correctly does not have.

What belongs in a BEHAVIOURAL pod gate is exactly what a mis-staged or
mis-checked-out behavioural pod would get wrong, checkable on its CPU before
anything bills science:

* the committed bindings -- the design, the execution preregistration, the
  replay plan -- agree with each other and with their own hashes;
* the staged bytes -- both battery roles, the recovery pack, both calibration
  mixtures -- match the pins they ship;
* the pure derivations -- seeds, schedule, execution protocol -- reproduce the
  preregistered values;
* the governance and dispatch surface imports and refuses what it must.

DELIBERATELY ABSENT: anything reading the dev host's durable store
(`require_arms_present`, `session_contract`, `build_payload` -- the pod proves
arm identities at materialization, with the digest gates), anything needing
the teacher (the root-state re-derivation runs in the driver, after
TEACHER_READY), and anything touching a network.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[6]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from stages.phase_d1 import behavioural as B  # noqa: E402
from stages.phase_d1 import behavioural_materialize as M  # noqa: E402


def _prereg() -> dict:
    return json.loads(
        (REPO / "logs/stages/stage-1/phase_d1/plans/"
                "d1_behavioural_preregistration.json").read_text())


class TestTheCommittedBindingsAgree:

    def test_the_design_is_the_preregistered_revision_with_no_blockers(self):
        design = B.design(REPO)
        assert design["open_blockers"] == []
        assert _prereg()["design"]["design_hash"] == design["design_hash"]

    def test_the_preregistration_matches_its_own_hash(self):
        import importlib.util

        doc = _prereg()
        src = (REPO / "scripts/stages/stage-1/phase_d1/"
                      "write_d1_behavioural_preregistration.py")
        spec = importlib.util.spec_from_file_location("_pw", src)
        writer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(writer)
        assert doc["preregistration_sha256"] == \
            writer.preregistration_identity(doc)

    def test_the_replay_plan_is_the_bound_one_and_fully_pinned(self):
        from stages.phase_d1 import replay_specs as R

        doc = _prereg()
        assert M.plan_sha256(REPO) == doc["arm_materialization"]["plan_sha256"]
        plan = M.load_plan(REPO / M.PLAN_REL)
        leaves = R.leaves_from_plan(plan)
        assert len(leaves) == 4
        members = {m["state_id"]: m for m in
                   json.loads((REPO / B.RETENTION_REL).read_text())
                   ["the_frozen_behavioural_finalists"]["members"]}
        for leaf in leaves:
            want = members[leaf.state_id]
            assert leaf.expected_artifact_digest == want["artifact_digest"]
            assert leaf.expected_weights_digest == want["weights_digest"]
            assert len(leaf.steps) == 4
            for step in leaf.steps:
                assert len(step.expected_artifact_digest) == 64
                assert len(step.expected_config_hash) == 64

    def test_the_incumbent_identity_is_the_designs(self):
        bound = _prereg()["arms"]["members"]
        b = next(m for m in bound if m["arm"] == "B")
        design = B.design(REPO)["incumbent"]
        for key in ("artifact_digest", "weights_digest",
                    "single_shard_sha256", "arch_signature"):
            assert b["identities"][key] == design[key]
        assert b["identities"]["artifact_digest"].startswith("53e30566")


class TestTheStagedBytesMatchTheirPins:

    def test_both_battery_roles_rehash_against_the_manifest(self):
        for role in ("d1_screening", "d1_confirmation"):
            identity = B.battery_role(role, REPO)
            assert identity["n_prompts"] == 950
            assert identity["n_scorable"] == 850
            assert len(identity["files"]) == 7

    def test_the_two_roles_are_disjoint_on_this_machine(self):
        assert B.roles_are_disjoint(REPO)["disjoint"] is True

    def test_the_recovery_pack_matches_the_frozen_recipe(self):
        from shared.recipes import E1_KD_HEAVY_0860K as recipe
        from stages.phase_c1.autoinit_c1_driver import PACK_DIR

        blocks = REPO / PACK_DIR / "blocks.npz"
        assert blocks.is_file(), "the recovery pack is not staged"
        got = hashlib.sha256(blocks.read_bytes()).hexdigest()
        assert got == recipe.pack_sha256

    def test_both_calibration_mixtures_resolve_through_their_pins(self):
        from aadistill.initialization.calibration.profiles import get_profile
        from shared.calibration import register_builtin_profiles

        register_builtin_profiles()
        for pid in ("calib.domain_balanced@v1", "calib.reasoning_heavy@v2"):
            profile = get_profile(pid)
            items = profile.resolve(REPO)
            assert items, pid

    def test_the_evaluation_tokenizer_sidecars_carry_their_pins(self):
        from stages.phase_c1.autoinit_c1_driver import (
            TOKENIZER_SIDECAR_SHA256, TOKENIZER_SOURCE,
        )

        for name, want in TOKENIZER_SIDECAR_SHA256.items():
            path = Path(TOKENIZER_SOURCE) / name
            assert path.is_file(), f"{name} is not staged"
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            assert got == want, name


class TestThePureDerivationsReproduceThePreregisteredValues:

    def test_the_seeds(self):
        doc = _prereg()
        assert list(B.screening_seeds(REPO)) == doc["seeds"]["screening"]
        assert list(B.confirmation_seeds(REPO)) == doc["seeds"]["confirmation"]
        assert list(B.excluded_seeds(REPO)) == doc["seeds"]["excluded"]

    def test_the_screening_schedule(self):
        probes = B.probes("screening", REPO)
        assert [p.probe_id for p in probes] == \
            _prereg()["screening"]["probe_ids"]

    def test_the_confirmation_schedule_narrows(self):
        probes = B.probes("confirmation", REPO, advancing_candidate="q1")
        assert len(probes) == 6
        assert {p.arm_id for p in probes} == {"q1", "B"}
        with pytest.raises(B.D1BehaviouralError):
            B.probes("confirmation", REPO)

    def test_the_candidates_replay_under_the_searchs_execution(self):
        from stages.phase_d1 import replay_specs as R

        plan = M.load_plan(REPO / M.PLAN_REL)
        leaves = R.leaves_from_plan(plan)
        agreement = R.verify_execution_config(leaves)
        assert agreement["using"] == {
            "micro_batch_size": 3,
            "calibration_batch_packing": "length_sorted_v1"}

    def test_the_incumbent_rebuilds_under_a_bsz1(self):
        assert M.incumbent_execution().as_fingerprint() == {
            "micro_batch_size": 1,
            "calibration_batch_packing": "original_order_v1"}


class TestTheGovernanceAndDispatchSurfaceHolds:

    def test_the_authorization_type_loads_and_refuses_an_edited_artifact(
            self, tmp_path):
        from stages.phase_d1.behavioural_governance import (
            D1BehaviouralAuthorization, D1BehaviouralRefused,
        )

        path = tmp_path / "edited.json"
        path.write_text(json.dumps({"authorization_sha256": "x" * 64,
                                    "schema": "wrong"}))
        with pytest.raises(D1BehaviouralRefused):
            D1BehaviouralAuthorization.load(path)

    def test_the_shell_dispatch_branch_exists_for_this_kind(self):
        src = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"
               ).read_text()
        assert '"$SESSION_KIND" = "d1_behavioural"' in src
        branch = src.split('"$SESSION_KIND" = "d1_behavioural"', 1)[1]
        branch = branch.split("elif", 1)[0]
        assert "allows_recovery_training is True" in branch
        assert "allows_beam_search is False" in branch

    def test_the_driver_and_both_scorers_are_present_and_parse(self):
        import importlib.util

        driver_src = (REPO / "scripts/stages/stage-1/phase_d1/"
                             "autoinit_d1_behavioural_driver.py")
        spec = importlib.util.spec_from_file_location("_pf_drv", driver_src)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        parser = driver.build_parser()
        assert parser is not None
        assert driver.SCORER.is_file()
        assert driver.CONFIRMATION_SCORER.is_file()

    def test_the_artifact_specs_load_through_the_real_loader(self):
        import importlib.util

        src = REPO / "scripts/shared/pod/collect_artifacts.py"
        spec = importlib.util.spec_from_file_location("_pf_collect", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for rel in (
                "configs/stages/stage-1/phase_d1/d1_behavioural_artifacts.json",
                "configs/stages/stage-1/phase_d1/"
                "d1_behavioural_artifacts_failed.json"):
            assert mod.load_specs(str(REPO / rel)), rel
