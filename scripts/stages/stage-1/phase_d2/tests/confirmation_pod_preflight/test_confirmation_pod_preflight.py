"""The D2 SHARED-CONFIRMATION pod's blocking CPU gate. Pod-faithful.

`SetupManifest.test_paths` for the confirmation points HERE. What belongs here
is what a mis-staged or mis-checked-out confirmation pod would get wrong,
checkable on its CPU before anything trains:

* the bindings agree: the preregistration and the shared protocol with their
  own hashes, the committed selection, and the frozen screening ranking that
  names the candidate;
* this checkout's trainer, generator and scorer are the protocol's pins;
* the staged bytes -- both battery roles, the recovery pack, the evaluation
  tokenizer -- match the pins they ship;
* the schedule, seeds and verdict parameters reproduce the registered values;
* the governance and dispatch surface loads and refuses what it must.

DELIBERATELY ABSENT, as on the screening pod: the candidate's bytes (the
driver hashes them at all four identities before anything trains), anything
needing the teacher, and anything touching the network or the dev host.
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

from stages.phase_d2 import d2_confirmation as CF  # noqa: E402


class TestTheBindingsAgree:

    def test_the_preregistration_matches_its_own_hash(self):
        from aadistill.infrastructure.manifest import sha256_json
        from stages.phase_d2.write_d2_preregistration import NON_SCIENCE_KEYS

        doc = CF.preregistration(REPO)
        science = {k: v for k, v in doc.items() if k not in NON_SCIENCE_KEYS}
        assert sha256_json(science) == doc["preregistration_hash"]

    def test_the_shared_protocol_verifies_and_is_the_bound_one(self):
        assert CF.protocol(REPO)["protocol_hash"] == \
            CF.behaviour(REPO)["protocol"]["protocol_hash"]

    def test_the_frozen_screening_names_exactly_one_committed_candidate(self):
        cand = CF.advancing(REPO)
        assert cand["arm"] in {"c1", "c2", "c3", "c4"}
        assert len(CF.arms(REPO, arm_root="/nonexistent")) == 1

    def test_this_checkout_is_the_protocols_trainer_generator_and_scorer(self):
        from stages.d_series import shared_evaluation as SE

        assert SE.tree_matches(CF.protocol(REPO), REPO) == {
            "trainer": [], "generation": [], "scoring": []}

    def test_the_recovery_is_the_protocols(self):
        assert CF.recovery(REPO)["recipe_id"] == "e1_p1_kd_heavy@0.86M"


class TestTheStagedBytesMatchTheirPins:

    def test_the_confirmation_role_rehashes_and_is_the_bound_one(self):
        bat = CF.battery(REPO)
        assert bat["role"] == "d1_confirmation"
        assert (bat["n_prompts"], bat["n_scorable"], len(bat["files"])) == \
            (950, 850, 7)

    def test_the_two_roles_are_disjoint_on_this_machine(self):
        from stages.phase_d1 import behavioural as D1B

        assert D1B.roles_are_disjoint(REPO)["disjoint"] is True

    def test_the_recovery_pack_matches_the_frozen_recipe(self):
        from shared.recipes import E1_KD_HEAVY_0860K as recipe
        from stages.phase_c1.autoinit_c1_driver import PACK_DIR

        blocks = REPO / PACK_DIR / "blocks.npz"
        assert blocks.is_file(), "the recovery pack is not staged"
        assert hashlib.sha256(blocks.read_bytes()).hexdigest() == \
            recipe.pack_sha256

    def test_the_evaluation_tokenizer_sidecars_carry_their_pins(self):
        from stages.phase_c1.autoinit_c1_driver import (
            TOKENIZER_SIDECAR_SHA256, TOKENIZER_SOURCE,
        )

        for name, want in TOKENIZER_SIDECAR_SHA256.items():
            path = Path(TOKENIZER_SOURCE) / name
            assert path.is_file(), f"{name} is not staged"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == want, name


class TestTheScheduleAndVerdictAreTheRegisteredOnes:

    def test_the_seeds_are_the_shared_confirmation_rungs(self):
        assert sorted(CF.seeds(REPO)) == sorted(
            CF.protocol(REPO)["rungs"]["shared_confirmation"]["seeds"])

    def test_three_probes_of_one_candidate(self):
        probes = CF.probes(REPO, arm_root="/nonexistent")
        assert len(probes) == 3 and len({p.arm_id for p in probes}) == 1

    def test_the_verdict_parameters_are_registered(self):
        rule = CF.rule(REPO)
        assert (rule["sesoi"], rule["alpha_one_sided"],
                rule["seed_robustness_min_positive"]) == (0.01, 0.025, 2)


class TestTheGovernanceAndDispatchSurfaceHolds:

    def test_the_authorization_type_refuses_an_edited_artifact(self, tmp_path):
        from stages.phase_d2.d2_screening_authorization import (
            D2ConfirmationAuthorization, D2ScreeningRefused,
        )

        path = tmp_path / "edited.json"
        path.write_text(json.dumps({"authorization_sha256": "x" * 64,
                                    "schema": "wrong"}))
        with pytest.raises(D2ScreeningRefused):
            D2ConfirmationAuthorization.load(path)

    def test_the_shell_dispatch_branch_exists_for_this_kind(self):
        src = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"
               ).read_text()
        branch = src.split('"$SESSION_KIND" = "d2_confirmation"', 1)[1]
        branch = branch.split("\nelif", 1)[0]
        assert "D2ConfirmationAuthorization.load(" in branch
        assert "assert a.authorizes_d2_confirmation is True" in branch
        for flag in ("authorizes_d2_screening", "allows_beam_search",
                     "authorizes_candidate_reselection",
                     "authorizes_reference_retraining",
                     "authorizes_fresh_confirmation", "authorizes_d3",
                     "authorizes_promotion", "automatic_followon_start"):
            assert f"assert a.{flag} is False" in branch, flag

    def test_the_pod_entry_points_are_present_and_parse(self):
        import ast

        for rel in ("scripts/stages/stage-1/phase_d2/"
                    "autoinit_d2_screening_driver.py",
                    "scripts/stages/stage-1/phase_d2/d2_confirmation.py",
                    "scripts/stages/stage-1/phase_d2/rescore_d2_probe.py",
                    "scripts/stages/stage-1/phase_d1/score_d1_confirmation.py",
                    "scripts/shared/evaluation/uncapped_eval_token_ids.py"):
            ast.parse((REPO / rel).read_text())

    def test_the_artifact_specs_load_through_the_real_loader(self):
        import importlib.util

        src = REPO / "scripts/shared/pod/collect_artifacts.py"
        spec = importlib.util.spec_from_file_location("_pf_collect_d2c", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for rel in ("configs/stages/stage-1/phase_d2/"
                    "d2_confirmation_artifacts.json",
                    "configs/stages/stage-1/phase_d2/"
                    "d2_confirmation_artifacts_failed.json"):
            assert mod.load_specs(str(REPO / rel)), rel
