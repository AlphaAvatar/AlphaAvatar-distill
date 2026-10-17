"""The D3 PROBES pod's blocking CPU gate. Pod-faithful.

`SetupManifest.test_paths` for the probes points HERE, and only a probes pod
runs it -- which exists only after a structural run completed with
STRUCTURAL_CHANGE. What a mis-staged probes pod would get wrong, checked on its
CPU before anything trains:

* the bindings agree: the registration, the shared protocol and the
  structural closeout that names the D3 initialization;
* this checkout's trainer, generator and scorer are the protocol's pins;
* the staged bytes -- the confirmation battery, the recovery pack, the
  evaluation tokenizer -- match their pins;
* the schedule is the registered one: one D3 arm, the three D1 confirmation
  seeds, q2's and B's imported units as comparators;
* the governance and dispatch surface loads and refuses what it must.

DELIBERATELY ABSENT: the D3 product's bytes (the driver hashes them at all
four identities before anything trains) and anything touching the network.
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

from stages.phase_d3 import d3_probes as P  # noqa: E402


def _a_structural_change_is_on_record() -> bool:
    """Whether this checkout carries a structural closeout that names a D3
    initialization. The premise is a COMMITTED RECORD, so it travels in the
    bundle and a pod sees what the dev host sees: a probes pod cannot exist
    without it (its launcher's relay inputs are read from it). Before the
    structural session, the dev suite has nothing for this gate to check."""
    try:
        P.structural_product(REPO)
    except P.D3ProbesError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _a_structural_change_is_on_record(),
    reason=("no structural run has completed with STRUCTURAL_CHANGE; this is "
            "the probes pod's gate and a probes pod needs that record"))


class TestTheBindingsAgree:

    def test_the_registration_verifies(self):
        assert len(P.registration(REPO)["registration_hash"]) == 64

    def test_the_structural_product_is_named_by_one_completed_run(self):
        product = P.structural_product(REPO)
        assert len(product["identities"]["artifact_digest"]) == 64

    def test_this_checkout_is_the_protocols_trainer_generator_and_scorer(self):
        from stages.d_series import shared_evaluation as SE

        assert SE.tree_matches(P.protocol(REPO), REPO) == {
            "trainer": [], "generation": [], "scoring": []}

    def test_the_recovery_is_the_protocols(self):
        assert P.recovery(REPO)["recipe_id"] == "e1_p1_kd_heavy@0.86M"


class TestTheStagedBytesMatchTheirPins:

    def test_the_confirmation_role_rehashes(self):
        bat = P.battery(REPO)
        assert bat["role"] == "d1_confirmation"
        assert (bat["n_prompts"], bat["n_scorable"]) == (950, 850)

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


class TestTheScheduleIsTheRegisteredOne:

    def test_three_probes_of_the_one_d3_arm_at_the_confirmation_seeds(self):
        probes = P.probes(REPO, arm_root="/nonexistent")
        assert [p.seed for p in probes] == [396880939, 766527341, 1004399678]
        assert {p.arm_id for p in probes} == {P.ARM_ID}

    def test_the_comparators_are_q2_and_b_imported_units(self):
        units = P.comparator_units(REPO)
        assert sorted(units) == ["B", "q2"]
        assert all(len(v) == 3 for v in units.values())


class TestTheGovernanceAndDispatchSurfaceHolds:

    def test_the_authorization_type_refuses_an_edited_artifact(self, tmp_path):
        from stages.phase_d3.d3_authorization import (
            D3AuthorizationRefused, D3ProbesAuthorization,
        )

        path = tmp_path / "edited.json"
        path.write_text(json.dumps({"authorization_sha256": "x" * 64,
                                    "schema": "wrong"}))
        with pytest.raises(D3AuthorizationRefused):
            D3ProbesAuthorization.load(path)

    def test_the_shell_dispatch_branch_exists_for_this_kind(self):
        src = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh"
               ).read_text()
        branch = src.split('"$SESSION_KIND" = "d3_probes"', 1)[1]
        branch = branch.split("\nelif", 1)[0]
        assert "D3ProbesAuthorization.load(" in branch
        assert "assert a.authorizes_d3_probes is True" in branch
        for flag in ("authorizes_d3_structural",
                     "authorizes_fixed_path_materialization",
                     "authorizes_reference_retraining",
                     "authorizes_screening_seeds",
                     "authorizes_fresh_confirmation", "authorizes_promotion",
                     "authorizes_automatic_followon_start"):
            assert f"assert a.{flag} is False" in branch, flag
