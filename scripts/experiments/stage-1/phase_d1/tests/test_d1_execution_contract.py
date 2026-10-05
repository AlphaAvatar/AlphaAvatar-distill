"""The D1 execution contract, proved before the first expansion.

`d1_design.json :: execution_wiring_required` named three things the core PERMITS a
caller to omit, each of which yields a run whose records are valid-looking and
scientifically unusable, and it named the test owed when a driver existed. This is
that test.

It is an EXPERIMENT-suite test, not a core one: it checks that THIS experiment's
session implements THIS experiment's frozen design. The reusable halves — that a
search carries a distribution support at all, that a measurer's partition is
checked per measurement — live in `tests/initialization/`.

The session is built for real on the CPU with the frozen assets, because the
question is what the wiring produces and a mocked evaluator would be testing the
mock. No GPU: none of these facts is a kernel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/autoinit"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.phase_d1 import d1_session as S  # noqa: E402


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """One wired TREATMENT session, on the CPU, against the frozen assets."""
    S._register_frozen_operators()
    return S.build_session(
        arm=S.TREATMENT_ARM, workdir=tmp_path_factory.mktemp("d1"),
        run_id="contract-test", device="cpu")


@pytest.fixture(scope="module")
def design():
    return S.design()


class TestTheThreeRequirementsTheDesignNamed:

    def test_the_frozen_suite_content_hash_reaches_the_evaluator(self, session):
        """Otherwise the protocol id cannot tell two suites with the same
        structure and different items apart, and `suite_content_identity` is the
        string "unbound"."""
        content = session.suite_content_sha256
        assert content and len(content) == 64, content
        assert session.evaluator.suite_content_sha256 == content
        #: And it is the ASSET's own value, not something recomputed here.
        manifest = json.loads(
            (REPO / S.STATE_EVAL_ROOT / "manifest.json").read_text())
        assert content == manifest["content_sha256"]

    def test_the_arm_policy_and_the_declared_environment_are_both_bound(
            self, session, design):
        from aadistill.initialization.scoring.positions import SUPERVISED_TARGET_V1

        assert session.config.position_policy.policy_hash == \
            SUPERVISED_TARGET_V1.policy_hash
        assert session.evaluator.position_policy.policy_hash == \
            SUPERVISED_TARGET_V1.policy_hash
        #: The design's own declared hash for the treatment arm, so this cannot
        #: pass against a policy the design does not name.
        assert SUPERVISED_TARGET_V1.policy_hash == \
            design["scoring_policy"]["treatment"]["policy_hash"]
        env = S.numerics()
        assert env.device_type == "cuda"
        assert env.compute_dtype == "bfloat16"
        assert env.accumulation_dtype == "float32"

    def test_the_protocol_id_is_declared_and_equals_the_evaluators(self, session):
        """DECLARED, not learned from the first measurement: the config hash must
        carry the protocol before any work is done under it."""
        declared = session.config.measurement_protocol_id
        assert declared, "the config declares no protocol id"
        assert declared == session.evaluator.measurement_protocol_id

    def test_a_search_built_from_it_accepts_the_declaration(self, session,
                                                            tmp_path):
        """The construction check is the one that fires before anything expensive,
        so the session must satisfy it rather than merely look right."""
        import dataclasses

        from aadistill.initialization.planning.search import BeamSearch, SearchError
        from aadistill.initialization.specs.arch import get_adapter

        ev = session.evaluator
        ok = BeamSearch(
            adapter=get_adapter("qwen3"), config=session.config,
            root_teacher_id="contract/teacher", root_teacher_sha256="ab" * 32,
            root_loader=lambda: (_ for _ in ()).throw(
                AssertionError("no teacher is loaded by a contract test")),
            calibration_loader=lambda profile: [],
            measurer=lambda m, d: ev.evaluate(m, d),
            execution=S.execution(), numerics=S.numerics())
        assert ok.measurement_protocol_id == ev.measurement_protocol_id
        #: And a session whose protocol was taken under a DIFFERENT partition is
        #: refused at construction, which is what makes the equality meaningful.
        from aadistill.initialization.scoring.support import FULL_VOCAB_V1
        full = dataclasses.replace(session.config,
                                   distribution_support=FULL_VOCAB_V1)
        with pytest.raises(SearchError):
            BeamSearch(
                adapter=get_adapter("qwen3"), config=full,
                root_teacher_id="contract/teacher", root_teacher_sha256="ab" * 32,
                root_loader=lambda: None,
                calibration_loader=lambda profile: [],
                measurer=ev, execution=S.execution(), numerics=S.numerics())


class TestTheSupportTheDesignDidNotHaveToName:
    """It did not exist when the contract was written; it is load-bearing now."""

    def test_the_search_and_the_evaluator_reduce_over_the_d_series_partition(
            self, session):
        from experiments.phase_d_series import scoring_protocol as SP

        want = SP.D_SERIES_SUPPORT.as_dict()
        assert session.config.distribution_support.as_dict() == want
        assert session.evaluator.distribution_support.as_dict() == want
        assert want["top_k"] == SP.D_SERIES_TOP_K == 200

    def test_the_partition_is_in_the_config_hash(self, session):
        """A coarsened estimand must not share an identity with a full-vocab run."""
        import dataclasses

        from aadistill.initialization.scoring.support import FULL_VOCAB_V1

        full = dataclasses.replace(session.config,
                                   distribution_support=FULL_VOCAB_V1)
        assert full.config_hash != session.config.config_hash
        assert "distribution_support" in session.config.as_dict()
        assert "distribution_support" not in full.as_dict()


class TestTheFrozenScopeIsWhatRuns:
    """Everything the funding amendment froze, checked against the design."""

    def test_the_execution_protocol_is_bsz3_length_sorted(self, session, design):
        proto = design["execution_protocol"]
        assert proto["micro_batch_size"] == 3
        assert proto["calibration_batch_packing"] == "length_sorted_v1"
        ex = S.execution()
        assert ex.micro_batch_size == proto["micro_batch_size"]
        assert ex.calibration_batch_packing == proto["calibration_batch_packing"]
        assert session.evaluator.execution.micro_batch_size == 3

    def test_the_beam_is_width_6_with_one_warmup_level(self, session, design):
        beam = design["search_stage"]["schedule"]
        assert (beam["width"], beam["warmup_levels"]) == (6, 1)
        assert session.config.schedule.width == 6
        assert session.config.schedule.warmup_levels == 1

    def test_the_space_is_the_four_frozen_operators(self, session, design):
        declared = design["search_stage"]["frozen_implementations"]
        #: KIND -> impl_id. The kinds are asserted too, so a space that dropped a
        #: whole structural kind fails here rather than looking like four entries.
        assert set(declared) == {"DEPTH", "FFN", "RESIDUAL_WIDTH", "ATTENTION"}
        frozen = set(declared.values())
        assert frozen == {
            "depth.causal_kl_greedy_v1", "ffn.activation_importance_v0",
            "width.global_pca_v0", "attention.activation_importance_v1"}
        assert set(session.config.allowed_impls) == frozen

    def test_both_calibration_profiles_are_active(self, session, design):
        declared = set(design["search_stage"]["profiles"])
        assert declared == {"calib.domain_balanced@v1", "calib.reasoning_heavy@v2"}
        assert {p.qualified_id for p in session.config.profiles} == declared

    def test_the_target_is_the_frozen_a3_path_target(self, session):
        """Not re-derived: a session must not invent the experiment's own shape."""
        from experiments.phase_a3 import a3_session as A3S

        spec = A3S.path_spec(workdir_device="cpu")
        assert session.config.target_spec.spec_hash == spec.target_spec.spec_hash
        assert session.config.seed == spec.seed

    def test_the_recovery_recipe_is_the_frozen_one(self, design):
        assert design["recovery_recipe"] == "E1_KD_HEAVY_0860K"

    def test_the_behavioural_design_is_top_2_with_2_and_3_seeds(self, design):
        b = design["behavioural_design"]
        assert (b["top_k"], b["screening_seeds"], b["confirmation_seeds"]) == \
            (2, 2, 3)
        assert b["screening_probes"] + b["confirmation_probes"] == 12


class TestTheContractCheckerItself:
    """A checker that has only ever seen a correct session is not known to check.

    Each mutation is one requirement, broken on its own.
    """

    def test_the_wired_session_passes(self, session):
        report = S.assert_session_contract(session)
        assert report["arm"] == S.TREATMENT_ARM
        assert len(report["checked"]) >= 8
        assert report["_authorizes"].startswith("nothing")

    @pytest.mark.parametrize("what,mutate", [
        ("the protocol id is unset",
         lambda s, d: d.replace(s.config, measurement_protocol_id=None)),
        ("the protocol id is somebody else's",
         lambda s, d: d.replace(s.config, measurement_protocol_id="f" * 64)),
        ("the support falls back to the full vocabulary",
         lambda s, d: d.replace(s.config, distribution_support=_full_vocab())),
        ("the arm is scored by the control's policy",
         lambda s, d: d.replace(s.config, position_policy=_control())),
        ("the beam is widened",
         lambda s, d: d.replace(s.config, schedule=_wider(s.config.schedule))),
        ("an operator is dropped from the space",
         lambda s, d: d.replace(
             s.config, allowed_impls=tuple(sorted(s.config.allowed_impls))[:3])),
        ("a profile is dropped",
         lambda s, d: d.replace(s.config, profiles=s.config.profiles[:1])),
        ("the session claims a different design revision",
         lambda s, d: d.replace(s.config, run_id=s.config.run_id)),
    ])
    def test_each_broken_requirement_is_refused(self, session, what, mutate):
        import dataclasses

        broken = dataclasses.replace(
            session, config=mutate(session, dataclasses))
        if what.startswith("the session claims"):
            broken = dataclasses.replace(broken, design_hash="0" * 24)
        with pytest.raises(S.D1SessionError, match="does not implement"):
            S.assert_session_contract(broken)

    def test_an_unbound_suite_content_hash_is_refused(self, session):
        import dataclasses

        for bad in (None, "", "unbound", "UNBOUND_SUITE_CONTENT"):
            broken = dataclasses.replace(session, suite_content_sha256=bad)
            with pytest.raises(S.D1SessionError, match="does not implement"):
                S.assert_session_contract(broken)

    def test_an_unknown_arm_cannot_be_built(self):
        with pytest.raises(S.D1SessionError, match="unknown D1 arm"):
            S.position_policy("whatever")


class TestTheControlArmIsAlsoWired:
    """Both arms, because a contract satisfied by one of two is half a contract."""

    def test_it_builds_and_passes_with_the_incumbent_policy(self, tmp_path):
        from aadistill.initialization.scoring.positions import ALL_POSITIONS_V1

        S._register_frozen_operators()
        control = S.build_session(arm=S.CONTROL_ARM, workdir=tmp_path,
                                  run_id="contract-control", device="cpu")
        S.assert_session_contract(control)
        assert control.config.position_policy.policy_hash == \
            ALL_POSITIONS_V1.policy_hash

    def test_the_two_arms_have_different_protocol_ids_and_config_hashes(
            self, session, tmp_path):
        S._register_frozen_operators()
        control = S.build_session(arm=S.CONTROL_ARM, workdir=tmp_path,
                                  run_id=session.config.run_id, device="cpu")
        assert control.config.measurement_protocol_id != \
            session.config.measurement_protocol_id
        assert control.config.config_hash != session.config.config_hash


def _full_vocab():
    from aadistill.initialization.scoring.support import FULL_VOCAB_V1

    return FULL_VOCAB_V1


def _control():
    from aadistill.initialization.scoring.positions import ALL_POSITIONS_V1

    return ALL_POSITIONS_V1


def _wider(schedule):
    import dataclasses

    return dataclasses.replace(schedule, width=schedule.width + 1)
