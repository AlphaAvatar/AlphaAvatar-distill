"""D2's search chain at $0: preregistration, session contract, driver pieces.

    pytest scripts/stages/stage-1/phase_d2/tests

The session is built for real -- the frozen suite, the registries, the A3
target -- on the CPU, and checked against the preregistration. The driver's
GPU stages are not run here; what IS tested is every decision they make
from data: the grant binding, the committed-set size, the KL-only control and
the objective telemetry the cost model reads.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]

from stages.phase_d2 import autoinit_d2_driver as DRV  # noqa: E402
from stages.phase_d2 import d2_session as S  # noqa: E402
from stages.phase_d2 import write_d2_preregistration as W  # noqa: E402

SUITE = REPO / "artifacts/stages/stage-1/state_eval_v1/items.jsonl"


def _grant(tmp_path, **over):
    """A real `D2Authorization`, serialized by its own `as_dict`."""
    from stages.phase_d2.d2_authorization import (
        AUTHORIZED_STAGES, D2_SEARCH_POLICY, STAGE_CONDITIONS, D2Authorization,
    )
    contract = {"preregistration_hash": "p" * 64, "measurement_protocol_id": "m",
                "config_hash": "c", "objective_hash": "o",
                "ranking_policy_hash": "r"}
    fields = dict(
        authorization_id="autoinit.v1.phase_d2", granted_utc="2026-10-13T00:00:00Z",
        granted_by="test", plan_id="autoinit.v1.phase_d2",
        plan_hash=contract["preregistration_hash"], expected_usd=11.0,
        hard_cap_usd=25.0, authorized_stages=AUTHORIZED_STAGES,
        stage_conditions=STAGE_CONDITIONS, scope_note="test",
        authorized_session_commit="a" * 40,
        harness_source_digest="h" * 64, harness_source_files=("x.py",),
        preregistration_hash=contract["preregistration_hash"], run_id="r1",
        measurement_protocol_id="m", config_hash="c", suite_content_sha256="s",
        objective_hash="o", ranking_policy_hash="r",
        pricing={"pricing_in_force": "records_max"},
        action_policy=D2_SEARCH_POLICY)
    fields.update(over)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(D2Authorization(**fields).as_dict()))
    return str(path), contract


class TestThePreregistration:

    def test_the_committed_record_is_what_the_writer_derives(self):
        assert W.main([]) == 0, "run write_d2_preregistration.py --write"

    def test_it_changes_one_variable_from_d1(self):
        doc = S.preregistration()
        d1 = json.loads((REPO / W.D1_DESIGN).read_text())["search_stage"]
        assert doc["search"]["frozen_implementations"] == \
            d1["frozen_implementations"]
        assert doc["search"]["profiles"] == d1["profiles"]
        assert doc["search"]["beam"] == d1["beam"]
        assert doc["search"]["ranking_policy"]["id"] == \
            "beam.pareto_scoring_objective@v1"
        assert doc["retention"]["k"] == 4
        assert doc["authorizes"] == "nothing"

    def test_the_hash_covers_the_science_not_the_status(self):
        doc = S.preregistration()
        science = {k: v for k, v in doc.items()
                   if k not in ("status", "preregistration_hash", "budget",
                                "authorizes")}
        from aadistill.infrastructure.manifest import sha256_json
        assert sha256_json(science) == doc["preregistration_hash"]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not SUITE.is_file():
        pytest.skip("state_eval_v1 is not staged on this host")
    return S.build_session(workdir=tmp_path_factory.mktemp("d2"),
                           run_id="d2_test", device="cpu")


class TestTheSessionContract:

    def test_a_built_session_implements_the_preregistration(self, session):
        contract = S.assert_session_contract(session)
        doc = S.preregistration()
        assert contract["objective_hash"] == doc["objective"]["objective_hash"]
        assert session.config.scoring_objective is not None
        assert session.evaluator.scoring_objective is not None
        assert session.config.seed == doc["search"]["seed"]

    def test_the_measurer_exposes_what_the_search_checks(self, session):
        m = session.measurer()
        assert m.scoring_objective is session.evaluator.scoring_objective
        assert m.distribution_support is session.evaluator.distribution_support
        assert m.measurement_protocol_id == session.config.measurement_protocol_id

    def test_a_row_selection_the_evaluator_does_not_run_is_refused(self, session):
        with pytest.raises(S.D2SessionError, match="row selection"):
            S.assert_session_contract(
                S.D2Session(**{**session.__dict__, "prediction_rows": "scored_v1"}))

    def test_another_preregistration_revision_is_refused(self, session, tmp_path):
        doc = S.preregistration()
        doc["search"]["seed"] = int(doc["search"]["seed"]) + 1
        doc["preregistration_hash"] = "0" * 64
        target = tmp_path / S.PREREGISTRATION
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(doc))
        with pytest.raises(S.D2SessionError) as err:
            S.assert_session_contract(session, repo_root=tmp_path)
        assert "seed" in str(err.value) and "revision" in str(err.value)

    def test_the_d2_identity_differs_from_d1s(self, session):
        from stages.phase_d1 import d1_session as D1S

        d1 = D1S.build_session(arm=D1S.TREATMENT_ARM, workdir=Path("/tmp/d1x"),
                               run_id="d2_test", device="cpu")
        assert d1.config.measurement_protocol_id != \
            session.config.measurement_protocol_id
        assert d1.config.config_hash != session.config.config_hash


class TestTheDriverDecisions:

    def test_the_grant_must_bind_this_sessions_identities(self, tmp_path):
        from stages.phase_d2.d2_authorization import D2AuthorizationRefused

        auth, contract = _grant(tmp_path)
        out = DRV.load_authorization(auth, contract=contract, run_id="r1",
                                     prediction_rows="all_v1")
        assert out["hard_cap_usd"] == 25.0
        with pytest.raises(D2AuthorizationRefused):
            DRV.load_authorization(auth, contract=contract, run_id="other",
                                   prediction_rows="all_v1")
        for key in ("config_hash", "objective_hash"):
            with pytest.raises(DRV.D2DriverError, match=key):
                DRV.load_authorization(auth, contract={**contract, key: "x"},
                                       run_id="r1", prediction_rows="all_v1")
        with pytest.raises(DRV.D2DriverError, match="prediction_rows"):
            DRV.load_authorization(auth, contract=contract, run_id="r1",
                                   prediction_rows="scored_v1")

    def test_the_kl_only_control_is_reported_and_never_used(self):
        result = SimpleNamespace(finalists=lambda policy, k: SimpleNamespace(
            selected_ids=("s1", "s9", "s3", "s7")[:k]))
        out = DRV.kl_only_control(result, ["s1", "s2", "s3", "s4"], 4)
        from aadistill.initialization.planning.ranking import PARETO_V1

        assert out["ranking_policy"] == PARETO_V1.qualified_id
        assert out["overlap_with_committed"] == ["s1", "s3"]
        assert out["used_for_selection"] is False

    def test_a_short_committed_set_is_not_a_complete_search(self):
        session = SimpleNamespace(config=SimpleNamespace(policy="P"))
        result = SimpleNamespace(finalists=lambda policy, k: SimpleNamespace(
            selected=("only-one",), retention="quality_only"))
        with pytest.raises(DRV.D2DriverError, match="NOT COMPLETE"):
            DRV.commit(session, result, Path("/nonexistent"), 4)

    def test_objective_telemetry_sums_what_the_cost_model_reads(self, tmp_path):
        rows = [
            {"impl_id": "depth.causal_kl_greedy_v1", "operator_seconds": 100.0,
             "operator_timing": {"objective_seconds": 4.0},
             "reference_cache": {"mode": "cached"},
             "state_evaluation_seconds": {"seconds": 10.0}},
            {"impl_id": "ffn.activation_importance_v0", "operator_seconds": 9.0,
             "operator_timing": {}, "state_evaluation_seconds": {"seconds": 11.0}}]
        path = tmp_path / "telemetry.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = DRV.objective_telemetry(path)
        assert out["depth_expansions"] == 1
        assert out["depth_objective_seconds"] == 4.0
        assert out["state_evaluation_seconds"] == 21.0
        assert out["depth_cache_modes"] == ["cached"]
        assert DRV.objective_telemetry(tmp_path / "absent") == {"available": False}

    def test_the_driver_uses_d1s_terminal_markers(self):
        from stages.phase_d1 import autoinit_d1_driver as D1

        assert (DRV.SUCCESS_MARKER, DRV.FAILURE_MARKER, DRV.CHECK_ONLY_MARKER) \
            == (D1.SUCCESS_MARKER, D1.FAILURE_MARKER, D1.CHECK_ONLY_MARKER)
