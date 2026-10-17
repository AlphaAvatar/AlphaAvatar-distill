"""The generic forward-baseline resolver, on synthetic records.

Nothing here reads this project's selection: a property pinned by construction
keeps holding whichever baseline is selected. The real record's agreement with
its history is the application layer's test (`scripts/shared/tests`).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

import pytest

from aadistill.initialization.baseline import (
    STATE_EVALUATION_REFERENCE, BaselineError, load_baseline,
    resolve_selection,
)
from aadistill.initialization.execution import ExecutionConfig
from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1
from aadistill.initialization.scoring.kl_reference import ROOT_REFERENCE_V1
from aadistill.initialization.scoring.positions import (
    ALL_POSITIONS_V1, SUPERVISED_TARGET_V1,
)
from aadistill.initialization.specs.materialization import (
    NumericalEnvironment, numerical_execution_fingerprint,
)

ENV = NumericalEnvironment(device_type="cuda", compute_dtype="bfloat16",
                           accumulation_dtype="float32")
FORWARD = {"micro_batch_size": 2, "calibration_batch_packing": "length_sorted_v1",
           "prediction_rows": "scored_v1"}
HISTORICAL = dict(FORWARD, prediction_rows="all_v1")


def _fp(raw):
    return numerical_execution_fingerprint(ExecutionConfig(**raw), ENV)


def _artifact(label, execution, digest_char):
    return {"label": label, "semantic_state_id": "s" * 32,
            "artifact_digest": digest_char * 64, "weights_digest": "w" * 64,
            "single_shard_sha256": "h" * 64, "arch_signature": "a" * 64,
            "execution": dict(execution),
            "numerical_execution_fingerprint": _fp(execution)}


def record() -> dict:
    return {
        "schema": "aadistill.initialization.forward_baseline/v1",
        "baseline_id": "toy.baseline", "version": 1,
        "method": {
            "operators": {"KIND_A": {"impl_id": "kind_a.toy_v1",
                                     "signature_hash": "1" * 64},
                          "KIND_B": {"impl_id": "kind_b.toy_v1",
                                     "signature_hash": "2" * 64}},
            "position_policy": {"qualified_id": SUPERVISED_TARGET_V1.qualified_id,
                                "policy_hash": SUPERVISED_TARGET_V1.policy_hash},
            "distribution_support": {"support_id": "reference_topk_tail_v1",
                                     "top_k": 7},
            "kl_reference": {"operator_local": {"KIND_A": "parent_state"},
                             "state_evaluation": STATE_EVALUATION_REFERENCE},
            "search": {
                "schedule": {"qualified_id": SCHEDULE_V1.qualified_id,
                             "schedule_hash": SCHEDULE_V1.schedule_hash},
                "ranking": {"qualified_id": PARETO_V1.qualified_id,
                            "policy_hash": PARETO_V1.policy_hash}}},
        "execution": dict(FORWARD),
        "numerics": ENV.as_dict(),
        "numerical_execution_fingerprint": _fp(FORWARD),
        "reference_artifacts": [_artifact("historical", HISTORICAL, "d")],
        "forward_materializations": [],
    }


def _write(tmp_path, doc, *, role="init", select=True):
    rec = tmp_path / "record.json"
    rec.write_text(json.dumps(doc))
    sel = tmp_path / "selection.json"
    digest = hashlib.sha256(rec.read_bytes()).hexdigest()
    sel.write_text(json.dumps({
        "schema": "aadistill.initialization.baseline_selection/v1",
        "selections": {role: {"record": "record.json", "sha256": digest}}}))
    return rec, sel


@dataclass
class _Impl:
    kind: str
    signature_hash: str
    accepts_kl_reference: bool = False


def _registry(**over):
    impls = {"kind_a.toy_v1": _Impl("KIND_A", "1" * 64, True),
             "kind_b.toy_v1": _Impl("KIND_B", "2" * 64)}
    impls.update(over)

    def get(impl_id):
        if impl_id not in impls:
            raise KeyError(impl_id)
        return impls[impl_id]
    return get


class TestAValidRecordResolves:
    def test_through_its_selection(self, tmp_path):
        _, sel = _write(tmp_path, record())
        b = resolve_selection(sel, "init")
        assert b.qualified_id == "toy.baseline@v1"
        assert b.execution == ExecutionConfig(**FORWARD)
        assert b.execution_fingerprint == _fp(FORWARD)
        assert b.position_policy is SUPERVISED_TARGET_V1
        assert b.distribution_support.as_dict() == {
            "support_id": "reference_topk_tail_v1", "top_k": 7}
        assert b.schedule is SCHEDULE_V1 and b.ranking_policy is PARETO_V1
        assert b.implementation_ids() == {"KIND_A": "kind_a.toy_v1",
                                          "KIND_B": "kind_b.toy_v1"}
        assert b.require_registered_operators(_registry()) == \
            b.implementation_ids()

    def test_the_parent_reference_is_expressed_by_omission(self, tmp_path):
        doc = record()
        b = resolve_selection(_write(tmp_path, doc)[1], "init")
        assert b.kl_reference_config("KIND_A") == {}
        assert b.kl_reference_config("KIND_B") == {}

    def test_a_declared_root_reference_is_handed_on(self, tmp_path):
        doc = record()
        doc["method"]["kl_reference"]["operator_local"]["KIND_A"] = \
            ROOT_REFERENCE_V1
        b = resolve_selection(_write(tmp_path, doc)[1], "init")
        assert b.kl_reference_config("KIND_A") == {
            "kl_reference": ROOT_REFERENCE_V1}


class TestAnIdentityDoesNotCrossExecutionProfiles:
    """The defect this guards: handing a historical artifact's digest to a
    materialization under another profile as its expected bytes."""

    def test_the_forward_profile_gets_no_historical_artifact(self, tmp_path):
        b = resolve_selection(_write(tmp_path, record())[1], "init")
        assert b.reference_artifact_for(b.execution) is None
        assert b.forward_materialization() is None

    def test_the_historical_profile_gets_its_own_artifact(self, tmp_path):
        b = resolve_selection(_write(tmp_path, record())[1], "init")
        art = b.reference_artifact_for(ExecutionConfig(**HISTORICAL))
        assert art is not None and art.label == "historical"
        assert art.identities["artifact_digest"] == "d" * 64

    def test_a_shared_semantic_state_is_not_enough(self, tmp_path):
        """Same semantic state id, different fingerprint: still no match."""
        b = resolve_selection(_write(tmp_path, record())[1], "init")
        other = ExecutionConfig(**dict(HISTORICAL, micro_batch_size=1))
        assert b.reference_artifact_for(other) is None

    def test_a_forward_materialization_must_carry_the_forward_fingerprint(
            self, tmp_path):
        doc = record()
        doc["forward_materializations"] = [_artifact("wrong", HISTORICAL, "e")]
        with pytest.raises(BaselineError, match="forward profile"):
            resolve_selection(_write(tmp_path, doc)[1], "init")

    def test_a_forward_materialization_is_returned_for_the_forward_profile(
            self, tmp_path):
        doc = record()
        doc["forward_materializations"] = [_artifact("fwd", FORWARD, "f")]
        b = resolve_selection(_write(tmp_path, doc)[1], "init")
        assert b.forward_materialization().label == "fwd"


class TestTheRecordIsRefusedWhen:
    def _refused(self, tmp_path, doc, match):
        with pytest.raises(BaselineError, match=match):
            resolve_selection(_write(tmp_path, doc)[1], "init")

    def test_it_was_edited_after_selection(self, tmp_path):
        rec, sel = _write(tmp_path, record())
        doc = json.loads(rec.read_text())
        doc["version"] = 2
        rec.write_text(json.dumps(doc))
        with pytest.raises(BaselineError, match="edited after it was selected"):
            resolve_selection(sel, "init")

    def test_the_stated_fingerprint_disagrees(self, tmp_path):
        doc = record()
        doc["numerical_execution_fingerprint"] = "0" * 64
        self._refused(tmp_path, doc, "fingerprint")

    def test_an_artifact_fingerprint_disagrees_with_its_profile(self, tmp_path):
        doc = record()
        doc["reference_artifacts"][0]["numerical_execution_fingerprint"] = \
            _fp(FORWARD)
        self._refused(tmp_path, doc, "one fact")

    def test_an_artifact_lacks_an_identity(self, tmp_path):
        doc = record()
        del doc["reference_artifacts"][0]["weights_digest"]
        self._refused(tmp_path, doc, "weights_digest")

    def test_the_row_policy_is_unknown(self, tmp_path):
        doc = record()
        doc["execution"]["prediction_rows"] = "some_rows_v9"
        self._refused(tmp_path, doc, "prediction_rows")

    def test_the_position_policy_hash_disagrees(self, tmp_path):
        doc = record()
        doc["method"]["position_policy"]["policy_hash"] = \
            ALL_POSITIONS_V1.policy_hash
        self._refused(tmp_path, doc, "hashes to")

    def test_the_position_policy_is_unknown(self, tmp_path):
        doc = record()
        doc["method"]["position_policy"]["qualified_id"] = "positions.none@v1"
        self._refused(tmp_path, doc, "no scoring position policy")

    def test_the_schedule_is_not_shipped(self, tmp_path):
        doc = record()
        doc["method"]["search"]["schedule"]["schedule_hash"] = "0" * 64
        self._refused(tmp_path, doc, "schedule")

    def test_the_ranking_is_not_shipped(self, tmp_path):
        doc = record()
        doc["method"]["search"]["ranking"]["qualified_id"] = "beam.other@v1"
        self._refused(tmp_path, doc, "ranking")

    def test_state_evaluation_names_a_reference_the_evaluator_lacks(
            self, tmp_path):
        doc = record()
        doc["method"]["kl_reference"]["state_evaluation"] = "parent_state"
        self._refused(tmp_path, doc, "state evaluation reference")

    def test_an_operator_reference_is_unknown(self, tmp_path):
        doc = record()
        doc["method"]["kl_reference"]["operator_local"]["KIND_A"] = "sibling"
        self._refused(tmp_path, doc, "KL reference")

    def test_a_reference_names_an_absent_operator(self, tmp_path):
        doc = record()
        doc["method"]["kl_reference"]["operator_local"]["KIND_Z"] = \
            "parent_state"
        self._refused(tmp_path, doc, "names no operator")

    def test_the_support_is_unknown(self, tmp_path):
        doc = record()
        doc["method"]["distribution_support"] = {"support_id": "made_up_v1"}
        self._refused(tmp_path, doc, "distribution_support")

    def test_the_role_is_not_selected(self, tmp_path):
        _, sel = _write(tmp_path, record())
        with pytest.raises(BaselineError, match="selects nothing"):
            resolve_selection(sel, "other_role")

    def test_the_schema_is_wrong(self, tmp_path):
        doc = record()
        doc["schema"] = "something/v1"
        rec, _ = _write(tmp_path, doc)
        with pytest.raises(BaselineError, match="schema"):
            load_baseline(rec)


class TestTheLiveRegistryIsChecked:
    def _baseline(self, tmp_path, doc=None):
        return resolve_selection(_write(tmp_path, doc or record())[1], "init")

    def test_an_unregistered_implementation(self, tmp_path):
        b = self._baseline(tmp_path)
        reg = _registry()

        def get(impl_id):
            if impl_id == "kind_b.toy_v1":
                raise KeyError(impl_id)
            return reg(impl_id)
        with pytest.raises(BaselineError, match="not registered"):
            b.require_registered_operators(get)

    def test_a_changed_signature(self, tmp_path):
        b = self._baseline(tmp_path)
        with pytest.raises(BaselineError, match="new baseline"):
            b.require_registered_operators(_registry(
                **{"kind_b.toy_v1": _Impl("KIND_B", "9" * 64)}))

    def test_an_implementation_of_another_kind(self, tmp_path):
        b = self._baseline(tmp_path)
        with pytest.raises(BaselineError, match="implements"):
            b.require_registered_operators(_registry(
                **{"kind_b.toy_v1": _Impl("KIND_A", "2" * 64)}))

    def test_a_reference_handed_to_an_implementation_that_refuses_one(
            self, tmp_path):
        doc = copy.deepcopy(record())
        doc["method"]["kl_reference"]["operator_local"]["KIND_B"] = \
            ROOT_REFERENCE_V1
        b = self._baseline(tmp_path, doc)
        with pytest.raises(BaselineError, match="does not accept"):
            b.require_registered_operators(_registry())
