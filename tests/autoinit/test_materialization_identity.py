"""Four identities, and the collision A3 measured that needs all four.

A3 ran the incumbent ATTENTION operator at two batching protocols and produced
`53e30566c5f7` and `7dd2f6f6980b` — reproducibly, on three machines, with an
identical `result_spec_hash`, the same operator, the same hashed config and the
same semantics. One semantic state, two sets of bytes. `compute_state_id` is
deliberately blind to execution, so resume, dedup and checkpoint ownership would
have treated them as one.

These tests pin the resolution:

* `semantic_state_id` still collides, because it should — two protocols of one
  path are one hypothesis;
* `numerical_execution_fingerprint` separates them, and covers only the fields
  declared byte-affecting;
* `materialization_id` is the pair, and is what resume may key on;
* `artifact_digest` is observed and bound once.

The companion assertion lives in `test_execution_config_is_not_identity.py`,
which asserts the collision itself; this file asserts the second coordinate that
makes the collision harmless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.specs.materialization import (  # noqa: E402
    FINGERPRINT_FIELDS,
    MaterializationError,
    MaterializationIdentity,
    NumericalEnvironment,
    materialization_id,
    numerical_execution_fingerprint,
)

#: The two protocols A3 compared.
A_BSZ1 = ExecutionConfig(micro_batch_size=1,
                         calibration_batch_packing="original_order_v1")
A_BSZ3 = ExecutionConfig(micro_batch_size=3,
                         calibration_batch_packing="length_sorted_v1")

POD = NumericalEnvironment(device_type="cuda", compute_dtype="bfloat16")
SEMANTIC = "a3semanticstateid" + "0" * 15


def identity(execution, environment=POD, semantic=SEMANTIC):
    return MaterializationIdentity.build(semantic_state_id=semantic,
                                         execution=execution,
                                         environment=environment)


class TestTheA3Collision:
    def test_one_path_two_protocols_is_one_semantic_state(self):
        assert identity(A_BSZ1).semantic_state_id == \
            identity(A_BSZ3).semantic_state_id

    def test_and_two_materializations(self):
        assert identity(A_BSZ1).materialization_id != \
            identity(A_BSZ3).materialization_id

    def test_a_cross_protocol_record_is_refused_by_name(self):
        with pytest.raises(MaterializationError, match="byte-affecting"):
            identity(A_BSZ1).require_same_materialization(
                identity(A_BSZ3), what="journal entry")

    def test_the_same_protocol_is_admitted(self):
        identity(A_BSZ1).require_same_materialization(identity(A_BSZ1))
        assert identity(A_BSZ1).same_materialization(identity(A_BSZ1))

    def test_a_serialized_record_compares_the_same_way(self):
        """Because that is how it arrives: out of a journal, as a mapping."""
        record = identity(A_BSZ3).as_dict()
        assert not identity(A_BSZ1).same_materialization(record)
        assert identity(A_BSZ3).same_materialization(record)

    def test_an_absent_record_is_not_a_match(self):
        assert not identity(A_BSZ1).same_materialization({})


class TestWhatTheFingerprintCovers:
    def test_batch_size_and_packing_both_move_it(self):
        base = numerical_execution_fingerprint(A_BSZ1, POD)
        size_only = numerical_execution_fingerprint(
            ExecutionConfig(micro_batch_size=3,
                            calibration_batch_packing="original_order_v1"), POD)
        packing_only = numerical_execution_fingerprint(
            ExecutionConfig(micro_batch_size=1,
                            calibration_batch_packing="length_sorted_v1"), POD)
        assert len({base, size_only, packing_only}) == 3

    def test_the_device_class_moves_it(self):
        """A CPU dry run must not be able to satisfy a GPU resume: different
        kernels, different reductions, different bytes."""
        cpu = NumericalEnvironment(device_type="cpu", compute_dtype="float32")
        assert numerical_execution_fingerprint(A_BSZ1, cpu) != \
            numerical_execution_fingerprint(A_BSZ1, POD)

    def test_the_compute_dtype_moves_it(self):
        fp32 = NumericalEnvironment(device_type="cuda", compute_dtype="float32")
        assert numerical_execution_fingerprint(A_BSZ1, fp32) != \
            numerical_execution_fingerprint(A_BSZ1, POD)

    def test_a_device_ORDINAL_is_refused(self):
        """A3 reproduced one digest on three different rented L40S pods, so the
        particular card is not part of the identity — and an ordinal in an
        identity makes a resume depend on scheduling. AGENTS.md P3 forbids a CUDA
        ordinal in reusable core for the same reason."""
        with pytest.raises(MaterializationError, match="ordinal"):
            NumericalEnvironment(device_type="cuda:0", compute_dtype="bfloat16")

    def test_an_unstated_condition_is_refused_rather_than_defaulted(self):
        with pytest.raises(MaterializationError, match="non-empty string"):
            NumericalEnvironment(device_type="", compute_dtype="bfloat16")

    def test_the_execution_config_owns_which_of_its_fields_count(self):
        """`specs.materialization` ASKS rather than enumerates, so a new
        execution knob is declared in one place."""
        assert set(A_BSZ1.as_fingerprint()) == set(
            ExecutionConfig.FINGERPRINT_FIELDS)
        assert set(ExecutionConfig.FINGERPRINT_FIELDS) <= set(FINGERPRINT_FIELDS)

    def test_an_object_with_no_fingerprint_view_is_refused(self):
        class NotAnExecutionConfig:
            micro_batch_size = 3

        with pytest.raises(MaterializationError, match="as_fingerprint"):
            numerical_execution_fingerprint(NotAnExecutionConfig(), POD)

    def test_an_undeclared_field_is_refused_rather_than_absorbed(self):
        """A fingerprint that absorbs whatever it is handed forks on runtime
        detail and stops being an identity."""
        class Chatty(ExecutionConfig):
            def as_fingerprint(self):
                return {**super().as_fingerprint(), "log_level": "debug"}

        with pytest.raises(MaterializationError, match="FINGERPRINT_FIELDS"):
            numerical_execution_fingerprint(Chatty(), POD)

    def test_a_missing_field_is_refused(self):
        class Forgetful(ExecutionConfig):
            def as_fingerprint(self):
                return {"micro_batch_size": 1}

        with pytest.raises(MaterializationError, match="missing"):
            numerical_execution_fingerprint(Forgetful(), POD)


class TestBindingTheBytes:
    def test_the_identity_exists_before_the_bytes_do(self):
        """It is what decides whether to build them."""
        assert identity(A_BSZ3).artifact_digest is None

    def test_binding_records_the_pair(self):
        bound = identity(A_BSZ3).bind("7dd2f6f6980b")
        assert bound.artifact_digest == "7dd2f6f6980b"
        assert bound.materialization_id == identity(A_BSZ3).materialization_id

    def test_rebinding_to_different_bytes_is_refused(self):
        bound = identity(A_BSZ3).bind("7dd2f6f6980b")
        with pytest.raises(MaterializationError, match="cannot be rebound"):
            bound.bind("53e30566c5f7")

    def test_rebinding_to_the_same_bytes_is_idempotent(self):
        bound = identity(A_BSZ3).bind("7dd2f6f6980b")
        assert bound.bind("7dd2f6f6980b").artifact_digest == "7dd2f6f6980b"

    def test_an_empty_digest_is_refused(self):
        with pytest.raises(MaterializationError, match="empty artifact digest"):
            identity(A_BSZ3).bind("")


class TestRefusals:
    def test_no_semantic_id_is_refused(self):
        with pytest.raises(MaterializationError, match="semantic state id"):
            materialization_id("", "fingerprint")

    def test_no_fingerprint_is_refused_by_naming_the_collision(self):
        with pytest.raises(MaterializationError, match="collision A3 found"):
            materialization_id(SEMANTIC, "")

    def test_the_id_is_deterministic(self):
        assert identity(A_BSZ3).materialization_id == \
            identity(A_BSZ3).materialization_id

    def test_a_different_semantic_state_is_a_different_materialization(self):
        """Both coordinates matter, and this is the one a fingerprint alone
        would miss."""
        other = identity(A_BSZ3, semantic="b" * 32)
        assert other.numerical_execution_fingerprint == \
            identity(A_BSZ3).numerical_execution_fingerprint
        assert other.materialization_id != identity(A_BSZ3).materialization_id


class TestNoFakeOperatorIdentity:
    """A3 was explicitly forbidden from registering
    `attention.activation_importance_bsz3`, and for the right reason: the
    operator's semantics did not change, so an id change would lie about the
    science and multiply the registry by every execution knob forever.

    The fingerprint is a second coordinate, not a renaming of the first, and the
    registry must stay clean of execution-variant ids.
    """

    def test_no_registered_implementation_names_a_batch_size(self):
        from aadistill.initialization.operators.base import (
            registered_implementations,
        )
        from aadistill.initialization.operators.register import (
            register_builtin_operators,
        )
        from aadistill.initialization.operators.attention.gqa import (
            activation_importance,
        )
        register_builtin_operators()
        activation_importance.register()
        offenders = [i for i in registered_implementations()
                     if "bsz" in i or "batch" in i or "packed" in i]
        assert offenders == [], (
            f"{offenders} name an execution protocol in an operator id. "
            "Execution belongs in the numerical fingerprint; an operator id "
            "names an algorithm.")
