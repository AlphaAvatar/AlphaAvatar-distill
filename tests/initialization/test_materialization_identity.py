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
    root_materialization_id,
)

#: The two protocols A3 compared.
A_BSZ1 = ExecutionConfig(micro_batch_size=1,
                         calibration_batch_packing="original_order_v1")
A_BSZ3 = ExecutionConfig(micro_batch_size=3,
                         calibration_batch_packing="length_sorted_v1")

POD = NumericalEnvironment(device_type="cuda", compute_dtype="bfloat16")
SEMANTIC = "a3semanticstateid" + "0" * 15

#: A stand-in parent materialization. Every child binds one, so the helper
#: supplies a fixed value and the lineage tests vary it deliberately.
PARENT = "parentmaterialization" + "0" * 11


def identity(execution, environment=POD, semantic=SEMANTIC, parent=PARENT):
    return MaterializationIdentity.build(semantic_state_id=semantic,
                                         execution=execution,
                                         environment=environment,
                                         parent_materialization_id=parent)


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
            materialization_id("", "fingerprint", PARENT)

    def test_no_parent_is_refused_by_naming_the_consumed_bytes(self):
        """The third required term, and it has no default on purpose.

        An optional parent would reintroduce the collision one level up the
        tree: two children of differently materialized parents, same path, same
        protocol, sharing one id and therefore able to resume each other.
        """
        with pytest.raises(MaterializationError) as caught:
            materialization_id(SEMANTIC, "fingerprint", "")
        assert "parent materialization" in str(caught.value)
        assert "root_materialization_id" in str(caught.value)

    def test_no_fingerprint_is_refused_by_naming_the_collision(self):
        """And the assertion is on the CONTENT, not on a sentence.

        This matched the phrase "collision A3 found" and broke the moment the
        module's prose was rewritten to keep a campaign's experiment label out
        of reusable core — a refusal that still says exactly the right thing,
        failed by a test that had locked how it said it. What matters is that
        the message names the missing fingerprint and the collision it prevents;
        the words are the module's to choose.
        """
        with pytest.raises(MaterializationError) as caught:
            materialization_id(SEMANTIC, "", PARENT)
        message = str(caught.value)
        assert "fingerprint" in message
        assert "collision" in message

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


class TestParentLineage:
    """A child's bytes are a function of the bytes it consumed.

    Without the parent term, two children of two differently materialized
    parents — same path, same execution protocol — would share one
    materialization id and could resume each other. That is the original
    collision one level up the tree, and it is the reason the term is required
    rather than optional.
    """

    def test_a_different_parent_is_a_different_materialization(self):
        a = identity(A_BSZ1, parent="parent-A" + "0" * 24)
        b = identity(A_BSZ1, parent="parent-B" + "0" * 24)
        assert a.semantic_state_id == b.semantic_state_id
        assert a.numerical_execution_fingerprint == \
            b.numerical_execution_fingerprint
        assert a.materialization_id != b.materialization_id

    def test_all_three_terms_move_it_independently(self):
        base = identity(A_BSZ1)
        variants = {
            base.materialization_id,
            identity(A_BSZ3).materialization_id,                     # protocol
            identity(A_BSZ1, semantic="b" * 32).materialization_id,  # path
            identity(A_BSZ1, parent="c" * 32).materialization_id,    # parent
        }
        assert len(variants) == 4, (
            "each of the three terms must move the id on its own; "
            f"got {len(variants)} distinct ids from four combinations")

    def test_the_lineage_is_readable_not_only_hashed(self):
        """A record should be able to say WHICH bytes it consumed, not merely
        that it consumed some. Carried as a field for that reason."""
        child = identity(A_BSZ1)
        assert child.parent_materialization_id == PARENT
        assert child.as_dict()["parent_materialization_id"] == PARENT

    def test_binding_the_digest_keeps_the_lineage(self):
        bound = identity(A_BSZ3).bind("7dd2f6f6980b")
        assert bound.parent_materialization_id == PARENT
        assert bound.materialization_id == identity(A_BSZ3).materialization_id


class TestTheRootIdentity:
    """A root is not something this project built, so its materialization is
    its published revision — and nothing else."""

    def test_it_is_the_pinned_teacher_and_not_the_execution(self):
        a = MaterializationIdentity.root(
            semantic_state_id=SEMANTIC, root_teacher_id="org/teacher",
            root_teacher_sha256="ab" * 32)
        b = MaterializationIdentity.root(
            semantic_state_id=SEMANTIC, root_teacher_id="org/teacher",
            root_teacher_sha256="ab" * 32)
        assert a.materialization_id == b.materialization_id
        #: No parent, by construction.
        assert a.parent_materialization_id is None
        #: And the fingerprint IS the root id: there was no execution of ours to
        #: fingerprint, and claiming the teacher's bytes depend on our batch
        #: size would be false.
        assert a.numerical_execution_fingerprint == a.materialization_id

    def test_a_different_revision_is_a_different_root(self):
        a = MaterializationIdentity.root(
            semantic_state_id=SEMANTIC, root_teacher_id="org/teacher",
            root_teacher_sha256="ab" * 32)
        b = MaterializationIdentity.root(
            semantic_state_id=SEMANTIC, root_teacher_id="org/teacher",
            root_teacher_sha256="cd" * 32)
        assert a.materialization_id != b.materialization_id

    def test_an_unpinned_root_is_refused(self):
        for bad in ({"root_teacher_id": "", "root_teacher_sha256": "ab" * 32},
                    {"root_teacher_id": "org/t", "root_teacher_sha256": ""}):
            with pytest.raises(MaterializationError, match="unpinned root|needs the teacher"):
                root_materialization_id(**bad)

    def test_it_names_no_experiment(self):
        """Generic by construction: it takes the two fields that pin ANY
        teacher, whatever family, stage or scale."""
        import inspect

        params = set(inspect.signature(root_materialization_id).parameters)
        assert params == {"root_teacher_id", "root_teacher_sha256"}


class TestTheFingerprintGrowthRule:
    """Item 5: the boundary is written down, so a future field is added
    deliberately rather than by whoever hits the problem first."""

    def test_the_rule_names_both_sides(self):
        from aadistill.initialization.specs.materialization import (
            FINGERPRINT_GROWTH_RULE,
        )

        rule = FINGERPRINT_GROWTH_RULE.lower()
        assert "in:" in rule and "out:" in rule
        #: The in-side must name an arithmetic-path control, the out-side a
        #: where/when fact. Checked by category rather than by exact wording.
        assert any(w in rule for w in ("backend", "kernel", "attention"))
        assert any(w in rule for w in ("ordinal", "provider", "host"))

    def test_the_declared_fields_are_all_arithmetic_path(self):
        """No `where`/`when` field may be in the fingerprint itself."""
        from aadistill.initialization.specs.materialization import (
            FINGERPRINT_FIELDS,
        )

        forbidden = ("provider", "pod", "ordinal", "host", "driver", "run_id",
                     "workdir", "utc", "log")
        for field in FINGERPRINT_FIELDS:
            assert not any(f in field for f in forbidden), field
