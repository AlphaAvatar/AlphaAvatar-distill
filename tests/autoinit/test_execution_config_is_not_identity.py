"""Batch size is execution provenance, never scientific identity.

`OperatorStep.identity()` feeds `compute_state_id`, and it reads `impl_id`, the
implementation signature, the profile hash, the operator's `config_hash` and the
seed. If a micro-batch size could reach any of those, then two runs of the same
science on different hardware would be different scientific states, a resume
would not find its own journal, and a frozen path would stop reproducing its own
id for a reason that has nothing to do with what was computed.

The guarantee is structural rather than an exclusion list: execution settings
live on `OperatorContext.execution`, a separate field from `OperatorContext.
config`, and only `config` is hashed. These tests pin both halves — that the
identity does not move, and that the batch size is nonetheless recorded as
evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.profiles import NO_CALIBRATION  # noqa: E402
from aadistill.initialization.execution import (  # noqa: E402
    DEFAULT_EXECUTION,
    DEFAULT_MICRO_BATCH_SIZE,
    ExecutionConfig,
    ExecutionError,
)
from aadistill.initialization.operators.attention.gqa import activation_importance as attention_activation  # noqa: E402
from aadistill.initialization.operators.base import OperatorContext  # noqa: E402
from aadistill.initialization.operators.depth.causal_kl_greedy import (  # noqa: E402
    DEPTH_CAUSAL_KL_GREEDY_V1)
from aadistill.initialization.operators.ffn.dense.activation_importance import (  # noqa: E402
    FFN_ACTIVATION_IMPORTANCE_V0)
from aadistill.initialization.operators.width.residual.global_pca import (  # noqa: E402
    WIDTH_GLOBAL_PCA_V0)
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402
from aadistill.initialization.specs.state import (  # noqa: E402
    OperatorStep,
    compute_state_id,
)

from conftest import TEACHER_GEOMETRY, build_tiny_model  # noqa: E402

BATCH_SIZES = (1, 2, 4, 16)


def ragged_items(n: int = 6, vocab: int = 128, seed: int = 909):
    torch.manual_seed(seed)
    out = []
    for k, length in enumerate((21, 13, 17, 9, 19, 11)[:n]):
        out.append({"item_id": f"i{k}",
                    "input_ids": torch.randint(1, vocab, (1, length)),
                    "domain": "general" if k % 2 else "math",
                    "subtype": "text" if k % 2 else "arith"})
    return out


@pytest.fixture
def model():
    m = build_tiny_model(TEACHER_GEOMETRY)
    m.config.use_cache = False
    return m


@pytest.fixture
def items():
    return ragged_items()


@pytest.fixture(autouse=True)
def registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


def context(model, items, batch_size, target, packing="original_order_v1"):
    """The SAME science every time; only `execution` moves."""
    return OperatorContext(
        adapter=QWEN3_ADAPTER, model=model,
        parent_spec=ArchSpec.of("qwen3", TEACHER_GEOMETRY), target_spec=target,
        profile=NO_CALIBRATION, calibration_items=items, seed=0, device="cpu",
        config={"n_calibration_items": len(items)},
        execution=ExecutionConfig(micro_batch_size=batch_size,
                                  calibration_batch_packing=packing))


# --- the structural guarantee ----------------------------------------------


class TestTheBoundaryIsStructural:
    def test_execution_is_a_separate_field_from_the_hashed_config(self, model, items):
        """Not "a key we remember to exclude" -- a different field entirely."""
        ctx = context(model, items, 8, ArchSpec.of("qwen3", TEACHER_GEOMETRY))
        assert ctx.execution.micro_batch_size == 8
        assert "micro_batch_size" not in ctx.config
        assert not any("batch" in k for k in ctx.config), (
            "an execution setting reached the mapping that is hashed into "
            "OperatorStep.config_hash")

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_the_operator_config_that_gets_hashed_does_not_move(
            self, model, items, batch_size):
        """The hash the search actually computes, at every batch size.

        Reproduces `BeamSearch._expand_one` exactly rather than approximating
        it: the value under test is the one that reaches `OperatorStep`.
        """
        ctx = context(model, items, batch_size,
                      ArchSpec.of("qwen3", TEACHER_GEOMETRY))
        config_hash = sha256_json(
            {k: v for k, v in ctx.config.items() if k != "n_calibration_items"})
        assert config_hash == sha256_json({})

    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_the_state_id_is_the_same_at_every_batch_size(self, batch_size):
        """The quantity that actually matters, end to end.

        Same parent, same operator, same calibration, same seed, different
        micro-batch size -> the same content-derived state id. A resume keyed on
        this must find the work a differently-batched run left behind.
        """
        ctx = context(build_tiny_model(TEACHER_GEOMETRY), ragged_items(),
                      batch_size, ArchSpec.of("qwen3", TEACHER_GEOMETRY))
        step = OperatorStep(
            index=0, kind="ATTENTION",
            impl_id="attention.activation_importance_v1",
            impl_signature_hash="sig", profile_id=NO_CALIBRATION.qualified_id,
            profile_hash=NO_CALIBRATION.profile_hash,
            config_hash=sha256_json({k: v for k, v in ctx.config.items()
                                     if k != "n_calibration_items"}),
            seed=ctx.seed, result_spec_hash="spec")
        got = compute_state_id("root-sha", "target-sha", (step,))
        expected = compute_state_id("root-sha", "target-sha", (OperatorStep(
            index=0, kind="ATTENTION",
            impl_id="attention.activation_importance_v1",
            impl_signature_hash="sig", profile_id=NO_CALIBRATION.qualified_id,
            profile_hash=NO_CALIBRATION.profile_hash,
            config_hash=sha256_json({}), seed=0, result_spec_hash="spec"),))
        assert got == expected

    @pytest.mark.parametrize("packing",
                             ["original_order_v1", "length_sorted_v1"])
    def test_the_packing_policy_never_reaches_the_hashed_config(
            self, model, items, packing):
        """A-bsz3 is A. The grouping must not fork a state id.

        This is the whole premise of treating the batching protocol as an
        EXECUTION optimization: A-bsz3 and canonical A must be the same
        scientific state, so that whether they agree is a question about
        arithmetic rather than about identity. If the packing reached
        `config`, the two would be different states by construction and the
        comparison would be meaningless.
        """
        ctx = context(model, items, 3,
                      ArchSpec.of("qwen3", TEACHER_GEOMETRY), packing=packing)
        assert ctx.execution.calibration_batch_packing == packing
        assert "calibration_batch_packing" not in ctx.config
        assert not any("pack" in k for k in ctx.config), (
            "the packing policy reached the mapping that is hashed into "
            "OperatorStep.config_hash")
        config_hash = sha256_json(
            {k: v for k, v in ctx.config.items() if k != "n_calibration_items"})
        assert config_hash == sha256_json({}), (
            "the hashed config moved with the packing policy")

    def test_execution_settings_are_absent_from_the_identity_tuple(self):
        """Whatever else `identity()` grows, it must not grow this."""
        step = OperatorStep(
            index=0, kind="ATTENTION", impl_id="x", impl_signature_hash="s",
            profile_id="p", profile_hash="ph", config_hash="ch", seed=0,
            result_spec_hash="rs")
        identity = step.identity()
        assert "micro_batch_size" not in identity
        assert "calibration_batch_packing" not in identity
        assert not any("batch" in k or "batch" in str(v)
                       for k, v in identity.items())
        assert not any("pack" in k or "pack" in str(v)
                       for k, v in identity.items())


# --- and it is still recorded ----------------------------------------------


class TestItIsRecordedAsEvidence:
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_every_batching_operator_traces_the_size_it_actually_used(
            self, model, items, batch_size):
        parent = ArchSpec.of("qwen3", TEACHER_GEOMETRY)
        cases = [
            (attention_activation.ATTENTION_ACTIVATION_IMPORTANCE_V1,
             parent.replace(num_attention_heads=2)),
            (FFN_ACTIVATION_IMPORTANCE_V0, parent.replace(intermediate_size=24)),
            (WIDTH_GLOBAL_PCA_V0, parent.replace(hidden_size=16)),
            (DEPTH_CAUSAL_KL_GREEDY_V1, parent.replace(num_hidden_layers=4)),
        ]
        for impl, target in cases:
            outcome = impl.apply(context(model, items, batch_size, target))
            assert outcome.trace.get("micro_batch_size") == batch_size, (
                f"{impl.impl_id} did not record the batch size it ran at")

    def test_the_trace_is_not_part_of_the_identity(self):
        """Why recording it is safe: `identity()` does not read `trace`."""
        common = dict(index=0, kind="ATTENTION", impl_id="x",
                      impl_signature_hash="s", profile_id="p", profile_hash="ph",
                      config_hash="ch", seed=0, result_spec_hash="rs")
        quiet = OperatorStep(**common)
        loud = OperatorStep(**common, trace={"micro_batch_size": 64})
        assert quiet.identity() == loud.identity()
        assert (compute_state_id("r", "t", (quiet,))
                == compute_state_id("r", "t", (loud,)))


# --- and it is a SEMANTIC id, which is not a materialization id ------------


class TestSemanticIdentityIsNotMaterializationIdentity:
    """The other half of the guarantee above, and it is a limitation.

    Everything in this file so far establishes that the batching knobs cannot
    move the state id. That is what makes A-bsz1 and A-bsz3 the same
    SCIENTIFIC state, and it is correct. It also means the state id cannot
    distinguish two executions that produced different BYTES -- so it is not
    sufficient on its own to own, resume or deduplicate a checkpoint.

    Stated here as a tested property rather than a paragraph, because the
    consequence is a live adoption precondition: see
    `logs/stages/stage-1/phase_c3/plans/a_bsz3_adoption.json ::
    identity_semantics`, and the resume refusal in
    `test_corrections.py :: test_resume_refuses_a_record_from_a_different_numerical_protocol`.
    """

    def test_the_identity_carries_nothing_that_could_name_an_artifact(self):
        """Not just "no batch size" -- no bytes at all.

        A state id derived without any reference to what was produced cannot
        be the authority for which artifact a resume may adopt.
        """
        step = OperatorStep(
            index=0, kind="ATTENTION", impl_id="x", impl_signature_hash="s",
            profile_id="p", profile_hash="ph", config_hash="ch", seed=0,
            result_spec_hash="rs")
        identity = step.identity()
        assert set(identity) == {"impl_id", "impl_signature_hash",
                                 "profile_hash", "config_hash", "seed"}, (
            "the identity tuple changed shape; the claim that it names no "
            "artifact has to be rechecked")
        assert not any("digest" in k or "artifact" in k for k in identity)

    def test_two_executions_that_differ_in_bytes_would_collide_on_one_id(self):
        """The hazard, in one assertion, with the fix it points at.

        Two steps identical in every identity-bearing field, whose executions
        differed and whose artifacts therefore might differ, are ONE state id.
        Asserted as a disjunction so that it stays true after the mechanism is
        built: if `compute_state_id` ever binds the execution fingerprint, the
        ids will differ and the adoption precondition can be lifted.
        """
        common = dict(index=0, kind="ATTENTION",
                      impl_id="attention.activation_importance_v1",
                      impl_signature_hash="sig", profile_id="p",
                      profile_hash="ph", config_hash="ch", seed=0,
                      result_spec_hash="rs")
        bsz1 = OperatorStep(
            **common, trace={"micro_batch_size": 1,
                             "calibration_batch_packing": "original_order_v1"})
        bsz3 = OperatorStep(
            **common, trace={"micro_batch_size": 3,
                             "calibration_batch_packing": "length_sorted_v1"})
        a = compute_state_id("root", "target", (bsz1,))
        b = compute_state_id("root", "target", (bsz3,))
        if a != b:
            pytest.fail(
                "compute_state_id now distinguishes two execution protocols. "
                "That would be a DIFFERENT resolution from the one the "
                "repository took -- see `specs.materialization`, which adds a "
                "second coordinate rather than forking the hypothesis -- so "
                "update `identity_semantics` in plans/a_bsz3_adoption.json and "
                "decide deliberately which mechanism owns this.")
        #: They collide, and that is correct: two batching protocols of one path
        #: are one hypothesis and must keep sharing a semantic id.
        assert a == b

    def test_the_collision_is_resolved_one_level_up(self):
        """And this is the mechanism that makes it harmless.

        `semantic_state_id` collides by design, so resume and dedup may not key
        on it alone. `materialization_id` is the semantic id PLUS the
        byte-affecting execution fingerprint, and it separates exactly the two
        protocols A3 measured. Asserted here, next to the collision, so a reader
        who finds the hazard finds its answer in the same file.
        """
        from aadistill.initialization.execution import ExecutionConfig
        from aadistill.initialization.specs.materialization import (
            MaterializationIdentity, NumericalEnvironment,
        )

        env = NumericalEnvironment(device_type="cuda", compute_dtype="bfloat16")
        semantic = compute_state_id("root", "target", ())
        #: ONE parent for both, so the only thing separating them is the
        #: execution protocol — which is the claim.
        parent = "shared-parent-materialization" + "0" * 4
        bsz1 = MaterializationIdentity.build(
            semantic_state_id=semantic, environment=env,
            parent_materialization_id=parent,
            execution=ExecutionConfig(
                micro_batch_size=1,
                calibration_batch_packing="original_order_v1"))
        bsz3 = MaterializationIdentity.build(
            semantic_state_id=semantic, environment=env,
            parent_materialization_id=parent,
            execution=ExecutionConfig(
                micro_batch_size=3,
                calibration_batch_packing="length_sorted_v1"))
        assert bsz1.semantic_state_id == bsz3.semantic_state_id
        assert bsz1.materialization_id != bsz3.materialization_id


# --- the config object itself ----------------------------------------------


class TestTheExecutionConfig:
    def test_the_default_is_stated_and_shared(self):
        assert DEFAULT_EXECUTION.micro_batch_size == DEFAULT_MICRO_BATCH_SIZE
        assert OperatorContext(
            adapter=None, model=None, parent_spec=None, target_spec=None,
            profile=None, calibration_items=[], seed=0
        ).execution == DEFAULT_EXECUTION

    def test_it_refuses_a_size_that_is_not_a_positive_int(self):
        for bad in (0, -1, "4", 2.0, True):
            with pytest.raises(ExecutionError):
                ExecutionConfig(micro_batch_size=bad)

    def test_it_is_frozen_so_one_operator_cannot_retune_the_next(self):
        cfg = ExecutionConfig(micro_batch_size=4)
        with pytest.raises(Exception):
            cfg.micro_batch_size = 8

    def test_as_trace_is_what_an_operator_records(self):
        assert ExecutionConfig(micro_batch_size=7).as_trace() == {
            "micro_batch_size": 7,
            "calibration_batch_packing": "original_order_v1"}
        assert ExecutionConfig(micro_batch_size=3,
                              calibration_batch_packing="length_sorted_v1"
                              ).as_trace() == {
            "micro_batch_size": 3,
            "calibration_batch_packing": "length_sorted_v1"}

    def test_the_packing_policy_is_execution_and_is_validated(self):
        """The second execution knob, and it refuses an unknown name.

        A grouping cannot change the estimand, the aggregation or the
        selection RULE -- so it is execution. It can change the selection
        RESULT, because a different grouping pads to different widths and this
        project has measured bf16 GEMMs reducing shape-dependently. Execution,
        therefore, but not numerically free: exactly the distinction that makes
        its equivalence an empirical question.
        """
        for bad in ("", "sorted", "length_sorted", None, 3):
            with pytest.raises(ExecutionError):
                ExecutionConfig(calibration_batch_packing=bad)
