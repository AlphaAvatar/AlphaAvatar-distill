"""A-bsz3 is A: same operator, same state, a different execution protocol.

The variant exists to ask ONE question — can B3's batching protocol accelerate
the incumbent ATTENTION initialization without changing what A means? For that
question to be answerable at all, three things must hold, and each is a test
here:

1. the batching knobs must be EXECUTION, so A-bsz1 and A-bsz3 are the same
   scientific state and any difference between them is arithmetic;
2. the operator must actually honour the packing policy, and must keep
   `bsz1 + original_order` as the byte-exact reference path;
3. the comparison driver must RUN — shipping an unexecuted driver and calling
   it tooling is how a rehearsal proves nothing.

What these tests deliberately do NOT claim: that A-bsz3 reproduces A on the
real teacher. They run tiny float32 models on CPU, where there is no
shape-dependent bf16 GEMM to find. `attn_out` is in the group measured to
reduce shape-dependently on an L40S, and this operator's score is built from
it, so the numerical question is GPU-only by construction.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "scripts", REPO / "tests" / "autoinit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.calibration.packing import (  # noqa: E402
    LENGTH_SORTED_V1, ORIGINAL_ORDER_V1, pack,
)
from aadistill.initialization.calibration.profiles import (  # noqa: E402
    register_profile, unregister_profile,
)
from aadistill.initialization.execution import ExecutionConfig  # noqa: E402
from aadistill.initialization.operators.attention.gqa import (  # noqa: E402
    activation_importance as attention_activation,
)
from aadistill.initialization.planning.fixed_path import (  # noqa: E402
    FixedPathSpec, FixedPathStep, VerifiedSuffix, materialize_fixed_path,
)

from conftest import TEACHER_GEOMETRY, build_tiny_model  # noqa: E402
from test_calibration_item_preparation import (  # noqa: E402
    DEPTH, FFN, TARGET, WIDTH, write_tiny_mixture,
)

from experiments.phase_c3.a_bsz3 import (  # noqa: E402
    A_BSZ1, A_BSZ3, ATTENTION_IMPL_ID, PROTOCOLS, execution_comparison,
)

START = 3


@pytest.fixture(scope="module", autouse=True)
def _registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


# --- 1. A-bsz3 is the same scientific state as A ---------------------------


def test_the_variant_is_the_incumbent_operator_and_imports_no_causal_kl():
    """A-bsz3 must not quietly become a second operator."""
    assert ATTENTION_IMPL_ID == "attention.activation_importance_v1"
    src = (REPO / "scripts/experiments/phase_c3/a_bsz3.py").read_text()
    assert "causal_kl" not in src.replace("`causal_kl`", ""), (
        "A-bsz3 references causal-KL in executable text; it is the incumbent "
        "operator under a different execution protocol and nothing else")
    assert "attention.activation_importance_bsz3" not in src, (
        "an execution knob was encoded as an implementation id, which would "
        "answer the equivalence question by definition instead of measuring it")


def test_both_protocols_carry_the_same_operator_and_differ_only_in_execution():
    assert A_BSZ1.micro_batch_size == 1
    assert A_BSZ1.calibration_batch_packing == ORIGINAL_ORDER_V1
    assert A_BSZ3.micro_batch_size == 3
    assert A_BSZ3.calibration_batch_packing == LENGTH_SORTED_V1
    assert set(PROTOCOLS) == {"A_bsz1", "A_bsz3"}


# --- 2. the operator honours the policy, and the reference path is exact ---


def _run(execution, items, seed=0):
    from aadistill.initialization.operators.base import (
        OperatorContext, get_implementation,
    )
    from aadistill.initialization.specs.arch import ArchSpec

    torch.manual_seed(seed)
    model = build_tiny_model(TEACHER_GEOMETRY)
    model.config.use_cache = False
    parent = ArchSpec.of("qwen3", TEACHER_GEOMETRY)
    target = parent.replace(
        num_attention_heads=TEACHER_GEOMETRY["num_attention_heads"] // 2)
    ctx = OperatorContext(
        adapter=QWEN3_ADAPTER, model=model, parent_spec=parent,
        target_spec=target, profile=_no_calibration(), calibration_items=items,
        seed=0, device="cpu", config={"n_calibration_items": len(items)},
        execution=execution)
    return get_implementation(ATTENTION_IMPL_ID).apply(ctx)


def _no_calibration():
    from aadistill.initialization.calibration.profiles import NO_CALIBRATION
    return NO_CALIBRATION


def _items(n=7):
    torch.manual_seed(11)
    return [{"item_id": f"i{k}",
             "input_ids": torch.randint(1, 128, (1, length)),
             "domain": "general", "subtype": "text"}
            for k, length in enumerate((23, 9, 31, 13, 17, 11, 27)[:n])]


def test_the_reference_path_is_bsz1_original_order_and_is_flagged_as_such():
    """The condition that reproduces the frozen C1 selection by construction."""
    items = _items()
    assert _run(A_BSZ1, items).trace["reference_path"] is True
    for execution in (ExecutionConfig(1, LENGTH_SORTED_V1),
                      ExecutionConfig(3, ORIGINAL_ORDER_V1), A_BSZ3):
        assert _run(execution, items).trace["reference_path"] is False


def test_the_packing_policy_is_traced_and_never_silently_ignored():
    """A setting the operator does not honour is worse than one it refuses."""
    trace = _run(A_BSZ3, _items()).trace
    assert trace["micro_batch_size"] == 3
    assert trace["calibration_batch_packing"] == LENGTH_SORTED_V1


def test_every_protocol_sees_the_same_valid_token_count():
    """Padded positions are masked OUT of the accumulator.

    If this ever fails it is a masking defect, not a numerical result: the two
    protocols would be averaging over different denominators and the
    comparison would be void rather than informative.
    """
    items = _items()
    counts = {name: _run(cfg, items).trace["calibration_tokens"]
              for name, cfg in PROTOCOLS.items()}
    assert len(set(counts.values())) == 1, counts
    assert next(iter(counts.values())) == sum(
        int(i["input_ids"].shape[-1]) for i in items)


def test_the_operator_reports_kept_heads_and_selection_margins():
    """Scores alone cannot say whether a flip was close; margins can."""
    trace = _run(A_BSZ1, _items()).trace
    kept = trace["kept_q_heads_per_layer"]
    margins = trace["selection_margin_per_layer"]
    assert kept and margins and len(kept) == len(margins)
    n_kv = TEACHER_GEOMETRY["num_key_value_heads"]
    assert all(len(m) == n_kv for m in margins), (
        "a margin is per GQA group, because the selection is per GQA group")
    assert all(m >= 0 for row in margins for m in row), (
        "a margin is kept-minus-dropped and cannot be negative")


def test_on_cpu_float32_every_grouping_agrees_which_is_the_algorithmic_claim():
    """The ALGORITHM is grouping-invariant. That is all this shows.

    It is worth asserting because it separates the two failure modes: if the
    real GPU comparison finds a difference, this test says the difference is
    arithmetic rather than a bug in how the grouping aggregates.
    """
    items = _items()
    digests = set()
    for execution in (A_BSZ1, ExecutionConfig(1, LENGTH_SORTED_V1),
                      ExecutionConfig(3, ORIGINAL_ORDER_V1), A_BSZ3):
        outcome = _run(execution, items)
        sd = outcome.model.state_dict()
        digests.add(tuple(
            (k, tuple(sd[k].flatten().tolist()[:4])) for k in sorted(sd)))
    assert len(digests) == 1, (
        "the groupings disagree on CPU float32, where there is no "
        "shape-dependent GEMM -- so the aggregation itself is grouping "
        "dependent and that is a defect, not a numerical finding")


# --- 3. the $0 comparison, on the REAL frozen mixture ----------------------


def test_the_zero_cost_comparison_reads_the_real_mixture():
    d = execution_comparison()
    assert d["n_items"] == 67
    assert d["valid_tokens"] == 59830
    a = d["protocols"]["A_bsz1"]["padding"]
    b = d["protocols"]["A_bsz3"]["padding"]
    assert a["padded_positions"] == 0, (
        "bsz1 pays no padding by definition; any other value means `pack` "
        "stopped producing one-item groups")
    assert b["n_groups"] == 23 == -(-67 // 3)
    assert 0 < b["padding_over_valid"] < 0.05


def test_length_sorting_is_what_makes_bsz3_cheap():
    """The protocol's whole claim, checked against the alternative.

    Consecutive original-order grouping at the same batch size pads far worse.
    Asserting the ORDERING rather than a literal keeps this true if the frozen
    mixture is ever re-cut.
    """
    from aadistill.initialization.calibration.packing import padding_profile
    from experiments.phase_c3.a_bsz3 import item_token_counts

    lengths = item_token_counts()
    sorted_cost = padding_profile(lengths, 3, packing=LENGTH_SORTED_V1)
    consecutive = padding_profile(lengths, 3, packing=ORIGINAL_ORDER_V1)
    assert sorted_cost["padded_positions"] < consecutive["padded_positions"]
    assert sorted_cost["n_groups"] == consecutive["n_groups"], (
        "the two policies must differ in padding, not in forward count")


# --- 4. the structural driver RUNS, at toy scale ---------------------------


def test_the_structural_comparison_driver_runs_end_to_end(tmp_path):
    """The real `structural_half`, on the real suffix API, at toy scale.

    Not a stub and not a transcript: this builds a pinned prefix, verifies it,
    and drives both protocols through `materialize_fixed_path_suffix` exactly
    as a session would. What it proves is that the driver executes and its
    plumbing is right -- the prefix is not re-run, path identity is preserved,
    and both protocols produce a comparable record.

    It does NOT prove numerical equivalence on the real teacher, and a passing
    `artifact_digest_identical` here is a statement about CPU float32.
    """
    sys.path.insert(0, str(REPO / "scripts" / "autoinit"))
    from compare_a_bsz3 import structural_half

    profile, _ = write_tiny_mixture(tmp_path)
    register_profile(profile, replace=True)
    try:
        pid = profile.qualified_id
        prefix = [FixedPathStep(DEPTH, pid), FixedPathStep(FFN, pid),
                  FixedPathStep(WIDTH, pid)]
        common = dict(
            family="qwen3", target_spec=_target(),
            root_repo_id="test/teacher", root_revision="deadbeef",
            device="cpu")
        probe = FixedPathSpec(path_id="test.a.probe",
                              steps=(*prefix,
                                     FixedPathStep(ATTENTION_IMPL_ID, pid)),
                              **common)
        observed = materialize_fixed_path(
            probe, adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
            workdir=tmp_path / "probe", repo_root=tmp_path)
        parent_digest = observed[2].identity.artifact_digest

        pinned = [*prefix[:2],
                  FixedPathStep(WIDTH, pid,
                                expected_artifact_digest=parent_digest,
                                label="pre-ATTENTION parent")]
        spec = FixedPathSpec(
            path_id="test.a", steps=(*pinned,
                                     FixedPathStep(ATTENTION_IMPL_ID, pid,
                                                   label="A")), **common)
        inc = materialize_fixed_path(
            spec, adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
            workdir=tmp_path / "pinned", repo_root=tmp_path)

        verified = VerifiedSuffix(
            start_index=START, parent=inc[2],
            expected_parent_artifact_digest=parent_digest,
            expected_path_hash=spec.spec_hash,
            prefix_reference_steps=tuple(spec.steps[:START]),
            expected_suffix_steps=((ATTENTION_IMPL_ID, pid),))

        out = structural_half(
            spec, adapter=QWEN3_ADAPTER,
            root_loader=lambda: QWEN3_ADAPTER.load(
                inc[2].checkpoint_path, device="cpu"),
            verified=verified, workdir=tmp_path / "cmp", repo_root=tmp_path,
            device="cpu")
    finally:
        unregister_profile(profile.qualified_id)

    assert set(out) == {"A_bsz1", "A_bsz3", "_comparison"}
    c = out["_comparison"]
    assert "INVALID" not in c, c.get("INVALID")
    assert c["calibration_tokens_identical"] is True
    assert c["classification"] in {"PURE_EXECUTION_OPTIMIZATION",
                                   "DISTINCT_NUMERICAL_PROTOCOL"}
    #: The protocols were really applied, not defaulted.
    assert out["A_bsz1"]["traced_execution"]["micro_batch_size"] == 1
    assert out["A_bsz3"]["traced_execution"]["micro_batch_size"] == 3
    assert out["A_bsz3"]["traced_execution"]["calibration_batch_packing"] == \
        LENGTH_SORTED_V1
    assert out["A_bsz1"]["traced_execution"]["reference_path"] is True
    assert out["A_bsz3"]["traced_execution"]["reference_path"] is False
    #: And the comparison actually computed a selection diff.
    assert "kept_head_selection" in c


def _target():
    """The tiny target the shared fixture uses, so DEPTH/FFN/WIDTH apply."""
    from aadistill.initialization.specs.arch import ArchSpec
    return ArchSpec.of("qwen3", TARGET)

