"""Executing one arm's tail from the other arm's verified parent.

The two C1 arms share a three-step prefix and differ only in ATTENTION. Stage D
replays the incumbent arm and gates its step-2 output against the frozen parent
digest; stage F then builds the treatment arm from that same parent. It did so by
handing `materialize_fixed_path` the FULL four-step treatment spec together with
the step-2 checkpoint — and `materialize_fixed_path` starts at step 0.

That is not a slow path. It raises:

    step 0 (depth.causal_kl_greedy_v1): not applicable to qwen3(...) —
    num_hidden_layers already at target (4)

reproduced here at `$0` in `test_mutation_the_old_full_path_call_still_fails`,
which is the mutation proof for everything else in this module: restore the old
call and the suite goes red for exactly that reason.

The fix is a suffix entry point rather than a one-step spec. A synthesized
one-step `FixedPathSpec` would hash differently, and an arm's identity IS its
full frozen path — a result bound to a one-step path is a result for a different
experiment. So the full spec is passed unchanged, its hash is checked against the
frozen one, and only the executed index range narrows. `StepResult.index` and the
checkpoint directory keep their original numbering.

Everything the caller asserts about the parent is refused before an operator
runs. Those refusals are the bulk of this file, because the entry point's whole
purpose is to SKIP verification work that was done elsewhere — so an unchecked
premise is not a missing optimization, it is the entire risk.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.operators import attention_activation  # noqa: E402
from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402
from aadistill.initialization.calibration.profiles import (# noqa: E402
    register_profile,
    unregister_profile,
)
from aadistill.initialization.planning.fixed_path import (# noqa: E402
    FixedPathError,
    FixedPathRootDeviceMismatch,
    FixedPathSpec,
    FixedPathStep,
    FixedPathSuffixRefused,
    VerifiedSuffix,
    materialize_fixed_path,
    materialize_fixed_path_suffix,
    verify_root_placement,
    write_suffix_execution_record,
)
from aadistill.initialization.operators.base import get_implementation  # noqa: E402

from conftest import TEACHER_GEOMETRY, build_tiny_model  # noqa: E402
from test_calibration_item_preparation import (  # noqa: E402
    ATTENTION, DEPTH, FFN, TARGET, WIDTH, write_tiny_mixture,
)

#: The incumbent's own ATTENTION consumes no calibration, exactly as C1's does.
INCUMBENT_ATTENTION = "attention.weight_proxy_v0"
NO_CALIB = "calib.none@v1"
START = 3                       # len(prefix); the only index the suffix runs


@pytest.fixture(scope="module", autouse=True)
def _treatment_registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


def _spec(steps, path_id, *, device="cpu") -> FixedPathSpec:
    return FixedPathSpec(
        path_id=path_id, family="qwen3",
        target_spec=ArchSpec.of("qwen3", TARGET), steps=tuple(steps),
        root_repo_id="test/teacher", root_revision="deadbeef", device=device)


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    """A tiny two-arm world built exactly the way `build_arm_specs` builds C1's.

    Three passes, and each is doing real work:

    1. an UNPINNED incumbent run, only to learn what the digests are — the same
       chicken-and-egg the real session solved by running Phase B first;
    2. a PINNED incumbent run, which is stage D: step 2 gated against the parent
       digest, step 3 against the incumbent digest. Its `steps[2]` is a genuinely
       gated `StepResult`, not one whose `digest_matches` a fixture set to True;
    3. the treatment arm, built by `replace_tail` from the pinned incumbent, so
       the shared prefix is shared BY CONSTRUCTION rather than by inspection.
    """
    root = tmp_path_factory.mktemp("suffix")
    profile, _ = write_tiny_mixture(root)
    register_profile(profile, replace=True)
    pid = profile.qualified_id

    prefix = [FixedPathStep(DEPTH, pid), FixedPathStep(FFN, pid),
              FixedPathStep(WIDTH, pid)]
    probe = _spec([*prefix, FixedPathStep(INCUMBENT_ATTENTION, NO_CALIB)],
                  "test.incumbent.probe")
    observed = materialize_fixed_path(
        probe, adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=root / "probe", repo_root=root)
    parent_digest = observed[2].identity.artifact_digest
    incumbent_digest = observed[3].identity.artifact_digest

    pinned_prefix = [*prefix[:2], FixedPathStep(
        WIDTH, pid, expected_artifact_digest=parent_digest,
        label="pre-ATTENTION parent")]
    incumbent = _spec(
        [*pinned_prefix,
         FixedPathStep(INCUMBENT_ATTENTION, NO_CALIB,
                       expected_artifact_digest=incumbent_digest,
                       label="incumbent")],
        "test.incumbent")
    inc_steps = materialize_fixed_path(
        incumbent, adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=root / "incumbent", repo_root=root)

    treatment = incumbent.replace_tail(
        START, FixedPathStep(ATTENTION, pid, label="treatment ATTENTION"),
        path_id="test.treatment")

    yield types.SimpleNamespace(
        root=root, profile=profile, pid=pid,
        incumbent=incumbent, treatment=treatment,
        inc_steps=inc_steps, parent=inc_steps[2],
        parent_digest=parent_digest, incumbent_digest=incumbent_digest)
    unregister_profile(profile.qualified_id)


def _verified(arms, **over) -> VerifiedSuffix:
    kw = dict(
        start_index=START,
        parent=arms.parent,
        expected_parent_artifact_digest=arms.parent_digest,
        expected_path_hash=arms.treatment.spec_hash,
        prefix_reference_steps=tuple(arms.incumbent.steps[:START]),
        expected_suffix_steps=((ATTENTION, arms.pid),),
    )
    kw.update(over)
    return VerifiedSuffix(**kw)


def _run_suffix(arms, workdir, *, spec=None, verified=None, **kw):
    return materialize_fixed_path_suffix(
        spec or arms.treatment, adapter=QWEN3_ADAPTER,
        root_loader=lambda: QWEN3_ADAPTER.load(arms.parent.checkpoint_path,
                                               device="cpu"),
        workdir=workdir, verified=verified or _verified(arms),
        repo_root=arms.root, **kw)


# --- the parent this all rests on -------------------------------------------

def test_the_parent_is_a_genuinely_gated_step(arms):
    """No fixture set these. Stage D's gate did."""
    assert arms.parent.index == 2
    assert arms.parent.digest_expected == arms.parent_digest
    assert arms.parent.digest_matches is True
    assert arms.parent.identity.artifact_digest == arms.parent_digest


def test_the_arms_share_their_prefix_by_construction(arms):
    assert arms.treatment.steps[:START] == arms.incumbent.steps[:START]
    assert arms.treatment.steps[START] != arms.incumbent.steps[START]
    assert arms.treatment.spec_hash != arms.incumbent.spec_hash
    assert len(arms.treatment.steps) == 4


# --- the production path -----------------------------------------------------

def test_only_the_original_step_three_executes(arms, tmp_path, monkeypatch):
    """The whole claim, measured from inside the operators themselves."""
    ran: list[str] = []
    for impl_id in (DEPTH, FFN, WIDTH, ATTENTION, INCUMBENT_ATTENTION):
        impl = get_implementation(impl_id)
        real = impl.execute
        monkeypatch.setattr(
            impl, "execute",
            lambda ctx, _i=impl_id, _r=real: (ran.append(_i), _r(ctx))[1])

    results, evidence = _run_suffix(arms, tmp_path / "trt")

    assert ran == [ATTENTION], f"executed {ran}"
    assert DEPTH not in ran and FFN not in ran and WIDTH not in ran
    assert [r.index for r in results] == [START]
    assert results[0].impl_id == ATTENTION
    assert evidence["executed_step_indices"] == [START]
    assert evidence["prefix_step_indices_not_executed"] == [0, 1, 2]


def test_the_result_keeps_its_original_index_and_checkpoint_name(arms, tmp_path):
    results, _ = _run_suffix(arms, tmp_path / "trt")
    assert results[0].index == START
    assert Path(results[0].checkpoint_path).name == "03_attention"


def test_the_output_stays_bound_to_the_full_treatment_path(arms, tmp_path):
    """A one-step spec would have been the easy fix and the wrong one."""
    results, evidence = _run_suffix(arms, tmp_path / "trt")
    assert evidence["expected_path_hash"] == arms.treatment.spec_hash
    assert evidence["actual_path_hash"] == arms.treatment.spec_hash
    assert results[0].result_spec_hash == ArchSpec.of("qwen3", TARGET).spec_hash
    assert results[0].digest_expected is None, (
        "the treatment output was never pinned; nothing may claim it matched")
    assert results[0].digest_matches is None


def test_the_suffix_output_differs_from_the_incumbent(arms, tmp_path):
    """Two arms, one prefix, different ATTENTION — so different outputs."""
    results, _ = _run_suffix(arms, tmp_path / "trt")
    assert results[0].identity.artifact_digest != arms.incumbent_digest


def test_the_record_says_it_is_not_a_replay(arms, tmp_path):
    results, evidence = _run_suffix(arms, tmp_path / "trt")
    import json

    path = write_suffix_execution_record(
        arms.treatment, results, tmp_path / "rec.json",
        runtime={"torch": torch.__version__}, suffix_evidence=evidence,
        calibration={"profile_id": arms.pid})
    rec = json.loads(path.read_text())
    assert rec["schema"] == "aadistill.autoinit.fixed_path_suffix_execution/v1"
    assert rec["is_replay"] is False
    assert rec["output_digest_was_pre_pinned"] is False
    assert rec["path_hash"] == arms.treatment.spec_hash
    assert rec["executed_step_indices"] == [START]
    assert rec["verified_prefix"]["expected_parent_artifact_digest"] == \
        arms.parent_digest
    assert rec["output"]["index"] == START
    assert rec["n_pinned"] == 0


def test_mutation_the_old_full_path_call_still_fails(arms, tmp_path):
    """Restore stage F's old call and the defect comes straight back.

    This is the mutation proof for the module: if the suffix entry point were
    unnecessary, this would pass.
    """
    with pytest.raises(FixedPathError) as exc:
        materialize_fixed_path(
            arms.treatment, adapter=QWEN3_ADAPTER,
            root_loader=lambda: QWEN3_ADAPTER.load(arms.parent.checkpoint_path,
                                                   device="cpu"),
            workdir=tmp_path / "old", repo_root=arms.root)
    assert "step 0" in str(exc.value)
    assert "num_hidden_layers already at target" in str(exc.value)


# --- fail closed, before any operator runs -----------------------------------

def _refused(arms, tmp_path, monkeypatch, **over):
    """Assert the refusal AND that nothing was executed."""
    ran: list[str] = []
    for impl_id in (DEPTH, FFN, WIDTH, ATTENTION):
        impl = get_implementation(impl_id)
        monkeypatch.setattr(
            impl, "execute",
            lambda ctx, _i=impl_id: pytest.fail(f"{_i} ran after a refusal"))
    spec = over.pop("spec", None)
    with pytest.raises(FixedPathSuffixRefused) as exc:
        _run_suffix(arms, tmp_path / "refused", spec=spec,
                    verified=_verified(arms, **over))
    assert ran == []
    assert not (tmp_path / "refused" / "steps").exists()
    return exc.value


def test_a_wrong_parent_digest_is_refused(arms, tmp_path, monkeypatch):
    exc = _refused(arms, tmp_path, monkeypatch,
                   expected_parent_artifact_digest="0" * 64)
    assert "realized artifact digest is not the frozen one" in exc.reason


def test_an_unpinned_parent_is_refused(arms, tmp_path, monkeypatch):
    """A parent that was never gated cannot be a verified prefix."""
    import copy

    loose = copy.copy(arms.parent)
    loose.digest_expected = None
    loose.digest_matches = None
    exc = _refused(arms, tmp_path, monkeypatch, parent=loose)
    assert "was not pinned to the frozen parent digest" in exc.reason


def test_a_parent_whose_gate_did_not_match_is_refused(arms, tmp_path, monkeypatch):
    import copy

    bad = copy.copy(arms.parent)
    bad.digest_matches = False
    exc = _refused(arms, tmp_path, monkeypatch, parent=bad)
    assert "did not record a digest MATCH" in exc.reason


def test_a_parent_at_the_wrong_step_index_is_refused(arms, tmp_path, monkeypatch):
    import copy

    wrong = copy.copy(arms.parent)
    wrong.index = 1
    exc = _refused(arms, tmp_path, monkeypatch, parent=wrong)
    assert "must continue from step" in exc.reason


def test_a_non_shared_prefix_is_refused(arms, tmp_path, monkeypatch):
    """The parent on disk belongs to the prefix that produced it, not to any
    path that merely has three steps."""
    other = (FixedPathStep(FFN, arms.pid), *arms.incumbent.steps[1:START])
    exc = _refused(arms, tmp_path, monkeypatch, prefix_reference_steps=other)
    assert "not the prefix that was actually executed" in exc.reason


def test_a_changed_treatment_path_hash_is_refused(arms, tmp_path, monkeypatch):
    exc = _refused(arms, tmp_path, monkeypatch, expected_path_hash="0" * 64)
    assert "not the frozen path" in exc.reason


def test_a_start_index_other_than_the_frozen_one_is_refused(arms, tmp_path,
                                                            monkeypatch):
    exc = _refused(arms, tmp_path, monkeypatch, start_index=2)
    assert "must continue from step" in exc.reason


def test_a_start_index_outside_the_path_is_refused(arms, tmp_path, monkeypatch):
    exc = _refused(arms, tmp_path, monkeypatch, start_index=4)
    assert "not a proper suffix" in exc.reason


def test_a_start_index_of_zero_is_refused(arms, tmp_path, monkeypatch):
    """Zero is the whole path; that is the other entry point's job."""
    exc = _refused(arms, tmp_path, monkeypatch, start_index=0)
    assert "not a proper suffix" in exc.reason


def test_a_different_suffix_operator_is_refused(arms, tmp_path, monkeypatch):
    exc = _refused(arms, tmp_path, monkeypatch,
                   expected_suffix_steps=((INCUMBENT_ATTENTION, NO_CALIB),))
    assert "not the frozen ones" in exc.reason


def test_a_wrong_parent_result_spec_hash_is_refused(arms, tmp_path, monkeypatch):
    """Checked against the model that actually loaded, not against the record."""
    import copy

    wrong = copy.copy(arms.parent)
    wrong.result_spec_hash = "0" * 64
    exc = _refused(arms, tmp_path, monkeypatch, parent=wrong)
    assert "architecture is not the one the verified step recorded" in exc.reason


def test_altered_parent_checkpoint_bytes_are_refused(arms, tmp_path, monkeypatch):
    """The record says what was written; this asks what is there NOW.

    A byte flipped deep in the tensor data leaves a loadable checkpoint with the
    same architecture — only the weights differ — so every structural check
    passes and only re-identification from disk catches it.
    """
    import copy
    import shutil

    victim = tmp_path / "tampered"
    shutil.copytree(arms.parent.checkpoint_path, victim)
    shard = next(p for p in sorted(victim.iterdir())
                 if p.suffix == ".safetensors")
    raw = bytearray(shard.read_bytes())
    at = len(raw) - 9                      # inside the data block, not the header
    raw[at] ^= 0xFF
    shard.write_bytes(bytes(raw))

    moved = copy.copy(arms.parent)
    moved.checkpoint_path = str(victim)
    exc = _refused(arms, tmp_path, monkeypatch, parent=moved)
    assert "no longer identifies to the frozen parent digest" in exc.reason


# --- exhaustive root placement, without a GPU --------------------------------

class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.register_buffer("scale", torch.ones(3))


def test_a_uniform_cpu_module_is_accepted():
    ev = verify_root_placement(_Tiny(), "cpu")
    assert ev["resolved"] == "cpu"
    assert ev["devices"] == ["cpu"]
    assert ev["n_parameters"] == 2 and ev["n_buffers"] == 1
    assert ev["n_tensors_checked"] == 3


def test_buffers_are_checked_not_just_parameters():
    """`model_device()` reads the FIRST parameter and stops. A buffer left on
    the host is invisible to it and fatal to an operator that reads it."""
    m = _Tiny()
    m.scale = m.scale.to("meta")            # the only other device a CPU box has
    with pytest.raises(FixedPathRootDeviceMismatch) as exc:
        verify_root_placement(m, "cpu")
    assert "buffer scale" in str(exc.value.evidence["meta_tensors"])
    # and the first parameter — all `model_device` ever looks at — was fine
    assert next(m.parameters()).device.type == "cpu"


def test_a_meta_parameter_is_refused():
    m = _Tiny()
    m.lin.weight = torch.nn.Parameter(torch.empty(3, 3, device="meta"))
    with pytest.raises(FixedPathRootDeviceMismatch) as exc:
        verify_root_placement(m, "cpu")
    assert exc.value.evidence["n_meta"] == 1
    assert "parameter lin.weight" in exc.value.evidence["meta_tensors"]


def test_a_module_with_no_tensors_is_refused():
    with pytest.raises(FixedPathRootDeviceMismatch, match="no tensors"):
        verify_root_placement(torch.nn.Module(), "cpu")


def test_an_unset_optional_buffer_is_skipped_not_refused():
    m = _Tiny()
    m.register_buffer("maybe", None)
    assert verify_root_placement(m, "cpu")["n_buffers"] == 1


#: Placement cases a CPU-only box cannot ALLOCATE. `verify_root_placement`'s
#: contract is over `.device`, so the stub supplies exactly that and nothing
#: else — and the real-tensor cases above cover the paths that can be allocated,
#: so this is not the only evidence for any behaviour.
class _Placed:
    def __init__(self, **where):
        self._p = {n: types.SimpleNamespace(device=torch.device(d))
                   for n, d in where.items() if not n.startswith("buf_")}
        self._b = {n[4:]: types.SimpleNamespace(device=torch.device(d))
                   for n, d in where.items() if n.startswith("buf_")}

    def named_parameters(self):
        return list(self._p.items())

    def named_buffers(self):
        return list(self._b.items())


def test_mixed_cpu_and_cuda_parameters_are_refused():
    m = _Placed(a="cuda:0", b="cpu")
    with pytest.raises(FixedPathRootDeviceMismatch) as exc:
        verify_root_placement(m, "cuda")
    assert exc.value.evidence["devices"] == ["cpu", "cuda:0"]
    assert "split across devices" in exc.value.evidence["reason"]


def test_a_buffer_on_the_wrong_device_is_refused():
    m = _Placed(a="cuda:0", buf_scale="cpu")
    with pytest.raises(FixedPathRootDeviceMismatch) as exc:
        verify_root_placement(m, "cuda")
    assert "buffer scale" in str(exc.value.evidence["examples"])


def test_mixed_cuda_ordinals_are_refused():
    m = _Placed(a="cuda:0", b="cuda:1")
    with pytest.raises(FixedPathRootDeviceMismatch) as exc:
        verify_root_placement(m, "cuda")
    assert exc.value.evidence["devices"] == ["cuda:0", "cuda:1"]


def test_an_unindexed_cuda_declaration_accepts_one_consistent_ordinal():
    assert verify_root_placement(_Placed(a="cuda:3", b="cuda:3"),
                                 "cuda")["resolved"] == "cuda:3"


def test_an_indexed_declaration_requires_that_exact_ordinal():
    assert verify_root_placement(_Placed(a="cuda:1"), "cuda:1")["resolved"] == "cuda:1"
    with pytest.raises(FixedPathRootDeviceMismatch):
        verify_root_placement(_Placed(a="cuda:0"), "cuda:1")


def test_the_verifier_does_not_move_the_model():
    """Relocating a misplaced model would destroy the evidence that it was."""
    m = _Tiny()
    with pytest.raises(FixedPathRootDeviceMismatch):
        verify_root_placement(m, "cuda")
    assert next(m.parameters()).device.type == "cpu"


def test_the_suffix_entry_point_checks_placement_too(arms, tmp_path, monkeypatch):
    """Both entry points, not just the whole-path one."""
    for impl_id in (DEPTH, FFN, WIDTH, ATTENTION):
        impl = get_implementation(impl_id)
        monkeypatch.setattr(
            impl, "execute",
            lambda ctx, _i=impl_id: pytest.fail(f"{_i} ran on a misplaced root"))
    cuda_spec = FixedPathSpec(
        path_id="test.treatment", family="qwen3",
        target_spec=arms.treatment.target_spec, steps=arms.treatment.steps,
        root_repo_id=arms.treatment.root_repo_id,
        root_revision=arms.treatment.root_revision, device="cuda")
    with pytest.raises(FixedPathRootDeviceMismatch):
        _run_suffix(arms, tmp_path / "dev", spec=cuda_spec,
                    verified=_verified(arms,
                                       expected_path_hash=cuda_spec.spec_hash))


# --- the deadline the fixed path never passed --------------------------------

class _FakeClockDeadline:
    """Expires after `allow` checks. No wall clock, so the test cannot flake."""

    def __init__(self, allow: int):
        self.allow, self.checks, self.fired_at = allow, 0, ""

    def check(self, where: str = "") -> None:
        self.checks += 1
        if self.checks > self.allow:
            self.fired_at = where
            raise RuntimeError(f"budget spent at {where}")


def test_the_fixed_path_hands_the_deadline_to_the_operator(arms, tmp_path,
                                                           monkeypatch):
    seen: list = []
    impl = get_implementation(ATTENTION)
    real = impl.execute
    monkeypatch.setattr(
        impl, "execute",
        lambda ctx, _r=real: (seen.append(ctx.deadline), _r(ctx))[1])

    d = _FakeClockDeadline(allow=10_000)
    _run_suffix(arms, tmp_path / "dl", deadline=d)
    assert seen and seen[0] is d


def test_an_expired_deadline_stops_further_candidate_work(arms, tmp_path):
    """DEPTH checks once per candidate, so an expired budget bounds the work.

    Measured, not asserted: the run that stops is compared against the number of
    candidates the unbounded run evaluates.
    """
    spec = _spec([FixedPathStep(DEPTH, arms.pid)], "test.depth_only")

    counter = _FakeClockDeadline(allow=10_000)
    materialize_fixed_path(
        spec, adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=tmp_path / "full", repo_root=arms.root, deadline=counter)
    full_checks = counter.checks
    assert full_checks > 2, "DEPTH must consult the deadline per candidate"

    stopped = _FakeClockDeadline(allow=2)
    with pytest.raises(RuntimeError, match="budget spent"):
        materialize_fixed_path(
            spec, adapter=QWEN3_ADAPTER,
            root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
            workdir=tmp_path / "stopped", repo_root=arms.root, deadline=stopped)
    assert stopped.checks == 3 < full_checks
    assert stopped.fired_at, "the deadline records where it fired"
    assert not (tmp_path / "stopped" / "steps").exists(), (
        "an expired deadline must stop before the step is written")


def test_no_deadline_is_still_allowed(arms, tmp_path, monkeypatch):
    """`None` must stay a no-op: the field was optional before this change."""
    seen: list = []
    impl = get_implementation(ATTENTION)
    real = impl.execute
    monkeypatch.setattr(
        impl, "execute",
        lambda ctx, _r=real: (seen.append(ctx.deadline), _r(ctx))[1])
    _run_suffix(arms, tmp_path / "nodl")
    assert seen == [None]
