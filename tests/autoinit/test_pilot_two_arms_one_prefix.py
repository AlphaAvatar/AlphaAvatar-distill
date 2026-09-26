"""The pilot replays the frozen prefix ONCE and runs both arms off it.

Attempt 18's evidence puts the frozen parent replay at ~24 minutes. Doing it
per arm would buy nothing — a second replay can only reproduce the parent, in
which case it was redundant, or fail to, in which case the pilot is over — and
it would let the two arms be compared against two different checkpoints that
merely happen to share a digest.

Everything here EXECUTES, at toy scale on CPU, through the same three
functions the paid driver calls: `replay_prefix`, `verified_parent`,
`run_arm`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.initialization.operators.attention.gqa import causal_kl  # noqa: E402
from aadistill.initialization.operators.register import (  # noqa: E402
    register_builtin_operators)
from aadistill.initialization.planning.fixed_path import (  # noqa: E402
    FixedPathSpec, FixedPathSuffixRefused)
from aadistill.initialization.specs.arch import ArchSpec  # noqa: E402

from conftest import build_tiny_model  # noqa: E402
from experiments.phase_c3 import pilot  # noqa: E402

PARENT = dict(hidden_size=32, num_hidden_layers=4, intermediate_size=64,
              num_attention_heads=8, num_key_value_heads=2, head_dim=8,
              vocab_size=128, tie_word_embeddings=True)
TARGET = dict(PARENT, num_hidden_layers=3, intermediate_size=48,
              hidden_size=16, num_attention_heads=4)


@pytest.fixture(autouse=True)
def registered():
    register_builtin_operators()
    causal_kl.register(replace=True)
    yield
    causal_kl.unregister()


def toy_items(vocab=128, seed=5):
    g = torch.Generator().manual_seed(seed)
    out = []
    for subtype, lens in (("general", (11, 15)), ("alpha", (9, 13)),
                          ("beta", (14, 10))):
        for k, length in enumerate(lens):
            out.append({
                "item_id": f"{subtype}/{k}",
                "domain": "general" if subtype == "general" else "task",
                "subtype": subtype,
                "input_ids": torch.randint(0, vocab, (1, length), generator=g),
            })
    return out


ITEMS = {"calib.domain_balanced@v1": None, "calib.reasoning_heavy@v2": None}


def calib():
    items = toy_items()
    return {k: items for k in ITEMS}


def toy_prefix(digest: str | None = None) -> FixedPathSpec:
    """`pilot.prefix_spec()`'s shape at a CPU geometry.

    Built from `pilot.prefix_steps()`, the real constructor, so the profiles
    and the order are the frozen ones; only the target and the root are toy,
    and the pin is the digest this toy parent actually produces.
    """
    steps = list(pilot.prefix_steps(pin_parent=False))
    if digest is not None:
        last = steps[-1]
        steps[-1] = type(last)(impl_id=last.impl_id, profile_id=last.profile_id,
                               expected_artifact_digest=digest,
                               label="pre-ATTENTION parent")
    return FixedPathSpec(
        path_id="toy.shared-prefix", family="qwen3",
        target_spec=ArchSpec.of("qwen3", TARGET), steps=tuple(steps),
        root_repo_id="toy/root", root_revision="toy", device="cpu", seed=0)


def toy_arm(batch_size: int, digest: str | None = None) -> FixedPathSpec:
    steps = tuple(toy_prefix(digest).steps) + (pilot.causal_step(batch_size),)
    return FixedPathSpec(
        path_id=f"toy.{pilot.arm_id(batch_size)}", family="qwen3",
        target_spec=ArchSpec.of("qwen3", TARGET), steps=steps,
        root_repo_id="toy/root", root_revision="toy", device="cpu", seed=0)


def run_prefix(tmp_path, digest=None):
    results = []
    pilot.replay_prefix(
        toy_prefix(digest), adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(PARENT),
        workdir=tmp_path / "work", repo_root=REPO,
        calibration_items=calib(), on_step=results.append)
    return results


def run_both(tmp_path):
    """One prefix, then both arms as suffixes from its parent."""
    prefix = run_prefix(tmp_path)
    digest = prefix[-1].identity.artifact_digest
    #: Re-run the prefix pinned to what it produced, so the parent is GATED
    #: exactly as the real one is against `eea90c91…`.
    prefix = run_prefix(tmp_path / "pinned", digest)
    out = {}
    for bs in (1, 4):
        arm = toy_arm(bs, digest)
        verified = pilot.verified_parent(prefix, arm,
                                         expected_digest=digest)
        results, evidence = pilot.run_arm(
            arm, adapter=QWEN3_ADAPTER,
            parent_loader=lambda p=prefix[-1].checkpoint_path: (
                QWEN3_ADAPTER.load_checkpoint(p)
                if hasattr(QWEN3_ADAPTER, "load_checkpoint")
                else _load(p)),
            workdir=tmp_path / "pinned" / "work", verified=verified,
            repo_root=REPO, calibration_items=calib())
        out[bs] = (results, evidence)
    return prefix, out


def _load(path):
    from transformers import Qwen3ForCausalLM

    return Qwen3ForCausalLM.from_pretrained(path).float().eval()


# --- one prefix, two arms -------------------------------------------------

def test_the_prefix_is_executed_once_and_both_arms_reuse_it(tmp_path):
    prefix, arms = run_both(tmp_path)
    assert [r.kind for r in prefix] == ["DEPTH", "FFN", "RESIDUAL_WIDTH"]
    for bs, (results, _) in arms.items():
        assert [r.index for r in results] == [pilot.CAUSAL_STEP_INDEX], (
            f"arm B{bs} executed more than its causal step")
        assert results[0].kind == "ATTENTION"


def test_both_arms_re_identify_the_same_parent(tmp_path):
    """From the FILES, not from the record: `materialize_fixed_path_suffix`
    re-identifies the checkpoint on disk before either arm's causal step."""
    prefix, arms = run_both(tmp_path)
    digest = prefix[-1].identity.artifact_digest
    seen = set()
    for _bs, (_results, evidence) in arms.items():
        assert evidence["parent_artifact_digest"] == digest
        assert evidence["expected_parent_artifact_digest"] == digest
        assert evidence["premise"] == "verified"
        seen.add(evidence["parent_checkpoint"])
    assert len(seen) == 1, f"the arms used different parents: {seen}"


def test_the_arms_keep_their_original_step_index_and_path_hash(tmp_path):
    """A one-step spec for the tail would have a different path hash, and the
    arm's identity is the full frozen path -- so the suffix run must be bound
    to the FOUR-step spec and keep the original step numbering."""
    prefix, arms = run_both(tmp_path)
    digest = prefix[-1].identity.artifact_digest
    for bs, (results, evidence) in arms.items():
        assert evidence["expected_path_hash"] == toy_arm(bs, digest).spec_hash
        assert evidence["actual_path_hash"] == evidence["expected_path_hash"]
        assert results[0].index == pilot.CAUSAL_STEP_INDEX == 3
        assert evidence["executed_step_indices"] == [3]
    #: And the two arms are DIFFERENT paths, so a shared hash would mean the
    #: batch size had fallen out of the identity.
    assert (arms[1][1]["expected_path_hash"]
            != arms[4][1]["expected_path_hash"])


def test_the_two_arms_differ_only_in_the_batch_size(tmp_path):
    _prefix, arms = run_both(tmp_path)
    b1, b4 = arms[1][0][0], arms[4][0][0]
    assert b1.impl_id == b4.impl_id and b1.profile_id == b4.profile_id
    assert b1.trace["calibration_forward_batch_size"] == 1
    assert b4.trace["calibration_forward_batch_size"] == 4


def test_each_arm_carries_its_own_causal_evidence(tmp_path):
    """§2: the evidence must survive the executor, not only the operator."""
    _prefix, arms = run_both(tmp_path)
    for bs, (results, _) in arms.items():
        selection = results[0].selection
        assert "causal_head_evidence" in selection, (
            f"arm B{bs} lost its scientific evidence at the executor boundary")
        ev = selection["causal_head_evidence"]
        assert ev["calibration_forward_batch_size"] == bs
        assert len(ev["head_scores"]) == TARGET["num_hidden_layers"]
        assert "kept_heads" in selection


def test_the_evidence_reaches_the_serialized_record(tmp_path):
    """`StepResult.as_dict()` is what a replay record is written from."""
    import json

    _prefix, arms = run_both(tmp_path)
    doc = arms[4][0][0].as_dict()
    assert "causal_head_evidence" in doc["selection"]
    assert json.loads(json.dumps(doc))["selection"]["causal_head_evidence"][
        "calibration_forward_batch_size"] == 4


def test_a_historical_step_serializes_exactly_as_before(tmp_path):
    """The new key is absent from every other operator's outcome, so the
    whitelist growing must move no historical serialization."""
    prefix = run_prefix(tmp_path)
    for step in prefix:
        assert "causal_head_evidence" not in step.selection
        assert "causal_head_evidence" not in step.as_dict()["selection"]


def test_a_wrong_parent_digest_refuses_before_any_forward(tmp_path):
    """The gate is the guarantee that both arms share ONE parent."""
    prefix = run_prefix(tmp_path)
    digest = prefix[-1].identity.artifact_digest
    arm = toy_arm(4, digest)
    verified = pilot.verified_parent(prefix, arm, expected_digest=digest)
    wrong = type(verified)(
        start_index=verified.start_index, parent=verified.parent,
        expected_parent_artifact_digest="0" * 64,
        expected_path_hash=verified.expected_path_hash,
        prefix_reference_steps=verified.prefix_reference_steps,
        expected_suffix_steps=verified.expected_suffix_steps)
    with pytest.raises(FixedPathSuffixRefused):
        pilot.run_arm(arm, adapter=QWEN3_ADAPTER,
                      parent_loader=lambda: _load(prefix[-1].checkpoint_path),
                      workdir=tmp_path / "w2", verified=wrong,
                      repo_root=REPO, calibration_items=calib())


def test_verified_parent_refuses_a_parent_at_the_wrong_index(tmp_path):
    prefix = run_prefix(tmp_path)
    digest = prefix[-1].identity.artifact_digest
    arm = toy_arm(1, digest)
    with pytest.raises(ValueError, match="must continue from"):
        pilot.verified_parent(prefix[:1], arm, expected_digest=digest)


def test_verified_parent_refuses_an_empty_prefix():
    with pytest.raises(ValueError, match="no steps"):
        pilot.verified_parent([], toy_arm(1))


def test_run_arm_has_no_execution_parameter():
    """Same reason as `replay_prefix`, and a sharper one: this step's batch
    size comes from its hashed config, so an execution knob here would be
    inert at best and misleading at worst."""
    import inspect

    assert "execution" not in inspect.signature(pilot.run_arm).parameters


def test_the_causal_step_index_is_derived_from_the_prefix():
    assert pilot.CAUSAL_STEP_INDEX == len(pilot.PREFIX_STEPS) == 3
    assert len(pilot.arm_spec(1, repo_root=REPO).steps) == 4


def test_the_real_prefix_spec_is_pinned_and_three_steps():
    spec = pilot.prefix_spec(repo_root=REPO)
    assert len(spec.steps) == 3
    pinned = [s for s in spec.steps if s.expected_artifact_digest]
    assert [s.impl_id for s in pinned] == ["width.global_pca_v0"]
    assert pinned[0].expected_artifact_digest == pilot.FROZEN_PARENT_DIGEST
    #: Same root and target as the arms, so the parent it builds is theirs.
    arm = pilot.arm_spec(4, repo_root=REPO)
    assert spec.target_spec == arm.target_spec
    assert (spec.root_repo_id, spec.root_revision) == (arm.root_repo_id,
                                                       arm.root_revision)
    assert tuple(spec.steps) == tuple(arm.steps[:3])


def test_the_default_expected_digest_is_the_frozen_parent():
    """The toy passes its own; the PILOT must not be able to."""
    import inspect

    sig = inspect.signature(pilot.verified_parent)
    assert sig.parameters["expected_digest"].default == pilot.FROZEN_PARENT_DIGEST


def test_the_real_arms_accept_a_parent_produced_by_the_real_prefix_shape(tmp_path):
    """The premise checks on the REAL arm specs, against a REAL StepResult.

    `verify_suffix_premise` is everything that needs no model and no disk, so
    the pilot's own specs can be checked against each other at $0 rather than
    discovering a mismatched prefix on a paid pod. The StepResult is taken
    from a toy prefix execution and re-pointed at the real arms, because the
    only fields the premise reads are the index, the digests and the spec
    hash — not the geometry.
    """
    from dataclasses import replace as dc_replace

    from aadistill.initialization.planning.fixed_path import (
        verify_suffix_premise)

    prefix = run_prefix(tmp_path)
    digest = prefix[-1].identity.artifact_digest
    prefix = run_prefix(tmp_path / "pinned", digest)
    parent = dc_replace(prefix[-1], impl_id="width.global_pca_v0",
                        profile_id="calib.reasoning_heavy@v2")
    for bs in (1, 4):
        arm = pilot.arm_spec(bs, repo_root=REPO)
        ev = verify_suffix_premise(
            arm, pilot.verified_parent([parent], arm, expected_digest=digest))
        assert ev["premise"] == "verified"
        assert ev["expected_suffix_steps"] == [
            [pilot.CAUSAL_IMPL_ID, pilot.CAUSAL_PROFILE_ID]]
        assert ev["start_index"] == 3
