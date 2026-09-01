"""The fixed-path executor: exact order, and a digest gate that fails closed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.autoinit.arch import ArchSpec  # noqa: E402
from aadistill.autoinit.calibration import (  # noqa: E402
    register_profile,
    unregister_profile,
)
from aadistill.autoinit.fixed_path import (  # noqa: E402
    FixedPathDigestMismatch,
    FixedPathError,
    FixedPathSpec,
    FixedPathStep,
    materialize_fixed_path,
    write_replay_record,
)

from conftest import TEACHER_GEOMETRY, build_tiny_model, make_items, make_profile  # noqa: E402

TARGET = dict(TEACHER_GEOMETRY, intermediate_size=24, num_attention_heads=2)
PROFILE_ID = "test.balanced@v1"


@pytest.fixture(autouse=True)
def c1_operator_registered():
    """Explicit registration, and it must not leak into other tests' searches."""
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


@pytest.fixture
def profile_registered():
    p = make_profile("balanced")
    register_profile(p, replace=True)
    yield p
    unregister_profile(p.profile_id)


@pytest.fixture
def calib():
    return {PROFILE_ID: make_items()}


def spec(steps, **kw):
    return FixedPathSpec(
        path_id=kw.pop("path_id", "test.path"), family="qwen3",
        target_spec=ArchSpec.of("qwen3", TARGET), steps=tuple(steps),
        root_repo_id="test/teacher", root_revision="deadbeef", **kw)


def run(s, tmp_path, calib):
    return materialize_fixed_path(
        s, adapter=QWEN3_ADAPTER, root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=tmp_path, calibration_items=calib)


FFN = FixedPathStep("ffn.activation_importance_v0", PROFILE_ID)
ATTN = FixedPathStep("attention.weight_proxy_v0", "calib.none@v1")


# --- order is exact ---------------------------------------------------------

def test_steps_run_in_the_declared_order(tmp_path, profile_registered, calib):
    out = run(spec([FFN, ATTN]), tmp_path, calib)
    assert [r.kind for r in out] == ["FFN", "ATTENTION"]
    assert [r.index for r in out] == [0, 1]
    assert out[0].identity.num_parameters != out[1].identity.num_parameters


def test_the_reverse_order_is_a_different_path_and_a_different_artifact(
        tmp_path, profile_registered, calib):
    """Order is not cosmetic: FFN selects neurons from the *current* activations,
    so running it after ATTENTION selects different ones. An executor that could
    silently permute would silently change the experiment."""
    a = run(spec([FFN, ATTN], path_id="a"), tmp_path / "a", calib)
    b = run(spec([ATTN, FFN], path_id="b"), tmp_path / "b", calib)
    assert [r.kind for r in a] == ["FFN", "ATTENTION"]
    assert [r.kind for r in b] == ["ATTENTION", "FFN"]
    # Same destination geometry, different weights.
    assert a[-1].result_spec_hash == b[-1].result_spec_hash
    assert a[-1].identity.artifact_digest != b[-1].identity.artifact_digest


def test_the_path_label_and_hash_change_with_order():
    a, b = spec([FFN, ATTN]), spec([ATTN, FFN])
    assert a.path_label != b.path_label
    assert a.spec_hash != b.spec_hash
    assert a.kinds == ("FFN", "ATTENTION") and b.kinds == ("ATTENTION", "FFN")


def test_a_kind_cannot_be_applied_twice():
    with pytest.raises(FixedPathError, match="repeats kind"):
        spec([FFN, FFN])


def test_an_empty_path_is_refused():
    with pytest.raises(FixedPathError, match="at least one step"):
        spec([])


def test_an_inapplicable_step_is_refused_before_it_runs(tmp_path, profile_registered,
                                                        calib):
    """Target already at the parent's value -> the operator has nothing to do."""
    s = FixedPathSpec(
        path_id="noop", family="qwen3",
        target_spec=ArchSpec.of("qwen3", TEACHER_GEOMETRY),   # == the root
        steps=(FFN,), root_repo_id="r", root_revision="v")
    with pytest.raises(FixedPathError, match="not applicable"):
        run(s, tmp_path, calib)


# --- the digest gate --------------------------------------------------------

def test_a_matching_digest_passes_and_is_recorded(tmp_path, profile_registered, calib):
    first = run(spec([FFN, ATTN]), tmp_path / "1", calib)
    pinned = first[0].identity.artifact_digest
    out = run(spec([FixedPathStep("ffn.activation_importance_v0", PROFILE_ID,
                                  expected_artifact_digest=pinned, label="ffn"), ATTN]),
              tmp_path / "2", calib)
    assert out[0].digest_matches is True
    assert out[1].digest_matches is None          # unpinned steps stay unjudged


def test_a_mismatched_digest_stops_the_path_and_carries_the_evidence(
        tmp_path, profile_registered, calib):
    bad = "0" * 64
    with pytest.raises(FixedPathDigestMismatch) as e:
        run(spec([FixedPathStep("ffn.activation_importance_v0", PROFILE_ID,
                                expected_artifact_digest=bad, label="pinned-ffn"),
                  ATTN]), tmp_path, calib)
    err = e.value
    assert err.step_index == 0 and err.expected == bad
    assert err.actual != bad and len(err.actual) == 64
    assert "STOP" in str(err)
    # The evidence a reviewer needs is attached, not merely logged somewhere.
    assert err.evidence["steps"][0]["selection"]["kept_neurons"]
    assert err.evidence["path"]["path_label"]


def test_the_mismatch_stops_before_the_next_step_runs(tmp_path, profile_registered,
                                                      calib):
    with pytest.raises(FixedPathDigestMismatch) as e:
        run(spec([FixedPathStep("ffn.activation_importance_v0", PROFILE_ID,
                                expected_artifact_digest="0" * 64),
                  ATTN]), tmp_path, calib)
    assert len(e.value.evidence["steps"]) == 1, "ATTENTION ran after the gate failed"


# --- the C1 arm constructor -------------------------------------------------

def test_replace_tail_builds_two_arms_that_share_a_prefix(tmp_path, profile_registered,
                                                          calib):
    incumbent = spec([FFN, ATTN], path_id="incumbent")
    treatment = incumbent.replace_tail(
        1, FixedPathStep("attention.activation_importance_v1", PROFILE_ID),
        path_id="treatment")
    assert incumbent.steps[0] == treatment.steps[0]
    assert incumbent.steps[1] != treatment.steps[1]
    assert incumbent.spec_hash != treatment.spec_hash

    a = run(incumbent, tmp_path / "i", calib)
    b = run(treatment, tmp_path / "t", calib)
    # The shared prefix really is shared: identical artifact, bit for bit.
    assert a[0].identity.artifact_digest == b[0].identity.artifact_digest
    # And the arms differ only at the varied step.
    assert a[1].kind == b[1].kind == "ATTENTION"
    assert a[1].impl_id != b[1].impl_id


def test_replace_tail_rejects_an_out_of_range_index():
    with pytest.raises(FixedPathError, match="out of range"):
        spec([FFN, ATTN]).replace_tail(5, ATTN, path_id="x")


# --- it is not a search -----------------------------------------------------

def test_the_module_does_not_depend_on_the_beam():
    """Checked on the import graph, not on prose — the docstring names
    `BeamSearch` precisely to explain why it is not used."""
    import ast

    path = (Path(__file__).resolve().parents[2]
            / "src/aadistill/autoinit/fixed_path.py")
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    for forbidden in ("search", "ranking", "phase_a", "phase_b", "state"):
        assert not any(m == forbidden or m.endswith(f".{forbidden}")
                       or f".{forbidden}." in m for m in imported), \
            f"fixed_path imports {forbidden}: {sorted(imported)}"

    import aadistill.autoinit.fixed_path as fp
    assert not hasattr(fp, "BeamSearch") and not hasattr(fp, "SearchConfig")


def test_a_null_calibration_step_needs_no_items(tmp_path, profile_registered):
    """`calib.none@v1` resolves to the sentinel, exactly as the search does."""
    out = materialize_fixed_path(
        FixedPathSpec(path_id="attn-only", family="qwen3",
                      target_spec=ArchSpec.of("qwen3",
                                              dict(TEACHER_GEOMETRY,
                                                   num_attention_heads=2)),
                      steps=(ATTN,), root_repo_id="r", root_revision="v"),
        adapter=QWEN3_ADAPTER,
        root_loader=lambda: build_tiny_model(TEACHER_GEOMETRY),
        workdir=tmp_path, calibration_items={})
    assert out[0].profile_id == "calib.none@v1"


def test_the_replay_record_is_written_even_though_nothing_was_pinned(
        tmp_path, profile_registered, calib):
    s = spec([FFN, ATTN])
    out = run(s, tmp_path, calib)
    p = write_replay_record(s, out, tmp_path / "replay.json",
                            runtime={"torch": "test"}, root_binding={"repo": "test"})
    import json
    rec = json.loads(p.read_text())
    assert rec["n_pinned"] == 0 and rec["all_pinned_digests_matched"] is True
    assert [s["kind"] for s in rec["steps"]] == ["FFN", "ATTENTION"]
    assert rec["path_hash"] == s.spec_hash
