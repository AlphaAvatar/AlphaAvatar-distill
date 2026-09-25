"""The C2 baseline: found when the search produced it, rebuilt once when not.

Zero cost, CPU only, no checkpoint and no model. That is possible because B's
identity is *derivable from its committed components*: the shard sha256 in
`c1_arm_identities.json` reproduces its `weights_digest`, and that plus the
config sha256 and the arch signature reproduces its `artifact_digest`. So a
faithful B identity can be constructed here, and the consumer can be run against
the real output shape rather than against a fake that agrees with it.

That distinction has cost this project money. A harness once certified a
defective line because its stub matched what the consumer expected instead of
what the producer emits; `materialize_fixed_path` is stubbed in the rebuild test
below, and what it returns is a `StepResult` carrying a `CheckpointIdentity`
whose digests are computed, not asserted.

Three things are under test and they are different:

1. **the frozen identities are what committed evidence says** — re-derived from
   the preregistration and the arm-identities record, not trusted here;
2. **both branches behave** — B present means no rebuild, B absent means exactly
   one, and a second invocation is refused;
3. **every gate fails closed** — a construction that is not the frozen one, a
   searched leaf with B's path but different bytes, and a rebuild that produces
   something else are all refusals with a decomposed explanation.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO / _extra) not in sys.path:
        sys.path.insert(0, str(REPO / _extra))

from aadistill.initialization.specs.artifact import (  # noqa: E402
    CheckpointIdentity, ShardRecord,
)
from experiments.phase_c2 import baseline as B  # noqa: E402


@pytest.fixture
def registered():
    """`attention.activation_importance_v1` is not a shipped default."""
    from aadistill.initialization.operators.attention.gqa import activation_importance as attention_activation
    from experiments.phase_c2.search_space import register_c2_operators

    register_c2_operators()
    try:
        yield
    finally:
        attention_activation.unregister()


# --- 1. the frozen identities, re-derived -----------------------------------

def test_the_construction_hash_is_the_one_the_preregistration_froze():
    doc = json.loads((REPO / B.PREREGISTRATION).read_text())
    assert doc["fixed_path"]["treatment_spec_hash"] == B.B_SPEC_HASH
    assert doc["replay_gates"]["expected_parent_digest"] == B.B_PARENT_DIGEST
    #: And the path label the search will produce, in the same order.
    assert doc["fixed_path"]["treatment_path_label"] == B.leaf_label()


def test_the_content_identities_are_what_attempt_18_recorded():
    doc = json.loads((REPO / B.ARM_IDENTITIES).read_text())
    t = doc["treatment"]
    assert t["artifact_digest"] == B.B_ARTIFACT_DIGEST
    assert t["weights_digest"] == B.B_WEIGHTS_DIGEST
    assert t["config_sha256"] == B.B_CONFIG_SHA256
    assert t["arch_signature"] == B.B_ARCH_SIGNATURE
    assert t["single_shard_sha256"] == B.B_SINGLE_SHARD_SHA256
    assert t["num_parameters"] == B.B_NUM_PARAMETERS
    assert (t["impl_id"], t["profile_id"]) == B.B_PATH[-1][1:]
    #: The record says the output digest was NOT pre-pinned, which is why this
    #: module calls it observed rather than preregistered. If that ever became
    #: True the provenance sentence in `baseline.py` would be wrong.
    assert doc["treatment_output_digest_was_pre_pinned"] is False
    assert doc["runtime"]["transformers"] == B.B_RUNTIME["transformers"]


def test_the_aggregate_digest_is_derivable_from_its_own_components():
    """The consistency check that makes the constants mutually checkable.

    `artifact_digest` is `sha256_json` over the weights digest, the index, the
    config, the arch signature and the tokenizer — and `weights_digest` is in
    turn an aggregate over the shard manifest. So B's recorded aggregate must be
    reproducible from B's recorded parts, and if it is not then at least one of
    the seven constants above is wrong.
    """
    identity = b_identity()
    assert identity.weights_digest == B.B_WEIGHTS_DIGEST
    assert identity.artifact_digest == B.B_ARTIFACT_DIGEST


def b_identity(**overrides: Any) -> CheckpointIdentity:
    """A faithful B identity, built from its committed components."""
    fields = dict(
        path="/workspace/aad/artifacts/autoinit/phase_c2_search/baseline_rebuild",
        shards=(ShardRecord("model.safetensors", B.B_SINGLE_SHARD_SHA256,
                            1192099840),),
        index_sha256=None, config_sha256=B.B_CONFIG_SHA256,
        arch_signature=B.B_ARCH_SIGNATURE, tokenizer_sha256=None,
        num_parameters=B.B_NUM_PARAMETERS)
    return CheckpointIdentity(**{**fields, **overrides})


def test_the_frozen_construction_comes_from_c1s_own_constructor(registered):
    """Imported, not re-declared: one construction of the thing whose sameness
    is the point."""
    spec = B.frozen_baseline_spec(device="cuda")
    assert spec.spec_hash == B.B_SPEC_HASH
    assert B.path_identity_of_spec(spec) == B.B_PATH
    assert B.assert_frozen_construction(spec)["verified"] is True
    #: The prefix carries the parent pin C1 replayed three times.
    assert spec.steps[-2].expected_artifact_digest == B.B_PARENT_DIGEST


def test_a_construction_that_is_not_the_frozen_one_is_refused(registered):
    """The $0 gate, and the whole reason it runs before any compute."""
    from aadistill.initialization.planning.fixed_path import FixedPathStep

    spec = B.frozen_baseline_spec(device="cuda")
    #: The same four kinds, ATTENTION on the OTHER mixture. A legitimate C2
    #: candidate, and emphatically not B.
    other = spec.replace_tail(
        len(spec.steps) - 1,
        FixedPathStep("attention.activation_importance_v1",
                      "calib.reasoning_heavy@v2"),
        path_id=spec.path_id)
    assert other.spec_hash != B.B_SPEC_HASH
    with pytest.raises(B.BaselineError, match="not the frozen treatment path"):
        B.assert_frozen_construction(other)


def test_the_device_is_part_of_the_construction_hash(registered):
    """Not a quirk to work around: a CPU-built path is a different path, and
    `"cuda"` is the value C1 froze."""
    assert B.frozen_baseline_spec(device="cpu").spec_hash != B.B_SPEC_HASH


# --- 2. detection ----------------------------------------------------------

@dataclass
class FakeStep:
    kind: str
    impl_id: str
    profile_id: str


@dataclass
class FakeState:
    steps: tuple
    artifact_digest: str
    state_id: str = "f" * 32
    path_label: str = ""
    _complete: bool = True

    def is_complete_leaf(self) -> bool:
        return self._complete


@dataclass
class FakeResult:
    leaves: list


def b_leaf(digest: str = B.B_ARTIFACT_DIGEST, **kw) -> FakeState:
    return FakeState(steps=tuple(FakeStep(*t) for t in B.B_PATH),
                     artifact_digest=digest, path_label=B.leaf_label(), **kw)


def other_leaf() -> FakeState:
    steps = list(B.B_PATH)
    steps[-1] = ("ATTENTION", "attention.activation_importance_v1",
                 "calib.reasoning_heavy@v2")
    return FakeState(steps=tuple(FakeStep(*t) for t in steps),
                     artifact_digest="a" * 64, state_id="a" * 32)


def test_a_searched_b_is_found_among_other_leaves():
    found = B.find_searched_baseline(FakeResult([other_leaf(), b_leaf()]))
    assert found is not None
    assert found.artifact_digest == B.B_ARTIFACT_DIGEST


def test_no_b_among_the_leaves_is_an_absence_not_an_error():
    assert B.find_searched_baseline(FakeResult([other_leaf()])) is None
    assert B.find_searched_baseline(FakeResult([])) is None


def test_a_leaf_with_bs_path_but_different_bytes_is_refused():
    """The finding that must not be silently used as a baseline.

    Same operators, same profiles, same seed, same teacher — different bytes.
    That is a statement about determinism or about an operator, and comparing
    candidates against it would compare them against a different B.
    """
    with pytest.raises(B.BaselineError, match="different bytes"):
        B.find_searched_baseline(FakeResult([b_leaf(digest="b" * 64)]))


def test_an_incomplete_state_with_bs_prefix_is_not_b():
    """Only a complete leaf can be B; a 3-operator intermediate cannot."""
    partial = FakeState(steps=tuple(FakeStep(*t) for t in B.B_PATH[:3]),
                        artifact_digest=B.B_ARTIFACT_DIGEST, _complete=False)
    assert B.find_searched_baseline(FakeResult([partial])) is None


# --- 3. the fallback, both branches ----------------------------------------

def make_fallback(tmp_path, monkeypatch, *, steps=None, calls=None):
    """A fallback whose executor is replaced by one returning the REAL shape."""
    from aadistill.initialization.planning import fixed_path as FP

    def fake_materialize(spec, **kwargs):
        if calls is not None:
            calls.append(spec.path_id)
        return steps if steps is not None else [real_shaped_step()]

    monkeypatch.setattr(B, "materialize_fixed_path", fake_materialize)
    #: `rebuild_minutes` and `afford` are required: the rebuild runs on its OWN
    #: allowance, checked against the soft stop immediately before
    #: materialization, and on a clock it builds itself. Accepting a
    #: pre-constructed `Deadline` would leak the beam's, which after a
    #: full-envelope search has nothing left. See
    #: `tests/pod/test_phase_c2_reserve_partition.py`.
    return B.BaselineFallback(adapter=object(), workdir=tmp_path,
                              rebuild_minutes=27.665, afford=lambda *_: None,
                              repo_root=REPO, device="cuda",
                              say=lambda *_: None)


def real_shaped_step(identity: CheckpointIdentity | None = None):
    """A `StepResult` in the shape the executor actually emits."""
    from aadistill.initialization.planning.fixed_path import StepResult

    art = identity if identity is not None else b_identity()
    return StepResult(
        index=3, impl_id="attention.activation_importance_v1",
        profile_id="calib.domain_balanced@v1", kind="ATTENTION",
        result_spec_hash=B.B_ARCH_SIGNATURE, identity=art,
        checkpoint_path=art.path, seconds=11.27,
        digest_expected=B.B_ARTIFACT_DIGEST, digest_matches=True)


def test_b_present_means_no_rebuild(registered, tmp_path, monkeypatch):
    calls: list = []
    fallback = make_fallback(tmp_path, monkeypatch, calls=calls)
    entries = fallback(FakeResult([other_leaf(), b_leaf()]), lambda: None)

    assert entries == [], "a rebuilt B was injected beside a searched one"
    assert calls == [], "the executor ran when nothing needed rebuilding"
    assert fallback.outcome["resolution"] == "searched"
    assert fallback.outcome["rebuilt"] is False


def test_b_absent_means_exactly_one_rebuild(registered, tmp_path, monkeypatch):
    calls: list = []
    fallback = make_fallback(tmp_path, monkeypatch, calls=calls)
    entries = fallback(FakeResult([other_leaf()]), lambda: None)

    assert len(entries) == 1
    assert calls == [B.REBUILT_PATH_ID], calls
    entry = entries[0]
    #: The `retained_candidates` entry shape, with the digest that makes
    #: `make_retained_state` refuse a checkpoint that is not B — a third,
    #: independent gate after the $0 construction check and the executor's pin.
    assert entry["expected_artifact_digest"] == B.B_ARTIFACT_DIGEST
    assert entry["candidate_id"] == B.REBUILT_STATE_ID
    assert Path(entry["checkpoint_dir"]).name == "baseline_rebuild"

    out = fallback.outcome
    assert out["resolution"] == "rebuilt" and out["rebuilds"] == 1
    assert all(out["identity_matches"].values()), out["identity_matches"]
    #: The pinned variant's own hash is NOT the frozen one, and the record says
    #: so rather than leaving a reader to assume they should match.
    assert out["pinned_spec_hash"] != B.B_SPEC_HASH
    assert out["construction"]["spec_hash"] == B.B_SPEC_HASH


def test_the_rebuild_path_actually_invokes_the_construction_gate(
        registered, tmp_path, monkeypatch):
    """The gate must be CALLED, not merely exist.

    Found by mutation: replacing `assert_frozen_construction(spec)` in
    `rebuild()` with a dict literal left every other test green. The gate was
    unit-tested, the constants were checked, and nothing proved the production
    path went through it — a mechanism with no caller, which tests prove works
    and do not prove anything uses.

    So this points `frozen_baseline_spec` at a path that is NOT the frozen one
    and requires the rebuild to refuse before it executes anything.
    """
    from aadistill.initialization.planning.fixed_path import FixedPathStep

    frozen = B.frozen_baseline_spec(device="cuda")
    impostor = frozen.replace_tail(
        len(frozen.steps) - 1,
        FixedPathStep("attention.activation_importance_v1",
                      "calib.reasoning_heavy@v2"),
        path_id=frozen.path_id)
    monkeypatch.setattr(B, "frozen_baseline_spec",
                        lambda **_: impostor)

    executed: list = []
    fallback = make_fallback(tmp_path, monkeypatch, calls=executed)
    with pytest.raises(B.BaselineError, match="not the frozen treatment path"):
        fallback(FakeResult([other_leaf()]), lambda: None)
    assert executed == [], (
        "the executor ran before the construction was verified; the $0 gate "
        "exists to refuse BEFORE the 27 minutes are spent")


def test_the_fallback_refuses_a_second_invocation(registered, tmp_path,
                                                  monkeypatch):
    """Single-shot, structurally. Two decisions would be two baselines."""
    fallback = make_fallback(tmp_path, monkeypatch)
    fallback(FakeResult([b_leaf()]), lambda: None)
    with pytest.raises(B.BaselineError, match="invoked twice"):
        fallback(FakeResult([b_leaf()]), lambda: None)


def test_a_rebuild_that_produces_something_else_is_refused(
        registered, tmp_path, monkeypatch):
    """Belt and braces: the executor's pin should already have raised.

    Kept because that pin lives in a different module, and the baseline the
    experiment's meaning rests on should not be protected by exactly one check
    in somebody else's file.
    """
    wrong = b_identity(config_sha256="c" * 64)
    assert wrong.artifact_digest != B.B_ARTIFACT_DIGEST
    fallback = make_fallback(tmp_path, monkeypatch,
                             steps=[real_shaped_step(wrong)])
    with pytest.raises(B.BaselineError, match="not B"):
        fallback(FakeResult([other_leaf()]), lambda: None)


def test_a_mismatch_says_which_component_moved_and_what_it_would_mean():
    """"The digest differs" is not a diagnosis."""
    weights_moved = B.explain_mismatch(
        b_identity(shards=(ShardRecord("model.safetensors", "d" * 64, 1),)))
    assert set(weights_moved["moved"]) >= {"weights_digest", "artifact_digest",
                                           "single_shard_sha256"}
    assert "scientific finding" in \
        weights_moved["what_each_would_mean"]["weights_digest"]

    config_moved = B.explain_mismatch(b_identity(config_sha256="e" * 64))
    assert "weights_digest" not in config_moved["moved"], (
        "a config-only difference must not read as different weights")
    assert "rope_theta" in \
        config_moved["what_each_would_mean"]["config_sha256"]
    #: C1's runtime travels with the explanation, because a config that moved
    #: between transformers versions is a runtime fact and a reader needs to be
    #: able to see that possibility.
    assert config_moved["c1_runtime"]["transformers"] == "5.13.1"
