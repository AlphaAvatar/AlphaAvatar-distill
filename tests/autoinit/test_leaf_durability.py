"""A successful Stage-1 selection must survive a failing Stage 2.

Attempt 11 spent 180.3 minutes producing five valid, measured, selected leaves
and lost every checkpoint, because persistence happened only after Stage-5
selection and Stage 2 failed six seconds after Stage 1 passed. Collection *did*
run on that failure path — the manifest came home with `rc=0` — so the fix is to
make the selected leaves a collected **artifact** at the Stage-1/2 boundary
rather than a Stage-5 fetch product.

These use a tiny real checkpoint written by the real adapter, so the digest that
is compared is the digest the search would have recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.artifact import identify_checkpoint  # noqa: E402
from aadistill.autoinit.leaf_durability import (  # noqa: E402
    LeafDurabilityError, free_bytes_at, persist_selected_leaves,
)


@pytest.fixture(scope="module")
def tiny_leaf(tmp_path_factory):
    """A real, tiny, weight-only checkpoint — what `Qwen3Adapter.save()` writes."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    adapter = get_adapter("qwen3")
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    d = tmp_path_factory.mktemp("leaf_src") / "state"
    adapter.save(model, str(d))
    spec = adapter.spec_from_config(cfg)
    identity = identify_checkpoint(d, adapter=adapter, spec=spec,
                                   num_parameters=adapter.param_count(spec))
    # The producer's contract: weight-only, so the digest carries no tokenizer.
    assert identity.tokenizer_sha256 is None
    assert not (d / "tokenizer.json").is_file()
    return {
        "adapter": adapter, "spec": spec, "dir": d,
        "leaf": {"state_id": "abc123def456", "checkpoint_path": str(d),
                 "artifact_digest": identity.artifact_digest,
                 "total_bytes": identity.total_bytes,
                 "num_parameters": identity.num_parameters}}


def _persist(tiny_leaf, dest, **kw):
    return persist_selected_leaves(
        leaves=[tiny_leaf["leaf"]], destination=dest,
        adapter=tiny_leaf["adapter"], spec=tiny_leaf["spec"], **kw)


def test_a_selected_leaf_is_persisted_and_its_identity_reverified(tiny_leaf, tmp_path):
    out = _persist(tiny_leaf, tmp_path / "durable", margin_bytes=0)
    assert out["n_leaves"] == 1
    entry = out["leaves"][0]
    assert entry["artifact_digest"] == tiny_leaf["leaf"]["artifact_digest"]
    assert entry["recorded_digest"] == entry["artifact_digest"]
    assert entry["tokenizer_sha256"] is None
    assert (Path(entry["path"]) / "config.json").is_file()


def test_the_persisted_copy_is_weight_only(tiny_leaf, tmp_path):
    """No tokenizer files are added: `artifact_digest` folds in
    `tokenizer_sha256`, so adding them would move the identity the search
    metrics hang on."""
    out = _persist(tiny_leaf, tmp_path / "durable", margin_bytes=0)
    landed = {p.name for p in Path(out["leaves"][0]["path"]).iterdir()}
    assert not landed & {"tokenizer.json", "tokenizer_config.json", "vocab.json"}


def test_a_copy_that_acquires_a_tokenizer_is_refused(tiny_leaf, tmp_path):
    """The guard, exercised. Asserting only that the OUTPUT has no tokenizer
    passes trivially when the source has none — it never reaches the check.
    This copies a tokenizer in, which both trips the guard and (because
    `tokenizer_sha256` enters the digest) would move the identity."""
    from transformers import AutoTokenizer

    canonical = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    if not (canonical / "tokenizer.json").is_file():
        pytest.skip("the canonical initialization is not staged here")

    def adds_a_tokenizer(src, dst):
        import shutil
        shutil.copytree(src, dst)
        AutoTokenizer.from_pretrained(canonical).save_pretrained(dst)

    with pytest.raises(LeafDurabilityError) as exc:
        _persist(tiny_leaf, tmp_path / "durable", margin_bytes=0,
                 copier=adds_a_tokenizer)
    # Either guard is a correct refusal, and both are about the same thing: the
    # persisted artifact is no longer the one the search measured.
    assert ("carries tokenizer files" in str(exc.value)
            or "after transfer the checkpoint" in str(exc.value)), exc.value


def test_a_digest_that_changed_in_transit_is_refused(tiny_leaf, tmp_path):
    """A copy that silently truncated passes every check made at the source."""
    def corrupting(src, dst):
        import shutil
        shutil.copytree(src, dst)
        cfg = json.loads((dst / "config.json").read_text())
        cfg["intermediate_size"] = cfg["intermediate_size"] + 1
        (dst / "config.json").write_text(json.dumps(cfg))

    with pytest.raises(LeafDurabilityError, match="after transfer the checkpoint"):
        _persist(tiny_leaf, tmp_path / "durable", margin_bytes=0, copier=corrupting)


def test_a_destination_without_room_is_refused_before_anything_moves(
        tiny_leaf, tmp_path):
    """The space is checked before the first byte, because a persistence step
    that runs out halfway has already destroyed what it was protecting.

    The real numbers: attempt 11's five leaves are 5.55 GiB, the relay had
    ~1.03 GiB free and the dev box 3.4 GiB.
    """
    moved = []
    dest = tmp_path / "durable"
    with pytest.raises(LeafDurabilityError, match="cannot durably preserve"):
        _persist(tiny_leaf, dest, free_bytes=lambda p: 1,
                 copier=lambda s, d: moved.append(d))
    assert not moved, "a copy started despite insufficient space"
    assert not dest.exists(), "the destination was created before the check passed"


def test_the_headroom_check_uses_a_margin(tiny_leaf, tmp_path):
    """Exactly-enough is not enough: filling a volume to the last byte is its
    own failure."""
    need = tiny_leaf["leaf"]["total_bytes"]
    with pytest.raises(LeafDurabilityError, match="cannot durably preserve"):
        _persist(tiny_leaf, tmp_path / "d1", free_bytes=lambda p: need,
                 margin_bytes=1 << 20)
    out = _persist(tiny_leaf, tmp_path / "d2",
                   free_bytes=lambda p: need + (1 << 20), margin_bytes=1 << 20)
    assert out["n_leaves"] == 1


def test_a_missing_source_checkpoint_is_refused(tiny_leaf, tmp_path):
    leaf = {**tiny_leaf["leaf"], "checkpoint_path": str(tmp_path / "gone")}
    with pytest.raises(LeafDurabilityError, match="is not on disk"):
        persist_selected_leaves(
            leaves=[leaf], destination=tmp_path / "durable",
            adapter=tiny_leaf["adapter"], spec=tiny_leaf["spec"], margin_bytes=0)


def test_an_empty_selection_is_refused(tiny_leaf, tmp_path):
    """Zero leaves is not "nothing to do" — it means selection did not happen."""
    with pytest.raises(LeafDurabilityError, match="no selected leaves"):
        persist_selected_leaves(
            leaves=[], destination=tmp_path / "durable",
            adapter=tiny_leaf["adapter"], spec=tiny_leaf["spec"])


def test_free_bytes_walks_up_to_an_existing_ancestor(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "d"
    assert free_bytes_at(deep) == free_bytes_at(tmp_path)
    assert free_bytes_at(deep) > 0


# --- the wiring, not just the mechanism -------------------------------------

def test_stage1_persists_before_stage2_and_fails_closed():
    """Read off the real driver: the boundary must sit between SEARCH_DONE and
    stage 1 returning True, and a failure must fail STAGE 1 — not warn."""
    src = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    stage1 = src[src.index("def stage1(self)"):src.index("def probe_config")]

    assert "persist_selected_leaves(" in stage1
    assert stage1.index("SEARCH_DONE") < stage1.index("persist_selected_leaves("), (
        "persistence runs before the selection is complete")
    assert (stage1.index("persist_selected_leaves(")
            < stage1.index("return self.record(1, True")), (
        "persistence runs after stage 1 has already reported success, so stage 2 "
        "could start without it")
    # The except block must RETURN a stage-1 failure, not log and continue.
    import ast

    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "stage1")
    handlers = [h for n in ast.walk(fn) if isinstance(n, ast.Try) for h in n.handlers
                if "LeafDurabilityError" in ast.unparse(h.type or ast.Constant(None))]
    assert len(handlers) == 1, "no LeafDurabilityError handler in stage1"
    body = ast.unparse(handlers[0])
    assert "return self.record(1, False" in body.replace("\n", " ").replace("  ", " "), (
        f"a durability failure does not fail stage 1 closed: {body[:300]}")


def test_the_destination_is_collected_on_the_failure_path():
    """Under AUDIT, because collection walks that tree on every path including
    failure — which is exactly what attempt 11 needed and did not have."""
    import ast

    src = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "selected_leaf_dir")
    # The returned expression only — text slicing here over-ran into the
    # surrounding constants, several of which legitimately mention REPO.
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    expr = ast.unparse(ret.value)
    # Derived from AUDIT, never rebuilt from REPO: as a module constant it
    # ignored the rehearsal's audit-root redirection and wrote 193 MB of real
    # leaf directories into the repository's own artifact tree.
    assert expr.startswith("AUDIT /"), expr
    assert "REPO" not in expr, expr


def test_only_the_selected_leaves_are_persisted():
    """43 states were searched; five were selected. The other 38 are recorded
    and intentionally not preserved."""
    src = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    stage1 = src[src.index("def stage1(self)"):src.index("def probe_config")]
    call = stage1[stage1.index("persist_selected_leaves("):]
    call = call[:call.index("except LeafDurabilityError")]
    assert "for s in self.leaves" in call, call[:400]
    assert "found.states" not in call and "all_states" not in call
