"""A completed unit is persisted once, re-identified on arrival, and never altered after.

A synthetic checkpoint directory (a safetensors shard and a config) is enough:
the properties are the persistence order, the identity check and that a
completed record survives every later attempt byte for byte.
"""
from __future__ import annotations

import json

import pytest

from stages.reasoning_data import persist_unit as PU

from aadistill.initialization.adapters import register_builtin_adapters
from aadistill.initialization.specs.arch import get_adapter
from aadistill.runtime.leaf_durability import LeafDurabilityError

ARCH, NPARAM = "a" * 64, 1234


@pytest.fixture()
def adapter():
    register_builtin_adapters()
    return get_adapter("qwen3")


def _ckpt(d, fill=b"\x00"):
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(fill * 4096)
    (d / "config.json").write_text(json.dumps({"hidden_size": 8, "num_hidden_layers": 1}))
    return d


@pytest.fixture()
def ckpt(tmp_path):
    return _ckpt(tmp_path / "trainer_out" / "model")


def _save(ckpt, root, adapter, **kw):
    kw.setdefault("inputs", {"pack": "x"})
    return PU.persist_checkpoint("A-s1", ckpt, root, adapter=adapter, arch_signature=ARCH,
                                 num_parameters=NPARAM, **kw)


def _snapshot(d):
    """Every file under `d`: bytes and mtime, to prove nothing was touched."""
    return {str(p.relative_to(d)): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(d.rglob("*")) if p.is_file()}


def test_a_completed_checkpoint_is_copied_and_re_identified(tmp_path, ckpt, adapter):
    root = tmp_path / "durable"
    rec = _save(ckpt, root, adapter)
    assert rec["status"] == "persisted" and rec["persisted"] and rec["verified"]["matched"]
    assert (root / "A-s1/checkpoint/model.safetensors").read_bytes() == (ckpt / "model.safetensors").read_bytes()
    assert not (root / "A-s1/checkpoint.partial").exists()
    assert not (root / "A-s1/checkpoint.inflight.json").exists()
    on_disk = json.loads((root / "A-s1/checkpoint.descriptor.json").read_text())
    assert on_disk["persisted"] and on_disk["artifact_digest"] == rec["artifact_digest"]
    assert "status" not in on_disk


def test_no_room_refuses_before_any_byte_and_writes_only_the_inflight_record(tmp_path, ckpt, adapter):
    root = tmp_path / "durable"
    with pytest.raises(LeafDurabilityError, match="nothing was copied"):
        _save(ckpt, root, adapter, capacity=100)
    inflight = json.loads((root / "A-s1/checkpoint.inflight.json").read_text())
    assert inflight["persisted"] is False and inflight["artifact_digest"] and "nothing was copied" in inflight["error"]
    assert not (root / "A-s1/checkpoint.descriptor.json").exists()
    assert not (root / "A-s1/checkpoint").exists() and not (root / "A-s1/checkpoint.partial").exists()
    rec = _save(ckpt, root, adapter)          # the retry, with room
    assert rec["status"] == "persisted" and PU.completed(root, "A-s1", "checkpoint")["persisted"]
    assert not (root / "A-s1/checkpoint.inflight.json").exists()


def test_bytes_that_change_in_transit_are_not_given_the_final_name(tmp_path, ckpt, adapter, monkeypatch):
    real = PU.shutil.copytree

    def corrupting(src, dst, **kw):
        real(src, dst, **kw)
        (dst / "model.safetensors").write_bytes(b"\x01" * 4096)
    monkeypatch.setattr(PU.shutil, "copytree", corrupting)
    with pytest.raises(LeafDurabilityError, match="do not re-identify"):
        _save(ckpt, tmp_path / "d", adapter)
    assert not (tmp_path / "d/A-s1/checkpoint").exists()
    assert not (tmp_path / "d/A-s1/checkpoint.descriptor.json").exists()


def test_the_same_checkpoint_saved_again_is_already_persisted_and_nothing_is_touched(tmp_path, ckpt, adapter):
    root = tmp_path / "durable"
    _save(ckpt, root, adapter)
    before = _snapshot(root)
    rec = _save(ckpt, root, adapter)
    assert rec["status"] == "already_persisted" and rec["consistent"] is True
    assert _snapshot(root) == before


@pytest.mark.parametrize("change", ["weights", "inputs"])
def test_a_different_identity_is_refused_and_the_completed_unit_is_untouched(tmp_path, ckpt, adapter, change):
    root = tmp_path / "durable"
    _save(ckpt, root, adapter)
    before = _snapshot(root)
    other = _ckpt(tmp_path / "other", b"\x07") if change == "weights" else ckpt
    inputs = {"pack": "y"} if change == "inputs" else {"pack": "x"}
    with pytest.raises(LeafDurabilityError, match="different checkpoint is already persisted"):
        _save(other, root, adapter, inputs=inputs)
    assert _snapshot(root) == before


def test_no_room_on_a_completed_unit_leaves_its_completed_record_intact(tmp_path, ckpt, adapter):
    root = tmp_path / "durable"
    _save(ckpt, root, adapter)
    before = _snapshot(root)
    assert _save(ckpt, root, adapter, capacity=100)["status"] == "already_persisted"
    with pytest.raises(LeafDurabilityError, match="different checkpoint"):
        _save(_ckpt(tmp_path / "other", b"\x07"), root, adapter, capacity=100)
    assert _snapshot(root) == before
    assert PU.completed(root, "A-s1", "checkpoint")["persisted"] is True


def test_an_attempt_stopped_between_rename_and_commit_is_committed_only_for_its_own_identity(
        tmp_path, ckpt, adapter, monkeypatch):
    root = tmp_path / "durable"
    real = PU._commit
    monkeypatch.setattr(PU, "_commit", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _save(ckpt, root, adapter)
    monkeypatch.setattr(PU, "_commit", real)
    assert (root / "A-s1/checkpoint").is_dir() and not (root / "A-s1/checkpoint.descriptor.json").exists()
    before = _snapshot(root)
    with pytest.raises(LeafDurabilityError, match="unknown provenance"):
        _save(_ckpt(tmp_path / "other", b"\x07"), root, adapter)
    assert _snapshot(root) == before
    rec = _save(ckpt, root, adapter)
    assert rec["status"] == "persisted" and rec["completed_after_interruption"] is True


def test_a_stale_non_completed_descriptor_is_refused_not_overwritten(tmp_path, ckpt, adapter):
    """A v1-style descriptor with persisted=false is not this module's completed
    record; it is preserved for diagnosis."""
    root = tmp_path / "durable"
    (root / "A-s1").mkdir(parents=True)
    (root / "A-s1/checkpoint.descriptor.json").write_text(json.dumps({"persisted": False}))
    with pytest.raises(LeafDurabilityError, match="not a completed record"):
        _save(ckpt, root, adapter)
    assert json.loads((root / "A-s1/checkpoint.descriptor.json").read_text()) == {"persisted": False}


# ---- file sets ---------------------------------------------------------------------------------

def test_raw_evaluation_outputs_are_persisted_with_their_hashes(tmp_path):
    f = tmp_path / "generations.jsonl"
    f.write_text('{"item": 1, "raw": "..."}\n')
    rec = PU.persist_files("B-s2", [f], tmp_path / "durable")
    assert rec["status"] == "persisted"
    assert (tmp_path / "durable/B-s2/evaluation/generations.jsonl").read_text() == f.read_text()
    assert rec["files"][0]["sha256"] == PU._sha256(f)


def _two_same_named(tmp_path):
    a, b = tmp_path / "out/fluid/summary.json", tmp_path / "out/capability/summary.json"
    for p, t in ((a, "fluid"), (b, "capability")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
    return a, b


def test_same_named_inputs_without_a_base_are_refused_before_any_write(tmp_path):
    a, b = _two_same_named(tmp_path)
    with pytest.raises(LeafDurabilityError, match="two inputs have the logical name 'summary.json'"):
        PU.persist_files("B-s2", [a, b], tmp_path / "durable")
    assert not (tmp_path / "durable").exists()


def test_same_named_inputs_keep_distinct_logical_paths_under_a_base(tmp_path):
    a, b = _two_same_named(tmp_path)
    root = tmp_path / "durable"
    rec = PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out")
    assert [r["name"] for r in rec["files"]] == ["capability/summary.json", "fluid/summary.json"]
    assert (root / "B-s2/evaluation/fluid/summary.json").read_text() == "fluid"
    assert (root / "B-s2/evaluation/capability/summary.json").read_text() == "capability"


@pytest.mark.parametrize("name", ["../x", "/abs", "a/./b", ""])
def test_a_logical_name_that_escapes_or_is_not_normalized_is_refused(tmp_path, name):
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(LeafDurabilityError, match="not a normalized relative logical name"):
        PU.persist_files("B-s2", {name: f}, tmp_path / "durable")


def test_the_same_file_set_again_is_already_persisted_and_a_different_one_is_refused(tmp_path):
    a, b = _two_same_named(tmp_path)
    root = tmp_path / "durable"
    PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out")
    before = _snapshot(root)
    again = PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out")
    assert again["status"] == "already_persisted" and again["consistent"] is True
    a.write_text("changed")
    with pytest.raises(LeafDurabilityError, match="different evaluation file set is already persisted"):
        PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out")
    with pytest.raises(LeafDurabilityError, match="different evaluation file set"):
        PU.persist_files("B-s2", [b], root, base=tmp_path / "out", capacity=1)
    assert _snapshot(root) == before


def test_no_room_for_a_file_set_then_a_retry_that_succeeds(tmp_path):
    a, b = _two_same_named(tmp_path)
    root = tmp_path / "durable"
    with pytest.raises(LeafDurabilityError, match="nothing was copied"):
        PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out", capacity=1)
    assert not (root / "B-s2/evaluation").exists() and PU.completed(root, "B-s2", "evaluation") is None
    assert "nothing was copied" in json.loads((root / "B-s2/evaluation.inflight.json").read_text())["error"]
    rec = PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out")
    assert rec["status"] == "persisted" and not (root / "B-s2/evaluation.inflight.json").exists()
    before = _snapshot(root)
    assert PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out",
                            capacity=1)["status"] == "already_persisted"
    assert _snapshot(root) == before


def test_a_file_set_landed_without_a_record_of_it_is_unknown_provenance(tmp_path):
    a, b = _two_same_named(tmp_path)
    root = tmp_path / "durable"
    (root / "B-s2/evaluation").mkdir(parents=True)
    (root / "B-s2/evaluation/stray.txt").write_text("?")
    with pytest.raises(LeafDurabilityError, match="unknown provenance"):
        PU.persist_files("B-s2", [a, b], root, base=tmp_path / "out")
    assert (root / "B-s2/evaluation/stray.txt").read_text() == "?"
