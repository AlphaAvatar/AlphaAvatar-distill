"""A completed unit is persisted with its descriptor first and re-identified on arrival.

A synthetic checkpoint directory (a safetensors shard and a config) is enough:
the property is the persistence order and the identity check, not the model.
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


@pytest.fixture()
def ckpt(tmp_path):
    d = tmp_path / "trainer_out" / "model"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\x00" * 4096)
    (d / "config.json").write_text(json.dumps({"hidden_size": 8, "num_hidden_layers": 1}))
    return d


def test_a_completed_checkpoint_is_copied_and_re_identified(tmp_path, ckpt, adapter):
    root = tmp_path / "durable"
    rec = PU.persist_checkpoint("A-s1", ckpt, root, adapter=adapter, arch_signature=ARCH,
                                num_parameters=NPARAM, inputs={"pack": "x"})
    assert rec["persisted"] and rec["verified"]["matched"]
    assert (root / "A-s1/checkpoint/model.safetensors").read_bytes() == (ckpt / "model.safetensors").read_bytes()
    assert not (root / "A-s1/checkpoint.partial").exists()
    on_disk = json.loads((root / "A-s1/checkpoint.descriptor.json").read_text())
    assert on_disk["persisted"] and on_disk["artifact_digest"] == rec["artifact_digest"]


def test_no_room_refuses_before_any_byte_but_the_descriptor_is_written(tmp_path, ckpt, adapter):
    root = tmp_path / "durable"
    with pytest.raises(LeafDurabilityError, match="nothing was copied"):
        PU.persist_checkpoint("A-s1", ckpt, root, adapter=adapter, arch_signature=ARCH,
                              num_parameters=NPARAM, inputs={}, capacity=100)
    desc = json.loads((root / "A-s1/checkpoint.descriptor.json").read_text())
    assert desc["persisted"] is False and desc["artifact_digest"]
    assert not (root / "A-s1/checkpoint").exists() and not (root / "A-s1/checkpoint.partial").exists()


def test_bytes_that_change_in_transit_are_not_given_the_final_name(tmp_path, ckpt, adapter, monkeypatch):
    real = PU.shutil.copytree

    def corrupting(src, dst, **kw):
        real(src, dst, **kw)
        (dst / "model.safetensors").write_bytes(b"\x01" * 4096)
    monkeypatch.setattr(PU.shutil, "copytree", corrupting)
    with pytest.raises(LeafDurabilityError, match="do not re-identify"):
        PU.persist_checkpoint("A-s1", ckpt, tmp_path / "d", adapter=adapter, arch_signature=ARCH,
                              num_parameters=NPARAM, inputs={})
    assert not (tmp_path / "d/A-s1/checkpoint").exists()


def test_a_completed_unit_is_never_overwritten(tmp_path, ckpt, adapter):
    root = tmp_path / "durable"
    PU.persist_checkpoint("A-s1", ckpt, root, adapter=adapter, arch_signature=ARCH, num_parameters=NPARAM, inputs={})
    with pytest.raises(LeafDurabilityError, match="never overwritten"):
        PU.persist_checkpoint("A-s1", ckpt, root, adapter=adapter, arch_signature=ARCH, num_parameters=NPARAM, inputs={})


def test_raw_evaluation_outputs_are_persisted_with_their_hashes(tmp_path):
    f = tmp_path / "generations.jsonl"
    f.write_text('{"item": 1, "raw": "..."}\n')
    rec = PU.persist_files("B-s2", [f], tmp_path / "durable")
    assert rec["persisted"] and (tmp_path / "durable/B-s2/evaluation/generations.jsonl").read_text() == f.read_text()
    assert rec["files"][0]["sha256"] == PU._sha256(f)
