"""Publishing the leaves to a transport repo must catch a bad far end.

The transport exists because continuation attempt 2 could not push 1.110 GiB
across the dev-box uplink inside the launcher's 600 s per-asset timeout. Moving
the bytes to a second repo fixes the delivery problem and creates a new one: a
copy that is *present* but *wrong*. A paid session would find that out after
staging, on a billing pod.

So the publisher verifies three independent ways, and these tests drive the real
verifier against a hub that lies in each of the ways that matter.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PUBLISH = REPO / "scripts/autoinit/publish_selected_leaves.py"


def load_publisher():
    spec = importlib.util.spec_from_file_location("publish_leaves", PUBLISH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["publish_leaves"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pub():
    return load_publisher()


def test_the_transport_repo_is_not_the_main_relay(pub):
    """A transport path that pointed at the main relay would reintroduce the
    quota problem it exists to avoid."""
    from aadistill.infrastructure.session import MAIN_RELAY

    assert pub.TRANSPORT_REPO != MAIN_RELAY
    assert "transport" in pub.TRANSPORT_REPO


def test_the_remote_paths_identify_the_attempt_and_the_state(pub):
    p = pub.remote_path("abc123", "model.safetensors")
    assert p.startswith("phase_a_attempt12/")
    assert "abc123" in p and p.endswith("model.safetensors")


def test_the_canonical_store_is_still_the_owner(pub):
    """The repo is a delivery path. If this ever points at the transport repo,
    the scientific owner has quietly moved."""
    assert str(pub.CANONICAL_STORE) == "/home/ecs-user/aad-artifacts/autoinit/phase_a"
    src = PUBLISH.read_text()
    assert "TRANSPORT ONLY" in src


def test_the_before_manifest_refuses_a_drifted_canonical_shard(pub, tmp_path,
                                                                monkeypatch):
    """Every digest is recomputed from the canonical bytes and checked against
    the attempt-12 record. A local copy that drifted makes everything
    downstream meaningless, so it must stop here.

    Driven against a synthetic store since 2026-08-22. It used to read the real
    5.55 GiB canonical checkpoints, so on any host without them — every pod —
    `build_before()` exited at "canonical leaf missing" and the drift assertion
    below never ran at all. What this test is about is the *refusal*, and
    synthetic bytes exercise it identically while letting it run anywhere.
    """
    store = tmp_path / "canonical"
    leaf = store / "leaf0"
    leaf.mkdir(parents=True)
    (leaf / "model.safetensors").write_bytes(b"the real weights")
    (leaf / "config.json").write_text('{"a": 1}')
    honest_shard = pub.sha256_file(leaf / "model.safetensors")

    evidence = tmp_path / "durability.json"
    evidence.write_text(json.dumps({"leaves": [{
        "state_id": "leaf0",
        "identity": {"single_shard_sha256": honest_shard,
                     "artifact_digest": "d" * 64, "config_sha256": "c" * 64,
                     "arch_signature": "a" * 64, "total_bytes": 16}}]}))
    monkeypatch.setattr(pub, "CANONICAL_STORE", store)
    monkeypatch.setattr(pub, "EVIDENCE", evidence)

    real = pub.sha256_file

    def wrong(path: Path):
        if str(path).endswith(".safetensors"):
            return "0" * 64
        return real(path)

    monkeypatch.setattr(pub, "sha256_file", wrong)
    with pytest.raises(SystemExit) as exc:
        pub.build_before()
    assert "does not match the attempt-12 record" in str(exc.value)


# --- the far end lies, three ways -------------------------------------------

def fake_leaf(tmp_path: Path, name: str, body: bytes) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(body)
    (d / "config.json").write_text('{"a": 1}')
    return d


def manifest_for(pub, files):
    """A minimal manifest shaped like `build_before()`'s output."""
    return {"repo": pub.TRANSPORT_REPO, "prefix": pub.PREFIX, "n_leaves": 1,
            "leaves": [{"selected_order": 0, "state_id": "leaf0",
                        "canonical_source": "/nonexistent",
                        "artifact_digest": "d" * 64,
                        "single_shard_sha256": "s" * 64,
                        "files": files}]}


def test_a_corrupted_remote_file_is_caught_by_the_round_trip(pub, tmp_path,
                                                             monkeypatch):
    """The hub serves different bytes than were uploaded."""
    good = b"the real weights"
    digest = hashlib.sha256(good).hexdigest()
    man = manifest_for(pub, [{"filename": "model.safetensors", "size_bytes": len(good),
                              "sha256": digest,
                              "remote_path": "phase_a_attempt12/leaf0/model.safetensors"}])

    served = tmp_path / "served.bin"
    served.write_bytes(b"corrupted bytes!")          # same length, wrong content
    monkeypatch.setattr(pub, "remote_oids", lambda m: {
        "phase_a_attempt12/leaf0/model.safetensors":
            {"size_bytes": len(good), "lfs_sha256": digest}})
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(served))
    monkeypatch.setattr(pub, "EVIDENCE", tmp_path / "evidence.json")
    (tmp_path / "evidence.json").write_text(json.dumps({"leaves": [
        {"state_id": "leaf0", "identity": {}}]}))

    result = pub.verify(man)
    assert any("round-trip sha256" in p for p in result["problems"]), result


def test_a_size_mismatch_at_the_far_end_is_caught(pub, tmp_path, monkeypatch):
    man = manifest_for(pub, [{"filename": "model.safetensors", "size_bytes": 1000,
                              "sha256": "a" * 64,
                              "remote_path": "phase_a_attempt12/leaf0/model.safetensors"}])
    monkeypatch.setattr(pub, "remote_oids", lambda m: {
        "phase_a_attempt12/leaf0/model.safetensors":
            {"size_bytes": 999, "lfs_sha256": "a" * 64}})
    monkeypatch.setattr(pub, "EVIDENCE", tmp_path / "e.json")
    (tmp_path / "e.json").write_text(json.dumps({"leaves": [
        {"state_id": "leaf0", "identity": {}}]}))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(tmp_path / "e.json"))
    result = pub.verify(man)
    assert any("999 bytes remote" in p for p in result["problems"]), result


def test_an_lfs_oid_that_disagrees_is_caught_without_downloading(pub, tmp_path,
                                                                 monkeypatch):
    """The cheap check: the hub's own content hash, compared before any bytes
    move. A mismatch here means the upload landed something else."""
    man = manifest_for(pub, [{"filename": "model.safetensors", "size_bytes": 10,
                              "sha256": "a" * 64,
                              "remote_path": "phase_a_attempt12/leaf0/model.safetensors"}])
    monkeypatch.setattr(pub, "remote_oids", lambda m: {
        "phase_a_attempt12/leaf0/model.safetensors":
            {"size_bytes": 10, "lfs_sha256": "b" * 64}})
    monkeypatch.setattr(pub, "EVIDENCE", tmp_path / "e.json")
    (tmp_path / "e.json").write_text(json.dumps({"leaves": [
        {"state_id": "leaf0", "identity": {}}]}))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(tmp_path / "e.json"))
    result = pub.verify(man)
    assert any("LFS oid" in p for p in result["problems"]), result


def test_a_file_absent_from_the_far_end_is_caught(pub, tmp_path, monkeypatch):
    man = manifest_for(pub, [{"filename": "model.safetensors", "size_bytes": 10,
                              "sha256": "a" * 64,
                              "remote_path": "phase_a_attempt12/leaf0/model.safetensors"}])
    monkeypatch.setattr(pub, "remote_oids", lambda m: {})
    monkeypatch.setattr(pub, "EVIDENCE", tmp_path / "e.json")
    (tmp_path / "e.json").write_text(json.dumps({"leaves": [
        {"state_id": "leaf0", "identity": {}}]}))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(tmp_path / "e.json"))
    result = pub.verify(man)
    assert any("absent from the transport repo" in p for p in result["problems"])


def test_verification_writes_no_manifest_when_it_fails(pub):
    """`--verify` exits non-zero and the session refuses an unverified manifest,
    so a bad far end cannot reach a paid run."""
    src = PUBLISH.read_text()
    assert 'man["verified"] = not result["problems"]' in src
    assert "raise SystemExit(1)" in src
    launcher = (REPO / "scripts/pod/autoinit_recovery_continuation_launch.py").read_text()
    assert 'return bool(man.get("verified"))' in launcher, (
        "the session would accept an unverified transport manifest")
    assert "no verified transport" in launcher, (
        "the $0 leaf gate no longer refuses an unverified transport")


# --- the attempt-3 regression ----------------------------------------------
#
# This module is part of the suite the pod's SETUP GATE runs, even though the
# publisher it exercises is a dev-box tool the paid session never executes.
# `verify()` used to `mkdtemp` into the literal `/home/ecs-user/aad-scratch`,
# which a pod does not have, so five tests here raised `FileNotFoundError` and
# recovery continuation attempt 3 died at $0.2011 — with the transport it was
# testing already proven on that same pod.

def test_the_round_trip_needs_no_dev_box_directory(pub, tmp_path, monkeypatch):
    """The attempt-3 condition: no configured scratch, no dev-box scratch.

    `verify()` must still run. Pointing `DEV_BOX_SCRATCH` at a path that does
    not exist reproduces a pod exactly for this code path, without needing one.
    """
    monkeypatch.delenv(pub.SCRATCH_ENV, raising=False)
    monkeypatch.setattr(pub, "DEV_BOX_SCRATCH", tmp_path / "no-such-scratch")
    assert pub.scratch_dir() is None, (
        "with neither root present the publisher must defer to tempfile, not "
        "name a directory the host may not have")

    man = manifest_for(pub, [{"filename": "model.safetensors", "size_bytes": 10,
                              "sha256": "a" * 64,
                              "remote_path": "phase_a_attempt12/leaf0/model.safetensors"}])
    monkeypatch.setattr(pub, "remote_oids", lambda m: {
        "phase_a_attempt12/leaf0/model.safetensors":
            {"size_bytes": 10, "lfs_sha256": "a" * 64}})
    monkeypatch.setattr(pub, "EVIDENCE", tmp_path / "e.json")
    (tmp_path / "e.json").write_text(json.dumps({"leaves": [
        {"state_id": "leaf0", "identity": {}}]}))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(tmp_path / "e.json"))

    # The assertion is that this returns at all. Before the fix it raised
    # FileNotFoundError from mkdtemp, which is what the pod saw.
    result = pub.verify(man)
    assert "problems" in result and "round_trip" in result


def test_a_configured_scratch_root_is_preferred_when_it_exists(pub, tmp_path,
                                                               monkeypatch):
    """The other branch of the same choice, so neither is left unexecuted."""
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setenv(pub.SCRATCH_ENV, str(configured))
    monkeypatch.setattr(pub, "DEV_BOX_SCRATCH", tmp_path / "no-such-scratch")
    assert pub.scratch_dir() == str(configured)

    # A configured root that is not there is a preference, not a requirement.
    monkeypatch.setenv(pub.SCRATCH_ENV, str(tmp_path / "absent"))
    assert pub.scratch_dir() is None
