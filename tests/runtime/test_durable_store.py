"""The durable object store: generic interface, real backends, real identity.

A completed unit of work must survive the resource that produced it and be
consumable by a replacement. The two legs have opposite constraints -- the
upload runs where nothing is billing and may be slow, the download runs on the
billed replacement and must not be -- and a transport that inverts them
charges the slow leg at the accelerator's rate.

What is tested here is not throughput. It is that the interface carries no
vendor, that a transfer cannot happen without the ARRIVED bytes being
re-identified, and that every way of getting that wrong is refused.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.runtime import durable_store as DS  # noqa: E402

from experiments.durable_stores import (LocalDirStore,  # noqa: E402
                                        S3CompatibleStore)


# --- the interface names no vendor and no experiment ------------------------

def test_the_generic_module_names_no_vendor_protocol_or_experiment():
    """The property that lets the same core serve a later stage.

    A vendor, an endpoint, a bucket or a region in the interface would make
    the next backend an edit to generic runtime code. So would an experiment
    name, a model family or a parameter count.
    """
    src = (REPO / "src/aadistill/runtime/durable_store.py").read_text()
    banned = {
        "vendor or protocol": r"\b([Ss]3|R2|runpod|RunPod|cloudflare|"
                              r"[Hh]ugging|minio|aws|AWS|boto|azure|gcs)\b",
        "experiment": r"\b(phase_c[0-9]|attempt[0-9]|c2_behavioural|behavioural)\b",
        "model family": r"\b([Qq]wen|llama|mistral)\b",
        "a parameter count": r"\b\d{3}[,_]\d{3}[,_]\d{3}\b",
        "provider paths": r"(/workspace|aad-artifacts|runpod\.io)",
        "budgets": r"\$\d+\.\d{2}",
    }
    found = {k: sorted({m.group(0) for m in re.finditer(v, src)})
             for k, v in banned.items()}
    found = {k: v for k, v in found.items() if v}
    assert not found, f"the generic store interface encodes {found}"


@pytest.mark.parametrize("cls", [LocalDirStore, S3CompatibleStore])
def test_both_backends_satisfy_the_protocol(cls):
    assert isinstance(cls.__new__(cls), DS.DurableStore)


def test_the_object_store_backend_takes_its_endpoint_as_an_argument():
    """No vendor in the ADAPTER's logic either -- only at its construction.

    Two different services differ by the strings passed in, so a third needs
    no code change.
    """
    a = S3CompatibleStore(endpoint_url="https://one.example/", bucket="b1",
                          access_key_env="A_KEY", secret_key_env="A_SECRET")
    b = S3CompatibleStore(endpoint_url="https://two.example/", bucket="b2",
                          access_key_env="B_KEY", secret_key_env="B_SECRET")
    assert a.endpoint_url != b.endpoint_url and a.bucket != b.bucket
    src = (REPO / "scripts/experiments/durable_stores.py").read_text()
    body = src.split("class S3CompatibleStore", 1)[1]
    for host in ("runpod.io", "cloudflarestorage.com", "amazonaws.com"):
        assert host not in body, f"{host} is hardcoded into the adapter"


def test_credentials_are_never_defaulted_or_read_from_the_repository():
    """And the credential check runs BEFORE the client-library import.

    Both refuse, but "the key is not set" is actionable where "a library is
    missing" sends someone to install a dependency they may not need. The
    import came first and hid the useful message behind it.
    """
    s = S3CompatibleStore(endpoint_url="https://x.example/", bucket="b",
                          access_key_env="NOPE_KEY", secret_key_env="NOPE_SEC")
    with pytest.raises(RuntimeError, match="not in the environment"):
        s._client()
    body = (REPO / "scripts/experiments/durable_stores.py").read_text().split(
        "def _client", 1)[1].split("\n    def ")[0]
    assert body.index("not in the environment") < body.index("import boto3")


def test_a_configured_backend_with_no_client_library_says_so(monkeypatch):
    """The other refusal, reached only once credentials exist."""
    monkeypatch.setenv("SOME_KEY", "ak")
    monkeypatch.setenv("SOME_SECRET", "sk")
    s = S3CompatibleStore(endpoint_url="https://x.example/", bucket="b",
                          access_key_env="SOME_KEY",
                          secret_key_env="SOME_SECRET")
    try:
        import boto3  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="client library"):
            s._client()
    else:
        pytest.skip("a client library is installed on this host")


# --- a transfer cannot happen without the arrived bytes being re-identified -

class _Probe:
    """A checkpoint-shaped directory, built for the purpose.

    Not the real one: these tests must pass on a machine that has never run
    the experiment. The real 2.22 GiB probe is exercised separately, where it
    exists, by the campaign's own validation.
    """

    @staticmethod
    def make(root: Path, payload: bytes = b"weights" * 1024) -> Path:
        d = root / "probe"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(payload)
        (d / "config.json").write_text(json.dumps({"hidden_size": 8}))
        return d


class _Identity:
    """The identity functions, replaced by content hashing.

    `identify_for_transfer` needs an architecture adapter and a real
    safetensors file. What this file tests is the STORE's contract -- that it
    identifies before sending, re-identifies what arrives, and refuses a
    mismatch -- so the identity is stubbed to something equally strict and
    equally derived from bytes.
    """

    @staticmethod
    def install(monkeypatch):
        import hashlib

        def digest(d: Path) -> str:
            h = hashlib.sha256()
            for f in sorted(Path(d).rglob("*")):
                if f.is_file():
                    h.update(f.relative_to(d).as_posix().encode())
                    h.update(f.read_bytes())
            return h.hexdigest()

        class _Ident:
            def __init__(self, d):
                self._d = digest(d)
            def as_dict(self):
                return {"artifact_digest": self._d, "path": "x"}
            @property
            def artifact_digest(self):
                return self._d
            @property
            def weights_digest(self):
                return self._d

        import aadistill.runtime.leaf_durability as LD
        monkeypatch.setattr(LD, "identify_for_transfer",
                            lambda d, **k: _Ident(d))
        monkeypatch.setattr(
            LD, "verify_transferred_leaf",
            lambda d, rec, **k: {"matched": digest(d) == rec["artifact_digest"],
                                 "observed": digest(d)})


def test_a_round_trip_verifies_and_reports_both_legs(tmp_path, monkeypatch):
    _Identity.install(monkeypatch)
    src = _Probe.make(tmp_path / "src")
    store = LocalDirStore(tmp_path / "backend")
    up = DS.upload_checkpoint(store, src, "k/one", adapter=None,
                              arch_signature="sig", num_parameters=8)
    assert up.direction == "upload" and up.bytes > 0 and up.n_files == 2
    assert store.stat_tree("k/one")["n_files"] == 2
    down = DS.restore_checkpoint(store, "k/one", tmp_path / "arrived",
                                 adapter=None, identity=up.identity)
    assert down.verified and down.direction == "download"
    assert (tmp_path / "arrived" / "model.safetensors").is_file()
    assert up.identity["artifact_digest"] == down.identity["artifact_digest"]
    #: and both legs are reported, because the whole point is that one of them
    #: is cheap and the other is not
    assert up.as_dict()["mb_per_second"] >= 0
    assert down.as_dict()["mb_per_second"] >= 0


def test_a_corrupted_object_is_refused_and_left_in_place(tmp_path, monkeypatch):
    """The bytes stay as evidence about the transport."""
    _Identity.install(monkeypatch)
    src = _Probe.make(tmp_path / "src")
    store = LocalDirStore(tmp_path / "backend")
    up = DS.upload_checkpoint(store, src, "k/one", adapter=None,
                              arch_signature="sig", num_parameters=8)
    w = store.root / "k/one/model.safetensors"
    raw = bytearray(w.read_bytes())
    raw[-1] ^= 0x01
    w.write_bytes(bytes(raw))
    with pytest.raises(DS.DurableStoreError, match="did NOT re-identify"):
        DS.restore_checkpoint(store, "k/one", tmp_path / "arrived",
                              adapter=None, identity=up.identity)
    assert (tmp_path / "arrived" / "model.safetensors").is_file(), (
        "the arrival was deleted; it is evidence about the transport")


def test_a_restore_into_an_occupied_path_is_refused(tmp_path, monkeypatch):
    _Identity.install(monkeypatch)
    src = _Probe.make(tmp_path / "src")
    store = LocalDirStore(tmp_path / "backend")
    up = DS.upload_checkpoint(store, src, "k/one", adapter=None,
                              arch_signature="sig", num_parameters=8)
    (tmp_path / "arrived").mkdir()
    with pytest.raises(DS.DurableStoreError, match="already exists"):
        DS.restore_checkpoint(store, "k/one", tmp_path / "arrived",
                              adapter=None, identity=up.identity)


def test_uploading_something_that_is_not_a_directory_is_refused(tmp_path):
    store = LocalDirStore(tmp_path / "backend")
    f = tmp_path / "loose.bin"
    f.write_bytes(b"x")
    with pytest.raises(DS.DurableStoreError, match="not a directory"):
        DS.upload_checkpoint(store, f, "k", adapter=None,
                             arch_signature="s", num_parameters=1)


def test_a_missing_object_raises_rather_than_producing_an_empty_restore(
        tmp_path, monkeypatch):
    _Identity.install(monkeypatch)
    store = LocalDirStore(tmp_path / "backend")
    with pytest.raises(FileNotFoundError):
        DS.restore_checkpoint(store, "absent", tmp_path / "arrived",
                              adapter=None, identity={"artifact_digest": "x"})


# --- the store is not allowed to interpret a key ---------------------------

@pytest.mark.parametrize("key", ["../escape", "a/../../escape", "/abs/escape"])
def test_a_key_cannot_escape_the_store_root(tmp_path, key):
    store = LocalDirStore(tmp_path / "backend")
    with pytest.raises(ValueError, match="escapes the store root"):
        store._at(key)


def test_the_local_backend_reports_capacity_and_the_object_one_does_not():
    """A capacity gate can only check a backend that answers.

    Object storage does not report remaining space, and inventing a number
    would make a gate green about something nobody measured -- so it returns
    None and the caller must ask whoever owns the quota.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        assert LocalDirStore(t).free_bytes() > 0
    assert S3CompatibleStore(
        endpoint_url="https://x.example/", bucket="b",
        access_key_env="K", secret_key_env="S").free_bytes() is None
