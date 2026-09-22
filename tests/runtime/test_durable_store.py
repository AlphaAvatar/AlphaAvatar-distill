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

@pytest.mark.parametrize("key", ["../escape", "a/../../escape", "/abs/escape",
                                 "..", "../../etc/passwd"])
def test_a_key_cannot_escape_the_store_root(tmp_path, key):
    store = LocalDirStore(tmp_path / "backend")
    with pytest.raises(ValueError, match="escapes the store root"):
        store._at(key)


def test_a_sibling_whose_name_merely_starts_with_the_root_is_outside_it(
        tmp_path):
    """`str.startswith` is not a containment predicate, and this used it.

    With a root of `<t>/backend`, the sibling `<t>/backend_evil/x` has the
    root as a STRING prefix and passed. Containment compares path COMPONENTS.
    """
    (tmp_path / "backend_evil").mkdir(parents=True)
    store = LocalDirStore(tmp_path / "backend")
    with pytest.raises(ValueError, match="escapes the store root"):
        store._at("../backend_evil/x")
    #: and the string predicate it replaced WOULD have allowed it, which is
    #: what makes this a regression rather than a restatement
    escaped = (store.root / "../backend_evil/x").resolve()
    assert str(escaped).startswith(str(store.root.resolve())), (
        "the sibling no longer collides by string prefix, so this test no "
        "longer exercises the defect it was written for")
    assert not escaped.is_relative_to(store.root.resolve())


def test_a_legitimate_nested_key_is_allowed(tmp_path):
    """So the refusals above are known to be selective."""
    store = LocalDirStore(tmp_path / "backend")
    assert store._at("campaign/probe/one").is_relative_to(
        store.root.resolve())


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


# --- an upload is not durable until the STORED bytes say so ----------------

class TestTheUploadAcknowledgement:
    """`verified=True` is what a producer releases its only local copy on.

    It used to mean "the source identity was computed and the PUT returned",
    which proves nothing about the bytes in the backend.
    """

    def _setup(self, tmp_path, monkeypatch):
        _Identity.install(monkeypatch)
        src = _Probe.make(tmp_path / "src")
        return src, LocalDirStore(tmp_path / "backend")

    def test_it_is_established_by_reading_the_stored_bytes_back(
            self, tmp_path, monkeypatch):
        src, store = self._setup(tmp_path, monkeypatch)
        up = DS.upload_checkpoint(store, src, "k", adapter=None,
                                  arch_signature="sig", num_parameters=8,
                                  scratch=tmp_path)
        assert up.verified
        assert up.as_dict()["verified_by"] == "readback of the stored bytes"
        assert up.as_dict()["source_identity_matched"] is True

    def test_corrupted_stored_bytes_fail_the_upload_not_only_the_restore(
            self, tmp_path, monkeypatch):
        """The whole point: the producer must not release on a bad store.

        The transport is made to corrupt what it writes, so the PUT returns
        success and the stored bytes are wrong -- which is precisely the case
        a source-side identity cannot see.
        """
        src, store = self._setup(tmp_path, monkeypatch)
        real_put = store.put_tree

        def corrupting_put(local_dir, key):
            out = real_put(local_dir, key)
            f = store.root / key / "model.safetensors"
            raw = bytearray(f.read_bytes())
            raw[-1] ^= 0x01
            f.write_bytes(bytes(raw))
            return out

        monkeypatch.setattr(store, "put_tree", corrupting_put)
        with pytest.raises(DS.DurableStoreError, match="do not re-identify"):
            DS.upload_checkpoint(store, src, "k", adapter=None,
                                 arch_signature="sig", num_parameters=8,
                                 scratch=tmp_path)
        assert (store.root / "k").is_dir(), (
            "the stored object was removed; it is evidence about the transport")

    def test_verify_false_records_itself_and_is_not_an_acknowledgement(
            self, tmp_path, monkeypatch):
        src, store = self._setup(tmp_path, monkeypatch)
        up = DS.upload_checkpoint(store, src, "k", adapter=None,
                                  arch_signature="sig", num_parameters=8,
                                  verify=False, scratch=tmp_path)
        assert up.verified is False
        assert up.as_dict()["verified_by"] is None
        assert "must NOT be treated as a durable acknowledgement" in \
            up.as_dict()["_not_verified"]

    def test_the_scratch_readback_leaves_nothing_behind(
            self, tmp_path, monkeypatch):
        """It is transient, not a second durable copy -- including on failure."""
        src, store = self._setup(tmp_path, monkeypatch)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        DS.upload_checkpoint(store, src, "k", adapter=None,
                             arch_signature="sig", num_parameters=8,
                             scratch=scratch)
        assert list(scratch.iterdir()) == []

    def test_no_backend_metadata_is_trusted(self):
        """Provider-neutral: only `get_tree` is used to verify.

        An ETag is a digest of digests whose value depends on the client's
        part size, so it is not a cryptographic identity of the content.
        """
        import ast

        src = (REPO / "src/aadistill/runtime/durable_store.py").read_text()
        #: CODE ONLY. The docstring explains WHY an ETag is not an identity,
        #: so a substring search over the source flags the very sentence that
        #: forbids it -- a test that cannot tell code from the prose
        #: describing it fails on a correct implementation.
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_stored_identity_matches")
        body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                               and isinstance(fn.body[0].value, ast.Constant)
                               ) else fn.body
        code = "\n".join(ast.unparse(s) for s in body)
        for meta in ("etag", "ETag", "checksum", "ContentMD5", "md5"):
            assert meta not in code, f"{meta} is trusted as an identity"
        assert "store.get_tree(" in code, (
            "the readback does not go through the store's own transport, so "
            "it is not provider-neutral")


# --- a scientific artifact's key is immutable -------------------------------

class TestKeysAreImmutable:
    def _setup(self, tmp_path, monkeypatch):
        _Identity.install(monkeypatch)
        return _Probe.make(tmp_path / "src"), LocalDirStore(tmp_path / "backend")

    def test_re_uploading_the_same_artifact_is_idempotent(
            self, tmp_path, monkeypatch):
        src, store = self._setup(tmp_path, monkeypatch)
        first = DS.upload_checkpoint(store, src, "k", adapter=None,
                                     arch_signature="sig", num_parameters=8,
                                     scratch=tmp_path)
        again = DS.upload_checkpoint(store, src, "k", adapter=None,
                                     arch_signature="sig", num_parameters=8,
                                     scratch=tmp_path)
        assert again.as_dict()["already_present"] is True and again.verified
        assert again.identity["artifact_digest"] == \
            first.identity["artifact_digest"]

    def test_a_different_artifact_at_the_same_key_is_refused(
            self, tmp_path, monkeypatch):
        src, store = self._setup(tmp_path, monkeypatch)
        DS.upload_checkpoint(store, src, "k", adapter=None,
                             arch_signature="sig", num_parameters=8,
                             scratch=tmp_path)
        other = _Probe.make(tmp_path / "other", payload=b"different" * 1024)
        with pytest.raises(DS.DurableStoreError, match="already occupied by a "
                                                       "DIFFERENT artifact"):
            DS.upload_checkpoint(store, other, "k", adapter=None,
                                 arch_signature="sig", num_parameters=8,
                                 scratch=tmp_path)

    def test_an_occupied_key_with_verify_false_is_refused(
            self, tmp_path, monkeypatch):
        """Without a readback there is no way to know it is the same artifact."""
        src, store = self._setup(tmp_path, monkeypatch)
        DS.upload_checkpoint(store, src, "k", adapter=None,
                             arch_signature="sig", num_parameters=8,
                             scratch=tmp_path)
        with pytest.raises(DS.DurableStoreError, match="verify=False"):
            DS.upload_checkpoint(store, src, "k", adapter=None,
                                 arch_signature="sig", num_parameters=8,
                                 verify=False, scratch=tmp_path)

    def test_the_backend_itself_refuses_rather_than_emptying(self, tmp_path):
        """It called `rmtree` first -- destroying possibly the only copy of a
        completed measurement to make room for unverified bytes."""
        src = _Probe.make(tmp_path / "src")
        store = LocalDirStore(tmp_path / "backend")
        store.put_tree(src, "k")
        with pytest.raises(FileExistsError, match="immutable"):
            store.put_tree(src, "k")
        assert (store.root / "k" / "model.safetensors").is_file()

    def test_a_stale_file_cannot_survive_into_a_later_restore(
            self, tmp_path, monkeypatch):
        """The failure mode the immutability rule exists for.

        An upload that merged would leave `stale.bin` under the prefix, and a
        restore would reassemble a directory that was never any measurement.
        """
        src, store = self._setup(tmp_path, monkeypatch)
        store.put_tree(src, "k")
        (store.root / "k" / "stale.bin").write_bytes(b"left over")
        smaller = _Probe.make(tmp_path / "smaller")
        with pytest.raises((DS.DurableStoreError, FileExistsError)):
            DS.upload_checkpoint(store, smaller, "k", adapter=None,
                                 arch_signature="sig", num_parameters=8,
                                 scratch=tmp_path)

    def test_the_object_backend_refuses_a_populated_prefix_too(self):
        """Object stores have no directory to replace, so it is explicit."""
        body = (REPO / "scripts/experiments/durable_stores.py").read_text()
        s3 = body.split("class S3CompatibleStore", 1)[1]
        put = s3.split("def put_tree", 1)[1].split("\n    def ")[0]
        assert "self.stat_tree(key)" in put and "FileExistsError" in put


# --- the pod-side backend, exercised over a real HTTP server ---------------

class TestThePresignedFetchPath:
    """The side that downloads holds no credential and no client library.

    Giving the pod the bucket keys would put a long-lived secret on a machine
    the project does not own, in an environment whose logs are collected as
    evidence, and would make every pod carry an S3 client for one fetch. A
    signed, expiring URL per object is less to leak and nothing to install.

    Run against a REAL http server over a REAL directory. A stubbed transport
    would prove that the test's fake works; four paid pods in this project
    have died inside lines no `$0` path ever executed.
    """

    @staticmethod
    def _serve(root: Path):
        import functools
        import http.server
        import threading

        handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                    directory=str(root))
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    def test_it_fetches_a_real_tree_and_re_identifies_it(self, tmp_path,
                                                         monkeypatch):
        from experiments.durable_stores import (PresignedFetchPlan,
                                                PresignedHttpStore)
        _Identity.install(monkeypatch)
        src = _Probe.make(tmp_path / "src")
        #: `_Probe.make` returns the probe DIRECTORY, one level below the root
        #: it was given. Serving the root instead put every object one path
        #: segment deeper than the plan named them.
        srv, base = self._serve(src)
        try:
            files = {f.name: f.stat().st_size
                     for f in src.iterdir() if f.is_file()}
            plan = PresignedFetchPlan(
                "k", {n: f"{base}/{n}" for n in files}, files, 3600)
            store = PresignedHttpStore(plan)
            assert store.stat_tree("k")["n_files"] == len(files)
            up_identity = {"artifact_digest": __import__("hashlib").sha256(
                b"").hexdigest()}
            #: the real generic restore, over the real socket
            import aadistill.runtime.leaf_durability as LD
            ident = LD.identify_for_transfer(src, adapter=None,
                                             arch_signature="s",
                                             num_parameters=1)
            rec = DS.restore_checkpoint(
                store, "k", tmp_path / "arrived", adapter=None,
                identity={"artifact_digest": ident.artifact_digest})
            assert rec.verified and rec.n_files == len(files)
            for name in files:
                assert (tmp_path / "arrived" / name).read_bytes() == \
                    (src / name).read_bytes(), f"{name} did not survive"
        finally:
            srv.shutdown()

    def test_it_cannot_upload_and_says_why(self, tmp_path):
        from experiments.durable_stores import (PresignedFetchPlan,
                                                PresignedHttpStore)
        store = PresignedHttpStore(PresignedFetchPlan("k", {"a": "u"},
                                                      {"a": 1}, 60))
        with pytest.raises(NotImplementedError, match="READ access only"):
            store.put_tree(tmp_path, "k")

    def test_a_plan_covers_exactly_one_object(self, tmp_path):
        from experiments.durable_stores import (PresignedFetchPlan,
                                                PresignedHttpStore)
        store = PresignedHttpStore(PresignedFetchPlan("k", {"a": "u"},
                                                      {"a": 1}, 60))
        with pytest.raises(KeyError, match="covers exactly one object"):
            store.get_tree("other", tmp_path / "x")

    def test_a_malicious_object_name_cannot_escape_the_restore_root(
            self, tmp_path):
        """The narrow hardening: a key suffix becomes a local path."""
        from experiments.durable_stores import (PresignedFetchPlan,
                                                PresignedHttpStore)
        srv, base = self._serve(tmp_path)
        try:
            store = PresignedHttpStore(PresignedFetchPlan(
                "k", {"../escaped": f"{base}/x"}, {"../escaped": 1}, 60))
            with pytest.raises(ValueError, match="outside the restore root"):
                store.get_tree("k", tmp_path / "arrived")
        finally:
            srv.shutdown()

    def test_a_plan_never_prints_its_signatures(self):
        from experiments.durable_stores import PresignedFetchPlan

        plan = PresignedFetchPlan(
            "k", {"a": "https://x/?X-Amz-Signature=DEADBEEF"}, {"a": 1}, 60)
        for rendered in (repr(plan), str(plan), str(plan.redacted())):
            assert "DEADBEEF" not in rendered, (
                "a signature reached a rendered form; a plan grants read "
                "access without a credential and must never be logged, "
                "committed or written into evidence")
        assert plan.redacted()["n_objects"] == 1
