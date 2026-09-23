"""A durable object store for checkpoint directories, with no vendor in it.

**STATUS (2026-09-23): NO PRODUCTION CALLER.** This module was written for a
route that was superseded before it ever ran. The plan was to pre-stage a
campaign's completed checkpoints into a remote object store and have each
replacement resource fetch them with pre-signed links; every such store
reachable from this environment requires an account-creation and payment step
that only a maintainer can perform. The application layer now pre-stages those
bytes onto a block volume the provider already holds and each resource
attaches, which needs no credential anywhere and removes the fetch from the
billed session entirely.

It is kept because the contract it encodes -- that a transfer's identity cannot
be skipped -- is generic and outlives the backend question, and because it
names no provider to go stale. Do not read its presence as evidence that this
project has an object-store backend. It does not.

WHY THIS EXISTS. A completed unit of work has to survive the provider resource
that produced it, and a replacement resource has to be able to consume it. The
two legs have opposite constraints:

* the UPLOAD can be slow, because it runs on a dev box while no accelerator is
  billing. Hours there cost nothing;
* the DOWNLOAD must be fast, because it runs on the billed replacement
  resource. Hours there are the experiment's budget.

A transport that inverts this -- pushing multi-GiB objects from a dev box to a
machine that is already billing -- charges the slow leg at the accelerator's
rate. One campaign priced that at more than a fresh session costs.

WHAT THIS IS NOT. It is not a file transfer library and it does not move bytes
itself: a `DurableStore` implementation does that, and implementations live in
the application layer where a vendor, an endpoint and a credential belong.
Nothing here names a provider, a bucket, an account, a region, a protocol or
an experiment. The same interface must serve a local directory, an
object-storage endpoint, a hub repository or anything a later stage needs, for
any model family and any artifact size.

WHAT IT ADDS over a plain `put`/`get`. Identity. A restored checkpoint is only
a measurement if the bytes that ARRIVED re-identify to what was announced, and
that check belongs beside the transfer rather than in whatever calls it:

    upload   ->  identify the source, record it with the object
    download ->  re-identify from the ARRIVED bytes, refuse on mismatch

`identify_for_transfer` and `verify_transferred_leaf` do the identity work;
this module composes them with a transport so no caller can transfer without
checking, and no check can be satisfied by a manifest instead of by bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class DurableStoreError(RuntimeError):
    """A transfer did not happen, or happened and did not verify."""


@runtime_checkable
class DurableStore(Protocol):
    """What a durable backend must do. Four methods, no vendor.

    `key` is an opaque object name the caller chooses. Implementations must
    treat it as a path-like identifier and must not interpret its parts.
    """

    def put_tree(self, local_dir: Path, key: str) -> dict[str, Any]:
        """Upload a directory to an UNOCCUPIED key.

        MUST refuse a key that already holds anything. A scientific artifact's
        key is immutable: an upload that merges into a populated prefix leaves
        whatever the previous upload wrote and the new source does not, and a
        later restore then reassembles a directory that was never any one
        measurement. Overwriting silently is worse -- it destroys the only
        remaining copy of whatever was there.

        Returns `{bytes, n_files, seconds, ...}`.
        """

    def get_tree(self, key: str, local_dir: Path) -> dict[str, Any]:
        """Download into `local_dir`, which must not already exist."""

    def stat_tree(self, key: str) -> dict[str, Any] | None:
        """`{bytes, n_files}` if the object exists, else None."""

    def free_bytes(self) -> int | None:
        """Remaining capacity, or None when the backend does not report it."""


@dataclass(frozen=True)
class TransferRecord:
    """One checkpoint's journey, and whether its identity survived it."""

    key: str
    direction: str
    bytes: int
    n_files: int
    seconds: float
    identity: dict[str, Any]
    verified: bool
    store: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def mb_per_second(self) -> float:
        return (self.bytes / 1e6 / self.seconds) if self.seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "direction": self.direction,
                "bytes": self.bytes, "gib": round(self.bytes / 2**30, 3),
                "n_files": self.n_files, "seconds": round(self.seconds, 2),
                "mb_per_second": round(self.mb_per_second, 3),
                "identity": self.identity, "verified": self.verified,
                "store": self.store, **self.detail}


def upload_checkpoint(store: DurableStore, local_dir: str | Path, key: str, *,
                      adapter: Any, arch_signature: str,
                      num_parameters: int, verify: bool = True,
                      scratch: str | Path | None = None) -> TransferRecord:
    """Identify a checkpoint, upload it UNCHANGED, then verify what is STORED.

    Nothing here re-saves, re-serialises or converts: a transport that rewrites
    the bytes it carries is not transporting the same artifact.

    **A successful PUT is not durability.** `verified` used to mean "the source
    identity was computed and the upload call returned", which proves nothing
    about the bytes now in the backend. That distinction becomes load-bearing
    at exactly the moment it matters: the producer releases its only local copy
    on the strength of this record. So the identity is re-derived from what the
    BACKEND holds, by reading it back through the store's own `get_tree` into a
    scratch directory that is deleted afterwards.

    The readback is a round trip through the real transport, which is the
    property an acknowledgement should assert, and it reuses
    `verify_transferred_leaf` rather than adding a second identity
    construction. It is deliberately NOT an ETag comparison: a multipart
    upload's ETag is a digest of digests whose value depends on the part size
    the client chose, so it is not a cryptographic identity of the content and
    two correct backends can disagree about it.

    `verify=False` exists for a caller that will verify separately and wants
    the upload timing alone; it is not the default and it records itself.
    """
    import shutil
    import tempfile

    from aadistill.runtime.leaf_durability import identify_for_transfer

    src = Path(local_dir)
    if not src.is_dir():
        raise DurableStoreError(f"{src} is not a directory to upload")

    #: IMMUTABLE KEYS. An occupied key is refused rather than merged into or
    #: overwritten, unless what is already there is byte-identical to what is
    #: being uploaded -- which makes a re-upload idempotent instead of
    #: destructive, and is established by reading the stored bytes rather than
    #: by comparing sizes.
    existing = store.stat_tree(key)
    if existing:
        if not verify:
            raise DurableStoreError(
                f"{key!r} is already occupied ({existing}) and verify=False, "
                "so whether it already holds this exact artifact cannot be "
                "established. A scientific artifact's key is immutable.")
        same = _stored_identity_matches(
            store, key, adapter=adapter, arch_signature=arch_signature,
            num_parameters=num_parameters, source=src, scratch=scratch)
        if same["matched"]:
            return TransferRecord(
                key=key, direction="upload", bytes=int(existing["bytes"]),
                n_files=int(existing["n_files"]), seconds=0.0,
                identity=same["identity"], verified=True,
                store=type(store).__name__,
                detail={"already_present": True,
                        "verified_by": "readback of the stored bytes",
                        "_idempotent": (
                            "the key already held this exact artifact, "
                            "established by re-deriving the identity from the "
                            "backend's bytes. Nothing was uploaded and nothing "
                            "was overwritten.")})
        raise DurableStoreError(
            f"{key!r} is already occupied by a DIFFERENT artifact "
            f"({same.get('why')}). A scientific artifact's key is immutable: "
            "merging would let a stale file survive into a later restore, and "
            "overwriting would destroy whatever is there. Choose a fresh key.")
    ident = identify_for_transfer(
        src, adapter=adapter, arch_signature=arch_signature,
        num_parameters=num_parameters)
    #: The DICT a destination re-derives against, built the way every other
    #: consumer in this repository builds it: the record's own `as_dict` plus
    #: the derived digests that are properties rather than fields. Assembling
    #: a subset by hand here is how a sender comes to record `None` for a
    #: field the receiver computes.
    identity = {**ident.as_dict(),
                "artifact_digest": ident.artifact_digest,
                "weights_digest": ident.weights_digest,
                "arch_signature": arch_signature,
                "num_parameters": int(num_parameters)}
    out = store.put_tree(src, key)
    if not verify:
        return TransferRecord(
            key=key, direction="upload", bytes=int(out["bytes"]),
            n_files=int(out["n_files"]), seconds=float(out["seconds"]),
            identity=identity, verified=False,
            store=type(store).__name__,
            detail={"verified_by": None,
                    "_not_verified": (
                        "verify=False: the upload call returned and nothing "
                        "has read the stored bytes. This record must NOT be "
                        "treated as a durable acknowledgement, and the "
                        "producer's local copy must not be released on it.")})

    check = _stored_identity_matches(
        store, key, adapter=adapter, arch_signature=arch_signature,
        num_parameters=num_parameters, source=src, scratch=scratch)
    if not check["matched"]:
        raise DurableStoreError(
            f"{key!r} uploaded but the STORED bytes do not re-identify to what "
            f"was announced: {check.get('why')}. The object is left in place as "
            "evidence about the transport, and the producer's local copy must "
            "NOT be released.")
    return TransferRecord(
        key=key, direction="upload", bytes=int(out["bytes"]),
        n_files=int(out["n_files"]), seconds=float(out["seconds"]),
        identity=check["identity"], verified=True,
        store=type(store).__name__,
        detail={"verified_by": "readback of the stored bytes",
                "source_identity_matched": (
                    check["identity"].get("artifact_digest")
                    == identity.get("artifact_digest")),
                "_verified_means": (
                    "the identity was re-derived from the bytes the BACKEND "
                    "holds, by reading them back through the store. A PUT that "
                    "returned successfully proves nothing about what is stored, "
                    "and this record is what a producer releases its only local "
                    "copy on.")})


def _stored_identity_matches(store: DurableStore, key: str, *, adapter: Any,
                             arch_signature: str, num_parameters: int,
                             source: Path | None,
                             scratch: str | Path | None) -> dict[str, Any]:
    """Re-derive a stored object's identity by reading it back.

    Provider-neutral by construction: it uses only `get_tree`, so a backend
    that can be read can be verified, and no backend-specific metadata --
    an ETag, a checksum header, a vendor field -- is trusted or required.

    The scratch copy is transient and is always removed, including on failure.
    It is not a second durable copy; it exists for as long as it takes to hash.
    """
    import shutil
    import tempfile

    from aadistill.runtime.leaf_durability import (identify_for_transfer,
                                                   verify_transferred_leaf)

    base = Path(scratch) if scratch else Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=base))
    back = tmp / "readback"
    try:
        store.get_tree(key, back)
        ident = identify_for_transfer(
            back, adapter=adapter, arch_signature=arch_signature,
            num_parameters=num_parameters)
        identity = {**ident.as_dict(),
                    "artifact_digest": ident.artifact_digest,
                    "weights_digest": ident.weights_digest,
                    "arch_signature": arch_signature,
                    "num_parameters": int(num_parameters)}
        if source is None:
            return {"matched": True, "identity": identity}
        v = verify_transferred_leaf(
            back, {**identify_for_transfer(
                source, adapter=adapter, arch_signature=arch_signature,
                num_parameters=num_parameters).as_dict(),
                "arch_signature": arch_signature,
                "num_parameters": int(num_parameters)},
            adapter=adapter)
        return {"matched": bool(v.get("matched")), "identity": identity,
                "why": v}
    except Exception as exc:                                    # noqa: BLE001
        return {"matched": False, "identity": {},
                "why": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def restore_checkpoint(store: DurableStore, key: str, local_dir: str | Path, *,
                       adapter: Any, identity: dict[str, Any]
                       ) -> TransferRecord:
    """Download to a NEW path, then re-identify from the ARRIVED bytes.

    Refuses on mismatch, and refuses rather than deleting: a checkpoint that
    arrived wrong is evidence about the transport and the caller decides what
    to do with it.

    `local_dir` must not exist. Restoring over an existing directory could
    leave a mixture of two transfers whose identity is neither one's.
    """
    from aadistill.runtime.leaf_durability import verify_transferred_leaf

    dest = Path(local_dir)
    if dest.exists():
        raise DurableStoreError(
            f"{dest} already exists; a restore into an occupied path could "
            "leave a mixture of two transfers whose identity is neither's")
    out = store.get_tree(key, dest)
    v = verify_transferred_leaf(dest, identity, adapter=adapter)
    rec = TransferRecord(
        key=key, direction="download", bytes=int(out["bytes"]),
        n_files=int(out["n_files"]), seconds=float(out["seconds"]),
        identity=dict(identity), verified=bool(v.get("matched")),
        store=type(store).__name__, detail={"verification": v})
    if not rec.verified:
        raise DurableStoreError(
            f"{key} arrived at {dest} and did NOT re-identify to what was "
            f"announced: {v}. The bytes are left in place as evidence about "
            "the transport; nothing may consume them as a measurement.")
    return rec
