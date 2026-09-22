"""A durable object store for checkpoint directories, with no vendor in it.

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
        """Upload a directory. Returns `{bytes, n_files, seconds, ...}`."""

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
                      num_parameters: int) -> TransferRecord:
    """Identify a checkpoint, then upload it UNCHANGED.

    The identity is computed BEFORE the transfer and travels with the record,
    because that is what the destination has to re-derive. Nothing here
    re-saves, re-serialises or converts: a transport that rewrites the bytes it
    carries is not transporting the same artifact, and the whole point is that
    the object at the far end is the same scientific product.
    """
    from aadistill.runtime.leaf_durability import identify_for_transfer

    src = Path(local_dir)
    if not src.is_dir():
        raise DurableStoreError(f"{src} is not a directory to upload")
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
    return TransferRecord(
        key=key, direction="upload", bytes=int(out["bytes"]),
        n_files=int(out["n_files"]), seconds=float(out["seconds"]),
        identity=identity, verified=True,
        store=type(store).__name__,
        detail={"_verified_means": (
            "the identity was computed from the SOURCE bytes and recorded "
            "with the object. Whether the arrival matches is the download's "
            "question, and it is asked there.")})


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
