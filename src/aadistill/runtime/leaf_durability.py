"""Make a successful Stage-1 selection survive a failing Stage 2.

A paid session spent **180.3 minutes** producing valid, measured, selected
leaves — and then lost every one of their checkpoints, because
persistence happens only after Stage-5 selection and Stage 2 failed six seconds
after Stage 1 passed. The search *record* came home; the weights did not. The
science that survived is a ranking whose artifacts no longer exist, and
regenerating them costs another full search.

This is the durability boundary that closes that failure class: the five
selected leaves are persisted **after selection and before Stage 2**, so the
expensive, already-succeeded half of the session cannot be destroyed by the
cheap half that follows it.

Three properties, each learned from a specific failure:

* **Weight-only, byte-identical.** A searched leaf is a model artifact and
  carries no tokenizer files by design. `CheckpointIdentity.artifact_digest`
  folds in `tokenizer_sha256` when one is present, so adding tokenizer files
  here would change the identity the search metrics are attached to. The
  tokenizer is a separate consumer dependency — see
  `aadistill.models.tokenizer_contract`.
* **Verified after transfer, not before.** The digest is recomputed from the
  destination and required to equal the one the search recorded. A copy that
  silently truncated is a copy that passed every check made only at the source.
* **Headroom is measured, and refusal is the answer.** In that session the
  leaves were 5.55 GiB against ~1.03 GiB free on the relay and 3.4 GiB on the
  dev box. A
  persistence step that discovers this halfway through has already destroyed the
  thing it was protecting, so the space is checked before the first byte moves.

Only the selected leaves are persisted. The search produced 43 states; the other
38 are recorded and intentionally not preserved.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aadistill.initialization.specs.artifact import (
    _tokenizer_digest,
    identify_checkpoint,
)

__all__ = ["LeafDurabilityError", "free_bytes_at", "identify_for_transfer",
           "persist_selected_leaves", "verify_transferred_leaf"]


class LeafDurabilityError(RuntimeError):
    """A selected leaf could not be durably preserved. Stage 2 must not start."""


def identify_for_transfer(directory: str | Path, *, adapter: Any,
                          arch_signature: str, num_parameters: int):
    """Identify a checkpoint directory the way a transfer will be checked.

    `identify_checkpoint` takes an `ArchSpec` and reads `spec.spec_hash`, which
    suits a checkpoint a search just constructed. A *trained* checkpoint has no
    spec of its own — training changes weights, not architecture — so its
    `arch_signature` and `num_parameters` come from the initialization it was
    trained from, and are passed in.

    This exists so the SOURCE and the DESTINATION compute identity by one
    construction rather than two. `artifact_digest` covers `tokenizer_sha256`,
    so a sender that recorded `None` while the receiver hashed the tokenizer
    files that arrived would produce a mismatch on every transfer of a
    checkpoint carrying a tokenizer — a disagreement about bookkeeping,
    reported as corruption. `verify_transferred_leaf` calls this too.
    """
    from aadistill.initialization.specs.artifact import (
        CheckpointIdentity, ShardRecord,
    )
    from aadistill.infrastructure.manifest import sha256_file, sha256_json

    import json as _json

    d = Path(directory)
    if not d.is_dir():
        raise LeafDurabilityError(f"{d}: not a checkpoint directory")
    names = sorted(adapter.weight_files(str(d)))
    if not names:
        raise LeafDurabilityError(f"{d}: no weight shards")
    config = d / "config.json"
    if not config.is_file():
        raise LeafDurabilityError(f"{d}: config.json is missing")
    index_name = adapter.index_file(str(d))
    return CheckpointIdentity(
        path=str(d),
        shards=tuple(ShardRecord(n, sha256_file(d / n), (d / n).stat().st_size)
                     for n in names),
        config_sha256=sha256_json(_json.loads(config.read_text())),
        arch_signature=arch_signature,
        num_parameters=int(num_parameters),
        index_sha256=sha256_file(d / index_name) if index_name else None,
        tokenizer_sha256=_tokenizer_digest(d))


def free_bytes_at(destination: str | Path) -> int:
    """Free space at the nearest existing ancestor of `destination`."""
    d = Path(destination)
    while not d.exists() and d != d.parent:
        d = d.parent
    return shutil.disk_usage(d).free


def persist_selected_leaves(
    *,
    leaves: Sequence[Mapping[str, Any]],
    destination: str | Path,
    adapter: Any,
    spec: Any,
    margin_bytes: int = 512 * 2**20,
    free_bytes: Callable[[str | Path], int] = free_bytes_at,
    copier: Callable[[Path, Path], None] | None = None,
) -> dict:
    """Copy each selected leaf to `destination` and re-verify its identity.

    `leaves` supplies, per leaf, `state_id`, `checkpoint_path`, `num_parameters`,
    the `artifact_digest` the search recorded and `total_bytes`. Raises
    `LeafDurabilityError` — before moving anything — when the destination cannot
    hold them, and after each copy when the recomputed digest disagrees.
    """
    if not leaves:
        raise LeafDurabilityError(
            "no selected leaves were handed to the durability boundary. Stage 1 "
            "selects before Stage 2 starts; an empty set here means the "
            "selection did not happen and Stage 2 must not proceed.")

    dest = Path(destination)
    required = sum(int(leaf["total_bytes"]) for leaf in leaves)
    available = free_bytes(dest)
    if available < required + margin_bytes:
        raise LeafDurabilityError(
            f"cannot durably preserve {len(leaves)} selected leaves: they need "
            f"{required / 2**30:.2f} GiB (plus a {margin_bytes / 2**30:.2f} GiB "
            f"margin) and {dest} has {available / 2**30:.2f} GiB free. Refusing "
            "before the first byte moves — a persistence step that runs out of "
            "space halfway has already destroyed what it was protecting. Free "
            "space or choose a destination; do not start Stage 2.")

    dest.mkdir(parents=True, exist_ok=True)
    copy = copier or (lambda s, d: shutil.copytree(s, d, dirs_exist_ok=True))
    persisted = []
    for leaf in leaves:
        state_id = leaf["state_id"]
        source = Path(leaf["checkpoint_path"])
        recorded = leaf["artifact_digest"]
        if not source.is_dir():
            raise LeafDurabilityError(
                f"{state_id}: the measured checkpoint {source} is not on disk, "
                "so the selection cannot be preserved at all")
        target = dest / state_id
        copy(source, target)

        identity = identify_checkpoint(
            target, adapter=adapter, spec=spec,
            num_parameters=int(leaf["num_parameters"]))
        if identity.artifact_digest != recorded:
            raise LeafDurabilityError(
                f"{state_id}: after transfer the checkpoint identifies as "
                f"{identity.artifact_digest} but the search recorded {recorded}. "
                "The persisted copy is not the artifact the metrics were "
                "measured on; refusing to continue.")
        if identity.tokenizer_sha256 is not None:
            raise LeafDurabilityError(
                f"{state_id}: the persisted copy carries tokenizer files. A "
                "searched leaf is weight-only by design, and adding them moves "
                "the artifact digest the search metrics hang on. The tokenizer "
                "is a separate consumer dependency.")
        persisted.append({
            "state_id": state_id,
            #: The whole identity, so a verifier on another machine can rebuild
            #: `artifact_digest` from LOCAL bytes plus the two fields no file
            #: carries -- `arch_signature` and `num_parameters`.
            "identity": identity.as_dict(),
            "weights_digest": identity.weights_digest,
            "config_sha256": identity.config_sha256,
            "arch_signature": identity.arch_signature,
            "num_parameters": identity.num_parameters,
            "source": str(source),
            "path": str(target),
            "artifact_digest": identity.artifact_digest,
            "recorded_digest": recorded,
            "single_shard_sha256": identity.single_shard_sha256,
            "total_bytes": identity.total_bytes,
            "tokenizer_sha256": identity.tokenizer_sha256,
        })

    return {
        "schema": "aadistill.autoinit.selected_leaf_durability/v1",
        "destination": str(dest),
        "n_leaves": len(persisted),
        "required_bytes": required,
        "free_bytes_before": available,
        "leaves": persisted,
        "note": ("persisted after Stage-1 selection and before Stage 2, so a "
                 "Stage-2 failure cannot destroy a search that already "
                 "succeeded. Weight-only: no tokenizer files are added."),
    }


def verify_transferred_leaf(directory: str | Path, record: Mapping[str, Any], *,
                            adapter: Any) -> dict:
    """Re-identify a leaf that has landed on another machine.

    Staging a leaf on the pod is not durability; it is a copy that dies with the
    pod. This is the other half — run **after** the transfer, on the destination,
    rebuilding the identity from the bytes that actually arrived.

    `arch_signature` and `num_parameters` are taken from the record because no
    file carries them: they come from the adapter's `ArchSpec`, which lives where
    the search ran. **Everything else is recomputed from local bytes**, so a
    transfer that truncated a shard or mangled a config is caught here rather
    than assumed away.
    """
    d = Path(directory)
    if not d.is_dir():
        raise LeafDurabilityError(f"{d}: nothing arrived")

    #: ONE construction, shared with the sender. Re-deriving the identity here
    #: would be a second implementation of the thing whose sameness is the
    #: entire point of the comparison below.
    try:
        local = identify_for_transfer(
            d, adapter=adapter, arch_signature=record["arch_signature"],
            num_parameters=record["num_parameters"])
    except LeafDurabilityError as exc:
        raise LeafDurabilityError(f"{d}: {exc}; the transfer is incomplete") from exc

    recorded = record["artifact_digest"]
    return {
        "path": str(d),
        "artifact_digest": local.artifact_digest,
        "recorded_digest": recorded,
        "matched": local.artifact_digest == recorded,
        "weights_digest": local.weights_digest,
        "weights_digest_matched": local.weights_digest == record.get("weights_digest"),
        "config_sha256": local.config_sha256,
        #: Three-valued on purpose: `None` when the record does not carry a
        #: config hash at all. "not recorded" and "did not match" are different
        #: findings, and older records predate this field.
        #:
        #: Read with `.get`, never by subscript, and the distinction is load
        #: bearing rather than stylistic: a caller derives this function's
        #: REQUIRED record fields by scanning its own source for subscripted
        #: record reads, and asserts that whoever writes the record declares
        #: every one of them. A subscript here would announce a requirement no
        #: writer satisfies and none has reason to — this field is optional by
        #: design. (That scan reads COMMENTS too, so this note spells out no
        #: subscripted field name of its own.)
        "config_matched": (None if record.get("config_sha256") is None
                           else local.config_sha256 == record.get("config_sha256")),
        "arch_signature": local.arch_signature,
        "num_parameters": local.num_parameters,
        "single_shard_sha256": local.single_shard_sha256,
        "shard_matched": local.single_shard_sha256 == record.get("single_shard_sha256"),
        "tokenizer_sha256": local.tokenizer_sha256,
        "total_bytes": local.total_bytes,
    }
