"""Checkpoint identity that survives sharding.

``model.safetensors`` is not a checkpoint; it is one possible file layout for one.
A depth-only 4B intermediate already sits near the default shard threshold in some
supported Transformers versions, and a 30B-class teacher will always be sharded.
An identity pinned to a single filename would silently hash the wrong thing — or
nothing — exactly when the models get large enough for it to matter.

So a checkpoint's identity is an **artifact digest** over everything that decides
what the weights are and how a runtime will read them:

* every weight shard, sorted deterministically, each with its own sha256 and size;
* the shard index file when one exists (it maps parameter names to shards, so two
  identical shard sets under different indices are different checkpoints);
* the config;
* the architecture signature (the ``ArchSpec`` hash);
* the tokenizer, when the checkpoint carries one.

The per-shard hashes are kept individually, not just folded into the aggregate,
because this project's frozen record names single-file hashes — ``86fbba78…`` is
the sha256 of the canonical Stage-1 ``model.safetensors``. ``single_shard_sha256``
returns exactly that number for an unsharded checkpoint, so a historical hash
stays checkable while the aggregate digest is what metrics bind to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_file, sha256_json
from .arch import ArchSpec

SCHEMA = "aadistill.autoinit.checkpoint_identity/v1"


class ArtifactError(RuntimeError):
    """A checkpoint directory is not a usable artifact."""


@dataclass(frozen=True)
class ShardRecord:
    filename: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"filename": self.filename, "sha256": self.sha256,
                "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class CheckpointIdentity:
    """What identifies a materialized checkpoint, sharded or not."""

    path: str
    shards: tuple[ShardRecord, ...]
    config_sha256: str
    arch_signature: str
    num_parameters: int
    index_sha256: str | None = None
    tokenizer_sha256: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.shards:
            raise ArtifactError(f"{self.path}: no weight shards found")
        names = [s.filename for s in self.shards]
        if names != sorted(names):
            raise ArtifactError(
                f"{self.path}: shards must be in deterministic sorted order, got {names}")
        if len(set(names)) != len(names):
            raise ArtifactError(f"{self.path}: duplicate shard filenames {names}")

    @property
    def weights_digest(self) -> str:
        """Aggregate over the sorted shard manifest.

        Hashes ``filename:sha256`` lines rather than concatenated bytes, so it is
        computable without re-reading gigabytes and is stable under a rename only
        if the rename is meaningless — which for a shard index it is not.
        """
        payload = "".join(f"{s.filename}:{s.sha256}\n" for s in self.shards)
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def artifact_digest(self) -> str:
        """The identity metrics bind to: weights **and** how they will be read."""
        return sha256_json({
            "schema": SCHEMA,
            "weights_digest": self.weights_digest,
            "index_sha256": self.index_sha256,
            "config_sha256": self.config_sha256,
            "arch_signature": self.arch_signature,
            "tokenizer_sha256": self.tokenizer_sha256,
        })

    @property
    def single_shard_sha256(self) -> str | None:
        """The lone shard's own sha256, or None when sharded.

        Exists so a frozen single-file hash from the historical record — the
        canonical Stage-1 init's ``86fbba78…`` — remains directly checkable.
        """
        return self.shards[0].sha256 if len(self.shards) == 1 else None

    @property
    def total_bytes(self) -> int:
        return sum(s.size_bytes for s in self.shards)

    @property
    def is_sharded(self) -> bool:
        return len(self.shards) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "path": self.path,
            "shards": [s.as_dict() for s in self.shards],
            "n_shards": len(self.shards),
            "total_bytes": self.total_bytes,
            "weights_digest": self.weights_digest,
            "artifact_digest": self.artifact_digest,
            "single_shard_sha256": self.single_shard_sha256,
            "index_sha256": self.index_sha256,
            "config_sha256": self.config_sha256,
            "arch_signature": self.arch_signature,
            "tokenizer_sha256": self.tokenizer_sha256,
            "num_parameters": self.num_parameters,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> CheckpointIdentity:
        return cls(
            path=record["path"],
            shards=tuple(ShardRecord(s["filename"], s["sha256"], s["size_bytes"])
                         for s in record["shards"]),
            config_sha256=record["config_sha256"],
            arch_signature=record["arch_signature"],
            num_parameters=record["num_parameters"],
            index_sha256=record.get("index_sha256"),
            tokenizer_sha256=record.get("tokenizer_sha256"),
            extra=record.get("extra", {}))


def _tokenizer_digest(directory: Path) -> str | None:
    """Hash of the tokenizer files a checkpoint carries, if any.

    Included because a checkpoint whose tokenizer differs is a different artifact
    even at identical weights — the project has already been bitten once by a
    config field (`rope_parameters`) changing what a runtime does with unchanged
    weights.
    """
    names = sorted(n for n in ("tokenizer.json", "tokenizer_config.json",
                               "vocab.json", "merges.txt", "special_tokens_map.json")
                   if (directory / n).is_file())
    if not names:
        return None
    payload = "".join(f"{n}:{sha256_file(directory / n)}\n" for n in names)
    return hashlib.sha256(payload.encode()).hexdigest()


def identify_checkpoint(directory: str | Path, *, adapter, spec: ArchSpec,
                        num_parameters: int) -> CheckpointIdentity:
    """Build the identity of a checkpoint on disk.

    Shard discovery goes through the adapter, because which files hold weights is
    a runtime/format question, not a search question.
    """
    d = Path(directory)
    if not d.is_dir():
        raise ArtifactError(f"{d} is not a checkpoint directory")

    filenames = list(adapter.weight_files(str(d)))
    if not filenames:
        raise ArtifactError(
            f"{d}: the {adapter.family} adapter found no weight shards. A "
            "checkpoint that wrote nothing must not be measured.")
    shards = tuple(
        ShardRecord(name, sha256_file(d / name), (d / name).stat().st_size)
        for name in sorted(filenames))

    config_path = d / "config.json"
    if not config_path.is_file():
        raise ArtifactError(f"{d}: config.json is missing")
    config_sha = sha256_json(json.loads(config_path.read_text()))

    index_name = adapter.index_file(str(d))
    index_sha = sha256_file(d / index_name) if index_name else None

    return CheckpointIdentity(
        path=str(d), shards=shards, config_sha256=config_sha,
        arch_signature=spec.spec_hash, num_parameters=num_parameters,
        index_sha256=index_sha, tokenizer_sha256=_tokenizer_digest(d))


def verify_frozen_single_file_hash(identity: CheckpointIdentity,
                                   expected_sha256: str) -> bool:
    """Check a historical single-file hash against a (possibly sharded) artifact.

    Returns False rather than raising when the artifact is sharded: a sharded
    rebuild of a checkpoint that was frozen unsharded is not a hash mismatch, it
    is a different file layout, and conflating the two would make a legitimate
    re-materialization look like corruption.
    """
    return identity.single_shard_sha256 == expected_sha256


def shard_summary(identities: Sequence[CheckpointIdentity]) -> dict[str, Any]:
    return {
        "n_artifacts": len(identities),
        "n_sharded": sum(1 for i in identities if i.is_sharded),
        "max_shards": max((len(i.shards) for i in identities), default=0),
        "total_bytes": sum(i.total_bytes for i in identities),
    }
