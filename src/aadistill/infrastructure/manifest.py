"""Hashing and JSON manifest helpers shared by all pipeline stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    """Hash of a JSON-serializable object, independent of key order."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write a whole file, or leave the previous one untouched.

    `Path.write_text` truncates first and writes second. When the filesystem
    filled on 2026-09-11 that gap became two tracked files of zero bytes: the
    writer opened them, emptied them, and then could not write. Both were
    recoverable from git, which is luck rather than design — the same sequence
    on a gitignored artifact loses it.

    So: write a sibling temp file, flush it to the platter, then `os.replace`,
    which is atomic within a filesystem. A full disk now fails on the temp file
    and the original is still there. The temp is cleaned up on failure so a
    crashed write does not leave litter next to the thing it was protecting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_manifest(path: str | Path, manifest: dict) -> None:
    write_text_atomic(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
