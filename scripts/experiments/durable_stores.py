"""Concrete durable-store backends. The APPLICATION layer, where vendors live.

`aadistill.runtime.durable_store` defines what a backend must do and composes
it with identity checking. It names no provider, because a provider is an
operational choice that changes between experiments and stages. These are the
choices this repository currently has.

* `LocalDirStore` -- a directory on a filesystem. The dev box's canonical
  artifact store is one, so this is not a test double: it is the backend the
  campaign's ten completed probes already live in, and it makes the whole
  transfer path executable at `$0`.
* `S3CompatibleStore` -- any endpoint that speaks the S3 API. It takes the
  endpoint, bucket and credentials as ARGUMENTS; it does not know whether it
  is talking to one vendor or another, and adding a third changes nothing here.

Credentials are read from the environment, never from a constant, and never
logged.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any


class LocalDirStore:
    """A directory tree under `root`. Real storage, not a stub.

    `free_bytes` reports the filesystem's, which is what a capacity gate has
    to check: "a durability mechanism needs a backend with the capacity to
    hold what it is protecting" is only checkable if the backend answers.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _at(self, key: str) -> Path:
        #: Keys are opaque and must not escape the root. A key containing `..`
        #: would write outside the store, which is the one interpretation of a
        #: key this class is allowed to make.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"key {key!r} escapes the store root")
        return p

    def put_tree(self, local_dir: Path, key: str) -> dict[str, Any]:
        dest = self._at(key)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        shutil.copytree(local_dir, dest)
        dt = time.time() - t0
        files = [f for f in dest.rglob("*") if f.is_file()]
        return {"bytes": sum(f.stat().st_size for f in files),
                "n_files": len(files), "seconds": dt, "at": str(dest)}

    def get_tree(self, key: str, local_dir: Path) -> dict[str, Any]:
        src = self._at(key)
        if not src.is_dir():
            raise FileNotFoundError(f"no object at {key!r} in {self.root}")
        t0 = time.time()
        shutil.copytree(src, local_dir)
        dt = time.time() - t0
        files = [f for f in Path(local_dir).rglob("*") if f.is_file()]
        return {"bytes": sum(f.stat().st_size for f in files),
                "n_files": len(files), "seconds": dt, "from": str(src)}

    def stat_tree(self, key: str) -> dict[str, Any] | None:
        p = self._at(key)
        if not p.is_dir():
            return None
        files = [f for f in p.rglob("*") if f.is_file()]
        return {"bytes": sum(f.stat().st_size for f in files),
                "n_files": len(files)}

    def free_bytes(self) -> int | None:
        self.root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.root).free


class S3CompatibleStore:
    """Any endpoint speaking the S3 API. The endpoint is an argument.

    Deliberately ignorant of which service is behind it: one vendor's
    object storage, another's, a self-hosted one, or a local emulator are the
    same object to this class. That is the property that keeps a vendor out of
    the generic interface AND out of this file's logic -- only its
    construction site knows.

    Credentials come from the environment by the names the CALLER gives, so a
    second backend with different keys needs no change here and no key is ever
    written down.

    `boto3` is imported lazily: the dev box does not need it installed to
    price a continuation, and a pod that does not use this backend should not
    be made to carry it.
    """

    def __init__(self, *, endpoint_url: str, bucket: str,
                 access_key_env: str, secret_key_env: str,
                 region: str = "auto") -> None:
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.region = region
        self._ak_env, self._sk_env = access_key_env, secret_key_env

    def _client(self):
        #: CREDENTIALS FIRST. Both checks refuse, but this one is free and it
        #: is the more useful answer: "the key is not set" is actionable where
        #: "a library is missing" sends someone to install a dependency they
        #: may not need yet. Checking the import first hid the credential
        #: message behind it.
        ak, sk = os.environ.get(self._ak_env), os.environ.get(self._sk_env)
        if not ak or not sk:
            raise RuntimeError(
                f"the object-storage credentials are not in the environment "
                f"({self._ak_env}, {self._sk_env}). They are never read from a "
                "file in the repository and never defaulted.")
        try:
            import boto3
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError(
                "an S3-API client library is needed for an object-storage "
                "backend and is not installed. It is a lazy import on "
                "purpose: nothing else in this repository requires it, and a "
                "backend nobody has configured should not add a dependency "
                "to every pod.") from exc
        return boto3.client(
            "s3", endpoint_url=self.endpoint_url, region_name=self.region,
            aws_access_key_id=ak, aws_secret_access_key=sk)

    def put_tree(self, local_dir: Path, key: str) -> dict[str, Any]:
        c = self._client()
        files = sorted(f for f in Path(local_dir).rglob("*") if f.is_file())
        t0, total = time.time(), 0
        for f in files:
            rel = f.relative_to(local_dir).as_posix()
            #: Uploaded as-is. No re-serialisation, no dtype conversion, no
            #: recompression: the object at the far end must be the same
            #: scientific artifact, merely transported.
            c.upload_file(str(f), self.bucket, f"{key}/{rel}")
            total += f.stat().st_size
        return {"bytes": total, "n_files": len(files),
                "seconds": time.time() - t0, "at": f"{self.bucket}/{key}"}

    def get_tree(self, key: str, local_dir: Path) -> dict[str, Any]:
        c = self._client()
        dest = Path(local_dir)
        dest.mkdir(parents=True, exist_ok=False)
        t0, total, n = time.time(), 0, 0
        paginator = c.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{key}/"):
            for obj in page.get("Contents") or []:
                rel = obj["Key"][len(key) + 1:]
                if not rel:
                    continue
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                c.download_file(self.bucket, obj["Key"], str(out))
                total += int(obj["Size"])
                n += 1
        if n == 0:
            raise FileNotFoundError(
                f"no object under {key!r} in {self.bucket}")
        return {"bytes": total, "n_files": n, "seconds": time.time() - t0,
                "from": f"{self.bucket}/{key}"}

    def stat_tree(self, key: str) -> dict[str, Any] | None:
        c = self._client()
        total, n = 0, 0
        paginator = c.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{key}/"):
            for obj in page.get("Contents") or []:
                total += int(obj["Size"])
                n += 1
        return {"bytes": total, "n_files": n} if n else None

    def free_bytes(self) -> int | None:
        #: Object storage does not report remaining capacity, and guessing one
        #: would make a capacity gate green about a number nobody measured.
        #: A quota-bearing backend's headroom is asked of the backend by
        #: whoever configured the quota.
        return None
