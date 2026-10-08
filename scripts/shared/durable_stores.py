"""Concrete durable-store backends. The APPLICATION layer, where vendors live.

**STATUS (2026-09-23): NO PRODUCTION CALLER.** See
`aadistill.runtime.durable_store`. The S3-compatible route these backends were
written for needs an account this environment cannot create, and the C2
behavioural campaign pre-stages its probes onto an attached provider network
volume instead. Nothing in the launch path imports this module.

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


#: How long a fetch plan's signatures stay valid. Long enough for a pod to be
#: created, set up and restore ten probes; short enough that a leaked plan
#: stops working the same day. It is a URL lifetime, not a session budget.
PRESIGN_TTL_SECONDS = 12 * 3600


def safe_join(root: Path, rel: str) -> Path:
    """`root / rel`, refusing anything that resolves outside `root`.

    A remote object's key suffix becomes a local path on restore, so a key
    containing `..` -- or an absolute one -- would write outside the
    destination. Our own uploader is the only writer to the buckets in use, so
    this is low-risk today; it is here because a restore path is exactly the
    place a future stage would inherit the assumption without noticing, and
    because the same store is meant to serve Stage 0 through 6 and beyond.

    Component semantics, not a string prefix: a sibling directory whose name
    merely begins with the root's is outside it.
    """
    base = Path(root).resolve()
    out = (base / rel).resolve()
    if out != base and not out.is_relative_to(base):
        raise ValueError(
            f"object suffix {rel!r} resolves outside the restore root {base}")
    return out


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
        #:
        #: `str.startswith` is NOT a containment predicate and this used it:
        #: with a root of `/tmp/backend`, the sibling `/tmp/backend_evil/x`
        #: has the root as a string prefix and passed. A resolved
        #: `is_relative_to` compares path COMPONENTS, so a sibling whose name
        #: merely begins with the root's is outside it.
        try:
            return safe_join(self.root, key)
        except ValueError:
            raise ValueError(f"key {key!r} escapes the store root") from None

    def put_tree(self, local_dir: Path, key: str) -> dict[str, Any]:
        dest = self._at(key)
        #: AN OCCUPIED KEY IS REFUSED, never emptied. This called
        #: `shutil.rmtree(dest)` first, which destroys whatever is there --
        #: possibly the only remaining copy of a completed measurement -- to
        #: make room for bytes nobody has verified yet. A scientific
        #: artifact's key is immutable; the generic layer decides whether an
        #: existing object is the same artifact, and it does so by reading it.
        if dest.exists():
            raise FileExistsError(
                f"{key!r} is already occupied at {dest}. A scientific "
                "artifact's key is immutable: this store will not overwrite "
                "or merge into it.")
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
        #: AN OCCUPIED PREFIX IS REFUSED. Writing each file under an existing
        #: prefix is a directory SYNC, not an upload: anything the previous
        #: upload wrote and this source does not still sits there, and a later
        #: restore reassembles a directory that was never any one measurement.
        #: Object stores have no directories to replace, so the refusal has to
        #: be explicit here.
        if self.stat_tree(key):
            raise FileExistsError(
                f"{key!r} is already occupied in {self.bucket}. A scientific "
                "artifact's key is immutable: this store will not overwrite "
                "or merge into it, because a stale object left under the "
                "prefix would survive into a later restore.")
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
                out = safe_join(dest, rel)
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

    def presign_fetch_plan(self, key: str, *,
                           expires_in: int = PRESIGN_TTL_SECONDS
                           ) -> "PresignedFetchPlan":
        """Sign a time-limited GET per object under `key`.

        The credential stays here. What crosses to the machine that downloads
        is a set of URLs that expire, which is both less to leak and less to
        install: the fetching side needs no client library at all.
        """
        c = self._client()
        urls, sizes = {}, {}
        paginator = c.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{key}/"):
            for obj in page.get("Contents") or []:
                rel = obj["Key"][len(key) + 1:]
                if not rel:
                    continue
                urls[rel] = c.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": obj["Key"]},
                    ExpiresIn=int(expires_in))
                sizes[rel] = int(obj["Size"])
        if not urls:
            raise FileNotFoundError(f"no object under {key!r} in {self.bucket}")
        return PresignedFetchPlan(key, urls, sizes, expires_in)

    def free_bytes(self) -> int | None:
        #: Object storage does not report remaining capacity, and guessing one
        #: would make a capacity gate green about a number nobody measured.
        #: A quota-bearing backend's headroom is asked of the backend by
        #: whoever configured the quota.
        return None


class PresignedFetchPlan:
    """Signed, time-limited GET URLs for one object, and nothing else.

    WHY THIS EXISTS. The pod is the side that downloads, and giving it the
    bucket credentials would put a long-lived secret on a machine the project
    does not own, in an environment whose logs are collected as evidence. It
    would also make every pod carry an S3 client library it needs for one
    fetch.

    A plan is the alternative: the dev box, which already holds the
    credentials, signs a URL per object; the pod fetches them over plain HTTP
    and holds no secret that outlives the plan.

    **A plan IS a secret while it is valid.** Each URL carries a signature that
    grants read access without any further credential, so a plan must never be
    committed, logged, or written into evidence. `redacted()` is what a record
    may contain.
    """

    def __init__(self, key: str, urls: dict[str, str],
                 sizes: dict[str, int], expires_in: int) -> None:
        self.key, self._urls = key, dict(urls)
        self.sizes, self.expires_in = dict(sizes), int(expires_in)

    def __len__(self) -> int:
        return len(self._urls)

    def items(self):
        return self._urls.items()

    def redacted(self) -> dict[str, Any]:
        """What may appear in a record: shape and sizes, never a signature."""
        return {"key": self.key, "n_objects": len(self._urls),
                "bytes": sum(self.sizes.values()),
                "relative_paths": sorted(self._urls),
                "expires_in_seconds": self.expires_in,
                "_urls_are_omitted": (
                    "each URL carries a signature granting read access without "
                    "a credential. A plan is a secret while it is valid and is "
                    "never committed, logged or written into evidence.")}

    def __repr__(self) -> str:          # pragma: no cover - defensive
        return f"<PresignedFetchPlan {self.key} n={len(self._urls)} REDACTED>"

    __str__ = __repr__


class PresignedHttpStore:
    """A read-only backend that fetches a `PresignedFetchPlan` over HTTP.

    It satisfies the same `DurableStore` shape as any other, so the pod runs
    the SAME `restore_checkpoint` -- identity re-derived from the bytes that
    arrived -- with no branch for how they got there. It cannot upload, and
    says so rather than pretending.
    """

    def __init__(self, plan: PresignedFetchPlan) -> None:
        self.plan = plan

    def put_tree(self, local_dir: Path, key: str) -> dict[str, Any]:
        raise NotImplementedError(
            "a presigned fetch plan grants READ access only. The side that "
            "uploads is the side that holds the credentials, and that is "
            "deliberately not this one.")

    def get_tree(self, key: str, local_dir: Path) -> dict[str, Any]:
        import urllib.request

        if key != self.plan.key:
            raise KeyError(
                f"this plan is for {self.plan.key!r}, not {key!r}; a plan "
                "covers exactly one object")
        dest = Path(local_dir)
        dest.mkdir(parents=True, exist_ok=False)
        t0, total = time.time(), 0
        for rel, url in sorted(self.plan.items()):
            out = safe_join(dest, rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=600) as r, \
                    open(out, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            total += out.stat().st_size
        return {"bytes": total, "n_files": len(self.plan),
                "seconds": time.time() - t0, "from": "presigned plan"}

    def stat_tree(self, key: str) -> dict[str, Any] | None:
        if key != self.plan.key or not len(self.plan):
            return None
        return {"bytes": sum(self.plan.sizes.values()),
                "n_files": len(self.plan)}

    def free_bytes(self) -> int | None:
        return None
