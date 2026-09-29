"""Ask Hugging Face whether an upload of a given size would be accepted. $0.

**Why this exists.** AGENTS.md P8.2.1 requires confirming that a durable
large-artifact backend has ROOM before a long run that will produce large
completed artifacts. A durability mechanism can be perfectly correct — never
raising, disturbing no stage, recording each unit's identity, its inputs' hashes
and a content hash together with the exact reason it could not write — and still
preserve nothing, because the backend is full. The only defence is to ask
beforehand, which is what this does.

Callers own the sizes and the decision. A stage that knows how many artifacts it
will produce, and how large each one is, asks here and refuses its own launch;
this module states no experiment, no artifact count and no threshold.

**How.** The git-LFS batch endpoint performs its quota check *before* any bytes
move, so asking costs seconds and stores nothing. A `200` carrying an `upload`
action per object means the upload would be accepted; a `403` with
``Private repository storage limit reached`` means it would not.

**The one trap.** Use a FRESH RANDOM oid. An oid that already exists dedups and
returns success regardless of quota, which is a false PASS — the probe would
report room that does not exist, which is worse than not asking.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

#: git-LFS media type, required on both headers or the endpoint 406s.
_LFS_JSON = "application/vnd.git-lfs+json"


@dataclass(frozen=True)
class CapacityVerdict:
    """What the endpoint said, in a form a gate can print."""

    accepted: bool
    status: int
    detail: str
    n_objects: int
    total_bytes: int

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 2 ** 30

    def __str__(self) -> str:
        return (f"{self.n_objects} object(s), {self.total_gib:.2f} GiB: "
                f"{'ACCEPTED' if self.accepted else 'REFUSED'} "
                f"({self.status}) {self.detail}")


def would_accept(repo_id: str, sizes: list[int], token: str,
                 *, repo_type: str = "model", timeout: int = 90,
                 ) -> CapacityVerdict:
    """Would `repo_id` accept new LFS objects of exactly these sizes?

    `sizes` is the real shape of the intended upload — nine probes is nine
    entries, not one entry of nine times the size. The quota is account-wide
    (measured 2026-08-22: a brand-new private repo received no allowance of its
    own), so the answer is about the ACCOUNT, whichever repo is asked.

    Raises on a transport failure rather than returning a verdict, because
    "could not ask" and "was refused" must not look alike to a caller deciding
    whether to spend money.
    """
    import requests

    if not sizes:
        raise ValueError("no sizes to ask about")
    if any(s <= 0 for s in sizes):
        raise ValueError(f"sizes must be positive, got {sorted(sizes)[:3]}")

    prefix = "datasets/" if repo_type == "dataset" else ""
    objects = [{"oid": secrets.token_hex(32), "size": int(s)} for s in sizes]
    response = requests.post(
        f"https://huggingface.co/{prefix}{repo_id}.git/info/lfs/objects/batch",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": _LFS_JSON, "Content-Type": _LFS_JSON},
        json={"operation": "upload", "transfers": ["basic", "multipart"],
              "objects": objects, "hash_algo": "sha_256"},
        timeout=timeout)

    total = sum(int(s) for s in sizes)
    if response.status_code == 200:
        body = response.json()
        granted = sum(1 for o in body.get("objects", ())
                      if "upload" in (o.get("actions") or {}))
        #: A 200 is not sufficient. The endpoint answers per object, and an
        #: object it will not take simply arrives without an upload action.
        return CapacityVerdict(
            accepted=granted == len(objects), status=200,
            detail=f"{granted}/{len(objects)} granted an upload action",
            n_objects=len(objects), total_bytes=total)

    #: Every status here is a definite NO about this upload, not a failure to
    #: ask. `403` is the quota refusal; `413`/`422` are size refusals — a
    #: single object above the per-file maximum comes back `422 Unprocessable
    #: Entity` rather than `403`, and reporting that as "could not ask" would
    #: send a caller looking for a network fault that is not there.
    if response.status_code in (400, 403, 413, 422):
        try:
            message = response.json().get("message", response.text)
        except ValueError:
            message = response.text
        return CapacityVerdict(accepted=False, status=response.status_code,
                               detail=str(message)[:200] or
                               f"refused with {response.status_code} and no message",
                               n_objects=len(objects), total_bytes=total)

    response.raise_for_status()
    raise RuntimeError(f"unexpected LFS batch status {response.status_code}: "
                       f"{response.text[:200]}")
