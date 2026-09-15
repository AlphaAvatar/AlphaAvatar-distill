"""Can a pod, right now, obtain the exact authorized code?

A session can verify everything about the *contents* of its session commit —
the executable digest, the lineage, the authorization, the frozen science — and
still create a pod that cannot **reach** that commit, because no bundle was ever
built for it. That has happened, for real money, and "regenerate the bundle and
re-upload it" being a documented step is what made it possible: a documented
step depends on somebody remembering. `docs/core-provenance.md` records which
session paid for it.

The answer splits in two, and the split is the design:

* **preparation may mutate the relay.** `stage_bundle` builds, verifies and
  uploads. It refuses to overwrite a different object already under the
  canonical name, because a silent overwrite is how one session's bundle
  becomes another's.
* **the gate is read-only.** `roundtrip` downloads the object the pod would
  fetch, verifies it, clones it, and checks that the resulting checkout is the
  exact session commit carrying the exact authorization bytes, with an
  executable set that digests to the authorized value. It writes nothing.

Everything an experiment owns is a parameter: the relay repository, the path
prefix inside it, the label used in messages, and — at call time — the session
commit, the authorization path and bytes, and the expected executable digest
with the file set it covers. `TransportSpec` is the whole vocabulary.

The name is derived, never chosen. `--bundle c1` was an alias for nothing;
`aad_autoinit_<first 8 hex of the session commit>.bundle` cannot be, because the
commit is in it.

**There is a second copy of this logic under `scripts/experiments/`, on
purpose.** That copy is where it was written, and it is a member of a closed
experiment's frozen executable set: importing this module from it would move
that experiment's executable digest and invalidate records describing completed
attempts, for no benefit to work that will not run again. Whoever reopens it
should collapse the two then. Until that happens, the shared rule the two must
not disagree about — the digest formula — is imported here from
`aadistill.governance.closure`, so neither file owns a private copy of it.
`docs/core-provenance.md` names the copy and the reason.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


class BundleTransportError(RuntimeError):
    """The pod could not obtain the authorized commit, or would obtain another."""


@dataclass(frozen=True)
class TransportSpec:
    """One experiment's transport identity. Supplied, never assumed."""

    #: What to call this session in a refusal and in the upload's commit
    #: message. Supplied, because one experiment's name read oddly in a message
    #: about somebody else's launch.
    label: str
    #: The relay repository the pod fetches from.
    relay_repo: str
    #: The path prefix inside it.
    transfer_prefix: str = "transfer"
    #: The bundle filename prefix. Shared across sessions by default: the commit
    #: is what distinguishes one bundle from another, not the name.
    bundle_prefix: str = "aad_autoinit_"
    commit_abbrev: int = 8

    def __post_init__(self) -> None:
        for name in ("label", "relay_repo", "transfer_prefix", "bundle_prefix"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"incomplete transport spec: {name} is empty")
        if self.commit_abbrev < 7:
            raise ValueError(
                f"commit_abbrev={self.commit_abbrev} is too short to identify a "
                "commit; git's own default abbreviation is 7")

    def bundle_name(self, session_commit: str) -> str:
        """`<prefix><first N hex of the commit>.bundle`. Derived, never chosen."""
        c = (session_commit or "").strip().lower()
        if (len(c) < self.commit_abbrev
                or any(ch not in "0123456789abcdef" for ch in c)):
            raise BundleTransportError(
                f"{session_commit!r} is not a hex commit id, so no canonical "
                "bundle name can be derived from it")
        return f"{self.bundle_prefix}{c[:self.commit_abbrev]}.bundle"

    def repo_path(self, session_commit: str) -> str:
        return f"{self.transfer_prefix}/{self.bundle_name(session_commit)}"

    def require_canonical(self, bundle: str, session_commit: str) -> str:
        """An alias must fail at `$0`, not at `SETUP_RC=1` on a billing pod."""
        want = self.bundle_name(session_commit)
        if bundle != want:
            raise BundleTransportError(
                f"--bundle {bundle!r} is not the canonical name for session "
                f"commit {session_commit[:12]}…, which is {want!r}. A session "
                "once passed a short alias for nothing here and the pod 404'd "
                "on it.")
        return want


def _git(*args: str, cwd: Path | None = None, check: bool = True):
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    if check and out.returncode != 0:
        raise BundleTransportError(
            f"git {' '.join(args)} failed ({out.returncode}): {out.stderr.strip()}")
    return out


def sha256_bytes(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def build_bundle(spec: TransportSpec, repo_root: Path, session_commit: str,
                 dest: Path) -> dict[str, Any]:
    """A bundle containing that exact commit, verified before it is trusted.

    `git bundle create <dest> <commit>` alone is not enough: a bundle can verify
    cleanly and still not contain the commit anyone cares about, which is exactly
    the stale-bundle case. So the commit is re-derived from the bundle's own
    heads afterwards.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    #: `git bundle create <dest> <bare sha>` refuses with "Refusing to create
    #: empty bundle": a bundle records REFS, and a raw commit id is not one. So
    #: a temporary ref is pointed at the commit, bundled, and removed again --
    #: which also keeps the bundle to exactly that commit's history rather than
    #: every branch `--all` would sweep in.
    ref = f"refs/heads/_bundle_{session_commit[:12]}"
    _git("update-ref", ref, session_commit, cwd=repo_root)
    try:
        _git("bundle", "create", str(dest), ref, cwd=repo_root)
    finally:
        _git("update-ref", "-d", ref, cwd=repo_root, check=False)
    verify = _git("bundle", "verify", str(dest), cwd=repo_root)
    heads = _git("bundle", "list-heads", str(dest), cwd=repo_root).stdout
    if session_commit not in heads:
        raise BundleTransportError(
            f"{dest} verifies but does not carry {session_commit[:12]}…; its "
            f"heads are:\n{heads.strip()}")
    digest, size = sha256_bytes(dest)
    return {"path": str(dest), "sha256": digest, "bytes": size,
            "session_commit": session_commit,
            "canonical_name": spec.bundle_name(session_commit),
            "verify": verify.stdout.strip() or verify.stderr.strip(),
            "heads": heads.strip()}


def checkout_from_bundle(bundle: Path, workdir: Path,
                         session_commit: str) -> Path:
    """Exactly what the pod does: clone the bundle, then check the commit out.

    `autoinit_preflight_setup.sh` runs `git clone -q "$WS/$BUNDLE_NAME" "$REPO"`
    and then `git checkout -q "$SESSION_COMMIT"`. Reproducing both steps is the
    point — a bundle whose default HEAD is unusable but whose objects are
    present is fine for the pod, and a check that only cloned would reject it
    for the wrong reason.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    repo = workdir / "checkout"
    if repo.exists():
        shutil.rmtree(repo)
    _git("clone", "-q", str(bundle), str(repo))
    out = _git("checkout", "-q", session_commit, cwd=repo, check=False)
    if out.returncode != 0:
        raise BundleTransportError(
            f"the bundle clones but does not contain {session_commit[:12]}…: "
            f"{out.stderr.strip()[-300:]}")
    return repo


def roundtrip(spec: TransportSpec, *, session_commit: str,
              local_bundle_sha256: str, authorization_bytes: bytes,
              authorization_path: str, expected_harness_digest: str,
              harness_files: Sequence[str],
              download: Callable[[str, str, Path], Any],
              workdir: Path) -> dict[str, Any]:
    """READ-ONLY. Does the object the pod would fetch carry the authorized code?

    `download` is injected — it takes `(repo_id, path_in_repo, dest_dir)` and
    returns the downloaded path — so the gate can be driven against a local
    fixture in tests and against the real relay in production without the
    verification logic differing between the two.

    Every check is an equality against something the launcher already holds, so
    a bundle that is merely *a* bundle, or *a* checkout, cannot satisfy it.
    """
    from aadistill.governance.closure import digest_of

    if not harness_files:
        raise BundleTransportError(
            "no harness file set was supplied, so the digest below would be "
            "computed over nothing and would match nothing. An empty set passes "
            "vacuously, which is the one outcome a transport gate must not have.")

    name = spec.bundle_name(session_commit)
    path_in_repo = spec.repo_path(session_commit)
    record: dict[str, Any] = {
        "session_commit": session_commit,
        "canonical_bundle_name": name,
        "relay_repo": spec.relay_repo,
        "relay_path": path_in_repo,
        "n_harness_files": len(harness_files),
    }

    fetched = Path(download(spec.relay_repo, path_in_repo, workdir))
    if not fetched.is_file():
        raise BundleTransportError(
            f"{spec.relay_repo}:{path_in_repo} did not download to a file")
    remote_sha, size = sha256_bytes(fetched)
    record["bytes"] = size
    record["remote_sha256"] = remote_sha
    record["local_sha256"] = local_bundle_sha256
    if remote_sha != local_bundle_sha256:
        raise BundleTransportError(
            f"the relay object {path_in_repo} hashes to {remote_sha} but the "
            f"staged bundle hashes to {local_bundle_sha256}. The pod would fetch "
            "different bytes than were prepared.")

    verify = subprocess.run(["git", "bundle", "verify", str(fetched)],
                            capture_output=True, text=True)
    record["bundle_verify_rc"] = verify.returncode
    if verify.returncode != 0:
        raise BundleTransportError(
            "`git bundle verify` failed on the round-tripped remote object: "
            f"{(verify.stdout + verify.stderr).strip()[-600:]}")

    repo = checkout_from_bundle(fetched, workdir / "rt", session_commit)
    head = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    record["roundtrip_head"] = head
    if head != session_commit:
        raise BundleTransportError(
            f"the round-tripped bundle checks out {head[:12]}…, not the session "
            f"commit {session_commit[:12]}…. A pod would run a different tree.")

    blob = _git("show", f"{head}:{authorization_path}", cwd=repo, check=False)
    if blob.returncode != 0:
        raise BundleTransportError(
            f"the round-tripped checkout does not contain {authorization_path}; "
            "the driver would load no authorization, or another one")
    got = blob.stdout.encode()
    record["authorization_sha256"] = hashlib.sha256(got).hexdigest()
    record["authorization_matches"] = got == authorization_bytes
    if got != authorization_bytes:
        raise BundleTransportError(
            f"the {authorization_path} inside the bundle is not the artifact "
            "this launcher is loading; the pod would run under a different grant")

    entries = []
    for rel in sorted(harness_files):
        f = repo / rel
        if not f.is_file():
            raise BundleTransportError(
                "the round-tripped checkout is missing declared executable "
                f"source {rel!r}")
        entries.append({"path": rel, "sha256": sha256_bytes(f)[0]})
    #: The SHARED formula, imported rather than restated. Two copies of a digest
    #: rule are two rules as soon as one of them is edited.
    digest = digest_of(entries)
    record["roundtrip_harness_digest"] = digest
    record["expected_harness_digest"] = expected_harness_digest
    if digest != expected_harness_digest:
        raise BundleTransportError(
            f"the executable set inside the bundle digests to {digest}, "
            f"authorized {expected_harness_digest}. The pod would run code this "
            "authorization was not granted against.")

    record["ok"] = True
    record["answers"] = ("a pod can obtain the exact authorized code from the "
                         "relay right now, not merely that a local file once "
                         "existed")
    return record


def hf_download(repo_id: str, path_in_repo: str, dest_dir: Path) -> Path:
    """The real relay fetch, matching what the pod's setup does."""
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id, path_in_repo, local_dir=str(dest_dir)))


def stage_bundle(spec: TransportSpec, repo_root: Path, session_commit: str, *,
                 workdir: Path, api=None,
                 download: Callable[[str, str, Path], Any] | None = None,
                 ) -> dict[str, Any]:
    """Build, verify and upload. MUTATES THE RELAY; the gate does not.

    Refuses to overwrite an object already under the canonical name whose bytes
    differ. Two sessions cannot share an 8-hex prefix by accident, but a rebuilt
    bundle for the same commit can differ byte-for-byte — git bundles are not
    reproducible — and silently replacing one a launcher has already verified is
    how the gate's answer stops being true.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError

    api = api or HfApi()
    fetch = download or hf_download
    built = build_bundle(spec, repo_root, session_commit,
                         workdir / spec.bundle_name(session_commit))
    path_in_repo = spec.repo_path(session_commit)

    existing = None
    try:
        remote = fetch(spec.relay_repo, path_in_repo, workdir / "existing")
        existing = sha256_bytes(Path(remote))[0]
    except Exception as exc:                                   # noqa: BLE001
        if not isinstance(exc, EntryNotFoundError) and "404" not in str(exc):
            raise
    if existing is not None and existing != built["sha256"]:
        raise BundleTransportError(
            f"{path_in_repo} already exists on the relay with sha256 {existing}, "
            f"which is not the bundle just built ({built['sha256']}). Refusing "
            "to overwrite a different remote object: a launcher may already have "
            "verified those bytes.")
    if existing == built["sha256"]:
        built["upload"] = "already present with identical bytes; nothing uploaded"
        return built

    api.upload_file(path_or_fileobj=built["path"], path_in_repo=path_in_repo,
                    repo_id=spec.relay_repo, repo_type="model",
                    commit_message=f"{spec.label} session bundle for "
                                   f"{session_commit[:12]}")
    built["upload"] = f"uploaded to {spec.relay_repo}:{path_in_repo}"
    return built
