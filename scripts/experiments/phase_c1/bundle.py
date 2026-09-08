"""Can a pod, right now, obtain the exact authorized code?

C1 attempt 1 passed all eight pre-provider gates, created a pod and died at
`SETUP_RC=1`: the pod could not fetch `transfer/c1`, because no git bundle had
been created for the session commit. `$0.0786` for a 404.

Every gate that ran verified the *contents* of the session commit — the harness
digest, the lineage, the authorization, the preregistration, the frozen science,
the teacher binding, the battery, the artifact specs. Not one asked whether the
pod could **reach** that commit. "Regenerate the git bundle and re-upload it" was
a documented step in `scripts/pod/AGENTS.md`, so it depended on an operator
remembering it, and the operator did not.

This module answers the transport question instead of assuming it, and splits the
answer in two:

* **preparation may mutate the relay.** `stage_bundle` builds, verifies and
  uploads. It refuses to overwrite a different object already sitting under the
  canonical name, because a silent overwrite is how one session's bundle becomes
  another's.
* **the gate is read-only.** `roundtrip` downloads the object the pod would
  fetch, verifies it, clones it, and checks that the resulting checkout is the
  exact session commit carrying the exact authorization, with a harness that
  digests to the authorized value. It writes nothing anywhere.

The name is derived, never chosen. `--bundle c1` was an alias for nothing;
`aad_autoinit_<first 8 hex of the session commit>.bundle` cannot be, because the
commit is in it.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

RELAY_REPO = "AlphaAvatar/aadistill-artifacts"
TRANSFER_PREFIX = "transfer"
BUNDLE_PREFIX = "aad_autoinit_"
COMMIT_ABBREV = 8


class C1BundleError(RuntimeError):
    """The pod could not obtain the authorized commit, or would obtain another."""


def canonical_bundle_name(session_commit: str) -> str:
    """`aad_autoinit_<first 8 hex of the commit>.bundle`. Derived, never chosen."""
    c = (session_commit or "").strip().lower()
    if len(c) < COMMIT_ABBREV or any(ch not in "0123456789abcdef" for ch in c):
        raise C1BundleError(
            f"{session_commit!r} is not a hex commit id, so no canonical bundle "
            "name can be derived from it")
    return f"{BUNDLE_PREFIX}{c[:COMMIT_ABBREV]}.bundle"


def canonical_repo_path(session_commit: str) -> str:
    return f"{TRANSFER_PREFIX}/{canonical_bundle_name(session_commit)}"


def require_canonical_bundle_arg(bundle: str, session_commit: str) -> str:
    """An alias must fail at `$0`, not at `SETUP_RC=1` on a billing pod."""
    want = canonical_bundle_name(session_commit)
    if bundle != want:
        raise C1BundleError(
            f"--bundle {bundle!r} is not the canonical name for session commit "
            f"{session_commit[:12]}…, which is {want!r}. Attempt 1 passed "
            f"`--bundle c1`, an alias for nothing, and the pod 404'd on it.")
    return want


def _git(*args: str, cwd: Path | None = None, check: bool = True):
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
    if check and out.returncode != 0:
        raise C1BundleError(
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


def build_bundle(repo_root: Path, session_commit: str, dest: Path) -> dict[str, Any]:
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
    ref = f"refs/heads/_c1_bundle_{session_commit[:12]}"
    _git("update-ref", ref, session_commit, cwd=repo_root)
    try:
        _git("bundle", "create", str(dest), ref, cwd=repo_root)
    finally:
        _git("update-ref", "-d", ref, cwd=repo_root, check=False)
    verify = _git("bundle", "verify", str(dest), cwd=repo_root)
    heads = _git("bundle", "list-heads", str(dest), cwd=repo_root).stdout
    if session_commit not in heads:
        raise C1BundleError(
            f"{dest} verifies but does not carry {session_commit[:12]}…; its heads "
            f"are:\n{heads.strip()}")
    digest, size = sha256_bytes(dest)
    return {"path": str(dest), "sha256": digest, "bytes": size,
            "session_commit": session_commit,
            "canonical_name": canonical_bundle_name(session_commit),
            "verify": verify.stdout.strip() or verify.stderr.strip(),
            "heads": heads.strip()}


def checkout_from_bundle(bundle: Path, workdir: Path,
                         session_commit: str) -> Path:
    """Exactly what the pod does: clone the bundle, then check the commit out.

    `autoinit_preflight_setup.sh` runs `git clone -q "$WS/$BUNDLE_NAME" "$REPO"`
    and then `git checkout -q "$SESSION_COMMIT"`. Reproducing both steps is the
    point -- a bundle whose default HEAD is unusable but whose objects are
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
        raise C1BundleError(
            f"the bundle clones but does not contain {session_commit[:12]}…: "
            f"{out.stderr.strip()[-300:]}")
    return repo


def roundtrip(*, session_commit: str, local_bundle_sha256: str,
              authorization_bytes: bytes, authorization_path: str,
              expected_harness_digest: str, harness_files: tuple[str, ...],
              download, workdir: Path) -> dict[str, Any]:
    """READ-ONLY. Does the object the pod would fetch carry the authorized code?

    `download` is injected — it takes `(repo_id, path_in_repo, dest_dir)` and
    returns the downloaded path — so the gate can be driven against a local
    fixture in tests and against the real relay in production without the
    verification logic differing between the two.

    Every check is an equality against something the launcher already holds, so a
    bundle that is merely *a* bundle, or *a* checkout, cannot satisfy it.
    """
    name = canonical_bundle_name(session_commit)
    path_in_repo = canonical_repo_path(session_commit)
    record: dict[str, Any] = {
        "session_commit": session_commit,
        "canonical_bundle_name": name,
        "relay_repo": RELAY_REPO,
        "relay_path": path_in_repo,
    }

    fetched = download(RELAY_REPO, path_in_repo, workdir)
    fetched = Path(fetched)
    if not fetched.is_file():
        raise C1BundleError(
            f"{RELAY_REPO}:{path_in_repo} did not download to a file")
    remote_sha, size = sha256_bytes(fetched)
    record["bytes"] = size
    record["remote_sha256"] = remote_sha
    record["local_sha256"] = local_bundle_sha256
    if remote_sha != local_bundle_sha256:
        raise C1BundleError(
            f"the relay object {path_in_repo} hashes to {remote_sha} but the "
            f"staged bundle hashes to {local_bundle_sha256}. The pod would fetch "
            "different bytes than were prepared.")

    verify = subprocess.run(["git", "bundle", "verify", str(fetched)],
                            capture_output=True, text=True)
    record["bundle_verify_rc"] = verify.returncode
    if verify.returncode != 0:
        raise C1BundleError(
            f"`git bundle verify` failed on the round-tripped remote object: "
            f"{(verify.stdout + verify.stderr).strip()[-600:]}")

    repo = checkout_from_bundle(fetched, workdir / "rt", session_commit)
    head = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    record["roundtrip_head"] = head
    if head != session_commit:
        raise C1BundleError(
            f"the round-tripped bundle checks out {head[:12]}…, not the session "
            f"commit {session_commit[:12]}…. A pod would run a different tree.")

    blob = _git("show", f"{head}:{authorization_path}", cwd=repo, check=False)
    if blob.returncode != 0:
        raise C1BundleError(
            f"the round-tripped checkout does not contain {authorization_path}; "
            "the driver would load no authorization, or another one")
    got = blob.stdout.encode()
    record["authorization_sha256"] = hashlib.sha256(got).hexdigest()
    record["authorization_matches"] = got == authorization_bytes
    if got != authorization_bytes:
        raise C1BundleError(
            f"the {authorization_path} inside the bundle is not the artifact this "
            "launcher is loading; the pod would run under a different grant")

    entries = []
    for rel in sorted(harness_files):
        f = repo / rel
        if not f.is_file():
            raise C1BundleError(
                f"the round-tripped checkout is missing declared harness source "
                f"{rel!r}")
        entries.append({"path": rel, "sha256": sha256_bytes(f)[0]})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    record["roundtrip_harness_digest"] = digest
    record["expected_harness_digest"] = expected_harness_digest
    if digest != expected_harness_digest:
        raise C1BundleError(
            f"the harness inside the bundle digests to {digest}, authorized "
            f"{expected_harness_digest}. The pod would run code this "
            "authorization was not granted against.")

    record["ok"] = True
    record["answers"] = ("a pod can obtain the exact authorized code from the "
                         "relay right now, not merely that a local file once existed")
    return record


def hf_download(repo_id: str, path_in_repo: str, dest_dir: Path) -> Path:
    """The real relay fetch, matching what the pod's setup does."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id, path_in_repo, local_dir=str(dest_dir)))


def stage_bundle(repo_root: Path, session_commit: str, *, workdir: Path,
                 api=None) -> dict[str, Any]:
    """Build, verify and upload. MUTATES THE RELAY; the gate does not.

    Refuses to overwrite an object already under the canonical name whose bytes
    differ. Two sessions cannot share an 8-hex prefix by accident, but a rebuilt
    bundle for the same commit can differ byte-for-byte (git bundles are not
    reproducible), and silently replacing one that a launcher has already
    verified is how the gate's answer stops being true.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError

    api = api or HfApi()
    built = build_bundle(repo_root, session_commit, workdir / canonical_bundle_name(session_commit))
    path_in_repo = canonical_repo_path(session_commit)

    existing = None
    try:
        remote = hf_download(RELAY_REPO, path_in_repo, workdir / "existing")
        existing = sha256_bytes(Path(remote))[0]
    except Exception as exc:                                   # noqa: BLE001
        if not isinstance(exc, EntryNotFoundError) and "404" not in str(exc):
            raise
    if existing is not None and existing != built["sha256"]:
        raise C1BundleError(
            f"{path_in_repo} already exists on the relay with sha256 {existing}, "
            f"which is not the bundle just built ({built['sha256']}). "
            "Refusing to overwrite a different remote object: a launcher may "
            "already have verified those bytes.")
    if existing == built["sha256"]:
        built["upload"] = "already present with identical bytes; nothing uploaded"
        return built

    api.upload_file(path_or_fileobj=built["path"], path_in_repo=path_in_repo,
                    repo_id=RELAY_REPO, repo_type="model",
                    commit_message=f"C1 session bundle for {session_commit[:12]}")
    built["upload"] = f"uploaded to {RELAY_REPO}:{path_in_repo}"
    return built
