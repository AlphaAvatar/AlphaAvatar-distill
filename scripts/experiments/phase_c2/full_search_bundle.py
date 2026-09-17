"""Bundle transport for ONE Phase-C2 full-search session.

A thin instance. The build, the upload, the download and the verification logic
all remain `aadistill.infrastructure.bundle_transport`'s. There is no second
transport framework here, and the shared relay means a full-search bundle sits
beside every other session's under the same convention.

**What the bundle does and does not carry.** It carries the repository at the
session commit, so every tracked input the search reads travels with it —
including the two telemetry files and the level record the cost model pools its
per-expansion table over, which are named in the executable closure precisely so
their bytes are part of the identity a grant binds. It does NOT carry the
calibration item files or the frozen metric suite: those live in the out-of-tree
artifact store, which is gitignored, so the launcher stages them over the relay
as `LocalAsset`s that `full_search.staged_assets` derives.

That split is not a convenience. The CUDA engineering validation lost two paid
subruns learning it, one per half: the first because nothing under `logs/`
reached the pod, the second because nothing under `artifacts/` did.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aadistill.infrastructure import bundle_transport as _bt
from aadistill.infrastructure.bundle_transport import (  # noqa: F401
    BundleTransportError, TransportSpec, build_bundle, checkout_from_bundle,
    hf_download, sha256_bytes,
)

from experiments.phase_c2 import full_search as FSG

REPO = Path(__file__).resolve().parents[3]

#: The relay repository and prefix are SHARED with every other session: the
#: commit is what distinguishes one bundle from another, not the name. Only the
#: label differs, so a refusal about this session reads as being about it.
RELAY_REPO = "AlphaAvatar/aadistill-artifacts"

FULL_SEARCH_TRANSPORT = TransportSpec(
    label="Phase-C2 full joint re-search",
    relay_repo=RELAY_REPO,
    transfer_prefix="transfer",
)


def canonical_bundle_name(session_commit: str) -> str:
    return FULL_SEARCH_TRANSPORT.bundle_name(session_commit)


def canonical_repo_path(session_commit: str) -> str:
    return FULL_SEARCH_TRANSPORT.repo_path(session_commit)


def require_canonical_bundle_arg(bundle: str, session_commit: str) -> str:
    return FULL_SEARCH_TRANSPORT.require_canonical(bundle, session_commit)


def full_search_executable_set(repo_root: str | Path = REPO
                               ) -> tuple[str, tuple[str, ...]]:
    """`(digest, files)` for the live derived FULL-SEARCH closure.

    Both halves from one derivation. An authorization that declared one file set
    and bound a digest over another could never pass its own round-trip, because
    the gate re-digests the declared set -- so deriving them apart is how the
    two come to disagree.
    """
    live = FSG.current_executable(repo_root)
    return live["digest"], tuple(row["path"] for row in live["files"])


def roundtrip(*, session_commit: str, local_bundle_sha256: str,
              authorization_bytes: bytes, authorization_path: str,
              workdir: Path, download=hf_download,
              expected_harness_digest: str | None = None,
              harness_files: tuple[str, ...] | None = None,
              repo_root: str | Path = REPO) -> dict[str, Any]:
    """READ-ONLY transport gate for the full search. Writes nothing.

    The expected digest and file set default to the LIVE FULL-SEARCH closure and
    may be overridden only so a test can present a deliberately wrong pair and
    watch the gate refuse it.
    """
    if expected_harness_digest is None or harness_files is None:
        digest, files = full_search_executable_set(repo_root)
        expected_harness_digest = expected_harness_digest or digest
        harness_files = harness_files or files
    return _bt.roundtrip(
        FULL_SEARCH_TRANSPORT, session_commit=session_commit,
        local_bundle_sha256=local_bundle_sha256,
        authorization_bytes=authorization_bytes,
        authorization_path=authorization_path,
        expected_harness_digest=expected_harness_digest,
        harness_files=harness_files, download=download, workdir=Path(workdir))


def stage_bundle(session_commit: str, *, workdir: Path, api=None,
                 download=None, repo_root: str | Path = REPO) -> dict[str, Any]:
    """Build, verify and upload the full-search session bundle. MUTATES THE RELAY.

    Not called until an authorized session commit exists to bundle.
    """
    return _bt.stage_bundle(FULL_SEARCH_TRANSPORT, Path(repo_root),
                            session_commit, workdir=Path(workdir), api=api,
                            download=download)


__all__ = ["BundleTransportError", "FULL_SEARCH_TRANSPORT", "RELAY_REPO",
           "build_bundle", "canonical_bundle_name", "canonical_repo_path",
           "checkout_from_bundle", "full_search_executable_set", "hf_download",
           "require_canonical_bundle_arg", "roundtrip", "sha256_bytes",
           "stage_bundle"]
