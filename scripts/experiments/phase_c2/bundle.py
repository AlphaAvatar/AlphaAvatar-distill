"""C2's transport identity. The mechanism is in infrastructure, not here.

`aadistill.infrastructure.bundle_transport` answers the transport question —
build the exact commit into a bundle, verify it, upload it, download what the
pod would fetch, clone it, and check that the checkout is the exact session
commit carrying the exact authorization bytes with the authorized executable
digest. This file supplies the four facts that are C2's: the relay repository,
the path prefix inside it, the label that appears in refusals, and where the
authorization lives.

The executable set the round-trip re-digests is the LIVE derived closure, not a
hand-maintained list. That matters here more than anywhere: the round-trip
digests the files *inside the bundle* and compares them to the authorized value,
so a set that named a path the closure does not cover would certify a bundle
whose uncovered files could be anything.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.bundle_transport import (  # noqa: E402
    BundleTransportError,
    TransportSpec,
    build_bundle,
    checkout_from_bundle,
    hf_download,
    sha256_bytes,
)
from aadistill.infrastructure import bundle_transport as _bt  # noqa: E402

#: The same relay every session in this project uses, and the same `transfer/`
#: prefix. Declared rather than inherited: a session that fetched from
#: somewhere else would be a different transport question.
C2_TRANSPORT = TransportSpec(
    label="Phase-C2 Search-1",
    relay_repo="AlphaAvatar/aadistill-artifacts",
    transfer_prefix="transfer",
)


def canonical_bundle_name(session_commit: str) -> str:
    return C2_TRANSPORT.bundle_name(session_commit)


def canonical_repo_path(session_commit: str) -> str:
    return C2_TRANSPORT.repo_path(session_commit)


def require_canonical_bundle_arg(bundle: str, session_commit: str) -> str:
    return C2_TRANSPORT.require_canonical(bundle, session_commit)


def c2_authorization_path(run_id: str | None = None,
                          stage_id: str | None = None) -> str:
    """Where this run's authorization lives, as the round-trip must find it.

    Resolved through the same convention as every other run role, so the
    issuer, the launcher gate and the bundle check cannot disagree about which
    file the pod will load.
    """
    from experiments.phase_c2.session import c2_authorization_path as declared

    return declared(run_id, stage_id)


def c2_executable_set(repo_root: str | Path = REPO) -> tuple[str, tuple[str, ...]]:
    """`(digest, files)` for the live derived C2 closure.

    Both halves come from one derivation, which is the invariant C1 learned to
    state explicitly: an authorization that declares one file set and binds a
    digest over another can never pass its own round-trip, because the gate
    re-digests the declared set.
    """
    from experiments.phase_c2.session import c2_harness_digest

    live = c2_harness_digest(repo_root)
    return live["digest"], tuple(f["path"] for f in live["files"])


def roundtrip(*, session_commit: str, local_bundle_sha256: str,
              authorization_bytes: bytes, authorization_path: str,
              workdir: Path, download=hf_download,
              expected_harness_digest: str | None = None,
              harness_files: tuple[str, ...] | None = None,
              repo_root: str | Path = REPO) -> dict[str, Any]:
    """READ-ONLY transport gate for C2. Writes nothing anywhere.

    The expected digest and file set default to the LIVE closure and may be
    overridden only so a test can present a deliberately wrong pair and watch
    the gate refuse it.
    """
    if expected_harness_digest is None or harness_files is None:
        digest, files = c2_executable_set(repo_root)
        expected_harness_digest = expected_harness_digest or digest
        harness_files = harness_files or files
    return _bt.roundtrip(
        C2_TRANSPORT, session_commit=session_commit,
        local_bundle_sha256=local_bundle_sha256,
        authorization_bytes=authorization_bytes,
        authorization_path=authorization_path,
        expected_harness_digest=expected_harness_digest,
        harness_files=harness_files, download=download, workdir=Path(workdir))


def stage_bundle(session_commit: str, *, workdir: Path, api=None,
                 download=None, repo_root: str | Path = REPO) -> dict[str, Any]:
    """Build, verify and upload C2's session bundle. MUTATES THE RELAY."""
    return _bt.stage_bundle(C2_TRANSPORT, Path(repo_root), session_commit,
                            workdir=Path(workdir), api=api, download=download)


__all__ = ["BundleTransportError", "C2_TRANSPORT", "build_bundle",
           "c2_authorization_path", "c2_executable_set",
           "canonical_bundle_name", "canonical_repo_path",
           "checkout_from_bundle", "hf_download", "require_canonical_bundle_arg",
           "roundtrip", "sha256_bytes", "stage_bundle"]
