"""Bundle transport for the baseline-completion session. Thin over the generic.

The pod's setup downloads `AlphaAvatar/aadistill-artifacts:transfer/<BUNDLE>`
and checks out from it BEFORE any scientific stage, so a formal path has to
prove at `$0` that the canonical object exists and carries three things: the
exact authorized session commit, the exact authorization bytes the launcher
holds, and the exact executable closure the grant binds.

`experiments.phase_c2.bundle` answers that question for Search-1 and cannot
answer it here: it re-derives the SEARCH-1 closure and resolves the SEARCH-1
authorization path. A bundle verified against those would be verified against
the wrong code and the wrong permission -- and would pass, which is worse than
failing.

So this module supplies the three things that differ and reuses everything else:
the relay repository, the transfer prefix, the bundle naming, the build, the
upload, the download and the verification logic all remain
`aadistill.infrastructure.bundle_transport`'s. There is no second transport
framework here, and the shared relay means a completion bundle sits beside every
other session's under the same convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aadistill.infrastructure import bundle_transport as _bt
from aadistill.infrastructure.bundle_transport import (
    BundleTransportError, TransportSpec, build_bundle, checkout_from_bundle,
    hf_download, sha256_bytes,
)

from experiments.phase_c2 import baseline_completion as BC

REPO = Path(__file__).resolve().parents[3]

#: The relay repository and prefix are SHARED with every other session: the
#: commit is what distinguishes one bundle from another, not the name. Only the
#: label differs, so a refusal about this session reads as being about it.
RELAY_REPO = "AlphaAvatar/aadistill-artifacts"

COMPLETION_TRANSPORT = TransportSpec(
    label="Phase-C2 baseline completion",
    relay_repo=RELAY_REPO,
    transfer_prefix="transfer",
)


def canonical_bundle_name(session_commit: str) -> str:
    return COMPLETION_TRANSPORT.bundle_name(session_commit)


def canonical_repo_path(session_commit: str) -> str:
    return COMPLETION_TRANSPORT.repo_path(session_commit)


def require_canonical_bundle_arg(bundle: str, session_commit: str) -> str:
    return COMPLETION_TRANSPORT.require_canonical(bundle, session_commit)


def completion_executable_set(repo_root: str | Path = REPO
                              ) -> tuple[str, tuple[str, ...]]:
    """`(digest, files)` for the live derived COMPLETION closure.

    Both halves from one derivation. An authorization that declared one file set
    and bound a digest over another could never pass its own round-trip, because
    the gate re-digests the declared set -- so deriving them apart is how the
    two come to disagree.
    """
    live = BC.current_executable(repo_root)
    return live["digest"], tuple(row["path"] for row in live["files"])


def roundtrip(*, session_commit: str, local_bundle_sha256: str,
              authorization_bytes: bytes, authorization_path: str,
              workdir: Path, download=hf_download,
              expected_harness_digest: str | None = None,
              harness_files: tuple[str, ...] | None = None,
              repo_root: str | Path = REPO) -> dict[str, Any]:
    """READ-ONLY transport gate for baseline completion. Writes nothing.

    The expected digest and file set default to the LIVE COMPLETION closure and
    may be overridden only so a test can present a deliberately wrong pair and
    watch the gate refuse it.
    """
    if expected_harness_digest is None or harness_files is None:
        digest, files = completion_executable_set(repo_root)
        expected_harness_digest = expected_harness_digest or digest
        harness_files = harness_files or files
    return _bt.roundtrip(
        COMPLETION_TRANSPORT, session_commit=session_commit,
        local_bundle_sha256=local_bundle_sha256,
        authorization_bytes=authorization_bytes,
        authorization_path=authorization_path,
        expected_harness_digest=expected_harness_digest,
        harness_files=harness_files, download=download, workdir=Path(workdir))


def stage_bundle(session_commit: str, *, workdir: Path, api=None,
                 download=None, repo_root: str | Path = REPO) -> dict[str, Any]:
    """Build, verify and upload the completion session bundle. MUTATES THE RELAY.

    Not called this round: no authorized session commit exists to bundle.
    """
    return _bt.stage_bundle(COMPLETION_TRANSPORT, Path(repo_root), session_commit,
                            workdir=Path(workdir), api=api, download=download)


__all__ = ["BundleTransportError", "COMPLETION_TRANSPORT", "RELAY_REPO",
           "build_bundle", "canonical_bundle_name", "canonical_repo_path",
           "checkout_from_bundle", "completion_executable_set", "hf_download",
           "require_canonical_bundle_arg", "roundtrip", "sha256_bytes",
           "stage_bundle"]
