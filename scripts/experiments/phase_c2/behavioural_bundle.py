"""Bundle transport for ONE behavioural-selection session.

A thin instance. The build, the upload, the download and the verification all
remain `aadistill.infrastructure.bundle_transport`'s. The relay repository and
prefix are shared with every other session — the commit distinguishes one
bundle from another, not the name — so only the label differs, and a refusal
about this session reads as being about it.

**What the bundle carries.** The repository at the session commit, so the frozen
protocol, both battery identity records, attempt 3's selection and journal, the
teacher binding and the recovery recipe all travel with it. Those are not
context: they ARE the plan. The seeds, the batteries, the SESOI, the guardrails,
the bootstrap seed, the six arms' pinned construction paths and the identity
each is gated against are read out of them.

It does NOT carry anything under `artifacts/`, which is gitignored: the
calibration mixtures, the recovery pack, the evaluation tokenizer sidecars and
both batteries reach the pod over the relay and the scp path instead. Two paid
subruns of another session were lost learning that split, one per half — the
first because nothing under `logs/` reached the pod, the second because nothing
under `artifacts/` did.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aadistill.infrastructure import bundle_transport as _bt
from aadistill.infrastructure.bundle_transport import (  # noqa: F401
    BundleTransportError, TransportSpec, build_bundle, checkout_from_bundle,
    hf_download, sha256_bytes,
)

from experiments.phase_c2 import behavioural_governance as BG

REPO = Path(__file__).resolve().parents[3]

RELAY_REPO = "AlphaAvatar/aadistill-artifacts"

BEHAVIOURAL_TRANSPORT = TransportSpec(
    label="Phase-C2 behavioural selection",
    relay_repo=RELAY_REPO,
    transfer_prefix="transfer",
)


def canonical_bundle_name(session_commit: str) -> str:
    return BEHAVIOURAL_TRANSPORT.bundle_name(session_commit)


def canonical_repo_path(session_commit: str) -> str:
    return BEHAVIOURAL_TRANSPORT.repo_path(session_commit)


def require_canonical_bundle_arg(bundle: str, session_commit: str) -> str:
    return BEHAVIOURAL_TRANSPORT.require_canonical(bundle, session_commit)


def behavioural_executable_set(repo_root: str | Path = REPO
                               ) -> tuple[str, tuple[str, ...]]:
    """`(digest, files)` for the live derived behavioural closure.

    Both halves from ONE derivation. An authorization that declared one file set
    and digested another would be checkable against neither.
    """
    live = BG.current_executable(repo_root)
    return live["digest"], tuple(row["path"] for row in live["files"])


def stage_bundle(session_commit: str, *, workdir: str | Path,
                 repo_root: str | Path = REPO) -> dict[str, Any]:
    """Build, upload and verify this session's bundle on the shared relay."""
    return _bt.stage_bundle(BEHAVIOURAL_TRANSPORT, Path(repo_root),
                            session_commit, workdir=Path(workdir))


def roundtrip(*, session_commit: str, local_bundle_sha256: str,
              authorization_bytes: bytes, authorization_path: str,
              workdir: Path, download=hf_download,
              expected_harness_digest: str | None = None,
              harness_files: tuple[str, ...] | None = None,
              repo_root: str | Path = REPO) -> dict[str, Any]:
    """READ-ONLY transport gate. Writes nothing.

    Downloads the object a pod would fetch, checks out the session commit from
    it, and re-digests the authorized executable set INSIDE that checkout — so
    the gate verifies what the pod will actually run rather than what this
    machine happens to hold.

    The expected digest and file set default to the LIVE closure and may be
    overridden only so a test can present a deliberately wrong pair and watch
    the gate refuse it.
    """
    if expected_harness_digest is None or harness_files is None:
        digest, files = behavioural_executable_set(repo_root)
        expected_harness_digest = expected_harness_digest or digest
        harness_files = harness_files or files
    return _bt.roundtrip(
        BEHAVIOURAL_TRANSPORT, session_commit=session_commit,
        local_bundle_sha256=local_bundle_sha256,
        authorization_bytes=authorization_bytes,
        authorization_path=authorization_path,
        expected_harness_digest=expected_harness_digest,
        harness_files=harness_files, download=download, workdir=Path(workdir))
