"""Bundle transport for ONE replay session.

A thin instance. The build, the upload, the download and the verification logic
all remain `aadistill.infrastructure.bundle_transport`'s. The relay repository
and prefix are shared with every other session — the commit is what
distinguishes one bundle from another, not the name — so only the label differs,
and a refusal about this session reads as being about it.

**What the bundle carries, and why it is the whole plan.** It carries the
repository at the session commit, so attempt 3's selection, its compact journal
and its telemetry all travel with it. Those three files are not context for this
session: they ARE its plan. The specs, the twenty digest pins, the identity every
leaf is checked against and the per-path cost bound admission control spends
against are all read out of them, which is why they are named in the executable
closure and why their bytes are part of the identity the authorization binds.

It does NOT carry the calibration item files: those live in the out-of-tree
artifact store, which is gitignored, so the launcher stages them over the relay.
Two paid subruns of another session were lost learning that split, one per half
— the first because nothing under `logs/` reached the pod, the second because
nothing under `artifacts/` did.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aadistill.infrastructure import bundle_transport as _bt
from aadistill.infrastructure.bundle_transport import (  # noqa: F401
    BundleTransportError, TransportSpec, build_bundle, checkout_from_bundle,
    hf_download, sha256_bytes,
)

from experiments.phase_c2 import replay as RG

REPO = Path(__file__).resolve().parents[3]

RELAY_REPO = "AlphaAvatar/aadistill-artifacts"

REPLAY_TRANSPORT = TransportSpec(
    label="Phase-C2 replay-only artifact reconstruction",
    relay_repo=RELAY_REPO,
    transfer_prefix="transfer",
)


def canonical_bundle_name(session_commit: str) -> str:
    return REPLAY_TRANSPORT.bundle_name(session_commit)


def canonical_repo_path(session_commit: str) -> str:
    return REPLAY_TRANSPORT.repo_path(session_commit)


def require_canonical_bundle_arg(bundle: str, session_commit: str) -> str:
    return REPLAY_TRANSPORT.require_canonical(bundle, session_commit)


def replay_executable_set(repo_root: str | Path = REPO
                          ) -> tuple[str, tuple[str, ...]]:
    """`(digest, files)` for the live derived REPLAY closure.

    Both halves from one derivation. An authorization that declared one file set
    and digested another would be checkable against neither.
    """
    live = RG.current_executable(repo_root)
    return live["digest"], tuple(row["path"] for row in live["files"])


def stage_bundle(session_commit: str, *, workdir: str | Path,
                 repo_root: str | Path = REPO) -> dict[str, Any]:
    """Build, upload and verify this session's bundle on the shared relay."""
    return _bt.stage_bundle(REPLAY_TRANSPORT, Path(repo_root), session_commit,
                            workdir=Path(workdir))
