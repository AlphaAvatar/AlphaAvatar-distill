"""C1's executable closure, against THIS repository.

Split from `tests/architecture/test_closure.py` by the 2026-10-03 boundary
round. That file keeps the deriver's properties, pinned by construction against
a synthetic tree under `tmp_path`; these four ask whether C1's committed closure
snapshot still describes the live tree, which is a question about one closed
experiment's record rather than about the mechanism.

They are therefore historical verification: they go stale whenever core source
legitimately changes, and the response is to regenerate the snapshot
(`scripts/maintenance/architecture/derive_closure.py --write`) when C1 evidence needs to
describe the current tree — not to constrain the core suite with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aadistill.governance.closure import compare

REPO = Path(__file__).resolve().parents[5]


@pytest.fixture
def repo_root() -> Path:
    return REPO


class TestRealTree:
    """Facts that are only meaningful about the actual repository."""

    def test_current_digest_computes(self, repo_root):
        from stages.phase_c1.authorization import c1_current_executable
        doc = c1_current_executable(repo_root)
        assert doc["n_files"] > 40
        assert len(doc["digest"]) == 64

    def test_historical_declaration_fails_closed(self, repo_root):
        """An old authorization must not be revalidatable on the migrated tree."""
        from aadistill.governance.authorization import AuthorizationError
        from stages.phase_c1.authorization import c1_historical_harness_digest
        with pytest.raises(AuthorizationError, match="is missing"):
            c1_historical_harness_digest(repo_root)

    def test_closure_covers_the_paid_resource_path(self, repo_root):
        """Whatever can create or terminate a billed pod is part of identity.

        These three were the modules the silent walk missed.
        """
        from stages.phase_c1.authorization import c1_current_executable
        paths = {r["path"] for r in c1_current_executable(repo_root)["files"]}
        for rel in ("src/aadistill/infrastructure/provider.py",
                    "src/aadistill/infrastructure/remote.py",
                    "src/aadistill/infrastructure/log_relay.py"):
            assert rel in paths, f"{rel} can spend money and must bind"

    def test_snapshot_matches_live(self, repo_root):
        """The committed snapshot is kept current, so drift is reviewable."""
        from stages.phase_c1.authorization import (
            CURRENT_CLOSURE_SNAPSHOT, c1_current_executable)
        recorded = json.loads((repo_root / CURRENT_CLOSURE_SNAPSHOT).read_text())
        drift = compare(c1_current_executable(repo_root), recorded)
        assert drift["added_files"] == [] and drift["removed_files"] == [], (
            "re-run scripts/maintenance/architecture/derive_closure.py --write")
