"""Phase B's post-freeze accounting: which files the generic verifier works on.

`aadistill.governance.post_freeze` holds the mechanism -- what a declared drift
must satisfy, how a historical amendment ledger is verified, how the setup
script's `SESSION_KIND` branches are hashed -- and now knows none of the paths
below. It used to name all four, which made a generic accounting mechanism
unusable by any other phase without editing it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from aadistill.governance import post_freeze as _pf  # noqa: E402

#: Where the declaration lives. One file, so a second undeclared change cannot
#: hide behind a differently-named note.
NOTE_PATH = "logs/stages/stage-1/phase_b/analyses/autoinit_phase_b_post_freeze_changes.json"
#: The append-only ledger of reviewed, post-completion drift.
HISTORICAL_LEDGER_PATH = "logs/stages/stage-1/phase_b/analyses/autoinit_phase_b_historical_amendments.json"
#: The ledger begins where the sealed v1 note stopped.
SEALED_LEGACY_NOTE = NOTE_PATH
#: What Phase B preregistered.
PREREGISTRATION_PATH = "logs/stages/stage-1/phase_b/plans/autoinit_phase_b_preregistration.json"
#: The single dispatcher every launchable session shares.
SETUP_SCRIPT = "scripts/pod/autoinit_preflight_setup.sh"

PATHS = dict(note_path=NOTE_PATH, setup_script=SETUP_SCRIPT)
HISTORICAL_PATHS = dict(ledger_path=HISTORICAL_LEDGER_PATH,
                        sealed_note=SEALED_LEGACY_NOTE,
                        setup_script=SETUP_SCRIPT,
                        preregistration_path=PREREGISTRATION_PATH)


def dispatch_branch_hashes(repo_root=".") -> dict[str, str]:
    return _pf.dispatch_branch_hashes(repo_root, setup_script=SETUP_SCRIPT)


def accounted_for(frozen_digest: str, observed_digest: str, repo_root="."):
    return _pf.accounted_for(frozen_digest, observed_digest, repo_root, **PATHS)


def historical_accounted_for(frozen_digest: str, observed_digest: str,
                             repo_root="."):
    return _pf.historical_accounted_for(frozen_digest, observed_digest,
                                        repo_root, **HISTORICAL_PATHS)


__all__ = ["NOTE_PATH", "HISTORICAL_LEDGER_PATH", "SEALED_LEGACY_NOTE",
           "PREREGISTRATION_PATH", "SETUP_SCRIPT", "PATHS", "HISTORICAL_PATHS",
           "dispatch_branch_hashes", "accounted_for", "historical_accounted_for"]
