"""Which files implement this project's scorer, trainer and generation protocol.

The application layer. `aadistill.initialization.planning.recovery` and
`.generation` hold the digest mechanisms and now name none of these scripts: a
reusable core can say what a source digest IS without listing one project's
files, and the contract version belongs to whoever declares the set, because two
digests over different sets are not comparable and the version is what says so.

Every path and version is transcribed without edit, so every digest is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.planning import generation as _gen  # noqa: E402
from aadistill.initialization.planning import recovery as _rec  # noqa: E402

SOURCE_SETS_CONFIG = REPO / "configs/experiments/phase_a/source_sets.json"
_DOC = json.loads(SOURCE_SETS_CONFIG.read_text())

#: The current scorer. v2 names the pre-migration paths and is the record of
#: what completed runs bound; it is deliberately not computable here.
RECOVERY_SCORING_FILES_V3: tuple[str, ...] = tuple(_DOC["recovery_scoring"]["files_v3"])
RECOVERY_SCORING_FILES_V2: tuple[str, ...] = tuple(
    _DOC["recovery_scoring"]["files_v2_historical"])
RECOVERY_SCORING_CONTRACT_ID: str = _DOC["recovery_scoring"]["contract_id"]
RECOVERY_SCORING_CONTRACT_VERSION: int = _DOC["recovery_scoring"]["version"]

TRAINER_SOURCE_FILES_V1: tuple[str, ...] = tuple(_DOC["trainer"]["files"])
TRAINER_SOURCE_SET_VERSION: int = _DOC["trainer"]["set_version"]

GENERATION_SOURCE_FILES_V1: tuple[str, ...] = tuple(_DOC["generation"]["files"])
GENERATION_SOURCE_SET_VERSION: int = _DOC["generation"]["set_version"]


def recovery_scoring_contract(repo_root=".", *, files=None, version=None,
                              contract_id=None) -> dict:
    """This project's scoring contract, over the files it declares."""
    return _rec.recovery_scoring_contract(
        repo_root,
        files=RECOVERY_SCORING_FILES_V3 if files is None else files,
        version=RECOVERY_SCORING_CONTRACT_VERSION if version is None else version,
        contract_id=contract_id or RECOVERY_SCORING_CONTRACT_ID)


def trainer_source_digest(repo_root=".", files=None) -> dict:
    return _rec.trainer_source_digest(
        repo_root,
        files=TRAINER_SOURCE_FILES_V1 if files is None else files,
        set_version=TRAINER_SOURCE_SET_VERSION)


def generation_source_digest(repo_root=".", *, files=None) -> dict:
    return _gen.generation_source_digest(
        repo_root, files=GENERATION_SOURCE_FILES_V1 if files is None else files)


__all__ = ["SOURCE_SETS_CONFIG", "RECOVERY_SCORING_FILES_V2",
           "RECOVERY_SCORING_FILES_V3", "RECOVERY_SCORING_CONTRACT_ID",
           "RECOVERY_SCORING_CONTRACT_VERSION", "TRAINER_SOURCE_FILES_V1",
           "TRAINER_SOURCE_SET_VERSION", "GENERATION_SOURCE_FILES_V1",
           "GENERATION_SOURCE_SET_VERSION", "recovery_scoring_contract",
           "trainer_source_digest", "generation_source_digest"]
