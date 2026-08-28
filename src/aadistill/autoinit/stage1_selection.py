"""Make a completed Stage-1 search durable before anything else can fail.

Phase-B attempt 4 paid for and **completed** an eight-hour joint P=2 search. It
measured the canonical control, measured both imported finalists, and computed
its Top-N ranking. Then a local-name collision in the summary dict raised, the
stage failed, and the session ended with no authoritative record of which five
leaves had been selected — because the only place that record was ever going to
appear was the summary that never got built.

The search journal survived and is real audit evidence, but it is not a
selection: the existing restore contract requires actual checkpoint bytes with a
re-derived artifact identity, and a ranking reconstructed post hoc from a journal
is not that. So the science was complete and unusable at the same time.

This module closes that window. The moment a ranking exists it is committed to a
small, atomic, hash-bound artifact, before the control measurement, the retained
candidates, the summary, or any other bookkeeping that is **not required to
establish the search result**. Everything after that point may fail without
losing which five leaves won and where their bytes are.

Deliberately minimal. It is not a second search-result format and it does not
replace the summary; it records exactly what a later stage — or a failed-run
collector — needs to identify and secure the selection.

`generated_utc` is recorded and **excluded from the commitment hash**, for the
reason the preregistration excludes it: provenance is not commitment, and a
timestamp inside the identity would make the same selection hash differently on
every regeneration.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json

SCHEMA = "aadistill.autoinit.stage1_selection/v1"

#: The filename, beside the search journal in the search workdir. Both are
#: collected by the same artifact spec, so a run that produces one produces the
#: other on the same path.
FILENAME = "stage1_selection.json"

#: Excluded from the commitment hash. Provenance, not commitment.
_UNCOMMITTED = ("selection_sha256", "generated_utc")


def journal_sha256(path: str | Path) -> str:
    """Hash of the search journal this selection was drawn from."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Every field `leaf_durability.verify_transferred_leaf` reads off the record.
#: Named here so the writer and its consumer cannot drift apart silently again.
TRANSFER_VERIFICATION_FIELDS = ("artifact_digest", "arch_signature",
                                "num_parameters", "weights_digest",
                                "single_shard_sha256")


def build(*, search_config, ranking, suite, policy, profiles,
          journal_path: str | Path) -> dict[str, Any]:
    """The record. Pure — it computes, it does not write."""
    selected = [
        {
            "state_id": state.state_id,
            "path": state.path_label,
            "artifact_digest": state.artifact_digest,
            "single_shard_sha256": state.checkpoint_sha256,
            "checkpoint_path": state.checkpoint_path,
            "num_parameters": state.num_parameters,
            "impl_ids": [step.impl_id for step in state.steps],
            "calibration_profiles": sorted({step.profile_id for step in state.steps}),
            # REQUIRED BY `verify_transferred_leaf`, which rebuilds the identity
            # from the bytes that arrived and takes these two from the record
            # because no file carries them. Attempt 5 omitted `arch_signature`,
            # so every one of five transfers reported NOT MATCHED on a KeyError
            # while the bytes were in fact correct — a secured gate that cried
            # wolf at exactly the moment it must be believed.
            "arch_signature": state.artifact.arch_signature if state.artifact else None,
            "weights_digest": state.artifact.weights_digest if state.artifact else None,
        }
        for state in ranking.selected
    ]
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": (
            "Written the moment a Stage-1 ranking exists, before the control "
            "measurement, the retained candidates, the summary, or any other "
            "bookkeeping. If this file exists the search COMPLETED and these five "
            "leaves were selected, whatever happened afterwards. It is not a "
            "search result and not a substitute for the summary."),
        "search": {
            "run_id": search_config.run_id,
            "config_hash": search_config.config_hash,
            "seed": search_config.seed,
            "target_spec_hash": search_config.target_spec.spec_hash,
            "workdir": str(search_config.workdir),
        },
        "journal": {
            "path": str(journal_path),
            "sha256": journal_sha256(journal_path),
        },
        "policy": {"qualified_id": policy.qualified_id, "hash": policy.policy_hash},
        "suite": {"qualified_id": suite.qualified_id, "hash": suite.suite_hash},
        "profiles": [
            {"qualified_id": p.qualified_id, "profile_hash": p.profile_hash,
             "content_sha256": p.content_sha256}
            for p in profiles
        ],
        "selected": selected,
        "n_selected": len(selected),
        # Why these five and not the others. Without it the file records an
        # outcome and not a decision, and a later stage could not defend it.
        "decisions": list(ranking.decisions),
    }
    body["selection_sha256"] = sha256_json(
        {k: v for k, v in body.items() if k not in _UNCOMMITTED})
    return body


def write(record: dict[str, Any], directory: str | Path) -> Path:
    """Atomically, so a crash mid-write cannot leave a half-file that parses.

    `os.replace` is atomic within a filesystem, and the temporary file is created
    in the destination directory precisely so it is the same filesystem.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / FILENAME
    tmp = directory / f".{FILENAME}.partial"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, final)
    return final


def load(path: str | Path) -> dict[str, Any]:
    """Read and verify. A selection that fails its own hash is not a selection."""
    record = json.loads(Path(path).read_text())
    stated = record.get("selection_sha256")
    recomputed = sha256_json({k: v for k, v in record.items() if k not in _UNCOMMITTED})
    if stated != recomputed:
        raise ValueError(
            f"{path} does not match its own selection_sha256; it has been edited "
            "since it was written")
    return record


def commit(*, search_config, ranking, suite, policy, profiles,
           journal_path: str | Path, directory: str | Path) -> Path:
    """Build and write in one call, which is how callers should use this."""
    return write(build(search_config=search_config, ranking=ranking, suite=suite,
                       policy=policy, profiles=profiles, journal_path=journal_path),
                 directory)
