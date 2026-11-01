"""Where a student initialization came from, as its search RECORDED it -- for the lineage gate.

`aadistill.training.lineage` compares an initialization's provenance with a run's declared lineage: the root
teacher's id (``<model>@<revision>``), its shard-set hash, and the checkpoint's sha256. This reads those three facts
from a committed initialization-search record (the ``search_record.json`` the search drivers write: ``root_teacher``
and ``commit.selected_rows``) after checking the record's own sha256 against the one the lineage declares, so a
config cannot vouch for its own initialization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ProvenanceError(ValueError):
    """The record is missing, is not the declared one, or does not hold the checkpoint."""


def read(record: str | Path, record_sha256: str, checkpoint_sha256: str) -> dict:
    """{"root_teacher_id", "root_teacher_sha256", "checkpoint_sha256", "state_id"} of the selected row whose
    single-shard sha256 is ``checkpoint_sha256`` (refused when the record is absent, altered, or holds no such row)."""
    p = Path(record)
    if not p.is_file():
        raise ProvenanceError(f"the initialization's search record {p} is not here")
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record_sha256:
        raise ProvenanceError(f"{p} is not the record the lineage binds (sha256 {record_sha256[:12]})")
    doc = json.loads(raw)
    root = doc.get("root_teacher") or {}
    rows = [r for r in (doc.get("commit") or {}).get("selected_rows") or []
            if r.get("single_shard_sha256") == checkpoint_sha256]
    if not rows:
        raise ProvenanceError(f"{p} selected no checkpoint with sha256 {checkpoint_sha256[:12]}")
    return {"root_teacher_id": root.get("root_teacher_id"), "root_teacher_sha256": root.get("root_teacher_sha256"),
            "checkpoint_sha256": rows[0]["single_shard_sha256"], "state_id": rows[0].get("state_id")}
