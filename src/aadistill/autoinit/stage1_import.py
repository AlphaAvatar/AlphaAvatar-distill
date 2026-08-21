"""Import a *completed, verified* Phase-A Stage-1 result. Nothing weaker.

Attempts 11 and 12 independently produced the same Stage-1 search — identical
config hash, identical 43 states and 7 complete leaves, identical five selected
state ids in order, identical first depth decision — and attempt 12 preserved the
five selected checkpoints off-pod with re-verified artifact and shard digests.
Recomputing them a third time would cost 203 minutes and add no scientific
information. So a recovery continuation should *import* that result and begin at
Stage 2.

**This is not a deserializer, and must not become one.** There is deliberately no
`InitializationState.from_dict`: the append-only journal is evidence, not a
trusted serialization format, and a permissive loader would turn every recorded
line into something a later session could be talked into believing. What this
module does instead is **re-derive** the parts that can be re-derived and refuse
anything that does not match:

* the search's `config_hash` must equal the one being imported;
* the selected state ids must match **exactly, and in order** — a reordering is a
  different ranking, not a cosmetic difference;
* every checkpoint is re-identified **from local bytes**, and both its
  `artifact_digest` and its `single_shard_sha256` must equal the Stage-1 record;
* every state's geometry must equal the target geometry, so `is_complete_leaf`
  is true for the reason it is supposed to be true and not because a field was
  copied in;
* every state must carry a Stage-1 evaluation whose `artifact_digest` is its own
  — the same binding `attach_evaluation` enforces during a live search, applied
  to the imported record;
* the control is rebuilt **independently** from its frozen checkpoint and hash,
  never from the journal.

**The control comes back UNMEASURED, and that is not an oversight.** Its Stage-1
evaluation is not in the persisted evidence — the journal has no
`retained_canonical` row and `search_result["control"]` carries identity only —
because in a live search the control is measured on the GPU alongside everything
else. So a continuation must measure it itself, on the same suite, before
`admit_leaves` will accept it. That is one evaluation against a 203-minute
search, and it is the honest price: a control whose step-0 metrics were invented
rather than measured is not comparable to the leaves it is the baseline for.

The reconstructed states carry only what a recovery candidate needs. They are not
resumable search nodes: the operator steps are recorded for provenance, and
nothing here will let a caller continue searching from them.

Everything this module produces still goes through `admit_leaves`, which is the
same gate a live search's output passes. Importing does not bypass it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arch import ArchSpec
from .artifact import CheckpointIdentity
from .leaf_durability import verify_transferred_leaf
from .state import (
    InitializationState, StateEvaluation, StateValidity, make_control_state,
)

__all__ = ["Stage1ImportError", "ImportedStage1", "import_stage1_result"]


class Stage1ImportError(RuntimeError):
    """The persisted Stage-1 result could not be verified. Do not proceed."""


class ImportedStage1:
    """A verified Stage-1 result: the leaves, the control, and its evidence."""

    def __init__(self, *, leaves: list, control: Any, summary: dict,
                 verification: dict) -> None:
        self.leaves = leaves
        self.control = control
        self.summary = summary
        self.verification = verification


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise Stage1ImportError(message)


def _state_records(states_path: Path, wanted: Sequence[str]) -> dict[str, dict]:
    """The journal lines for exactly the wanted ids, and no others."""
    found: dict[str, dict] = {}
    for line in states_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("state_id")
        if sid in wanted:
            _require(sid not in found,
                     f"{sid} appears twice in {states_path}; the journal is "
                     "append-only and a duplicated state id means two different "
                     "records claim the same identity")
            found[sid] = row
    missing = [s for s in wanted if s not in found]
    _require(not missing,
             f"the state journal {states_path} has no record for {missing}; the "
             "selected leaves cannot be reconstructed from evidence that does "
             "not contain them")
    return found


def import_stage1_result(
    *,
    search_result: Mapping[str, Any],
    states_path: str | Path,
    checkpoint_store: str | Path,
    durability: Mapping[str, Any],
    adapter: Any,
    expected_config_hash: str,
    target_geometry: Mapping[str, Any],
    control_dir: str | Path,
    control_sha256: str | None,
    control_id: str = "qwen3_0p6b_init_v0",
) -> ImportedStage1:
    """Rebuild the five selected leaves and the control, or refuse.

    `search_result` and `states_path` are committed evidence; `durability` is the
    Stage-1 record of what each leaf hashed to; `checkpoint_store` holds the
    bytes. The bytes are what decide — the records only say what the bytes must
    turn out to be.
    """
    store = Path(checkpoint_store)
    states_path = Path(states_path)

    # --- 1. the search identity -------------------------------------------
    got = search_result.get("config_hash")
    _require(got == expected_config_hash,
             f"the persisted search declares config_hash {str(got)[:16]}… but "
             f"this continuation is bound to {expected_config_hash[:16]}…; a "
             "different search is a different scientific result")

    selected = [s["state_id"] for s in search_result["top_n"]["selected"]]
    recorded = [rec["state_id"] for rec in durability["leaves"]]
    _require(selected == recorded,
             f"the durability record lists {recorded} but the search selected "
             f"{selected}. Exact order matters: the ranking IS the result, and a "
             "reordering is a different one.")
    _require(len(selected) == len(set(selected)),
             f"duplicate state ids among the selected leaves: {selected}")

    rows = _state_records(states_path, selected)
    target = ArchSpec.of("qwen3", dict(target_geometry))

    # --- 2. every leaf, re-derived from its own bytes ----------------------
    leaves, evidence = [], []
    for rec in durability["leaves"]:
        sid = rec["state_id"]
        row = rows[sid]
        leaf_dir = store / sid
        _require(leaf_dir.is_dir(),
                 f"{sid}: no checkpoint at {leaf_dir}. The Stage-1 result cannot "
                 "be imported from a record whose weights are absent.")

        # The bytes decide. `verify_transferred_leaf` recomputes shard hashes,
        # the config hash, the index and the tokenizer digest locally, taking
        # only `arch_signature` and `num_parameters` from the record because no
        # file carries them.
        try:
            v = verify_transferred_leaf(leaf_dir, rec, adapter=adapter)
        except Exception as exc:                       # noqa: BLE001
            # One failure type at this boundary. A caller deciding whether to
            # start a paid continuation should catch one thing, not guess which
            # layer refused.
            raise Stage1ImportError(
                f"{sid}: the persisted checkpoint could not be verified: "
                f"{type(exc).__name__}: {exc}") from exc
        _require(v["matched"],
                 f"{sid}: the local checkpoint identifies as "
                 f"{v['artifact_digest'][:16]}… but Stage 1 recorded "
                 f"{rec['artifact_digest'][:16]}…")
        _require(v["shard_matched"],
                 f"{sid}: the local shard hashes to {v['single_shard_sha256']} "
                 f"but Stage 1 recorded {rec.get('single_shard_sha256')}")

        spec = ArchSpec.of("qwen3", row["arch_spec"])
        _require(spec.spec_hash == row["arch_spec_hash"],
                 f"{sid}: the recorded arch_spec does not hash to its own "
                 "recorded arch_spec_hash; the journal line is inconsistent")
        _require(spec.spec_hash == target.spec_hash,
                 f"{sid}: geometry {spec.describe()} is not the target "
                 f"{target.describe()}. A recovery candidate must BE the target "
                 "size; an intermediate would rank well and deploy never.")

        art = CheckpointIdentity.from_dict({**row["artifact"], "path": str(leaf_dir)})
        _require(art.artifact_digest == v["artifact_digest"],
                 f"{sid}: the journal's artifact record and the bytes on disk "
                 "disagree about the artifact digest")

        ev = row.get("evaluation")
        _require(isinstance(ev, dict) and ev,
                 f"{sid}: the journal carries no Stage-1 evaluation; a recovery "
                 "candidate must arrive with its own hash-bound measurement, not "
                 "acquire one later")
        _require(ev.get("artifact_digest") == art.artifact_digest,
                 f"{sid}: the recorded evaluation was measured on "
                 f"{str(ev.get('artifact_digest'))[:16]}… but this artifact "
                 f"digests to {art.artifact_digest[:16]}…. Metrics bind to "
                 "artifacts; importing does not relax that.")

        # Constructed field by field from re-derived values. Deliberately not a
        # `from_dict`: every argument below is either recomputed from bytes
        # (`spec`, `art`) or a scalar whose consistency has just been checked.
        state = InitializationState(
            state_id=sid,
            parent_id=row.get("parent_id"),
            root_teacher_id=row["root_teacher_id"],
            root_teacher_sha256=row["root_teacher_sha256"],
            spec=spec, target_spec=target,
            steps=(),                      # provenance only; not resumable
            num_parameters=int(row["num_parameters"]),
            depth=int(row.get("depth", 0)),
            seed=row.get("seed"),
            checkpoint_path=str(leaf_dir),
            artifact=art,
            validity=StateValidity.VALIDATED,
            provenance=row.get("provenance", "search"),
            notes={**dict(row.get("notes") or {}),
                   "imported": "phase-a stage-1, re-verified from bytes",
                   "path_label": row.get("path_label", "")},
        )
        state.attach_evaluation(StateEvaluation(
            artifact_digest=ev["artifact_digest"],
            suite_id=ev["suite_id"], suite_hash=ev["suite_hash"],
            reference=ev["reference"], values=dict(ev["values"]),
            positions=int(ev["positions"]), detail=dict(ev.get("detail") or {}),
            measured_utc=ev.get("measured_utc"),
            runtime=dict(ev.get("runtime") or {})))
        state.validity = StateValidity.MEASURED
        state.require_recovery_admissible()

        leaves.append(state)
        evidence.append({
            "state_id": sid, "path": str(leaf_dir),
            "artifact_digest": art.artifact_digest,
            "single_shard_sha256": v["single_shard_sha256"],
            "arch_spec_hash": spec.spec_hash,
            "path_label": row.get("path_label", ""),
            "verified_from_bytes": True,
        })

    _require([s.state_id for s in leaves] == selected,
             "the reconstructed leaves are not in the selected order")

    # --- 3. the control, rebuilt independently ------------------------------
    from .artifact import identify_checkpoint

    init_dir = Path(control_dir)
    _require(init_dir.is_dir(), f"no canonical control at {init_dir}")
    from transformers import AutoConfig

    control_spec = adapter.spec_from_config(AutoConfig.from_pretrained(str(init_dir)))
    control_artifact = identify_checkpoint(
        init_dir, adapter=adapter, spec=control_spec,
        num_parameters=adapter.param_count(control_spec))
    control = make_control_state(
        control_id=control_id, artifact=control_artifact, spec=control_spec,
        target_spec=target, num_parameters=adapter.param_count(control_spec),
        root_teacher_id=leaves[0].root_teacher_id,
        root_teacher_sha256=leaves[0].root_teacher_sha256,
        description=("the retained canonical initialization; rebuilt from its "
                     "frozen checkpoint, never from the journal"),
        expected_single_file_sha256=control_sha256)

    # UNMEASURED by construction. `require_recovery_admissible` will refuse it
    # until the caller measures it on the suite, which is the gate working: the
    # control's step-0 metrics have to be measured, not imported from a record
    # that does not contain them.
    return ImportedStage1(
        leaves=leaves, control=control, summary=dict(search_result),
        verification={
            "schema": "aadistill.autoinit.stage1_import/v1",
            "config_hash": expected_config_hash,
            "selected_in_order": selected,
            "n_leaves": len(leaves),
            "target_spec_hash": target.spec_hash,
            "control_id": control_id,
            "control_single_shard_sha256": control_artifact.single_shard_sha256,
            "leaves": evidence,
            "control_is_unmeasured": True,
            "note": ("every leaf re-identified from local bytes and required to "
                     "match the Stage-1 artifact and shard digests; the control "
                     "rebuilt from its frozen checkpoint and returned UNMEASURED "
                     "because its evaluation is not in the evidence; nothing "
                     "reranked"),
        })
