"""The immutable search manifest.

One file that lets another agent, months later, say exactly what was searched,
what was produced, what was thrown away and why. P4's list applied to a search
rather than to a training run: not only the hyperparameters but the *algorithm
state* — which implementation ids at which versions, under which adapter, on
which calibration profiles, ranked by which policy hash.

Two properties are deliberate.

**Pruned states are first-class.** A manifest that recorded only the beam would
make a search that discarded the eventual winner indistinguishable from one that
never generated it. Every state that was expanded appears, with its checkpoint
hash, its metrics, its operator trace and the sentence explaining why it was
dropped — even when its weights are gone.

**The manifest is written from the run, never hand-assembled.** ``build_manifest``
takes the ``SearchResult`` and the objects that configured it, so a field cannot
drift from what actually executed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json
from .arch import ArchitectureAdapter
from .calibration import CalibrationProfile
from .operators.base import registry_ledger
from .ranking import BeamRankingPolicy, BeamSchedule, RankingResult
from .search import SearchResult
from .state import StateValidity

SCHEMA = "aadistill.autoinit.search_manifest/v1"


def build_manifest(
    result: SearchResult,
    *,
    adapter: ArchitectureAdapter,
    profiles: Sequence[CalibrationProfile],
    policy: BeamRankingPolicy,
    teacher: Mapping[str, Any],
    control: Mapping[str, Any] | None = None,
    top_n: RankingResult | None = None,
    recovery_config: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    selected_winner: str | None = None,
    notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = result.config
    states = result.states

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "config": config.as_dict(),
        "config_hash": config.config_hash,

        "teacher": dict(teacher),
        "target_architecture": {
            "spec": config.target_spec.as_dict(),
            "spec_hash": config.target_spec.spec_hash,
            "num_parameters": adapter.param_count(config.target_spec),
        },
        "adapter": adapter.identity(),
        "operator_registry": registry_ledger(),
        "calibration_profiles": [p.as_dict() for p in profiles],
        "state_eval_suite": config.suite.as_dict(),
        "beam_ranking_policy": policy.as_dict(),
        "beam_schedule": config.schedule.as_dict(),
        "stats_spec": config.stats_spec.as_dict(),
        "recovery_control": dict(control) if control else None,
        "seeds": {"search_seed": config.seed},

        "levels": [level.as_dict() for level in result.levels],
        "states": [s.as_dict() for s in states.values()],
        "state_index": {
            "expanded": [sid for sid, s in states.items()
                         if s.validity is not StateValidity.PLANNED],
            "pruned": [sid for sid, s in states.items()
                       if s.validity is StateValidity.PRUNED],
            "invalid": [sid for sid, s in states.items()
                        if s.validity is StateValidity.INVALID],
            "complete_leaves": [s.state_id for s in result.complete_leaves],
            "resumed": list(result.resumed),
        },
        "artifact_digests": {
            sid: s.artifact_digest for sid, s in states.items() if s.artifact_digest
        },
        "artifacts": {
            sid: s.artifact.as_dict() for sid, s in states.items() if s.artifact
        },
        "leaf_set": [
            {
                "state_id": s.state_id,
                "path": s.path_label,
                "impl_ids": list(s.impl_ids),
                "calibration_profiles": list(s.profile_ids),
                "provenance": s.provenance,
                "artifact_digest": s.artifact_digest,
                "single_shard_sha256": s.checkpoint_sha256,
                "n_shards": len(s.artifact.shards) if s.artifact else 0,
                "num_parameters": s.num_parameters,
                "arch_spec_hash": s.spec.spec_hash,
                "matches_target": s.is_complete_leaf(),
                "metrics": dict(s.evaluation.values) if s.evaluation else None,
            }
            for s in result.complete_leaves
        ],
        "recovery_top_n": top_n.as_dict() if top_n else None,
        "recovery_config": dict(recovery_config or {}),
        "selected_winner": selected_winner,
        "cost_accounting": dict(cost or {}),
        "environment": dict(environment or {}),
        "summary": result.summary(),
        "notes": dict(notes or {}),
    }

    # Every leaf must be exactly at the target. A manifest that recorded a
    # mismatched leaf would be recording a protocol violation as a result.
    bad = [leaf["state_id"] for leaf in manifest["leaf_set"] if not leaf["matches_target"]]
    if bad:
        raise ValueError(
            f"leaves {bad} do not match the target architecture; intermediate states "
            "cannot be recorded as recovery candidates")

    manifest["manifest_hash"] = sha256_json(
        {k: v for k, v in manifest.items() if k != "generated_utc"})
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return p


def verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive the manifest hash and re-check the leaf invariant."""
    recomputed = sha256_json(
        {k: v for k, v in manifest.items()
         if k not in ("generated_utc", "manifest_hash")})
    target_hash = manifest["target_architecture"]["spec_hash"]
    mismatched = [
        leaf["state_id"] for leaf in manifest.get("leaf_set", [])
        if leaf.get("arch_spec_hash") != target_hash
    ]
    return {
        "manifest_hash_matches": recomputed == manifest.get("manifest_hash"),
        "recomputed_hash": recomputed,
        "leaves_match_target": not mismatched,
        "mismatched_leaves": mismatched,
        "n_states": len(manifest.get("states", [])),
        "n_leaves": len(manifest.get("leaf_set", [])),
    }
