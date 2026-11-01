#!/usr/bin/env python3
"""The hybrid full search's state-evaluation suite (state_eval_hybrid@v1): its binding, then the shared loader. $0.

The shared loader (`shared.evaluation.load_state_eval.load`) reads any suite directory; phase_d1's
`load_state_eval_suite` binds Thinking-2507's frozen `state_eval_v1` and is not edited. This module is the hybrid's
binding: `load(root, teacher_identity, types)` refuses a suite whose manifest does not name the root teacher (model,
revision, weight shards), its tokenizer, chat template, both modes, one critical-token group per mode and the
full-search type table it was built from, or whose items file is not the manifest's -- and only then loads it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class BindingError(ValueError):
    """The suite is not this search's suite."""


#: each mode's critical-token classes, as the hybrid generates them (full_search_types.json :: critical_tokens); the
#: suite declares them as critical groups, so the two modes weigh 50:50 in state.critical_token_kl (group_balanced/v1)
CRITICAL_GROUPS = {"H-T": ["think_close", "final_answer", "eos"], "H-N": ["final_answer", "eos"]}


def binding_problems(manifest: dict, teacher_identity: dict, types_sha256: str, items_sha256: str) -> list[str]:
    bad = []
    root = manifest.get("root_teacher") or {}
    want = {"model": teacher_identity["model"], "revision": teacher_identity["revision"],
            "weight_shards_sha256": sorted(s["lfs_sha256"] for s in teacher_identity["weight_shards"]),
            "chat_template_sha256": teacher_identity["chat_template_sha256"],
            "vocabulary_sha256": teacher_identity["vocabulary_sha256"]}
    for k, v in want.items():
        got = sorted(root.get(k) or []) if k == "weight_shards_sha256" else root.get(k)
        if got != v:
            bad.append(f"root_teacher.{k} is {str(got)[:40]}, the search's root is {str(v)[:40]}")
    if sorted((manifest.get("modes") or {})) != ["H-N", "H-T"]:
        bad.append("the suite must hold both modes, H-T and H-N")
    if manifest.get("critical_groups") != CRITICAL_GROUPS:
        bad.append(f"the suite's critical groups are {manifest.get('critical_groups')}, not one group per mode "
                   f"{CRITICAL_GROUPS}: the modes would not weigh 50:50 in the critical-token KL")
    if manifest.get("types_sha256") != types_sha256:
        bad.append("the suite was not built from this type table")
    if manifest.get("items_sha256") != items_sha256:
        bad.append("items.jsonl is not the manifest's")
    if not manifest.get("content_sha256"):
        bad.append("no content_sha256")
    return bad


def load(root, teacher_identity: dict, types_path):
    from shared.evaluation.load_state_eval import load as shared_load
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    items = hashlib.sha256((root / "items.jsonl").read_bytes()).hexdigest()
    types = hashlib.sha256(Path(types_path).read_bytes()).hexdigest()
    bad = binding_problems(manifest, teacher_identity, types, items)
    if bad:
        raise BindingError("state_eval_hybrid refused: " + "; ".join(bad))
    return shared_load(root)
