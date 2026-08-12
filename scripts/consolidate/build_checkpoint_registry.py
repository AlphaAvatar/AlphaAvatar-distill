#!/usr/bin/env python3
"""Canonical inventory of every checkpoint, with a retention class for each.

    PYTHONPATH=src python scripts/consolidate/build_checkpoint_registry.py \
        --out logs/checkpoint_registry.json

Nothing is deleted here. This produces the registry that a deletion pass may later
act on, and the rule is that no file is removed before it appears in this file with a
class and a justification.

Retention classes, from the consolidation brief:

    canonical                the accepted result of a completed experiment
    control                  a baseline another accepted result is measured against
    behavioral_anchor        the current best behaviour, or a reference for it
    diagnostic               produced to answer a question, not to be promoted
    reproducibility_required needed to reproduce an accepted endpoint
    superseded               a later checkpoint replaced its role
    duplicate               byte-identical or trivially reconstructable copy
    failed_or_partial        an interrupted or OOM-killed run
    deletable               safe to remove once a tombstone records it

A checkpoint is `deletable` only when it is none of the protective classes, its
scientific status is captured elsewhere, and its hash survives in this registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# path suffix -> (experiment, role, retention, why)
CLASSIFY = {
    "stage1/qwen3_0p6b_init_v0/checkpoint": (
        "E1-E8", "positional Stage-1 initialization (FP lineage)", "canonical",
        "the pinned control initialization every E1/E4/E6/E7/E8 arm descends from; "
        "reproducing any accepted result needs it"),
    "stage1/qwen3_0p6b_init_v0/random_baseline": (
        "E1", "random-init baseline", "control",
        "the comparison that established PCA/structural init beats random"),
    "stage1/e8_contribution_init_v1": (
        "E8a", "contribution-guided fully-compressed initialization (FC)",
        "canonical", "E8a's accepted treatment initialization, hash 7a0694a5…, "
        "single-variable against the control"),
    "stage1/e8b_dc_init": (
        "E8b", "depth-only contribution initialization (DC)", "reproducibility_required",
        "deterministic function of the pinned teacher and verified bitwise against "
        "bypassed_blocks; rebuildable on any pod, so a local copy is convenience"),
    "audit/e8_baseline_init_reproduction": (
        "E8a", "reproduction check of the control initialization", "duplicate",
        "byte-identical reproduction of qwen3_0p6b_init_v0 used to prove the rebuild "
        "path; its value is the recorded hash match, not the bytes"),
    "stage3/rescued/e1_r0860k_sa_pca": (
        "E1/E6", "0.86M rung, seed sa", "behavioral_anchor",
        "the rung the frozen 150-prompt battery was sampled from"),
    "stage3/rescued/e1_r0860k_sb_pca": (
        "E1/E6", "0.86M rung, seed sb", "behavioral_anchor",
        "second seed of the anchor rung"),
    "stage3/rescued/e1_r2960k_sb_pca": (
        "E1/E6", "2.96M rung, seed sb", "canonical",
        "part of the accepted E1 scale curve"),
    "stage3/rescued/e1_r5500k_sb_pca": (
        "E1/E6", "5.50M rung, seed sb", "canonical",
        "highest accepted rung of the E1 scale curve"),
    "stage3/rescued/e1_r2960k_sb_rand": (
        "E1", "2.96M rung from random init, seed sb", "control",
        "the random-init arm of the scale comparison"),
    "stage3/rescued/e1_r5500k_sb_rand": (
        "E1", "5.50M rung from random init, seed sb", "control",
        "the random-init arm at the highest rung"),
    "stage3/rescued/e1_ctl_r0250k_sa_pca_stepmatched": (
        "E1", "0.25M step-matched control", "diagnostic",
        "a step-matching control for the rung comparison; its conclusion is recorded "
        "in EXPERIMENTS.md and does not depend on the weights"),
}
PROTECTED = {"canonical", "control", "behavioral_anchor", "reproducibility_required"}


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def classify(rel: str) -> tuple[str, str, str, str]:
    for suffix, meta in CLASSIFY.items():
        if rel.endswith(suffix) or suffix in rel:
            return meta
    return ("unknown", "unclassified", "diagnostic",
            "not matched by the classification table — review before any deletion")


def referenced_by(rel: str, name: str) -> list[str]:
    """Which tracked documents or tests mention this checkpoint."""
    hits = []
    for d in ("logs", "configs", "tests", "scripts", "README.md"):
        base = REPO_ROOT / d
        files = [base] if base.is_file() else list(base.rglob("*"))
        for f in files:
            if not f.is_file() or f.suffix not in (".md", ".json", ".py", ".txt"):
                continue
            try:
                if name in f.read_text(errors="ignore"):
                    hits.append(str(f.relative_to(REPO_ROOT)))
            except Exception:
                continue
    return sorted(set(hits))[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/checkpoint_registry.json")
    ap.add_argument("--hash", action="store_true",
                    help="compute weight sha256 (slow; the registry is not "
                         "authoritative for deletion without it)")
    args = ap.parse_args()

    entries = []
    for w in sorted((REPO_ROOT / "artifacts").rglob("model.safetensors")):
        d = w.parent
        rel = str(d.relative_to(REPO_ROOT))
        exp, role, retention, why = classify(rel)
        cfg = d / "config.json"
        arch = {}
        if cfg.is_file():
            c = json.loads(cfg.read_text())
            arch = {k: c.get(k) for k in
                    ("num_hidden_layers", "hidden_size", "intermediate_size",
                     "num_attention_heads", "num_key_value_heads", "vocab_size")}
        name = d.name if d.name != "checkpoint" else d.parent.name
        e = {
            "canonical_name": name,
            "path_local": rel,
            "experiment": exp, "role": role,
            "retention": retention, "retention_reason": why,
            "architecture": arch,
            "size_bytes": dir_size(d),
            "size_human": f"{dir_size(d) / 2**30:.2f} GiB",
            "weights_sha256": sha256_file(w) if args.hash else None,
            "config_sha256": (hashlib.sha256(cfg.read_bytes()).hexdigest()
                              if cfg.is_file() else None),
            "referenced_by": referenced_by(rel, name),
            "protected": retention in PROTECTED,
        }
        entries.append(e)

    total = sum(e["size_bytes"] for e in entries)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "retention_classes": sorted({e["retention"] for e in entries}),
        "protected_classes": sorted(PROTECTED),
        "n_checkpoints": len(entries),
        "total_size_human": f"{total / 2**30:.2f} GiB",
        "deletion_rule": "a checkpoint may be deleted only if it is not protected, "
                         "its scientific status is recorded elsewhere, its hash is in "
                         "this registry, and a tombstone is written",
        "checkpoints": entries,
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(entries)} checkpoints, {report['total_size_human']}")
    for e in entries:
        print(f"  {e['retention']:24s} {e['size_human']:>10s}  {e['path_local']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
