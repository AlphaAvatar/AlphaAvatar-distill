#!/usr/bin/env python
"""Inventory every model checkpoint this project holds, and classify each one.

Covers both stores: the development machine and every Hugging Face repository
the project uses. For each checkpoint it records where it lives, which run made
it, what it contains, its size and hash, whether a second hash-verified copy
exists, what still needs it, and a proposed action.

The classification is **declared here, not inferred**. `REQUIRED` names the
checkpoints Experiment 2 or the historical record depends on and why; everything
else is judged against a short list of provable-obsolescence rules. Anything
that matches neither is reported as `decide` — the instruction is explicit that
an ambiguous role means retain and flag, not delete.

Deleting nothing is the default. `--apply-local` acts only on entries whose
action is `delete` **and** whose store is the dev box; Hugging Face deletions are
never performed here, because on a repo at its LFS limit removing a file from the
current revision reclaims no quota (the object stays referenced by history) and
the operations that would reclaim it are exactly the ones that must not run
unattended.

Usage:
    scripts/pod/checkpoint_inventory.py --out artifacts/stage3/checkpoint_inventory.json
    scripts/pod/checkpoint_inventory.py --out … --apply-local
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

RELAY = "AlphaAvatar/aadistill-artifacts"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth")

# --------------------------------------------------------------------------
# What must survive, and why. Keys are substrings matched against the entry id.
# --------------------------------------------------------------------------
REQUIRED = {
    "stage1/qwen3_0p6b_init_v0/checkpoint": (
        "the pinned Stage 1 PCA/sandwich fork point (sha256 86fbba78…); every "
        "Experiment 2 arm starts from it"),
    "stage1/qwen3_0p6b_init_v0/random_baseline": (
        "Experiment 1's second initialization axis; the init-dominates finding "
        "is not reproducible without it"),
    "e1_r0860k_sa_pca": (
        "Experiment 2's D0 control, seed 20260726 — the fixed-step endpoint the "
        "capability battery scores"),
    "e1_r0860k_sb_pca": (
        "Experiment 2's D0 control, seed 20260801"),
    "e1_ctl_r0250k_sa_pca_stepmatched": (
        "the step-matched compute control; the only evidence that held-out NLL "
        "tracks optimizer steps rather than unique data"),
}

# Provable obsolescence. Everything here must be defensible without judgement
# about future research value.
OBSOLETE_RULES = {
    "_smoke_ladder": (
        "two-step CPU smoke test of the ladder loader, superseded by 24 real "
        "Experiment 1 arms on the same code path; its manifest and train_log "
        "are retained and the weights cannot enter any comparison"),
}


def classify(entry_id: str, has_second_copy: bool) -> tuple[str, str]:
    for key, why in REQUIRED.items():
        if key in entry_id:
            return "retain", why
    for key, why in OBSOLETE_RULES.items():
        if key in entry_id:
            return "delete", why
    if "trainer_state" in entry_id:
        return ("delete" if any(k in entry_id for k in OBSOLETE_RULES)
                else "decide",
                "optimizer state from a completed run that will not be resumed")
    return "decide", (
        "historical Experiment 1 / Stage 3 weights: not required by the "
        "D0→D1→L1→R1/R2 chain and not scored by the frozen battery, but their "
        "diagnostic, reproducibility and future-control value is not proven "
        "zero — retained and flagged per the standing instruction")


def scan_local(root: Path, do_hash: bool) -> list[dict]:
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in WEIGHT_SUFFIXES:
            continue
        rel = path.relative_to(REPO_ROOT)
        out.append({
            "store": "devbox",
            "id": str(rel),
            "path": str(path),
            "contents": ("optimizer_state" if path.suffix in (".pt", ".pth")
                         else "model_weights"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path) if do_hash else None,
        })
    return out


def scan_relay(repo: str) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo, repo_type="model", files_metadata=True)
    out = []
    for sib in info.siblings:
        if not sib.rfilename.endswith(WEIGHT_SUFFIXES):
            continue
        out.append({
            "store": "huggingface",
            "repo": repo,
            "revision": info.sha,
            "id": f"{repo}:{sib.rfilename}",
            "path": sib.rfilename,
            "contents": ("optimizer_state"
                         if sib.rfilename.endswith((".pt", ".pth"))
                         else "model_weights"),
            "bytes": sib.size,
            # LFS records the object's own sha256; that is what a second copy is
            # compared against, and it is why a dev-box copy can be verified
            # without downloading the relay object.
            "sha256": sib.lfs.sha256 if sib.lfs else None,
        })
    return out


def parse_arm(entry: dict) -> dict:
    """Best-effort (experiment, arm, rung, seed, step) from the path."""
    parts = Path(entry["path"]).parts
    arm = next((p for p in parts if p.startswith(("e1_", "e2_", "s1_", "s2", "kdconf",
                                                  "tt", "qwen3_"))), None)
    step = next((p for p in parts if p.startswith("step_")), None)
    rung = seed = experiment = None
    if arm and arm.startswith("e1_r"):
        experiment = "experiment_1"
        body = arm[4:]
        rung = body.split("_")[0]
        seed = next((s for s in ("sa", "sb") if f"_{s}_" in arm), None)
    elif arm and arm.startswith("e1_ctl"):
        experiment = "experiment_1_control"
    elif arm and arm.startswith("e2_"):
        experiment = "experiment_2"
    elif arm:
        experiment = "stage1" if arm.startswith("qwen3_") else "stage3_pre_e1"
    return {"experiment": experiment, "arm": arm, "rung": rung, "seed": seed,
            "step": int(step.split("_")[1]) if step else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--repo", default=RELAY)
    ap.add_argument("--no-relay", action="store_true")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip local hashing (fast; loses duplicate detection)")
    ap.add_argument("--apply-local", action="store_true",
                    help="delete dev-box entries whose action is 'delete'")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    entries = scan_local(args.artifacts, not args.no_hash)
    if not args.no_relay:
        entries += scan_relay(args.repo)

    by_hash: dict[str, list[str]] = {}
    for e in entries:
        if e["sha256"]:
            by_hash.setdefault(e["sha256"], []).append(e["id"])

    for e in entries:
        copies = by_hash.get(e["sha256"] or "", [])
        e["copies"] = copies
        e["has_second_copy"] = len(copies) > 1
        e.update(parse_arm(e))
        action, why = classify(e["id"], e["has_second_copy"])
        e["action"], e["rationale"] = action, why

    df = shutil.disk_usage(REPO_ROOT)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "devbox_disk": {"total_bytes": df.total, "used_bytes": df.used,
                        "free_bytes": df.free},
        "totals": {},
        "entries": sorted(entries, key=lambda e: (e["store"], e["id"])),
    }
    for store in ("devbox", "huggingface"):
        rows = [e for e in entries if e["store"] == store]
        report["totals"][store] = {
            "files": len(rows),
            "bytes": sum(e["bytes"] or 0 for e in rows),
            "retain": sum(1 for e in rows if e["action"] == "retain"),
            "delete": sum(1 for e in rows if e["action"] == "delete"),
            "decide": sum(1 for e in rows if e["action"] == "decide"),
            "delete_bytes": sum(e["bytes"] or 0 for e in rows
                                if e["action"] == "delete"),
        }

    deleted = []
    if args.apply_local:
        for e in entries:
            if e["store"] != "devbox" or e["action"] != "delete":
                continue
            path = Path(e["path"])
            if path.is_file():
                os.remove(path)
                deleted.append({"id": e["id"], "bytes": e["bytes"],
                                "sha256": e["sha256"],
                                "rationale": e["rationale"]})
        df2 = shutil.disk_usage(REPO_ROOT)
        report["devbox_disk_after"] = {"total_bytes": df2.total,
                                       "used_bytes": df2.used,
                                       "free_bytes": df2.free}
    report["deleted"] = deleted
    report["huggingface_note"] = (
        "No Hugging Face deletion is performed by this tool. The relay is at its "
        "private-LFS limit, and removing a file from the current revision does "
        "not reclaim quota there — the object stays referenced by history "
        "(measured 2026-08-02: deleting 19.07 GB of superseded weights dropped "
        "the tree to 80.31 GB and reclaimed nothing). The operations that would "
        "reclaim it — history rewriting, LFS purging, repo deletion — are out of "
        "scope and need explicit approval.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print(f"{'store':12s} {'files':>6} {'GiB':>8} {'retain':>7} {'delete':>7} {'decide':>7}")
    for store, t in report["totals"].items():
        print(f"{store:12s} {t['files']:>6} {t['bytes'] / 2**30:>8.2f} "
              f"{t['retain']:>7} {t['delete']:>7} {t['decide']:>7}")
    if deleted:
        freed = sum(d["bytes"] for d in deleted)
        print(f"\ndeleted {len(deleted)} dev-box files, {freed / 2**30:.2f} GiB")
        for d in deleted:
            print(f"  {d['bytes'] / 2**30:6.2f} GiB  {d['id']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
