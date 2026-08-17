#!/usr/bin/env python3
"""Measure the four storage areas separately, and append the reading to the log.

    PYTHONPATH=src python scripts/consolidate/measure_storage.py \
        --label before-cleanup --note "baseline, nothing deleted yet"

Four areas, because a single total hides what actually changed:

    repository working tree   what git tracks, plus the working copy around it
                              and the .git directory, reported separately
    local artifact storage    artifacts/ **and /home/ecs-user/aad-artifacts** —
                              generated, gitignored, refetchable. The second is
                              out of tree and holds most of the bytes; a report
                              that measures only the first understates local
                              storage by more than an order of magnitude
    relay / LFS storage       AlphaAvatar/aadistill-artifacts, read from the
                              registry's cached listing (--relay to re-query)
    scratch / session dirs    /home/ecs-user/aad-scratch — per-session working
                              directories, bundles, poller output, quarantine

Readings append to logs/storage_measurements.json. A before/after pair with the
same definitions is the only honest way to say what a cleanup reclaimed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = Path("/home/ecs-user/aad-scratch")
EXTERNAL_ARTIFACTS = Path("/home/ecs-user/aad-artifacts")
RELAY_REPO = "AlphaAvatar/aadistill-artifacts"


def tree_bytes(root: Path, skip: tuple[str, ...] = ()) -> tuple[int, int]:
    """(bytes, files) under root, skipping any path containing a skipped part."""
    total = files = 0
    if not root.exists():
        return 0, 0
    for p in root.rglob("*"):
        if any(s in p.parts for s in skip):
            continue
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
                files += 1
            except OSError:
                continue
    return total, files


def git_tracked_bytes() -> tuple[int, int]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=False).stdout.splitlines()
    total = files = 0
    for rel in out:
        p = REPO_ROOT / rel
        if p.is_file():
            total += p.stat().st_size
            files += 1
    return total, files


def relay_bytes(requery: bool) -> dict:
    registry = REPO_ROOT / "logs/checkpoint_registry.json"
    if not requery and registry.is_file():
        relay = json.loads(registry.read_text()).get("relay", {})
        if relay.get("files"):
            return {"repo": RELAY_REPO,
                    "bytes": sum(f["size_bytes"] for f in relay["files"]),
                    "files": len(relay["files"]),
                    "source": "logs/checkpoint_registry.json (cached listing)",
                    "queried_utc": relay.get("queried_utc")}
    from huggingface_hub import HfApi                      # noqa: PLC0415
    info = HfApi().repo_info(RELAY_REPO, repo_type="model", files_metadata=True)
    sizes = [(f.lfs.size if f.lfs else f.size) or 0 for f in (info.siblings or [])]
    return {"repo": RELAY_REPO, "bytes": sum(sizes), "files": len(sizes),
            "source": "huggingface_hub repo_info",
            "queried_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def measure(requery_relay: bool) -> dict:
    tracked_b, tracked_n = git_tracked_bytes()
    # The working tree as a reader sees it: no build/venv/vcs internals, and
    # artifacts/ counted separately because it is the generated area.
    work_b, work_n = tree_bytes(REPO_ROOT,
                                skip=(".git", ".venv", "artifacts", "__pycache__",
                                      ".pytest_cache", ".ruff_cache"))
    git_b, _ = tree_bytes(REPO_ROOT / ".git")
    art_b, art_n = tree_bytes(REPO_ROOT / "artifacts")
    ext_b, ext_n = tree_bytes(EXTERNAL_ARTIFACTS)
    logs_b, logs_n = tree_bytes(REPO_ROOT / "logs")
    data_b, data_n = tree_bytes(REPO_ROOT / "data")
    scratch_b, scratch_n = tree_bytes(SCRATCH_ROOT)
    return {
        "repository_working_tree": {
            "definition": "the tree excluding .git, .venv, artifacts/ and caches",
            "bytes": work_b, "files": work_n,
            "of_which_logs_bytes": logs_b, "of_which_logs_files": logs_n,
            "of_which_data_bytes": data_b, "of_which_data_files": data_n,
            "git_tracked_bytes": tracked_b, "git_tracked_files": tracked_n,
            "git_dir_bytes": git_b,
        },
        "local_artifact_storage": {
            "definition": "artifacts/ in the tree PLUS the out-of-tree store, "
                          "both generated and gitignored",
            "bytes": art_b + ext_b, "files": art_n + ext_n,
            "in_tree": {"path": "artifacts", "bytes": art_b, "files": art_n},
            "out_of_tree": {"path": str(EXTERNAL_ARTIFACTS),
                            "bytes": ext_b, "files": ext_n},
        },
        "relay_lfs_storage": relay_bytes(requery_relay),
        "scratch_session_directories": {
            "definition": "per-session working directories, bundles, poller "
                          "output and pod-simulator quarantine",
            "path": str(SCRATCH_ROOT), "bytes": scratch_b, "files": scratch_n,
        },
    }


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return str(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True,
                    help="e.g. before-cleanup / after-cleanup")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="logs/storage_measurements.json")
    ap.add_argument("--relay", action="store_true",
                    help="re-query the artifact store instead of reading the "
                         "registry's cached listing")
    args = ap.parse_args()

    reading = {
        "label": args.label,
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                 capture_output=True, text=True,
                                 check=False).stdout.strip(),
        "note": args.note,
        "areas": measure(args.relay),
    }

    out = REPO_ROOT / args.out
    doc = (json.loads(out.read_text()) if out.is_file() else
           {"schema": "aadistill.storage_measurements/v1",
            "rule": "before and after are measured with the same definitions, in "
                    "the same four areas, or the difference means nothing",
            "readings": []})
    doc["readings"] = [r for r in doc["readings"] if r["label"] != args.label]
    doc["readings"].append(reading)
    doc["readings"].sort(key=lambda r: r["measured_utc"])
    out.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"[{args.label}] {reading['commit'][:12]}")
    for area, v in reading["areas"].items():
        print(f"  {area:32s} {human(v['bytes']):>12s}  "
              f"({v.get('files', '?')} files)")
    if len(doc["readings"]) > 1:
        first, last = doc["readings"][0], doc["readings"][-1]
        print(f"\n  delta {first['label']} -> {last['label']}:")
        for area in last["areas"]:
            d = last["areas"][area]["bytes"] - first["areas"][area]["bytes"]
            print(f"    {area:32s} {human(d):>12s}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
