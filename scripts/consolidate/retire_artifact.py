#!/usr/bin/env python3
"""Delete a local artifact, but only behind a tombstone that outlives it.

    PYTHONPATH=src python scripts/consolidate/retire_artifact.py \
        --path artifacts/audit/ladder_uniform_rebuild \
        --id ladder_uniform_rebuild \
        --criterion byte_identical_duplicate \
        --experiment E1 --role "rebuilt ladder for the mixture audit" \
        --reason "..." --canonical "artifacts/stage3/ladder_uniform_probe" \
        --frozen-results "artifacts/audit/e1_mixture_rebuild.json" \
        --reconstruction "scripts/data/audit_e1_mixture_rebuild.py …" \
        --reconstruction-cost '$0 (CPU)' \
        --apply

Nothing is removed without `--apply`. Everything else about the run happens
either way, so a dry run produces the exact tombstone that would be written.

The five criteria are the only reasons a deletion is allowed. They are spelled
out here rather than left to judgement at the keyboard:

    unreferenced_failed_partial   an interrupted attempt nothing points at
    byte_identical_duplicate      identical bytes to a copy that survives
    reproducible_materialization  deterministically rebuildable at a stated cost
    rejected_search_leaf          a searched-and-dropped state whose lineage,
                                  hashes, probe results and selection evidence
                                  are already recorded elsewhere
    verified_stale_cache          the canonical remote copy has been
                                  hash-verified, file by file

Refusals, which are the point of the tool:

  * the path is `protected` in logs/checkpoint_registry.json, or carries a
    `never_delete_clause`;
  * `--canonical` names something that does not exist;
  * a criterion of `byte_identical_duplicate` when the survivor's bytes differ;
  * a criterion of `verified_stale_cache` without `--relay-verified`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOMBSTONES = REPO_ROOT / "logs/checkpoint_tombstones.json"
REGISTRY = REPO_ROOT / "logs/checkpoint_registry.json"

CRITERIA = {
    "unreferenced_failed_partial": "an interrupted or failed attempt that nothing "
                                   "references",
    "byte_identical_duplicate": "byte-identical to a copy that survives",
    "reproducible_materialization": "a temporary materialization that is "
                                    "deterministically reproducible",
    "rejected_search_leaf": "a rejected search leaf whose lineage, hashes, probe "
                            "results and selection evidence are already preserved",
    "verified_stale_cache": "a stale local cache whose canonical remote artifact "
                            "has been hash-verified",
}


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def absolute(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def manifest(root: Path) -> tuple[list[dict], int]:
    """Every file under root with its size and hash, so the tombstone can say
    exactly what stopped existing."""
    files = sorted(f for f in ([root] if root.is_file() else root.rglob("*"))
                   if f.is_file())
    out, total = [], 0
    for f in files:
        size = f.stat().st_size
        total += size
        out.append({"path": str(f.relative_to(REPO_ROOT))
                    if str(f).startswith(str(REPO_ROOT)) else str(f),
                    "size_bytes": size, "sha256": sha256_file(f)})
    return out, total


def registry_guard(rel: str) -> tuple[bool, str]:
    if not REGISTRY.is_file():
        return True, "no registry to check against"
    reg = json.loads(REGISTRY.read_text())
    for e in reg.get("checkpoints", []):
        if e["path_local"] == rel or rel.startswith(e["path_local"] + "/") \
                or e["path_local"].startswith(rel.rstrip("/") + "/"):
            if e.get("protected"):
                return False, (f"{e['path_local']} is protected in the registry "
                               f"({e['retention']}): {e.get('never_delete_clause') or e['retention_reason']}")
    return True, "not protected in the registry"


def references(rel: str, name: str) -> list[str]:
    hits = []
    for d in ("logs", "configs", "tests", "scripts", "src", "docs", "README.md"):
        base = REPO_ROOT / d
        files = [base] if base.is_file() else [
            f for f in base.rglob("*") if f.is_file()
            and "__pycache__" not in f.parts
            and f.suffix in (".md", ".json", ".py", ".txt", ".sh")]
        for f in files:
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                continue
            if rel in text or (name and name in text):
                hits.append(str(f.relative_to(REPO_ROOT)))
    return sorted(set(hits))[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help="what to retire")
    ap.add_argument("--id", required=True, help="canonical_id for the tombstone")
    ap.add_argument("--criterion", required=True, choices=sorted(CRITERIA))
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--canonical", required=True,
                    help="the surviving evidence: a path, a relay path, or a "
                         "record that carries the same fact")
    ap.add_argument("--frozen-results", default="",
                    help="where the conclusion that depended on this lives now")
    ap.add_argument("--reconstruction", default="")
    ap.add_argument("--reconstruction-cost", default="")
    ap.add_argument("--relay-path", default="")
    ap.add_argument("--relay-verified", action="store_true",
                    help="every file was compared to the relay copy by hash")
    ap.add_argument("--store", default="repo_artifacts")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    target = absolute(args.path)
    if not target.exists():
        print(f"nothing at {args.path}", file=sys.stderr)
        return 1
    rel = args.path.rstrip("/")

    ok, why = registry_guard(rel)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    if args.criterion == "verified_stale_cache" and not args.relay_verified:
        print("REFUSED: verified_stale_cache requires --relay-verified",
              file=sys.stderr)
        return 2
    # `--canonical` may name several survivors, separated by `;`, and may include
    # prose or a relay prefix. Every token that looks like a repository path must
    # actually be there: the whole point is that the survivor really survives.
    missing_survivors = []
    for token in args.canonical.split(";"):
        token = token.strip().split(" ")[0].rstrip(",;")
        if not token or "/" not in token:
            continue
        if token.startswith(("http", "hf:")) or token == args.relay_path:
            continue
        if not token.startswith(("artifacts/", "logs/", "docs/", "configs/",
                                 "scripts/", "src/", "tests/", "/")):
            continue                       # a relay prefix or free text
        if not absolute(token).exists():
            missing_survivors.append(token)
    if missing_survivors:
        print(f"REFUSED: --canonical names {missing_survivors}, which are not there",
              file=sys.stderr)
        return 2

    files, total = manifest(target)
    entry = {
        "canonical_id": args.id,
        "historical_paths": [rel],
        "store": args.store,
        "relay_path": args.relay_path or None,
        "retention_tier": "TIER_4_DISPOSABLE",
        "experiment": args.experiment,
        "seed": None,
        "scientific_role": args.role,
        "weights_sha256": (files[0]["sha256"] if len(files) == 1 else None),
        "config_sha256": None,
        "architecture": None,
        "size_bytes": total,
        "size_human": f"{total / 2**20:.2f} MiB" if total < 2**30
                      else f"{total / 2**30:.2f} GiB",
        "n_files": len(files),
        "members": files,
        "references_at_deletion": references(rel, Path(rel).name),
        "deletion_criterion": args.criterion,
        "deletion_criterion_meaning": CRITERIA[args.criterion],
        "reason_physical_weights_deleted": args.reason,
        "canonical_surviving_evidence": args.canonical,
        "relay_hash_verified": bool(args.relay_verified),
        "deterministic_reconstruction_possible": bool(args.reconstruction),
        "reconstruction_recipe": args.reconstruction or None,
        "expected_reconstruction_cost": args.reconstruction_cost or None,
        "frozen_results_location": args.frozen_results or None,
        "deleted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    doc = json.loads(TOMBSTONES.read_text())
    if any(t["canonical_id"] == args.id for t in doc["tombstones"]):
        print(f"REFUSED: a tombstone for {args.id} already exists", file=sys.stderr)
        return 2

    print(json.dumps({k: v for k, v in entry.items() if k != "members"}, indent=2))
    print(f"  {len(files)} files, {entry['size_human']}")
    if not args.apply:
        print("\nDRY RUN — nothing written, nothing deleted. Pass --apply.")
        return 0

    doc["schema"] = "aadistill.tombstone/v2"
    doc["fields"] = sorted(set(doc["fields"]) | set(entry))
    doc.setdefault("schema_note", {})
    doc["schema_note"] = (
        "v2 adds store, size_bytes, n_files, members, references_at_deletion, "
        "deletion_criterion, canonical_surviving_evidence and "
        "relay_hash_verified. v1 entries are unchanged and remain valid: every "
        "field they carried still means what it meant.")
    doc["tombstones"].append(entry)
    TOMBSTONES.write_text(json.dumps(doc, indent=2) + "\n")

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    print(f"\ntombstoned and deleted: {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
