#!/usr/bin/env python3
"""Delete a local artifact, but only behind a tombstone that outlives it.

    PYTHONPATH=src python scripts/maintenance/consolidation/retire_artifact.py \
        --path artifacts/audit/ladder_uniform_rebuild \
        --id ladder_uniform_rebuild \
        --criterion byte_identical_duplicate \
        --experiment E1 --role "rebuilt ladder for the mixture audit" \
        --reason "..." --canonical "artifacts/shared/instruments/ladder_uniform_probe" \
        --frozen-results "artifacts/audit/e1_mixture_rebuild.json" \
        --reconstruction "scripts/stages/stage-3/e1/audit_e1_mixture_rebuild.py …" \
        --reconstruction-cost '$0 (CPU)' \
        --apply

Nothing is removed without `--apply`. Everything else about the run happens
either way, so a dry run produces the exact tombstone that would be written.

The six criteria are the only reasons a deletion is allowed. They are spelled
out here rather than left to judgement at the keyboard:

    unreferenced_failed_partial   an interrupted attempt nothing points at
    byte_identical_duplicate      identical bytes to a copy that survives
    reproducible_materialization  deterministically rebuildable at a stated cost
    rejected_search_leaf          a searched-and-dropped state whose lineage,
                                  hashes, probe results and selection evidence
                                  are already recorded elsewhere
    verified_stale_cache          the canonical remote copy has been
                                  hash-verified, file by file
    agents_2_5_standing_retirement
                                  AGENTS.md 2.5: terminal, evidence and hashes
                                  durable, no consumer, not an anchor

How the bytes could come back is three separate facts, never one flag:

    --reconstruction / --reconstruction-kind
                     a recipe and what it is: `retrain` (a new training run: new weights, not these bytes),
                     `rebuild` (a computation from preserved inputs), `copy` (fetch a surviving copy) or `none`
    --byte-identical whether the recipe gives back THESE bytes: `verified` (shown by an earlier rebuild whose hash
                     matched; `--byte-identical-evidence` says where), `not_verified`, or `not_expected` (a retrain)
    --same-hash-copy where a byte-identical copy survives (a local one is re-hashed)

`deterministic_reconstruction_possible` is derived: true only when `--byte-identical verified`. Until 2026-11-14 it
was `bool(--reconstruction)`, so any recipe -- a retrain included -- read as a deterministic rebuild.

Refusals, which are the point of the tool:

  * the path is `protected` in logs/maintenance/inventories/checkpoint_registry.json, or carries a
    `never_delete_clause` -- unless `--maintainer-decision` quotes the recorded maintainer decision that
    retires it (logs/budget/decisions.md); the tombstone then carries the decision and the class and
    clause it lifted (`protection_lifted`);
  * `--canonical` names something that does not exist;
  * a criterion of `byte_identical_duplicate` when the survivor's bytes differ;
  * a criterion of `verified_stale_cache` without `--relay-verified`;
  * a recipe without its kind, `--byte-identical verified` without evidence or for a retrain, a
    `reproducible_materialization` whose rebuild is not verified byte-identical, a `byte_identical_duplicate`
    without a same-hash copy, and a local same-hash copy whose bytes differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOMBSTONES = REPO_ROOT / "logs/maintenance/inventories/checkpoint_tombstones.json"
REGISTRY = REPO_ROOT / "logs/maintenance/inventories/checkpoint_registry.json"

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
    "agents_2_5_standing_retirement": "a terminal historical checkpoint whose evidence "
                                      "and hashes are durable and that no consumer or "
                                      "anchor needs (AGENTS.md 2.5)",
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


RECONSTRUCTION_KINDS = ("retrain", "rebuild", "copy", "none")
BYTE_IDENTICAL = ("verified", "not_verified", "not_expected")


def reconstruction_fields(recipe: str, kind: str | None, byte_identical: str | None, evidence: str,
                          copies: list[str], criterion: str, sha256: str | None = None) -> dict:
    """The tombstone's reconstruction fields, kept apart: a recipe and its kind, whether it gives back these bytes,
    and where a same-hash copy survives. Raises ValueError on an inconsistent combination."""
    if recipe and not kind:
        raise ValueError("--reconstruction needs --reconstruction-kind (retrain, rebuild, copy or none)")
    kind = kind or "none"
    if byte_identical is None:
        byte_identical = "not_expected" if kind == "retrain" else "not_verified"
    if byte_identical == "verified" and not evidence.strip():
        raise ValueError("--byte-identical verified needs --byte-identical-evidence: the rebuild whose hash matched")
    if byte_identical == "verified" and kind == "retrain":
        raise ValueError("a retrain does not give back the same bytes; it is a new checkpoint")
    if criterion == "reproducible_materialization" and byte_identical != "verified":
        raise ValueError("reproducible_materialization needs a rebuild verified byte-identical")
    if criterion == "byte_identical_duplicate" and not copies:
        raise ValueError("byte_identical_duplicate needs --same-hash-copy")
    for c in copies:
        p = absolute(c.split(" ")[0])
        if sha256 and p.is_file() and sha256_file(p) != sha256:
            raise ValueError(f"--same-hash-copy {c} does not hash to {sha256}")
    return {"reconstruction_recipe": recipe or None, "reconstruction_kind": kind,
            "byte_identical_reconstruction": byte_identical,
            "byte_identical_evidence": evidence.strip() or None,
            "same_hash_copies": copies,
            #: derived, never set from the presence of a recipe
            "deterministic_reconstruction_possible": byte_identical == "verified"}


def registry_guard(rel: str) -> tuple[bool, str, dict | None]:
    if not REGISTRY.is_file():
        return True, "no registry to check against", None
    reg = json.loads(REGISTRY.read_text())
    for e in reg.get("checkpoints", []):
        if e["path_local"] == rel or rel.startswith(e["path_local"] + "/") \
                or e["path_local"].startswith(rel.rstrip("/") + "/"):
            if e.get("protected"):
                return False, (f"{e['path_local']} is protected in the registry "
                               f"({e['retention']}): {e.get('never_delete_clause') or e['retention_reason']}"), \
                    {"path_local": e["path_local"], "retention": e["retention"],
                     "never_delete_clause": e.get("never_delete_clause"), "retention_reason": e.get("retention_reason")}
    return True, "not protected in the registry", None


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
    ap.add_argument("--reconstruction", default="", help="the recipe")
    ap.add_argument("--reconstruction-kind", choices=RECONSTRUCTION_KINDS, default=None)
    ap.add_argument("--byte-identical", choices=BYTE_IDENTICAL, default=None,
                    help="whether the recipe gives back these bytes (default: not_expected for a retrain, "
                         "else not_verified)")
    ap.add_argument("--byte-identical-evidence", default="",
                    help="the earlier rebuild whose hash matched (required with --byte-identical verified)")
    ap.add_argument("--same-hash-copy", action="append", default=[],
                    help="where a byte-identical copy survives (repeatable; a local one is re-hashed)")
    ap.add_argument("--reconstruction-cost", default="")
    ap.add_argument("--relay-path", default="")
    ap.add_argument("--relay-verified", action="store_true",
                    help="every file was compared to the relay copy by hash")
    ap.add_argument("--store", default="repo_artifacts")
    ap.add_argument("--maintainer-decision", default="",
                    help="the recorded maintainer decision (logs/budget/decisions.md, quoted with its date) that "
                         "retires an artifact the registry protects")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    target = absolute(args.path)
    if not target.exists():
        print(f"nothing at {args.path}", file=sys.stderr)
        return 1
    rel = args.path.rstrip("/")

    ok, why, guard = registry_guard(rel)
    if not ok and not args.maintainer_decision.strip():
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
    try:
        recon = reconstruction_fields(args.reconstruction, args.reconstruction_kind, args.byte_identical,
                                      args.byte_identical_evidence, args.same_hash_copy, args.criterion,
                                      files[0]["sha256"] if len(files) == 1 else None)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
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
        **recon,
        "expected_reconstruction_cost": args.reconstruction_cost or None,
        "frozen_results_location": args.frozen_results or None,
        **({"protection_lifted": {**guard, "maintainer_decision": args.maintainer_decision.strip()}}
           if guard else {}),
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
    #: the owner's totals are a running account (retire_eligible_checkpoints.py advances them the same way)
    if "totals" in doc:
        t = doc["totals"]
        t.update(tombstones_total=t["tombstones_total"] + 1, active=t["active"] + 1,
                 active_retired_bytes=t["active_retired_bytes"] + total)
        t["active_retired_human"] = f"{t['active_retired_bytes'] / 2**30:.3f} GiB"
    TOMBSTONES.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    print(f"\ntombstoned and deleted: {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
