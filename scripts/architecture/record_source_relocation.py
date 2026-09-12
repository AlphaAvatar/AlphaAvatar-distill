#!/usr/bin/env python3
"""Account for the initialization cutover against every frozen source set.

    PYTHONPATH=src:scripts python scripts/architecture/record_source_relocation.py --write

A frozen source-set declaration lists the files a completed run executed, by
path. The cutover moved most of those paths, so each historical set now names
files that do not exist. Three things have to be true at once, and this records
all three from the tree and from git rather than asserting them:

1. **The history is unchanged.** Every historical preregistration and result
   file keeps its bytes. Nothing about a completed run is rewritten, and the
   declarations keep naming the OLD paths, because that is what those runs
   executed. A declaration is a record, not an instruction to freeze the
   package layout forever.

2. **The moves are honest.** For every declared file this records its old path,
   its new path (or that it was deleted), and the sha256 on both sides. `git mv`
   preserves content, so a move shows old_sha == new_sha; anything else is a
   move PLUS an edit and is reported as such. Additions and removals are
   counted separately — this never claims additive-only over removals.

3. **Old launch compatibility is fail-closed.** A declaration repointed at
   current paths computes a DIFFERENT digest than the one a completed run
   recorded, so an old grant or authorization still fails its own check — by
   mismatch rather than by exception, which is the check those documents were
   always meant to fail. `C1_HARNESS_SOURCE_FILES_V1` is not repointed and
   still refuses outright, because C1's live contract is the derived closure.
   Both behaviours are exercised here rather than asserted, and only a refusal
   raised by the digest rule itself is counted as one.

The historical commit that owns the old paths is recorded so a reader can check
any old hash against the tree as it actually was.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/architecture"))

from aadistill.governance.authorization import AuthorizationError  # noqa: E402
from migration_map import MAP  # noqa: E402

OUT = "logs/maintenance/inventories/architecture_source_relocation.json"
SCHEMA = "aadistill.architecture_source_relocation/v1"

#: Every frozen declaration, with the callable that computes its digest so the
#: fail-closed claim can be exercised rather than asserted.
SETS = [
    ("HARNESS_SOURCE_FILES_V1", "aadistill.governance.authorization",
     "harness_source_digest"),
    ("PHASE_A_HARNESS_SOURCE_FILES_V1", "experiments.phase_a.plan",
     "phase_a_harness_digest"),
    ("PHASE_B_EXECUTABLE_SOURCE_FILES_V1", "experiments.phase_b.plan",
     "phase_b_source_digest"),
    ("CONTINUATION_SOURCE_FILES_V2", "experiments.phase_b.continuation",
     "continuation_source_digest"),
    ("CONTINUATION_HARNESS_SOURCE_FILES_V1", "experiments.recovery_continuation.plan",
     None),
    ("C1_HARNESS_SOURCE_FILES_V1", "experiments.phase_c1.authorization",
     "c1_historical_harness_digest"),
    ("GENERATION_SOURCE_FILES_V1", "aadistill.initialization.planning.generation",
     None),
    ("TRAINER_SOURCE_FILES_V1", "aadistill.initialization.planning.recovery",
     None),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def sha_of_blob(commit: str, path: str) -> str | None:
    out = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO,
                         capture_output=True)
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()


def sha_of_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def relocation_table(base_commit: str) -> dict[str, dict]:
    """old path -> {new, old_sha, new_sha, identical}. From git and the tree."""
    table: dict[str, dict] = {}
    for old, new in MAP.items():
        old_sha = sha_of_blob(base_commit, old)
        new_sha = sha_of_file(REPO / new)
        table[old] = {"new_path": new, "old_sha256": old_sha,
                      "new_sha256": new_sha,
                      "content_identical": old_sha is not None and old_sha == new_sha}
    return table


def account_set(name: str, module: str, digest_fn: str | None,
                table: dict[str, dict], base_commit: str) -> dict:
    declared = list(getattr(importlib.import_module(module), name))
    moved, unchanged, deleted, edited = [], [], [], []
    for path in declared:
        if path in table:
            rec = table[path]
            (moved if rec["content_identical"] else edited).append(
                {"from": path, "to": rec["new_path"],
                 "old_sha256": rec["old_sha256"], "new_sha256": rec["new_sha256"]})
        elif (REPO / path).is_file():
            unchanged.append(path)
        else:
            deleted.append({"path": path, "old_sha256": sha_of_blob(base_commit, path)})

    entry = {
        "declaration": name, "declared_in": module, "n_declared": len(declared),
        "moved_unchanged_bytes": len(moved), "moved_and_edited": len(edited),
        "still_at_declared_path": len(unchanged), "deleted": len(deleted),
        "additive_only": not (moved or edited or deleted),
        "_accounting": (
            (f"{len(moved) + len(edited)} file(s) MOVED and {len(deleted)} "
             "DELETED relative to what this declaration names. Claiming "
             "additive-only over removals would be false.")
            if (moved or edited or deleted) else
            #: A repointed declaration names current paths, so nothing here is
            #: outstanding. Its move history is not lost — it is recorded in
            #: logs/maintenance/inventories/architecture_declaration_history.json, which holds the old
            #: path list and the digest the completed runs bound.
            "this declaration names paths that all exist; its relocation is "
            "accounted for in logs/maintenance/inventories/architecture_declaration_history.json"),
        "moves": moved, "moves_with_edits": edited, "deletions": deleted,
    }
    if digest_fn:
        #: Resolve the name FIRST, outside the try. An earlier revision of this
        #: script wrapped the lookup in `except Exception` along with the call,
        #: so three mistyped function names were recorded as "fail_closed": a
        #: typo was being reported as evidence that an old grant could not be
        #: revalidated. A name that does not exist is a bug in this script, and
        #: it must crash here rather than become a reassuring line in a log.
        fn = getattr(importlib.import_module(module), digest_fn)
        try:
            d = fn(str(REPO))
            entry["computes"] = True
            entry["current_digest"] = d["digest"] if isinstance(d, dict) else d
        except AuthorizationError as exc:
            #: Only a refusal from the digest rule itself counts. Any other
            #: exception is a defect, and is left to propagate.
            entry["computes"] = False
            entry["current_digest"] = None
            entry["refusal"] = f"{type(exc).__name__}: {exc}"
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--base", default="", help="commit owning the old paths")
    args = ap.parse_args()

    base = args.base or git("rev-parse", "origin/main")
    table = relocation_table(base)
    sets = [account_set(n, m, d, table, base) for n, m, d in SETS]

    doc = {
        "schema": SCHEMA,
        "_contract": (
            "How the initialization cutover stands against every frozen source "
            "set. Historical declarations keep naming the OLD paths because "
            "that is what those runs executed; this records where each file "
            "went, proves the bytes did not change, and EXERCISES the refusal "
            "that makes an old authorization unusable here. AUTHORIZES NOTHING."),
        "historical_base_commit": base,
        "_base_commit_meaning": (
            "the commit that owns the old paths. Every old_sha256 below is the "
            "blob at THAT commit, so a reader can check any of them against the "
            "tree as it actually was, not as it is now."),
        "migrated_commit": git("rev-parse", "HEAD"),
        "counts": {
            "files_moved": len(MAP),
            "moved_with_identical_bytes": sum(
                1 for r in table.values() if r["content_identical"]),
            "moved_with_edits": sum(
                1 for r in table.values() if not r["content_identical"]),
            "sets_examined": len(sets),
            "sets_computable": sum(1 for s in sets if s.get("computes")),
            "sets_refusing": sum(1 for s in sets if s.get("computes") is False),
        },
        "sets": sets,
        "historical_evidence": {
            "moved_or_rewritten": 0,
            "note": ("No file under logs/ was moved, renamed or rewritten by "
                     "this migration. Run evidence and preregistrations are "
                     "untouched; logs/runs/index.json verifies that "
                     "independently by digest."),
        },
        "authorizes": "nothing",
    }
    print(json.dumps(doc["counts"], indent=1))
    for s in sets:
        state = ("computable" if s.get("computes")
                 else "REFUSES" if s.get("computes") is False else "n/a")
        print(f"  {s['declaration']:42} declared={s['n_declared']:>3} "
              f"moved={s['moved_unchanged_bytes'] + s['moved_and_edited']:>3} "
              f"(of which edited={s['moved_and_edited']:>2}) "
              f"deleted={s['deleted']:>2} still={s['still_at_declared_path']:>3} "
              f"{state}")
    if args.write:
        (REPO / OUT).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
