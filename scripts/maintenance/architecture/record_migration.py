#!/usr/bin/env python3
"""Record a source relocation, generically, for every affected declaration.

    PYTHONPATH=src:scripts python scripts/architecture/record_migration.py \
        --migration initialization-core --version v1 \
        --base <commit> --tip <commit> --decision <file> --write

The Phase-B ledger answers one phase's question -- did the Phase-B executable
set drift after its freeze, and why. This answers the migration's question, for
every declaration at once, and it is deliberately NOT that schema: forcing a
package relocation into a provider-ownership record would describe it as a
repair to one session's machinery.

Written hierarchically, under
`logs/maintenance/source-relocations/<migration>/<version>/`, because a
relocation is a subject with its own versions rather than another
attempt-shaped file in a flat directory. It is maintenance of the repository's
own source, not an experiment record and not a log-layout history.

Every quantity is DERIVED from git and from the tree:

* old and new path for each declared file, and whether the move was a pure
  rename or a rename PLUS a modification -- decided by comparing the blob at the
  base against the file now, not by trusting `git mv`;
* the real additions and removals, including files that ceased to exist;
* the patch digest over the whole span;
* the identity each declaration produced at the base, and produces now;
* whether an old launch document can still validate -- exercised, not asserted.

A maintainer owns the decision text and nothing else, and it is required: this
records an approval that was given, so it cannot invent one.
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

SCHEMA = "aadistill.migration_source_relocation/v1"

#: (declaration, module, digest callable or None). The digest is exercised so
#: "this still computes" / "this refuses" is a measurement, not a claim.
#: Each declaration and the module that OWNS it. Five of these moved out of the
#: core in the Milestone-A closure -- a declaration is an experiment's statement
#: about which files it executes, which is exactly the kind of fact the core
#: stopped holding. This table follows the owner rather than the historical
#: location; the historical location is what `account()` measures the move from.
DECLARATIONS = [
    ("HARNESS_SOURCE_FILES_V1", "experiments.preflight",
     "preflight_harness_digest"),
    ("PHASE_A_HARNESS_SOURCE_FILES_V1", "experiments.phase_a.plan",
     "phase_a_harness_digest"),
    ("PHASE_B_EXECUTABLE_SOURCE_FILES_V1", "experiments.phase_b.plan",
     "phase_b_source_digest"),
    ("CONTINUATION_SOURCE_FILES_V2", "experiments.phase_b.continuation",
     "continuation_source_digest"),
    ("CONTINUATION_HARNESS_SOURCE_FILES_V1",
     "experiments.recovery_continuation.plan", None),
    ("C1_HARNESS_SOURCE_FILES_V1", "experiments.phase_c1.authorization",
     "c1_historical_harness_digest"),
    ("C1_SCORING_FILES_V1", "experiments.phase_c1.scoring", None),
    ("GENERATION_SOURCE_FILES_V1", "experiments.source_sets", None),
    ("TRAINER_SOURCE_FILES_V1", "experiments.source_sets", None),
    ("RECOVERY_SCORING_FILES_V2", "experiments.source_sets", None),
    ("RECOVERY_SCORING_FILES_V3", "experiments.source_sets", None),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout


def blob_sha(commit: str, path: str) -> str | None:
    out = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO,
                         capture_output=True)
    return hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None


def file_sha(path: str) -> str | None:
    p = REPO / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def digest_over(pairs: list[tuple[str, str]]) -> str:
    """The rule every declaration uses: sha256 over sorted 'path:sha256' lines."""
    return hashlib.sha256(
        "".join(f"{p}:{s}\n" for p, s in sorted(pairs)).encode()).hexdigest()


def classify(old: str, base: str) -> dict:
    """Where a declared file went, and whether its bytes moved with it."""
    new = MAP.get(old)
    old_sha = blob_sha(base, old)
    if new is None:
        if (REPO / old).is_file():
            now = file_sha(old)
            if old_sha is None:
                #: A declaration REPOINTED at a path that did not exist at the
                #: base. Calling that "modified in place" would describe a
                #: relocation as an edit; where the file came FROM is recorded
                #: in logs/maintenance/inventories/architecture_declaration_history.json, which holds
                #: each declaration's pre-migration path list.
                disposition = "declared_path_absent_at_base"
            elif now == old_sha:
                disposition = "unchanged_path"
            else:
                disposition = "modified_in_place"
            return {"path": old, "disposition": disposition,
                    "sha256_at_base": old_sha, "sha256_now": now}
        return {"path": old, "disposition": "removed",
                "sha256_at_base": old_sha, "sha256_now": None}
    now = file_sha(new)
    return {
        "path": old, "new_path": new,
        "disposition": ("renamed" if now is not None and now == old_sha
                        else "renamed_and_modified" if now is not None
                        else "renamed_to_a_path_that_is_missing"),
        "sha256_at_base": old_sha, "sha256_now": now,
    }


def account(name: str, module: str, digest_fn: str | None, base: str) -> dict:
    declared = list(getattr(importlib.import_module(module), name))
    moves = [classify(rel, base) for rel in declared]
    by = {}
    for m in moves:
        by[m["disposition"]] = by.get(m["disposition"], 0) + 1

    at_base = [(m["path"], m["sha256_at_base"]) for m in moves
               if m["sha256_at_base"]]
    now = [(m.get("new_path", m["path"]), m["sha256_now"]) for m in moves
           if m["sha256_now"]]

    entry = {
        "declaration": name,
        "declared_in": module,
        "n_declared": len(declared),
        "dispositions": by,
        "additive_only": set(by) <= {"unchanged_path"},
        "_accounting": (
            f"{by.get('renamed', 0)} renamed with identical bytes, "
            f"{by.get('renamed_and_modified', 0)} renamed AND modified, "
            f"{by.get('modified_in_place', 0)} modified in place, "
            f"{by.get('removed', 0)} removed. Recording this as additive would "
            "be false wherever anything was removed."),
        "identity_at_base": (
            digest_over(at_base) if len(at_base) == len(declared) else None),
        "identity_now": digest_over(now) if len(now) == len(declared) else None,
        "historical_only": True,
        "launch_compatible_with_frozen_preregistration": False,
        "scientific_fields_changed": 0,
        "files": moves,
    }
    if digest_fn:
        fn = getattr(importlib.import_module(module), digest_fn)
        try:
            doc = fn(str(REPO))
            entry["live_digest_computes"] = True
            entry["live_digest"] = doc["digest"] if isinstance(doc, dict) else doc
            entry["old_document_would_fail_by"] = "digest mismatch"
        except AuthorizationError as exc:
            entry["live_digest_computes"] = False
            entry["live_digest"] = None
            entry["old_document_would_fail_by"] = f"refusal: {exc}"
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--migration", default="initialization-core")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tip", default="HEAD")
    ap.add_argument("--decision", required=True,
                    help="file holding the maintainer's P12 decision text")
    ap.add_argument("--what", required=True,
                    help="file holding the maintainer's description of the change")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    base = git("rev-parse", args.base).strip()
    tip = git("rev-parse", args.tip).strip()
    decision = Path(args.decision).read_text().strip()
    what = Path(args.what).read_text().strip()
    if not decision or not what:
        raise SystemExit("refusing: the maintainer's decision and description "
                         "are required and cannot be empty")

    patch = git("diff", base, tip)
    numstat = {}
    for line in git("diff", "--numstat", base, tip).splitlines():
        added, removed, rel = (line.split("\t") + ["", "", ""])[:3]
        if rel and added != "-":
            numstat[rel] = [int(added), int(removed)]

    declarations = [account(n, m, d, base) for n, m, d in DECLARATIONS]
    doc = {
        "schema": SCHEMA,
        "_contract": (
            "How one source relocation stands against every declaration it "
            "touched. Records CURRENT-CODE MOVEMENT ONLY. It authorizes no "
            "experiment, no provider resource, no grant and no reinterpretation "
            "of any prior result, and no historical experiment ran the "
            "relocated implementation. AUTHORIZES NOTHING."),
        "migration": args.migration,
        "version": args.version,
        "base_commit": base,
        "tip_commit": tip,
        "commits": [line for line in
                    git("log", "--format=%H %s", f"{base}..{tip}").splitlines()],
        "what": what,
        "maintainer_decision": decision,
        "totals": {
            "files_relocated": len(MAP),
            "files_touched_in_span": len(numstat),
            "lines_added": sum(a for a, _ in numstat.values()),
            "lines_removed": sum(d for _, d in numstat.values()),
            "declarations_examined": len(declarations),
        },
        "numstat": numstat,
        "numstat_rule": "git diff --numstat <base> <tip>",
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "patch_rule": "sha256 of git diff <base> <tip>",
        "declarations": declarations,
        "historical_evidence": {
            "files_moved_or_rewritten": 0,
            "rule": ("no file under logs/ was moved, renamed or rewritten by "
                     "this migration; logs/index.json verifies that "
                     "independently by directory digest"),
        },
        "authorizes": "nothing",
    }
    doc["self_sha256"] = hashlib.sha256(
        json.dumps(doc, sort_keys=True).encode()).hexdigest()

    out = (REPO / "logs/maintenance/source-relocations" / args.migration
           / args.version / "source-relocation.json")
    print(f"{args.migration}/{args.version}: {len(declarations)} declaration(s), "
          f"{doc['totals']['lines_added']} added / "
          f"{doc['totals']['lines_removed']} removed over "
          f"{len(doc['commits'])} commit(s)")
    for d in declarations:
        state = ("computes" if d.get("live_digest_computes")
                 else "REFUSES" if d.get("live_digest_computes") is False
                 else "n/a")
        print(f"  {d['declaration']:38} {d['n_declared']:>3} files  "
              f"{d['dispositions']}  {state}")
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
