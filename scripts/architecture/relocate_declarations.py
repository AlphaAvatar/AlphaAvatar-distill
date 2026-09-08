#!/usr/bin/env python3
"""Point the frozen source declarations at the paths that now exist.

    PYTHONPATH=src:scripts python scripts/architecture/relocate_declarations.py --write

A declaration like `PHASE_A_HARNESS_SOURCE_FILES_V1` was doing two jobs at once,
and until the cutover they had the same answer:

* the **historical record** of what a completed, authorized run executed;
* the **current contract** a future run has to bind.

Those now disagree, and neither can be dropped. The split used here is:

* the Python declaration describes the CURRENT tree, so live code computes an
  identity again — a contract that refuses to compute is not protecting
  anything, it just moves the failure to launch time;
* the historical record moves to a JSON ledger, `logs/architecture_declaration_
  history.json`, holding the old path list, the digest it produced at the base
  commit, and where each file went.

Old authorizations stay unusable, but for a better reason than an exception: the
recorded digest no longer matches the recomputed one, which is the check those
documents were always supposed to fail. That is exercised, not assumed —
`tests/architecture/test_declaration_relocation.py` feeds a real recorded
authorization back in and requires a refusal.

Paths are rewritten inside the existing tuple literals, so the comments that
explain why each file is in the set survive. A file that no longer exists at all
is dropped and recorded as a removal; this never claims additive-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/architecture"))

from migration_map import MAP  # noqa: E402

OUT = "logs/architecture_declaration_history.json"

#: (declaration, file it lives in, module, the version constant to bump)
DECLARATIONS = [
    ("PHASE_A_HARNESS_SOURCE_FILES_V1", "scripts/experiments/phase_a/plan.py",
     "experiments.phase_a.plan", "PHASE_A_HARNESS_SOURCE_SET_VERSION"),
    ("PHASE_B_EXECUTABLE_SOURCE_FILES_V1", "scripts/experiments/phase_b/plan.py",
     "experiments.phase_b.plan", "PHASE_B_SOURCE_SET_VERSION"),
    ("CONTINUATION_SOURCE_FILES_V2", "scripts/experiments/phase_b/continuation.py",
     "experiments.phase_b.continuation", "CONTINUATION_SOURCE_SET_VERSION"),
    ("CONTINUATION_HARNESS_SOURCE_FILES_V1",
     "scripts/experiments/recovery_continuation/plan.py",
     "experiments.recovery_continuation.plan", None),
    ("HARNESS_SOURCE_FILES_V1", "src/aadistill/governance/authorization.py",
     "aadistill.governance.authorization", "HARNESS_SOURCE_SET_VERSION"),
    #: C1_HARNESS_SOURCE_FILES_V1 is deliberately ABSENT. C1's live contract is
    #: now the derived closure (`c1_current_executable`), so repointing the list
    #: as well would leave two implementations owning one identity. Its history
    #: is recorded below like the others; the list itself stays historical.
    ("C1_SCORING_FILES_V1", "scripts/experiments/phase_c1/scoring.py",
     "experiments.phase_c1.scoring", "C1_SCORING_CONTRACT_VERSION"),
]


def git_sha(commit: str, path: str) -> str | None:
    out = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO,
                         capture_output=True)
    return hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None


def digest_over(pairs: list[tuple[str, str]]) -> str:
    """The same rule every declaration uses: sorted 'path:sha256' lines."""
    return hashlib.sha256(
        "".join(f"{p}:{s}\n" for p, s in sorted(pairs)).encode()).hexdigest()


def tuple_span(text: str, name: str) -> tuple[int, int]:
    """The character span of `name`'s tuple literal, parens included."""
    match = re.search(rf"^{re.escape(name)}[^=]*= \(", text, re.M)
    if not match:
        raise SystemExit(f"cannot locate the {name} literal")
    start = match.end() - 1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise SystemExit(f"unbalanced parentheses in {name}")


def relocate(name: str, rel_file: str, declared: list[str], base: str,
             write: bool) -> dict:
    path = REPO / rel_file
    text = path.read_text()
    start, end = tuple_span(text, name)
    body = text[start:end]

    moves, removals, kept = [], [], []
    for old in declared:
        new = MAP.get(old)
        if new:
            moves.append({"from": old, "to": new,
                          "bytes_identical": git_sha(base, old) == (
                              hashlib.sha256((REPO / new).read_bytes()).hexdigest()
                              if (REPO / new).is_file() else None)})
            body = body.replace(f'"{old}"', f'"{new}"')
        elif (REPO / old).is_file():
            kept.append(old)
        else:
            removals.append({"path": old, "sha256_at_base": git_sha(base, old),
                             "why": "the package shell was deleted by the "
                                    "initialization consolidation"})
            #: Drop the line, and any trailing comment that belonged to it.
            body = re.sub(rf'^ *"{re.escape(old)}",.*\n', "", body, flags=re.M)

    if write and body != text[start:end]:
        path.write_text(text[:start] + body + text[end:])

    old_pairs = [(p, git_sha(base, p)) for p in declared]
    new_paths = [MAP.get(p, p) for p in declared if MAP.get(p) or (REPO / p).is_file()]
    new_pairs = [(p, hashlib.sha256((REPO / p).read_bytes()).hexdigest())
                 for p in new_paths]
    return {
        "declaration": name,
        "declared_in": rel_file,
        "n_declared_historically": len(declared),
        "n_declared_now": len(new_paths),
        "moved": len(moves),
        "moved_with_identical_bytes": sum(1 for m in moves if m["bytes_identical"]),
        "removed": len(removals),
        "unchanged_paths": len(kept),
        "additive_only": False,
        "_why_not_additive": (
            "files were MOVED, and package shells were REMOVED where the "
            "consolidation deleted them. Recording this as additive would be "
            "false."),
        "historical_digest_at_base": (
            digest_over([(p, s) for p, s in old_pairs if s])
            if all(s for _, s in old_pairs) else None),
        "current_digest": digest_over(new_pairs),
        "historical_paths": declared,
        "moves": moves,
        "removals": removals,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    base = subprocess.run(["git", "rev-parse", args.base], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()

    import importlib
    records = []
    for name, rel_file, module, _version in DECLARATIONS:
        declared = list(getattr(importlib.import_module(module), name))
        rec = relocate(name, rel_file, declared, base, args.write)
        records.append(rec)
        print(f"  {name:38} {rec['n_declared_historically']:>3} -> "
              f"{rec['n_declared_now']:>3} files "
              f"(moved={rec['moved']}, identical_bytes="
              f"{rec['moved_with_identical_bytes']}, removed={rec['removed']})")

    doc = {
        "schema": "aadistill.architecture_declaration_history/v1",
        "_contract": (
            "The historical identity of each frozen source declaration, kept "
            "here after the Python declarations were repointed at current "
            "paths. `historical_digest_at_base` is what the declaration "
            "produced on the tree the completed runs actually executed; it is "
            "the value recorded in their authorizations, and it is retained so "
            "those records stay checkable. AUTHORIZES NOTHING, and re-pointing "
            "a declaration does not make any old grant usable — the recomputed "
            "digest no longer matches what those documents recorded."),
        "base_commit": base,
        "migrated_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True).stdout.strip(),
        "declarations": records,
        "authorizes": "nothing",
    }
    if args.write:
        (REPO / OUT).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
