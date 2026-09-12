#!/usr/bin/env python3
"""Repoint markdown links that a relocation broke, by resolving them.

    PYTHONPATH=src python scripts/consolidate/fix_doc_links.py          # report
    PYTHONPATH=src python scripts/consolidate/fix_doc_links.py --write

A markdown link is relative to the file that contains it, so moving either end
breaks it — and a relocation moves both ends at once. Rewriting absolute
`logs/...` strings is not enough: `[roadmap](phase_c_roadmap.md)` contains no
`logs/` to rewrite, and it broke in two directions at the same time. A document
that stayed still now points at a name that left, and a document that moved into
a subdirectory now resolves its own relative links from the wrong directory.

So this resolves every link against its containing file and repairs only the
ones that do not exist, by finding the target's new location. A repair is made
only when exactly ONE file in the tree carries that basename; an ambiguous name
is reported and left alone, because guessing between two candidates is how a
citation comes to point at the wrong evidence.

It does not invent targets. A link whose basename exists nowhere is reported as
still broken — that is a real dangling reference and the answer is to fix the
document, not to silence the check.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: `[text](target)`, excluding external links and in-page anchors.
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#")

SEARCH_ROOTS = ("logs", "docs", "configs", "scripts", "src", "tests")


def protected_dirs(root: Path) -> tuple[str, ...]:
    """Directories the run index registers with a digest.

    A README inside one of these is covered by that digest, so repointing a
    link in it changes registered historical evidence. This edited 15 of them
    before the guard existed, and `test_every_historical_component_is_byte_
    identical` is what caught it. A stale link inside frozen evidence is
    correct: the evidence records where things were when it was written.
    """
    import json
    p = root / "logs/index.json"
    if not p.is_file():
        return ()
    idx = json.loads(p.read_text())
    registered = {rel for e in idx.get("runs", [])
                  for rel in (e.get("components") or {}).values()
                  if (root / rel).is_dir()}
    #: And the archive. Those documents are kept VERBATIM and say so in their
    #: own header; repointing a link inside one would make that false. A
    #: citation there records where a file was when the document was written,
    #: and the forward mapping is in the relocation record.
    registered.add("logs/archive")
    return tuple(sorted(registered))


def index_basenames(root: Path) -> dict[str, list[Path]]:
    """Basename -> every path carrying it. DIRECTORIES included.

    A link may point at a directory -- `[attempt 7](autoinit_..._attempt7/)` --
    and a files-only index reports every one of those as unrepairable while the
    directory sits one level away.
    """
    out: dict[str, list[Path]] = collections.defaultdict(list)
    for top in SEARCH_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if ".git" not in p.parts:
                out[p.name].append(p)
    for p in root.glob("*.md"):
        out[p.name].append(p)
    return out


def relocation_map(root: Path) -> dict[str, str]:
    """old path -> new path, from every migration manifest present.

    Consulted BEFORE any basename search, because the manifest KNOWS where an
    object went. Searching by name cannot tell `attempt10` of one experiment
    from `attempt10` of another, and reported those as ambiguous while the
    answer was recorded.
    """
    out: dict[str, str] = {}
    for m in sorted(root.glob("logs/migrations/*/manifest.json")):
        try:
            doc = json.loads(m.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for e in doc.get("entries", []):
            if e.get("old_path") and e.get("new_path"):
                out[e["old_path"]] = e["new_path"]
    return out


def via_manifest(doc: Path, target: str, root: Path,
                 moved: dict[str, str]) -> str | None:
    """Resolve a link through the migration record, as a repo-relative path."""
    raw = target.split("#", 1)[0].rstrip("/")
    candidates = []
    resolved = (doc.parent / raw).resolve()
    try:
        candidates.append(resolved.relative_to(root).as_posix())
    except ValueError:
        pass
    candidates.append(raw.lstrip("./"))
    if not raw.startswith(("/", ".")):
        candidates.append(f"logs/{raw}")
    for c in candidates:
        if c in moved:
            return moved[c]
    #: A file INSIDE a directory that moved: the directory is in the record,
    #: the file never moved on its own. Longest prefix first so a nested move
    #: wins over its parent.
    for c in candidates:
        for old in sorted(moved, key=len, reverse=True):
            if c.startswith(old + "/"):
                return moved[old] + c[len(old):]
    #: Last resort: the link names a file by a name that no longer exists
    #: because the object was RENAMED by the migration -- `STATE.md` is now
    #: `state/current.md`. Unique basenames only; an ambiguous one is left for
    #: a human, because guessing here points a citation at the wrong evidence.
    base = Path(raw).name
    hits = {new for old, new in moved.items() if Path(old).name == base}
    return hits.pop() if len(hits) == 1 else None


def narrow(target: str, cands: list[Path], root: Path) -> list[Path]:
    """Prefer candidates whose path ENDS with the link's own path suffix.

    `autoinit_..._attempt7/phase_a_result.json` names three segments and only
    one path in the tree ends with them, but a basename index sees only
    `phase_a_result.json` and calls it ambiguous. Matching the suffix uses the
    information the link already carries instead of discarding it.
    """
    parts = [x for x in Path(target.split("#", 1)[0].rstrip("/")).parts
             if x not in ("..", ".")]
    if len(parts) < 2:
        return cands
    suffix = "/".join(parts)
    exact = [c for c in cands
             if c.relative_to(root).as_posix().endswith(suffix)]
    return exact or cands


def resolve(doc: Path, target: str, root: Path) -> Path:
    t = target.split("#", 1)[0]
    return (doc.parent / t).resolve() if not t.startswith("/") else root / t.lstrip("/")


def repair(root: Path = REPO_ROOT, write: bool = False) -> dict:
    names = index_basenames(root)
    moved = relocation_map(root)
    frozen = protected_dirs(root)
    fixed, unresolved, ambiguous, skipped = [], [], [], []
    for doc in sorted(root.rglob("*.md")):
        if ".git" in doc.parts:
            continue
        rel = doc.relative_to(root).as_posix()
        if any(rel.startswith(d + "/") for d in frozen):
            skipped.append(rel)
            continue
        try:
            text = original = doc.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for m in LINK.finditer(original):
            target = m.group(2)
            if target.startswith(SKIP_PREFIX):
                continue
            dest = resolve(doc, target, root)
            if dest.exists():
                continue
            #: A directory link written with a trailing slash resolves to a
            #: directory that may legitimately have moved too.
            #: The record first: it knows, where a name search only guesses.
            known = via_manifest(doc, target, root, moved)
            if known and (root / known).exists():
                new = Path(__import__("os").path.relpath(
                    root / known, doc.parent)).as_posix()
                if target.endswith("/"):
                    new += "/"
                text = text.replace(f"]({target})", f"]({new})")
                fixed.append({"doc": doc.relative_to(root).as_posix(),
                              "from": target, "to": new, "via": "manifest"})
                continue
            base = Path(target.split("#", 1)[0].rstrip("/")).name
            cands = narrow(target, names.get(base, []), root)
            if len(cands) == 1:
                new = Path(
                    __import__("os").path.relpath(cands[0], doc.parent)).as_posix()
                if target.endswith("/"):
                    new += "/"
                text = text.replace(f"]({target})", f"]({new})")
                fixed.append({"doc": doc.relative_to(root).as_posix(),
                              "from": target, "to": new})
            elif len(cands) > 1:
                ambiguous.append({"doc": doc.relative_to(root).as_posix(),
                                  "target": target,
                                  "candidates": [c.relative_to(root).as_posix()
                                                 for c in cands]})
            else:
                unresolved.append({"doc": doc.relative_to(root).as_posix(),
                                   "target": target})
        if write and text != original:
            doc.write_text(text)
    return {"fixed": fixed, "unresolved": unresolved, "ambiguous": ambiguous,
            "skipped_registered_evidence": sorted(set(skipped))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    r = repair(write=a.write)
    print(f"{'repaired' if a.write else 'repairable'}: {len(r['fixed'])}")
    print(f"ambiguous (left alone): {len(r['ambiguous'])}")
    print(f"still broken          : {len(r['unresolved'])}")
    print(f"skipped (registered evidence): "
          f"{len(r['skipped_registered_evidence'])}")
    for u in r["unresolved"][:15]:
        print(f"   {u['doc']} -> {u['target']}")
    for u in r["ambiguous"][:5]:
        print(f"   ? {u['doc']} -> {u['target']} : {u['candidates'][:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
