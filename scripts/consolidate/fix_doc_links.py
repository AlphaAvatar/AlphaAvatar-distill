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
    p = root / "logs/runs/index.json"
    if not p.is_file():
        return ()
    idx = json.loads(p.read_text())
    return tuple(sorted({rel for e in idx.get("runs", [])
                         for rel in (e.get("components") or {}).values()
                         if (root / rel).is_dir()}))


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
