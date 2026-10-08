#!/usr/bin/env python3
"""Rewrite every import of the old packages to the new module map.

    PYTHONPATH=src python scripts/architecture/rewrite_imports.py --apply

Two kinds of edit, and the distinction matters:

* **import statements** are rewritten from the AST, so a module name inside a
  string or a docstring is never touched. `logs/…` prose describing
  `aadistill.autoinit.arch` stays exactly as written — which is correct,
  because prose about a historical layout is a historical fact.
* **`sys.path` shims** that scripts and tests use to reach `scripts/pod` and
  friends are left alone; they are directory paths, not module names.

Relative imports inside the moved packages are resolved against the module's
NEW location, because the layer depth changes: `autoinit/operators/base.py`
was two levels under `aadistill`, `initialization/operators/base.py` is three,
so `from ..arch import` has to become `from ..specs.arch import`.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/architecture"))

from migration_map import MAP, module_map, module_of  # noqa: E402

NEW_OF_OLD = module_map()
#: new module -> its file, so relative imports can be re-levelled
NEW_FILE_OF_MODULE = {module_of(new): new for new in MAP.values()}


def resolve_relative(module_file: str, level: int, target: str | None) -> str:
    """Absolute module for a relative import, from the file's OLD location."""
    pkg = module_of(module_file).rsplit(".", 1)[0] if not module_file.endswith(
        "__init__.py") else module_of(module_file)
    parts = pkg.split(".")
    if level > 1:
        parts = parts[: -(level - 1)] or parts[:1]
    base = ".".join(parts)
    return f"{base}.{target}" if target else base


def new_module_for(old_module: str) -> str | None:
    if old_module in NEW_OF_OLD:
        return NEW_OF_OLD[old_module]
    for old, new in NEW_OF_OLD.items():          # longest-first
        if old_module.startswith(old + "."):
            return new + old_module[len(old):]
    return None


def rewrite_file(path: Path, old_rel: str | None = None) -> tuple[str, int]:
    """Return (new_source, n_edits). Edits are line-based on import statements."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, 0
    lines = src.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []      # (start, end, replacement)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            changed = False
            names = []
            for a in node.names:
                new = new_module_for(a.name)
                if new:
                    changed = True
                    names.append(f"{new} as {a.asname}" if a.asname else new)
                else:
                    names.append(f"{a.name} as {a.asname}" if a.asname else a.name)
            if changed:
                indent = " " * node.col_offset
                edits.append((node.lineno, node.end_lineno,
                              f"{indent}import {', '.join(names)}\n"))
        elif isinstance(node, ast.ImportFrom):
            if node.level and old_rel:
                absolute = resolve_relative(old_rel, node.level, node.module)
            elif node.level:
                continue                          # relative, outside the move
            else:
                absolute = node.module or ""
            new = new_module_for(absolute)
            if not new:
                continue
            names = ", ".join(f"{a.name} as {a.asname}" if a.asname else a.name
                              for a in node.names)
            indent = " " * node.col_offset
            tail = ""
            raw = "".join(lines[node.lineno - 1:node.end_lineno])
            if "  # noqa" in raw:
                tail = "  # noqa: E402"
            stmt = f"{indent}from {new} import {names}{tail}\n"
            if len(stmt) > 88 and len(node.names) > 1:
                inner = "".join(f"{indent}    {a.name}"
                                f"{f' as {a.asname}' if a.asname else ''},\n"
                                for a in node.names)
                stmt = f"{indent}from {new} import ({tail.strip() or ''}\n{inner}{indent})\n"
            edits.append((node.lineno, node.end_lineno, stmt))

    if not edits:
        return src, 0
    for start, end, text in sorted(edits, reverse=True):
        lines[start - 1:end] = [text]
    return "".join(lines), len(edits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    #: new file -> old file, so a moved module's relative imports resolve
    old_of_new = {new: old for old, new in MAP.items()}

    total_files = total_edits = 0
    for base in ("src", "scripts", "tests"):
        for path in sorted((REPO / base).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            new_src, n = rewrite_file(path, old_of_new.get(rel))
            if n:
                total_files += 1
                total_edits += n
                if args.apply:
                    path.write_text(new_src)
    print(f"{'rewrote' if args.apply else 'would rewrite'} {total_edits} import "
          f"statement(s) in {total_files} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
