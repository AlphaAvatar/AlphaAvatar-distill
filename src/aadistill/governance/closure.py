"""Derive what a session would execute, by walking import edges.

A frozen source-set declaration answers two different questions, and until the
initialization cutover they had the same answer, so nothing forced them apart:

* **what did that completed run execute?** — a historical fact, fixed to the
  paths that existed then, and deliberately uncomputable afterwards so an old
  authorization cannot be revalidated against a tree it never ran on;
* **what would the next run execute?** — a property of the CURRENT tree.

This module answers the second, and *derives* it rather than restating a list.
From declared entry points it walks `import` edges transitively across the
configured source roots, so a module that becomes reachable is included whether
or not anyone remembered to add it. A hand-maintained list cannot make that
promise: the previous C1 declaration named a setup script for months after
nothing executed it, and its own comment said so.

Deriving live also removes a failure mode rather than moving it. A *recorded*
file list goes stale the moment a listed file is edited, and a stale list that
still loads is worse than one that is missing — it reports a confident identity
for the wrong set of files. Callers may still compare against a recorded
snapshot, but nothing here trusts one.

The entry points themselves are deliberately NOT declared here. They are an
experiment-instance fact and belong with the experiment; this module owns only
the mechanism.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

SCHEMA = "aadistill.executable_closure/v1"

#: Source roots a caller puts on sys.path, in resolution order.
DEFAULT_ROOTS = ("src", "scripts")

#: Import roots that are ours. An unresolvable import outside these is a
#: third-party or stdlib module and carries no repository identity.
INTERNAL_ROOTS = ("aadistill", "experiments")


class ClosureError(RuntimeError):
    """A closure could not be derived over a complete, existing file set."""


def _candidates(mod: str, roots: tuple[str, ...]) -> list[str]:
    rel = mod.replace(".", "/")
    return [f"{root}/{rel}{suffix}"
            for root in roots for suffix in (".py", "/__init__.py")]


def _resolve(repo: Path, mod: str, roots: tuple[str, ...]) -> str | None:
    for cand in _candidates(mod, roots):
        if (repo / cand).is_file():
            return cand
    return None


def _absolutize(rel: str, level: int, module: str, roots: tuple[str, ...]) -> str:
    """`from ..pkg import x` in a file -> the absolute module name.

    A relative import is invisible to a walk that only reads `node.module`, and
    silently skipping one under-counts the closure — it would report a complete
    set while omitting whatever that edge reaches. Resolving them is mechanical,
    so the walk does that rather than requiring every file to be written in a
    particular style.
    """
    parts = rel.split("/")
    if parts[0] in roots:
        parts = parts[1:]
    #: The containing directory is the current package, for a plain module and
    #: for an `__init__.py` alike: in `a/b/__init__.py`, level 1 means `a.b`.
    pkg = parts[:-1]
    if level - 1 > len(pkg):
        raise ClosureError(
            f"{rel} imports {level} level(s) up, past the top of its package")
    base = pkg[:len(pkg) - (level - 1)]
    return ".".join(base + (module.split(".") if module else []))


def _imports_of(repo: Path, rel: str, roots: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """(modules definitely imported, names that MIGHT be submodules).

    `from pkg import name` is ambiguous in the AST: `name` may be a submodule or
    an ordinary attribute. Both are probed, and only what resolves to a file is
    kept — but a probe that fails is not a miss, or every `from x import Class`
    would be reported and bury the real ones.
    """
    path = repo / rel
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ClosureError(f"cannot parse {rel}: {exc}") from exc
    definite: set[str] = set()
    probes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            definite.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = (_absolutize(rel, node.level, node.module or "", roots)
                      if node.level else node.module)
            if not module:
                continue
            definite.add(module)
            probes.update(f"{module}.{a.name}" for a in node.names)
    return definite, probes


def walk(repo_root: str | Path, entry_points: tuple[str, ...],
         roots: tuple[str, ...] = DEFAULT_ROOTS) -> tuple[list[str], list[str]]:
    """(files reachable from the entry points, unresolved internal modules)."""
    repo = Path(repo_root)
    missing = [e for e in entry_points if not (repo / e).is_file()]
    if missing:
        raise ClosureError(f"entry point(s) missing: {missing}")

    seen = set(entry_points)
    unresolved: set[str] = set()
    stack = list(entry_points)
    while stack:
        rel = stack.pop()
        if not rel.endswith(".py"):
            continue
        definite, probes = _imports_of(repo, rel, roots)
        for mod in definite | probes:
            target = _resolve(repo, mod, roots)
            if target is None:
                if mod in definite and mod.split(".")[0] in INTERNAL_ROOTS:
                    unresolved.add(mod)
                continue
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return sorted(seen), sorted(unresolved)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_of(rows: list[dict]) -> str:
    """sha256 over sorted 'path:sha256' lines. Same rule as every other set."""
    return hashlib.sha256(
        "".join(f"{r['path']}:{r['sha256']}\n" for r in rows).encode()).hexdigest()


def derive(repo_root: str | Path, experiment_id: str,
           entry_points: tuple[str, ...], declared_inputs: tuple[str, ...] = (),
           roots: tuple[str, ...] = DEFAULT_ROOTS) -> dict:
    """The live executable closure, hashed from the tree as it is now.

    `declared_inputs` are the non-python files that decide execution or
    authorization — the setup script, artifact specs, the experiment
    configuration. No import edge reaches them, and their bytes change what runs
    just as surely as a module's do, so they are named explicitly and hashed
    identically.
    """
    repo = Path(repo_root)
    files, unresolved = walk(repo, entry_points, roots)
    if unresolved:
        raise ClosureError(
            f"unresolved internal import(s): {sorted(unresolved)[:8]}; the "
            "closure would describe a smaller set than would actually run")
    missing = [d for d in declared_inputs if not (repo / d).is_file()]
    if missing:
        raise ClosureError(f"declared non-python input(s) missing: {missing}")

    rows = [{"path": p, "sha256": sha256_of(repo / p),
             "bytes": (repo / p).stat().st_size}
            for p in sorted(set(files) | set(declared_inputs))]
    return {
        "schema": SCHEMA,
        "_contract": (
            "What a CURRENT session would execute, derived by walking import "
            "edges from the declared entry points, plus the non-python inputs "
            "that decide execution or authorization. Derived live from the tree "
            "at the moment it was computed. This is NOT a historical record and "
            "describes no completed run. AUTHORIZES NOTHING."),
        "experiment_id": experiment_id,
        "entry_points": list(entry_points),
        "declared_non_python_inputs": list(declared_inputs),
        "source_roots": list(roots),
        "derivation": ("transitive AST import closure; `from pkg import name` "
                       "probes name as a submodule and keeps it only if it "
                       "resolves to a file"),
        "n_files": len(rows),
        "digest": digest_of(rows),
        "rule": "sha256 over sorted 'path:sha256' lines of the derived set",
        "files": rows,
        "authorizes": "nothing",
    }


def compare(live: dict, recorded: dict) -> dict:
    """How a live closure differs from a recorded snapshot of one.

    Used to report drift, never to gate on it: a recorded snapshot is evidence
    of what the set looked like when it was written, and an edit since then is
    ordinary, whereas a file appearing or disappearing is a change in what would
    run.
    """
    live_paths = {r["path"] for r in live["files"]}
    rec_paths = {r["path"] for r in recorded["files"]}
    rec_sha = {r["path"]: r["sha256"] for r in recorded["files"]}
    return {
        "digest_matches": live["digest"] == recorded["digest"],
        "added_files": sorted(live_paths - rec_paths),
        "removed_files": sorted(rec_paths - live_paths),
        "changed_files": sorted(
            r["path"] for r in live["files"]
            if r["path"] in rec_sha and r["sha256"] != rec_sha[r["path"]]),
    }
