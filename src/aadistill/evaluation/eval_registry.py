"""Permanent evaluation sets: content identity, protection and per-capability reports.

An evaluation set is a directory whose manifest names every content file with its
sha256 and size. A registry names every set with its status and the history of the
manifest digests it has been authorized to have, each entry citing the authorization
behind it. This module verifies both, refuses a set whose bytes no longer match the
identity the registry authorizes, and renders a per-capability report in which a
capability a model cannot be measured on is reported as such instead of being scored
zero.

It names no set, no benchmark, no stage and no path: callers pass the registry file.

    problems = verify(registry_path, "some-set")
    require_intact(registry_path, "some-set")      # raises EvalSetIntegrityError
    table = capability_table(declared, outcomes)
"""
from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from aadistill.infrastructure.manifest import sha256_file

REGISTRY_SCHEMA = "aadistill.eval_set_registry/v1"
MANIFEST_SCHEMA = "aadistill.eval_set_manifest/v1"

#: Statuses whose content may change only through a new authorized identity.
PROTECTED = frozenset({"candidate", "frozen"})
STATUSES = PROTECTED | {"draft", "retired"}

#: Outcome statuses a capability unit can have. Only "measured" carries a score.
OUTCOMES = ("measured", "unsupported", "withheld", "not_run")


class EvalSetIntegrityError(RuntimeError):
    """A registered evaluation set does not match its authorized identity."""


def load_registry(path: str | Path) -> dict:
    reg = json.loads(Path(path).read_text())
    if reg.get("schema") != REGISTRY_SCHEMA:
        raise EvalSetIntegrityError(f"{path}: schema {reg.get('schema')!r} != {REGISTRY_SCHEMA!r}")
    ids = [e["eval_set_id"] for e in reg.get("eval_sets", [])]
    if len(ids) != len(set(ids)):
        raise EvalSetIntegrityError(f"{path}: duplicate eval_set_id")
    return reg


def entry(registry: Mapping, eval_set_id: str) -> dict:
    for e in registry.get("eval_sets", []):
        if e["eval_set_id"] == eval_set_id:
            return dict(e)
    raise KeyError(f"no registered eval set {eval_set_id!r}")


def current_identity(e: Mapping) -> str:
    history = e.get("identity_history") or []
    if not history:
        raise EvalSetIntegrityError(f"{e['eval_set_id']}: no identity recorded")
    return history[-1]["manifest_sha256"]


def _walk(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def verify(registry_path: str | Path, eval_set_id: str) -> list[str]:
    """Every way the set's bytes differ from its authorized identity; [] if none."""
    registry_path = Path(registry_path)
    e = entry(load_registry(registry_path), eval_set_id)
    base = registry_path.parent
    root = base / e["root"]
    manifest_path = base / e["manifest"]
    problems: list[str] = []
    if e.get("status") not in STATUSES:
        problems.append(f"unknown status {e.get('status')!r}")
    if e.get("status") in PROTECTED:
        for i, h in enumerate(e.get("identity_history") or []):
            if not (h.get("authorization") or "").strip():
                problems.append(f"identity_history[{i}] cites no authorization")
    if not manifest_path.is_file():
        return problems + [f"manifest missing: {e['manifest']}"]
    digest = sha256_file(manifest_path)
    if digest != current_identity(e):
        problems.append(f"manifest sha256 {digest} != authorized {current_identity(e)}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"manifest schema {manifest.get('schema')!r} != {MANIFEST_SCHEMA!r}")
    if manifest.get("eval_set_id") != eval_set_id:
        problems.append(f"manifest names {manifest.get('eval_set_id')!r}")
    files = manifest.get("files") or {}
    for rel, ident in sorted(files.items()):
        p = root / rel
        if not p.is_file():
            problems.append(f"missing content file {rel}")
            continue
        if p.stat().st_size != ident["bytes"]:
            problems.append(f"{rel}: {p.stat().st_size} bytes != {ident['bytes']}")
        elif sha256_file(p) != ident["sha256"]:
            problems.append(f"{rel}: sha256 differs from the manifest")
    withheld = set(manifest.get("withheld") or {})
    docs = list(manifest.get("documentation") or [])
    if manifest_path.is_relative_to(root):
        docs.append(manifest_path.relative_to(root).as_posix())
    for rel in sorted(_walk(root) - set(files)):
        if rel not in withheld and any(fnmatch.fnmatchcase(rel, pat) for pat in docs):
            continue
        if rel in withheld:
            problems.append(f"{rel} is declared withheld but present in the set")
        else:
            problems.append(f"unlisted file {rel}")
    return problems


def require_intact(registry_path: str | Path, eval_set_id: str) -> dict:
    """The set's manifest, if its bytes match the authorized identity; else raise."""
    problems = verify(registry_path, eval_set_id)
    if problems:
        raise EvalSetIntegrityError(f"{eval_set_id}: " + "; ".join(problems))
    e = entry(load_registry(registry_path), eval_set_id)
    return json.loads((Path(registry_path).parent / e["manifest"]).read_text())


def withheld_intact(path: str | Path, ident: Mapping) -> bool:
    """A withheld file kept outside the repository still has its recorded identity."""
    p = Path(path)
    return p.is_file() and p.stat().st_size == ident["bytes"] and sha256_file(p) == ident["sha256"]


def capability_table(declared: Mapping[str, Sequence[str]],
                     outcomes: Mapping[str, Mapping]) -> dict:
    """Per capability, the outcome of every unit that measures it — never pooled.

    `declared` maps a capability to the units (subsets) that measure it;
    `outcomes` maps a unit to {"status": one of OUTCOMES, ...metrics}. A unit with
    no outcome is "not_run". A capability is "measured" only when every one of its
    units was; a unit that is unsupported, withheld or not run is reported with
    that status and contributes no number, so it can never read as zero.
    """
    table = {}
    for cap, units in declared.items():
        rows = {}
        for u in units:
            o = dict(outcomes.get(u) or {"status": "not_run"})
            if o.get("status") not in OUTCOMES:
                raise ValueError(f"{u}: outcome status {o.get('status')!r} not in {OUTCOMES}")
            rows[u] = o
        statuses = {o["status"] for o in rows.values()}
        if statuses == {"measured"}:
            status = "measured"
        elif "measured" in statuses:
            status = "partial"
        elif len(statuses) == 1:
            status = statuses.pop()
        else:
            status = "not_measured"
        table[cap] = {"status": status, "units": rows}
    return table
