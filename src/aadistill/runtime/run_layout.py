"""Where a run's files live — for any stage, without knowing what a stage is.

The previous version of this contract was compression-shaped. It hard-coded
`logs/runs` as the root, fixed the component vocabulary to
`replay/treatment/training/evaluation/decision`, and required every manifest to
carry `adapter`, `source_spec_hash`, `target_spec_hash`, `compression` and
`operator_path`. That describes one experiment. A dataset-preprocessing run has
no operator path; a throughput benchmark has no source spec; a teacher rollout
has neither, and none of them has a "treatment".

So this module knows three things and no more:

* a run is identified by `(experiment_id, run_id)` under a **caller-supplied**
  root — the core names no directory in the repository;
* a run declares a mapping of **role -> relative path**, where the roles are the
  caller's vocabulary, not ours;
* those paths must be relative, contained in the run, and uniquely owned.

Everything scientific — which roles exist, which are required, what "replay"
means — belongs to an :class:`ArtifactSpec` supplied by the experiment. The core
validates structure, containment, uniqueness and hashes; it never interprets a
role name. That is what lets the same schema carry a preprocessing run, a
rollout run, a recovery training run, a quantized benchmark and a composite
formal experiment without a source change here.

`aadistill.autoinit.run_layout` was the previous home. It named `logs/runs`, and
so could not be reused by a caller writing anywhere else.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

RUN_LAYOUT_VERSION = 3
MANIFEST_SCHEMA = "aadistill.runtime.run_manifest/v3"

#: A single path segment. Anything else is a traversal attempt or a nesting
#: nobody declared.
_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")

#: A role name. Free vocabulary — the core does not enumerate roles — but it
#: must be a usable key: no separators, no leading dot.
_ROLE = re.compile(r"^[a-z0-9][a-z0-9_.]*$")


class RunLayoutError(Exception):
    """The requested path or identifier is not inside the declared layout."""


def _check(kind: str, value: str, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise RunLayoutError(
            f"{kind} {value!r} is not valid here: lowercase letters, digits and "
            "underscores (roles may also contain dots), starting with a letter "
            "or digit. A path separator, '..' or an absolute path is refused "
            "rather than resolved into somewhere outside the run.")
    return value


@dataclass(frozen=True)
class ArtifactSpec:
    """Which roles an experiment requires. Supplied by the experiment.

    The core holds no vocabulary of its own: a Stage-4 rollout run and a C1
    isolation run differ only in the spec they bring. `required` is what must be
    present for the run to be complete; `optional` is what may be present.
    Anything outside both is refused, so a typo becomes an error rather than an
    unreferenced file.
    """

    spec_id: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for r in (*self.required, *self.optional):
            _check("role", r, _ROLE)
        dupes = sorted(set(self.required) & set(self.optional))
        if dupes:
            raise RunLayoutError(
                f"{dupes} are declared both required and optional in "
                f"{self.spec_id!r}; a role is one or the other")

    @property
    def known(self) -> frozenset[str]:
        return frozenset(self.required) | frozenset(self.optional)

    def check(self, roles: Iterable[str]) -> tuple[bool, str]:
        present = set(roles)
        missing = sorted(set(self.required) - present)
        if missing:
            return False, f"{self.spec_id}: required role(s) {missing} absent"
        unknown = sorted(present - self.known)
        if unknown:
            return False, (f"{self.spec_id}: role(s) {unknown} are not declared "
                           "by this spec")
        return True, f"{self.spec_id}: {len(present)} role(s), all declared"


@dataclass(frozen=True)
class RunLayout:
    """One run's directory, under a root the caller owns."""

    run_root: Path
    experiment_id: str
    run_id: str
    layout_version: int = RUN_LAYOUT_VERSION
    #: A path segment between the experiment and the run, supplied by the
    #: caller's convention. Empty by default, so every existing layout composes
    #: exactly as before. It exists because a convention may keep an
    #: experiment's runs in a child directory beside its plans and results
    #: rather than directly under the experiment -- which is a statement about
    #: that convention, not about any experiment, and so belongs in the
    #: caller's hands.
    runs_subdir: str = ""

    def __post_init__(self) -> None:
        _check("experiment_id", self.experiment_id, _ID)
        _check("run_id", self.run_id, _ID)

    @property
    def rel_root(self) -> str:
        """Relative to `run_root`. The core states no repository location."""
        mid = f"{self.runs_subdir}/" if self.runs_subdir else ""
        return f"{self.experiment_id}/{mid}{self.run_id}"

    @property
    def root(self) -> Path:
        p = Path(self.run_root) / self.experiment_id
        if self.runs_subdir:
            p = p / self.runs_subdir
        return p / self.run_id

    def path(self, relative: str) -> Path:
        """Resolve a caller-declared relative path inside this run."""
        if not isinstance(relative, str) or not relative or relative.startswith("/"):
            raise RunLayoutError(f"{relative!r} is not a relative path")
        resolved = (self.root / relative).resolve()
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise RunLayoutError(
                f"{relative!r} resolves to {resolved}, outside the run root {root}")
        return resolved

    def contains(self, path: str | Path) -> bool:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(self.run_root) / candidate
        try:
            candidate.resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        return True

    def create(self, roles: Mapping[str, str] | None = None) -> "RunLayout":
        """Make the run root, and the parent of each declared role path."""
        self.root.mkdir(parents=True, exist_ok=True)
        for rel in (roles or {}).values():
            target = self.path(rel)
            (target if rel.endswith("/") else target.parent).mkdir(
                parents=True, exist_ok=True)
        return self


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_of(path: str | Path) -> dict[str, Any]:
    """sha256 over sorted 'relpath:sha256' lines, for a file or a directory.

    One value that moves if any byte changes, if a file is added, or if one is
    removed — which is what "unchanged" has to mean for a directory.
    """
    target = Path(path)
    if target.is_file():
        entries = [(target.name, sha256_file(target))]
    else:
        entries = sorted((str(p.relative_to(target)), sha256_file(p))
                         for p in target.rglob("*") if p.is_file())
    digest = hashlib.sha256(
        "".join(f"{n}:{s}\n" for n, s in entries).encode()).hexdigest()
    return {"digest": digest, "n_files": len(entries),
            "rule": "sha256 over sorted 'relpath:sha256' lines"}


def build_run_manifest(
    layout: RunLayout,
    *,
    plan: Mapping[str, Any],
    implementation: Mapping[str, Any],
    status: Mapping[str, Any],
    roles: Mapping[str, str],
    spec: ArtifactSpec | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A manifest for any run. Every scientific field comes from the caller.

    `plan` and `implementation` are opaque here: the core records and hashes
    them, and never reads a key. That is deliberate — the moment this function
    knows that a plan has an `operator_path`, it stops being able to describe a
    dataset build.
    """
    from ..infrastructure.manifest import sha256_json

    for role, rel in roles.items():
        _check("role", role, _ROLE)
        layout.path(rel)                       # refuses traversal/escape
    owners: dict[str, list[str]] = {}
    for role, rel in roles.items():
        owners.setdefault(rel, []).append(role)
    clashes = {rel: sorted(rs) for rel, rs in owners.items() if len(rs) > 1}
    if clashes:
        raise RunLayoutError(
            f"one path, one owner — these are claimed twice: {clashes}")
    if spec is not None:
        ok, why = spec.check(roles)
        if not ok:
            raise RunLayoutError(why)

    doc: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "layout_version": layout.layout_version,
        "_contract": (
            "The index of ONE run, for any stage. Roles are the experiment's "
            "vocabulary; this schema does not interpret them. Every path is "
            "relative to this run's own directory. AUTHORIZES NOTHING."),
        "experiment_id": layout.experiment_id,
        "run_id": layout.run_id,
        "root": layout.rel_root,
        "artifact_spec": spec.spec_id if spec is not None else None,
        "plan": dict(plan),
        "implementation": dict(implementation),
        "status": dict(status),
        "roles": dict(sorted(roles.items())),
        "authorizes": "nothing",
    }
    if extra:
        for k, v in extra.items():
            if k in doc:
                raise RunLayoutError(f"{k!r} is a reserved manifest field")
            doc[k] = v
    doc["self_sha256"] = sha256_json(doc)
    return doc


def verify_run_manifest(doc: Mapping[str, Any], run_root: str | Path, *,
                        spec: ArtifactSpec | None = None,
                        require_files: bool = True) -> tuple[bool, str]:
    """Structure, containment, uniqueness, self-hash and — if given — the spec."""
    from ..infrastructure.manifest import sha256_json

    if doc.get("schema") != MANIFEST_SCHEMA:
        return False, f"schema {doc.get('schema')!r} is not {MANIFEST_SCHEMA!r}"
    stated = doc.get("self_sha256")
    if stated != sha256_json({k: v for k, v in doc.items() if k != "self_sha256"}):
        return False, "the manifest does not match its own self_sha256"
    try:
        layout = RunLayout(Path(run_root), doc["experiment_id"], doc["run_id"])
    except (RunLayoutError, KeyError) as exc:
        return False, f"unusable identifiers: {exc}"
    if doc.get("root") != layout.rel_root:
        return False, f"root {doc.get('root')!r} is not {layout.rel_root!r}"

    roles = doc.get("roles") or {}
    owners: dict[str, list[str]] = {}
    for role, rel in roles.items():
        owners.setdefault(rel, []).append(role)
    for rel, rs in owners.items():
        if len(rs) > 1:
            a, b = sorted(rs)[:2]
            return False, f"{rel!r} is claimed by both {a!r} and {b!r}; one path, one owner"
    for role, rel in roles.items():
        try:
            full = layout.path(rel)
        except RunLayoutError as exc:
            return False, f"{role!r}: {exc}"
        if require_files and not full.exists():
            return False, f"{role!r} references {rel!r}, which does not exist"
    if spec is not None:
        ok, why = spec.check(roles)
        if not ok:
            return False, why
    return True, f"{len(roles)} role(s), all inside {layout.rel_root}"
