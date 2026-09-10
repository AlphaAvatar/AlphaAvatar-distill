"""Where a run's files live in THIS repository.

`aadistill.runtime.run_layout` is the mechanism. It identifies a run by
`(experiment_id, run_id)` under a root the **caller** supplies, validates
containment, uniqueness and hashes, and deliberately names no directory: that is
what lets it carry a Stage-0 collection run, a Stage-3 recovery run, a rollout
benchmark and a Phase-C isolation session without a source change.

This module is that caller — the one place in AlphaAvatar-distill that says the
root is `logs/runs` and that a run's files are grouped into five areas:

    logs/runs/<experiment_id>/<run_id>/
        manifest.json     the run's index; written by the run itself
        governance/       what permitted the run: grant, authorization, bundle
        runtime/          how it executed: session record, launcher log, watchdog
        evidence/         what it observed: driver evidence, replay records
        artifacts/        what it produced or brought home
        closeout/         how it ended: outcome, cost, teardown confirmation

The areas are a repository convention and nothing more. **Which roles exist
inside them is the experiment's vocabulary**, declared by the experiment through
an `ArtifactSpec`: a Stage-0 activation-collection run has no `replay_record`,
and a Phase-C isolation session has no `activation_cache`. Neither this module
nor the core enumerates roles, so a new stage adds a declaration rather than a
branch here.

What this module does enforce is the part that is genuinely shared: every role
path sits under exactly one declared area, so a run cannot accumulate loose files
in its own root — which is the flat-`logs/` habit this convention exists to
replace — and a run that has already been recorded is never silently reopened.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.runtime.run_layout import (  # noqa: E402
    ArtifactSpec, RunLayout, build_run_manifest, verify_run_manifest,
)

#: This repository's run root, relative to the repository root. The only place
#: it is written down.
RUNS_ROOT = "logs/runs"

#: The run's own index, inside the run.
MANIFEST_NAME = "manifest.json"

#: The five areas, and what each one answers. A role path must begin with one of
#: them. Ordered as a run moves through them, which is also the order a reader
#: wants: what allowed it, how it ran, what it saw, what it made, how it ended.
AREAS: tuple[str, ...] = (
    "governance", "runtime", "evidence", "artifacts", "closeout")


class RunConventionError(RuntimeError):
    """A role path, or a run, that does not fit this repository's convention."""


def area_of(relative: str) -> str:
    """The area a role path belongs to. Raises if it belongs to none.

    `artifacts/` — a trailing slash — is a whole-directory role and is fine.
    `outcome.json` is not: a file directly in the run root has no owner, and
    `../x` is refused by the core before it can resolve anywhere.
    """
    if not isinstance(relative, str) or not relative or relative.startswith("/"):
        raise RunConventionError(f"{relative!r} is not a relative path")
    #: Checked BEFORE the area, because `governance/../manifest.json` starts with
    #: a declared area and lands on the run's own index. The core would allow it
    #: -- it resolves inside the run root, which is all the core promises -- so
    #: the area rule has to mean the area the path RESOLVES to, not the one it
    #: is spelled with.
    if any(part in ("..", ".") for part in relative.split("/")):
        raise RunConventionError(
            f"{relative!r} contains a traversal segment; a role names a path "
            "inside its area, not a route out of it")
    head, sep, _ = relative.partition("/")
    if head not in AREAS:
        raise RunConventionError(
            f"{relative!r} is not inside one of {list(AREAS)}. A run's own root "
            f"holds {MANIFEST_NAME} and nothing else, so that every file in a "
            "run has a declared owner")
    if not sep:
        raise RunConventionError(
            f"{relative!r} names the area itself; write {relative}/ for the whole "
            "directory, or name a path inside it")
    return head


def check_roles(roles: Mapping[str, str]) -> None:
    """Every declared role sits under exactly one area."""
    for role, rel in roles.items():
        area_of(rel)


def layout_for(repo_root: Path | str, experiment_id: str,
               run_id: str) -> RunLayout:
    """This run's layout. Creates nothing and checks nothing on disk."""
    return RunLayout(run_root=Path(repo_root) / RUNS_ROOT,
                     experiment_id=experiment_id, run_id=run_id)


def manifest_path(layout: RunLayout) -> Path:
    return layout.root / MANIFEST_NAME


def is_recorded(layout: RunLayout) -> bool:
    """Has this run already written its manifest?"""
    return manifest_path(layout).is_file()


def open_run(repo_root: Path | str, experiment_id: str, run_id: str, *,
             roles: Mapping[str, str]) -> RunLayout:
    """Create this run's directories, refusing to write into an occupied one.

    A recorded run is finished: its manifest names paths and a status that a
    second execution would invalidate silently. An *unrecorded* run that already
    holds files is worse — it is a run whose launcher died before it could write
    its manifest, and overwriting it destroys the only evidence of what
    happened. Both are refused.

    Every attempt this repository has run carries its own label (`attempt9`,
    `attempt3r`), so a collision is a mistake rather than a resumption, and there
    is no resume path to protect because none of these sessions has one. If one
    is ever added it reopens the run explicitly, rather than by this function
    forgetting to look.
    """
    check_roles(roles)
    layout = layout_for(repo_root, experiment_id, run_id)
    if is_recorded(layout):
        raise RunConventionError(
            f"{layout.rel_root} is already recorded ({MANIFEST_NAME} exists). "
            "Use a new run_id; a recorded run is not reopened, because its "
            "manifest describes an execution that this one would overwrite")
    occupied = sorted(p.relative_to(layout.root).as_posix()
                      for p in layout.root.rglob("*") if p.is_file()
                      ) if layout.root.exists() else []
    if occupied:
        raise RunConventionError(
            f"{layout.rel_root} already holds {len(occupied)} file(s) "
            f"({occupied[:3]}) and has no {MANIFEST_NAME}: a run whose launcher "
            "died before recording itself. Use a new run_id rather than writing "
            "over the only evidence of what happened")
    return layout.create(dict(roles))


def record_run(layout: RunLayout, *, spec: ArtifactSpec,
               plan: Mapping[str, Any], implementation: Mapping[str, Any],
               status: Mapping[str, Any], roles: Mapping[str, str],
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write this run's manifest, then verify it the way a reader will.

    Verification is not ceremony: `build_run_manifest` validates the declaration,
    while `verify_run_manifest` re-reads it against the filesystem and so is the
    step that notices a role pointing at a file the run never produced.
    """
    check_roles(roles)
    doc = build_run_manifest(layout, plan=plan, implementation=implementation,
                             status=status, roles=roles, spec=spec, extra=extra)
    path = manifest_path(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    ok, why = verify_run_manifest(doc, layout.run_root, spec=spec)
    if not ok:
        raise RunConventionError(f"{layout.rel_root}: {why}")
    return doc


def read_run(repo_root: Path | str, experiment_id: str,
             run_id: str) -> dict[str, Any]:
    """One recorded run's manifest, verified before it is returned."""
    layout = layout_for(repo_root, experiment_id, run_id)
    path = manifest_path(layout)
    if not path.is_file():
        raise RunConventionError(f"{layout.rel_root} has no {MANIFEST_NAME}")
    doc = json.loads(path.read_text())
    ok, why = verify_run_manifest(doc, layout.run_root)
    if not ok:
        raise RunConventionError(f"{layout.rel_root}: {why}")
    return doc


def present_roles(layout: RunLayout,
                  roles: Mapping[str, str]) -> dict[str, str]:
    """The declared roles whose paths actually exist.

    A session records what it produced, not what it hoped to: an aborted run has
    a session record and a launcher log and no evidence, and must still be able
    to write a manifest. Whether the surviving set is *complete* is the
    `ArtifactSpec`'s question, not this one's.

    A whole-directory role counts only when the directory has something in it.
    `open_run` creates every declared role's directory up front, so existence
    alone would report an empty `artifacts/` as an artifact the run produced.
    """
    out: dict[str, str] = {}
    for role, rel in roles.items():
        target = layout.path(rel)
        if target.is_dir():
            if any(target.rglob("*")):
                out[role] = rel
        elif target.exists():
            out[role] = rel
    return out


__all__ = ["AREAS", "MANIFEST_NAME", "RUNS_ROOT", "ArtifactSpec",
           "RunConventionError", "RunLayout", "area_of", "check_roles",
           "is_recorded", "layout_for", "manifest_path", "open_run",
           "present_roles", "read_run", "record_run"]
