"""Where a run's files live in THIS repository.

`aadistill.runtime.run_layout` is the mechanism. It identifies a run by
`(experiment_id, run_id)` under a root the **caller** supplies, validates
containment, uniqueness and hashes, and deliberately names no directory: that is
what lets it carry a Stage-0 collection run, a Stage-3 recovery run, a rollout
benchmark and a Phase-C isolation session without a source change.

This module is that caller — the one place in AlphaAvatar-distill that says the
root is `logs/runs` and that a run's files are grouped into five areas:

    logs/runs/stage-<stage_id>/<experiment_id>/<run_id>/
        manifest.json     the run's index; written by the run itself
        README.md         what this directory is; documentation, never evidence
        governance/       what permitted the run: grant, authorization, bundle
        runtime/          how it executed: session record, launcher log, watchdog
        evidence/         what it observed: driver evidence, replay records
        artifacts/        what it produced or brought home
        closeout/         how it ended: outcome, cost, teardown confirmation

The stage is DECLARED by the experiment's config and passed in; it is not
inferred from a name. Runs that predate the stage grouping stay at
`logs/runs/<experiment_id>/<run_id>/` and are found there — their grants,
readiness records and closeouts name those paths, and moving a run whose
identity was issued against its location breaks the lineage that makes it
evidence. One index reads both.

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
import re
import sys
from collections.abc import Iterable, Mapping
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

#: New runs are grouped by the pipeline stage their work belongs to:
#:
#:     logs/runs/stage-<stage_id>/<experiment_id>/<run_id>/
#:
#: The stage is DECLARED by the experiment's configuration and passed in. It is
#: never inferred from an experiment name, a directory name or the string "c1":
#: phase (A/B/C), pipeline stage (0-6) and attempt are three different
#: dimensions, and collapsing them is how `phase_c1` came to look like a stage.
#:
#: `shared` exists because not all managed work is a pipeline stage. Storage
#: maintenance, CUDA engineering validation and cross-stage infrastructure runs
#: get an explicit bucket rather than being filed under whichever stage they
#: touched last.
STAGE_PREFIX = "stage-"

#: A pipeline stage id is a decimal number. NOT a fixed list: `range(7)` encoded
#: "this project has stages 0-6", which is a fact about today's AGENTS.md rather
#: than about the mechanism, and it would have refused a Stage 7 that a future
#: config declared -- forcing a public-code edit to add a stage, which is the
#: opposite of declaring one. A wider fixed range would have the same shape.
#:
#: What the pattern still buys is path safety: one path segment, digits only, so
#: a stage id can never introduce a separator, a traversal or a hidden file.
PIPELINE_STAGE = re.compile(r"^(0|[1-9][0-9]*)$")

#: Buckets for work that is not a pipeline stage at all. Named, because these
#: are not numbers and must not be invented ad hoc.
#: `shared` is work that is not a pipeline stage at all -- engineering
#: validation, storage maintenance. `unscoped` is a run whose pipeline stage no
#: frozen record DETERMINES: the log-layout-v1 migration put every legacy run
#: there rather than guess, because only `phase_c1` declares a stage and the
#: other experiments' frozen records describe DRIVER stages, a different
#: dimension. A stage that is merely unknown must not be invented.
NON_PIPELINE_STAGES = ("shared", "unscoped")

#: The run's own index, inside the run.
MANIFEST_NAME = "manifest.json"

#: The one file permitted in a run's own root beside its manifest.
#:
#: Named explicitly, not matched by extension. "Ignore Markdown" would let any
#: .md file accumulate unowned in a run root, which is the habit the five areas
#: exist to prevent; this is one filename, and `open_run` still refuses every
#: other pre-existing file.
README_NAME = "README.md"

#: The five areas, and what each one answers. A role path must begin with one of
#: them. Ordered as a run moves through them, which is also the order a reader
#: wants: what allowed it, how it ran, what it saw, what it made, how it ended.
AREAS: tuple[str, ...] = (
    "governance", "runtime", "evidence", "artifacts", "closeout")


#: Marks a writable output location as belonging to exactly one run.
#:
#: A run's *directory* under `logs/runs` is not the only place it writes. The
#: session's scratch root is where the launcher log, the relayed driver streams
#: and the collected artifact manifest actually accumulate, and it is passed in
#: by the operator, created with `exist_ok=True`, and shared with read-only
#: inputs and caches. Nothing tied it to a run, so a fresh run id aimed at a
#: previous attempt's scratch collected that attempt's evidence as its own.
CLAIM_NAME = ".aad_output_claim.json"
CLAIM_SCHEMA = "aadistill.run_output_claim/v1"


class RunConventionError(RuntimeError):
    """A role path, or a run, that does not fit this repository's convention."""


class OutputOwnershipError(RunConventionError):
    """An output location belongs to a different run, or to nobody known.

    A subclass because callers refuse on both for the same reason, and because
    the two are genuinely different findings: a foreign claim is a collision,
    an unclaimed directory that already holds outputs is UNKNOWN ownership.
    Neither is resolved by guessing.
    """


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
    #: The run's own README is the ONE root file besides the manifest. It is
    #: matched by exact name: a rule that ignored `*.md` would let any Markdown
    #: accumulate in a run root unowned, which is what the five areas exist to
    #: prevent. A README describes the directory; it is never evidence, never a
    #: product, and never proof that the run executed.
    if relative == README_NAME:
        return "root"
    head, sep, _ = relative.partition("/")
    if head not in AREAS:
        raise RunConventionError(
            f"{relative!r} is not inside one of {list(AREAS)}. A run's own root "
            f"holds {MANIFEST_NAME}, {README_NAME} and nothing else, so that "
            "every file in a run has a declared owner")
    if not sep:
        raise RunConventionError(
            f"{relative!r} names the area itself; write {relative}/ for the whole "
            "directory, or name a path inside it")
    return head


def check_roles(roles: Mapping[str, str]) -> None:
    """Every declared role sits under exactly one area."""
    for role, rel in roles.items():
        area_of(rel)


def stage_segment(stage_id: str) -> str:
    """`stage-3`, `stage-shared`. Refuses anything undeclared.

    A stage id is a DECLARATION, so an unrecognised one is an error rather than
    a new directory: the failure mode this prevents is a typo silently creating
    `logs/runs/stage-c1/` and a second place for one experiment's runs to live.
    """
    if not isinstance(stage_id, str) or not (
            PIPELINE_STAGE.match(stage_id) or stage_id in NON_PIPELINE_STAGES):
        raise RunConventionError(
            f"stage_id {stage_id!r} is not a pipeline stage number or one of "
            f"{list(NON_PIPELINE_STAGES)}. A pipeline stage is a decimal number "
            "declared by the experiment -- this mechanism does not cap which "
            "ones exist, so a stage the pipeline grows later needs a config "
            "entry and no change here. Work that is not a pipeline stage at all "
            "-- storage maintenance, engineering validation, cross-stage "
            "infrastructure -- is declared 'shared' rather than filed under a "
            "stage it merely touched. Phase (A/B/C) and attempt are different "
            "dimensions and are not stage ids.")
    return f"{STAGE_PREFIX}{stage_id}"


def runs_root_for(repo_root: Path | str, stage_id: str | None) -> Path:
    """Where runs of a given stage live.

    `stage_id=None` is the LEGACY root, `logs/runs/<experiment_id>/<run_id>/`.
    It exists to read attempts 1-12 and the CUDA subruns where they are: those
    have issued or frozen identities, and moving a run whose grant, readiness
    record and closeout all name its path would break the lineage that makes it
    evidence. New runs declare a stage; old runs are found, not relocated.
    """
    base = Path(repo_root) / RUNS_ROOT
    return base if stage_id is None else base / stage_segment(stage_id)


def rel_run_dir(experiment_id: str, run_id: str,
                stage_id: str | None = None) -> str:
    """A run's directory as a REPO-RELATIVE path, stage segment included.

    The one string an operator is told and a gate computes. It existed as
    `f"{RUNS_ROOT}/{experiment_id}/{run_id}"` at four call sites, and when runs
    were grouped by stage all four silently kept naming the pre-stage location.
    Three were messages. The fourth was `grant_provenance_gate`, which computes
    where THIS session's grant must be: it would have looked under the legacy
    root while `open_run` put the run under the stage, so a correctly placed
    grant is refused and a grant placed where the gate looks is outside the run
    it belongs to. That is a `$0` refusal at gate time -- but only if it is
    noticed before a launch, and it consumes a one-use chain either way.
    """
    stage = "" if stage_id is None else f"{stage_segment(stage_id)}/"
    return f"{RUNS_ROOT}/{stage}{experiment_id}/{run_id}"


def layout_for(repo_root: Path | str, experiment_id: str, run_id: str,
               stage_id: str | None = None) -> RunLayout:
    """This run's layout. Creates nothing and checks nothing on disk.

    `stage_id` is the experiment's declaration. It defaults to None so the
    legacy root stays readable by every existing caller, and so that a caller
    which has not yet declared a stage fails by writing where it always did
    rather than by writing somewhere new and unindexed.
    """
    return RunLayout(run_root=runs_root_for(repo_root, stage_id),
                     experiment_id=experiment_id, run_id=run_id)


def manifest_path(layout: RunLayout) -> Path:
    return layout.root / MANIFEST_NAME


def is_recorded(layout: RunLayout) -> bool:
    """Has this run already written its manifest?"""
    return manifest_path(layout).is_file()


def open_run(repo_root: Path | str, experiment_id: str, run_id: str, *,
             roles: Mapping[str, str],
             prepared: Iterable[str] = (),
             stage_id: str | None = None) -> RunLayout:
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

    **`prepared` names roles written before the run opens, by someone else.**
    Not every file in a run is produced by the run: a maintainer's grant is
    authored and committed *while the tree is still clean*, because the
    launch-bound readiness sweep and the authorization that follows it both
    require it to be there already. Without this parameter the only two places
    such an input could go were a flat `logs/<experiment>_attemptN_grant.json` —
    the habit this convention exists to end — or nowhere, because the occupancy
    rule could not tell a declared, expected input from the residue of a dead
    launcher.

    It is an exemption from **that one rule** and nothing else. A prepared role
    must be declared in `roles`, so it lands in the manifest and has an owner; a
    name that is not a declared role is a typo, and silently widening the
    exemption to cover it is how it would stop meaning anything. Everything else
    in the run root still refuses: a recorded run, an undeclared file, a
    half-written evidence tree. Nothing here reads, validates or trusts the
    prepared file — whether it is the *right* grant is the caller's gate to
    apply, and C1 applies it against the authorization that names its hash.
    """
    check_roles(roles)
    prepared = tuple(prepared)
    unknown = sorted(set(prepared) - set(roles))
    if unknown:
        raise RunConventionError(
            f"prepared role(s) {unknown} are not declared in `roles`. A "
            "prepared input is exempted from the occupancy rule, so it has to "
            "be a role this run will record an owner for; an undeclared name "
            "would exempt a path nothing is accountable for")
    layout = layout_for(repo_root, experiment_id, run_id, stage_id)
    if is_recorded(layout):
        raise RunConventionError(
            f"{layout.rel_root} is already recorded ({MANIFEST_NAME} exists). "
            "Use a new run_id; a recorded run is not reopened, because its "
            "manifest describes an execution that this one would overwrite")
    #: A whole-directory role (`artifacts/`) exempts what is under it; a file
    #: role exempts exactly itself.
    exempt_files = {roles[r] for r in prepared if not roles[r].endswith("/")}
    exempt_trees = tuple(roles[r] for r in prepared if roles[r].endswith("/"))
    #: And the README, by name. A directory that already carries its own
    #: description is not a dead launcher's residue -- but this exempts exactly
    #: one filename, so a run root holding anything else is still refused, and a
    #: run holding ONLY a README is still an unexecuted run rather than a
    #: failed one.
    exempt_files.add(README_NAME)
    occupied = sorted(
        rel for rel in (p.relative_to(layout.root).as_posix()
                        for p in layout.root.rglob("*") if p.is_file())
        if rel not in exempt_files
        and not any(rel.startswith(t) for t in exempt_trees)
    ) if layout.root.exists() else []
    if occupied:
        raise RunConventionError(
            f"{layout.rel_root} already holds {len(occupied)} undeclared "
            f"file(s) ({occupied[:3]}) and has no {MANIFEST_NAME}: a run whose "
            "launcher died before recording itself. Use a new run_id rather "
            "than writing over the only evidence of what happened"
            + (f" ({len(prepared)} prepared role(s) were exempt and are not "
               "counted here)" if prepared else ""))
    return layout.create(dict(roles))


def read_output_claim(root: Path | str) -> dict[str, Any] | None:
    """The claim on this output location, or None if it carries none."""
    path = Path(root) / CLAIM_NAME
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise OutputOwnershipError(
            f"{path} is not readable as a claim ({exc}); ownership of this "
            "output location is unknown and it is not safe to write into")
    if doc.get("schema") != CLAIM_SCHEMA:
        raise OutputOwnershipError(
            f"{path} carries schema {doc.get('schema')!r}, not {CLAIM_SCHEMA!r}")
    return doc


def claim_output_root(root: Path | str, experiment_id: str, run_id: str, *,
                      outputs: Iterable[str]) -> dict[str, Any]:
    """Claim a writable output location for this run, or refuse.

    Three cases, and the middle one is the defect this exists for:

    * claimed by THIS run — idempotent, so a resumed or re-entered launcher
      does not trip over its own marker;
    * claimed by another run — refused, naming the owner;
    * unclaimed but already holding one of this run's declared `outputs` —
      refused as UNKNOWN ownership. Not "probably stale", not "older than the
      session": mtime is a guess, and the alternative to guessing is to say so.

    `outputs` is what the run will write and later collect — NOT everything in
    the directory. A scratch root legitimately holds shared read-only inputs
    and a model cache, and refusing on those would make the rule unusable, so
    they neither claim nor block.

    Writes only `CLAIM_NAME`. Nothing here deletes, moves or rewrites a
    pre-existing file: a refused location is left exactly as it was found.
    """
    root = Path(root)
    declared = sorted(set(outputs))
    existing = read_output_claim(root) if root.exists() else None
    if existing is not None:
        owner = (existing.get("experiment_id"), existing.get("run_id"))
        if owner != (experiment_id, run_id):
            raise OutputOwnershipError(
                f"{root} is the output location of {owner[0]}/{owner[1]}, not "
                f"{experiment_id}/{run_id}. Two runs must not share a writable "
                "output root: the second would collect the first's evidence as "
                "its own. Use a fresh scratch directory")
        return existing

    #: An entry containing `*` is a pattern, because some outputs are named
    #: after a resource that does not exist when the claim is made -- a
    #: watchdog journal carries its pod id, and a session may hold several
    #: resources in turn. Without this the journals stopped counting as
    #: evidence that a scratch root belongs to a run.
    present = [rel for rel in declared
               if (any(root.glob(rel)) if "*" in rel else (root / rel).exists())]
    if present:
        raise OutputOwnershipError(
            f"{root} already holds {len(present)} of this run's declared "
            f"outputs ({present[:3]}) and carries no claim, so it belongs to "
            "an execution this run cannot identify. It is left untouched; use "
            "a fresh scratch directory rather than writing over evidence whose "
            "owner is unknown")

    root.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": CLAIM_SCHEMA,
        "_contract": ("This directory is the writable OUTPUT root of exactly "
                      "one run. Shared read-only inputs and caches may live "
                      "here too; only the declared outputs below decide "
                      "ownership. AUTHORIZES NOTHING."),
        "experiment_id": experiment_id,
        "run_id": run_id,
        "declared_outputs": declared,
    }
    (root / CLAIM_NAME).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def require_output_claim(root: Path | str, experiment_id: str,
                         run_id: str) -> dict[str, Any]:
    """Refuse unless this output location is demonstrably this run's.

    Asked again at closeout rather than assumed from the open: the two happen
    at opposite ends of a session, and what is collected has to be what THIS
    execution produced.
    """
    claim = read_output_claim(root)
    if claim is None:
        raise OutputOwnershipError(
            f"{root} carries no output claim, so nothing in it can be shown to "
            f"belong to {experiment_id}/{run_id}; refusing to collect it")
    owner = (claim.get("experiment_id"), claim.get("run_id"))
    if owner != (experiment_id, run_id):
        raise OutputOwnershipError(
            f"{root} is claimed by {owner[0]}/{owner[1]}; refusing to collect "
            f"its contents as {experiment_id}/{run_id}'s evidence")
    return claim


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


def read_run(repo_root: Path | str, experiment_id: str, run_id: str,
             stage_id: str | None = None) -> dict[str, Any]:
    """One recorded run's manifest, verified before it is returned."""
    layout = layout_for(repo_root, experiment_id, run_id, stage_id)
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
            #: A README does not count as content. It describes the area; it is
            #: not something the run produced, and a whole-directory role whose
            #: only file is its own description would otherwise be reported as
            #: an artifact the run made.
            if any(q.name != README_NAME for q in target.rglob("*") if q.is_file()):
                out[role] = rel
        elif target.exists():
            out[role] = rel
    return out


#: What each area is for, in one line. The run-level README links the manifest;
#: these say what belongs in the directory they sit in, so a reader who opens
#: `evidence/` knows why it is not `artifacts/`.
AREA_PURPOSE: dict[str, str] = {
    "governance": ("what permitted this run: the maintainer grant, the one-use "
                   "authorization, the readiness record and the bundle record. "
                   "INPUTS and one-use snapshots, not products."),
    "runtime": ("how it executed: the session record the runner writes on every "
                "path, the launcher console, the watchdog journal."),
    "evidence": ("what it observed: driver evidence, replay records, the marker "
                 "stream. Observations, not conclusions."),
    "artifacts": ("what it produced or brought home. Large objects stay in the "
                  "session scratch and are referenced by hash from "
                  "`artifacts/manifest.json`; this directory holds reviewable "
                  "text."),
    "closeout": ("how it ended: the outcome classification, the measured cost "
                 "and the provider teardown confirmation."),
}


def write_run_readmes(layout: RunLayout, *, experiment_id: str, run_id: str,
                      stage_id: str | None, roles: Mapping[str, str]) -> list[str]:
    """Describe this run's directories. Documentation only.

    A README here is never evidence, never a product and never proof that the
    run executed: `present_roles` ignores it, `record_run_index` does not count
    a README-only directory as a run, and nothing derived from a run reads one.

    It also does not restate facts that have an owner. No cumulative cost, no
    current `main` SHA, no authorization status, no result -- those live in the
    ledger, in git and in the manifest, and a second hand-maintained copy is how
    they go stale. What it carries is what this directory is for, and where the
    canonical index is.
    """
    written: list[str] = []
    where = f"stage-{stage_id}/" if stage_id else ""
    rows = "".join(f"| `{a}/` | {AREA_PURPOSE[a]} |\n" for a in AREAS)
    (layout.root / README_NAME).write_text(
        f"# {experiment_id} / {run_id}\n"
        f"\n"
        f"One run of `{experiment_id}`, at "
        f"`logs/runs/{where}{experiment_id}/{run_id}/`.\n"
        f"\n"
        f"**Canonical index: `{MANIFEST_NAME}` in this directory.** It names "
        f"every role this run recorded, its status and its cost. Read it rather "
        f"than this file for anything factual: this README describes the layout "
        f"and is not evidence that the run executed, succeeded or was "
        f"authorized.\n"
        f"\n"
        f"| area | what is in it |\n"
        f"| --- | --- |\n"
        f"{rows}"
        f"\n"
        f"The repository-wide index of every run is `logs/runs/index.json`.\n")
    written.append(README_NAME)
    for area in AREAS:
        d = layout.root / area
        if not d.is_dir():
            continue
        mine = sorted(r for r, rel in roles.items() if rel.split("/")[0] == area)
        declares = ("Roles this run declares here: "
                    + ", ".join(f"`{m}`" for m in mine) + ".\n"
                    if mine else
                    "This run declares no role in this area.\n")
        (d / README_NAME).write_text(
            f"# {area}/\n"
            f"\n"
            f"{AREA_PURPOSE[area]}\n"
            f"\n"
            f"{declares}"
            f"\n"
            f"Described by `../{MANIFEST_NAME}`, which is authoritative.\n")
        written.append(f"{area}/{README_NAME}")
    return written


__all__ = ["AREAS", "CLAIM_NAME", "CLAIM_SCHEMA", "MANIFEST_NAME",
           "RUNS_ROOT", "ArtifactSpec", "OutputOwnershipError",
           "RunConventionError", "RunLayout", "area_of", "check_roles",
           "claim_output_root", "is_recorded", "layout_for", "manifest_path",
           "open_run", "present_roles", "read_output_claim", "read_run",
           "rel_run_dir",
           "record_run", "require_output_claim"]
