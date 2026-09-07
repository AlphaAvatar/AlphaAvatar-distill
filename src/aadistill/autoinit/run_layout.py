"""Where a run's files live. Generic, future-only, and derived from config.

`logs/` has 240 top-level entries. Nine of them are C1 attempt directories, nine
more are the grants for those attempts, and the rest is everything else the
project has ever recorded — flat, sorted alphabetically, with `autoinit_c1_attempt3r`
next to `autoinit_causal_depth_pricing_bound.json`. Finding what an attempt
produced means knowing the naming convention that was in use when it ran, and
three conventions have been in use.

So runs get a layout:

    logs/runs/<experiment_id>/<attempt_id>/
        manifest.json
        governance/{grant,authorization,bundle}.json
        runtime/{session,provider}.json  runtime/status.jsonl
        evidence/{replay,treatment,decision}.json  evidence/{training,evaluation}/
        closeout/{outcome,teardown,budget}.json

Three properties are deliberate.

**Generic, not C1's.** There is no `C1Attempt10Paths`. `RunLayout` takes an
experiment id and an attempt id and knows nothing else; Stage 4 and Stage 5 use
the same class and the same component names. A layout that encodes one
experiment's vocabulary has to be rewritten for the next one, and then the two
disagree.

**No conditionals on the subject.** Model name, geometry, compression ratio,
stage number and attempt number never appear in a branch here. Components are a
declared mapping; manifest fields come from the caller's configuration. Anything
that must vary by experiment is data the caller supplies, so adding an experiment
is writing a config rather than editing this file.

**Future-only.** Attempts 1-9 are not moved, renamed or copied. They are
*registered* in `logs/runs/index.json` as `legacy-v1` references, with a digest
each, so the index can answer "what runs exist" without anything being rewritten
— and so a test can prove they stayed byte-for-byte identical. Moving completed
evidence to tidy a directory is how provenance gets lost; the first migration is
new evidence only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

RUN_LAYOUT_VERSION = 2
RUNS_ROOT = "logs/runs"
INDEX_PATH = f"{RUNS_ROOT}/index.json"
INDEX_SCHEMA = "aadistill.autoinit.run_index/v1"
MANIFEST_SCHEMA = "aadistill.autoinit.run_manifest/v2"

#: The layout, as data. A component is addressed by a dotted name and resolves to
#: exactly one path under the run root. Directory components end in "/".
#:
#: This is a mapping and not a set of methods on purpose: `layout.path(name)` is
#: total over this dict, so a new component is one line here and no branch
#: anywhere, and an unknown name is an error rather than a silently invented path.
COMPONENTS: dict[str, str] = {
    "manifest": "manifest.json",
    "governance.grant": "governance/grant.json",
    "governance.authorization": "governance/authorization.json",
    "governance.bundle": "governance/bundle.json",
    "runtime.session": "runtime/session.json",
    "runtime.provider": "runtime/provider.json",
    "runtime.status": "runtime/status.jsonl",
    "evidence.replay": "evidence/replay.json",
    "evidence.treatment": "evidence/treatment.json",
    "evidence.training": "evidence/training/",
    "evidence.evaluation": "evidence/evaluation/",
    "evidence.decision": "evidence/decision.json",
    "closeout.outcome": "closeout/outcome.json",
    "closeout.teardown": "closeout/teardown.json",
    "closeout.budget": "closeout/budget.json",
}

#: Components a manifest must reference. Everything else is optional because a
#: run that never reached that stage has nothing to point at — and a manifest
#: that names a file which does not exist is worse than one that omits it.
REQUIRED_MANIFEST_SECTIONS = ("governance", "runtime", "evidence", "closeout")

#: An id is a single path segment. Anything else is a traversal attempt or a
#: nested layout nobody declared.
_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class RunLayoutError(Exception):
    """The requested path is not inside the declared layout."""


def _check_id(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise RunLayoutError(
            f"{kind} {value!r} is not a valid identifier: lowercase letters, "
            "digits and underscores, starting with a letter or digit. A path "
            "separator, '..' or an absolute path is refused here rather than "
            "resolved into somewhere outside the run.")
    return value


@dataclass(frozen=True)
class RunLayout:
    """One run's directory. Construct with ids; ask it for components."""

    experiment_id: str
    attempt_id: str
    repo_root: Path = Path(".")
    runs_root: str = RUNS_ROOT
    layout_version: int = RUN_LAYOUT_VERSION

    def __post_init__(self) -> None:
        _check_id("experiment_id", self.experiment_id)
        _check_id("attempt_id", self.attempt_id)

    @property
    def rel_root(self) -> str:
        return f"{self.runs_root}/{self.experiment_id}/{self.attempt_id}"

    @property
    def root(self) -> Path:
        return Path(self.repo_root) / self.rel_root

    def path(self, component: str) -> Path:
        """The absolute path of a declared component. Total over COMPONENTS."""
        if component not in COMPONENTS:
            raise RunLayoutError(
                f"{component!r} is not a declared run component; the layout is "
                f"{sorted(COMPONENTS)}")
        resolved = (self.root / COMPONENTS[component].rstrip("/")).resolve()
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise RunLayoutError(
                f"{component!r} resolves to {resolved}, outside the run root "
                f"{root}")
        return resolved

    def rel(self, component: str) -> str:
        """Repo-relative, for writing into a manifest."""
        return f"{self.rel_root}/{COMPONENTS[component].rstrip('/')}"

    def contains(self, path: str | Path) -> bool:
        """Is `path` inside this run? Used to refuse dual-writes elsewhere."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(self.repo_root) / candidate
        try:
            candidate.resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        return True

    def create(self) -> "RunLayout":
        """Make the directory skeleton. Creates no files."""
        for spec in COMPONENTS.values():
            target = self.root / spec
            (target if spec.endswith("/") else target.parent).mkdir(
                parents=True, exist_ok=True)
        return self


def directory_digest(path: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    """sha256 over sorted 'relpath:sha256' lines. The same rule as the harness.

    A single value that changes if any byte of any file changes, if a file is
    added, or if one is removed — which is what "preserved byte-for-byte" has to
    mean for a directory.
    """
    root = Path(repo_root)
    target = root / path if not Path(path).is_absolute() else Path(path)
    if target.is_file():
        entries = [(target.name, hashlib.sha256(target.read_bytes()).hexdigest())]
    else:
        entries = sorted(
            (str(p.relative_to(target)),
             hashlib.sha256(p.read_bytes()).hexdigest())
            for p in target.rglob("*") if p.is_file())
    digest = hashlib.sha256(
        "".join(f"{name}:{sha}\n" for name, sha in entries).encode()).hexdigest()
    return {"digest": digest, "n_files": len(entries),
            "rule": "sha256 over sorted 'relpath:sha256' lines"}


def build_run_manifest(
    layout: RunLayout,
    *,
    config: Mapping[str, Any],
    present: Iterable[str] = (),
) -> dict[str, Any]:
    """A run manifest, entirely from `layout` and `config`.

    Every value below is either structural (derived from the layout) or supplied
    by the caller's configuration. Nothing branches on which experiment, model,
    geometry, stage or attempt this is — that is the property that lets Stage 4
    and Stage 5 use this unchanged.

    `present` names the components this run actually has, so the manifest points
    only at files that exist. A manifest naming an absent file is a broken
    reference; a manifest omitting a stage the run never reached is just true.
    """
    from ..infrastructure.manifest import sha256_json

    required = ("adapter", "source_spec_hash", "target_spec_hash",
                "compression", "operator_path", "protocols", "status")
    missing = [k for k in required if k not in config]
    if missing:
        raise RunLayoutError(
            f"the run configuration is missing {missing}. These are derived by "
            "the caller from its own frozen objects; this builder refuses to "
            "invent them, because a manifest that guesses an identity is not "
            "evidence of one.")

    present = sorted(set(present))
    unknown = [c for c in present if c not in COMPONENTS]
    if unknown:
        raise RunLayoutError(f"undeclared components {unknown}")

    def section(prefix: str) -> dict[str, str]:
        return {name.split(".", 1)[1]: layout.rel(name)
                for name in present
                if name.startswith(f"{prefix}.")}

    doc: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "layout_version": layout.layout_version,
        "_contract": (
            "The index of ONE run. Every reference is inside this run's own "
            "directory; nothing here is written anywhere else in logs/, and "
            "nothing here authorizes anything."),
        "experiment_id": layout.experiment_id,
        "attempt_id": layout.attempt_id,
        "root": layout.rel_root,
        "architecture_adapter": config["adapter"],
        "source_spec_hash": config["source_spec_hash"],
        "target_spec_hash": config["target_spec_hash"],
        "compression": config["compression"],
        "operator_path": config["operator_path"],
        "protocols": config["protocols"],
        "status": config["status"],
        "components": {s: section(s) for s in REQUIRED_MANIFEST_SECTIONS},
        "authorizes": "nothing",
    }
    for optional in ("notes", "supersedes", "budget"):
        if optional in config:
            doc[optional] = config[optional]
    doc["self_sha256"] = sha256_json(doc)
    return doc


def verify_run_manifest(doc: Mapping[str, Any], repo_root: str | Path = ".",
                        *, require_files: bool = True) -> tuple[bool, str]:
    """Is this manifest internally consistent and are its references real?"""
    from ..infrastructure.manifest import sha256_json

    if doc.get("schema") != MANIFEST_SCHEMA:
        return False, f"schema {doc.get('schema')!r} is not {MANIFEST_SCHEMA!r}"
    stated = doc.get("self_sha256")
    if stated != sha256_json({k: v for k, v in doc.items() if k != "self_sha256"}):
        return False, "the manifest does not match its own self_sha256"

    try:
        layout = RunLayout(doc["experiment_id"], doc["attempt_id"],
                           repo_root=Path(repo_root))
    except (RunLayoutError, KeyError) as exc:
        return False, f"unusable ids: {exc}"
    if doc.get("root") != layout.rel_root:
        return False, f"root {doc.get('root')!r} is not {layout.rel_root!r}"

    #: Collected in full BEFORE any check, deliberately. A single pass reports
    #: whichever problem it meets first, and for a duplicated path that depends
    #: on iteration order: if the wrong entry comes first it is reported as a
    #: wrong path and the duplicate — the actual defect, two owners for one file
    #: — is never named. Two passes make the diagnosis independent of dict order.
    declared: list[tuple[str, str]] = []
    for sname, section in (doc.get("components") or {}).items():
        for cname, rel in section.items():
            full = f"{sname}.{cname}"
            if full not in COMPONENTS:
                return False, f"{full!r} is not a declared component"
            declared.append((full, rel))

    owners: dict[str, list[str]] = {}
    for full, rel in declared:
        owners.setdefault(rel, []).append(full)
    for rel, claimants in owners.items():
        if len(claimants) > 1:
            a, b = sorted(claimants)[:2]
            return False, (f"{rel!r} is claimed by both {a!r} and {b!r}; "
                           "one path, one owner")

    for full, rel in declared:
        if rel != layout.rel(full):
            return False, f"{full!r} points at {rel!r}, not {layout.rel(full)!r}"
        if require_files and not (Path(repo_root) / rel).exists():
            return False, f"{full!r} references {rel!r}, which does not exist"
    return True, f"{len(declared)} components, all inside {layout.rel_root}"


def load_index(repo_root: str | Path = ".") -> dict[str, Any]:
    path = Path(repo_root) / INDEX_PATH
    if not path.is_file():
        return {"schema": INDEX_SCHEMA, "runs": []}
    return json.loads(path.read_text())


def resolve_run(entry: Mapping[str, Any], repo_root: str | Path = ".") -> tuple[bool, str]:
    """Does an index entry point at something that is really there?

    Handles both layout versions, because the index describes both: a v1 entry is
    a legacy reference validated by its recorded digest, a v2 entry resolves
    through its own manifest.
    """
    root = Path(repo_root)
    version = int(entry.get("layout_version", 0))
    target = root / entry["root"]
    if not target.exists():
        return False, f"{entry['root']} does not exist"
    if version == 1:
        live = directory_digest(entry["root"], root)
        if live["digest"] != entry.get("digest"):
            return False, (f"{entry['root']} has changed: recorded "
                           f"{str(entry.get('digest'))[:12]}…, live "
                           f"{live['digest'][:12]}…")
        return True, f"legacy-v1, {live['n_files']} files, unchanged"
    if version == RUN_LAYOUT_VERSION:
        layout = RunLayout(entry["experiment_id"], entry["attempt_id"],
                           repo_root=root)
        manifest = layout.path("manifest")
        if not manifest.is_file():
            return False, f"{entry['root']} has no manifest.json"
        return verify_run_manifest(json.loads(manifest.read_text()), root)
    return False, f"unknown layout_version {version!r}"
