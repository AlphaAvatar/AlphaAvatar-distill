#!/usr/bin/env python3
"""One index entry per logical RUN, not per artifact root.

    PYTHONPATH=src python scripts/architecture/record_run_index.py --write

The first version of this index counted 77 "runs". It was counting artifact
roots: `logs/stages/stage-1/phase_c1/runs/attempt9` and `logs/budget/approvals/autoinit_c1_attempt9_grant.json`
are one attempt with two surviving components, and the index recorded them as
two peers. A reader asking "how many C1 attempts have run?" got 19 for a phase
that has had 9.

So an entry is now keyed by `(experiment_id, run_id)` and carries a
**components** map — role -> existing path — plus a digest per component and one
aggregate over all of them. Nothing is moved, renamed or copied; every path is
exactly where it has always been. The correction is to the schema and the count,
which is what the maintainer approved under P12.

Roles here are the vocabulary of the legacy artifacts, discovered by pattern, so
that adding a convention means writing the rule down rather than remembering to
append a path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.runtime.run_layout import digest_of  # noqa: E402

#: At the root of `logs/`: the index spans every stage.
OUT = "logs/index.json"
SCHEMA = "aadistill.runtime.run_index/v2"

#: (pattern, experiment_id, run_id template, role). A run is the (experiment,
#: run) pair; every match contributes a COMPONENT to it. Directories and files
#: alike — `..._grant.json` is the `grant` role of the same run whose evidence
#: directory is the `root` role.
CONVENTIONS: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"^autoinit_c1_attempt(?P<n>\d+r?)$"), "phase_c1", "attempt{n}", "root"),
    (re.compile(r"^autoinit_c1_attempt(?P<n>\d+r?)_grant\.json$"), "phase_c1", "attempt{n}", "grant"),
    (re.compile(r"^autoinit_continuation_b_attempt(?P<n>\d+)$"), "continuation_b", "attempt{n}", "root"),
    (re.compile(r"^autoinit_continuation_b_attempt(?P<n>\d+)\.json$"), "continuation_b", "attempt{n}", "outcome"),
    (re.compile(r"^autoinit_recovery_continuation_attempt(?P<n>\d+)$"), "recovery_continuation", "attempt{n}", "root"),
    (re.compile(r"^autoinit_recovery_continuation_attempt(?P<n>\d+)_grant\.json$"), "recovery_continuation", "attempt{n}", "grant"),
    (re.compile(r"^autoinit_attempt(?P<n>\d+)_probe_reuse\.json$"), "recovery_continuation", "attempt{n}", "probe_reuse"),
    (re.compile(r"^autoinit_attempt(?P<n>\d+)_retention_verification\.json$"), "recovery_continuation", "attempt{n}", "retention"),
    (re.compile(r"^autoinit_phase_a_attempt(?P<n>\d+)$"), "phase_a", "attempt{n}", "root"),
    (re.compile(r"^autoinit_phase_a_attempt(?P<n>\d+)_grant\.json$"), "phase_a", "attempt{n}", "grant"),
    (re.compile(r"^autoinit_phase_b_attempt(?P<n>\d+)$"), "phase_b", "attempt{n}", "root"),
    (re.compile(r"^autoinit_phase_b_attempt(?P<n>\d+)\.json$"), "phase_b", "attempt{n}", "outcome"),
    (re.compile(r"^autoinit_phase_b_grant_attempt(?P<n>\d+)\.json$"), "phase_b", "attempt{n}", "grant"),
    (re.compile(r"^autoinit_device_canary_attempt(?P<n>\d+)$"), "device_canary", "attempt{n}", "root"),
    (re.compile(r"^autoinit_measurement_attempt(?P<n>\d+)$"), "measurement", "attempt{n}", "root"),
    #: Aggregate directories from an earlier convention: several attempts under
    #: one directory. Registered whole rather than split, because splitting
    #: means moving files.
    (re.compile(r"^autoinit_continuation_attempts$"), "recovery_continuation", "legacy_aggregate", "root"),
    (re.compile(r"^autoinit_phase_a_attempts$"), "phase_a", "legacy_aggregate", "root"),
]


#: Text the published table carries, so a reader meeting it in the index knows
#: what it is for and what it deliberately is not.
HISTORICAL_PATHS_NOTE = (
    "old path -> where that object is now. A frozen payload -- a consumed "
    "authorization, a closed run's manifest, a hash-anchored amendment -- names "
    "the path an object had when it was written. That statement stays true and "
    "is never rewritten, so a consumer reading one needs a way from that name "
    "to the current address. This is that way, and only that: paths, no "
    "payload, no hashes, no per-move prose. Git history holds the trees.")

HISTORICAL_PATHS_RESOLVE = (
    "exact key first; otherwise the longest key that is a parent directory of "
    "the query, with the remainder appended. No match means the path is "
    "current, or names an object that no longer exists.")

HISTORICAL_PATHS_MAINTENANCE = (
    "carried forward by this builder and extended by hand when an object "
    "moves. Nothing derives it: once a move is made, only the person making it "
    "knows the pairing, which is why it is recorded at the time rather than "
    "reconstructed later.")


def historical_paths(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """The old-path table the index carries.

    It lives in the index rather than in this file because 300 string pairs are
    data, and because the index is what a consumer already reads. Four
    migration manifests used to hold the same pairs alongside per-file digests
    and prose -- a second, editable copy of trees git already has.
    """
    p = repo_root / OUT
    if not p.is_file():
        return {}
    block = json.loads(p.read_text()).get("historical_paths") or {}
    return dict(block.get("map") or {})


def resolve_historical(rel: str, repo_root: Path = REPO_ROOT) -> str:
    """`rel` as it is addressed today; `rel` itself when nothing moved it."""
    table = historical_paths(repo_root)
    if rel in table:
        return table[rel]
    for old in sorted(table, key=len, reverse=True):
        if rel.startswith(old + "/"):
            return table[old] + rel[len(old):]
    return rel


def discover_legacy(repo_root: Path) -> dict[tuple[str, str], dict]:
    """Runs named by the FILENAME each had at `logs/` root.

    Those objects sit in the canonical layout now, so matching on the current
    tree finds nothing and every legacy run silently becomes `unrecorded` --
    losing the per-component digests that make "the historical evidence is
    unchanged" checkable. The historical-path table is how a name from then
    resolves to a path now.
    """
    runs: dict[tuple[str, str], dict] = {}
    moved = historical_paths(repo_root)
    #: The names as they were, resolved to where they are.
    candidates = [(Path(old).name, new) for old, new in moved.items()
                  if Path(old).parent.as_posix() == "logs"]
    candidates += [(q.name, q.relative_to(repo_root).as_posix())
                   for q in sorted((repo_root / "logs").iterdir())]
    for name, current in candidates:
        for pattern, experiment, tpl, role in CONVENTIONS:
            m = pattern.match(name)
            if not m:
                continue
            run_id = tpl.format(**m.groupdict())
            key = (experiment, run_id)
            rec = runs.setdefault(key, {
                "experiment_id": experiment, "run_id": run_id,
                "layout_version": 1, "components": {}, "component_digests": {},
            })
            rel = current
            if role in rec["components"]:
                raise SystemExit(
                    f"two paths claim role {role!r} of {experiment}/{run_id}: "
                    f"{rec['components'][role]} and {rel}. One path, one owner — "
                    "add a distinct role rather than overwriting.")
            rec["components"][role] = rel
            rec["component_digests"][role] = digest_of(repo_root / rel)["digest"]
            break
    import hashlib

    for rec in runs.values():
        rec["components"] = dict(sorted(rec["components"].items()))
        rec["component_digests"] = dict(sorted(rec["component_digests"].items()))
        rec["aggregate_digest"] = hashlib.sha256("".join(
            f"{role}:{rec['components'][role]}:{rec['component_digests'][role]}\n"
            for role in rec["components"]).encode()).hexdigest()
        rec["n_components"] = len(rec["components"])
    return runs


def discover_v3(repo_root: Path) -> list[dict]:
    """Hierarchical runs, found by the presence of a manifest."""
    from aadistill.runtime.run_layout import MANIFEST_SCHEMA

    #: STAGE-FIRST:
    #:     logs/stages/stage-<id>/<experiment>/runs/<run>/
    #: There is no `logs/runs/`. An experiment's runs sit beside its plans and
    #: results rather than in a parallel tree.
    #:
    #: `logs/cross-stage/` DOES NOT EXIST and is not created: every experiment
    #: resolved to exactly one stage. It is still scanned because
    #: `experiments.run_layout` still composes that path for `stage_id=None`,
    #: and that module is covered by the frozen C1 harness digest -- removing
    #: the fallback would move the digest and invalidate an authorized
    #: preregistration to tidy a directory that is already gone. A scan of a
    #: directory that does not exist costs nothing; an unindexed run would not.
    stages_root = repo_root / "logs/stages"
    cross_root = repo_root / "logs/cross-stage"
    out: list[dict] = []
    found: set = set()
    if stages_root.is_dir():
        found |= set(stages_root.glob("*/*/runs/*/manifest.json"))
    if cross_root.is_dir():
        found |= set(cross_root.glob("*/runs/*/manifest.json"))
    for manifest in sorted(found):
        doc = json.loads(manifest.read_text())
        if doc.get("schema") != MANIFEST_SCHEMA:
            continue
        rel = manifest.parent.relative_to(repo_root).as_posix()
        #: `.../stage-<id>/<experiment>/runs/<run>/manifest.json`
        parts = manifest.parent.relative_to(repo_root).parts
        stage = parts[2] if parts[1] == "stages" else None
        out.append({
            "experiment_id": doc["experiment_id"], "run_id": doc["run_id"],
            "stage": stage.removeprefix("stage-") if stage else None,
            "layout": "stage_grouped" if stage else "legacy_two_level",
            "layout_version": doc["layout_version"],
            "components": {"root": rel, "manifest": f"{rel}/manifest.json"},
            "manifest_sha256": doc["self_sha256"],
            "n_components": len(doc.get("roles") or {}),
        })
    return out


def discover_unrecorded(repo_root: Path) -> list[dict]:
    """Run directories under `logs/runs` that hold files and no valid manifest.

    The index used to skip these silently, and it was not a hypothetical gap:
    the three 2026-09-10 CUDA stage-F subruns each wrote a real directory here
    and none of them wrote a manifest, so an index whose contract reads "every
    run this repository has recorded" reported `runs_current: 0` while three
    runs sat on disk. Silence is the wrong answer twice over — it also hides the
    case that matters operationally, a launcher that died before it could record
    itself.

    They are reported, with their digest, and NOT promoted into `runs`: a run
    that never declared its own roles has not been recorded, and back-filling a
    manifest from the outside would be this index inventing evidence about an
    execution it did not watch.
    """
    from aadistill.runtime.run_layout import MANIFEST_SCHEMA

    #: Same roots as `discover_v3`, and for the same reason: `cross-stage/`
    #: does not exist, and is scanned so that a run written there by the
    #: retained `stage_id=None` path could not go unreported.
    stages_root = repo_root / "logs/stages"
    cross_root = repo_root / "logs/cross-stage"
    out: list[dict] = []
    if not stages_root.is_dir() and not cross_root.is_dir():
        return out
    #: A stage directory is not itself a run: `stage-3` holds experiments, so
    #: only `stage-<id>/<experiment>/runs/<run>` is a candidate.
    candidates = {p for p in stages_root.glob("*/*/runs/*") if p.is_dir()}
    candidates |= {p for p in cross_root.glob("*/runs/*") if p.is_dir()}
    for run_dir in sorted(candidates):
        manifest = run_dir / "manifest.json"
        if manifest.is_file():
            try:
                if json.loads(manifest.read_text()).get("schema") == MANIFEST_SCHEMA:
                    continue
            except json.JSONDecodeError:
                pass
        files = [p for p in run_dir.rglob("*") if p.is_file()]
        #: A directory holding only its own README has not executed and has not
        #: failed. Counting it as an unrecorded run would report a launcher that
        #: died where nothing ever started.
        described_only = {p.name for p in files} == {"README.md"}
        if not files or described_only:
            continue
        rel = run_dir.relative_to(repo_root).as_posix()
        #: A run whose only files are governance inputs has not executed — it is
        #: PREPARED. A maintainer grant is committed into the run before the
        #: launch-bound sweep, so this state is now reachable on purpose, and
        #: reporting it as "the launcher did not reach its closeout" would be a
        #: false statement about a session that never started. Derived from the
        #: convention's own area vocabulary, so no role name is interpreted here.
        areas = {Path(p).relative_to(run_dir).parts[0] for p in files
                 if len(Path(p).relative_to(run_dir).parts) > 1}
        prepared_only = areas == {"governance"}
        #: A run that never opened still owes an account of itself, and the
        #: closeout is where it gives one. Read it rather than guessing: with a
        #: closeout present, "the launcher did not reach its closeout" is the
        #: one thing that is certainly false. Attempt 13 aborted inside
        #: `open_run`, so no manifest exists and none is invented -- but its
        #: closeout says why, and this reports what the closeout says.
        closeout = run_dir / "closeout/outcome.json"
        stated = None
        if closeout.is_file():
            try:
                stated = json.loads(closeout.read_text()).get("classification")
            except (json.JSONDecodeError, OSError):
                stated = None
        out.append({
            #: `.../<experiment>/runs/<run>`: the experiment is the
            #: grandparent. Reading the parent gave the literal "runs", which
            #: then appeared in the index as an experiment of that name.
            "experiment_id": run_dir.parent.parent.name, "run_id": run_dir.name,
            "root": rel, "n_files": len(files),
            "digest": digest_of(run_dir)["digest"],
            "why": ("no run manifest, and only governance inputs are present: "
                    "PREPARED but not executed" if prepared_only else
                    f"no run manifest; its closeout states: {stated}" if stated
                    else
                    "no valid run manifest; predates the run-manifest convention "
                    "or the launcher did not reach its closeout"),
        })
    return out


def build_index(repo_root: Path) -> dict:
    legacy = sorted(discover_legacy(repo_root).values(),
                    key=lambda r: (r["experiment_id"], r["run_id"]))
    modern = discover_v3(repo_root)
    #: A legacy run REGISTERED with its component digests is not also an
    #: unrecorded directory. Before the layout migration the two could not
    #: overlap -- legacy runs lived at `logs/` root and this scan looked only
    #: under `logs/runs/`. They live under `logs/runs/` now, so every legacy
    #: run was being counted twice, once with its digests and once as evidence
    #: nobody had registered.
    registered = {rel for r in legacy for rel in r["components"].values()}
    unrecorded = [u for u in discover_unrecorded(repo_root)
                  if u["root"] not in registered]
    return {
        "schema": SCHEMA,
        "_contract": (
            "Every run this repository has recorded, ONE ENTRY PER LOGICAL RUN. "
            "A legacy run references its surviving components in place: nothing "
            "was moved, renamed or copied, and each component carries a digest "
            "so that is checkable. Registering a run confers nothing. A run "
            "directory that exists but recorded no manifest is reported under "
            "`unrecorded`, never silently dropped."),
        "granularity": ("one entry per (experiment_id, run_id). The v1 index "
                        "counted artifact roots, so an attempt with an evidence "
                        "directory and a grant file appeared twice."),
        "historical_paths": {
            "_what_this_is": HISTORICAL_PATHS_NOTE,
            "_resolve": HISTORICAL_PATHS_RESOLVE,
            "_maintenance": HISTORICAL_PATHS_MAINTENANCE,
            "map": dict(sorted(historical_paths(repo_root).items())),
        },
        "counts": {
            "runs_legacy_v1": len(legacy),
            "runs_current": len(modern),
            "runs_unrecorded": len(unrecorded),
            "legacy_components": sum(r["n_components"] for r in legacy),
            #: Over `runs`, which is legacy + modern. It counted `legacy` alone
            #: while `runs` held both, so every experiment that had moved to the
            #: current layout under-reported itself -- and the more a phase
            #: migrated, the more wrong its own count became.
            "by_experiment": _by_experiment(legacy + modern, unrecorded),
            "_by_experiment_scope": (
                "counts BOTH layouts, split by what each entry is. `recorded` "
                "is a run that wrote its own manifest; `unrecorded` is a "
                "directory holding evidence without one. They are reported "
                "separately because a missing manifest is a real difference in "
                "what the entry can be trusted to say about itself, not a "
                "presentation detail."),
        },
        "kinds": {
            "_what_this_is": (
                "what each entry IS, so a reader is not left inferring it from "
                "the shape of a path. A run that executed, a run that is "
                "prepared and has not, historical evidence with no manifest, a "
                "legacy aggregate that spans several runs, and a directory that "
                "only describes itself are five different things."),
            "recorded_run": (
                "wrote a valid manifest naming the roles it produced. In "
                "`runs`."),
            "prepared_not_executed": (
                "holds governance inputs only -- a grant committed before the "
                "launch-bound sweep -- and no runtime, evidence or closeout. "
                "In `unrecorded`, flagged by its `why`. It is not a launcher "
                "that died; nothing started."),
            "historical_evidence_no_manifest": (
                "predates the manifest convention or the launcher never reached "
                "its closeout. In `unrecorded` with a digest."),
            "legacy_aggregate": (
                "one entry whose components span several directories, recorded "
                "before per-run directories existed. In `runs`, with "
                "`components` naming each part in place."),
            "description_only": (
                "a directory holding nothing but its own README. Counted "
                "nowhere: it is navigation, not a run, and reporting it as an "
                "unrecorded run would describe a launcher that never ran."),
        },
        "runs": legacy + modern,
        "unrecorded": unrecorded,
        "authorizes": "nothing",
    }


def _by_experiment(recorded: list[dict], unrecorded: list[dict]) -> dict:
    """Per experiment, counted over every entry rather than one layout."""
    out: dict[str, dict] = {}
    for r in recorded:
        e = out.setdefault(r["experiment_id"], {"recorded": 0, "unrecorded": 0})
        e["recorded"] += 1
    for r in unrecorded:
        e = out.setdefault(r["experiment_id"], {"recorded": 0, "unrecorded": 0})
        e["unrecorded"] += 1
    for e in out.values():
        e["total"] = e["recorded"] + e["unrecorded"]
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    index = build_index(REPO_ROOT)
    for r in index["runs"]:
        print(f"  v{r['layout_version']}  {r['experiment_id']:22} {r['run_id']:16} "
              f"{r['n_components']} component(s): {','.join(r['components'])}")
    for u in index["unrecorded"]:
        print(f"  --  {u['experiment_id']:22} {u['run_id']:16} "
              f"{u['n_files']} file(s), NO MANIFEST: {u['root']}")
    print(f"\n{json.dumps(index['counts'], indent=1)}")
    if args.write:
        out = REPO_ROOT / OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(index, indent=1) + "\n")
        print(f"wrote {OUT}")
    else:
        print("(dry run; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
