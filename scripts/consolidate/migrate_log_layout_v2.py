#!/usr/bin/env python3
"""Stage → Experiment → Run: the canonical `logs/` hierarchy.

    PYTHONPATH=src python scripts/consolidate/migrate_log_layout_v2.py          # plan
    PYTHONPATH=src python scripts/consolidate/migrate_log_layout_v2.py --write

log-layout-v1 removed the loose root and the three parallel run trees, but left
`experiments/`, `runs/` and `validations/` as three top-level axes — so one
experiment's plan, its analyses, its validations and its runs still lived in
three different places. This makes **stage** the first dimension and keeps an
experiment's whole scientific context in one directory:

    logs/stages/stage-<id>/<experiment>/{plans,analyses,results,history,
                                         validations,runs/<run_id>/}

**Stage is read, never guessed from a name.** `phase_a`, `phase_c1` and
`continuation_b` are experiment identifiers; they say nothing about which
pipeline stage the work belongs to. The only reliable declaration in this
repository is `configs/experiments/<id>/authorization.json:stage_id`, and only
`phase_c1` carries one. Everything else goes to `cross-stage/<experiment>/`,
which is a positive statement — this experiment's stage is not established by
any frozen record — and not a second dumping ground.

**A validation belongs to what it serves.** The CUDA stage-F work exists to
validate C1's execution path, so it lives inside that experiment. Infrastructure
validation that serves no single experiment goes to `shared/validations/`.
`validation` is not a top-level category.

**Evidence is preserved, never rewritten.** Bytes, recorded digests,
authorization self-hashes, scientific results, budget facts, run identity and
git ancestry are untouched. An old path inside a frozen payload keeps meaning
what it meant — where the object was then — and this migration's manifest maps
it forward.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "logs/migrations/log-layout-v2/manifest.json"

#: A validation and the experiment it serves. Stated, because "which experiment
#: does this validate" is a judgement about purpose that a path cannot carry.
VALIDATION_OWNER: dict[str, tuple[str, str]] = {
    #: Exists to prove C1's stage-F device boundary on real CUDA.
    "cuda-stage-f": ("phase_c1", "validates the C1 execution path's device "
                                 "placement on real CUDA"),
    #: The C1 renderer-parity gate's evidence.
    "renderer-parity": ("phase_c1", "the C1 renderer-parity gate's evidence"),
}

#: Validation that serves no single experiment: shared infrastructure capability.
SHARED_VALIDATIONS = ("device-canary", "micro-preflight", "cpu-smoke",
                      "depth-backend", "dryrun", "relay-mirror")

#: Archive groups, by what the material is about.
ARCHIVE_REPOSITORY = ("handoffs", "indexes", "STATE_superseded", "CATALOG_detail",
                      "PROPOSAL", "README")


def git(*args: str, root: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_stage(experiment_id: str, root: Path) -> str | None:
    """The experiment's pipeline stage, from its configuration. Or nothing.

    Deliberately the ONLY source consulted. A run manifest records the stage it
    was opened under, which is the same declaration one step removed; a
    preregistration's `session_plan.stages` are DRIVER stages; and an operator
    named `composite.stage1_sandwich_v0` names an operator. None of those
    establish the pipeline stage of an experiment.
    """
    p = root / f"configs/experiments/{experiment_id}/authorization.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("stage_id")
    except (json.JSONDecodeError, OSError):
        return None


#: Not experiments at all, whatever they sit beside today.
NOT_AN_EXPERIMENT: dict[str, tuple[str, str]] = {
    "shared": ("logs/shared",
               "cross-cutting analyses that belong to no experiment: transport, "
               "relay capacity, tool rendering, probe reuse, pricing bounds"),
    "architecture": ("logs/maintenance/inventories",
                     "mechanical inventories of the source tree, produced by "
                     "repository maintenance rather than by an experiment"),
}


def experiment_home(experiment_id: str, root: Path) -> tuple[str, str]:
    """Where this experiment's directory belongs, and why."""
    if experiment_id in NOT_AN_EXPERIMENT:
        return NOT_AN_EXPERIMENT[experiment_id]
    stage = declared_stage(experiment_id, root)
    if stage:
        return (f"logs/stages/stage-{stage}/{experiment_id}",
                f"configs/experiments/{experiment_id}/authorization.json "
                f"declares stage_id {stage!r}")
    return (f"logs/cross-stage/{experiment_id}",
            "no frozen record establishes a pipeline stage for this experiment; "
            "its identifier names the experiment, not a stage")


def plan(root: Path = REPO_ROOT) -> dict:
    moves: list[dict] = []
    reasons: dict[str, str] = {}

    def add(src: Path, dest: str, binding: str, why: str) -> None:
        rel = src.relative_to(root).as_posix()
        moves.append({"old_path": rel, "new_path": dest, "binding": binding,
                      "kind": "directory" if src.is_dir() else "file",
                      "migration_reason": why})

    logs = root / "logs"

    # --- experiment material: plans / analyses / results / history ----------
    exp_root = logs / "experiments"
    if exp_root.is_dir():
        for d in sorted(p for p in exp_root.iterdir() if p.is_dir()):
            home, why = experiment_home(d.name, root)
            reasons[d.name] = why
            for child in sorted(d.iterdir()):
                if child.name == "README.md":
                    continue
                add(child, f"{home}/{child.name}", "experiment_material", why)

    # --- runs: under the experiment that owns them --------------------------
    runs_root = logs / "runs"
    if runs_root.is_dir():
        for group in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            for exp_dir in sorted(q for q in group.iterdir() if q.is_dir()):
                home, why = experiment_home(exp_dir.name, root)
                reasons[exp_dir.name] = why
                for run in sorted(r for r in exp_dir.iterdir() if r.is_dir()):
                    add(run, f"{home}/runs/{run.name}", "run", why)

    # --- validations: to what they serve ------------------------------------
    val_root = logs / "validations"
    if val_root.is_dir():
        for d in sorted(p for p in val_root.iterdir() if p.is_dir()):
            if d.name in VALIDATION_OWNER:
                exp, why = VALIDATION_OWNER[d.name]
                home, _ = experiment_home(exp, root)
                add(d, f"{home}/validations/{d.name}", "validation", why)
            else:
                add(d, f"logs/shared/validations/{d.name}", "validation",
                    "serves no single experiment: shared infrastructure "
                    "capability, validated once for every session that uses it")
        for f in sorted(p for p in val_root.iterdir() if p.is_file()
                        and p.name != "README.md"):
            add(f, f"logs/shared/validations/{f.name}", "validation",
                "shared validation material")

    # --- archive: mirror the same split -------------------------------------
    arc = logs / "archive"
    if arc.is_dir():
        for p in sorted(arc.iterdir()):
            if p.name == "README.md":
                continue
            if any(p.name.startswith(k) for k in ARCHIVE_REPOSITORY):
                add(p, f"logs/archive/repository/{p.name}", "archive",
                    "repository-wide history: a superseded state, catalog, "
                    "handoff or index snapshot")
            else:
                add(p, f"logs/archive/cross-stage/{p.name}", "archive",
                    "belongs to an experiment whose stage is not established")

    by_old = collections.Counter(m["old_path"] for m in moves)
    by_new = collections.Counter(m["new_path"] for m in moves)
    return {"moves": moves, "reasons": reasons, "conflicts": {
        "old_twice": [k for k, n in by_old.items() if n > 1],
        "new_twice": [k for k, n in by_new.items() if n > 1]}}


def identity(m: dict, root: Path, head: str) -> dict:
    src = root / m["old_path"]
    e = {**m, "source_commit": head, "canonical_owner": m["new_path"]}
    if m["kind"] == "file":
        e["sha256"] = sha256_of(src)
    else:
        files = sorted(q for q in src.rglob("*") if q.is_file())
        e["n_files"] = len(files)
        e["tree_sha256"] = hashlib.sha256("".join(
            f"{q.relative_to(src).as_posix()}:{sha256_of(q)}\n"
            for q in files).encode()).hexdigest()
    e["historical_binding"] = (
        "old_path is where this object was at " + head[:12] + "; payloads that "
        "name it are not rewritten, because that statement remains true")
    return e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    p = plan()
    if p["conflicts"]["old_twice"] or p["conflicts"]["new_twice"]:
        print("REFUSING: the plan is not one-to-one")
        print(json.dumps(p["conflicts"], indent=1))
        return 1

    by = collections.Counter(m["binding"] for m in p["moves"])
    print(f"objects to relocate : {len(p['moves'])}")
    for k, n in by.most_common():
        print(f"   {k:22s} {n}")
    print("\nexperiment homes:")
    for exp, why in sorted(p["reasons"].items()):
        home, _ = experiment_home(exp, REPO_ROOT)
        print(f"   {exp:24s} -> {home}")
    if not a.write:
        print("\n(plan only; pass --write)")
        return 0

    head = git("rev-parse", "HEAD").strip()
    entries = [identity(m, REPO_ROOT, head) for m in p["moves"]]
    for m in p["moves"]:
        (REPO_ROOT / m["new_path"]).parent.mkdir(parents=True, exist_ok=True)
        git("mv", m["old_path"], m["new_path"])

    drift = []
    for e in entries:
        dst = REPO_ROOT / e["new_path"]
        if e["kind"] == "file":
            if sha256_of(dst) != e["sha256"]:
                drift.append(e["new_path"])
        else:
            files = sorted(q for q in dst.rglob("*") if q.is_file())
            got = hashlib.sha256("".join(
                f"{q.relative_to(dst).as_posix()}:{sha256_of(q)}\n"
                for q in files).encode()).hexdigest()
            if got != e["tree_sha256"]:
                drift.append(e["new_path"])
    if drift:
        print(f"\nBYTES CHANGED during the move: {drift}")
        return 1

    out = REPO_ROOT / MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "aadistill.log_layout_migration/v2",
        "migration_id": "log-layout-v2",
        "_what_this_is": (
            "the relocation that made STAGE the first dimension of `logs/`. An "
            "experiment's plans, analyses, results, history, validations and "
            "runs are now one directory instead of three top-level axes."),
        "source_commit": head,
        "_stage_attribution": (
            "read from configs/experiments/<id>/authorization.json:stage_id and "
            "from nothing else. An experiment identifier is not a stage, a "
            "preregistration's session_plan stages are DRIVER stages, and an "
            "operator id names an operator. An experiment with no such "
            "declaration is in `cross-stage/`, which states that its stage is "
            "not established rather than guessing one."),
        "_old_paths": (
            "old_path is a fact about where an object was at the source commit. "
            "Payloads that record one -- consumed authorizations, closed "
            "manifests, archived prose -- are NOT rewritten; this manifest is "
            "how a reader follows one forward."),
        "experiment_homes": {
            exp: {"home": experiment_home(exp, REPO_ROOT)[0], "because": why}
            for exp, why in sorted(p["reasons"].items())},
        "counts": {"objects": len(entries), "by_binding": dict(by)},
        "verification": {
            "bytes_unchanged": True,
            "one_old_path_one_new_path": True,
            "no_new_path_claimed_twice": True,
            "method": ("per-file sha256 and per-directory tree hash captured "
                       "BEFORE the move and re-checked after it"),
        },
        "entries": entries,
        "authorizes": "nothing",
    }, indent=1) + "\n")
    print(f"\nmoved {len(entries)} object(s); recorded {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
