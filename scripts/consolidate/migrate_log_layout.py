#!/usr/bin/env python3
"""One canonical `logs/` layout, by evidence-preserving relocation.

    PYTHONPATH=src python scripts/consolidate/migrate_log_layout.py            # plan
    PYTHONPATH=src python scripts/consolidate/migrate_log_layout.py --write    # do it

`logs/` had three parallel run layouts and 179 loose files at its root. The
reason each file stayed was that something referenced it, which is a statement
about effort, not about evidence — and it was keeping the repository permanently
shaped by the order in which its experiments happened to run.

**What must not change, and does not.** Historical payloads, recorded digests,
authorization self-hashes, scientific results, budget amounts, frozen decisions,
run identity and git ancestry. Nothing is rewritten, nothing is force-pushed,
nothing is squashed. A relocated object is byte-identical to what it was.

**What does change.** Where the file sits at the CURRENT head. Git history keeps
the original path as the permanent record of where it was; an old path inside a
frozen payload keeps meaning what it meant — the historical location — and this
migration's manifest maps it forward. Rewriting those strings would be editing
evidence, so they are left exactly as written.

**Stage attribution is DECLARED, never inferred.** Only `phase_c1` declares a
pipeline stage (`configs/experiments/phase_c1/authorization.json:stage_id`).
Every other experiment's frozen records describe *driver* stages -- `stage 0`
attestation, `stage 1` build -- which are a different dimension, and an operator
called `composite.stage1_sandwich_v0` names an operator, not a pipeline stage.
Those runs go to `runs/unscoped/`, which exists precisely for a run whose stage
no frozen record determines. Filing them under a guessed stage would be
inventing provenance.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "logs/migrations/log-layout-v1/manifest.json"

#: Directories that already are canonical top level.
CANONICAL_TOP = {"README.md", "state", "budget", "experiments", "runs",
                 "validations", "maintenance", "migrations", "archive"}

#: `logs/<dir>` -> (experiment_id, run_id). The run directories of closed
#: experiments, named by the convention each used at the time.
RUN_DIR = re.compile(
    r"^autoinit_(?P<exp>c1|phase_a|phase_b|continuation_b|recovery_continuation"
    r"|measurement|device_canary)_(?P<run>attempt\d+r?)$")

#: Experiment id as written at the time -> the canonical experiment id.
EXPERIMENT_ALIAS = {"c1": "phase_c1"}

#: The ONLY experiment whose pipeline stage is declared. Read, not assumed.
def declared_stage(experiment_id: str, root: Path) -> str | None:
    p = root / f"configs/experiments/{experiment_id}/authorization.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("stage_id")
    except (json.JSONDecodeError, OSError):
        return None


#: Engineering validation, not a formal pipeline run: it answers a hardware or
#: integration question and produces no scientific measurement.
VALIDATION_DIRS = {
    "autoinit_device_canary_attempt1": "device-canary",
    "autoinit_device_canary_attempt2": "device-canary",
    "autoinit_preflight_run4": "micro-preflight",
}

#: Root directories that are neither a run nor a validation.
OTHER_DIRS = {
    "autoinit_continuation_attempts": "experiments/recovery_continuation/history",
    "autoinit_phase_a_attempts": "experiments/phase_a/history",
    "autoinit_permanent_controls": "experiments/phase_a/results",
    "autoinit_stage3_complete": "experiments/phase_a/results",
    "e7_canary": "experiments/early/history",
    "e7_canary_rerun": "experiments/early/history",
    "e8b_s2_dp_sa": "experiments/early/history",
    "e8b_step0_records": "experiments/early/history",
    "superseded": "archive/superseded",
}

#: Loose root FILES, by semantic owner. First match wins, so order matters.
#: Every rule names what the object IS, never its extension.
FILE_RULES: tuple[tuple[str, str], ...] = (
    # --- current state -----------------------------------------------------
    (r"^current_state\.json$", "state/current.json"),
    (r"^STATE\.md$", "state/current.md"),
    (r"^CATALOG\.md$", "state/ownership.md"),
    (r"^PHASE_INDEX\.md$", "state/phase_index.md"),
    (r"^supported_models\.md$", "state/supported_models.md"),
    (r"^artifact_manifests\.md$", "state/artifact_manifests.md"),
    # --- budget and approvals ----------------------------------------------
    (r"^BUDGET_LEDGER\.md$", "budget/ledger.md"),
    (r"^decisions\.md$", "budget/decisions.md"),
    (r".*_grant\d*\.json$", "budget/approvals"),
    (r"^autoinit_.*_authorization\.json$", "budget/approvals"),
    (r"^autoinit_c1_grant\.json$", "budget/approvals"),
    # --- engineering validation --------------------------------------------
    (r"^(c1_)?renderer_parity.*\.json$", "validations/renderer-parity"),
    (r"^autoinit_repeatability_cpu_smoke\.json$", "validations/cpu-smoke"),
    (r"^autoinit_depth_backend_equivalence\.json$", "validations/depth-backend"),
    (r"^autoinit_dryrun.*\.json$", "validations/dryrun"),
    (r"^relay_mirror_verification\.json$", "validations/relay-mirror"),
    # --- maintenance --------------------------------------------------------
    (r"^storage_.*\.json$", "maintenance/storage"),
    (r"^scratch_.*\.json$", "maintenance/storage"),
    (r"^checkpoint_.*\.json$", "maintenance/inventories"),
    (r"^log_inventory\.json$", "maintenance/inventories"),
    (r"^derived_cache_.*\.json$", "maintenance/cleanup"),
    (r"^c1_storage_recovery\.json$", "maintenance/storage"),
    (r"^architecture_.*\.json$", "maintenance/inventories"),
    # --- superseded documents ----------------------------------------------
    (r"^HANDOFF_next_session\.md$", "archive/handoffs"),
    (r"^EXPERIMENTS?(_INDEX)?\.md$", "archive/indexes"),
    # --- per-experiment material -------------------------------------------
    (r"^(autoinit_)?(phase_c1|phase_c0|c1)_.*preregistration.*\.json$",
     "experiments/phase_c1/plans"),
    (r"^(autoinit_)?phase_a_.*preregistration.*\.json$", "experiments/phase_a/plans"),
    (r"^(autoinit_)?phase_b_.*preregistration.*\.json$", "experiments/phase_b/plans"),
    (r"^autoinit_continuation_b_preregistration\.json$",
     "experiments/continuation_b/plans"),
    (r"^autoinit_recovery_continuation_preregistration\.json$",
     "experiments/recovery_continuation/plans"),
    (r"^(autoinit_)?c1_.*\.json$", "experiments/phase_c1/analyses"),
    (r"^phase_c[01]_.*\.json$", "experiments/phase_c1/plans"),
    (r"^phase_c_.*\.md$", "experiments/phase_c1/plans"),
    (r"^autoinit_c1_.*$", "experiments/phase_c1/analyses"),
    (r"^(autoinit_)?phase_a_.*$", "experiments/phase_a/analyses"),
    (r"^(autoinit_)?phase_b_.*$", "experiments/phase_b/analyses"),
    (r"^autoinit_continuation_b_.*$", "experiments/continuation_b/analyses"),
    (r"^autoinit_continuation_.*$", "experiments/recovery_continuation/analyses"),
    (r"^autoinit_recovery_.*$", "experiments/recovery_continuation/analyses"),
    (r"^recovery_.*$", "experiments/recovery_continuation/analyses"),
    (r"^autoinit_measurement_.*$", "experiments/measurement/analyses"),
    (r"^autoinit_device_canary.*$", "validations/device-canary"),
    (r"^autoinit_micro_preflight.*$", "validations/micro-preflight"),
    (r"^autoinit_preflight.*$", "validations/micro-preflight"),
    (r"^e[0-9].*$", "experiments/early/analyses"),
    (r"^autoinit_stage1_.*$", "experiments/phase_a/analyses"),
    #: Cross-cutting analyses that belong to no single experiment -- transport,
    #: relay capacity, tool rendering, probe reuse, pricing bounds. They are NOT
    #: "early": filing them under the E-series would make that directory mean
    #: "whatever was left", which is the bucket this migration removes.
    (r"^autoinit_.*$", "experiments/shared/analyses"),
)


def git(*args: str, root: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_id(rel: str, root: Path) -> str | None:
    """Git's own identity for the tracked content, when it has one."""
    out = subprocess.run(["git", "ls-files", "-s", "--", rel], cwd=root,
                         capture_output=True, text=True)
    parts = out.stdout.split()
    return parts[1] if len(parts) > 2 else None


def classify_dir(name: str, root: Path) -> tuple[str, str] | None:
    """`logs/<name>/` -> (new relative path under logs/, binding type)."""
    if name in CANONICAL_TOP:
        return None
    if name in VALIDATION_DIRS:
        return f"validations/{VALIDATION_DIRS[name]}/runs/{name}", "validation_run"
    if name in OTHER_DIRS:
        return f"{OTHER_DIRS[name]}/{name}", "experiment_material"
    m = RUN_DIR.match(name)
    if m:
        exp = EXPERIMENT_ALIAS.get(m.group("exp"), m.group("exp"))
        stage = declared_stage(exp, root)
        run = m.group("run")
        if stage:
            return f"runs/stage-{stage}/{exp}/{run}", "formal_run_declared_stage"
        return f"runs/unscoped/{exp}/{run}", "formal_run_no_declared_stage"
    return None


def classify_file(name: str) -> str | None:
    for pattern, dest in FILE_RULES:
        if re.match(pattern, name):
            return dest if dest.endswith((".json", ".md")) else f"{dest}/{name}"
    return None


def plan(root: Path = REPO_ROOT) -> dict:
    moves, unclassified = [], []
    logs = root / "logs"

    for p in sorted(logs.iterdir()):
        rel = p.relative_to(root).as_posix()
        if p.is_dir():
            got = classify_dir(p.name, root)
            if got:
                moves.append({"old_path": rel, "new_path": f"logs/{got[0]}",
                              "binding": got[1], "kind": "directory"})
            elif p.name not in CANONICAL_TOP:
                unclassified.append(rel)
        elif p.name != "README.md":
            dest = classify_file(p.name)
            if dest:
                moves.append({"old_path": rel, "new_path": f"logs/{dest}",
                              "binding": "loose_root_file", "kind": "file"})
            else:
                unclassified.append(rel)

    #: The second parallel run tree: `logs/runs/<experiment>/<run>`.
    legacy_runs = logs / "runs"
    for exp_dir in sorted(p for p in legacy_runs.iterdir()
                          if p.is_dir() and not p.name.startswith("stage-")
                          and p.name != "unscoped"):
        for run_dir in sorted(q for q in exp_dir.iterdir() if q.is_dir()):
            rel = run_dir.relative_to(root).as_posix()
            if exp_dir.name == "cuda_stage_f":
                moves.append({
                    "old_path": rel,
                    "new_path": f"logs/validations/cuda-stage-f/runs/{run_dir.name}",
                    "binding": "validation_run", "kind": "directory"})
                continue
            stage = declared_stage(exp_dir.name, root)
            dest = (f"runs/stage-{stage}/{exp_dir.name}/{run_dir.name}" if stage
                    else f"runs/unscoped/{exp_dir.name}/{run_dir.name}")
            moves.append({
                "old_path": rel, "new_path": f"logs/{dest}",
                "binding": ("formal_run_declared_stage" if stage
                            else "formal_run_no_declared_stage"),
                "kind": "directory"})

    #: One old path -> one new path, and one new path claimed once.
    by_old = collections.Counter(m["old_path"] for m in moves)
    by_new = collections.Counter(m["new_path"] for m in moves)
    conflicts = {
        "old_claimed_twice": [k for k, n in by_old.items() if n > 1],
        "new_claimed_twice": [k for k, n in by_new.items() if n > 1],
    }
    return {"moves": moves, "unclassified": unclassified, "conflicts": conflicts}


def record_entry(m: dict, root: Path, head: str) -> dict:
    """Everything needed to follow this object forward, and to check it."""
    src = root / m["old_path"]
    e = {**m, "source_commit": head}
    if m["kind"] == "file":
        e["sha256_before"] = sha256_of(src)
        e["git_blob"] = blob_id(m["old_path"], root)
    else:
        files = sorted(q for q in src.rglob("*") if q.is_file())
        e["n_files"] = len(files)
        e["tree_sha256"] = hashlib.sha256("".join(
            f"{q.relative_to(src).as_posix()}:{sha256_of(q)}\n"
            for q in files).encode()).hexdigest()
    e["evidence_preserving_because"] = (
        "the object's bytes are unchanged by the move; `git mv` keeps one owner "
        "and full history; the historical payloads that name the old path are "
        "not edited, because the old path is a true statement about where the "
        "object was at "
        f"{head[:12]} and this manifest is what carries it forward")
    return e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    p = plan()

    if p["conflicts"]["old_claimed_twice"] or p["conflicts"]["new_claimed_twice"]:
        print("REFUSING: the plan is not one-to-one")
        print(json.dumps(p["conflicts"], indent=1))
        return 1

    by_binding = collections.Counter(m["binding"] for m in p["moves"])
    print(f"objects to relocate : {len(p['moves'])}")
    for k, n in by_binding.most_common():
        print(f"   {k:32s} {n}")
    print(f"unclassified        : {len(p['unclassified'])}")
    for u in p["unclassified"][:20]:
        print(f"   {u}")
    if not a.write:
        print("\n(plan only; pass --write)")
        return 0
    if p["unclassified"]:
        print("\nREFUSING: every object needs an owner before anything moves")
        return 1

    head = git("rev-parse", "HEAD").strip()
    entries = [record_entry(m, REPO_ROOT, head) for m in p["moves"]]
    for m in p["moves"]:
        dest = REPO_ROOT / m["new_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        git("mv", m["old_path"], m["new_path"])

    #: Verified AFTER the move, against what was recorded before it.
    drift = []
    for e in entries:
        dst = REPO_ROOT / e["new_path"]
        if e["kind"] == "file":
            if sha256_of(dst) != e["sha256_before"]:
                drift.append(e["new_path"])
        else:
            files = sorted(q for q in dst.rglob("*") if q.is_file())
            got = hashlib.sha256("".join(
                f"{q.relative_to(dst).as_posix()}:{sha256_of(q)}\n"
                for q in files).encode()).hexdigest()
            if got != e["tree_sha256"] or len(files) != e["n_files"]:
                drift.append(e["new_path"])
    if drift:
        print(f"\nBYTES CHANGED during the move: {drift}")
        return 1

    out = REPO_ROOT / MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "aadistill.log_layout_migration/v1",
        "migration_id": "log-layout-v1",
        "_what_this_is": (
            "the one relocation that gave `logs/` a single canonical layout at "
            "the current head. Every entry maps an old path to exactly one new "
            "path, with the identity the object had before the move so the "
            "claim `nothing changed but the location` is checkable rather than "
            "asserted."),
        "source_commit": head,
        "_old_paths_in_history": (
            "an old path written inside a frozen payload -- a consumed "
            "authorization, a closed run's manifest, an archived document -- is "
            "NOT stale. It states where the object was when that payload was "
            "written, which remains true, and git history holds the tree that "
            "proves it. Those strings are not rewritten; this manifest is how a "
            "reader follows one forward."),
        "_stage_attribution": (
            "declared, never inferred. Only phase_c1 declares a pipeline stage "
            "(configs/experiments/phase_c1/authorization.json:stage_id). Other "
            "experiments' frozen records describe DRIVER stages, a different "
            "dimension, so their runs are `unscoped` -- which is what that "
            "directory is for."),
        "counts": {"objects": len(entries),
                   "by_binding": dict(by_binding)},
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
