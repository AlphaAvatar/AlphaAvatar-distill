#!/usr/bin/env python3
"""Complete inventory of the project's documentary storage, before anything is deleted.

    PYTHONPATH=src python scripts/consolidate/build_log_inventory.py \
        --out logs/log_inventory.json

Nothing is deleted here. This answers the questions a cleanup has to answer first:

  * which files are **byte-identical duplicates**, and which copy is canonical;
  * which files are **summaries** that restate canonical per-run evidence;
  * which files are **stale living-state snapshots**;
  * which are **scratch / poll / debug** output, and whether any of it carries
    evidence that exists nowhere else;
  * which are **historical records that must stay immutable**.

The disposition of a duplicate is *derived*, not asserted: the copy that
executable code, tests or a consumed record points at is the canonical one, and
the others are the copies. Where references cannot decide it, the choice is made
in `CANONICAL_OVERRIDES` below with a written reason, so the decision is
reviewable rather than implicit.

Categories
----------
    raw_execution_artifact  written by a run: driver evidence, launcher output,
                            watchdog journals, generations, per-sample records
    derived_summary         a report or roll-up whose facts belong to a run record
    living_state            describes the repository now; replaced, not appended
    scratch_debug           poller and monitor output
    immutable_record        evidence, frozen science, consumed authorizations
    index                   catalogues and registries: they own pointers, not facts
    narrative               prose that explains rather than records

Dispositions
------------
    keep_canonical          the one copy that survives
    keep_immutable          never rewrite: evidence, frozen identity, spent grant
    keep_referenced         load-bearing for code or tests
    keep_unique_evidence    scratch-shaped, but carries a fact held nowhere else
    delete_duplicate        byte-identical to a canonical copy that survives
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where documentary storage lives. `artifacts/` is deliberately excluded — it is
#: generated, gitignored storage and is inventoried by build_checkpoint_registry.py.
ROOTS = ("logs", "docs", "README.md", "AGENTS.md")

#: Searched for references. A path named here by executable code is load-bearing.
EXECUTABLE = ("src", "scripts", "tests")
DOCUMENTARY = ("logs", "docs", "configs", "README.md", "AGENTS.md")

#: Duplicate groups whose members are referenced equally (or not at all), so the
#: reference rule cannot pick. Each entry is (sha256 prefix -> canonical path, why).
CANONICAL_OVERRIDES = {
    "db5bb0cb33ed": (
        "logs/autoinit_stage3_complete/imported_controls.json",
        "the same control import served attempt 7 and the completed attempt 8. "
        "The completed run owns it: its products are the frozen Stage-3 artifacts, "
        "so a reader following the accepted result finds the file in place."),
    "b39bc39e5908": (
        "logs/autoinit_preflight_run4/preflight_evidence.json",
        "one session wrote one evidence file. `write_preregistration.py` and the "
        "materialized preregistration both cite this path; the copy under "
        "autoinit_permanent_controls/ is that file under a second name."),
}

#: Snapshots of a living-state file. Each names the git object that holds the
#: same bytes; the claim is *verified* at build time, not asserted, so "this is
#: already in history" cannot quietly stop being true.
STALE_SNAPSHOTS = {
    "logs/archive/current_state_20260817_full.json":
        ("3261f6b67e513a9c7c4260e3a7ccc91c847dc127", "logs/current_state.json"),
}

#: Files whose shape says "scratch" but which carry evidence held nowhere else.
#: Recorded explicitly so that a future pass does not delete them by category.
SCRATCH_WITH_UNIQUE_EVIDENCE = {
    "poll.log": "the only record of the pod id, the observed pod lifetime and the "
                "accrued spend of a paid attempt; the session record carries the "
                "final figure, not the trajectory",
    "monitor.log": "the launcher-side view of a run the driver never reported on",
}

CATEGORY_RULES = (
    # (predicate on repo-relative posix path, category)
    (lambda p: p.name in ("poll.log", "monitor.log"), "scratch_debug"),
    (lambda p: p.name in ("current_state.json", "STATE.md"), "living_state"),
    (lambda p: p.name in ("CATALOG.md", "EXPERIMENT_INDEX.md", "supported_models.md",
                          "artifact_manifests.md", "checkpoint_registry.json",
                          "checkpoint_tombstones.json", "log_inventory.json",
                          "storage_measurements.json"), "index"),
    (lambda p: p.name.endswith(("_report.md",)), "derived_summary"),
    (lambda p: p.suffix in (".jsonl", ".log", ".out"), "raw_execution_artifact"),
    (lambda p: p.name in ("session.json", "manifest.json") or
               p.name.endswith(("_evidence.json", "_session.json", "evidence.json")),
     "raw_execution_artifact"),
    (lambda p: "archive" in p.parts, "immutable_record"),
    (lambda p: p.suffix == ".md" and p.parts[0] in ("docs",), "narrative"),
    (lambda p: p.name in ("README.md", "AGENTS.md", "decisions.md",
                          "BUDGET_LEDGER.md", "EXPERIMENTS.md"), "narrative"),
    (lambda p: p.suffix == ".json", "immutable_record"),
    (lambda p: True, "narrative"),
)

#: Directories whose contents are a paid attempt's evidence chain. Anything in
#: one of these is immutable unless it is a byte-identical duplicate.
ATTEMPT_DIR_MARKERS = ("attempt", "_run4", "_canary", "permanent_controls",
                       "stage3_complete", "e8b_s2_dp_sa", "e8b_step0_records")


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def iter_files() -> list[Path]:
    out: list[Path] = []
    for r in ROOTS:
        p = REPO_ROOT / r
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file()
                    and "__pycache__" not in f.parts]
    return sorted(out)


def load_corpus() -> dict[str, str]:
    """Every tracked text file, read once, so reference search is one pass."""
    corpus: dict[str, str] = {}
    for r in EXECUTABLE + DOCUMENTARY:
        base = REPO_ROOT / r
        files = [base] if base.is_file() else [
            f for f in base.rglob("*")
            if f.is_file() and "__pycache__" not in f.parts
            and f.suffix in (".py", ".md", ".json", ".sh", ".txt", ".jsonl", ".toml")]
        for f in files:
            try:
                corpus[str(f.relative_to(REPO_ROOT))] = f.read_text(errors="ignore")
            except Exception:
                continue
    return corpus


def categorize(rel: Path) -> str:
    for pred, cat in CATEGORY_RULES:
        if pred(rel):
            return cat
    return "narrative"


def in_attempt_chain(rel: Path) -> bool:
    return any(m in part for part in rel.parts for m in ATTEMPT_DIR_MARKERS)


def snapshot_equivalences() -> list[dict]:
    """For each declared living-state snapshot, check the named git object really
    holds the same bytes. A superseded snapshot whose history copy is verified
    needs no second copy on disk; one that fails this check does."""
    out = []
    for rel, (rev, path_in_rev) in STALE_SNAPSHOTS.items():
        f = REPO_ROOT / rel
        blob = subprocess.run(["git", "show", f"{rev}:{path_in_rev}"],
                              cwd=REPO_ROOT, capture_output=True, check=False)
        in_history = (hashlib.sha256(blob.stdout).hexdigest()
                      if blob.returncode == 0 else None)
        on_disk = sha256_file(f) if f.is_file() else None
        out.append({
            "path": rel,
            "present_on_disk": f.is_file(),
            "git_reference": f"git show {rev}:{path_in_rev}",
            "sha256_on_disk": on_disk,
            "sha256_in_history": in_history,
            "verified_identical": bool(on_disk and on_disk == in_history),
            "note": ("a snapshot of a living-state file. Git already holds these "
                     "exact bytes at the reference above, which is a stabler "
                     "citation than a second file that can drift"),
        })
    return out


def verbatim_narrative_overlap(paths: list[Path]) -> list[dict]:
    """Long sentences that appear word for word in more than one prose file."""
    def sentences(p: Path) -> list[str]:
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            return []
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        out = []
        for line in text.splitlines():
            for s in re.split(r"(?<=[.;:])\s+", line.strip()):
                s = re.sub(r"\s+", " ", s).strip()
                if len(s) >= 60:
                    out.append(s)
        return out

    index: dict[str, set[str]] = collections.defaultdict(set)
    for p in paths:
        if p.suffix != ".md":
            continue
        rel = str(p.relative_to(REPO_ROOT))
        for s in sentences(p):
            index[s].add(rel)
    shared = [{"sentence": s, "files": sorted(v)} for s, v in index.items() if len(v) > 1]
    return sorted(shared, key=lambda e: (-len(e["files"]), e["sentence"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/log_inventory.json")
    args = ap.parse_args()

    files = iter_files()
    corpus = load_corpus()
    out_rel = args.out

    records = []
    by_hash: dict[str, list[str]] = collections.defaultdict(list)
    for f in files:
        rel = f.relative_to(REPO_ROOT)
        rel_s = str(rel)
        if rel_s == out_rel:                       # never inventory our own output
            continue
        digest = sha256_file(f)
        by_hash[digest].append(rel_s)
        refs = sorted(name for name, text in corpus.items()
                      if name != rel_s and rel_s in text)
        records.append({
            "path": rel_s,
            "size_bytes": f.stat().st_size,
            "sha256": digest,
            "category": categorize(rel),
            "in_paid_attempt_chain": in_attempt_chain(rel),
            "referenced_by": refs,
            "referenced_by_executable": [r for r in refs
                                         if r.split("/")[0] in EXECUTABLE],
        })

    # ---- duplicate groups, and which copy is canonical --------------------
    groups = []
    canonical_of: dict[str, tuple[str, str]] = {}
    for digest, members in by_hash.items():
        if len(members) < 2:
            continue
        by_path = {r["path"]: r for r in records}
        exec_refd = [m for m in members if by_path[m]["referenced_by_executable"]]
        any_refd = [m for m in members if by_path[m]["referenced_by"]]
        override = CANONICAL_OVERRIDES.get(digest[:12])
        if override:
            canonical, why = override
        elif len(exec_refd) == 1:
            canonical = exec_refd[0]
            why = ("the only copy executable code or a test names; the others are "
                   "copies of it")
        elif len(any_refd) == 1:
            canonical = any_refd[0]
            why = "the only copy any other file points at"
        else:
            canonical, why = sorted(members)[0], "UNDECIDED — review before deleting"
        groups.append({
            "sha256": digest,
            "size_bytes": by_path[members[0]]["size_bytes"],
            "members": sorted(members),
            "canonical": canonical,
            "canonical_reason": why,
            "reclaimable_bytes": by_path[members[0]]["size_bytes"] * (len(members) - 1),
            "decided_by": ("override" if override else
                           "executable reference" if len(exec_refd) == 1 else
                           "documentary reference" if len(any_refd) == 1 else "none"),
        })
        for m in members:
            canonical_of[m] = (canonical, why)

    # ---- disposition -----------------------------------------------------
    for r in records:
        path, cat = r["path"], r["category"]
        canonical, why = canonical_of.get(path, (None, None))
        if canonical and canonical != path:
            r["duplicate_of"] = canonical
            r["disposition"] = "delete_duplicate"
            r["disposition_reason"] = (
                f"byte-identical to {canonical}, which survives: {why}")
        elif canonical == path:
            r["duplicate_of"] = None
            r["disposition"] = "keep_canonical"
            r["disposition_reason"] = why
        elif r["referenced_by_executable"]:
            r["duplicate_of"] = None
            r["disposition"] = "keep_referenced"
            r["disposition_reason"] = (
                "named by " + ", ".join(r["referenced_by_executable"][:3]))
        elif cat == "scratch_debug":
            r["duplicate_of"] = None
            note = SCRATCH_WITH_UNIQUE_EVIDENCE.get(Path(path).name)
            r["disposition"] = "keep_unique_evidence" if note else "delete_scratch"
            r["disposition_reason"] = note or "scratch output with no unique record"
        elif cat in ("immutable_record", "raw_execution_artifact") or \
                r["in_paid_attempt_chain"]:
            r["duplicate_of"] = None
            r["disposition"] = "keep_immutable"
            r["disposition_reason"] = (
                "evidence of something that happened; AGENTS.md P11 and the "
                "CATALOG's HISTORICAL class forbid rewriting it")
        else:
            r["duplicate_of"] = None
            r["disposition"] = "keep_canonical"
            r["disposition_reason"] = "sole copy, no superseding owner"

    # ---- carry the record of anything already removed ---------------------
    # Regenerating this file must not erase the evidence that a copy existed.
    # A path that was inventoried before and is gone now becomes a `removed`
    # entry keeping its hash, so "attempt 6's engine probe was byte-identical
    # to 49593edb…" survives the deletion of the bytes.
    out_path = REPO_ROOT / out_rel
    removed = []
    if out_path.is_file():
        previous = json.loads(out_path.read_text())
        removed = list(previous.get("removed", []))
        known = {e["path"] for e in removed}
        live = {r["path"] for r in records}
        for old in previous.get("files", []):
            if old["path"] in live or old["path"] in known:
                continue
            removed.append({
                "path": old["path"],
                "sha256": old["sha256"],
                "size_bytes": old["size_bytes"],
                "category": old["category"],
                "was": old.get("disposition"),
                "canonical_survivor": old.get("duplicate_of"),
                "reason": old.get("disposition_reason"),
                "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        removed.sort(key=lambda e: e["path"])

    totals = collections.Counter(r["disposition"] for r in records)
    cats = collections.Counter(r["category"] for r in records)
    report = {
        "schema": "aadistill.log_inventory/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "roots": list(ROOTS),
        "rule": ("nothing is deleted before it appears here with a disposition and "
                 "a reason. A duplicate's canonical copy is derived from who "
                 "references it, not chosen by hand, except where "
                 "CANONICAL_OVERRIDES states otherwise and says why."),
        "totals": {
            "files": len(records),
            "bytes": sum(r["size_bytes"] for r in records),
            "by_disposition": dict(sorted(totals.items())),
            "by_category": dict(sorted(cats.items())),
            "duplicate_groups": len(groups),
            "reclaimable_bytes": sum(g["reclaimable_bytes"] for g in groups),
            "removed_previously": len(removed),
        },
        "removed": removed,
        "duplicate_groups": sorted(groups, key=lambda g: -g["reclaimable_bytes"]),
        "living_state_snapshots": snapshot_equivalences(),
        "verbatim_narrative_overlap": verbatim_narrative_overlap(files),
        "files": records,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(records)} files, {report['totals']['bytes'] / 2**20:.2f} MiB")
    for k, v in sorted(totals.items()):
        print(f"  {k:24s} {v}")
    print(f"  duplicate groups: {len(groups)}, "
          f"reclaimable {report['totals']['reclaimable_bytes'] / 1024:.1f} KiB")
    print(f"-> {out_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
