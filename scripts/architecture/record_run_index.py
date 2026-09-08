#!/usr/bin/env python3
"""One index entry per logical RUN, not per artifact root.

    PYTHONPATH=src python scripts/architecture/record_run_index.py --write

The first version of this index counted 77 "runs". It was counting artifact
roots: `logs/autoinit_c1_attempt9` and `logs/autoinit_c1_attempt9_grant.json`
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

OUT = "logs/runs/index.json"
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


def discover_legacy(repo_root: Path) -> dict[tuple[str, str], dict]:
    runs: dict[tuple[str, str], dict] = {}
    for entry in sorted((repo_root / "logs").iterdir()):
        for pattern, experiment, tpl, role in CONVENTIONS:
            m = pattern.match(entry.name)
            if not m:
                continue
            run_id = tpl.format(**m.groupdict())
            key = (experiment, run_id)
            rec = runs.setdefault(key, {
                "experiment_id": experiment, "run_id": run_id,
                "layout_version": 1, "components": {}, "component_digests": {},
            })
            rel = f"logs/{entry.name}"
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

    runs_root = repo_root / "logs/runs"
    out: list[dict] = []
    if not runs_root.is_dir():
        return out
    for manifest in sorted(runs_root.glob("*/*/manifest.json")):
        doc = json.loads(manifest.read_text())
        if doc.get("schema") != MANIFEST_SCHEMA:
            continue
        out.append({
            "experiment_id": doc["experiment_id"], "run_id": doc["run_id"],
            "layout_version": doc["layout_version"],
            "components": {"root": f"logs/runs/{doc['root']}",
                           "manifest": f"logs/runs/{doc['root']}/manifest.json"},
            "manifest_sha256": doc["self_sha256"],
            "n_components": len(doc.get("roles") or {}),
        })
    return out


def build_index(repo_root: Path) -> dict:
    legacy = sorted(discover_legacy(repo_root).values(),
                    key=lambda r: (r["experiment_id"], r["run_id"]))
    modern = discover_v3(repo_root)
    return {
        "schema": SCHEMA,
        "_contract": (
            "Every run this repository has recorded, ONE ENTRY PER LOGICAL RUN. "
            "A legacy run references its surviving components in place: nothing "
            "was moved, renamed or copied, and each component carries a digest "
            "so that is checkable. Registering a run confers nothing."),
        "granularity": ("one entry per (experiment_id, run_id). The v1 index "
                        "counted artifact roots, so an attempt with an evidence "
                        "directory and a grant file appeared twice."),
        "counts": {
            "runs_legacy_v1": len(legacy),
            "runs_current": len(modern),
            "legacy_components": sum(r["n_components"] for r in legacy),
            "by_experiment": {e: sum(1 for r in legacy if r["experiment_id"] == e)
                              for e in sorted({r["experiment_id"] for r in legacy})},
        },
        "runs": legacy + modern,
        "authorizes": "nothing",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    index = build_index(REPO_ROOT)
    for r in index["runs"]:
        print(f"  v{r['layout_version']}  {r['experiment_id']:22} {r['run_id']:16} "
              f"{r['n_components']} component(s): {','.join(r['components'])}")
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
