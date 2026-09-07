#!/usr/bin/env python3
"""Register existing runs in `logs/runs/index.json`. Moves and copies nothing.

    PYTHONPATH=src python scripts/autoinit/record_run_index.py --write

Attempts 1-9 stay exactly where they are. This records, for each, a
`legacy-v1` reference and a directory digest, so that:

* "what runs exist" has one answer instead of a naming convention;
* a test can prove the historical evidence is byte-for-byte unchanged;
* run-layout-v2 can be introduced for FUTURE runs without a repository-wide
  move, which is the migration that loses provenance.

Discovery is by pattern over `logs/`, not a hand-written list, so a run that
exists cannot be quietly left out of the index — and the no-new-flat-log test
then has something real to compare against.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.run_layout import (  # noqa: E402
    INDEX_PATH, INDEX_SCHEMA, RUN_LAYOUT_VERSION, RunLayout, directory_digest,
    resolve_run,
)

#: How the three historical naming conventions map onto (experiment, attempt).
#: A list of patterns rather than a list of paths: a convention is a rule, and
#: writing the rule down is what lets the enforcement test find an unregistered
#: run instead of trusting that somebody remembered to add it.
CONVENTIONS = [
    (re.compile(r"^autoinit_c1_attempt(?P<n>\d+r?)$"), "phase_c1", "attempt{n}"),
    (re.compile(r"^autoinit_continuation_b_attempt(?P<n>\d+)$"),
     "continuation_b", "attempt{n}"),
    (re.compile(r"^autoinit_recovery_continuation_attempt(?P<n>\d+)$"),
     "recovery_continuation", "attempt{n}"),
    (re.compile(r"^autoinit_phase_a_attempt(?P<n>\d+)$"), "phase_a", "attempt{n}"),
    (re.compile(r"^autoinit_phase_b_attempt(?P<n>\d+)$"), "phase_b", "attempt{n}"),
    (re.compile(r"^autoinit_device_canary_attempt(?P<n>\d+)$"),
     "device_canary", "attempt{n}"),
    (re.compile(r"^autoinit_measurement_attempt(?P<n>\d+)$"),
     "measurement", "attempt{n}"),
    #: Aggregate directories from an earlier convention: one directory holding
    #: several attempts. Registered as an aggregate rather than split apart,
    #: because splitting means moving files and the migration is future-only.
    (re.compile(r"^autoinit_continuation_attempts$"),
     "recovery_continuation", "legacy_aggregate"),
    (re.compile(r"^autoinit_phase_a_attempts$"), "phase_a", "legacy_aggregate"),
]

#: Flat per-attempt FILES that belong to a run but predate the layout. Recorded
#: as legacy references too, so "is every attempt artifact accounted for" has a
#: yes/no answer.
FILE_CONVENTIONS = [
    (re.compile(r"^autoinit_c1_attempt(?P<n>\d+r?)_grant\.json$"),
     "phase_c1", "attempt{n}", "governance.grant"),
    (re.compile(r"^autoinit_phase_a_attempt(?P<n>\d+)_grant\.json$"),
     "phase_a", "attempt{n}", "governance.grant"),
    (re.compile(r"^autoinit_phase_b_grant_attempt(?P<n>\d+)\.json$"),
     "phase_b", "attempt{n}", "governance.grant"),
    (re.compile(r"^autoinit_recovery_continuation_attempt(?P<n>\d+)_grant\.json$"),
     "recovery_continuation", "attempt{n}", "governance.grant"),
    (re.compile(r"^autoinit_phase_b_attempt(?P<n>\d+)\.json$"),
     "phase_b", "attempt{n}", "closeout.outcome"),
    (re.compile(r"^autoinit_continuation_b_attempt(?P<n>\d+)\.json$"),
     "continuation_b", "attempt{n}", "closeout.outcome"),
    (re.compile(r"^autoinit_attempt(?P<n>\d+)_probe_reuse\.json$"),
     "recovery_continuation", "attempt{n}", "evidence.replay"),
    (re.compile(r"^autoinit_attempt(?P<n>\d+)_retention_verification\.json$"),
     "recovery_continuation", "attempt{n}", "evidence.replay"),
]


def discover(repo_root: Path) -> list[dict]:
    logs = repo_root / "logs"
    found: list[dict] = []
    for entry in sorted(logs.iterdir()):
        rel = f"logs/{entry.name}"
        for pattern, experiment, attempt_tpl in CONVENTIONS:
            m = pattern.match(entry.name)
            if m and entry.is_dir():
                found.append({
                    "experiment_id": experiment,
                    "attempt_id": attempt_tpl.format(**m.groupdict()),
                    "layout_version": 1,
                    "kind": "directory",
                    "root": rel,
                })
                break
        else:
            for pattern, experiment, attempt_tpl, role in FILE_CONVENTIONS:
                m = pattern.match(entry.name)
                if m and entry.is_file():
                    found.append({
                        "experiment_id": experiment,
                        "attempt_id": attempt_tpl.format(**m.groupdict()),
                        "layout_version": 1,
                        "kind": "file",
                        "role": role,
                        "root": rel,
                    })
                    break
    for record in found:
        record.update(digest=directory_digest(record["root"], repo_root)["digest"],
                      n_files=directory_digest(record["root"], repo_root)["n_files"])
    return found


def discover_v2(repo_root: Path) -> list[dict]:
    """Run-layout-v2 runs, found by the presence of a manifest."""
    runs = repo_root / "logs/runs"
    out: list[dict] = []
    if not runs.is_dir():
        return out
    for manifest in sorted(runs.glob("*/*/manifest.json")):
        doc = json.loads(manifest.read_text())
        layout = RunLayout(doc["experiment_id"], doc["attempt_id"],
                           repo_root=repo_root)
        out.append({
            "experiment_id": doc["experiment_id"],
            "attempt_id": doc["attempt_id"],
            "layout_version": RUN_LAYOUT_VERSION,
            "kind": "directory",
            "root": layout.rel_root,
            "manifest_sha256": doc["self_sha256"],
        })
    return out


def build_index(repo_root: Path) -> dict:
    legacy = discover(repo_root)
    modern = discover_v2(repo_root)
    return {
        "schema": INDEX_SCHEMA,
        "_contract": (
            "Every run this repository has recorded, in one place. Legacy-v1 "
            "entries are REFERENCES: nothing was moved, renamed or copied, and "
            "each carries a digest so that can be proven. Registering a run "
            "confers nothing — this file authorizes no spend, no launch and no "
            "attempt."),
        "layout": {
            "current_version": RUN_LAYOUT_VERSION,
            "root": "logs/runs/<experiment_id>/<attempt_id>/",
            "rule": ("run-layout-v2 is FUTURE-ONLY. Historical runs keep their "
                     "paths; new runs are created under logs/runs/ and are not "
                     "dual-written anywhere else."),
        },
        "counts": {"legacy_v1": len(legacy), "v2": len(modern)},
        "runs": legacy + modern,
        "authorizes": "nothing",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    index = build_index(REPO_ROOT)
    problems = [f"{e['root']}: {why}" for e in index["runs"]
                for ok, why in [resolve_run(e, REPO_ROOT)] if not ok]
    for entry in index["runs"]:
        print(f"  {entry['layout_version']}  {entry['experiment_id']:22} "
              f"{entry['attempt_id']:10} {entry['n_files'] if 'n_files' in entry else '':>4} "
              f"{entry['root']}")
    print(f"\n{index['counts']['legacy_v1']} legacy-v1, {index['counts']['v2']} v2")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1

    if args.write:
        out = REPO_ROOT / INDEX_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(index, indent=1) + "\n")
        print(f"wrote {INDEX_PATH}")
    else:
        print("(dry run; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
