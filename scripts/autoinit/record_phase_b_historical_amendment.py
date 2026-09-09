#!/usr/bin/env python3
"""Append ONE reviewed amendment to the Phase-B historical ledger. Zero cost.

    PYTHONPATH=src python scripts/autoinit/record_phase_b_historical_amendment.py \
        --commit <source-repair-sha> \
        --what "..." --why-shared-owner "..." --why-not-c1-override "..." \
        --maintainer "..."

APPEND ONLY. It reads every existing entry, keeps them byte-for-byte, and adds
exactly one. It refuses to rewrite, renumber, reorder or drop history — which is
the failure mode of the v1 generator it stands beside, and the reason this is a
separate command rather than a flag on that one.

It records HISTORY. It confers nothing: every entry it writes asserts
`launch_compatible_with_frozen_preregistration: false`, and
`aadistill.governance.post_freeze.historical_accounted_for` refuses a ledger that
says otherwise. The paid Phase-B launch gate never reads this file.

Every quantity is DERIVED from git and from the tree — numstat including honest
removals, per-file before/after hashes, the patch hash, the sealed note's hash.
A field a maintainer owns (what, why, the P12 decision) is an argument; a field
the repository can compute is never taken on trust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.preflight import HARNESS_SOURCE_FILES_V1  # noqa: E402
from experiments.phase_a.plan import PHASE_A_HARNESS_SOURCE_FILES_V1  # noqa: E402
from experiments.phase_b.plan import PHASE_B_EXECUTABLE_SOURCE_FILES_V1, phase_b_source_digest  # noqa: E402
from experiments.phase_b.continuation import CONTINUATION_SOURCE_FILES_V2  # noqa: E402
from experiments.phase_c1.authorization import C1_HARNESS_SOURCE_FILES_V1  # noqa: E402
from aadistill.governance.post_freeze import (  # noqa: E402
    HISTORICAL_LEDGER_SCHEMA,
    entry_self_hash,
)
from experiments.phase_b.post_freeze import (  # noqa: E402
    HISTORICAL_LEDGER_PATH,
    SEALED_LEGACY_NOTE,
)

PREREG = "logs/autoinit_phase_b_preregistration.json"


def git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def sha_file(rel: str) -> str:
    return hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()


def sha_blob_at(commit: str, rel: str) -> str | None:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{commit}:{rel}"],
                         capture_output=True)
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", required=True,
                    help="the source-repair commit this amendment accounts for")
    ap.add_argument("--what", required=True)
    ap.add_argument("--why-shared-owner", required=True)
    ap.add_argument("--why-not-c1-override", required=True)
    ap.add_argument("--maintainer", required=True,
                    help="the human/P12 decision authorizing this amendment")
    ap.add_argument("--reviewed-base", required=True)
    ap.add_argument("--parent", default=None, help=(
        "the commit the change is measured FROM. Defaults to <commit>^, which "
        "is right for a single repair. A change that spans several commits -- "
        "the initialization migration spans thirteen -- must name its base, or "
        "the amendment reports one commit's numstat while claiming to account "
        "for all of them."))
    ap.add_argument("--ledger", default=None, help=(
        "write to this ledger instead of the committed one. For tests: a test "
        "that points the recorder at the repository can append to a governance "
        "artifact whenever its 'this will be refused' precondition stops "
        "holding, which is exactly what happened during the initialization "
        "migration."))
    args = ap.parse_args()

    commit = git("rev-parse", args.commit).strip()
    parent = git("rev-parse", args.parent or f"{commit}^").strip()

    ledger_path = Path(args.ledger) if args.ledger else REPO_ROOT / HISTORICAL_LEDGER_PATH
    prereg_doc = json.loads((REPO_ROOT / PREREG).read_text())
    frozen = prereg_doc["executable_source"]["digest"]
    live = phase_b_source_digest(REPO_ROOT)["digest"]

    if ledger_path.is_file():
        led = json.loads(ledger_path.read_text())
        if led.get("schema") != HISTORICAL_LEDGER_SCHEMA:
            raise SystemExit(f"refusing: {HISTORICAL_LEDGER_PATH} is not a "
                             f"{HISTORICAL_LEDGER_SCHEMA} ledger")
        existing = list(led.get("amendments") or [])
        anchors = led["anchors"]
        if anchors["phase_b_frozen_source_digest"] != frozen:
            raise SystemExit("refusing: the ledger anchors a different freeze")
    else:
        # The legacy note's post-freeze digest is where history stood before the
        # first amendment. Anchored, not copied forward.
        legacy = json.loads((REPO_ROOT / SEALED_LEGACY_NOTE).read_text())
        existing = []
        anchors = {
            "phase_b_preregistration": PREREG,
            "phase_b_preregistration_sha256": sha_file(PREREG),
            "phase_b_frozen_source_digest": frozen,
            "phase_b_frozen_set_version": prereg_doc["executable_source"].get(
                "set_version"),
            "sealed_legacy_note": SEALED_LEGACY_NOTE,
            "sealed_legacy_note_sha256": sha_file(SEALED_LEGACY_NOTE),
            "legacy_post_freeze_digest": legacy["post_freeze_digest"],
            "_why": ("the ledger begins where the sealed v1 note stopped. That "
                     "note is anchored by hash and never rewritten; this file "
                     "appends the facts it cannot express."),
        }

    previous = (existing[-1]["new_live_digest"] if existing
                else anchors["legacy_post_freeze_digest"])
    if previous == live:
        raise SystemExit("refusing: the ledger already accounts for this tree")

    changed = sorted(
        rel for rel in PHASE_B_EXECUTABLE_SOURCE_FILES_V1
        if sha_blob_at(parent, rel) != sha_file(rel))
    if not changed:
        raise SystemExit("refusing: no Phase-B set member differs from the parent")

    numstat: dict[str, list[int]] = {}
    for line in git("diff", "--numstat", parent, commit, "--", *changed).splitlines():
        a, d, rel = (line.split("\t") + ["", "", ""])[:3]
        if rel:
            numstat[rel] = [int(a), int(d)]
    patch = git("diff", parent, commit, "--", *changed)

    entry = {
        "amendment_id": f"PHB-HA-{len(existing) + 1:03d}",
        "schema_version": 1,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviewed_base_commit": args.reviewed_base,
        "source_repair_commit": commit,
        "source_repair_parent": parent,
        "previous_live_digest": previous,
        "new_live_digest": live,
        "changed_files": changed,
        "file_sha256_before": {rel: sha_blob_at(parent, rel) for rel in changed},
        "file_sha256_after": {rel: sha_file(rel) for rel in changed},
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "numstat": numstat,
        "numstat_rule": "git diff --numstat <parent> <commit> -- <changed_files>",
        "additive_only": False,
        "lines_removed": sum(v[1] for v in numstat.values()),
        "historical_only": True,
        "launch_compatible_with_frozen_preregistration": False,
        "phase_b_science_changed": False,
        "phase_b_results_changed": False,
        "operational_classification": (
            "post-provider resource ownership, watchdog establishment and "
            "verified teardown"),
        "what": args.what,
        "why_the_owner_is_shared_session_runner": args.why_shared_owner,
        "why_a_c1_only_override_was_rejected": args.why_not_c1_override,
        "maintainer_authorization": args.maintainer,
        "does_not_authorize": [
            "any Phase-B execution",
            "re-freezing the Phase-B preregistration",
            "reinterpreting the completed Phase-B result",
            "launch compatibility with the frozen preregistration",
            "any relaxation of autoinit_phase_b_launch.preregistration_gate",
        ],
        "previous_entry_sha256": existing[-1]["entry_sha256"] if existing else None,
        #: The same shared file sits in other hash-bound sets. Derived, so the
        #: blast radius is recorded here rather than discovered later.
        "also_affected_hash_bound_sets": sorted(
            name for name, files in (
                ("experiments.phase_a.plan.PHASE_A_HARNESS_SOURCE_FILES_V1",
                 PHASE_A_HARNESS_SOURCE_FILES_V1),
                ("experiments.phase_b.continuation."
                 "CONTINUATION_SOURCE_FILES_V2", CONTINUATION_SOURCE_FILES_V2),
                ("aadistill.governance.authorization.HARNESS_SOURCE_FILES_V1",
                 HARNESS_SOURCE_FILES_V1),
                ("experiments.phase_c1.authorization."
                 "C1_HARNESS_SOURCE_FILES_V1", C1_HARNESS_SOURCE_FILES_V1),
            ) if set(changed) & set(files)),
    }
    entry["entry_sha256"] = entry_self_hash(entry)

    ledger = {
        "schema": HISTORICAL_LEDGER_SCHEMA,
        "_what_this_is": (
            "an APPEND-ONLY ledger of reviewed, post-completion drift in the "
            "Phase-B executable source set. It exists because 'this tree may "
            "not launch Phase B' and 'nobody explained why the digest moved' "
            "are different facts and the repository had one mechanism for "
            "both. It records history and confers nothing."),
        "consumed_by_a_paid_launch_gate": False,
        "launch_compatibility_owner": (
            "scripts/pod/autoinit_phase_b_launch.py::preregistration_gate, via "
            "aadistill.governance.post_freeze.accounted_for — which is unchanged "
            "and still refuses non-additive drift"),
        "immutability": (
            f"{PREREG} and {SEALED_LEGACY_NOTE} are anchored by hash above and "
            "are never rewritten by this command."),
        "anchors": anchors,
        "amendments": [*existing, entry],
    }
    ledger_path.write_text(json.dumps(ledger, indent=1) + "\n")
    print(f"appended {entry['amendment_id']} to {HISTORICAL_LEDGER_PATH}")
    print(f"  {previous[:16]}… -> {live[:16]}…")
    print(f"  changed: {changed}")
    print(f"  numstat: {numstat}  (lines_removed={entry['lines_removed']})")
    print(f"  entries now: {len(ledger['amendments'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
