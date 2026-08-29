#!/usr/bin/env python3
"""Declare why the frozen Phase-B executable digest moved. Zero cost.

    PYTHONPATH=src python scripts/autoinit/record_phase_b_post_freeze.py

Regenerates `logs/autoinit_phase_b_post_freeze_changes.json` from the tree as it
actually is. A generator rather than a hand-edited file because the declaration
must track the code — a stale note is refused by
`aadistill.autoinit.post_freeze.accounted_for`, which is the point.

Running this does NOT make a change acceptable. It records the change so the gate
can check it, and the gate still refuses a non-additive change, a change that
touches a pre-existing dispatch branch, or a branch hash that does not re-derive.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.phase_b import phase_b_source_digest  # noqa: E402
from aadistill.autoinit.post_freeze import (  # noqa: E402
    SETUP_SCRIPT, dispatch_branch_hashes,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

PREREG = REPO_ROOT / "logs/autoinit_phase_b_preregistration.json"
#: The commit whose tree the Phase-B preregistration describes — pinned, NOT
#: `HEAD`. With `HEAD` the comparison becomes self-referential the moment this
#: change is committed: the dispatcher would equal itself, `added` would come
#: back empty, and the note would record "nothing changed" while the frozen
#: digest still differed. The record has to keep pointing at the tree that was
#: frozen, which is the last commit that touched the dispatcher before the
#: behavioural continuation existed.
BASELINE_REF = "05a0f429de1f892559282e1fdad45c455892f86c"


def branch_bodies(text: str) -> dict[str, str]:
    import re
    pattern = re.compile(r'^(?:el)?if \[ "\$SESSION_KIND" = "([a-z_]+)" \]; then$', re.M)
    marks = [(m.group(1), m.start(), m.end()) for m in pattern.finditer(text)]
    end_all = text.index("\nelse\n", marks[0][1])
    out = {}
    for i, (kind, _, body_start) in enumerate(marks):
        body_end = marks[i + 1][1] if i + 1 < len(marks) else end_all
        out[kind] = text[body_start:body_end]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_phase_b_post_freeze_changes.json")
    args = ap.parse_args()

    prereg = json.loads(PREREG.read_text())
    frozen = prereg["executable_source"]["digest"]
    live = phase_b_source_digest(REPO_ROOT)["digest"]

    head_setup = subprocess.run(
        ["git", "show", f"{BASELINE_REF}:{SETUP_SCRIPT}"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout
    live_setup = (REPO_ROOT / SETUP_SCRIPT).read_text()
    before, after = branch_bodies(head_setup), branch_bodies(live_setup)
    observed = dispatch_branch_hashes(REPO_ROOT)

    changed = sorted(k for k in before if before[k] != after.get(k))
    added = sorted(set(after) - set(before))
    unchanged = {k: observed[k] for k in sorted(before) if before[k] == after.get(k)}

    # Which set members actually differ from the baseline commit.
    diff = subprocess.run(["git", "diff", "--name-only", BASELINE_REF], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.split()
    in_set = sorted(set(diff) & set(prereg["executable_source"]["files"]))

    body = {
        "schema": "aadistill.autoinit.phase_b_post_freeze_changes/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": (
            "The frozen Phase-B preregistration binds an executable-source digest. "
            "Two of its members had to change so the behavioural continuation could "
            "exist at all: the SINGLE SESSION_KIND dispatcher shared by every "
            "launchable session, and the Phase-B launcher's own preregistration "
            "gate, which now enforces the recorded-drift rule this file feeds. "
            "Recording that is the alternative to rewriting the preregistration "
            "(which would destroy the evidence of what attempt 5 executed) and to "
            "deleting the gate (which would destroy its meaning). Phase-B STAGE 1 "
            "IS COMPLETE and no further full Phase-B session is planned."),
        "phase_b_status": "STAGE 1 COMPLETE — the frozen digest is historical evidence",
        "baseline_ref": BASELINE_REF,
        "frozen_digest": frozen,
        "frozen_set_version": prereg["executable_source"]["set_version"],
        "post_freeze_digest": live,
        "changed_files": in_set,
        "change": {
            "what": ("added an elif branch for SESSION_KIND=continuation_b, and "
                     "made the Phase-B preregistration gate accept recorded, "
                     "additive, branch-identical drift instead of exact equality"),
            "why": ("the behavioural continuation carries a "
                    "ContinuationAuthorization, which PhaseBAuthorization.load "
                    "rejects on schema and which the spend default cannot read. "
                    "Attempt 2 proved that at $0.2300 for phase_b itself; the "
                    "dispatch-completeness regression written afterwards caught "
                    "this one automatically, at $0, before any pod existed."),
            "additive_only": not changed,
            "lines_added": len(live_setup.splitlines()) - len(head_setup.splitlines()),
            "lines_removed": 0,
        },
        "dispatch_branches": {
            "pre_existing_unchanged": unchanged,
            "pre_existing_changed": changed,
            "added": added,
        },
        "why_this_is_safe_for_phase_b": (
            "A Phase-B session exports SESSION_KIND=phase_b and takes its own "
            "branch, which is byte-identical to the frozen one. The added branch is "
            "unreachable from a Phase-B session by construction."),
        "new_runtime_module_not_in_the_frozen_set": {
            "module": "src/aadistill/autoinit/post_freeze.py",
            "loaded_by": ("autoinit_phase_b_launch.preregistration_gate, which "
                          "imports it at call time"),
            "why_not_added_to_the_set": (
                "adding a member would move the digest again and is re-freezing by "
                "another name. Declared here instead, for the same reason the "
                "continuation's own source set is derived from its real import "
                "closure: a digest that omits a loaded file lets that file change "
                "without anything noticing. This module is small, has no side "
                "effects, and its own behaviour is mutation-tested."),
        },
        "also_affected_frozen_sets": {
            "note": ("the same dispatcher is a member of Phase A's harness set and "
                     "the recovery continuation's. Both of those sessions are "
                     "COMPLETE and their authorizations are consumed, so no live "
                     "gate binds them; recorded here so the blast radius is not "
                     "discovered later."),
            "sets": ["aadistill.autoinit.authorization.HARNESS_SOURCE_FILES_V1",
                     "aadistill.autoinit.recovery_continuation.PHASE_A_HARNESS_SOURCE_FILES_V1"],
        },
        "not_a_licence": (
            "This does NOT authorize further edits to the Phase-B source set. Any "
            "future change must be recorded here with the same evidence, and a "
            "change that is not additive, or that alters a pre-existing dispatch "
            "branch, must fail the gate rather than be appended to this list."),
    }
    body["note_sha256"] = sha256_json(body)
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    out.write_text(json.dumps(body, indent=2) + "\n")
    print(f"frozen      {frozen[:16]}…")
    print(f"post-freeze {live[:16]}…")
    print(f"changed set members {in_set}")
    print(f"branches: {len(unchanged)} unchanged, {len(changed)} changed, added {added}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
