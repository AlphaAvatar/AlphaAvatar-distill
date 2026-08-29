"""When a frozen executable digest moves for a reason worth recording.

`scripts/pod/autoinit_preflight_setup.sh` is simultaneously a member of three
frozen source sets — Phase A's harness, Phase B's executable, the recovery
continuation's — **and** the single `SESSION_KIND` dispatcher every launchable
session passes through. Adding a session therefore moves digests that were frozen
before that session existed, and there is no version of "add a session" that
avoids it: the runner hardcodes one setup script.

Three responses were available and two are wrong. Re-freezing the preregistration
destroys the evidence of what the completed run actually executed. Deleting the
gate destroys its meaning. So the gate changes **shape, not strength**: drift is
allowed only when it is declared, additive, and leaves every pre-existing dispatch
branch byte-identical — which is the property that actually protects an existing
session, since each takes its own branch and cannot reach a new one.

The note is an artifact this repository writes, so nothing here trusts it. Branch
hashes are re-derived from the script and compared; the note supplies the claim
and the script supplies the evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

#: Where the declaration lives. One file, so a second undeclared change cannot
#: hide behind a differently-named note.
NOTE_PATH = "logs/autoinit_phase_b_post_freeze_changes.json"

SETUP_SCRIPT = "scripts/pod/autoinit_preflight_setup.sh"

_BRANCH = re.compile(r'^(?:el)?if \[ "\$SESSION_KIND" = "([a-z_]+)" \]; then$', re.M)


def dispatch_branch_hashes(repo_root: str | Path = ".") -> dict[str, str]:
    """One hash per `SESSION_KIND` branch, sliced at the next branch keyword.

    Sliced at the next `elif` **of any kind**, not at a named one: an inserted
    branch would otherwise be absorbed into its predecessor's slice and the
    predecessor would read as changed when it is not.
    """
    text = (Path(repo_root) / SETUP_SCRIPT).read_text()
    marks = [(m.group(1), m.start(), m.end()) for m in _BRANCH.finditer(text)]
    if not marks:
        raise ValueError(f"{SETUP_SCRIPT} declares no SESSION_KIND branches")
    end_all = text.index("\nelse\n", marks[0][1])
    out: dict[str, str] = {}
    for i, (kind, _, body_start) in enumerate(marks):
        body_end = marks[i + 1][1] if i + 1 < len(marks) else end_all
        out[kind] = hashlib.sha256(text[body_start:body_end].encode()).hexdigest()
    return out


def accounted_for(frozen_digest: str, observed_digest: str,
                  repo_root: str | Path = ".") -> tuple[bool, str]:
    """Is this drift declared, additive, and harmless to existing sessions?

    Returns `(False, reason)` for an undeclared, stale, non-additive, or
    branch-touching change — all of which must still fail closed.
    """
    if frozen_digest == observed_digest:
        return True, "the executable is the one that was frozen"

    root = Path(repo_root)
    note_path = root / NOTE_PATH
    if not note_path.is_file():
        return False, (
            f"the executable digests to {observed_digest[:12]}… but the record "
            f"froze {frozen_digest[:12]}…, and nothing declares why")
    try:
        note = json.loads(note_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"{NOTE_PATH} is not readable: {exc}"

    if note.get("frozen_digest") != frozen_digest:
        return False, (f"{NOTE_PATH} describes a freeze at "
                       f"{str(note.get('frozen_digest'))[:12]}…, not "
                       f"{frozen_digest[:12]}…")
    if note.get("post_freeze_digest") != observed_digest:
        return False, (f"{NOTE_PATH} accounts for "
                       f"{str(note.get('post_freeze_digest'))[:12]}… but the tree "
                       f"digests to {observed_digest[:12]}…; a further change was "
                       "made and not declared")
    change = note.get("change") or {}
    if change.get("additive_only") is not True or change.get("lines_removed") != 0:
        return False, (f"{NOTE_PATH} declares a change that is not additive; a "
                       "removal or edit must fail closed")
    branches = note.get("dispatch_branches") or {}
    if branches.get("pre_existing_changed"):
        return False, (f"a pre-existing SESSION_KIND branch changed: "
                       f"{branches['pre_existing_changed']}. Existing sessions run "
                       "that code")

    # Re-derived, not believed.
    observed_branches = dispatch_branch_hashes(root)
    for kind, recorded in (branches.get("pre_existing_unchanged") or {}).items():
        if observed_branches.get(kind) != recorded:
            return False, (f"the {kind!r} dispatch branch does not hash to what "
                           f"{NOTE_PATH} records; it changed after the change was "
                           "reviewed")
    added = sorted(set(observed_branches) - set(branches.get("pre_existing_unchanged") or {}))
    if added != sorted(branches.get("added") or []):
        return False, (f"the script declares branches {added} but the note records "
                       f"{sorted(branches.get('added') or [])}")
    return True, (f"drift declared in {NOTE_PATH}: additive, "
                  f"{len(observed_branches)} branches, "
                  f"{len(branches.get('pre_existing_unchanged') or {})} pre-existing "
                  "byte-identical")
