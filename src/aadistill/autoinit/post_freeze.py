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

#: `[a-z0-9_]`, not `[a-z_]`. The original class could not match a kind with a
#: digit, so a `c1` branch was invisible here: its body was absorbed into the
#: PRECEDING branch's slice, and this module would then report that pre-existing
#: branch as changed and fail an additive, harmless drift. A gate that cannot see
#: a branch is not a stricter gate, it is a wrong one.
_BRANCH = re.compile(r'^(?:el)?if \[ "\$SESSION_KIND" = "([a-z0-9_]+)" \]; then$',
                     re.M)


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


# --- historical accounting, which is NOT launch compatibility ---------------
#
# `accounted_for` above answers one question: may a Phase-B launch run against
# this tree? It requires the drift to be additive and branch-identical, and it
# must keep saying NO to anything else — that is what protects a completed
# result from being reinterpreted.
#
# But "Phase B may not launch against this tree" and "nobody ever explained why
# the digest moved" are different facts, and the repository had only one
# mechanism for both. A reviewed, non-additive operational repair to a SHARED
# runtime file therefore had nowhere to be recorded: declaring it additive would
# be false, and not declaring it left an unexplained digest.
#
# So the two are separated. This ledger records history and confers NOTHING. It
# is deliberately not readable by `preregistration_gate`, and every entry has to
# say so in its own fields before it will verify.

HISTORICAL_LEDGER_PATH = "logs/autoinit_phase_b_historical_amendments.json"
HISTORICAL_LEDGER_SCHEMA = "aadistill.autoinit.phase_b_historical_amendments/v1"

#: The sealed v1 declaration. Anchored by hash, never rewritten: an amendment
#: that could edit the record it amends is not an amendment.
SEALED_LEGACY_NOTE = NOTE_PATH


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry_self_hash(entry: dict) -> str:
    """Hash of an entry with its own hash field removed."""
    body = {k: v for k, v in entry.items() if k != "entry_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    import subprocess

    out = subprocess.run(["git", "-C", str(repo_root), *args],
                         capture_output=True, text=True, timeout=120)
    return out.stdout if out.returncode == 0 else ""


def historical_accounted_for(frozen_digest: str, observed_digest: str,
                             repo_root: str | Path = ".") -> tuple[bool, str]:
    """Is the drift from `frozen_digest` to `observed_digest` reviewed HISTORY?

    Answers only that. A `True` here is not permission to launch anything, and
    every entry must assert `launch_compatible_with_frozen_preregistration:
    false` before it verifies — so a ledger cannot be written that quietly
    claims otherwise.

    Nothing is believed. The sealed note's hash, the per-file after-hashes, the
    numstat and the patch hash are all re-derived from the tree and from git.
    """
    if frozen_digest == observed_digest:
        return True, "the executable is the one that was frozen"

    root = Path(repo_root)
    path = root / HISTORICAL_LEDGER_PATH
    if not path.is_file():
        return False, (f"the executable digests to {observed_digest[:12]}… but "
                       f"the record froze {frozen_digest[:12]}…, and "
                       f"{HISTORICAL_LEDGER_PATH} does not exist")
    try:
        led = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"{HISTORICAL_LEDGER_PATH} is not readable: {exc}"

    if led.get("schema") != HISTORICAL_LEDGER_SCHEMA:
        return False, f"{HISTORICAL_LEDGER_PATH} declares schema {led.get('schema')!r}"
    if led.get("consumed_by_a_paid_launch_gate") is not False:
        return False, ("the ledger does not declare itself unusable by a paid "
                       "launch gate")

    anchors = led.get("anchors") or {}
    if anchors.get("phase_b_frozen_source_digest") != frozen_digest:
        return False, (f"the ledger anchors freeze "
                       f"{str(anchors.get('phase_b_frozen_source_digest'))[:12]}…, "
                       f"not {frozen_digest[:12]}…")

    prereg = root / "logs/autoinit_phase_b_preregistration.json"
    if not prereg.is_file():
        return False, "the immutable Phase-B preregistration is missing"
    if anchors.get("phase_b_preregistration_sha256") != _sha_file(prereg):
        return False, ("the immutable Phase-B preregistration has changed since "
                       "the ledger anchored it")
    sealed = root / SEALED_LEGACY_NOTE
    if not sealed.is_file():
        return False, f"the sealed legacy note {SEALED_LEGACY_NOTE} is missing"
    if anchors.get("sealed_legacy_note_sha256") != _sha_file(sealed):
        return False, ("the sealed legacy v1 declaration has been modified; it "
                       "is anchored by hash and must stay byte-identical")

    entries = led.get("amendments") or []
    if not entries:
        return False, "the ledger records no amendment"

    previous = anchors.get("legacy_post_freeze_digest")
    for i, e in enumerate(entries):
        where = f"amendment {e.get('amendment_id', i)}"
        if e.get("entry_sha256") != entry_self_hash(e):
            return False, f"{where}: self-hash does not match its contents"
        if e.get("previous_entry_sha256") != (
                entries[i - 1]["entry_sha256"] if i else None):
            return False, f"{where}: the entry chain is broken"
        if e.get("previous_live_digest") != previous:
            return False, (f"{where}: claims to follow "
                           f"{str(e.get('previous_live_digest'))[:12]}… but the "
                           f"chain is at {str(previous)[:12]}…")
        for flag, want in (("historical_only", True),
                           ("launch_compatible_with_frozen_preregistration", False),
                           ("phase_b_science_changed", False)):
            if e.get(flag) is not want:
                return False, f"{where}: {flag} must be {want!r}"
        if not (e.get("maintainer_authorization") or "").strip():
            return False, f"{where}: no human/P12 decision is recorded"
        previous = e.get("new_live_digest")

    if previous != observed_digest:
        return False, (f"the ledger accounts up to {str(previous)[:12]}… but the "
                       f"tree digests to {observed_digest[:12]}…; a further "
                       "change was made and not declared")

    ok, why = _rederive_last_entry(root, entries[-1])
    if not ok:
        return False, why
    return True, (f"historical drift accounted for across {len(entries)} "
                  f"amendment(s); launch compatibility is NOT implied")


def _rederive_last_entry(root: Path, e: dict) -> tuple[bool, str]:
    """Re-derive the newest entry's evidence from the tree and from git."""
    changed = list(e.get("changed_files") or [])
    if not changed:
        return False, "the newest amendment names no changed file"

    after = e.get("file_sha256_after") or {}
    for rel in changed:
        p = root / rel
        if not p.is_file():
            return False, f"{rel} is named by the amendment and is absent"
        if after.get(rel) != _sha_file(p):
            return False, (f"{rel} does not hash to what the amendment records "
                           "as its after-state")

    commit, parent = e.get("source_repair_commit"), e.get("source_repair_parent")
    if not commit or not parent:
        return False, "the amendment does not name its commit and parent"
    if _git(root, "rev-parse", f"{commit}^").strip() != parent:
        return False, (f"{commit[:12]}…'s parent is not the recorded "
                       f"{parent[:12]}…")

    observed = {}
    for line in _git(root, "diff", "--numstat", parent, commit, "--",
                     *changed).splitlines():
        a, d, rel = (line.split("\t") + ["", "", ""])[:3]
        if rel:
            observed[rel] = [int(a), int(d)]
    if observed != {k: list(v) for k, v in (e.get("numstat") or {}).items()}:
        return False, (f"the recorded numstat {e.get('numstat')} is not what git "
                       f"reports ({observed})")

    patch = _git(root, "diff", parent, commit, "--", *changed)
    if hashlib.sha256(patch.encode()).hexdigest() != e.get("patch_sha256"):
        return False, "the recorded patch hash is not the hash of that diff"
    return True, "re-derived"
