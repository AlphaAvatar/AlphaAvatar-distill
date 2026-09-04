"""The gates a session runs at $0, before a pod exists, as reusable factories.

These were overridden methods on three launcher subclasses — which is why the
micro-preflight quietly had no commit gate at all while the continuation and
Phase A each had their own copy of one, and nobody could see the asymmetry
without reading three files. A precheck is now a value in
`SessionSpec.precheck`, so a reader can count them.

Each returns `(ok, message)`. A false `ok` aborts before the provider is
contacted; the message is what the session record and the launch log say.

(The design in `docs/SESSION_ARCHITECTURE.md` names two new modules. This is a
third, deliberately: the shared gates run git and read the relay, and putting
that I/O in the module that defines the frozen types would make a declaration
module do work. The composition is the same either way.)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable

from .session import SessionContext


def lineage_from_authorized_base(repo_root: Path, base: str | None, commit: str,
                                 auth_path: str | tuple[str, ...]) -> dict:
    """Is `commit` the authorized base plus the permitted artifacts, and nothing else?

    `auth_path` accepts a tuple as well as a single path. That generalization was
    added on 2026-09-04 for `pod_environment_gate`, which asks the identical
    question of a different base — "is this tree the SWEPT commit plus only the
    readiness record?" — and would otherwise have needed a second copy of these
    fifty lines. Two copies of a lineage rule is how the original two session
    gates drifted, which is why this function exists at all.

    The existing caller passes a single path and is unchanged in strictness: one
    permitted path in, one permitted path out.

    The harness digest proves the *declared harness files* did not move, and the
    auth-blob check proves the driver will load this exact grant. Neither says
    anything about the rest of the tree. `authorized_session_commit` is
    necessarily issued against the clean PRE-authorization HEAD — the artifact
    cannot be committed before it exists — so the commit the pod actually checks
    out is always a later one, and until this check nothing constrained what else
    rode along in that gap.

    Closing it needs two facts, both from git: the final commit **descends from**
    the authorized base, and the only path that differs between them is the
    authorization artifact.

    Returns a record rather than a bool so the launcher can log exactly what it
    saw, including on the paths that refuse.
    """
    allowed = (auth_path,) if isinstance(auth_path, str) else tuple(auth_path)
    out = {"authorized_base": base, "session_commit": commit,
           "allowed_paths": list(allowed),
           "descends_from_base": None, "changed_paths": None,
           "unexpected_paths": None, "ok": False, "reason": ""}
    if not base:
        out["reason"] = ("the authorization declares no authorized_session_commit, "
                         "so there is no base to constrain the tree against")
        return out
    known = subprocess.run(["git", "cat-file", "-e", f"{base}^{{commit}}"],
                           capture_output=True, cwd=repo_root)
    if known.returncode != 0:
        out["reason"] = f"the authorized base {base} is not a commit in this repository"
        return out
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", base, commit],
                         capture_output=True, cwd=repo_root)
    out["descends_from_base"] = anc.returncode == 0
    if not out["descends_from_base"]:
        out["reason"] = (f"{commit} does not descend from the authorized base "
                         f"{base}; it is a different line of history")
        return out
    diff = subprocess.run(["git", "diff", "--name-only", base, commit],
                          capture_output=True, text=True, cwd=repo_root)
    if diff.returncode != 0:
        out["reason"] = f"could not diff {base}..{commit}: {diff.stderr.strip()[:200]}"
        return out
    changed = [p for p in diff.stdout.split("\n") if p.strip()]
    out["changed_paths"] = changed
    out["unexpected_paths"] = [p for p in changed if p not in allowed]
    if out["unexpected_paths"]:
        out["reason"] = (
            f"{len(out['unexpected_paths'])} path(s) other than "
            f"{list(allowed)} changed between the base and the session commit: "
            f"{out['unexpected_paths'][:8]}")
        return out
    out["ok"] = True
    out["reason"] = (f"only {sorted(set(changed))} differs from the base"
                     if changed else "identical to the base")
    return out


def session_commit_gate(repo_root: Path, auth_path: str, *,
                        check_lineage: bool) -> Callable[[SessionContext], tuple]:
    """The harness at `--session-commit` must be the authorized one.

    The pod does not run the dev box's working tree: it clones a bundle and
    checks out this commit. Continuation attempt 5 died at $0.1369 on a stale
    binding, and this is the gate that followed.

    Rather than compare two commit hashes — which cannot be equal, since the
    authorization artifact is written before it is committed — it asks the
    question that matters: do the harness files AT THAT COMMIT digest to the
    authorized value, and does that commit carry this exact authorization?

    `check_lineage` adds Phase A's stronger third question: is everything else in
    the tree unchanged from the authorized base? It is a parameter rather than a
    separate function because the difference between the two sessions was one
    call, and duplicating ninety lines to express it is how the two copies
    drifted in the first place.
    """
    def check(ctx: SessionContext) -> tuple[bool, str]:
        commit = ctx.args.session_commit
        entries, missing = [], []
        for rel in sorted(ctx.auth.harness_source_files):
            blob = subprocess.run(["git", "show", f"{commit}:{rel}"],
                                  capture_output=True, cwd=repo_root)
            if blob.returncode != 0:
                missing.append(rel)
                continue
            entries.append({"path": rel,
                            "sha256": hashlib.sha256(blob.stdout).hexdigest()})
        if missing:
            return False, f"{commit} does not contain {missing}"
        digest = hashlib.sha256(
            "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()
        ).hexdigest()
        auth_blob = subprocess.run(["git", "show", f"{commit}:{auth_path}"],
                                   capture_output=True, cwd=repo_root)
        carries_auth = (auth_blob.returncode == 0
                        and auth_blob.stdout == (repo_root / auth_path).read_bytes())
        record = {
            "session_commit": commit,
            "authorized_session_commit": ctx.auth.authorized_session_commit,
            "harness_digest_at_commit": digest,
            "authorized_harness_digest": ctx.auth.harness_source_digest,
            "harness_matches": digest == ctx.auth.harness_source_digest,
            "commit_carries_this_authorization": carries_auth,
            "rule": ("the pod checks out this commit from a bundle; its harness "
                     "must digest to the authorized value and it must carry the "
                     "authorization the driver will load"),
        }
        lineage = None
        if check_lineage:
            lineage = lineage_from_authorized_base(
                repo_root, ctx.auth.authorized_session_commit, commit, auth_path)
            record["lineage"] = lineage
        ctx.evidence["session_commit_check"] = record

        if digest != ctx.auth.harness_source_digest:
            return False, (
                f"the harness at {commit} digests to {digest}, authorized "
                f"{ctx.auth.harness_source_digest}. The pod would run code this "
                "authorization was not granted against.")
        if not carries_auth:
            return False, (
                f"{commit} does not carry this exact {auth_path}; the driver "
                "would load a different authorization, or none.")
        if lineage is not None and not lineage["ok"]:
            return False, lineage["reason"]
        tail = f" and {lineage['reason']}" if lineage else ""
        return True, (f"session commit {commit[:12]} verified: harness digests to "
                      f"{digest[:12]}…, carries the authorization{tail}")

    check.__name__ = ("session_commit_and_lineage" if check_lineage
                      else "session_commit")
    return check


def frozen_science_plan_gate(repo_root: Path, plan_path: str
                             ) -> Callable[[SessionContext], tuple]:
    """The frozen science plan must be the one the authorization names.

    Without this the driver's Stage 0 fails after setup has been paid for.
    """
    def check(ctx: SessionContext) -> tuple[bool, str]:
        frozen = repo_root / plan_path
        if not frozen.is_file():
            return False, (f"no frozen science plan at {frozen}; "
                           "assert_preregistered would have nothing to bind against")
        frozen_hash = json.loads(frozen.read_text()).get("plan_hash")
        ctx.evidence.setdefault("precheck", {})["frozen_plan_hash"] = frozen_hash
        if frozen_hash != ctx.auth.science_plan_hash:
            return False, (f"the frozen plan hashes to {frozen_hash} but the "
                           f"authorization names {ctx.auth.science_plan_hash}")
        return True, f"frozen science plan {frozen_hash[:12]}… matches the authorization"

    check.__name__ = "frozen_science_plan"
    return check


def local_files_gate(repo_root: Path, paths: tuple[str, ...], *, what: str
                     ) -> Callable[[SessionContext], tuple]:
    """Dev-box files a session reads that are not directories under a manifest.

    The continuation reads each permanent control's three record files; under
    `--transport relay` an unstaged one used to be discovered by the transport
    step, after setup had been paid for.
    """
    def check(ctx: SessionContext) -> tuple[bool, str]:
        missing = [p for p in paths if not (repo_root / p).is_file()]
        ctx.evidence.setdefault("precheck", {})[f"{what}_missing"] = missing
        if missing:
            return False, f"{what} missing: {missing}"
        return True, f"{what}: {len(paths)} present"

    check.__name__ = what
    return check
