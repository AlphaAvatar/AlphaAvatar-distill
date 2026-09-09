"""The spend authorization, as an object the launcher must obey.

The preflight *plan* is preregistered and historical; it still says
"PROPOSAL, NOT AUTHORIZED" and is not rewritten to reflect a later conversation.
Authorization is a separate, dated artifact that points at the plan by hash. Two
records, two lifetimes: the plan describes what would be done, the authorization
records who permitted how much of it, when.

The launcher loads it and cannot proceed past what it grants. That matters
because every budget overrun in this project's history came from a limit that
lived in prose: E6b overran by $0.56 with the number in a plan document, and a
finished corpus build idled ~$8.70 because teardown was tied to a generous
backstop rather than to completion.

**This module holds mechanism only.** Which actions an authorization may
express, which files its harness covers and which on-disk keys would claim a
permission are all supplied by the application: a governance primitive that
knows one experiment's stage names is not reusable, and the next experiment
would have to edit it rather than declare itself.

Absence of permission is denial. `ActionPolicy` must be supplied to load an
artifact at all, so an authorization cannot be read by a caller that has not
said what it is allowed to express.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aadistill.infrastructure.manifest import sha256_file, sha256_json


class AuthorizationError(RuntimeError):
    """An action exceeds or falls outside what was authorized."""


#: An action an authorization may express, and how a claim to it appears on
#: disk. Supplied by the application: `phase_a`, `beam_search` and the rest are
#: one project's vocabulary, and a governance primitive that enumerated them
#: could not serve a second experiment.
@dataclass(frozen=True)
class ActionPolicy:
    """What an authorization is permitted to express. Absence is denial."""

    policy_id: str
    #: Actions this authorization MAY grant. Everything else is denied.
    allowed: frozenset[str] = frozenset()
    #: on-disk key -> action. An artifact whose key is truthy for an action
    #: outside `allowed` is refused at load: that is a document claiming a
    #: permission its own policy does not have.
    wire_claims: dict[str, str] = field(default_factory=dict)
    #: action -> what to tell someone who asked for it. Optional.
    refusal_notes: dict[str, str] = field(default_factory=dict)

    def allows(self, action: str) -> bool:
        return action in self.allowed

    def refuse(self, action: str) -> None:
        note = self.refusal_notes.get(action)
        raise AuthorizationError(
            note or (f"{action!r} is not authorized under policy "
                     f"{self.policy_id!r}; absence of permission is denial. "
                     "Stop, report, and obtain a new authorization."))

    def check_claims(self, raw: dict[str, Any], *, where: str = "") -> None:
        """Refuse an artifact that claims a permission this policy denies."""
        claimed = [key for key, action in self.wire_claims.items()
                   if raw.get(key) and not self.allows(action)]
        if claimed:
            raise AuthorizationError(
                f"{where or 'this artifact'} claims {claimed}, which policy "
                f"{self.policy_id!r} does not grant; refusing to load it")


#: A policy that grants nothing and recognises no claim. The safe default for a
#: caller that has not declared one -- it can express no permission at all.
DENY_ALL = ActionPolicy(policy_id="deny_all")


def harness_source_digest(repo_root: str | Path = ".", *,
                          files: tuple[str, ...],
                          set_version: int | None = None) -> dict[str, Any]:
    """Digest a caller-declared harness set.

    `files` is required. It used to default to the preflight's list, which meant
    a caller that forgot to declare its own executable silently got a digest
    over somebody else's -- and the digest verified perfectly.
    """
    root = Path(repo_root)
    if not files:
        raise AuthorizationError(
            "no harness source files were declared; a digest over an empty or "
            "unstated set describes nothing and would authorize anything")
    declared = tuple(files)
    entries = []
    for rel in sorted(declared):
        path = root / rel
        if not path.is_file():
            raise AuthorizationError(
                f"declared harness source {rel!r} is missing; refusing to "
                "authorize a digest over a smaller harness than the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    #: The version belongs to whoever declared the set: two digests over
    #: different sets are not comparable, and the version is what says so.
    return {"digest": digest, "set_version": set_version, "files": entries}


@dataclass(frozen=True)
class SpendAuthorization:
    """What a named maintainer permitted, bound to a plan hash."""

    authorization_id: str
    granted_utc: str
    granted_by: str
    plan_id: str
    plan_hash: str
    expected_usd: float
    hard_cap_usd: float
    authorized_stages: tuple[int, ...]
    stage_conditions: dict[str, str]
    scope_note: str
    #: The executable identity permitted to consume this authorization. Not
    #: provenance: the launcher refuses to create a pod when the harness on disk
    #: does not match. An authorization granted against a rehearsed harness does
    #: not extend to an edited one.
    authorized_session_commit: str | None = None
    harness_source_digest: str | None = None
    #: WHICH files that digest covers. A session that runs a different
    #: executable -- the continuation runs its own launcher, driver and plan
    #: module -- must declare them here. This used to DEFAULT to the preflight's
    #: list, so a session that forgot to declare its own executable digested
    #: somebody else's and was admitted.
    harness_source_files: tuple[str, ...] = ()
    #: A ceiling on ONE launch, separate from the cumulative cap. The cumulative
    #: cap covers an effort that has already failed several times; without this,
    #: a single run could spend the whole of it. Named by the maintainer.
    per_launch_hard_usd: float | None = None
    #: Provenance only, never enforced.
    provenance_commit: str | None = None
    version: int = 1

    #: What this authorization may express. `DENY_ALL` grants nothing, so an
    #: authorization constructed without a policy can permit no action at all.
    action_policy: ActionPolicy = DENY_ALL

    def allows(self, action: str) -> bool:
        """Does this authorization grant `action`? Absence is denial."""
        return self.action_policy.allows(action)

    def refuse(self, action: str) -> None:
        """Refuse `action`, with the policy's own note if it has one."""
        self.action_policy.refuse(action)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "aadistill.autoinit.spend_authorization/v1",
            "authorization_id": self.authorization_id,
            "version": self.version,
            "granted_utc": self.granted_utc,
            "granted_by": self.granted_by,
            "plan_id": self.plan_id,
            "preflight_plan_hash": self.plan_hash,
            "expected_usd": self.expected_usd,
            "hard_cap_usd": self.hard_cap_usd,
            "authorized_stages": list(self.authorized_stages),
            "stage_conditions": dict(self.stage_conditions),
            "scope_note": self.scope_note,
            **{key: self.allows(action)
               for key, action in self.action_policy.wire_claims.items()},
            "authorized_session_commit": self.authorized_session_commit,
            "harness_source_digest": self.harness_source_digest,
            "harness_source_files": list(self.harness_source_files),
            "per_launch_hard_usd": self.per_launch_hard_usd,
            "provenance_commit": self.provenance_commit,
            "enforcement": (
                "the launcher loads this artifact and refuses to create a pod "
                "whose priced hard threshold exceeds hard_cap_usd, refuses a "
                "stage not in authorized_stages, and has no code path to Phase A"),
        }
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    # -- the checks the launcher and driver call -------------------------
    def require_plan(self, plan_hash: str) -> None:
        if plan_hash != self.plan_hash:
            raise AuthorizationError(
                f"this authorization is bound to preflight plan {self.plan_hash} "
                f"but the plan about to run hashes to {plan_hash}. An "
                "authorization does not transfer to a plan that changed.")

    def require_stage(self, stage: int) -> None:
        if stage not in self.authorized_stages:
            raise AuthorizationError(
                f"stage {stage} is not in the authorized set "
                f"{list(self.authorized_stages)}")

    def require_within_cap(self, projected_usd: float, *, what: str = "") -> None:
        if projected_usd > self.hard_cap_usd:
            raise AuthorizationError(
                f"{what or 'projected spend'} ${projected_usd:.2f} exceeds the "
                f"authorized hard cap ${self.hard_cap_usd:.2f}")

    def require_harness(self, repo_root: str | Path = ".") -> dict[str, Any]:
        """Refuse to run a harness this authorization was not granted against.

        A paid run that produces permanent artifacts must be executed by the code
        that was rehearsed. Whole-repository HEAD is the wrong identity here for
        the same reason it is wrong for the trainer — a docs commit would revoke a
        valid authorization — so this digests the declared harness set.
        """
        observed = harness_source_digest(repo_root, files=self.harness_source_files)
        if self.harness_source_digest is None:
            raise AuthorizationError(
                "this authorization declares no harness_source_digest, so it "
                "cannot authorize any executable. Re-issue it against the "
                f"rehearsed harness (observed {observed['digest']}).")
        if observed["digest"] != self.harness_source_digest:
            raise AuthorizationError(
                f"the harness on disk digests to {observed['digest']} but this "
                f"authorization was granted against {self.harness_source_digest}. "
                "The rehearsed harness and the executable harness differ; "
                "re-rehearse and re-issue rather than running an unrehearsed "
                "harness against a paid authorization.")
        return observed

    def require_within_launch_limit(self, hard_usd: float, *, what: str = "") -> None:
        """One launch may not spend the cumulative allowance of several."""
        if self.per_launch_hard_usd is None:
            return
        if hard_usd > self.per_launch_hard_usd:
            raise AuthorizationError(
                f"{what or 'planned hard threshold'} ${hard_usd:.4f} exceeds the "
                f"per-launch limit ${self.per_launch_hard_usd:.4f}. The "
                f"cumulative cap ${self.hard_cap_usd:.2f} covers an effort that "
                "has already failed several times; it is not one run's budget.")

    @classmethod
    def load(cls, path: str | Path, *,
             policy: ActionPolicy | None = None) -> "SpendAuthorization":
        """Read an artifact under a declared policy.

        `policy` is required, by class attribute or argument. A caller that has
        not said what the artifact may express cannot check whether it claims
        more, so reading it at all would be reading an unchecked permission.
        """
        resolved = policy or getattr(cls, "POLICY", None)
        if resolved is None:
            raise AuthorizationError(
                f"{cls.__name__}.load needs an ActionPolicy: without one there "
                "is nothing to check an artifact's claims against, and an "
                "unchecked claim is an unenforced permission")
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        resolved.check_claims(raw, where=str(path))
        return cls(
            authorization_id=raw["authorization_id"],
            granted_utc=raw["granted_utc"], granted_by=raw["granted_by"],
            plan_id=raw["plan_id"], plan_hash=raw["preflight_plan_hash"],
            expected_usd=float(raw["expected_usd"]),
            hard_cap_usd=float(raw["hard_cap_usd"]),
            authorized_stages=tuple(raw["authorized_stages"]),
            stage_conditions=dict(raw["stage_conditions"]),
            scope_note=raw["scope_note"],
            authorized_session_commit=raw.get("authorized_session_commit"),
            harness_source_digest=raw.get("harness_source_digest"),
            #: No fallback. An artifact that names no harness gets none, and
            #: `require_harness` then refuses -- which is right: inheriting
            #: another session's file list is how a digest comes to describe an
            #: executable nobody checked.
            harness_source_files=tuple(raw.get("harness_source_files") or ()),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


#: MICRO_PREFLIGHT_AUTHORIZATION moved to
#: `configs/experiments/micro_preflight/authorization.json`, loaded by
#: `scripts/experiments/micro_preflight.py`. A specific maintainer grant --
#: dollar amounts, a granted date, a plan hash -- is experiment instance data,
#: and the reusable core should describe what an authorization IS without
#: carrying one.


def authorization_from_dict(doc) -> "SpendAuthorization":
    """One authorization from a document holding this dataclass's own fields.

    Distinct from `SpendAuthorization.load`, which reads the ON-DISK GRANT
    schema and verifies its self-hash. This is the declaration form: the same
    values that used to be written out in this module, now supplied by the
    application. It grants nothing that the record it came from did not.
    """
    fields = dict(doc)
    for key in ("authorized_stages",):
        if fields.get(key) is not None:
            fields[key] = tuple(fields[key])
    for key in ("stage_conditions",):
        if fields.get(key) is not None:
            fields[key] = dict(fields[key])
    if fields.get("harness_source_files") is not None:
        fields["harness_source_files"] = tuple(fields["harness_source_files"])
    return SpendAuthorization(**fields)
