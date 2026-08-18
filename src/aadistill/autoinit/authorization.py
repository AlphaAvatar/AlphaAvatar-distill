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

`AUTHORIZED_STAGES` is the whole of it. Phase A is not on the list, and
`allows_phase_a` is a hard `False` rather than a flag someone could set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_file, sha256_json


class AuthorizationError(RuntimeError):
    """An action exceeds or falls outside what was authorized."""


#: The executable harness this authorization is granted against. Same rule and
#: same failure mode as the trainer and scoring source sets: a missing declared
#: file raises rather than yielding a digest over a smaller harness.
HARNESS_SOURCE_FILES_V1: tuple[str, ...] = (
    "scripts/pod/autoinit_preflight_launch.py",
    "scripts/pod/autoinit_preflight_driver.py",
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/autoinit_engine_probe.py",
    "scripts/pod/watchdog.py",
    "scripts/pod/collect_artifacts.py",
    # The session machinery. Added 2026-08-18 with the composition refactor: the
    # flow that used to live in `autoinit_preflight_launch.py` now lives here, so
    # a harness digest that did not cover it would certify a launcher that is a
    # hundred lines of declaration while the code that creates pods, relays logs
    # and tears down went unmeasured.
    "src/aadistill/infrastructure/session.py",
    "src/aadistill/infrastructure/session_runner.py",
    "src/aadistill/infrastructure/session_prechecks.py",
    "src/aadistill/autoinit/authorization.py",
    "src/aadistill/autoinit/generation.py",
)
#: Bumped with the three session modules. A digest computed over set 1 and one
#: computed over set 2 are not comparable, and the version is what says so.
HARNESS_SOURCE_SET_VERSION = 2


def harness_source_digest(repo_root: str | Path = ".", *,
                          files: tuple[str, ...] | None = None) -> dict[str, Any]:
    root = Path(repo_root)
    declared = tuple(files) if files is not None else HARNESS_SOURCE_FILES_V1
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
    return {"digest": digest, "set_version": HARNESS_SOURCE_SET_VERSION,
            "files": entries}


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
    #: executable — the continuation runs its own launcher, driver and plan
    #: module — must declare them here, or `require_harness` would digest the
    #: preflight's files and happily admit an edited continuation driver. The
    #: default keeps every existing artifact byte-identical.
    harness_source_files: tuple[str, ...] = HARNESS_SOURCE_FILES_V1
    #: A ceiling on ONE launch, separate from the cumulative cap. The cumulative
    #: cap covers an effort that has already failed several times; without this,
    #: a single run could spend the whole of it. Named by the maintainer.
    per_launch_hard_usd: float | None = None
    #: Provenance only, never enforced.
    provenance_commit: str | None = None
    version: int = 1

    #: Not a field. Phase A is separately unauthorized and this artifact cannot
    #: express permission for it.
    @property
    def allows_phase_a(self) -> bool:
        return False

    @property
    def automatic_phase_a_start(self) -> bool:
        return False

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
            "phase_a_authorized": self.allows_phase_a,
            "automatic_phase_a_start": self.automatic_phase_a_start,
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

    def refuse_phase_a(self) -> None:
        raise AuthorizationError(
            "Phase A is separately unauthorized and is not reachable from the "
            "preflight. Stop, report, and obtain a new authorization.")

    @classmethod
    def load(cls, path: str | Path) -> "SpendAuthorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        if raw.get("phase_a_authorized") or raw.get("automatic_phase_a_start"):
            raise AuthorizationError(
                "this artifact claims Phase A authorization, which it cannot "
                "grant; refusing to load it")
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
            harness_source_files=tuple(raw.get("harness_source_files")
                                       or HARNESS_SOURCE_FILES_V1),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


MICRO_PREFLIGHT_AUTHORIZATION = SpendAuthorization(
    authorization_id="autoinit.micro_preflight.2026-08-13",
    granted_utc="2026-08-13T00:00:00Z",
    granted_by="maintainer (session authorization)",
    plan_id="autoinit.micro_preflight",
    # Moved 2026-08-14 with the recovery_search_v2 migration: the plan's
    # Stage-3 description names the battery, so the battery change moves the
    # plan hash. The stages, their order and their stop conditions are
    # unchanged.
    plan_hash="afd08be777e1f84e29350ff6c65daaf9b8f72d8f7ea593ca92f0129572264295",
    expected_usd=4.20,
    hard_cap_usd=8.60,
    authorized_stages=(0, 1, 2, 3),
    stage_conditions={
        "0": "runtime attestation; protocol and generation-protocol materialization",
        "1": "cheap machine gates; any blocking failure stops the session",
        "2": "permanent canonical sa/sb controls, ONLY if stages 0 and 1 passed",
        "3": "control characterization; materializes the frozen thresholds",
        "teardown": "delete the pod, verify from the provider that it is gone, STOP",
    },
    # Filled by scripts/autoinit/issue_authorization.py at issue time, against
    # the rehearsed harness.
    authorized_session_commit=None,
    harness_source_digest=None,
    scope_note=(
        "micro-preflight only. If Stage 0/1 indicates that hardware, runtime, "
        "storage strategy, trainer semantics, evaluation semantics or any frozen "
        "identity must change, stop before permanent controls and return for "
        "review. Phase A remains separately unauthorized and must not start "
        "automatically."),
)
