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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json


class AuthorizationError(RuntimeError):
    """An action exceeds or falls outside what was authorized."""


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
    consuming_commit: str | None = None
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
            "consuming_commit": self.consuming_commit,
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
            consuming_commit=raw.get("consuming_commit"),
            version=int(raw.get("version", 1)))


MICRO_PREFLIGHT_AUTHORIZATION = SpendAuthorization(
    authorization_id="autoinit.micro_preflight.2026-08-13",
    granted_utc="2026-08-13T00:00:00Z",
    granted_by="maintainer (session authorization)",
    plan_id="autoinit.micro_preflight",
    plan_hash="37dbd7b22e3e884eff9d55f95c5ce25a212f823d2f396691c30d47930076f8ab",
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
    scope_note=(
        "micro-preflight only. If Stage 0/1 indicates that hardware, runtime, "
        "storage strategy, trainer semantics, evaluation semantics or any frozen "
        "identity must change, stop before permanent controls and return for "
        "review. Phase A remains separately unauthorized and must not start "
        "automatically."),
)
