"""Authorization for the Phase-A recovery continuation — its own harness.

The continuation has its own launcher, driver, budget and authorization path,
and until this module existed the only issuer bound
`PHASE_A_HARNESS_SOURCE_FILES_V1`. That set measures the full Phase-A
launcher/driver/search path and does **not** contain either continuation file.
Re-using the Phase-A issuer would therefore have produced an authorization whose
harness digest did not measure the executable the paid run actually executes —
the digest would have been green while the thing it certified was untested by it.

**Full Phase A and the recovery continuation are distinct operational
harnesses**, and are measured independently. Broadening the Phase-A set to cover
this session would make each one's identity move when the other changed, which is
the opposite of what a harness digest is for.

Refused by **schema**, not by convention, exactly as `PhaseAAuthorization` is
refused where a `SpendAuthorization` is expected: a full-Phase-A artifact cannot
be pressed into service here, and this artifact cannot authorize a search. That
matters because the two carry different ceilings — $23.0484 funds a beam search
this session does not run, and $16.7456 is what its own derivation prices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json
from .authorization import AuthorizationError
from .phase_a import (
    PHASE_A_HARNESS_SOURCE_FILES_V1, PHASE_A_PLAN_V1, PhaseAAuthorization,
    phase_a_harness_digest,
)

SCHEMA = "aadistill.autoinit.recovery_continuation_authorization/v1"

#: The one Phase-A harness file this session cannot reach.
#:
#: `autoinit_phase_a_driver.stage1` imports `run_phase_a_search` *inside the
#: method*, and the continuation overrides that method and never delegates to it,
#: so the module is never imported on this path. Digesting it anyway would make
#: an edit to code this session cannot execute revoke a valid authorization —
#: the same false coupling that pinning whole-repository HEAD has, one level
#: smaller.
SEARCH_ONLY_HARNESS_FILES: tuple[str, ...] = (
    "scripts/autoinit/phase_a_search.py",
)

#: What this session runs that a full Phase-A session does not.
CONTINUATION_ONLY_HARNESS_FILES: tuple[str, ...] = (
    # The session declaration and the driver it names.
    "scripts/pod/autoinit_recovery_continuation_launch.py",
    "scripts/pod/autoinit_recovery_continuation_driver.py",
    # The frozen identities, in the module that holds no search code. This is
    # what that extraction exists for: the continuation binds the same teacher,
    # geometry, canonical init and seed without importing the beam.
    "scripts/autoinit/phase_a_frozen.py",
    # What replaces the search, and what hands the card to recovery.
    "src/aadistill/autoinit/stage1_import.py",
    "src/aadistill/autoinit/device_handoff.py",
    # Executed on both sides of the transfer: the launcher's $0 precheck
    # re-identifies the five leaves with it, and the driver reads them with it.
    "src/aadistill/autoinit/leaf_durability.py",
    # This module: the schema, the refusals, and this list.
    "src/aadistill/autoinit/recovery_continuation.py",
)

#: THE EXECUTABLE CONTINUATION CLOSURE — what this session actually runs.
#:
#: DERIVED from the Phase-A set rather than transcribed from it. The two sessions
#: share a launcher, a driver, the pod setup and the session machinery: the
#: continuation launcher imports `continuation_budget`, the durability callables
#: and the shared parser, and the continuation driver *subclasses* `PhaseADriver`
#: for Stages 2-5. Those are imported, not shelled out to, and are therefore as
#: much "the executable" here as they are there.
#:
#: A second hand-maintained copy of fourteen shared paths would be a second thing
#: to keep in step, and the failure mode is silent: whichever list was forgotten
#: would certify a smaller harness than the one that runs. Deriving means adding
#: a shared module to the Phase-A set propagates here, which is correct — this
#: session executes it too.
#:
#: Deriving is not broadening. `PHASE_A_HARNESS_SOURCE_FILES_V1` is unchanged and
#: does not mention this session; the two digests are different numbers over
#: different sets, and each moves only when its own executable moves.
RECOVERY_CONTINUATION_HARNESS_FILES_V1: tuple[str, ...] = tuple(sorted(
    (set(PHASE_A_HARNESS_SOURCE_FILES_V1) - set(SEARCH_ONLY_HARNESS_FILES))
    | set(CONTINUATION_ONLY_HARNESS_FILES)))


def recovery_continuation_harness_digest(repo_root: str | Path = ".") -> dict[str, Any]:
    """Digest the continuation's own executable closure.

    Uses the same function `require_harness()` calls, so the issued digest and
    the verified digest cannot diverge by one of them being reimplemented.
    """
    return phase_a_harness_digest(
        repo_root, files=RECOVERY_CONTINUATION_HARNESS_FILES_V1)


@dataclass(frozen=True)
class RecoveryContinuationAuthorization(PhaseAAuthorization):
    """Permits importing a verified Stage-1 result and running Stages 2-5.

    A `PhaseAAuthorization` in structure — same frozen plan identities, same
    commit binding, same harness rule — and a different **type**, so neither can
    stand in for the other. `allows_beam_search` is a hard `False` here: this
    session cannot start a search whatever artifact it is pointed at, and that
    is a property rather than a promise in a comment.
    """

    harness_source_files: tuple[str, ...] = RECOVERY_CONTINUATION_HARNESS_FILES_V1

    @property
    def allows_beam_search(self) -> bool:
        """Never. The continuation imports Stage 1; it does not produce one."""
        return False

    @property
    def authorizes_recovery_continuation(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        # From the properties, never a literal: a subclass that changed one and
        # an artifact that still said `false` would be a document disagreeing
        # with the object that wrote it.
        payload["allows_beam_search"] = self.allows_beam_search
        payload["authorizes_recovery_continuation"] = (
            self.authorizes_recovery_continuation)
        payload["stage1_source"] = (
            "imported from the verified Phase-A attempt-12 result; this session "
            "runs no search and cannot reach one")
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "RecoveryContinuationAuthorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        # Refused by SCHEMA, not by convention. A full-Phase-A artifact carries
        # a $23.0484 ceiling that funds a beam search this session does not run;
        # accepting it here would authorize the continuation under a price
        # derived for different work.
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. "
                "A full Phase-A authorization measures the search harness and "
                "carries the search's ceiling; it cannot authorize the recovery "
                "continuation, which runs a different executable at a different "
                "price.")
        if raw.get("allows_beam_search"):
            raise AuthorizationError(
                f"{path} claims to allow a beam search. The continuation "
                "imports Stage 1 and cannot reach a search; an artifact that "
                "says otherwise describes a session that does not exist.")
        if not raw.get("authorizes_recovery_continuation"):
            raise AuthorizationError(
                f"{path} carries the continuation schema but does not assert "
                "authorizes_recovery_continuation; refusing to infer permission "
                "from a schema name alone")
        if raw.get("automatic_followon_start"):
            raise AuthorizationError(
                "this artifact claims an automatic follow-on start, which it "
                "cannot grant; the continuation stops for review")
        return cls(
            authorization_id=raw["authorization_id"],
            granted_utc=raw["granted_utc"], granted_by=raw["granted_by"],
            plan_id=raw["plan_id"],
            plan_hash=raw["phase_a_session_plan_hash"],
            science_plan_hash=raw["phase_a_science_plan_hash"],
            expected_usd=float(raw["expected_usd"]),
            hard_cap_usd=float(raw["hard_cap_usd"]),
            authorized_stages=tuple(raw["authorized_stages"]),
            stage_conditions=dict(raw["stage_conditions"]),
            scope_note=raw["scope_note"],
            authorized_session_commit=raw.get("authorized_session_commit"),
            harness_source_digest=raw.get("harness_source_digest"),
            harness_source_files=tuple(raw.get(
                "harness_source_files", RECOVERY_CONTINUATION_HARNESS_FILES_V1)),
            per_launch_hard_usd=(float(raw["per_launch_hard_usd"])
                                 if raw.get("per_launch_hard_usd") is not None
                                 else None),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


#: Written into `granted_by` by the template below and refused by the issuer, for
#: the same reason the Phase-A one is: a grant is a one-use maintainer decision
#: about a particular attempt at a particular cumulative spend, and executable
#: source is where such a decision goes stale silently while still reading as
#: though it applies.
CONTINUATION_GRANT_PROSE_REQUIRED = (
    "NO GRANT. This is the recovery-continuation authorization SCHEMA, not a "
    "grant. scripts/autoinit/issue_recovery_continuation_authorization.py "
    "requires --grant naming a one-use continuation grant document, and refuses "
    "to issue with this value in place.")

#: The schema, with the pricing left as placeholders the issuer fills from
#: `continuation_budget()`. No dollar figure is written here: the continuation's
#: price is derived, and a constant would drift from the derivation.
RECOVERY_CONTINUATION_AUTHORIZATION = RecoveryContinuationAuthorization(
    authorization_id="autoinit.recovery_continuation.PLACEHOLDER",
    granted_utc="PLACEHOLDER",
    granted_by=CONTINUATION_GRANT_PROSE_REQUIRED,
    plan_id=PHASE_A_PLAN_V1.plan_id,
    #: The SAME frozen session plan. A different operational identity, not a
    #: different science.
    plan_hash=PHASE_A_PLAN_V1.plan_hash,
    science_plan_hash="PLACEHOLDER",
    expected_usd=0.0, hard_cap_usd=0.0,
    #: Stage 1 is IMPORTED, not searched, and appears here because the driver
    #: still reports it as a stage. Stages 2-5 are the work.
    authorized_stages=(0, 1, 2, 3, 4, 5),
    stage_conditions={
        "1": ("import the verified attempt-12 Stage-1 result from staged bytes, "
              "measure the canonical control once on the frozen suite, and admit "
              "the complete set. NO SEARCH."),
        "2": "rung-1 recovery probes on the imported leaves",
        "5": "selection and report; Phase A remains a terminus"},
    scope_note=(
        "ONE recovery continuation: import the verified Phase-A attempt-12 "
        "Stage-1 result and run recovery Stages 2-5. It does NOT authorize a "
        "beam search, a fresh Stage 1, a re-ranking or re-selection of the five "
        "leaves, retraining the permanent controls, or any follow-on session. "
        "The search is structurally unreachable from this harness."),
    per_launch_hard_usd=None,
)
