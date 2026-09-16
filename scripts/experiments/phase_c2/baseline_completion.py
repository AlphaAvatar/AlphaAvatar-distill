"""The governance layer for ONE Phase-C2 baseline-completion session.

Attempt 4's beam completed and committed a ranking of five measured candidates;
its conditional rebuild then hit a reserve that could not fund the work, so B
carries no measurement and the B->C comparison was never computed. This module
is the permission and identity surface for the session that supplies exactly
that missing half.

**It is a different TYPE, and the difference is the point.** A
`BaselineCompletionAuthorization` reports `authorizes_c2_search1 = False` and
declares its own schema, so:

* the Search-1 launcher cannot load one -- `C2Authorization.load` refuses a
  foreign schema, which is the check that already stops a Phase-A, Phase-B or C1
  artifact standing in for a C2 one;
* this session's launcher cannot load a Search-1 authorization, for the same
  reason in the other direction;
* and a completion grant therefore cannot be spent on a beam, by construction
  rather than by instruction.

What it permits is one rebuild of the frozen B, one `state_eval` measurement of
it, and one comparison against candidate measurements it reads and never
recomputes. It does not permit a beam, a new candidate, a remeasurement of an
existing one, Search-2, or behavioural confirmation.

Every number comes from the completion pricing record and every identity from
the completion protocol; nothing here is retyped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aadistill.governance.authorization import AuthorizationError
from aadistill.governance.closure import ClosureError, derive, digest_of
from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase
from aadistill.infrastructure.session import BudgetSpec
from aadistill.infrastructure.manifest import sha256_json

from experiments.phase_c2.session import C2Authorization, C2ResourceScope

REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA = "aadistill.autoinit.c2_baseline_completion_authorization/v1"

PLAN_ID = "autoinit.v1.phase_c2.baseline_completion"
SESSION_ID = "autoinit-phase-c2-baseline-completion"

PROTOCOL = "logs/stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_protocol.json"
PRICING = "logs/stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_pricing.json"
FROZEN_INPUTS = ("logs/stages/stage-1/phase_c2/runs/attempt4/evidence/"
                 "c2_frozen_comparison_inputs.json")
SELECTION_RECORD = ("logs/stages/stage-1/phase_c2/runs/attempt4/evidence/"
                    "stage1_selection.json")

#: What this session executes. The Search-1 driver is NOT here, and neither is
#: anything that imports the beam runner.
ENTRY_POINTS: tuple[str, ...] = (
    "scripts/pod/autoinit_phase_c2_baseline_launch.py",
    "scripts/pod/autoinit_phase_c2_baseline_driver.py",
    "scripts/pod/collect_artifacts.py",
)

#: Non-python inputs the session reads and whose bytes therefore belong in its
#: executable identity: the protocol it obeys, the prices it is bound to, the
#: candidate measurements it compares against, the ranking it cites, and the
#: shell that stages it.
DECLARED_INPUTS: tuple[str, ...] = (
    PROTOCOL, PRICING, FROZEN_INPUTS, SELECTION_RECORD,
    "configs/experiments/phase_c2/frozen_assets.json",
    "scripts/pod/autoinit_preflight_setup.sh",
)

SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/pod", "scripts/autoinit")


def protocol(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """The protocol, verified against its own hash."""
    doc = json.loads((Path(repo_root) / PROTOCOL).read_text())
    stated = doc.get("protocol_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "protocol_sha256"})
    if stated != recomputed:
        raise AuthorizationError(
            f"{PROTOCOL} does not match its own protocol_sha256; it has been "
            "edited since it was written")
    return doc


def plan_hash(repo_root: str | Path = REPO_ROOT) -> str:
    """The protocol IS the plan for this session."""
    return protocol(repo_root)["protocol_sha256"]


def pricing(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    doc = json.loads((Path(repo_root) / PRICING).read_text())
    stated = doc.get("pricing_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items() if k != "pricing_sha256"})
    if stated != recomputed:
        raise AuthorizationError(
            f"{PRICING} does not match its own pricing_sha256; it has been "
            "edited since it was priced")
    return doc


def hard_ceiling_usd(repo_root: str | Path = REPO_ROOT) -> float:
    return float(pricing(repo_root)["totals"]["hard_ceiling_usd"])


def price_per_hour_usd(repo_root: str | Path = REPO_ROOT) -> float:
    return float(pricing(repo_root)["totals"]["price_per_hour"])


def rebuild_minutes(repo_root: str | Path = REPO_ROOT) -> float:
    """The DEPTH-bounded allowance for the whole B path, from the pricing.

    Not the 27.665 min that failed: the sum of the bounding line items for the
    work the rebuild performs, which the pricing derives from Attempt 4's own
    measurements.
    """
    doc = pricing(repo_root)
    wanted = {"depth_full_derivation", "ffn", "residual_width", "attention",
              "materialization_reload_identity_x4"}
    minutes = sum(item["minutes"] for item in doc["line_items_bounding"]
                  if item["item"] in wanted)
    if not minutes:
        raise AuthorizationError(
            f"{PRICING} names none of {sorted(wanted)}; the rebuild allowance "
            "cannot be derived from it")
    return round(minutes, 3)


def _minutes(doc: dict[str, Any], item: str) -> float:
    for row in doc["line_items_bounding"]:
        if row["item"] == item:
            return float(row["minutes"])
    raise AuthorizationError(f"{PRICING} has no line item {item!r}")


def budget_spec(repo_root: str | Path = REPO_ROOT) -> BudgetSpec:
    """The `BudgetSpec` whose plan reproduces the completion pricing record.

    `arms=0`: this session trains nothing, so the step term multiplies out and
    the measured floor is passed only so the below-floor guard cannot fire on a
    figure that means nothing here. Every phase below is a line item of the
    pricing document, named all the way into the session record.
    """
    doc = pricing(repo_root)
    return BudgetSpec(
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: baseline completion trains nothing. It rebuilds "
                     "one checkpoint and measures it once."),
        setup_minutes=_minutes(doc, "setup_and_staging"),
        transfer_minutes=_minutes(doc, "evidence_collection_and_sync"),
        other_phases=(
            Phase("driver_startup_and_teacher_load",
                  _minutes(doc, "driver_startup_teacher_load_bookkeeping")),
            Phase("depth_full_derivation", _minutes(doc, "depth_full_derivation")),
            Phase("ffn", _minutes(doc, "ffn")),
            Phase("residual_width", _minutes(doc, "residual_width")),
            Phase("attention", _minutes(doc, "attention")),
            Phase("materialization_reload_identity",
                  _minutes(doc, "materialization_reload_identity_x4")),
            Phase("state_eval_of_B_once", _minutes(doc, "state_eval_of_B_once")),
            Phase("b_to_c_comparison", _minutes(doc, "b_to_c_comparison_post_processing")),
            Phase("teardown", _minutes(doc, "teardown_and_provider_confirmation")),
        ),
        contingency_fraction=float(doc["totals"]["contingency_fraction"]),
        #: No beam, so no beam-composition risk. No conditional rebuild reserve
        #: either: the rebuild is not conditional here, it is the whole session,
        #: and it is priced in the path above rather than held beside it.
        soft_stop_reserves=(),
        artifact_recovery_reserve_minutes=float(
            doc["totals"]["artifact_recovery_reserve_minutes"]),
    )


class BaselineCompletionAuthorization(C2Authorization):
    """Permits exactly one baseline completion. Never a beam.

    Structurally a `C2Authorization` -- same commit binding, same derived-harness
    rule, same hash-of-itself check -- and a different type with a different
    schema, so the two cannot stand in for one another in either direction.
    """

    @property
    def authorizes_c2_search1(self) -> bool:
        """NEVER. That is the whole reason this type exists.

        Search-1's beam is a completed, consumed measurement. An artifact that
        could authorize another one would let a completion grant be spent on a
        ten-hour search, which is both a budget expansion and a scientific
        change.
        """
        return False

    @property
    def authorizes_c2_baseline_completion(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        #: From the properties, never literals.
        payload["authorizes_c2_search1"] = self.authorizes_c2_search1
        payload["authorizes_c2_baseline_completion"] = (
            self.authorizes_c2_baseline_completion)
        payload["scope"] = (
            "ONE rebuild of the frozen Phase-C1 treatment baseline B through its "
            "complete deterministic fixed path, ONE state_eval measurement of the "
            "rebuilt B on the frozen suite, and ONE B->C comparison against the "
            "five candidate measurements Attempt 4 froze and this session reads "
            "without recomputing. NOT a beam search, NOT a new candidate, NOT a "
            "remeasurement of any candidate, NOT Search-2, NOT behavioural "
            "confirmation, and no recovery training.")
        payload["forbids"] = [
            "rerunning the Search-1 beam",
            "generating any new C candidate",
            "remeasuring or replacing any frozen C measurement",
            "Search-2",
            "behavioural confirmation",
            "recovery training of any kind",
        ]
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "BaselineCompletionAuthorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = {k: v for k, v in raw.items() if k != "authorization_sha256"}
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. A "
                "Search-1 authorization prices a ten-hour beam and permits one; "
                "it cannot authorize a baseline completion, and a baseline "
                "completion's cannot authorize a beam.")
        if raw.get("authorizes_c2_search1"):
            raise AuthorizationError(
                f"{path} claims authorizes_c2_search1. A baseline-completion "
                "artifact that could authorize a beam is not one.")
        for forbidden in ("allows_phase_a", "allows_recovery_training"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. This session rebuilds one "
                    "checkpoint and measures it once.")
        scope = raw.get("resource_scope")
        return cls(
            **{k: v for k, v in raw.items()
               if k in cls.__dataclass_fields__ and k != "resource_scope"},
            resource_scope=C2ResourceScope.from_dict(scope) if scope else None)


def current_executable(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """What a completion session would execute NOW, derived live from the tree."""
    try:
        return derive(Path(repo_root), "phase_c2_baseline_completion",
                      ENTRY_POINTS, DECLARED_INPUTS, roots=SOURCE_ROOTS)
    except ClosureError as exc:
        raise AuthorizationError(
            f"cannot derive the baseline-completion executable set: {exc}") from exc


def executable_digest(repo_root: str | Path = REPO_ROOT) -> str:
    live = current_executable(repo_root)
    return live["digest"] if isinstance(live, dict) else digest_of(live)
