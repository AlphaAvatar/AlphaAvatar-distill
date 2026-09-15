"""The Phase-C2 Search-1 session: its plan, its authorization type, its budget.

Three properties make this a *type* rather than a policy comment.

**It cannot authorize anything else.** `allows_phase_a` is hard `False` and
`load` refuses any artifact whose schema is not this one. A Phase-A, Phase-B,
continuation or C1 grant carries a ceiling derived for different work over a
different harness; accepting one here would certify code this session does not
run and price it for work it does not do.

**It authorizes a search and nothing after it.** `authorizes_c2_search1` is the
only thing it says yes to. There is no recovery training, no probe, no battery
and no confirmation anywhere in this session, and `allows_recovery_training` is
`False` so a reader does not have to infer that from an absence.

**Its ceiling is derived from the pricing record, not typed in.**
`c2_budget_spec()` reads `logs/stages/stage-1/phase_c2/plans/phase_c2_pricing.json`
— which `load_pricing` refuses if it does not match its own `pricing_sha256` —
and builds the `BudgetSpec` from it, so there is exactly one place the
enforceable ceiling comes from.

The two reserves reach `plan_session` as named `Phase` entries in
`soft_stop_reserves`, which places them AFTER the contingency multiplier and
BEFORE the soft stop. That placement is the point: a reserve added as a phase
would be inflated by contingency, and one added after the soft stop would not
protect the work at all, because `afford()` refuses to start anything that would
cross the soft stop.

**The step time is a formality here and says so.** A search session trains
nothing, so `arms=0` and the step term multiplies to zero. The measured floor is
passed so the below-floor guard cannot be satisfied by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aadistill.governance.authorization import AuthorizationError
from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase
from aadistill.infrastructure.manifest import sha256_json
from aadistill.infrastructure.session import BudgetSpec
from experiments.phase_a.plan import PhaseAAuthorization

SCHEMA = "aadistill.autoinit.c2_authorization/v1"

PRICING_PATH = "logs/stages/stage-1/phase_c2/plans/phase_c2_pricing.json"
PLAN_PATH = "logs/stages/stage-1/phase_c2/plans/phase_c2_search1_plan.md"

#: Every file whose bytes decide what the paid C2 session executes. The
#: authorization measures THIS set; a grant that declares a different one is
#: certifying different code.
#:
#: The two artifact specs are in here for the reason the C1 set names its own:
#: `collect_artifacts.py` is a spec interpreter, and what actually decides which
#: evidence survives teardown is the declared pattern list. A session whose
#: evidence contract can be edited without moving the harness digest has an
#: unmeasured mutable input at exactly the point where loss is irreversible.
C2_HARNESS_SOURCE_FILES_V1: tuple[str, ...] = (
    # the session, end to end
    "scripts/pod/autoinit_phase_c2_launch.py",
    "scripts/pod/autoinit_phase_c2_driver.py",
    # the setup script `SessionRunner._launch` uploads and runs. `SetupManifest`
    # carries no setup-script field, so no session can substitute another.
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/start_job.py",
    "scripts/pod/watchdog.py",
    "scripts/pod/collect_artifacts.py",
    # the experiment layer: the space, the baseline, this file
    "scripts/experiments/phase_c2/__init__.py",
    "scripts/experiments/phase_c2/search_space.py",
    "scripts/experiments/phase_c2/baseline.py",
    #: The B->C comparison. It decides what the run's baseline evidence IS, and
    #: it is the only place a REBUILT baseline's state_eval result becomes
    #: durable — so its bytes decide whether the session can answer its own
    #: question after teardown. Configuration by file type, executable by
    #: consequence, exactly like the artifact specs below.
    "scripts/experiments/phase_c2/comparison.py",
    "scripts/experiments/phase_c2/session.py",
    # the search seam the driver calls, and the frozen identities it resolves
    "scripts/autoinit/phase_a_search.py",
    "scripts/autoinit/phase_a_frozen.py",
    "scripts/autoinit/load_state_eval.py",
    # the frozen C1 constructor the baseline rebuild imports rather than
    # re-deriving. Its bytes decide whether the rebuilt B is B.
    "scripts/experiments/phase_c1/session.py",
    "scripts/experiments/phase_c1/isolation.py",
    # the evidence contract
    "configs/autoinit/c2_artifacts.json",
    "configs/autoinit/c2_artifacts_failed.json",
)

C2_SOURCE_SET_VERSION = 1


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C2Stage:
    stage_id: str
    what: str
    blocking: bool = True


#: Two stages. A search session that trains nothing has no reason to have six,
#: and a stage that exists only to mirror another experiment's numbering is a
#: stage a reader has to look up.
C2_STAGES: tuple[C2Stage, ...] = (
    C2Stage("bind_identities",
            "image, executable digest, teacher revision, both calibration "
            "mixtures by spec AND content hash, the configured space, and the "
            "frozen baseline construction — all before anything expensive"),
    C2Stage("search_and_baseline",
            "the beam search over the Search-1 space, the durability boundary "
            "that commits the ranking the instant it exists, and the baseline "
            "B resolved exactly once: searched if the beam re-derived it, "
            "rebuilt through the frozen C1 fixed path if not"),
)


@dataclass(frozen=True)
class C2SessionContract:
    """The session's declared shape, hashable for a grant to bind to."""

    session_id: str = "autoinit.v1.phase_c2.search1"
    stages: tuple[C2Stage, ...] = C2_STAGES
    trains_anything: bool = False
    n_probes: int = 0
    battery_asset_id: str | None = None
    #: What the search-stage metric is, and what it is not.
    ranking_metric: str = "state_eval@v1 via PARETO_V1"
    metric_is_not: str = (
        "a demonstrated recovery improvement. The ranking is a hypothesis "
        "generator; a search winner is not a behavioural result and must never "
        "be reported as one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aadistill.autoinit.c2_session_contract/v1",
            "session_id": self.session_id,
            "stages": [{"stage_id": s.stage_id, "what": s.what,
                        "blocking": s.blocking} for s in self.stages],
            "trains_anything": self.trains_anything,
            "n_probes": self.n_probes,
            "battery_asset_id": self.battery_asset_id,
            "ranking_metric": self.ranking_metric,
            "metric_is_not": self.metric_is_not,
        }

    @property
    def contract_hash(self) -> str:
        return sha256_json(self.as_dict())


C2_SESSION_CONTRACT = C2SessionContract()

C2_PLAN_ID = "autoinit.v1.phase_c2.search1"


def c2_plan_hash() -> str:
    """The session plan a grant binds to.

    Derived from the contract and the configured space together, so a grant
    cannot survive a change to either. Restating the space here would be a
    second declaration of the thing whose fixity is the point.
    """
    from experiments.phase_c2.search_space import (
        C2_ALLOWED_IMPLS, C2_IMPL_PROFILES, C2_PROFILE_IDS,
    )
    from aadistill.initialization.planning.ranking import PARETO_V1, SCHEDULE_V1

    return sha256_json({
        "plan_id": C2_PLAN_ID,
        "contract": C2_SESSION_CONTRACT.as_dict(),
        "allowed_impls": sorted(C2_ALLOWED_IMPLS),
        "impl_profiles": {k: sorted(v) for k, v
                          in sorted(C2_IMPL_PROFILES.items())},
        "profiles": sorted(C2_PROFILE_IDS),
        "schedule": SCHEDULE_V1.as_dict(),
        "policy": PARETO_V1.qualified_id,
        "policy_hash": PARETO_V1.policy_hash,
        "order": "free; no kind is pinned to a position",
    })


# ---------------------------------------------------------------------------
# the authorization type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C2Authorization(PhaseAAuthorization):
    """Permits exactly the Phase-C2 Search-1 initialization search.

    Structurally a `PhaseAAuthorization` — same commit binding, same harness
    rule, same hash-of-itself check — and a different **type**, so neither can
    stand in for the other.
    """

    harness_source_files: tuple[str, ...] = C2_HARNESS_SOURCE_FILES_V1

    @property
    def allows_phase_a(self) -> bool:
        """Never. Phase A is a different, completed experiment."""
        return False

    @property
    def allows_recovery_training(self) -> bool:
        """Never. This session trains nothing and has no battery."""
        return False

    @property
    def authorizes_c2_search1(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        # From the properties, never a literal: a document that disagreed with
        # the object that wrote it would be worse than no document.
        payload["allows_phase_a"] = self.allows_phase_a
        payload["allows_recovery_training"] = self.allows_recovery_training
        payload["authorizes_c2_search1"] = self.authorizes_c2_search1
        payload["scope"] = (
            "ONE beam search over the Phase-C2 Search-1 space: four operator "
            "kinds, one implementation each, order free, ATTENTION branching "
            "over two calibration mixtures. Plus, conditionally, ONE rebuild of "
            "the frozen C1 baseline B when the beam does not re-derive it. No "
            "recovery training, no probes, no battery, no confirmation, and no "
            "Search-2.")
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "C2Authorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = {k: v for k, v in raw.items() if k != "authorization_sha256"}
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. "
                "A Phase-A, Phase-B, continuation or C1 grant measures a "
                "different harness and carries a ceiling derived for different "
                "work; it cannot authorize the C2 search.")
        for forbidden in ("allows_phase_a", "allows_recovery_training"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. C2 Search-1 runs a search and "
                    "trains nothing; an artifact claiming otherwise is not a C2 "
                    "authorization.")
        # Mapped explicitly, not by field name: `as_dict` serialises
        # `plan_hash` as `phase_a_session_plan_hash` and `science_plan_hash` as
        # `phase_a_science_plan_hash`, so a by-name filter drops both and the
        # constructor then fails on a required argument. Every sibling loader
        # maps them by hand for the same reason.
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
            harness_source_files=tuple(raw.get("harness_source_files")
                                       or C2_HARNESS_SOURCE_FILES_V1),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


def c2_harness_digest(repo_root: str | Path = ".") -> dict[str, Any]:
    from aadistill.governance.authorization import harness_source_digest

    return harness_source_digest(repo_root, files=C2_HARNESS_SOURCE_FILES_V1,
                                 set_version=C2_SOURCE_SET_VERSION)


# ---------------------------------------------------------------------------
# the budget, derived from the pricing record
# ---------------------------------------------------------------------------

def load_pricing(repo_root: str | Path = ".") -> dict[str, Any]:
    p = Path(repo_root) / PRICING_PATH
    if not p.is_file():
        raise AuthorizationError(
            f"{PRICING_PATH} is missing; the C2 budget is derived from the "
            "pricing record and must not be typed in a second time")
    doc = json.loads(p.read_text())
    stated = doc.get("pricing_sha256")
    check = {k: v for k, v in doc.items() if k != "pricing_sha256"}
    if stated != sha256_json(check):
        raise AuthorizationError(
            f"{PRICING_PATH} does not match its own pricing_sha256; it has been "
            "edited since it was generated")
    return doc


def _minutes(doc: dict[str, Any], substring: str) -> float:
    for item in doc["line_items"]:
        if substring in item["item"]:
            return float(item["minutes"])
    raise AuthorizationError(
        f"pricing record has no line item matching {substring!r}")


def _reserve(doc: dict[str, Any], name: str) -> float:
    for entry in doc["reserves"]:
        if entry["reserve"] == name:
            return float(entry["minutes"])
    raise AuthorizationError(f"pricing record declares no reserve {name!r}")


def c2_budget_spec(repo_root: str | Path = ".") -> BudgetSpec:
    """The `BudgetSpec` whose plan reproduces the pricing record.

    Every number is read out of that record. The two reserves stay named all
    the way into the session record, so a reader can tell what each one was for
    after the fact — which is the difference between "the session cost $12" and
    "the session rebuilt the baseline".
    """
    doc = load_pricing(repo_root)
    return BudgetSpec(
        #: A search trains nothing. The step term multiplies to zero and the
        #: measured floor is passed so the below-floor guard cannot fire by
        #: accident on a figure that means nothing here.
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: Search-1 trains nothing, so arms=0 and the step "
                     "term is zero. The measured E6b floor is passed so the "
                     "below-floor guard cannot be satisfied by accident"),
        setup_minutes=_minutes(doc, "session setup"),
        transfer_minutes=_minutes(doc, "bundle transfer"),
        other_phases=(
            Phase("teacher_fetch_verify", _minutes(doc, "teacher fetch")),
            Phase("machine_gates", _minutes(doc, "machine gates")),
            Phase("beam_search_depth_early", _minutes(doc, "beam search")),
            Phase("selection_and_manifest", _minutes(doc, "selection commit")),
            Phase("artifact_synchronization",
                  _minutes(doc, "artifact synchronization")),
        ),
        contingency_fraction=float(doc["totals"]["contingency_fraction"]),
        soft_stop_reserves=(
            Phase("beam_composition_risk", _reserve(doc, "beam_composition_risk")),
            Phase("baseline_rebuild_reserve",
                  _reserve(doc, "baseline_rebuild_reserve")),
        ),
        artifact_recovery_reserve_minutes=float(
            doc["totals"]["artifact_recovery_reserve_minutes"]),
    )


def c2_hard_ceiling_usd(repo_root: str | Path = ".") -> float:
    """The one place the enforceable ceiling comes from."""
    return float(load_pricing(repo_root)["totals"]["hard_ceiling_usd"])


def c2_price_per_hour_usd(repo_root: str | Path = ".") -> float:
    """The planned hourly rate, from the same hash-verified record.

    Planning evidence with a date on it. A launch re-queries `securePrice`
    before any provider resource exists and refuses a quote above
    `--max-price`, so a stale value here can only ever refuse a launch.
    """
    return float(load_pricing(repo_root)["hardware"]["price_per_hour_usd"])
