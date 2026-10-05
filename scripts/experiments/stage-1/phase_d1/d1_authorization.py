"""D1's ONE-USE authorization: four money conditions, all four checked.

A grant says what a maintainer decided. Everything else — the ceiling, the stages,
the design hash, the session contract, the identities — is DERIVED from the tree the
authorization will bind, and refused if the grant asserts it. A grant that can state
its own identity can over-authorize itself.

**FOUR CONDITIONS, not three.** The 2026-10-01 note records that the package total
binds too and looked implied only because it equalled formal + engineering. All four
are checked here, each against the live derived position:

    derived session ceiling  <=  per-session envelope
    cumulative + ceiling     <=  project cumulative cap
    ceiling                  <=  remaining FORMAL allowance
    ceiling                  <=  remaining PACKAGE total

And one that is not about money: D1's formal sessions must be funded by this
package at all. `funds_formal_sessions_of` is what makes a session's spend reach the
formal book instead of only the project cumulative, and a session outside that list
would spend from a book that reports it as never having happened.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[4]

BUDGET_TERMS = "configs/experiments/phase_c1/authorization.json"
DESIGN_REL = "logs/stages/stage-1/phase_d1/plans/d1_design.json"

#: Stages a D1 SEARCH session is authorized to execute. The search commits a
#: candidate set and stops: the behavioural rungs are separately authorized
#: sessions, because neither can be bound until the set it produced exists.
AUTHORIZED_STAGES = ("A", "B", "C", "D")
STAGE_CONDITIONS = {
    "A": ("setup and preflight: the frozen assets stage, the registries fill, "
          "and the session contract is asserted BEFORE any expansion"),
    "B": ("the root teacher materializes and the evaluator is primed with it; "
          "one teacher object, primed as the reference and handed to the search"),
    "C": ("the beam search runs the frozen space at width 6 with one warmup "
          "level; every expansion declares the D-series partition"),
    "D": ("commit_top_k and package. NO recovery, NO behavioural evaluation and "
          "NO promotion decision is computed in this session"),
}

#: A grant may not state what this module derives.
GRANT_MAY_NOT_STATE = (
    "authorization_id", "plan_id", "plan_hash", "science_plan_hash",
    "authorized_stages", "stage_conditions", "expected_usd",
    "per_launch_hard_usd", "design_hash", "measurement_protocol_id",
    "config_hash",
)


class D1AuthorizationRefused(RuntimeError):
    """A condition does not hold. A refusal is the product, not a warning."""


@dataclass(frozen=True)
class D1Authorization:
    """The issued artifact. `load` is the only way back in."""

    authorization_id: str
    granted_utc: str
    granted_by: str
    plan_id: str
    plan_hash: str
    science_plan_hash: str
    expected_usd: float
    hard_cap_usd: float
    per_launch_hard_usd: float
    authorized_stages: tuple[str, ...]
    stage_conditions: Mapping[str, str]
    session_commit: str
    design_hash: str
    arm: str
    measurement_protocol_id: str
    config_hash: str
    suite_content_sha256: str
    money: Mapping[str, Any] = field(default_factory=dict)
    one_use: str = ""
    authorizes: str = ""
    #: TYPED PERMISSIONS, so a D1 artifact cannot be mistaken for another
    #: experiment's and vice versa. The pod's dispatch branch asserts these before
    #: the driver starts: a missing branch falls through to the generic loader,
    #: which is a $0.2300 KeyError one step after the test gate passed.
    #:
    #: D1 is the first of these that DOES run a beam search, which is exactly why
    #: the flag is stated rather than assumed false by family resemblance.
    authorizes_d1_search: bool = True
    allows_beam_search: bool = True
    allows_recovery: bool = False
    allows_behavioural: bool = False
    automatic_followon_start: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aadistill.phase_d1.authorization/v1",
            "authorization_id": self.authorization_id,
            "granted_utc": self.granted_utc,
            "granted_by": self.granted_by,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "science_plan_hash": self.science_plan_hash,
            "expected_usd": round(float(self.expected_usd), 4),
            "hard_cap_usd": round(float(self.hard_cap_usd), 4),
            "per_launch_hard_usd": round(float(self.per_launch_hard_usd), 4),
            "authorized_stages": list(self.authorized_stages),
            "stage_conditions": dict(self.stage_conditions),
            "session_commit": self.session_commit,
            "design_hash": self.design_hash,
            "arm": self.arm,
            "measurement_protocol_id": self.measurement_protocol_id,
            "config_hash": self.config_hash,
            "suite_content_sha256": self.suite_content_sha256,
            "money": dict(self.money),
            "one_use": self.one_use,
            "authorizes": self.authorizes,
            "authorizes_d1_search": bool(self.authorizes_d1_search),
            "allows_beam_search": bool(self.allows_beam_search),
            "allows_recovery": bool(self.allows_recovery),
            "allows_behavioural": bool(self.allows_behavioural),
            "automatic_followon_start": bool(self.automatic_followon_start),
        }

    def require_plan(self, plan_hash: str) -> None:
        """Refuse an artifact that does not bind THIS session's plan.

        Called by the pod's dispatch branch before the driver starts. A plan hash
        that disagrees means the authorization was issued against a different
        design revision than the one the pod checked out.
        """
        if not plan_hash:
            raise D1AuthorizationRefused(
                "no plan hash was supplied to require_plan; an unbound check is "
                "not a check")
        if self.plan_hash != plan_hash:
            raise D1AuthorizationRefused(
                f"this authorization binds plan {self.plan_hash[:16]} and the "
                f"session declares {plan_hash[:16]}. The artifact was issued "
                "against a different design revision than this pod runs.")

    @classmethod
    def load(cls, path: str | Path) -> "D1Authorization":
        """Read an issued authorization back. The issuer round-trips through it.

        An artifact the issuer can write and the loader cannot read is a defect
        that surfaces on the pod, after the money is committed.
        """
        doc = json.loads(Path(path).read_text())
        if doc.get("schema") != "aadistill.phase_d1.authorization/v1":
            raise D1AuthorizationRefused(
                f"{path} is not a D1 authorization: schema {doc.get('schema')!r}")
        missing = [f for f in (
            "authorization_id", "granted_utc", "hard_cap_usd", "session_commit",
            "design_hash", "arm", "measurement_protocol_id", "config_hash",
            "suite_content_sha256", "authorized_stages") if not doc.get(f)]
        if missing:
            raise D1AuthorizationRefused(f"{path} omits {missing}")
        return cls(
            authorization_id=doc["authorization_id"],
            granted_utc=doc["granted_utc"],
            granted_by=doc.get("granted_by", ""),
            plan_id=doc["plan_id"], plan_hash=doc["plan_hash"],
            science_plan_hash=doc["science_plan_hash"],
            expected_usd=float(doc["expected_usd"]),
            hard_cap_usd=float(doc["hard_cap_usd"]),
            per_launch_hard_usd=float(doc["per_launch_hard_usd"]),
            authorized_stages=tuple(doc["authorized_stages"]),
            stage_conditions=dict(doc["stage_conditions"]),
            session_commit=doc["session_commit"],
            design_hash=doc["design_hash"], arm=doc["arm"],
            measurement_protocol_id=doc["measurement_protocol_id"],
            config_hash=doc["config_hash"],
            suite_content_sha256=doc["suite_content_sha256"],
            money=dict(doc.get("money") or {}),
            one_use=doc.get("one_use", ""), authorizes=doc.get("authorizes", ""),
            authorizes_d1_search=bool(doc.get("authorizes_d1_search", False)),
            allows_beam_search=bool(doc.get("allows_beam_search", False)),
            allows_recovery=bool(doc.get("allows_recovery", False)),
            allows_behavioural=bool(doc.get("allows_behavioural", False)),
            automatic_followon_start=bool(
                doc.get("automatic_followon_start", False)))


def live_money(repo_root: str | Path = REPO) -> dict[str, Any]:
    """The four limits and the live balances, DERIVED. Never transcribed."""
    import sys

    root = Path(repo_root)
    sys.path.insert(0, str(root / "scripts"))
    from consolidate.derive_budget import derive

    terms = json.loads((root / BUDGET_TERMS).read_text())
    ep, ap = terms["execution_package"], terms["accepted_pricing"]
    live = derive(root)
    return {
        "per_session_envelope_usd": float(ep["per_attempt_hard_ceiling_usd"]),
        "project_cap_usd": float(ap["cumulative_cap_usd"]),
        "formal_allowance_usd": float(ep["formal_allowance_usd"]),
        "package_total_usd": float(ep["package_total_usd"]),
        "formal_remaining_usd": float(live["formal"]["remaining_usd"]),
        "package_remaining_usd": float(live["package"]["remaining_usd"]),
        "project_remaining_usd": float(live["project"]["remaining_usd"]),
        "cumulative_spend_usd": float(live["project"]["cumulative_spend_usd"]),
        "funds_formal_sessions_of": list(
            ep["funds_formal_sessions_of"]["experiment_ids"]),
        "_derived_by": "scripts/consolidate/derive_budget.py",
    }


def session_ceiling(repo_root: str | Path = REPO) -> dict[str, float]:
    """The SEARCH session's derived price, through the existing machinery.

    `write_d1_design.topk_search_cost()` when a production basis exists, which is
    the figure `budget.chain.sessions.search` and `search_stage.cost` both carry;
    the frozen model otherwise. No second pricing formula.
    """
    import sys

    root = Path(repo_root)
    for extra in ("scripts", "scripts/autoinit"):
        p = str(root / extra)
        if p not in sys.path:
            sys.path.insert(0, p)
    import write_d1_design as w

    priced = w.topk_search_cost()
    if priced is None:
        from experiments.phase_d1 import search_space as d1

        w._ensure_the_frozen_operators_are_registered()
        session = d1.search_cost()
        basis = "FROZEN full-vocabulary planning basis: no production measurement"
    else:
        session = priced["search_session"]
        basis = ("MEASURED production Top-K invocation, rebuilt end to end: "
                 + str(priced["basis"]["operator_invocation_seconds_max"]) + " s")
    return {
        "hard_ceiling_usd": float(session["hard_ceiling_usd"]),
        "expected_usd": round(
            float(session["expected_minutes"]) / 60.0
            * float(session["price_per_hour"])
            + float(session["container_disk_usd"])
            * float(session["expected_minutes"])
            / float(session["hard_ceiling_minutes"]), 4),
        "hard_ceiling_minutes": float(session["hard_ceiling_minutes"]),
        "price_per_hour": float(session["price_per_hour"]),
        "_basis": basis,
    }


def check_the_four_conditions(*, ceiling: float,
                              money: Mapping[str, Any]) -> list[str]:
    """Every condition that fails, as a list. Empty means issuable.

    All four, every time: the package total binds separately and looked implied
    only while it equalled formal + engineering, which an amendment can change.
    """
    failed: list[str] = []
    if ceiling > money["per_session_envelope_usd"] + 5e-4:
        failed.append(
            f"the derived ceiling ${ceiling:.4f} exceeds the "
            f"${money['per_session_envelope_usd']:.4f} per-session envelope")
    projected = money["cumulative_spend_usd"] + ceiling
    if projected > money["project_cap_usd"] + 5e-4:
        failed.append(
            f"cumulative ${money['cumulative_spend_usd']:.4f} + ${ceiling:.4f} "
            f"= ${projected:.4f} exceeds the ${money['project_cap_usd']:.4f} "
            "project cap")
    if ceiling > money["formal_remaining_usd"] + 5e-4:
        failed.append(
            f"the ceiling ${ceiling:.4f} exceeds the remaining FORMAL allowance "
            f"${money['formal_remaining_usd']:.4f}")
    if ceiling > money["package_remaining_usd"] + 5e-4:
        failed.append(
            f"the ceiling ${ceiling:.4f} exceeds the remaining PACKAGE total "
            f"${money['package_remaining_usd']:.4f}")
    return failed


def build_payload(*, grant: Mapping[str, Any], session_commit: str,
                  granted_utc: str, arm: str,
                  repo_root: str | Path = REPO,
                  workdir: Path | None = None) -> dict[str, Any]:
    """The authorization payload, fully derived from the tree it will bind."""
    import tempfile

    from experiments.phase_d1 import d1_session as S

    root = Path(repo_root)
    for f in GRANT_MAY_NOT_STATE:
        if f in grant:
            raise D1AuthorizationRefused(
                f"the grant states {f!r}, which this module DERIVES. A grant that "
                "can state its own identity or its own stages can change the "
                "experiment or over-authorize itself.")

    design = json.loads((root / DESIGN_REL).read_text())
    blockers = list(design["open_blockers"])
    if blockers:
        raise D1AuthorizationRefused(
            f"the design still reports open blockers {blockers}; an "
            "authorization may not be issued over one")

    money = live_money(root)
    if S.EXPERIMENT_ID not in money["funds_formal_sessions_of"]:
        raise D1AuthorizationRefused(
            f"{S.EXPERIMENT_ID} is not in funds_formal_sessions_of "
            f"{money['funds_formal_sessions_of']}, so its formal spend would "
            "reach the project cumulative while the formal book reported it as "
            "never having happened")

    priced = session_ceiling(root)
    ceiling = priced["hard_ceiling_usd"]
    failed = check_the_four_conditions(ceiling=ceiling, money=money)
    if failed:
        raise D1AuthorizationRefused(
            "the session is not fundable:\n  - " + "\n  - ".join(failed))

    #: THE ENVELOPE IS NOT THE GRANT. A grant asking for the envelope would
    #: authorize more than the session was priced at.
    asked = grant.get("hard_cap_usd")
    if asked is not None and abs(float(asked) - ceiling) > 5e-4:
        raise D1AuthorizationRefused(
            f"the grant asks for ${float(asked):.4f}; the DERIVED ceiling is "
            f"${ceiling:.4f}. The authorization carries the derived figure.")
    stated_cap = grant.get("cumulative_cap_usd")
    if stated_cap is not None and \
            abs(float(stated_cap) - money["project_cap_usd"]) > 5e-4:
        raise D1AuthorizationRefused(
            f"the grant names cap ${float(stated_cap):.4f}, not "
            f"${money['project_cap_usd']:.4f}")

    #: THE SESSION IS BUILT AND CHECKED before a dollar is authorized, so the
    #: authorization binds the identities the run will actually carry.
    S._register_frozen_operators()
    tmp = Path(workdir or tempfile.mkdtemp(prefix="d1-auth-"))
    session = S.build_session(arm=arm, workdir=tmp, run_id="authorization-probe",
                              device="cpu", repo_root=root)
    contract = S.assert_session_contract(session, root)

    auth = D1Authorization(
        authorization_id=f"autoinit.v1.{S.EXPERIMENT_ID}",
        granted_utc=granted_utc,
        granted_by=str(grant.get("granted_by") or "")[:4000],
        plan_id=f"autoinit.v1.{S.EXPERIMENT_ID}",
        plan_hash=design["design_hash"],
        science_plan_hash=design["design_hash"],
        expected_usd=priced["expected_usd"],
        hard_cap_usd=ceiling,
        per_launch_hard_usd=ceiling,
        authorized_stages=AUTHORIZED_STAGES,
        stage_conditions=STAGE_CONDITIONS,
        session_commit=session_commit,
        design_hash=design["design_hash"],
        arm=arm,
        measurement_protocol_id=contract["measurement_protocol_id"],
        config_hash=contract["config_hash"],
        suite_content_sha256=contract["suite_content_sha256"],
        money={**money, "derived_session": priced,
               "four_conditions": "all four checked; see check_the_four_conditions"},
        one_use=("ONE grant, ONE issuance, ONE launcher session. A failure is this "
                 "session's result; a retry is a new grant and a new "
                 "authorization."),
        authorizes=("ONE D1 formal SEARCH session on the frozen design, stages "
                    f"{'/'.join(AUTHORIZED_STAGES)}. It does NOT authorize "
                    "recovery, behavioural screening, confirmation, promotion, "
                    "D2, D3, or any repetition of a completed measurement."),
    )
    return auth.as_dict()
