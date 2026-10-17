"""D2's ONE-USE search authorization: D1's money rules, D2's identities.

A grant says what the maintainer decided -- that THIS preregistration revision
is approved, who granted it, what it covers. Everything else is DERIVED from
the tree the authorization binds and refused if the grant states it: the
ceiling (re-priced at the live rate from `d2_budget.py`'s search price in
force), the stages, the session's protocol and config identities, the
objective and ranking hashes, the harness.

The money is checked by D1's four conditions, imported rather than copied --
per-session envelope, project cap, formal allowance, package total -- plus the
funding scope: `phase_d2` must be in `funds_formal_sessions_of`, which is a
maintainer amendment, never an autonomous one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from aadistill.governance.authorization import (
    ActionPolicy, AuthorizationError, SpendAuthorization,
)
from aadistill.infrastructure.manifest import sha256_json

REPO = Path(__file__).resolve().parents[4]

SCHEMA = "aadistill.phase_d2.authorization/v1"
#: `device` is a field of `SearchConfig.as_dict()`, so the issuer, the
#: launcher's $0 gate and the driver must build on the same one.
FORMAL_DEVICE = "cuda"

D2_SEARCH_POLICY = ActionPolicy(
    policy_id="phase_d2_search",
    allowed=frozenset({"beam_search", "d2_search"}),
    wire_claims={
        "allows_beam_search": "beam_search",
        "authorizes_d2_search": "d2_search",
        "allows_recovery": "recovery",
        "allows_behavioural": "behavioural",
        "automatic_followon_start": "automatic_followon_start",
    },
    wire_schema=SCHEMA,
    plan_hash_key="plan_hash",
    enforcement=(
        "the launcher loads this artifact, refuses a planned hard threshold "
        "above hard_cap_usd, a harness whose derived closure differs in "
        "membership or digest, and a preregistration revision other than the "
        "bound one; the driver re-checks every bound identity before the root"),
    refusal_notes={
        "recovery": "a D2 SEARCH trains nothing; recovery is a separate grant",
        "behavioural": ("screening and confirmation on the shared protocol are "
                        "separately authorized sessions"),
        "automatic_followon_start": ("the committed field is reviewed before "
                                     "any probe is trained"),
    },
)

AUTHORIZED_STAGES = ("A", "B", "C", "D")
STAGE_CONDITIONS = {
    "A": "setup; the session contract asserted against the preregistration",
    "B": "the root teacher once; the evaluator primed with it",
    "C": ("the beam search minimizing the D2 objective, under the soft stop "
          "and the reference-cache regression guard"),
    "D": "commit the preregistered Top-K and record the KL-only control",
}
GRANT_MAY_NOT_STATE = (
    "authorization_id", "plan_id", "plan_hash", "authorized_stages",
    "stage_conditions", "expected_usd", "per_launch_hard_usd", "hard_cap_usd",
    "measurement_protocol_id", "config_hash", "objective_hash",
)

D2_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/stages/stage-1/phase_d2/autoinit_d2_launch.py",
    "scripts/stages/stage-1/phase_d2/autoinit_d2_driver.py",
    "scripts/stages/stage-1/phase_d2/d2_session.py",
    "scripts/stages/stage-1/phase_d2/d2_authorization.py",
    "scripts/shared/pod/collect_artifacts.py",
)
#: Bytes no import edge reaches. `stages.d_series` modules are declared, as
#: D1's precedent does; the preregistration is bound by `require_plan`.
D2_DECLARED_INPUTS: tuple[str, ...] = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "configs/stages/stage-1/phase_c1/authorization.json",
    "configs/stages/stage-1/phase_d2/d2_search_artifacts.json",
    "configs/stages/stage-1/phase_d2/d2_search_artifacts_failed.json",
    "scripts/stages/stage-1/families/d_series/scoring_protocol.py",
    "scripts/stages/stage-1/families/d_series/objectives.py",
)
D2_SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/stages/stage-1")


class D2AuthorizationRefused(AuthorizationError):
    """A condition does not hold. An `AuthorizationError`, so the runner
    reports it as a refusal rather than an unhandled error."""


def d2_current_executable(repo_root: str | Path = REPO) -> dict[str, Any]:
    from aadistill.governance.closure import ClosureError, derive

    try:
        return derive(repo_root, "phase_d2", D2_ENTRY_POINTS, D2_DECLARED_INPUTS,
                      roots=D2_SOURCE_ROOTS)
    except ClosureError as exc:
        raise D2AuthorizationRefused(
            f"cannot derive the D2 executable set: {exc}") from exc


@dataclass(frozen=True)
class D2Authorization(SpendAuthorization):
    """The issued artifact: a `SpendAuthorization` by type, D2 by schema."""

    preregistration_hash: str = ""
    run_id: str = ""
    measurement_protocol_id: str = ""
    config_hash: str = ""
    suite_content_sha256: str = ""
    objective_hash: str = ""
    ranking_policy_hash: str = ""
    prediction_rows: str = "all_v1"
    pricing: Mapping[str, Any] = field(default_factory=dict)
    money: Mapping[str, Any] = field(default_factory=dict)
    one_use: str = ""
    authorizes_d2_search: bool = True
    allows_beam_search: bool = True
    allows_recovery: bool = False
    allows_behavioural: bool = False
    automatic_followon_start: bool = False

    def require_harness(self, repo_root: str | Path = REPO) -> dict[str, Any]:
        """Refuse drift in MEMBERSHIP and in digest (D1's rule, same reason)."""
        observed = d2_current_executable(repo_root)
        if not self.harness_source_digest:
            raise D2AuthorizationRefused("no harness_source_digest is bound")
        recorded = set(self.harness_source_files)
        live = {f["path"] for f in observed["files"]}
        if recorded and recorded != live:
            raise D2AuthorizationRefused(
                "the D2 executable closure MEMBERSHIP changed since the grant: "
                f"added {sorted(live - recorded)[:4]}, removed "
                f"{sorted(recorded - live)[:4]}")
        if observed["digest"] != self.harness_source_digest:
            raise D2AuthorizationRefused(
                f"the D2 harness digests to {observed['digest'][:16]}, the grant "
                f"bound {self.harness_source_digest[:16]}")
        return observed

    def require_plan(self, plan_hash: str) -> None:
        if not plan_hash or self.plan_hash != plan_hash:
            raise D2AuthorizationRefused(
                f"this authorization binds preregistration "
                f"{self.plan_hash[:16]} and the session declares "
                f"{str(plan_hash)[:16]}")

    def require_session_commit(self, commit: str) -> None:
        if not self.authorized_session_commit or \
                commit != self.authorized_session_commit:
            raise D2AuthorizationRefused(
                f"this authorization binds commit "
                f"{self.authorized_session_commit[:12]}, the session declares "
                f"{str(commit)[:12]}")

    def require_run_id(self, run_id: str) -> None:
        """`run_id` is a field of `config_hash`: a different run is a
        different identity, refused here at $0 rather than in stage A."""
        if not self.run_id or run_id != self.run_id:
            raise D2AuthorizationRefused(
                f"this authorization was issued for run {self.run_id!r}, the "
                f"session declares {run_id!r}")

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        payload.update({
            "preregistration_hash": self.preregistration_hash,
            "run_id": self.run_id,
            "measurement_protocol_id": self.measurement_protocol_id,
            "config_hash": self.config_hash,
            "suite_content_sha256": self.suite_content_sha256,
            "objective_hash": self.objective_hash,
            "ranking_policy_hash": self.ranking_policy_hash,
            "prediction_rows": self.prediction_rows,
            "pricing": dict(self.pricing),
            "money": dict(self.money),
            "one_use": self.one_use,
            "authorizes_d2_search": bool(self.authorizes_d2_search),
            "allows_beam_search": bool(self.allows_beam_search),
            "allows_recovery": bool(self.allows_recovery),
            "allows_behavioural": bool(self.allows_behavioural),
            "automatic_followon_start": bool(self.automatic_followon_start),
        })
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "D2Authorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = {k: v for k, v in raw.items() if k != "authorization_sha256"}
        if not stated or stated != sha256_json(check):
            raise D2AuthorizationRefused(
                f"{path} does not match its own authorization_sha256")
        if raw.get("schema") != SCHEMA:
            raise D2AuthorizationRefused(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}")
        missing = [f for f in (
            "authorization_id", "granted_utc", "hard_cap_usd",
            "authorized_session_commit", "preregistration_hash", "run_id",
            "measurement_protocol_id", "config_hash", "suite_content_sha256",
            "objective_hash", "ranking_policy_hash", "authorized_stages",
            "harness_source_digest") if not raw.get(f)]
        if missing:
            raise D2AuthorizationRefused(f"{path} omits {missing}")
        D2_SEARCH_POLICY.check_claims(raw, where=str(path))
        if not raw.get("allows_beam_search") or not raw.get("authorizes_d2_search"):
            raise D2AuthorizationRefused(f"{path} does not authorize a D2 search")
        return cls(
            authorization_id=raw["authorization_id"],
            granted_utc=raw["granted_utc"], granted_by=raw.get("granted_by", ""),
            plan_id=raw["plan_id"], plan_hash=raw["plan_hash"],
            expected_usd=float(raw["expected_usd"]),
            hard_cap_usd=float(raw["hard_cap_usd"]),
            authorized_stages=tuple(raw["authorized_stages"]),
            stage_conditions=dict(raw["stage_conditions"]),
            scope_note=raw.get("scope_note", ""),
            authorized_session_commit=raw["authorized_session_commit"],
            harness_source_digest=raw["harness_source_digest"],
            harness_source_files=tuple(raw.get("harness_source_files") or ()),
            per_launch_hard_usd=(float(raw["per_launch_hard_usd"])
                                 if raw.get("per_launch_hard_usd") else None),
            provenance_commit=raw.get("provenance_commit"),
            preregistration_hash=raw["preregistration_hash"],
            run_id=raw["run_id"],
            measurement_protocol_id=raw["measurement_protocol_id"],
            config_hash=raw["config_hash"],
            suite_content_sha256=raw["suite_content_sha256"],
            objective_hash=raw["objective_hash"],
            ranking_policy_hash=raw["ranking_policy_hash"],
            prediction_rows=raw.get("prediction_rows", "all_v1"),
            pricing=dict(raw.get("pricing") or {}),
            money=dict(raw.get("money") or {}),
            one_use=raw.get("one_use", ""),
            authorizes_d2_search=bool(raw["authorizes_d2_search"]),
            allows_beam_search=bool(raw["allows_beam_search"]),
            allows_recovery=bool(raw.get("allows_recovery", False)),
            allows_behavioural=bool(raw.get("allows_behavioural", False)),
            automatic_followon_start=bool(raw.get("automatic_followon_start",
                                                  False)),
            action_policy=D2_SEARCH_POLICY)


def session_ceiling(repo_root: str | Path = REPO,
                    prediction_rows: str = "all_v1") -> dict[str, Any]:
    """D2's search price IN FORCE for the row selection the session DECLARES,
    from `d2_budget.py`. No second formula.

    `records_max` until the verified stage-L evidence validates the reference
    cache fix, `cache_fix_validated` after; the budget record says which and
    carries the evidence it rests on. The row selection prices itself: a
    scored_v1 session takes scored_v1's lower bound ONLY while version 3's
    own scored-rows equivalence is adopted, and is refused otherwise -- never
    priced at the cheaper bound on another objective's evidence.
    """
    from stages.d_series import d2_budget as BUD

    budget = BUD.build()
    search = budget["steps"]["d2_search"]
    modes = search["by_prediction_rows"]
    if prediction_rows not in modes:
        raise D2AuthorizationRefused(
            f"prediction_rows {prediction_rows!r} has no price; priced: "
            f"{sorted(modes)}")
    mode = modes[prediction_rows]
    if not mode["eligible"]:
        raise D2AuthorizationRefused(
            f"prediction_rows={prediction_rows!r} is not priceable: its "
            f"eligibility is {mode['eligibility']}; the search would be "
            "priced at a discount no adopted evidence supports")
    priced = mode["priced_by"]
    rate = float(budget["rate_usd_per_hour_basis"])
    hard_min = float(priced["hard_ceiling_minutes"])
    exp_min = float(mode["expected"]["minutes"])
    disk_usd = float(mode["hard"]["disk_usd"])
    eligibility = mode["eligibility"]
    return {"hard_ceiling_usd": float(mode["hard"]["all_in_usd"]),
            "expected_usd": float(mode["expected"]["all_in_usd"]),
            "hard_ceiling_minutes": hard_min, "expected_minutes": exp_min,
            "session_overhead_minutes": float(priced["session_overhead_minutes"]),
            "container_disk_usd": disk_usd,
            "container_disk_gb": int(search["container_disk_gb"]),
            "price_per_hour": rate,
            "prediction_rows": prediction_rows,
            "prediction_rows_in_force": search["prediction_rows_in_force"],
            "scored_rows_evidence_sha256": (
                eligibility.get("evidence_session_sha256")
                if isinstance(eligibility, dict) else None),
            "pricing_in_force": search["pricing_in_force"],
            "calibration_evidence_sha256": (search["calibration_evidence"]
                                            .get("evidence_session_sha256")),
            "_basis": (f"{BUD.RECORD} :: steps.d2_search.by_prediction_rows."
                       f"{prediction_rows} ({search['pricing_in_force']})")}


def reprice_at(rate_usd_per_hour: float,
               repo_root: str | Path = REPO,
               prediction_rows: str = "all_v1") -> dict[str, Any]:
    """The SAME accepted minute bound, costed at the live rate."""
    accepted = session_ceiling(repo_root, prediction_rows)
    minutes, exp_min = accepted["hard_ceiling_minutes"], accepted["expected_minutes"]
    disk_per_hour = (accepted["container_disk_usd"] / (minutes / 60.0)
                     if minutes else 0.0)
    return {**accepted,
            "hard_ceiling_usd": round(minutes / 60.0 * (rate_usd_per_hour
                                                        + disk_per_hour), 4),
            "expected_usd": round(exp_min / 60.0 * (rate_usd_per_hour
                                                    + disk_per_hour), 4),
            "price_per_hour": float(rate_usd_per_hour),
            "accepted_at_price_per_hour": accepted["price_per_hour"]}


def build_payload(*, grant: Mapping[str, Any], session_commit: str,
                  granted_utc: str, run_id: str, prediction_rows: str = "all_v1",
                  repo_root: str | Path = REPO, workdir: Path | None = None,
                  live_rate: float | None = None) -> dict[str, Any]:
    """The authorization payload, derived from the tree it will bind."""
    import tempfile

    from stages.phase_d1.d1_authorization import (
        check_the_four_conditions, live_money, live_secure_price,
    )
    from stages.phase_d2 import d2_session as S

    root = Path(repo_root)
    for f in GRANT_MAY_NOT_STATE:
        if f in grant:
            raise D2AuthorizationRefused(
                f"the grant states {f!r}, which this module DERIVES")
    if not session_commit or len(session_commit) != 40:
        raise D2AuthorizationRefused(f"{session_commit!r} is not a full commit id")
    if not str(run_id or "").strip():
        raise D2AuthorizationRefused("no run_id; config_hash is a function of it")
    prereg = S.preregistration(root)
    approved = grant.get("approves_preregistration_hash")
    if approved != prereg["preregistration_hash"]:
        raise D2AuthorizationRefused(
            f"the grant approves preregistration {str(approved)[:16]} and the "
            f"tree carries {prereg['preregistration_hash'][:16]}; a search may "
            "only run under the revision the maintainer approved")
    if prediction_rows != "all_v1":
        from stages.d_series import d2_budget as BUD
        #: VERSION 3's own scored-rows equivalence (stage W). Stage V's PASS
        #: is version 2's -- another objective and another support -- and
        #: does not extend to version 3, any more than R0's or stage L's do.
        verdict = BUD.d2v3_validation_evidence().get("scored_rows_equivalence")
        if verdict != "PASS":
            raise D2AuthorizationRefused(
                f"prediction_rows={prediction_rows!r} needs THIS objective's "
                "own scored-rows equivalence (the D2 v3 GPU validation, "
                f"stage W), which is {verdict!r}; Stage V's verdict on "
                "version 2, R0's KL-only verdict and stage L's verdict on "
                "the superseded objective do not extend to it")

    money = live_money(root)
    if S.EXPERIMENT_ID not in money["funds_formal_sessions_of"]:
        raise D2AuthorizationRefused(
            f"{S.EXPERIMENT_ID} is not in funds_formal_sessions_of "
            f"{money['funds_formal_sessions_of']}: adding it is a maintainer "
            "amendment, and without it the formal book would not see the spend")
    quote = ({"gpu": "NVIDIA L40S", "usd_per_hour": float(live_rate),
              "_quoted": "supplied by the caller (tests only)"}
             if live_rate is not None else live_secure_price())
    priced = reprice_at(quote["usd_per_hour"], root, prediction_rows)
    failed = check_the_four_conditions(ceiling=priced["hard_ceiling_usd"],
                                       money=money)
    if failed:
        raise D2AuthorizationRefused(
            f"at ${quote['usd_per_hour']:.4f}/h the D2 search is not fundable "
            f"({priced['pricing_in_force']} price):\n  - " + "\n  - ".join(failed))

    tmp = Path(workdir or tempfile.mkdtemp(prefix="d2-auth-"))
    session = S.build_session(workdir=tmp, run_id=str(run_id).strip(),
                              device=FORMAL_DEVICE,
                              prediction_rows=prediction_rows, repo_root=root)
    contract = S.assert_session_contract(session, root)
    closure = d2_current_executable(root)
    auth = D2Authorization(
        authorization_id=f"autoinit.v1.{S.EXPERIMENT_ID}",
        granted_utc=granted_utc,
        granted_by=str(grant.get("granted_by") or "")[:4000],
        plan_id=f"autoinit.v1.{S.EXPERIMENT_ID}",
        plan_hash=prereg["preregistration_hash"],
        expected_usd=priced["expected_usd"],
        hard_cap_usd=priced["hard_ceiling_usd"],
        per_launch_hard_usd=priced["hard_ceiling_usd"],
        authorized_stages=AUTHORIZED_STAGES, stage_conditions=STAGE_CONDITIONS,
        scope_note=str(grant.get("covers") or "")[:4000],
        authorized_session_commit=session_commit,
        harness_source_digest=closure["digest"],
        harness_source_files=tuple(f["path"] for f in closure["files"]),
        provenance_commit=session_commit,
        preregistration_hash=prereg["preregistration_hash"],
        run_id=str(run_id).strip(),
        measurement_protocol_id=contract["measurement_protocol_id"],
        config_hash=contract["config_hash"],
        suite_content_sha256=session.suite_content_sha256,
        objective_hash=contract["objective_hash"],
        ranking_policy_hash=contract["ranking_policy_hash"],
        prediction_rows=prediction_rows,
        pricing={k: priced[k] for k in (
            "pricing_in_force", "calibration_evidence_sha256",
            "hard_ceiling_minutes", "expected_minutes", "prediction_rows",
            "scored_rows_evidence_sha256", "_basis")},
        money={**money, "derived_session": priced, "live_quote": quote},
        one_use=("ONE grant, ONE issuance, ONE launcher session; a retry is a "
                 "new authorization"),
        action_policy=D2_SEARCH_POLICY)
    payload = auth.as_dict()
    payload["authorizes"] = (
        "ONE D2 formal SEARCH session under the approved preregistration, "
        f"stages {'/'.join(AUTHORIZED_STAGES)}. Not recovery, not screening or "
        "confirmation, not D3, not a promotion.")
    payload["harness"] = {"n_files": closure["n_files"],
                          "entry_points": list(closure["entry_points"]),
                          "digest": closure["digest"]}
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload
