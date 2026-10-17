"""The new D3's ONE-USE authorizations: the STRUCTURAL session and the PROBES.

Two sessions, two grants, two types (maintainer decision 2026-10-16): the
structural ablation runs first; the three recovery probes are separately
authorized and only after the structural session completes with
STRUCTURAL_CHANGE. Each grant states what the maintainer decided -- the
registration it approves, the rung, a cumulative envelope -- and every other
field is DERIVED from the tree the artifact binds and refused if stated: the
ceiling (re-priced at the live rate from `d3_budget.py`), the contract it
binds (the treatment path hash; the probes contract hash), the harness.

WHAT EACH MAY NEVER REACH, by type rather than by prose: a beam or Full Search,
reopening D1 or D2, the cancelled student-confidence D3, recovery training (the
structural grant) or another structural run (the probes grant), retraining or
regenerating q2/B, the screening seeds, a fresh confirmation, a promotion or
any automatic follow-on.

The money is D1's four conditions (imported) and the funding scope
(`phase_d3` in `funds_formal_sessions_of` is a maintainer amendment), plus the
CUMULATIVE ENVELOPE of the rung, netted against every earlier subrun of that
rung by its booked cost (`_subrun_cost`, D2's fail-closed reading).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, NamedTuple

from aadistill.governance.authorization import (
    ActionPolicy, AuthorizationError, SpendAuthorization,
)
from aadistill.infrastructure.manifest import sha256_json

REPO = Path(__file__).resolve().parents[4]
RUNS_ROOT = "logs/stages/stage-1/phase_d3/runs"
SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/stages/stage-1")
AUTHORIZED_STAGES = ("A", "B", "C", "D")

#: Every action EITHER grant must refuse, with why. Shared so the two
#: policies cannot drift apart on what neither may do.
_SHARED_DENIALS = {
    "beam_search": ("the D3 control is D1-q2's fixed path; no beam or Full "
                    "Search is part of this design"),
    "reopen_d1_d2": "D1 and D2 are complete; their records are not re-run",
    "cancelled_d3": "the student-confidence D3 is cancelled and unexecuted",
    "reference_retraining": ("q2 and B are IMPORTED units; nothing about them "
                             "is retrained, regenerated or re-evaluated"),
    "screening_seeds": ("the historical screening seeds are not part of the "
                        "registered design"),
    "fresh_confirmation": "a fresh independent confirmation is its own grant",
    "promotion": "an exploratory comparison never promotes",
    "automatic_followon_start": ("the next step is a maintainer decision on "
                                 "the evidence"),
}


def _policy(policy_id: str, schema: str, allowed: set[str],
            own: Mapping[str, str], extra_denials: Mapping[str, str]
            ) -> ActionPolicy:
    denials = {**_SHARED_DENIALS, **extra_denials}
    claims = {**own, **{f"authorizes_{a}": a for a in denials}}
    return ActionPolicy(
        policy_id=policy_id, allowed=frozenset(allowed), wire_claims=claims,
        wire_schema=schema, plan_hash_key="plan_hash",
        enforcement=(
            "the launcher loads this artifact, refuses a planned hard "
            "threshold above hard_cap_usd, a harness whose derived closure "
            "differs, a registration or contract other than the bound one; "
            "the driver re-derives the contract on the pod and refuses a "
            "mismatch before any work"),
        refusal_notes=dict(denials))


STRUCTURAL_SCHEMA = "aadistill.phase_d3.structural_authorization/v1"
PROBES_SCHEMA = "aadistill.phase_d3.probes_authorization/v1"
STRUCTURAL_POLICY = _policy(
    "phase_d3_structural", STRUCTURAL_SCHEMA,
    {"fixed_path_materialization", "d3_structural"},
    {"allows_fixed_path_materialization": "fixed_path_materialization",
     "authorizes_d3_structural": "d3_structural"},
    {"recovery_training": "the structural session trains nothing",
     "d3_probes": ("the three probes are a separate grant, issued only after "
                   "STRUCTURAL_CHANGE")})
PROBES_POLICY = _policy(
    "phase_d3_probes", PROBES_SCHEMA,
    {"recovery_training", "d3_probes"},
    {"allows_recovery_training": "recovery_training",
     "authorizes_d3_probes": "d3_probes"},
    {"d3_structural": ("the structural session is complete; its product is "
                       "what these probes train from"),
     "fixed_path_materialization": "the initialization is the structural "
                                   "session's product, staged from the relay"})

STRUCTURAL_STAGE_CONDITIONS = {
    "A": ("setup on an admitted host; the registration, the registered "
          "treatment spec and its single-factor contract verified"),
    "B": ("the root state re-derived from the control plan; the execution "
          "agreement; the staged teacher verified"),
    "C": ("q2's fixed path from the teacher with DEPTH under the declared "
          "teacher reference: the teacher reference built first; step 0 "
          "pinned to q2's FFN digest; every step persisted on completion; "
          "the parent-reference comparability gate on the DEPTH step, which "
          "stops an inadmissible run before WIDTH"),
    "D": ("the structural comparison and outcome; the product's four "
          "identities; the product and evidence secured off-pod"),
}
PROBES_STAGE_CONDITIONS = {
    "A": ("setup on an admitted host; the D3 initial checkpoint staged from "
          "the relay and verified at all four identities; the contract "
          "re-derived and equal to the bound hash; the training runtime equal "
          "to the protocol's"),
    "B": ("the three probes: frozen recipe from the D3 initial checkpoint at "
          "each D1 confirmation seed, packaged, generated through the "
          "unchanged generator (token-id sidecar), admitted against the "
          "shared confirmation rung's evaluation protocol, scored by its "
          "pinned scorer"),
    "C": "per probe, the rollout rescoring diagnostic, failure-isolated",
    "D": ("stop at preservation; the exploratory comparison with q2 and B is "
          "computed off-pod at $0"),
}


class RungGovernance(NamedTuple):
    rung: str
    schema: str
    policy: ActionPolicy
    run_prefix: str
    stage_conditions: Mapping[str, str]
    budget_step: str
    entry_points: tuple[str, ...]
    declared_inputs: tuple[str, ...]
    required_claim: str


STRUCTURAL_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/stages/stage-1/phase_d3/autoinit_d3_structural_launch.py",
    "scripts/stages/stage-1/phase_d3/autoinit_d3_structural_driver.py",
    "scripts/stages/stage-1/phase_d3/d3_design.py",
    "scripts/stages/stage-1/phase_d3/d3_authorization.py",
    "scripts/shared/pod/collect_artifacts.py",
)
STRUCTURAL_DECLARED_INPUTS: tuple[str, ...] = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "configs/stages/stage-1/phase_c1/authorization.json",
    "configs/stages/stage-1/phase_d3/d3_structural_artifacts.json",
    "configs/stages/stage-1/phase_d3/d3_structural_artifacts_failed.json",
    "scripts/stages/stage-1/families/d_series/scoring_protocol.py",
    "logs/stages/stage-1/phase_d3/plans/d3_registration_v1.json",
    "logs/stages/stage-1/phase_d3/plans/d3_control_q2.json",
    "logs/stages/stage-1/phase_d3/plans/d3_parent_reference_admission_v1.json",
    "logs/stages/stage-1/phase_d1/plans/d1_behavioural_replay_plan.json",
)
PROBES_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/stages/stage-1/phase_d3/autoinit_d3_probes_launch.py",
    "scripts/stages/stage-1/phase_d2/autoinit_d2_screening_driver.py",
    "scripts/stages/stage-1/phase_d2/rescore_d2_probe.py",
    "scripts/stages/stage-1/phase_d3/d3_probes.py",
    "scripts/stages/stage-1/phase_d3/d3_authorization.py",
    "scripts/stages/stage-1/phase_d1/score_d1_confirmation.py",
    "scripts/shared/evaluation/uncapped_eval_token_ids.py",
    "scripts/shared/evaluation/uncapped_eval.py",
    "scripts/shared/training/train_stage3.py",
    "scripts/shared/pod/autoinit_engine_probe.py",
    "scripts/shared/pod/collect_artifacts.py",
)
PROBES_DECLARED_INPUTS: tuple[str, ...] = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "configs/stages/stage-1/phase_c1/authorization.json",
    "configs/stages/stage-1/phase_d3/d3_probes_artifacts.json",
    "configs/stages/stage-1/phase_d3/d3_probes_artifacts_failed.json",
    "configs/stages/stage-3/recipes/recovery.json",
    "configs/stages/stage-3/e1/e1_r0860k_sa_pca.json",
    "logs/stages/stage-1/phase_d3/plans/d3_registration_v1.json",
)

RUNG_GOVERNANCE: dict[str, RungGovernance] = {
    "structural": RungGovernance(
        rung="structural", schema=STRUCTURAL_SCHEMA, policy=STRUCTURAL_POLICY,
        run_prefix="d3_structural_",
        stage_conditions=STRUCTURAL_STAGE_CONDITIONS,
        budget_step="d3_structural", entry_points=STRUCTURAL_ENTRY_POINTS,
        declared_inputs=STRUCTURAL_DECLARED_INPUTS,
        required_claim="authorizes_d3_structural"),
    "probes": RungGovernance(
        rung="probes", schema=PROBES_SCHEMA, policy=PROBES_POLICY,
        run_prefix="d3_probes_", stage_conditions=PROBES_STAGE_CONDITIONS,
        budget_step="d3_probes", entry_points=PROBES_ENTRY_POINTS,
        declared_inputs=PROBES_DECLARED_INPUTS,
        required_claim="authorizes_d3_probes"),
}
GRANT_MAY_NOT_STATE = (
    "authorization_id", "plan_id", "plan_hash", "authorized_stages",
    "stage_conditions", "expected_usd", "per_launch_hard_usd", "hard_cap_usd",
    "contract_hash", "seeds", "n_probes",
)


class D3AuthorizationRefused(AuthorizationError):
    """A D3 condition does not hold. An `AuthorizationError`, so the runner
    reports a refusal rather than an unhandled error."""


def governance(rung: str) -> RungGovernance:
    if rung not in RUNG_GOVERNANCE:
        raise D3AuthorizationRefused(f"no D3 rung {rung!r}")
    return RUNG_GOVERNANCE[rung]


def current_executable(repo_root: str | Path = REPO,
                       rung: str = "structural") -> dict[str, Any]:
    from aadistill.governance.closure import ClosureError, derive

    gov = governance(rung)
    try:
        return derive(repo_root, f"phase_d3_{rung}", gov.entry_points,
                      gov.declared_inputs, roots=SOURCE_ROOTS)
    except ClosureError as exc:
        raise D3AuthorizationRefused(
            f"cannot derive the D3 {rung} executable set: {exc}") from exc


@dataclass(frozen=True)
class _D3Authorization(SpendAuthorization):
    """The fields both D3 artifacts carry; the subclass fixes the rung."""

    registration_hash: str = ""
    contract_hash: str = ""
    run_id: str = ""
    seeds: tuple[int, ...] = ()
    n_probes: int = 0
    pricing: Mapping[str, Any] = field(default_factory=dict)
    money: Mapping[str, Any] = field(default_factory=dict)
    rate_usd_per_hour: float | None = None
    hard_runtime_minutes: float | None = None
    one_use: str = ""
    allows_fixed_path_materialization: bool = False
    authorizes_d3_structural: bool = False
    allows_recovery_training: bool = False
    authorizes_d3_probes: bool = False
    authorizes_beam_search: bool = False
    authorizes_reopen_d1_d2: bool = False
    authorizes_cancelled_d3: bool = False
    authorizes_reference_retraining: bool = False
    authorizes_screening_seeds: bool = False
    authorizes_fresh_confirmation: bool = False
    authorizes_promotion: bool = False
    authorizes_automatic_followon_start: bool = False
    authorizes_recovery_training: bool = False
    authorizes_fixed_path_materialization: bool = False
    RUNG: ClassVar[str] = ""

    def require_harness(self, repo_root: str | Path = REPO) -> dict[str, Any]:
        observed = current_executable(repo_root, self.RUNG)
        if not self.harness_source_digest:
            raise D3AuthorizationRefused("no harness_source_digest is bound")
        recorded = set(self.harness_source_files)
        live = {f["path"] for f in observed["files"]}
        if recorded and recorded != live:
            raise D3AuthorizationRefused(
                f"the D3 {self.RUNG} executable closure MEMBERSHIP changed "
                f"since the grant: added {sorted(live - recorded)[:4]}, "
                f"removed {sorted(recorded - live)[:4]}")
        if observed["digest"] != self.harness_source_digest:
            raise D3AuthorizationRefused(
                f"the D3 {self.RUNG} harness digests to "
                f"{observed['digest'][:16]}, the grant bound "
                f"{self.harness_source_digest[:16]}")
        return observed

    def require_plan(self, plan_hash: str) -> None:
        if not plan_hash or self.plan_hash != plan_hash:
            raise D3AuthorizationRefused(
                f"this authorization binds registration {self.plan_hash[:16]} "
                f"and the session declares {str(plan_hash)[:16]}")

    def require_session_commit(self, commit: str) -> None:
        if not self.authorized_session_commit or \
                commit != self.authorized_session_commit:
            raise D3AuthorizationRefused(
                f"this authorization binds commit "
                f"{self.authorized_session_commit[:12]}, the session declares "
                f"{str(commit)[:12]}")

    def require_run_id(self, run_id: str) -> None:
        if not self.run_id or run_id != self.run_id:
            raise D3AuthorizationRefused(
                f"this authorization was issued for run {self.run_id!r}, the "
                f"session declares {run_id!r}")

    def require_contract(self, contract_hash: str) -> None:
        if not self.contract_hash or contract_hash != self.contract_hash:
            raise D3AuthorizationRefused(
                f"the session's contract hashes to {str(contract_hash)[:16]} "
                f"and this authorization binds {self.contract_hash[:16]}")

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = governance(self.RUNG).schema
        payload.update({
            "registration_hash": self.registration_hash,
            "contract_hash": self.contract_hash, "run_id": self.run_id,
            "seeds": list(self.seeds), "n_probes": int(self.n_probes),
            "pricing": dict(self.pricing), "money": dict(self.money),
            "rate_usd_per_hour": self.rate_usd_per_hour,
            "hard_runtime_minutes": self.hard_runtime_minutes,
            "one_use": self.one_use})
        for wire in self.action_policy.wire_claims:
            payload[wire] = bool(getattr(self, wire))
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path):
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = {k: v for k, v in raw.items() if k != "authorization_sha256"}
        if not stated or stated != sha256_json(check):
            raise D3AuthorizationRefused(
                f"{path} does not match its own authorization_sha256")
        gov = governance(cls.RUNG)
        if raw.get("schema") != gov.schema:
            raise D3AuthorizationRefused(
                f"{path} declares schema {raw.get('schema')!r}, not "
                f"{gov.schema!r}")
        missing = [f for f in (
            "authorization_id", "granted_utc", "hard_cap_usd",
            "authorized_session_commit", "registration_hash", "contract_hash",
            "run_id", "authorized_stages", "harness_source_digest",
            "hard_runtime_minutes") if not raw.get(f)]
        if missing:
            raise D3AuthorizationRefused(f"{path} omits {missing}")
        gov.policy.check_claims(raw, where=str(path))
        if not raw.get(gov.required_claim):
            raise D3AuthorizationRefused(f"{path} does not authorize D3 "
                                         f"{cls.RUNG}")
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
            registration_hash=raw["registration_hash"],
            contract_hash=raw["contract_hash"], run_id=raw["run_id"],
            seeds=tuple(int(s) for s in raw.get("seeds") or ()),
            n_probes=int(raw.get("n_probes") or 0),
            pricing=dict(raw.get("pricing") or {}),
            money=dict(raw.get("money") or {}),
            rate_usd_per_hour=raw.get("rate_usd_per_hour"),
            hard_runtime_minutes=float(raw["hard_runtime_minutes"]),
            one_use=raw.get("one_use", ""),
            **{claim: bool(raw.get(claim))
               for claim in gov.policy.wire_claims},
            action_policy=gov.policy)


@dataclass(frozen=True)
class D3StructuralAuthorization(_D3Authorization):
    """Permits exactly one D3 structural session of the registered treatment."""

    RUNG: ClassVar[str] = "structural"
    allows_fixed_path_materialization: bool = True
    authorizes_d3_structural: bool = True


@dataclass(frozen=True)
class D3ProbesAuthorization(_D3Authorization):
    """Permits exactly one D3 three-probe session from the D3 product."""

    RUNG: ClassVar[str] = "probes"
    allows_recovery_training: bool = True
    authorizes_d3_probes: bool = True


TYPES = {"structural": D3StructuralAuthorization,
         "probes": D3ProbesAuthorization}


def session_ceiling(repo_root: str | Path = REPO,
                    rung: str = "structural") -> dict[str, Any]:
    """The rung's price from `d3_budget.py`. No second formula."""
    from stages.phase_d3 import d3_budget as BUD

    gov = governance(rung)
    priced = BUD.build()
    step = priced["steps"][gov.budget_step]
    rate = float(priced["rate_usd_per_hour_basis"])
    disk_gb = (int(step["container_disk_gb"]) if "container_disk_gb" in step
               else 120)
    overhead = (float(step["session_overhead_minutes"])
                if "session_overhead_minutes" in step else None)
    if overhead is None:
        from stages.d_series import d2_budget as D2B

        design = json.loads((Path(repo_root) / D2B.D1_DESIGN).read_text())
        cell = design["budget"]["chain"]["sessions"]["confirmation"]
        overhead = float(cell["session_overhead_minutes"])
        disk_gb = int(cell["container_disk_gb"])
    return {"hard_ceiling_usd": float(step["hard"]["all_in_usd"]),
            "expected_usd": float(step["expected"]["all_in_usd"]),
            "hard_ceiling_minutes": float(step["hard"]["minutes"]),
            "expected_minutes": float(step["expected"]["minutes"]),
            "container_disk_usd": float(step["hard"]["disk_usd"]),
            "container_disk_gb": disk_gb,
            "session_overhead_minutes": overhead,
            "price_per_hour": rate,
            "_basis": f"{BUD.RECORD} :: steps.{gov.budget_step}"}


def reprice_at(rate_usd_per_hour: float, repo_root: str | Path = REPO,
               rung: str = "structural") -> dict[str, Any]:
    """The SAME accepted minute bound, costed at the live rate."""
    accepted = session_ceiling(repo_root, rung)
    minutes, exp_min = (accepted["hard_ceiling_minutes"],
                        accepted["expected_minutes"])
    disk_per_hour = accepted["container_disk_usd"] / (minutes / 60.0)
    return {**accepted,
            "hard_ceiling_usd": round(minutes / 60.0 * (rate_usd_per_hour
                                                        + disk_per_hour), 4),
            "expected_usd": round(exp_min / 60.0 * (rate_usd_per_hour
                                                    + disk_per_hour), 4),
            "price_per_hour": float(rate_usd_per_hour),
            "accepted_at_price_per_hour": accepted["price_per_hour"]}


def consumed_by_prior_subruns(repo_root: str | Path = REPO,
                              rung: str = "structural",
                              exclude: str | None = None) -> dict[str, Any]:
    """Every earlier subrun of the rung's booked cost; one unknown refuses.

    `exclude` is the run a closeout is booking right now: it has no closeout
    yet by definition, and is counted by the caller, once."""
    from stages.phase_d2.d2_screening_authorization import _subrun_cost

    root = Path(repo_root) / RUNS_ROOT
    rows, unknown = [], []
    for run in sorted(root.glob(f"{governance(rung).run_prefix}*")):
        if run.name == exclude:
            continue
        cost, basis = _subrun_cost(run)
        if cost is None:
            unknown.append(f"{run.name}: {basis}")
            continue
        rows.append({"run_id": run.name, "all_in_usd": cost, "basis": basis})
    if unknown:
        raise D3AuthorizationRefused(
            f"earlier D3 {rung} subruns have no booked cost: {unknown}; close "
            "each one out (cost booked, teardown verified) first")
    return {"subruns": rows,
            "consumed_usd": round(sum(r["all_in_usd"] for r in rows), 4)}


def build_payload(*, grant: Mapping[str, Any], session_commit: str,
                  granted_utc: str, run_id: str, rung: str = "structural",
                  repo_root: str | Path = REPO,
                  live_rate: float | None = None,
                  consumed_usd: float | None = None,
                  money: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The rung's authorization payload, derived from the tree it binds.

    `live_rate`, `consumed_usd` and `money` substitute the provider quote, the
    envelope reading and the live limits -- tests only.
    """
    from stages.phase_d1.d1_authorization import (
        check_the_four_conditions, live_money, live_secure_price,
    )
    from stages.phase_d3 import d3_budget as BUD
    from stages.phase_d3 import d3_design as D

    gov = governance(rung)
    root = Path(repo_root)
    for f in GRANT_MAY_NOT_STATE:
        if f in grant:
            raise D3AuthorizationRefused(
                f"the grant states {f!r}, which this module DERIVES")
    if grant.get("rung") != rung:
        raise D3AuthorizationRefused(
            f"the grant funds rung {grant.get('rung')!r}; this is the {rung} "
            "issuer")
    if not session_commit or len(session_commit) != 40:
        raise D3AuthorizationRefused(f"{session_commit!r} is not a full commit")
    run_id = str(run_id or "").strip()
    if not run_id.startswith(gov.run_prefix):
        raise D3AuthorizationRefused(
            f"run id {run_id!r} must start with {gov.run_prefix!r}: the "
            "envelope nets every subrun under that prefix")
    reg = D.registration(root)
    if grant.get("approves_registration_hash") != reg["registration_hash"]:
        raise D3AuthorizationRefused(
            "the grant approves a different registration than the tree's")
    envelope = grant.get("cumulative_envelope_usd")
    if envelope is None or float(envelope) <= 0:
        raise D3AuthorizationRefused("the grant states no "
                                     "cumulative_envelope_usd")
    money = dict(money if money is not None else live_money(root))
    if D.EXPERIMENT_ID not in money["funds_formal_sessions_of"]:
        raise D3AuthorizationRefused(
            f"{D.EXPERIMENT_ID} is not in funds_formal_sessions_of "
            f"{money['funds_formal_sessions_of']}: adding it is a maintainer "
            "amendment")
    quote = ({"gpu": "NVIDIA L40S", "usd_per_hour": float(live_rate),
              "_quoted": "supplied by the caller (tests only)"}
             if live_rate is not None else live_secure_price())
    priced = reprice_at(quote["usd_per_hour"], root, rung)
    spent = (consumed_by_prior_subruns(root, rung) if consumed_usd is None else
             {"subruns": [], "consumed_usd": round(float(consumed_usd), 4),
              "_source": "supplied by the caller"})
    remaining = round(float(envelope) - spent["consumed_usd"], 4)
    hard = priced["hard_ceiling_usd"]
    from stages.d_series import d2_budget as D2B

    reserve = float(D2B.TEARDOWN_RESERVE_USD)
    required = round(hard + reserve, 4)
    if remaining < required:
        raise D3AuthorizationRefused(
            f"the envelope has ${remaining:.4f} left after "
            f"${spent['consumed_usd']:.4f} of earlier subruns; a session needs "
            f"its ${hard:.4f} hard ceiling plus the ${reserve:.2f} teardown "
            f"reserve, ${required:.4f}. A larger envelope is a maintainer "
            "decision")
    failed = check_the_four_conditions(ceiling=hard, money=money)
    if failed:
        raise D3AuthorizationRefused(
            f"at ${quote['usd_per_hour']:.4f}/h the D3 {rung} session is not "
            "fundable:\n  - " + "\n  - ".join(failed))
    if rung == "structural":
        spec = D.registered_treatment_spec(root)
        contract_hash, seeds, n_probes = spec.spec_hash, (), 0
    else:
        from stages.phase_d3 import d3_probes as P

        contract = P.session_contract(root)
        contract_hash = P.contract_hash(contract)
        seeds, n_probes = tuple(contract["seeds"]), int(contract["n_probes"])
    closure = current_executable(root, rung)
    cls = TYPES[rung]
    auth = cls(
        authorization_id=f"autoinit.v1.{D.EXPERIMENT_ID}.{rung}",
        granted_utc=granted_utc,
        granted_by=str(grant.get("granted_by") or "")[:4000],
        plan_id=f"autoinit.v1.{D.EXPERIMENT_ID}.{rung}",
        plan_hash=reg["registration_hash"],
        expected_usd=priced["expected_usd"], hard_cap_usd=hard,
        per_launch_hard_usd=hard, authorized_stages=AUTHORIZED_STAGES,
        stage_conditions=dict(gov.stage_conditions),
        scope_note=str(grant.get("covers") or "")[:4000],
        authorized_session_commit=session_commit,
        harness_source_digest=closure["digest"],
        harness_source_files=tuple(f["path"] for f in closure["files"]),
        provenance_commit=session_commit,
        registration_hash=reg["registration_hash"],
        contract_hash=contract_hash, run_id=run_id, seeds=seeds,
        n_probes=n_probes,
        pricing={k: priced[k] for k in (
            "hard_ceiling_minutes", "expected_minutes", "_basis",
            "accepted_at_price_per_hour")},
        money={**money, "derived_session": priced, "live_quote": quote,
               "cumulative_envelope_usd": float(envelope),
               "prior_subruns": spent, "envelope_remaining_usd": remaining,
               "teardown_reserve_usd": reserve,
               "envelope_required_for_this_session_usd": required},
        rate_usd_per_hour=quote["usd_per_hour"],
        hard_runtime_minutes=priced["hard_ceiling_minutes"],
        one_use=("ONE grant issuance, ONE launcher session; a replacement "
                 "attempt is a new authorization netted against the "
                 "cumulative envelope"),
        action_policy=gov.policy)
    payload = auth.as_dict()
    payload["authorizes"] = (
        "ONE D3 STRUCTURAL session: q2's fixed path from the teacher with "
        "DEPTH under the declared teacher reference, its evidence and its "
        "product. No training, no probes, no search." if rung == "structural"
        else "ONE D3 PROBES session: three recovery probes from the D3 "
             "initial checkpoint at the D1 confirmation seeds, stopping at "
             "preservation; the exploratory comparison off-pod. No retraining "
             "of q2/B, no screening seeds, no promotion.")
    payload["harness"] = {"n_files": closure["n_files"],
                          "entry_points": list(closure["entry_points"]),
                          "digest": closure["digest"]}
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload


__all__ = ["D3AuthorizationRefused", "D3ProbesAuthorization",
           "D3StructuralAuthorization", "RUNG_GOVERNANCE", "TYPES",
           "build_payload", "consumed_by_prior_subruns", "current_executable",
           "governance", "reprice_at", "session_ceiling"]
