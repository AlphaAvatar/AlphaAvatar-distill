"""D2's ONE-USE behavioural SCREENING authorization. Its own type and schema.

The grant says what the maintainer decided -- that the screening rung of the
approved preregistration may run, under a stated cumulative envelope. Every
other field is DERIVED from the tree the authorization binds and refused if
the grant states it: the ceiling (re-priced at the live rate from
`d2_budget.py`'s screening step), the location-free session contract hash, the
seeds and probe count, the battery, the harness.

WHAT IT PERMITS: recovery training of the eight frozen screening probes and the
mechanical one-or-none selection, computed off the pod. WHAT IT MAY NEVER
REACH, by type rather than by prose: a beam search, a confirmation, a
re-selection of the committed field, retraining or regenerating B or q2, a
promotion, or an automatic follow-on start.

The money is D1's four conditions (imported) plus the funding scope, and the
CUMULATIVE ENVELOPE: a replacement attempt after a pre-science failure is a new
one-use authorization, issued only while what the envelope has left after every
earlier screening subrun's booked cost still holds a full session's hard
ceiling PLUS the teardown reserve; an earlier attempt whose cost is not booked
or whose teardown is not verified refuses the issuance (AGENTS.md P12.1).
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

SCHEMA = "aadistill.phase_d2.screening_authorization/v1"
RUNS_ROOT = "logs/stages/stage-1/phase_d2/runs"
RUN_PREFIX = "d2_screening_"

D2_SCREENING_POLICY = ActionPolicy(
    policy_id="phase_d2_screening",
    allowed=frozenset({"recovery_training", "behavioural_screening"}),
    wire_claims={
        "allows_recovery_training": "recovery_training",
        "authorizes_d2_screening": "behavioural_screening",
        "allows_beam_search": "beam_search",
        "authorizes_d2_confirmation": "d2_confirmation",
        "authorizes_candidate_reselection": "candidate_reselection",
        "authorizes_reference_retraining": "reference_retraining",
        "authorizes_promotion": "promotion",
        "automatic_followon_start": "automatic_followon_start",
    },
    wire_schema=SCHEMA,
    plan_hash_key="plan_hash",
    enforcement=(
        "the launcher loads this artifact, refuses a planned hard threshold "
        "above hard_cap_usd, a harness whose derived closure differs in "
        "membership or digest, a preregistration other than the bound one and "
        "a contract other than the bound one; the driver re-derives the "
        "contract on the pod and refuses a mismatch before training"),
    refusal_notes={
        "beam_search": "the D2 search is COMPLETE; this rung measures its field",
        "d2_confirmation": ("the shared confirmation is a separate grant that "
                            "names the candidate screening advanced"),
        "candidate_reselection": ("the field is the search's committed Top-4, "
                                  "read from its selection record"),
        "reference_retraining": ("B and q2 are IMPORTED units of the shared "
                                 "protocol; nothing about them is retrained, "
                                 "regenerated or re-evaluated"),
        "promotion": "screening nominates for the shared confirmation, at most",
        "automatic_followon_start": ("the advancing candidate is reviewed into "
                                     "a confirmation grant that names it"),
    },
)

AUTHORIZED_STAGES = ("A", "B", "C", "D")
STAGE_CONDITIONS = {
    "A": ("setup on an admitted host; the relay-staged arms verified at all "
          "four identities; the contract re-derived and equal to the bound "
          "hash; the training runtime equal to the protocol's"),
    "B": ("the eight probes: frozen recipe from each arm's verified "
          "checkpoint at its seed, packaged, generated through the unchanged "
          "generator (token-id sidecar), admitted against the shared rung's "
          "evaluation protocol, scored by the pinned scorer"),
    "C": ("per probe, after scoring: the rollout rescoring diagnostic on the "
          "candidate's own rollouts, failure-isolated"),
    "D": ("stop at preservation: every probe's evidence and trained "
          "checkpoint off-pod; the ranking and selection are computed off-pod "
          "at $0"),
}
GRANT_MAY_NOT_STATE = (
    "authorization_id", "plan_id", "plan_hash", "authorized_stages",
    "stage_conditions", "expected_usd", "per_launch_hard_usd", "hard_cap_usd",
    "contract_hash", "seeds", "n_probes", "battery_content_id",
)

ENTRY_POINTS: tuple[str, ...] = (
    "scripts/stages/stage-1/phase_d2/autoinit_d2_screening_launch.py",
    "scripts/stages/stage-1/phase_d2/autoinit_d2_screening_driver.py",
    "scripts/stages/stage-1/phase_d2/rescore_d2_probe.py",
    "scripts/stages/stage-1/phase_d2/d2_screening.py",
    "scripts/stages/stage-1/phase_d2/d2_screening_authorization.py",
    "scripts/stages/stage-1/phase_d1/score_d1_screening.py",
    "scripts/shared/evaluation/uncapped_eval_token_ids.py",
    "scripts/shared/evaluation/uncapped_eval.py",
    "scripts/shared/training/train_stage3.py",
    "scripts/shared/pod/autoinit_engine_probe.py",
    "scripts/shared/pod/collect_artifacts.py",
)
DECLARED_INPUTS: tuple[str, ...] = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "configs/stages/stage-1/phase_c1/authorization.json",
    "configs/stages/stage-1/phase_d2/d2_screening_artifacts.json",
    "configs/stages/stage-1/phase_d2/d2_screening_artifacts_failed.json",
    "configs/stages/stage-3/recipes/recovery.json",
    "configs/stages/stage-3/e1/e1_r0860k_sa_pca.json",
)
SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/stages/stage-1")


class D2ScreeningRefused(AuthorizationError):
    """A screening condition does not hold. An `AuthorizationError`, so the
    runner reports a refusal rather than an unhandled error."""


def current_executable(repo_root: str | Path = REPO) -> dict[str, Any]:
    from aadistill.governance.closure import ClosureError, derive

    try:
        return derive(repo_root, "phase_d2_screening", ENTRY_POINTS,
                      DECLARED_INPUTS, roots=SOURCE_ROOTS)
    except ClosureError as exc:
        raise D2ScreeningRefused(
            f"cannot derive the D2 screening executable set: {exc}") from exc


@dataclass(frozen=True)
class D2ScreeningAuthorization(SpendAuthorization):
    """Permits exactly one D2 screening session on the committed field."""

    preregistration_hash: str = ""
    shared_protocol_hash: str = ""
    contract_hash: str = ""
    run_id: str = ""
    seeds: tuple[int, ...] = ()
    n_probes: int = 0
    battery_content_id: str = ""
    selection_sha256: str = ""
    pricing: Mapping[str, Any] = field(default_factory=dict)
    money: Mapping[str, Any] = field(default_factory=dict)
    rate_usd_per_hour: float | None = None
    hard_runtime_minutes: float | None = None
    one_use: str = ""
    allows_recovery_training: bool = True
    authorizes_d2_screening: bool = True
    allows_beam_search: bool = False
    authorizes_d2_confirmation: bool = False
    authorizes_candidate_reselection: bool = False
    authorizes_reference_retraining: bool = False
    authorizes_promotion: bool = False
    automatic_followon_start: bool = False

    def require_harness(self, repo_root: str | Path = REPO) -> dict[str, Any]:
        observed = current_executable(repo_root)
        if not self.harness_source_digest:
            raise D2ScreeningRefused("no harness_source_digest is bound")
        recorded = set(self.harness_source_files)
        live = {f["path"] for f in observed["files"]}
        if recorded and recorded != live:
            raise D2ScreeningRefused(
                "the screening executable closure MEMBERSHIP changed since the "
                f"grant: added {sorted(live - recorded)[:4]}, removed "
                f"{sorted(recorded - live)[:4]}")
        if observed["digest"] != self.harness_source_digest:
            raise D2ScreeningRefused(
                f"the screening harness digests to {observed['digest'][:16]}, "
                f"the grant bound {self.harness_source_digest[:16]}")
        return observed

    def require_plan(self, plan_hash: str) -> None:
        if not plan_hash or self.plan_hash != plan_hash:
            raise D2ScreeningRefused(
                f"this authorization binds preregistration "
                f"{self.plan_hash[:16]} and the session declares "
                f"{str(plan_hash)[:16]}")

    def require_session_commit(self, commit: str) -> None:
        if not self.authorized_session_commit or \
                commit != self.authorized_session_commit:
            raise D2ScreeningRefused(
                f"this authorization binds commit "
                f"{self.authorized_session_commit[:12]}, the session declares "
                f"{str(commit)[:12]}")

    def require_run_id(self, run_id: str) -> None:
        if not self.run_id or run_id != self.run_id:
            raise D2ScreeningRefused(
                f"this authorization was issued for run {self.run_id!r}, the "
                f"session declares {run_id!r}")

    def require_contract(self, contract_hash: str) -> None:
        if not self.contract_hash or contract_hash != self.contract_hash:
            raise D2ScreeningRefused(
                f"the session's contract hashes to {str(contract_hash)[:16]} "
                f"and this authorization binds {self.contract_hash[:16]}: the "
                "field, seeds, recipe, battery, evaluation identity or "
                "comparator is not what was authorized")

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        payload.update({
            "preregistration_hash": self.preregistration_hash,
            "shared_protocol_hash": self.shared_protocol_hash,
            "contract_hash": self.contract_hash,
            "run_id": self.run_id,
            "seeds": list(self.seeds),
            "n_probes": int(self.n_probes),
            "battery_content_id": self.battery_content_id,
            "selection_sha256": self.selection_sha256,
            "pricing": dict(self.pricing),
            "money": dict(self.money),
            "rate_usd_per_hour": self.rate_usd_per_hour,
            "hard_runtime_minutes": self.hard_runtime_minutes,
            "one_use": self.one_use,
        })
        for wire in self.action_policy.wire_claims:
            payload[wire] = bool(getattr(self, wire))
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "D2ScreeningAuthorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = {k: v for k, v in raw.items() if k != "authorization_sha256"}
        if not stated or stated != sha256_json(check):
            raise D2ScreeningRefused(
                f"{path} does not match its own authorization_sha256")
        if raw.get("schema") != SCHEMA:
            raise D2ScreeningRefused(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}")
        missing = [f for f in (
            "authorization_id", "granted_utc", "hard_cap_usd",
            "authorized_session_commit", "preregistration_hash",
            "shared_protocol_hash", "contract_hash", "run_id", "seeds",
            "n_probes", "authorized_stages", "harness_source_digest",
            "hard_runtime_minutes") if not raw.get(f)]
        if missing:
            raise D2ScreeningRefused(f"{path} omits {missing}")
        D2_SCREENING_POLICY.check_claims(raw, where=str(path))
        if not raw.get("allows_recovery_training") or \
                not raw.get("authorizes_d2_screening"):
            raise D2ScreeningRefused(f"{path} does not authorize D2 screening")
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
            shared_protocol_hash=raw["shared_protocol_hash"],
            contract_hash=raw["contract_hash"], run_id=raw["run_id"],
            seeds=tuple(int(s) for s in raw["seeds"]),
            n_probes=int(raw["n_probes"]),
            battery_content_id=raw.get("battery_content_id", ""),
            selection_sha256=raw.get("selection_sha256", ""),
            pricing=dict(raw.get("pricing") or {}),
            money=dict(raw.get("money") or {}),
            rate_usd_per_hour=raw.get("rate_usd_per_hour"),
            hard_runtime_minutes=float(raw["hard_runtime_minutes"]),
            one_use=raw.get("one_use", ""),
            action_policy=D2_SCREENING_POLICY)


def session_ceiling(repo_root: str | Path = REPO) -> dict[str, Any]:
    """The screening price in force, from `d2_budget.py`. No second formula."""
    from stages.d_series import d2_budget as BUD

    budget = BUD.build()
    step = budget["steps"]["d2_screening"]
    design = json.loads((Path(repo_root) / BUD.D1_DESIGN).read_text())
    cell = design["budget"]["chain"]["sessions"]["screening"]
    return {"hard_ceiling_usd": float(step["hard"]["all_in_usd"]),
            "expected_usd": float(step["expected"]["all_in_usd"]),
            "hard_ceiling_minutes": float(step["hard"]["minutes"]),
            "expected_minutes": float(step["expected"]["minutes"]),
            "container_disk_usd": float(step["hard"]["disk_usd"]),
            "container_disk_gb": int(cell["container_disk_gb"]),
            "session_overhead_minutes": float(cell["session_overhead_minutes"]),
            "price_per_hour": float(budget["rate_usd_per_hour_basis"]),
            "n_probes": int(step["n_probes"]),
            "arm_staging": step.get("arm_staging"),
            "rescoring": step.get("rescoring"),
            "_basis": f"{BUD.RECORD} :: steps.d2_screening"}


def reprice_at(rate_usd_per_hour: float,
               repo_root: str | Path = REPO) -> dict[str, Any]:
    """The SAME accepted minute bound, costed at the live rate."""
    accepted = session_ceiling(repo_root)
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


def _subrun_cost(run: Path) -> tuple[float | None, str]:
    """(booked all-in cost, basis) of ONE earlier screening subrun, or
    (None, why it is unknown).

    FAIL CLOSED. Exactly two things count as known:

    * a closeout that books `money.all_in_usd` AND states
      `provider_teardown_verified: true` -- a cost with the resource possibly
      still billing is not a closed cost;
    * a record that no provider resource was ever created: a closeout saying
      so, or a session record with `provider_resource_created: false` and no
      pod id on it or on any host draw (a dry run, a pre-provider refusal).

    Everything else is UNKNOWN: an issued authorization with no session record
    (it may have been launched from elsewhere), a session record with a pod id
    and no closeout, an unreadable record, a closeout without a booked cost or
    without verified teardown. A potentially paid attempt is never read as $0.
    """
    closeout = run / "closeout" / "outcome.json"
    session = run / "runtime" / "session.json"
    if closeout.is_file():
        try:
            doc = json.loads(closeout.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"closeout unreadable ({type(exc).__name__})"
        cost = (doc.get("money") or {}).get("all_in_usd")
        if doc.get("provider_resource_created") is False and not cost:
            return 0.0, "closeout: no provider resource was created"
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            return None, "the closeout books no money.all_in_usd"
        if doc.get("provider_teardown_verified") is not True:
            return None, ("the closeout does not state "
                          "provider_teardown_verified: true")
        return float(cost), "closeout money.all_in_usd"
    if session.is_file():
        try:
            ev = json.loads(session.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"session record unreadable ({type(exc).__name__})"
        pods = [ev.get("pod_id"), *((d or {}).get("pod_id")
                                    for d in ev.get("draws") or [])]
        if ev.get("provider_resource_created") is False and not any(pods):
            return 0.0, "session record: nothing was created at the provider"
        return None, "a session record and no closeout"
    if run.is_dir() and any(run.rglob("*")):
        return None, ("an issued authorization with no session record and no "
                      "closeout")
    return 0.0, "an empty run directory"


def consumed_by_prior_subruns(repo_root: str | Path = REPO) -> dict[str, Any]:
    """Every earlier screening subrun's booked cost (`_subrun_cost`).

    Fails closed: one UNKNOWN anywhere refuses the whole issuance, because the
    envelope's remainder is then unknown.
    """
    root = Path(repo_root) / RUNS_ROOT
    rows, unknown = [], []
    for run in sorted(root.glob(f"{RUN_PREFIX}*")):
        cost, basis = _subrun_cost(run)
        if cost is None:
            unknown.append(f"{run.name}: {basis}")
            continue
        rows.append({"run_id": run.name, "all_in_usd": cost, "basis": basis})
    if unknown:
        raise D2ScreeningRefused(
            f"earlier screening subruns have no booked cost: {unknown}. The "
            "envelope's remainder is unknown and nothing is issued over it; "
            "close each one out (cost booked, teardown verified) first")
    return {"subruns": rows,
            "consumed_usd": round(sum(r["all_in_usd"] for r in rows), 4)}


def build_payload(*, grant: Mapping[str, Any], session_commit: str,
                  granted_utc: str, run_id: str,
                  repo_root: str | Path = REPO,
                  live_rate: float | None = None,
                  consumed_usd: float | None = None) -> dict[str, Any]:
    """The screening authorization payload, derived from the tree it binds."""
    from stages.phase_d1.d1_authorization import (
        check_the_four_conditions, live_money, live_secure_price,
    )
    from stages.d_series import d2_budget as BUD
    from stages.phase_d2 import d2_screening as SC

    root = Path(repo_root)
    for f in GRANT_MAY_NOT_STATE:
        if f in grant:
            raise D2ScreeningRefused(
                f"the grant states {f!r}, which this module DERIVES")
    if grant.get("rung") != "screening":
        raise D2ScreeningRefused(
            f"the grant funds rung {grant.get('rung')!r}; this is the "
            "screening issuer")
    if not session_commit or len(session_commit) != 40:
        raise D2ScreeningRefused(f"{session_commit!r} is not a full commit id")
    run_id = str(run_id or "").strip()
    if not run_id.startswith(RUN_PREFIX):
        raise D2ScreeningRefused(
            f"run id {run_id!r} must start with {RUN_PREFIX!r}: the envelope "
            "nets every subrun under that prefix")
    prereg = SC.preregistration(root)
    if grant.get("approves_preregistration_hash") != prereg[
            "preregistration_hash"]:
        raise D2ScreeningRefused(
            "the grant approves a different preregistration than the tree's")
    envelope = grant.get("cumulative_envelope_usd")
    if envelope is None or float(envelope) <= 0:
        raise D2ScreeningRefused(
            "the grant states no cumulative_envelope_usd: a screening grant "
            "funds the rung's attempts cumulatively, and an unbounded one is "
            "not a grant")

    money = live_money(root)
    if SC.EXPERIMENT_ID not in money["funds_formal_sessions_of"]:
        raise D2ScreeningRefused(
            f"{SC.EXPERIMENT_ID} is not in funds_formal_sessions_of")
    quote = ({"gpu": "NVIDIA L40S", "usd_per_hour": float(live_rate),
              "_quoted": "supplied by the caller (tests only)"}
             if live_rate is not None else live_secure_price())
    priced = reprice_at(quote["usd_per_hour"], root)
    spent = (consumed_by_prior_subruns(root) if consumed_usd is None else
             {"subruns": [], "consumed_usd": round(float(consumed_usd), 4),
              "_source": "supplied by the caller"})
    remaining = round(float(envelope) - spent["consumed_usd"], 4)
    hard = priced["hard_ceiling_usd"]
    #: THE ENVELOPE MUST HOLD THE SESSION AND ITS TEARDOWN. A ceiling that
    #: fits only without the reserve leaves a session that reaches its hard
    #: stop owing a teardown the grant cannot pay for.
    reserve = float(BUD.TEARDOWN_RESERVE_USD)
    required = round(hard + reserve, 4)
    if remaining < required:
        raise D2ScreeningRefused(
            f"the envelope has ${remaining:.4f} left after "
            f"${spent['consumed_usd']:.4f} of earlier subruns; a session needs "
            f"its derived ${hard:.4f} hard ceiling plus the ${reserve:.2f} "
            f"teardown reserve, ${required:.4f}. The probe schedule does not "
            "shrink to fit; a larger envelope is a maintainer decision")
    failed = check_the_four_conditions(ceiling=hard, money=money)
    if failed:
        raise D2ScreeningRefused(
            f"at ${quote['usd_per_hour']:.4f}/h the screening session is not "
            "fundable:\n  - " + "\n  - ".join(failed))

    contract = SC.session_contract(root)
    closure = current_executable(root)
    hard_minutes = priced["hard_ceiling_minutes"]
    auth = D2ScreeningAuthorization(
        authorization_id=f"autoinit.v1.{SC.EXPERIMENT_ID}.screening",
        granted_utc=granted_utc,
        granted_by=str(grant.get("granted_by") or "")[:4000],
        plan_id=f"autoinit.v1.{SC.EXPERIMENT_ID}.screening",
        plan_hash=prereg["preregistration_hash"],
        expected_usd=priced["expected_usd"], hard_cap_usd=hard,
        per_launch_hard_usd=hard,
        authorized_stages=AUTHORIZED_STAGES, stage_conditions=STAGE_CONDITIONS,
        scope_note=str(grant.get("covers") or "")[:4000],
        authorized_session_commit=session_commit,
        harness_source_digest=closure["digest"],
        harness_source_files=tuple(f["path"] for f in closure["files"]),
        provenance_commit=session_commit,
        preregistration_hash=prereg["preregistration_hash"],
        shared_protocol_hash=contract["shared_protocol"]["protocol_hash"],
        contract_hash=SC.contract_hash(contract), run_id=run_id,
        seeds=tuple(contract["seeds"]), n_probes=contract["n_probes"],
        battery_content_id=contract["battery"]["family_content_id"],
        selection_sha256=contract["field"]["selection_sha256"],
        pricing={k: priced[k] for k in (
            "hard_ceiling_minutes", "expected_minutes", "arm_staging",
            "rescoring", "_basis", "accepted_at_price_per_hour")},
        money={**money, "derived_session": priced, "live_quote": quote,
               "cumulative_envelope_usd": float(envelope),
               "prior_subruns": spent, "envelope_remaining_usd": remaining,
               "teardown_reserve_usd": reserve,
               "envelope_required_for_this_session_usd": required},
        rate_usd_per_hour=quote["usd_per_hour"],
        hard_runtime_minutes=hard_minutes,
        one_use=("ONE grant issuance, ONE launcher session; a replacement "
                 "attempt is a new authorization netted against the "
                 "cumulative envelope"),
        action_policy=D2_SCREENING_POLICY)
    payload = auth.as_dict()
    payload["authorizes"] = (
        "ONE D2 behavioural SCREENING session: the eight frozen probes and "
        "their diagnostics, stopping at preservation. Not a confirmation, not "
        "B/q2 retraining, not D3, not a promotion.")
    payload["harness"] = {"n_files": closure["n_files"],
                          "entry_points": list(closure["entry_points"]),
                          "digest": closure["digest"]}
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload


__all__ = ["D2ScreeningAuthorization", "D2ScreeningRefused",
           "D2_SCREENING_POLICY", "build_payload", "consumed_by_prior_subruns",
           "current_executable", "reprice_at", "session_ceiling"]
