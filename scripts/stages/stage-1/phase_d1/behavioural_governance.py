"""What a D1 BEHAVIOURAL authorization may permit, and what it may not.

A separate type from `D1Authorization`, with its own schema, because the two
permit disjoint things and neither may stand in for the other. The search's
artifact allows a beam and denies training; this one allows recovery training
and denies a beam. Delegating to the parent would refuse this artifact by
construction -- a sibling loader in this repository already hit exactly that by
inheriting a contract instead of satisfying it.

TWO RUNGS, TWO ARTIFACTS. Screening and confirmation are separately authorized
even though they share this type, because the confirmation rung cannot be bound
until screening has named an advancing candidate: its arms are *that* candidate
and B, and an artifact issued before the candidate exists would have to name it
or leave it open. `rung` is therefore a bound field and `advancing_candidate` is
required on a confirmation artifact and forbidden on a screening one.

WHAT THIS SESSION TRAINS. Ten probes for screening, six for confirmation, each
one a recovery run under the frozen recipe from one of the arms' checkpoints.
`allows_recovery_training` is True here and False on every other D1 artifact,
so an artifact that could reach the trainer is distinguishable from one that
could not by its type rather than by reading its prose.

WHAT IT MAY NEVER REACH, enforced by the policy rather than by a chain of
`if`s: another beam search, a re-selection of the candidate field, a
re-measurement of B's state evaluation, a promotion decision outside the frozen
decision rule, or an automatic follow-on start. The D1 search is complete and
its quality-order Top-4 frozen; this session measures that field and nothing
else.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aadistill.governance.authorization import (
    ActionPolicy,
    AuthorizationError,
    SpendAuthorization,
)
from aadistill.infrastructure.manifest import sha256_json

SCHEMA = "aadistill.phase_d1.behavioural_authorization/v1"

#: WHAT A D1 BEHAVIOURAL AUTHORIZATION MAY EXPRESS. Absence is denial.
#:
#: `recovery_training` and `behavioural_selection` are allowed and are the only
#: two. Everything the D1 search was permitted is denied here, and everything
#: this is permitted was denied there, so the two artifacts cannot be
#: substituted for one another even by a caller that loaded the wrong module.
D1_BEHAVIOURAL_POLICY = ActionPolicy(
    policy_id="phase_d1_behavioural",
    allowed=frozenset({"recovery_training", "behavioural_selection"}),
    wire_claims={
        "allows_recovery_training": "recovery_training",
        "authorizes_behavioural_selection": "behavioural_selection",
        "allows_beam_search": "beam_search",
        "authorizes_d1_search": "d1_search",
        "authorizes_candidate_reselection": "candidate_reselection",
        "authorizes_incumbent_remeasurement": "incumbent_remeasurement",
        "automatic_followon_start": "automatic_followon_start",
    },
    wire_schema=SCHEMA,
    plan_hash_key="plan_hash",
    enforcement=(
        "the launcher loads this artifact, refuses a pod whose priced hard "
        "threshold exceeds the bound all-in ceiling, refuses a harness whose "
        "derived closure differs from the bound one in membership or digest, "
        "refuses a plan hash that is not this design revision, refuses a rung "
        "whose probe schedule disagrees with the design's declared count, and "
        "has no code path to a beam search, a re-selection or a promotion "
        "outside the frozen decision rule"),
    refusal_notes={
        "beam_search": (
            "the D1 search is COMPLETE and its 92 expansions are a consumed "
            "measurement. This session measures the field that search "
            "committed; it cannot produce another one."),
        "d1_search": (
            "same artifact family, different session. A behavioural grant "
            "cannot fund a search even at a price that would fit."),
        "candidate_reselection": (
            "the Top-4 field is the maintainer's retention decision of "
            "2026-10-07, read from that record rather than recomputed. A "
            "session that could re-select would be choosing its own arms."),
        "incumbent_remeasurement": (
            "B's state evaluation stands. This session trains fresh probes "
            "FROM B's checkpoint as the matched arm of a paired comparison; "
            "it does not re-measure the result B was promoted on."),
        "automatic_followon_start": (
            "screening's advancing candidate is reviewed into a confirmation "
            "artifact that names it. Nothing chains automatically, because an "
            "artifact issued before the candidate existed could not have "
            "named it."),
    },
)


class D1BehaviouralRefused(AuthorizationError):
    """A D1 behavioural authorization does not permit what was asked of it."""


@dataclass(frozen=True)
class D1BehaviouralAuthorization(SpendAuthorization):
    """Permits exactly one D1 behavioural RUNG on one set of arms.

    Structurally a `SpendAuthorization` -- same commit binding, same derived
    harness rule, same hash-of-itself -- and a different type with a different
    schema, so it cannot stand in for the D1 search's artifact nor that one for
    it.

    The money fields are kept DISTINCT rather than folded into one: the
    provider bills container disk separately from the GPU, and the launcher
    derives its deadline from the bound rate rather than from a live quote, so
    an artifact re-derived at a different price cannot inherit the old dollar
    window.
    """

    #: THE POLICY, as the INSTANCE field the base actually reads.
    #: `action_policy` is a field defaulting to `DENY_ALL`, so a class-level
    #: `POLICY = ...` is decoration: the base's `as_dict`, `allows` and
    #: `refuse` all consult `self.action_policy`, and an artifact that only
    #: set the class attribute serialized under `deny_all` -- which refuses
    #: itself, with a message about a missing wire schema that names neither
    #: the policy nor the type. Defaulted here so a caller cannot forget it,
    #: and `POLICY` is kept as the module-level name callers reference.
    action_policy: ActionPolicy = D1_BEHAVIOURAL_POLICY

    #: WHICH RUNG, and it is load-bearing. The two rungs have different probe
    #: counts, different seeds, different batteries and different arms, and an
    #: artifact that did not say which would authorize whichever the session
    #: happened to run.
    rung: str = ""
    design_hash: str = ""
    #: The contract the session asserts on the pod, hashed. Binds the arms,
    #: the battery role, the recipe, the seeds and the probe schedule in one
    #: value, so a session running a different field is refused before it
    #: trains.
    contract_hash: str = ""
    battery_role: str = ""
    battery_content_id: str = ""
    recipe_id: str = ""
    n_probes: int = 0
    seeds: tuple[int, ...] = ()
    #: REQUIRED on a confirmation artifact, FORBIDDEN on a screening one.
    #: Screening does not know its advancing candidate yet -- that is what it
    #: is for -- and a confirmation artifact that left it open would permit
    #: confirming whichever candidate the session chose.
    advancing_candidate: str = ""
    run_id: str = ""
    money: Mapping[str, Any] = field(default_factory=dict)
    rate_usd_per_hour: float | None = None
    hard_runtime_minutes: float | None = None
    gpu_hard_usd: float | None = None
    disk_hard_usd: float | None = None
    all_in_hard_usd: float | None = None
    one_use: str = ""

    @property
    def allows_recovery_training(self) -> bool:
        return True

    @property
    def authorizes_behavioural_selection(self) -> bool:
        return True

    @property
    def allows_beam_search(self) -> bool:
        return False

    @property
    def authorizes_d1_search(self) -> bool:
        return False

    @property
    def authorizes_candidate_reselection(self) -> bool:
        return False

    @property
    def authorizes_incumbent_remeasurement(self) -> bool:
        return False

    @property
    def automatic_followon_start(self) -> bool:
        return False

    def require_rung(self, rung: str) -> None:
        """The artifact permits THIS rung. Fails closed on a mismatch."""
        if self.rung != rung:
            raise D1BehaviouralRefused(
                f"this artifact authorizes the {self.rung!r} rung and the "
                f"session is running {rung!r}. The two have different probe "
                "counts, seeds, batteries and arms; neither grant covers the "
                "other.")

    def require_contract(self, contract_hash: str) -> None:
        """The session's derived contract is the bound one.

        One comparison covering the arms, the battery, the recipe, the seeds
        and the schedule -- so a session that changed any of them is refused
        before it trains rather than after, and the refusal names the
        quantity rather than a symptom of it.
        """
        if not self.contract_hash:
            raise D1BehaviouralRefused(
                "this artifact binds no contract hash, so nothing checks that "
                "the session measures the field it was issued for")
        if self.contract_hash != contract_hash:
            raise D1BehaviouralRefused(
                f"the session's contract hashes to {contract_hash[:16]} and "
                f"this artifact binds {self.contract_hash[:16]}. Something in "
                "the arms, the battery, the recipe, the seeds or the probe "
                "schedule is not what was authorized.")

    def require_run_id(self, run_id: str) -> None:
        """The launch must be the run this artifact was issued for.

        The contract hash is not a function of the run id, so this is the one
        place a run-id mismatch is caught at all -- and it is caught at `$0`,
        where the search's equivalent check is caught before a provider is
        contacted.
        """
        if not self.run_id:
            raise D1BehaviouralRefused(
                "this authorization binds no run_id, so it cannot say which "
                "attempt it was issued for; a one-use artifact without a run "
                "identity is reusable by accident")
        if run_id != self.run_id:
            raise D1BehaviouralRefused(
                f"this authorization was issued for run {self.run_id!r} and "
                f"the session declares {run_id!r}. One grant, one issuance, "
                "one launcher session.")

    def require_advancing_candidate(self) -> str:
        """The candidate a confirmation rung is bound to. Screening has none."""
        if self.rung != "confirmation":
            raise D1BehaviouralRefused(
                f"the {self.rung!r} rung has no advancing candidate; only "
                "confirmation does, and screening is what produces it")
        if not self.advancing_candidate:
            raise D1BehaviouralRefused(
                "a confirmation artifact must NAME the advancing candidate. "
                "One that left it open would permit confirming whichever "
                "candidate the session chose, which is the selection it is "
                "supposed to be independent of.")
        return self.advancing_candidate

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        payload["rung"] = self.rung
        payload["design_hash"] = self.design_hash
        payload["contract_hash"] = self.contract_hash
        payload["battery_role"] = self.battery_role
        payload["battery_content_id"] = self.battery_content_id
        payload["recipe_id"] = self.recipe_id
        payload["n_probes"] = self.n_probes
        payload["seeds"] = list(self.seeds)
        payload["run_id"] = self.run_id
        payload["money"] = dict(self.money)
        payload["rate_usd_per_hour"] = self.rate_usd_per_hour
        payload["hard_runtime_minutes"] = self.hard_runtime_minutes
        payload["gpu_hard_usd"] = self.gpu_hard_usd
        payload["disk_hard_usd"] = self.disk_hard_usd
        payload["all_in_hard_usd"] = self.all_in_hard_usd
        payload["one_use"] = self.one_use
        if self.rung == "confirmation":
            payload["advancing_candidate"] = self.advancing_candidate
        #: Every claim, written out, so a reader of the artifact sees the
        #: permissions rather than having to know the type.
        for wire, _action in self.action_policy.wire_claims.items():
            payload[wire] = bool(getattr(self, wire))
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "D1BehaviouralAuthorization":
        """Read an issued behavioural authorization back, self-hash verified.

        An artifact the issuer can write and the loader cannot read fails on
        the pod, after the money is committed -- so the loader exists before
        the issuer does, and the issuer's last step is reading its own output
        through this.
        """
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if not stated or stated != sha256_json(check):
            raise D1BehaviouralRefused(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        if raw.get("schema") != SCHEMA:
            raise D1BehaviouralRefused(
                f"{path} declares schema {raw.get('schema')!r}, not "
                f"{SCHEMA!r}. Another session's grant measures a different "
                "harness and carries a ceiling derived for different work; it "
                "cannot authorize a D1 behavioural rung.")
        missing = [f for f in (
            "authorization_id", "granted_utc", "hard_cap_usd",
            "authorized_session_commit", "harness_source_digest",
            "rung", "design_hash", "contract_hash", "battery_role",
            "battery_content_id", "recipe_id", "n_probes", "seeds",
            "run_id") if not raw.get(f)]
        if missing:
            raise D1BehaviouralRefused(f"{path} omits {missing}")
        D1_BEHAVIOURAL_POLICY.check_claims(raw, where=str(path))
        if not raw.get("allows_recovery_training") or not raw.get(
                "authorizes_behavioural_selection"):
            raise D1BehaviouralRefused(
                f"{path} does not authorize a behavioural rung")
        rung = str(raw["rung"])
        if rung not in ("screening", "confirmation"):
            raise D1BehaviouralRefused(
                f"{path} names rung {rung!r}; there are two rungs and this is "
                "neither")
        advancing = str(raw.get("advancing_candidate") or "")
        if rung == "screening" and advancing:
            raise D1BehaviouralRefused(
                f"{path} is a SCREENING artifact naming an advancing "
                f"candidate {advancing!r}. Screening is what produces the "
                "candidate; an artifact that pre-named one would be a "
                "selection made before the measurement.")
        if rung == "confirmation" and not advancing:
            raise D1BehaviouralRefused(
                f"{path} is a CONFIRMATION artifact naming no advancing "
                "candidate; one that left it open would permit confirming "
                "whichever candidate the session chose.")
        return cls(
            authorization_id=raw["authorization_id"],
            granted_utc=raw["granted_utc"],
            granted_by=raw.get("granted_by", ""),
            plan_id=raw["plan_id"],
            plan_hash=raw[D1_BEHAVIOURAL_POLICY.plan_hash_key],
            expected_usd=float(raw["expected_usd"]),
            hard_cap_usd=float(raw["hard_cap_usd"]),
            authorized_stages=tuple(raw.get("authorized_stages") or ()),
            stage_conditions=dict(raw.get("stage_conditions") or {}),
            scope_note=raw.get("scope_note", ""),
            authorized_session_commit=raw["authorized_session_commit"],
            harness_source_digest=raw["harness_source_digest"],
            #: No fallback: an artifact that names no harness gets none, and
            #: `require_harness` then refuses.
            harness_source_files=tuple(raw.get("harness_source_files") or ()),
            per_launch_hard_usd=(float(raw["per_launch_hard_usd"])
                                 if raw.get("per_launch_hard_usd") else None),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)),
            action_policy=D1_BEHAVIOURAL_POLICY,
            rung=rung,
            design_hash=raw["design_hash"],
            contract_hash=raw["contract_hash"],
            battery_role=raw["battery_role"],
            battery_content_id=raw["battery_content_id"],
            recipe_id=raw["recipe_id"],
            n_probes=int(raw["n_probes"]),
            seeds=tuple(int(s) for s in raw["seeds"]),
            advancing_candidate=advancing,
            run_id=raw["run_id"],
            money=dict(raw.get("money") or {}),
            rate_usd_per_hour=(float(raw["rate_usd_per_hour"])
                               if raw.get("rate_usd_per_hour") else None),
            hard_runtime_minutes=(float(raw["hard_runtime_minutes"])
                                  if raw.get("hard_runtime_minutes") else None),
            gpu_hard_usd=(float(raw["gpu_hard_usd"])
                          if raw.get("gpu_hard_usd") else None),
            disk_hard_usd=(float(raw["disk_hard_usd"])
                           if raw.get("disk_hard_usd") is not None else None),
            all_in_hard_usd=(float(raw["all_in_hard_usd"])
                             if raw.get("all_in_hard_usd") else None),
            one_use=str(raw.get("one_use", "")))


#: The module-level name for the policy, kept so callers read
#: `behavioural_governance.POLICY` rather than reaching into the dataclass.
POLICY = D1_BEHAVIOURAL_POLICY

__all__ = ["D1_BEHAVIOURAL_POLICY", "POLICY", "D1BehaviouralAuthorization",
           "D1BehaviouralRefused", "SCHEMA"]
