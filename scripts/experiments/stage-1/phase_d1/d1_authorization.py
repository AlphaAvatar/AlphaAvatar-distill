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

from aadistill.governance.authorization import (
    ActionPolicy, AuthorizationError, SpendAuthorization,
)
from aadistill.infrastructure.manifest import sha256_json

REPO = Path(__file__).resolve().parents[4]

SCHEMA = "aadistill.phase_d1.authorization/v1"

#: A FORMAL D1 search is treatment-only. The control arm stays constructible at
#: `$0` for protocol-identity checks and is not issuable for a paid beam.
FORMAL_ARM = "supervised_target"

#: THE DEVICE A FORMAL D1 SEARCH EXECUTES ON, and therefore the device the
#: authorized `config_hash` must be computed with: `device` is a field of
#: `SearchConfig.as_dict()`. One constant, so the issuer, the launcher's $0
#: contract gate and the driver cannot disagree about it -- which they did, at
#: `cpu` / `cpu` / `cuda`.
FORMAL_DEVICE = "cuda"
#: WHAT A D1 SEARCH AUTHORIZATION MAY EXPRESS. Absence is denial.
#:
#: `beam_search` is ALLOWED and is the only one: D1 is the first of these
#: sessions that actually searches. Recovery, behavioural work and a follow-on
#: start are denied by the policy itself, so an artifact claiming one is refused
#: at load by `check_claims` rather than by a hand-written chain of `if`s.
D1_SEARCH_POLICY = ActionPolicy(
    policy_id="phase_d1_search",
    allowed=frozenset({"beam_search", "d1_search"}),
    wire_claims={
        "allows_beam_search": "beam_search",
        "authorizes_d1_search": "d1_search",
        "allows_recovery": "recovery",
        "allows_behavioural": "behavioural",
        "automatic_followon_start": "automatic_followon_start",
    },
    wire_schema=SCHEMA,
    plan_hash_key="plan_hash",
    enforcement=(
        "the launcher loads this artifact, refuses a pod whose priced hard "
        "threshold exceeds hard_cap_usd or per_launch_hard_usd, refuses a harness "
        "whose derived closure differs from the bound one in membership or "
        "digest, refuses a plan hash that is not this design revision, and has no "
        "code path to recovery, behavioural evaluation or a promotion decision"),
    refusal_notes={
        "recovery": ("a D1 SEARCH session trains nothing. Recovery of a selected "
                     "candidate is a separately authorized session and cannot be "
                     "reached from this artifact."),
        "behavioural": ("screening and confirmation are separately authorized "
                        "sessions; neither can be bound until the search has "
                        "committed its candidate set."),
        "automatic_followon_start": (
            "nothing chains off the D1 search. The committed Top-2 is reviewed "
            "before any probe is trained."),
    },
)



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


class D1AuthorizationRefused(AuthorizationError):
    """A condition does not hold. A refusal is the product, not a warning.

    An `AuthorizationError` by inheritance, deliberately: `SessionRunner` catches
    that type around its four authorization calls, so a D1-specific refusal raised
    anywhere in this module is reported as a refusal rather than escaping as an
    unhandled error after a pod may already exist.
    """

# ---------------------------------------------------------------------------
# the D1 executable closure
# ---------------------------------------------------------------------------

#: The entry points a D1 formal search session executes. The closure is DERIVED
#: from these by an import walk -- never a hand-maintained file list, which cannot
#: promise completeness and goes stale on the first edit while still reporting a
#: confident identity for the wrong set.
D1_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/pod/autoinit_d1_launch.py",
    "scripts/pod/autoinit_d1_driver.py",
    "scripts/experiments/stage-1/phase_d1/d1_session.py",
    "scripts/experiments/stage-1/phase_d1/d1_authorization.py",
    "scripts/pod/collect_artifacts.py",
)

#: Files no import edge reaches, whose bytes still decide what runs or what is
#: authorized. Hashed identically to the modules.
#:
#: The DESIGN is deliberately absent and is the one runtime input that cannot be
#: here: `plan_hash` is its hash, so including its bytes would be a fixed point
#: with no solution. It is bound by `require_plan` instead.
D1_DECLARED_INPUTS: tuple[str, ...] = (
    "scripts/pod/autoinit_preflight_setup.sh",
    "configs/experiments/phase_c1/authorization.json",
    "configs/autoinit/d1_search_artifacts.json",
    "configs/autoinit/d1_search_artifacts_failed.json",
    "scripts/experiments/stage-1/phase_d_series/scoring_protocol.py",
)

D1_SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/pod",
                                    "scripts/autoinit")

CURRENT_CLOSURE_SNAPSHOT = "configs/experiments/phase_d1/executable_closure.json"


def d1_current_executable(repo_root: str | Path = REPO) -> dict[str, Any]:
    """What a D1 session would execute NOW, derived live from the tree."""
    from aadistill.governance.closure import ClosureError, derive

    try:
        return derive(repo_root, "phase_d1", D1_ENTRY_POINTS, D1_DECLARED_INPUTS,
                      roots=D1_SOURCE_ROOTS)
    except ClosureError as exc:
        raise D1AuthorizationRefused(
            f"cannot derive the D1 executable set: {exc}") from exc


def d1_closure_drift(repo_root: str | Path = REPO) -> dict[str, Any] | None:
    """How the live closure differs from the recorded snapshot, or None.

    Reported, never gated on: an edited file is ordinary work, whereas a file
    appearing in or vanishing from the set is a change in what would run.
    """
    from aadistill.governance.closure import compare

    path = Path(repo_root) / CURRENT_CLOSURE_SNAPSHOT
    if not path.is_file():
        return None
    return compare(d1_current_executable(repo_root),
                   json.loads(path.read_text()))




@dataclass(frozen=True)
class D1Authorization(SpendAuthorization):
    """The issued artifact. A `SpendAuthorization` by TYPE, and D1 by schema.

    Subclassing the real base rather than reimplementing it is the point: the
    runner calls `require_plan`, `require_harness`, `require_within_cap` and
    `require_within_launch_limit`, and the first version of this class implemented
    only the first — so the launcher could not have reached a provider query. It
    would have failed locally on the authorization object, which is the cheap
    failure, but it would have failed having reported itself ready.

    What is D1's and therefore overridden here: the schema, the typed permissions,
    the arm, the scientific identities the money was authorized against, and a
    `require_harness` that refuses MEMBERSHIP drift as well as content drift.
    """

    #: D1's own, beyond the base's.
    design_hash: str = ""
    arm: str = ""
    #: THE RUN THIS GRANT WAS ISSUED FOR. Bound because `config_hash` below is a
    #: function of it: `run_id` is a field of `SearchConfig.as_dict()`, so the
    #: identity the money is authorized against is specific to one run id and
    #: carries no meaning without it.
    #:
    #: `build_payload` used to build its probe session with
    #: `run_id="authorization-probe"` while the driver built its own with the
    #: real run id, so the bound `config_hash` could never equal the session's
    #: and stage A's equality check would have refused EVERY launch -- after
    #: setup, the registries, the frozen assets and the contract had all been
    #: paid for. Verified: the committed grant bound
    #: 6301e12e2fd0738e… and run `d1_search_20261005_173304` computes
    #: b6a1806dca6dae9d….
    run_id: str = ""
    measurement_protocol_id: str = ""
    config_hash: str = ""
    suite_content_sha256: str = ""
    science_plan_hash: str = ""
    money: Mapping[str, Any] = field(default_factory=dict)
    one_use: str = ""
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

    # -- the harness ------------------------------------------------------

    def require_harness(self, repo_root: str | Path = REPO) -> dict[str, Any]:
        """Re-derive the live closure and refuse drift. Both kinds.

        The base compares a DIGEST over `harness_source_files`, which catches a
        content change. It cannot catch a MEMBERSHIP change, because the file list
        it digests is the authorization's own: a module added to or removed from
        what actually runs would be digested as the same set. So this re-derives
        the closure from the entry points and compares the file set first.

        An edited launcher, driver, session or setup script is an unrehearsed
        harness, and a paid run that produces permanent artifacts must not be
        executed by one.
        """
        observed = d1_current_executable(repo_root)
        if not self.harness_source_digest:
            raise D1AuthorizationRefused(
                "this D1 authorization declares no harness_source_digest, so it "
                "cannot authorize any executable. Re-issue it against the "
                f"rehearsed harness (observed {observed['digest'][:16]}).")
        recorded = set(self.harness_source_files)
        live = {f["path"] for f in observed["files"]}
        if recorded and recorded != live:
            added, gone = sorted(live - recorded), sorted(recorded - live)
            raise D1AuthorizationRefused(
                "the D1 executable closure MEMBERSHIP has changed since this "
                f"authorization was granted: {len(added)} added {added[:4]}, "
                f"{len(gone)} removed {gone[:4]}. A digest over the recorded list "
                "would not have seen this -- a module that joins or leaves what "
                "runs is a change in what runs. Re-derive and re-issue.")
        if observed["digest"] != self.harness_source_digest:
            raise D1AuthorizationRefused(
                f"the D1 harness digests to {observed['digest'][:16]} and this "
                f"authorization was granted against "
                f"{self.harness_source_digest[:16]}. The rehearsed harness and "
                "the executable harness differ; re-rehearse and re-issue rather "
                "than running an unrehearsed harness against a paid "
                "authorization.")
        return observed

    def require_plan(self, plan_hash: str) -> None:
        """Refuse an artifact that does not bind THIS session's plan.

        A plan hash that disagrees means the authorization was issued against a
        different design revision than the one the pod checked out.
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

    def require_session_commit(self, commit: str) -> None:
        """The pod checks out a commit; this is where it must be the bound one."""
        if not self.authorized_session_commit:
            raise D1AuthorizationRefused(
                "this authorization binds no session commit, so it cannot say "
                "which tree it authorized")
        if commit != self.authorized_session_commit:
            raise D1AuthorizationRefused(
                f"this authorization binds commit "
                f"{self.authorized_session_commit[:12]} and the session declares "
                f"{commit[:12]}")

    def require_run_id(self, run_id: str) -> None:
        """The launch must be the run this grant priced and hashed.

        Cheap, and it is the only place the mismatch is cheap. `config_hash`
        includes `run_id`, so a launch under a different run id carries a
        different config hash and stage A's equality check refuses it -- on a
        pod, after setup. Asked here, at `$0`, before a provider is contacted.
        """
        if not self.run_id:
            raise D1AuthorizationRefused(
                "this authorization binds no run_id, so the config_hash it "
                "carries cannot be attributed to any run")
        if run_id != self.run_id:
            raise D1AuthorizationRefused(
                f"this authorization was issued for run {self.run_id!r} and the "
                f"session declares {run_id!r}. `run_id` is a field of "
                "SearchConfig.config_hash, so the bound identity belongs to the "
                "other run and the driver would refuse it after setup.")

    def require_treatment_arm(self) -> None:
        """A FORMAL D1 search is treatment-only. The control is a $0 check.

        The frozen hypothesis is the combined protocol -- Top-K + bsz3 +
        supervised-target scoring -- as a challenger against incumbent B, and the
        funded chain prices ONE formal search. `all_positions` stays constructible
        for protocol-identity tests at $0; it must not be issuable for a second
        full paid beam.
        """
        if self.arm != FORMAL_ARM:
            raise D1AuthorizationRefused(
                f"this authorization names the {self.arm!r} arm. A FORMAL D1 "
                f"search is {FORMAL_ARM!r} only: the funded chain prices ONE "
                "search, and the control arm is a $0 contract check rather than "
                "a second paid beam.")

    # -- serialization ----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        payload.update({
            "design_hash": self.design_hash,
            "arm": self.arm,
            "run_id": self.run_id,
            "measurement_protocol_id": self.measurement_protocol_id,
            "config_hash": self.config_hash,
            "suite_content_sha256": self.suite_content_sha256,
            "science_plan_hash": self.science_plan_hash,
            "money": dict(self.money),
            "one_use": self.one_use,
            #: FROM THE FIELDS, never a literal: a document that disagreed with
            #: the object that wrote it would be worse than no document.
            "authorizes_d1_search": bool(self.authorizes_d1_search),
            "allows_beam_search": bool(self.allows_beam_search),
            "allows_recovery": bool(self.allows_recovery),
            "allows_behavioural": bool(self.allows_behavioural),
            "automatic_followon_start": bool(self.automatic_followon_start),
        })
        #: A HASH OF ITSELF, last, over everything above it.
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "D1Authorization":
        """Read an issued authorization back, self-hash verified.

        An artifact the issuer can write and the loader cannot read fails on the
        pod, after the money is committed.
        """
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if not stated or stated != sha256_json(check):
            raise D1AuthorizationRefused(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        if raw.get("schema") != SCHEMA:
            raise D1AuthorizationRefused(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. "
                "Another experiment's grant measures a different harness and "
                "carries a ceiling derived for different work; it cannot "
                "authorize a D1 search.")
        missing = [f for f in (
            "authorization_id", "granted_utc", "hard_cap_usd",
            "authorized_session_commit", "design_hash", "arm",
            #: `run_id` is REQUIRED, because `config_hash` is a function of it.
            #: An artifact that binds a config hash without saying which run it
            #: belongs to binds a number nothing can be compared against.
            "run_id",
            "measurement_protocol_id", "config_hash", "suite_content_sha256",
            "authorized_stages", "harness_source_digest") if not raw.get(f)]
        if missing:
            raise D1AuthorizationRefused(f"{path} omits {missing}")
        #: THE POLICY REFUSES A DENIED CLAIM, so the list of forbidden keys lives
        #: in one declaration rather than in a chain of `if`s here that a new
        #: permission could be added without.
        D1_SEARCH_POLICY.check_claims(raw, where=str(path))
        if not raw.get("allows_beam_search") or not raw.get("authorizes_d1_search"):
            raise D1AuthorizationRefused(
                f"{path} does not authorize a D1 beam search")
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
            design_hash=raw["design_hash"], arm=raw["arm"],
            run_id=raw["run_id"],
            measurement_protocol_id=raw["measurement_protocol_id"],
            config_hash=raw["config_hash"],
            suite_content_sha256=raw["suite_content_sha256"],
            science_plan_hash=raw.get("science_plan_hash", ""),
            money=dict(raw.get("money") or {}),
            one_use=raw.get("one_use", ""),
            authorizes_d1_search=bool(raw["authorizes_d1_search"]),
            allows_beam_search=bool(raw["allows_beam_search"]),
            allows_recovery=bool(raw.get("allows_recovery", False)),
            allows_behavioural=bool(raw.get("allows_behavioural", False)),
            automatic_followon_start=bool(
                raw.get("automatic_followon_start", False)),
            action_policy=D1_SEARCH_POLICY)


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
        "expected_minutes": float(session["expected_minutes"]),
        "container_disk_usd": float(session["container_disk_usd"]),
        "container_disk_gb": int(session.get("container_disk_gb") or 0),
        "price_per_hour": float(session["price_per_hour"]),
        "_basis": basis,
    }


def live_secure_price(gpu: str = "NVIDIA L40S") -> dict[str, Any]:
    """The provider's CURRENT securePrice for `gpu`. `$0`, and required.

    The package contract says the secure rate is re-queried immediately before
    authorization. Reading the measured basis' $1.09/h and letting the launcher
    abort later if the market moved is NOT that contract: it issues a ceiling at a
    rate nothing re-checked, and a lower live rate would silently authorize more
    than the session costs.
    """
    import json as _json
    import urllib.request

    from aadistill.infrastructure.provider import read_api_key

    key = read_api_key(str(Path("~/.runpod/config.toml").expanduser()))
    query = ("query { gpuTypes(input:{id:\"" + gpu + "\"}) { id securePrice "
             "lowestPrice(input:{gpuCount:1}) { stockStatus } } }")
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=_json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 #: REQUIRED. The provider sits behind a CDN that rejects
                 #: `Python-urllib/3.x` with HTTP 403 and body `error code: 1010`
                 #: -- which reads exactly like a credential failure and is not
                 #: one. The same query succeeds from curl, whose only relevant
                 #: difference is this header.
                 "User-Agent": "aadistill-d1-authorization/1.0"})
    body = _json.loads(urllib.request.urlopen(req, timeout=30).read())
    types = ((body.get("data") or {}).get("gpuTypes") or [])
    if not types or types[0].get("securePrice") in (None, 0):
        raise D1AuthorizationRefused(
            f"the provider returned no securePrice for {gpu!r}; an authorization "
            "may not be issued at a rate nothing quoted")
    row = types[0]
    return {"gpu": gpu, "usd_per_hour": float(row["securePrice"]),
            "stock_status": ((row.get("lowestPrice") or {}).get("stockStatus")),
            "_quoted": "live, immediately before issuance"}


def reprice_at(rate_usd_per_hour: float,
               repo_root: str | Path = REPO) -> dict[str, Any]:
    """The SAME accepted minute bound, costed at a different rate.

    NO GPU TIMING IS REPEATED and no minute assumption changes: the accepted
    `hard_ceiling_minutes` and the accepted container-disk model are re-costed, and
    only the dollar consequence moves. A lower live rate therefore LOWERS the
    authorized ceiling, which is the half of this contract that an abort-if-higher
    launcher check cannot provide.
    """
    accepted = session_ceiling(repo_root)
    minutes = accepted["hard_ceiling_minutes"]
    expected_minutes = accepted["expected_minutes"]
    #: The accepted disk model, scaled by nothing: container disk is billed per
    #: hour of the SAME bound, so it moves with the rate only through the hours.
    disk_per_hour = (accepted["container_disk_usd"] / (minutes / 60.0)
                     if minutes else 0.0)
    hard = round(minutes / 60.0 * rate_usd_per_hour
                 + minutes / 60.0 * disk_per_hour, 4)
    expected = round(expected_minutes / 60.0 * rate_usd_per_hour
                     + expected_minutes / 60.0 * disk_per_hour, 4)
    return {
        "hard_ceiling_usd": hard,
        "expected_usd": expected,
        "hard_ceiling_minutes": minutes,
        "expected_minutes": expected_minutes,
        "price_per_hour": float(rate_usd_per_hour),
        "accepted_at_price_per_hour": accepted["price_per_hour"],
        "accepted_hard_ceiling_usd": accepted["hard_ceiling_usd"],
        "container_disk_usd": round(minutes / 60.0 * disk_per_hour, 4),
        "_basis": accepted["_basis"],
        "_what_moved": (
            "ONLY the rate. The accepted "
            f"{minutes:.2f}-minute search bound and the accepted container-disk "
            "model are unchanged, and no GPU timing was repeated: this is the "
            "dollar consequence of an already-accepted minute bound."),
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
                  granted_utc: str, run_id: str, arm: str = FORMAL_ARM,
                  repo_root: str | Path = REPO,
                  workdir: Path | None = None,
                  live_rate: float | None = None) -> dict[str, Any]:
    """The authorization payload, fully derived from the tree it will bind.

    `live_rate` is quoted here by default, immediately before issuance, which is
    what the package contract requires. Passing one is for tests: a round that
    reads a committed rate is not re-querying it.

    `run_id` is REQUIRED and is not cosmetic. It is the run id the probe session
    below is built with, and therefore the run id the bound `config_hash`
    describes. This function used to pass the literal `"authorization-probe"`
    while the driver passed the real one, which made the bound config hash a
    value no launch could ever reproduce -- the single most expensive defect in
    the prepared chain, because every $0 gate passed and the refusal landed in
    stage A on a billing pod.
    """
    import tempfile

    from experiments.phase_d1 import d1_session as S

    root = Path(repo_root)
    for f in GRANT_MAY_NOT_STATE:
        if f in grant:
            raise D1AuthorizationRefused(
                f"the grant states {f!r}, which this module DERIVES. A grant that "
                "can state its own identity or its own stages can change the "
                "experiment or over-authorize itself.")

    #: TREATMENT ONLY, at the paid boundary. The frozen hypothesis is the combined
    #: protocol as a challenger against B and the funded chain prices ONE search;
    #: `all_positions` stays constructible at $0 for protocol-identity checks.
    if arm != FORMAL_ARM:
        raise D1AuthorizationRefused(
            f"a FORMAL D1 search authorization may not be issued for the {arm!r} "
            f"arm. It is {FORMAL_ARM!r} only: the funded chain prices ONE search, "
            "and the control is a $0 contract check rather than a second paid "
            "beam.")
    if not session_commit or len(session_commit) != 40:
        raise D1AuthorizationRefused(
            f"session_commit {session_commit!r} is not a full commit id; the pod "
            "checks out this tree and the authorization must bind which one")
    if not str(run_id or "").strip():
        raise D1AuthorizationRefused(
            "no run_id: the config_hash this authorization binds is computed "
            "with the run id, so an unnamed run produces an identity the "
            "launching session cannot match")

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

    #: THE LIVE RATE, then the SAME accepted minute bound costed at it. No GPU
    #: timing is repeated; only the dollar consequence moves. A lower live rate
    #: lowers the authorized ceiling, which an abort-if-higher launcher check
    #: cannot do.
    quote = ({"gpu": "NVIDIA L40S", "usd_per_hour": float(live_rate),
              "_quoted": "supplied by the caller (tests only)"}
             if live_rate is not None else live_secure_price())
    priced = reprice_at(quote["usd_per_hour"], root)
    ceiling = priced["hard_ceiling_usd"]
    failed = check_the_four_conditions(ceiling=ceiling, money=money)
    if failed:
        raise D1AuthorizationRefused(
            f"at the live rate ${quote['usd_per_hour']:.4f}/h the session is not "
            "fundable:\n  - " + "\n  - ".join(failed)
            + "\n\nThe science does not change to absorb a price movement: "
              "neither the beam nor the minute bound may be narrowed. Stop at $0.")

    #: THE ENVELOPE IS NOT THE GRANT, and neither is a stale ceiling. A grant
    #: asking for a figure the live rate no longer produces is refused.
    asked = grant.get("hard_cap_usd")
    if asked is not None and abs(float(asked) - ceiling) > 5e-4:
        raise D1AuthorizationRefused(
            f"the grant asks for ${float(asked):.4f}; the ceiling DERIVED at the "
            f"live ${quote['usd_per_hour']:.4f}/h is ${ceiling:.4f}. The "
            "authorization carries the derived figure.")
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
    #: THE RUN'S OWN ID AND THE DEVICE THE POD WILL USE.
    #:
    #: Both are fields of `SearchConfig.as_dict()` and therefore of
    #: `config_hash`, and this call got both wrong: `run_id="authorization-probe"`
    #: against the driver's real run id, and `device="cpu"` against the driver's
    #: `cuda`. Either alone made the bound `config_hash` unmatchable, so stage A's
    #: equality check would have refused every launch after setup was paid for.
    #:
    #: `cuda` is not a guess about this machine. It is what the FORMAL session
    #: runs on -- the driver refuses without a CUDA device, and `numerics()`
    #: declares `device_type="cuda"` unconditionally -- so a config built on
    #: `cpu` described a session this experiment cannot execute. Constructing it
    #: allocates nothing and needs no GPU on the issuing box.
    #:
    #: `workdir` stays a throwaway directory because it is NOT in `as_dict()`
    #: and so cannot move the identity.
    session = S.build_session(arm=arm, workdir=tmp, run_id=str(run_id).strip(),
                              device=FORMAL_DEVICE, repo_root=root)
    contract = S.assert_session_contract(session, root)

    #: THE HARNESS, derived live and bound BY MEMBERSHIP AND DIGEST.
    closure = d1_current_executable(root)

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
        scope_note=str(grant.get("covers") or "")[:4000],
        authorized_session_commit=session_commit,
        harness_source_digest=closure["digest"],
        harness_source_files=tuple(f["path"] for f in closure["files"]),
        provenance_commit=session_commit,
        design_hash=design["design_hash"],
        arm=arm,
        run_id=str(run_id).strip(),
        measurement_protocol_id=contract["measurement_protocol_id"],
        config_hash=contract["config_hash"],
        suite_content_sha256=contract["suite_content_sha256"],
        money={**money, "derived_session": priced, "live_quote": quote,
               "four_conditions": "all four checked at the LIVE rate; see "
                                  "check_the_four_conditions"},
        one_use=("ONE grant, ONE issuance, ONE launcher session. A failure is this "
                 "session's result; a retry is a new grant and a new "
                 "authorization."),
        action_policy=D1_SEARCH_POLICY,
    )
    payload = auth.as_dict()
    payload["authorizes"] = (
        "ONE D1 formal SEARCH session on the frozen design, stages "
        f"{'/'.join(AUTHORIZED_STAGES)}, on the {FORMAL_ARM} arm. It does NOT "
        "authorize recovery, behavioural screening, confirmation, promotion, D2, "
        "D3, or any repetition of a completed measurement.")
    payload["harness"] = {
        "n_files": closure["n_files"],
        "entry_points": list(closure["entry_points"]),
        "digest": closure["digest"],
        "_derived": ("live from the tree by aadistill.governance.closure, never a "
                     "hand-maintained list: a recorded file list goes stale on the "
                     "first edit and then reports a confident identity for the "
                     "wrong set"),
    }
    #: The self-hash is computed LAST, over the final payload.
    payload.pop("authorization_sha256", None)
    payload["authorization_sha256"] = sha256_json(payload)
    return payload
