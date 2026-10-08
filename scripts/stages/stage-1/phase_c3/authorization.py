"""The Phase-C3 authorization type and budget. Its own type, its own ceiling.

Three properties make this a *type* rather than a policy comment.

**It cannot authorize anything else, and nothing else can authorize it.**
`load` refuses any artifact whose schema is not this one. A C1 grant measures
a harness that does not contain the causal-KL operator and carries a ceiling
derived for six probes on two arms; a pilot campaign record authorizes
engineering, not science. Accepting either here would certify code this
session does not run and price it for work it does not do. `SESSION_KIND=c3`
has its own branch in `autoinit_preflight_setup.sh` for the same reason: a
missing branch is not a type error, it falls through to `spend` and loads a
`SpendAuthorization`. Phase-B attempt 2 paid `$0.2300` to establish that.

**Its ceiling is DERIVED from a live price, not typed in.** The 2026-09-28
amendment set a `$30.00` per-session *envelope* and made the provider price a
live input. `c3_budget_spec()` reads the live-pricing record produced
immediately before issuance, so the enforceable ceiling exists in exactly one
place and reflects what the provider actually charges today. **The envelope
is not the grant**: issuing at `$30.00` when the derived ceiling is `$22.1451`
would be a 36% over-authorization that every downstream gate would accept.

**Its harness is derived, not listed.** `c3_harness_digest()` walks the import
closure from the C3 entry points, so a hand-maintained list cannot go stale or
promise a completeness it does not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aadistill.governance.authorization import AuthorizationError
from aadistill.infrastructure.budget import Phase
from aadistill.infrastructure.manifest import sha256_json
from aadistill.infrastructure.session import BudgetSpec
from experiments.phase_a.plan import PhaseAAuthorization

SCHEMA = "aadistill.autoinit.c3_authorization/v1"

#: The executables a paid C3 session runs. The closure walk starts here.
C3_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/pod/autoinit_c3_launch.py",
    "scripts/pod/autoinit_c3_driver.py",
    "scripts/experiments/stage-1/phase_c3/session.py",
    "scripts/experiments/stage-1/phase_c3/formal_pricing.py",
    "scripts/autoinit/issue_c3_authorization.py",
    "scripts/pod/collect_artifacts.py",
)

#: Files no import edge reaches whose bytes still decide what runs.
#:
#: The PREREGISTRATION is deliberately absent, and it is the one runtime input
#: that cannot be here: it RECORDS this digest, so hashing its bytes into the
#: digest is a fixed point with no solution. It is bound instead by its own
#: `preregistration_sha256`, which `session.load_preregistration` verifies and
#: refuses, and which the launcher's gate compares against the authorization.
#:
#: The live-pricing record is absent for the same reason in reverse: it is
#: produced AFTER the harness is fixed, immediately before issuance, and the
#: authorization binds its figures directly.
C3_DECLARED_INPUTS: tuple[str, ...] = (
    #: The setup script the pod ACTUALLY runs. `SessionRunner._launch`
    #: uploads and executes this one; `SetupManifest` carries no
    #: setup-script field, so no session can substitute another. There is
    #: deliberately NO `autoinit_c3_remote.sh`: the formal chain has no
    #: per-session remote entrypoint, and declaring a file nothing executes
    #: is precisely the defect C1's harness list carried for months --
    #: it named a legacy script whose only remaining reference was that list.
    "scripts/pod/autoinit_preflight_setup.sh",
    "configs/experiments/phase_c1/authorization.json",
    "configs/autoinit/c3_artifacts.json",
    "configs/autoinit/c3_artifacts_failed.json",
)

#: The sys.path roots the C3 entry points insert; a walk that did not know
#: them would silently miss every helper imported by bare name.
C3_SOURCE_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/pod",
                                    "scripts/autoinit")

CURRENT_CLOSURE_SNAPSHOT = "configs/experiments/phase_c3/executable_closure.json"

LIVE_PRICING_PATH = "logs/stages/stage-1/phase_c3/plans/c3_live_pricing.json"

#: 3 arms x 3 seeds. DERIVED nowhere else: the plan owns the arms and seeds,
#: and `scripts/experiments/stage-1/phase_c3/tests/test_c3_session_contract.py` proves this equals what the
#: session contract computes from them.
C3_PROBES = 9

#: 1023 steps at the rate every retained probe of this recipe actually ran.
PROBE_STEPS = 1023


def c3_current_executable(repo_root: str | Path = ".") -> dict[str, Any]:
    """What a C3 session would execute NOW, derived live from the tree."""
    from aadistill.governance.closure import ClosureError, derive

    try:
        return derive(repo_root, "phase_c3", C3_ENTRY_POINTS,
                      C3_DECLARED_INPUTS, roots=C3_SOURCE_ROOTS)
    except ClosureError as exc:
        raise AuthorizationError(
            f"cannot derive the C3 executable set: {exc}") from exc


def c3_harness_digest(repo_root: str | Path = ".",
                      files: tuple[str, ...] | None = None) -> dict[str, Any]:
    """The digest a C3 grant binds — the CURRENT executable unless told otherwise."""
    if files is None:
        return c3_current_executable(repo_root)
    from aadistill.governance.closure import derive

    return derive(repo_root, "phase_c3", (), files, roots=C3_SOURCE_ROOTS)


def c3_harness_digest_value(repo_root: str | Path = ".") -> str:
    return c3_harness_digest(repo_root)["digest"]


class C3Authorization(PhaseAAuthorization):
    """Permits exactly the Phase-C3 three-arm causal-KL isolation.

    Structurally a `PhaseAAuthorization` — same commit binding, same harness
    rule, same hash-of-itself check — and a different **type**, so none of
    Phase A, Phase B, C1, C2 or an engineering pilot can stand in for it.
    """

    @property
    def allows_phase_a(self) -> bool:
        """Never. C3 replays one frozen path; it cannot start Phase A."""
        return False

    @property
    def allows_beam_search(self) -> bool:
        """Never. There is no search anywhere in this session."""
        return False

    @property
    def authorizes_c1_isolation(self) -> bool:
        """Never. C1 is complete; this artifact does not re-open it."""
        return False

    @property
    def authorizes_c3_isolation(self) -> bool:
        return True

    @property
    def allows_arm_elimination(self) -> bool:
        """Never. Nine probes, no successive halving, no forced winner."""
        return False

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        # From the properties, never a literal: a document that disagreed with
        # the object that wrote it would be worse than no document.
        payload["allows_phase_a"] = self.allows_phase_a
        payload["allows_beam_search"] = self.allows_beam_search
        payload["authorizes_c1_isolation"] = self.authorizes_c1_isolation
        payload["authorizes_c3_isolation"] = self.authorizes_c3_isolation
        payload["allows_arm_elimination"] = self.allows_arm_elimination
        payload["scope"] = (
            "3 arms x 3 fresh seeds = 9 fixed 0.86M recovery probes on one "
            "frozen path, evaluated once each on c1_confirmation_v1. The "
            "PRIMARY contrast is causal-B1 - incumbent B and owns the sole "
            "GO/NO_GO/INCONCLUSIVE verdict; the secondary (causal-B3 - "
            "causal-B1) and practical (causal-B3 - B) contrasts are reported "
            "completely and may not replace it. No search, no ranking, no "
            "successive halving, no tie-breaking, no arm elimination, no "
            "fourth seed, and no authorization to start C4.")
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "C3Authorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. "
                "A Phase-A, Phase-B, C1, C2 or engineering-pilot grant measures "
                "a different harness and carries a ceiling derived for "
                "different work; it cannot authorize the C3 isolation.")
        for forbidden in ("allows_phase_a", "allows_beam_search",
                          "allows_arm_elimination", "authorizes_c1_isolation"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. C3 replays one frozen path, "
                    "runs no search, eliminates no arm and does not re-open "
                    "C1; an artifact claiming otherwise is not a C3 "
                    "authorization.")
        # Mapped explicitly, not by field name: `as_dict` serialises
        # `plan_hash` as `phase_a_session_plan_hash` and `science_plan_hash` as
        # `phase_a_science_plan_hash`, so a by-name filter silently drops both
        # and the constructor then fails on a required argument.
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
            harness_source_files=tuple(raw.get("harness_source_files") or ()),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


# ---------------------------------------------------------------------------
# the budget, derived from the LIVE pricing record
# ---------------------------------------------------------------------------

def load_live_pricing(repo_root: str | Path = ".") -> dict[str, Any]:
    """The live re-price produced immediately before issuance.

    Refused rather than defaulted if absent: a C3 session priced from a stale
    or assumed rate is exactly what the amendment removed.
    """
    p = Path(repo_root) / LIVE_PRICING_PATH
    if not p.is_file():
        raise AuthorizationError(
            f"no live pricing record at {LIVE_PRICING_PATH}. Section 2 of the "
            "2026-09-28 decision requires a LIVE securePrice re-query "
            "immediately before the C3 authorization; run "
            "scripts/experiments/stage-1/phase_c3/formal_pricing.py first.")
    doc = json.loads(p.read_text())
    if not doc.get("FUNDABLE"):
        raise AuthorizationError(
            f"{LIVE_PRICING_PATH} reports FUNDABLE=false with shortfalls "
            f"{doc.get('shortfalls_usd')}; a session may not be issued against "
            "a price the envelopes do not cover.")
    return doc


def c3_budget_spec(repo_root: str | Path = ".") -> BudgetSpec:
    """The `BudgetSpec` whose plan reproduces the LIVE hard ceiling exactly.

    **It models the HARD case, with `contingency_fraction=0`.** The component
    table's hard bound already substitutes a worst-case setup and scales
    training and evaluation by the declared overrun factor; letting
    `plan_session` apply its own contingency on top of that would be two
    models of one quantity, and it double-counted by `$0.37` when first
    written -- the planner refused to build a plan that terminated above what
    it was authorized for, which is exactly the protection it exists to give.

    So the allowance is applied ONCE, here, from the same
    `hard_ceiling._allowances` the price is derived from. The result is that
    `spec.plan(...).total_usd` equals `c3_hard_ceiling_usd()` by construction
    rather than by coincidence, and `test_c3_formal_pricing` asserts it.
    """
    #: A RELATIVE import, not a `sys.path` insertion plus a top-level one. The
    #: insertion assumed this package sat directly under a directory on the
    #: path, which stopped being true when the experiment tree grew its stage
    #: level -- and a relative import never needed to know where the package
    #: lives. Deferred to call time because the pricing module reads live
    #: pricing and importing it at module scope would make a refusal happen at
    #: import rather than where it can be reported.
    from .formal_pricing import allowances, component_minutes

    load_live_pricing(repo_root)                 # refuses if absent or unfundable
    comps = component_minutes()
    a = allowances()
    overrun = a["train_and_eval_overrun_factor"]

    train_worst = comps["recovery_9_probes"] * overrun
    eval_worst = comps["evaluation_9_probes"] * overrun
    step_seconds = train_worst * 60.0 / (C3_PROBES * PROBE_STEPS)

    return BudgetSpec(
        arms=C3_PROBES,
        steps_per_arm=PROBE_STEPS,
        step_seconds=step_seconds,
        step_source=(
            f"measured 61.8 min/probe over attempt 18's six 0.86M probes of "
            f"this exact 1023-step recipe, scaled by the declared {overrun}x "
            f"hard-ceiling overrun factor"),
        setup_minutes=a["setup_worst_case_minutes"],
        eval_minutes_per_arm=eval_worst / C3_PROBES,
        #: ZERO here, because the closeout allowance is carried as the
        #: RECOVERY RESERVE below instead. It is one 15-minute component and
        #: may only be counted once: passing it as both put the plan $0.28
        #: above its own ceiling, since `plan_session` adds the reserve on
        #: top of the total. Held back as the reserve it does real work --
        #: the soft stop lands a teardown's worth below the hard ceiling,
        #: which is the margin P12.1 requires be preserved.
        transfer_minutes=0.0,
        other_phases=(
            Phase("prelaunch_gates", comps["gates"]),
            Phase("fixed_parent_replay", comps["parent_replay"]),
            Phase("incumbent_attention", comps["armA_incumbent_attention"]),
            Phase("causal_b1_scorer", comps["armB_causal_b1_scorer"]),
            Phase("causal_b3_scorer", comps["armC_causal_b3_scorer"]),
        ),
        #: ZERO, deliberately. See the docstring: the allowance is already in
        #: the minutes above, and a second multiplier here would price the
        #: same risk twice.
        contingency_fraction=0.0,
        artifact_recovery_reserve_minutes=float(
            comps["decide_bootstrap_closeout"]),
        #: so: priced work + reserve == the hard ceiling, exactly.
    )


def c3_hard_ceiling_usd(repo_root: str | Path = ".") -> float:
    """The one place the enforceable C3 ceiling comes from: the live record."""
    return float(load_live_pricing(repo_root)["price"]["hard_ceiling"]["usd"])


def c3_expected_usd(repo_root: str | Path = ".") -> float:
    return float(load_live_pricing(repo_root)["price"]["expected"]["usd"])


def c3_billed_rate_usd_per_hour(repo_root: str | Path = ".") -> float:
    """The LIVE rate the ceiling was derived at, including container disk."""
    return float(load_live_pricing(repo_root)["price"]["billed_rate_usd_per_hour"])
