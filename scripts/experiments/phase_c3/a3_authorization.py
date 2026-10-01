"""A3's one-use authorization type, and the budget derived from live pricing.

Structurally a `PhaseAAuthorization` — same commit binding, same harness rule,
same hash-of-itself check — and a different **type**, so no Phase-A, Phase-B,
C1, C2, C3 or engineering-pilot grant can stand in for it. A grant measures a
HARNESS and carries a ceiling derived for particular work; A3's harness
contains the A3 launcher, driver, session and comparison, none of which appear
in any earlier file set.

**The scope is the whole chain, and it says what it is not.** A3 runs one fixed
path, two executions of its last step, three recovery probes and three
evaluations. It runs no search, eliminates no arm, has no fourth seed, retrains
no control, computes no verdict on the pod, and authorizes neither the FFN
experiment nor D1/D2/D3 nor Stage-0 v2.

**The ceiling comes from the live re-price, never from a constant.** `$1.09/h`
is a historical observation; a different live `securePrice` is re-priced and
re-gated, not refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.governance.authorization import AuthorizationError  # noqa: E402
from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402
from aadistill.infrastructure.session import BudgetSpec  # noqa: E402
from experiments.phase_a.plan import PhaseAAuthorization  # noqa: E402

SCHEMA = "aadistill.autoinit.a3_authorization/v1"
LIVE_PRICING_PATH = "logs/stages/stage-1/phase_c3/plans/a3_live_pricing.json"

#: Every file whose bytes decide what an A3 session executes. A grant binds
#: this digest, so a repair to any of them invalidates the grant and forces a
#: fresh chain — which is the point: the launcher that ran is the launcher the
#: maintainer authorized.
A3_HARNESS_FILES: tuple[str, ...] = (
    "configs/autoinit/a3_artifacts.json",
    "configs/autoinit/a3_artifacts_failed.json",
    "configs/experiments/phase_c1/authorization.json",
    "scripts/autoinit/aggregate_a3.py",
    "scripts/autoinit/compare_a_bsz3.py",
    "scripts/autoinit/write_a3_design.py",
    "scripts/experiments/phase_c3/a3_authorization.py",
    "scripts/experiments/phase_c3/a3_pricing.py",
    "scripts/experiments/phase_c3/a3_session.py",
    "scripts/experiments/phase_c3/a_bsz3.py",
    "scripts/experiments/phase_c3/formal_pricing.py",
    "scripts/experiments/phase_c3/hardware.py",
    "scripts/pod/autoinit_a3_driver.py",
    "scripts/pod/autoinit_a3_launch.py",
    "src/aadistill/initialization/operators/attention/gqa/activation_importance.py",
)


def a3_harness_digest(repo_root: str | Path = REPO) -> dict[str, Any]:
    """The harness a grant measures, derived from the declared file set."""
    from aadistill.governance.authorization import harness_source_digest

    return harness_source_digest(repo_root, files=A3_HARNESS_FILES)


def a3_harness_digest_value(repo_root: str | Path = REPO) -> str:
    return a3_harness_digest(repo_root)["digest"]


class A3Authorization(PhaseAAuthorization):
    """Permits exactly the complete A3 chain, once."""

    @property
    def allows_phase_a(self) -> bool:
        """Never. A3 replays one frozen path; it cannot start Phase A."""
        return False

    @property
    def allows_beam_search(self) -> bool:
        """Never. There is no search anywhere in this session."""
        return False

    @property
    def authorizes_c1_isolation(self) -> bool:
        """Never. C1 is complete and this artifact does not re-open it."""
        return False

    @property
    def authorizes_c3_isolation(self) -> bool:
        """Never. C3 is complete with a NO_GO; A3 is a different experiment."""
        return False

    @property
    def authorizes_a3(self) -> bool:
        return True

    @property
    def allows_arm_elimination(self) -> bool:
        """Never. One arm, three seeds, no halving and no forced winner."""
        return False

    @property
    def allows_control_retraining(self) -> bool:
        """Never. attempt75's controls are reused, not rebuilt.

        Stated as a permission rather than left implicit: retraining them
        would spend three probes to reproduce evidence that exists, and would
        quietly make this a six-probe experiment the grant does not fund.
        """
        return False

    @property
    def allows_on_pod_decision(self) -> bool:
        """Never. The comparison runs off pod at $0.

        attempt75 trained, preserved and scored nine probes and then lost its
        decision artifact to a crash in the on-pod aggregation. A3's driver
        ends at preservation, and this permission says so in the artifact a
        launcher binds rather than only in prose.
        """
        return False

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        #: From the properties, never literals: a document disagreeing with
        #: the object that wrote it is worse than no document.
        for name in ("allows_phase_a", "allows_beam_search",
                     "authorizes_c1_isolation", "authorizes_c3_isolation",
                     "authorizes_a3", "allows_arm_elimination",
                     "allows_control_retraining", "allows_on_pod_decision"):
            payload[name] = getattr(self, name)
        payload["scope"] = (
            "ONE experiment end to end: replay the frozen parent "
            "eea90c91346a under its digest gate, verify A-bsz1 rebuilds the "
            "incumbent 53e30566c5f7, collect interleaved A-bsz1/A-bsz3 "
            "diagnostics, then train THREE A-bsz3 recovery probes at the "
            "three frozen C3 seeds and evaluate each once on "
            "c1_confirmation_v1. A differing A-bsz3 artifact digest is a "
            "FINDING and does not stop the chain. No search, no ranking, no "
            "arm elimination, no fourth seed, no retraining of attempt75's "
            "controls, no on-pod decision, and no authorization to start the "
            "FFN experiment, D1/D2/D3 or Stage-0 v2.")
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "A3Authorization":
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
                f"{path} declares schema {raw.get('schema')!r}, not "
                f"{SCHEMA!r}. A Phase-A, Phase-B, C1, C2, C3 or "
                "engineering-pilot grant measures a different harness and "
                "carries a ceiling derived for different work; it cannot "
                "authorize A3.")
        for forbidden in ("allows_phase_a", "allows_beam_search",
                          "allows_arm_elimination", "authorizes_c1_isolation",
                          "authorizes_c3_isolation",
                          "allows_control_retraining",
                          "allows_on_pod_decision"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. A3 replays one frozen path, "
                    "runs no search, eliminates no arm, re-opens neither C1 "
                    "nor C3, retrains no control and computes no on-pod "
                    "decision; an artifact claiming otherwise is not an A3 "
                    "authorization.")
        if not raw.get("authorizes_a3"):
            raise AuthorizationError(
                f"{path} does not claim authorizes_a3, so it authorizes no "
                "A3 session whatever else it permits")
        #: Mapped explicitly, not by field name: `as_dict` serialises
        #: `plan_hash` as `phase_a_session_plan_hash`, so a by-name filter
        #: drops it and the constructor then fails on a required argument.
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

def load_live_pricing(repo_root: str | Path = REPO) -> dict[str, Any]:
    """The live re-price produced immediately before issuance.

    Refused rather than defaulted if absent, and refused if it reports
    unfundable: a session priced from a stale or assumed rate is what the
    live-pricing requirement removed, and `FUNDABLE` is now a FOUR-condition
    answer including the package total.
    """
    p = Path(repo_root) / LIVE_PRICING_PATH
    if not p.is_file():
        raise AuthorizationError(
            f"no live pricing record at {LIVE_PRICING_PATH}; run "
            "scripts/experiments/phase_c3/a3_pricing.py --write first")
    doc = json.loads(p.read_text())
    if not doc.get("FUNDABLE"):
        raise AuthorizationError(
            f"{LIVE_PRICING_PATH} reports FUNDABLE=false with shortfalls "
            f"{doc.get('shortfalls_usd')}; a session may not be issued "
            "against a price the envelopes do not cover")
    checked = doc.get("_every_applicable_limit_is_checked") or []
    if "package_total_covers_derived" not in checked:
        raise AuthorizationError(
            f"{LIVE_PRICING_PATH} did not evaluate the package total. It "
            "equals formal + engineering, which is exactly why nothing "
            "checked it; an identity is not a check.")
    return doc


def a3_budget_spec(repo_root: str | Path = REPO) -> BudgetSpec:
    """The `BudgetSpec` whose plan reproduces the LIVE hard ceiling exactly.

    Models the HARD case with `contingency_fraction=0`. The component table's
    hard bound already substitutes a worst-case setup and scales every
    measured GPU component by the declared overrun factor; letting
    `plan_session` apply its own contingency on top would be two models of one
    quantity, and that double-counting once produced a plan that terminated
    above its own grant.

    The closeout allowance is carried ONCE, as the recovery reserve, because
    `plan_session` adds the reserve on top of the priced total. Passing it as
    both a phase and the reserve is how a plan lands above its ceiling.
    """
    from experiments.phase_c3.a3_pricing import (
        OVERRUN_FACTOR, SETUP_WORST_CASE_MINUTES, component_minutes,
    )

    load_live_pricing(repo_root)          # refuses if absent or unfundable
    hard = component_minutes(worst=True)
    #: A3 trains three probes of the same 1023-step recipe the controls used.
    n_probes, steps_per_probe = 3, 1023
    train_worst = hard["recovery_probes"]
    eval_worst = hard["evaluations"]
    return BudgetSpec(
        arms=n_probes,
        steps_per_arm=steps_per_probe,
        step_seconds=train_worst * 60.0 / (n_probes * steps_per_probe),
        step_source=(
            "measured 61.76 min/probe over attempt75's nine 0.86M probes of "
            f"this exact 1023-step recipe, scaled by the declared "
            f"{OVERRUN_FACTOR}x hard-ceiling overrun factor"),
        setup_minutes=SETUP_WORST_CASE_MINUTES,
        eval_minutes_per_arm=eval_worst / n_probes,
        transfer_minutes=0.0,
        other_phases=(
            Phase("prelaunch_gates", hard["pre_provider_gates"]),
            Phase("teacher_fetch_verify", hard["teacher_fetch_verify"]),
            Phase("register_operator", hard["register_operator"]),
            Phase("fixed_parent_replay", hard["parent_replay"]),
            Phase("initialization_diagnostics", hard["initialization_rounds"]),
            Phase("package", hard["aggregation"]),
        ),
        contingency_fraction=0.0,
        artifact_recovery_reserve_minutes=float(hard["collect_and_teardown"]),
    )


def a3_hard_ceiling_usd(repo_root: str | Path = REPO) -> float:
    return float(load_live_pricing(repo_root)["price"]["hard_ceiling"]["usd"])


def a3_expected_usd(repo_root: str | Path = REPO) -> float:
    return float(load_live_pricing(repo_root)["price"]["expected"]["usd"])


def a3_billed_rate_usd_per_hour(repo_root: str | Path = REPO) -> float:
    return float(load_live_pricing(repo_root)["price"]["billed_rate_usd_per_hour"])
