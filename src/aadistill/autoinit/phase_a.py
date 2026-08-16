"""Phase A: the session plan, its scope, and the only artifact that can permit it.

Every other session in this project runs under `SpendAuthorization`, whose
`allows_phase_a` is a hard `False` and whose `load()` refuses an artifact that
claims otherwise. That gate is not relaxed here, and this module does not import
it for modification: `PhaseAAuthorization` is a **separate type with a separate
schema**, so

* a `SpendAuthorization` artifact can never be loaded as a Phase-A grant, and
* a Phase-A artifact can never be loaded by the preflight or the continuation.

Both directions are refused by schema, not by convention. The preflight's setup
assertion (`assert a.allows_phase_a is False`) therefore continues to hold
byte-for-byte for the sessions it guards, because those sessions still load the
type whose answer is always `False`.

Two different plans bind a Phase-A run, and conflating them would let one move
under the other:

``PHASE_A_PLAN_V1``
    the *session* plan — which stages exist, which block, what each produces.
    The authorization binds to this by hash, exactly as the continuation binds
    to `CONTINUATION_PLAN_V1`.

``SuccessiveHalvingPlan("autoinit.v1.phase_a")``
    the *science* plan — searched leaves, survivors, seeds, thresholds,
    selection rules. `assert_preregistered` binds the executing science plan to
    the frozen preregistration. Its thresholds were materialized by Stage 3 on
    2026-08-15 and are not re-derived here.

Phase A is a terminus. It selects an initializer, names the canonical control as
the winner, or reports `unresolved_equivalence` — and then stops. Nothing in this
module can express permission for a follow-on experiment, which is why
`automatic_followon_start` is a property rather than a field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_file, sha256_json
from .authorization import AuthorizationError
from .recovery import PreflightPlan, PreflightStage, SEED_SA, SEED_SB, SEED_SC

SCHEMA = "aadistill.autoinit.phase_a_authorization/v1"

#: The executable that runs a Phase-A session. As with the continuation set, this
#: is NOT the preflight's list: Phase A has its own launcher, driver and plan
#: module, and an authorization digesting the preflight's files would admit an
#: edited Phase-A driver without noticing. The shared infrastructure appears in
#: both sets because both sessions execute it.
PHASE_A_HARNESS_SOURCE_FILES_V1: tuple[str, ...] = (
    "scripts/pod/autoinit_phase_a_launch.py",
    "scripts/pod/autoinit_phase_a_driver.py",
    "scripts/pod/autoinit_preflight_launch.py",   # the launcher it subclasses
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/autoinit_engine_probe.py",
    "scripts/pod/watchdog.py",
    "scripts/pod/collect_artifacts.py",
    # Imported by the driver, not shelled out to, and therefore just as much
    # "the executable" as the driver itself. `phase_a_search` runs the beam;
    # `write_preregistration` builds the plan the driver binds against, so an
    # edit to either changes what a paid run does.
    "scripts/autoinit/phase_a_search.py",
    "scripts/autoinit/write_preregistration.py",
    "src/aadistill/autoinit/authorization.py",
    "src/aadistill/autoinit/generation_compat.py",
    "src/aadistill/autoinit/phase_a.py",
    "src/aadistill/autoinit/generation.py",
)
PHASE_A_HARNESS_SOURCE_SET_VERSION = 1


def phase_a_harness_digest(repo_root: str | Path = ".", *,
                           files: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Digest the declared Phase-A harness. Same rule, same failure mode.

    A missing declared file raises rather than yielding a digest over a smaller
    harness than the one that runs.
    """
    root = Path(repo_root)
    declared = tuple(files) if files is not None else PHASE_A_HARNESS_SOURCE_FILES_V1
    entries = []
    for rel in sorted(declared):
        path = root / rel
        if not path.is_file():
            raise AuthorizationError(
                f"declared Phase-A harness source {rel!r} is missing; refusing to "
                "authorize a digest over a smaller harness than the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "set_version": PHASE_A_HARNESS_SOURCE_SET_VERSION,
            "files": entries}


#: The Phase-A session plan. Ordered so the cheap, refusable work happens before
#: the 9 recovery probes, which are ~92% of the session's cost.
PHASE_A_PLAN_V1 = PreflightPlan(
    plan_id="autoinit.v1.phase_a_session",
    version=1,
    stages=(
        PreflightStage(
            stage=0, name="attestation and preregistration binding", blocking=True,
            purpose=("re-establish the runtime and evaluation identity this "
                     "session measures under, and refuse to proceed unless the "
                     "science plan is the frozen one"),
            produces=("runtime fingerprint including image digest",
                      "attested evaluation protocol over recovery_search_v2",
                      "frozen-asset verification for state_eval_v1 and "
                      "recovery_search_v2",
                      "assert_preregistered against the frozen Phase-A plan",
                      "the materialized thresholds this run will select against"),
            stop_conditions=(
                "a frozen asset does not match its preregistered constant -> STOP",
                "the executing science plan does not hash to the frozen one -> "
                "STOP: a threshold moved after freezing",
                "the equivalence interval or feasibility floor is not "
                "materialized -> STOP: selection has no rule to apply",
                "the scoring contract digest differs from the pinned one -> STOP")),
        PreflightStage(
            stage=1, name="beam search over initialization paths", blocking=True,
            purpose=("materialize and measure the reachable initialization "
                     "states, and produce the searched leaves the recovery rungs "
                     "will probe"),
            produces=("one measured, hash-bound state per expansion",
                      "level records including every prune reason",
                      "the epsilon-Pareto ranking at each pruned level",
                      "5 admissible complete leaves",
                      "the search journal, resumable by content-derived state id"),
            stop_conditions=(
                "fewer than 5 admissible complete leaves exist -> STOP and report "
                "the shortfall rather than shrinking N",
                "a materialized state fails canonical reload, spec match, "
                "parameter count or finiteness -> STOP",
                "the state evaluation suite hash differs from the attested one -> "
                "STOP: the beam would rank on a different suite's questions")),
        PreflightStage(
            stage=2, name="recovery rung 1 on seed sa", blocking=True,
            purpose=("probe the 5 searched leaves and the retained canonical "
                     "control under one identical recovery recipe"),
            produces=("6 probes at 1023 steps each on seed sa",
                      "recovery_search_v2 battery result per probe",
                      "the rung-1 survivor selection under the frozen rule",
                      "per-probe journal entries, resumable"),
            stop_conditions=(
                "the canonical control is absent -> STOP: a comparison without "
                "the retained baseline is not the comparison the plan describes",
                "capability schema validation fails -> scoring defect, STOP",
                "a probe's evaluation protocol is not comparable to the "
                "attestation -> STOP")),
        PreflightStage(
            stage=3, name="recovery rung 2 on seed sb", blocking=True,
            purpose=("second seed for the 2 survivors and the control, because "
                     "the behaviour metric's seed-only spread is 0.1290"),
            produces=("3 probes at 1023 steps each on seed sb",
                      "pooled_counts@v2 aggregates over sa and sb",
                      "the final selection, or an explicit tie"),
            stop_conditions=(
                "the control did not advance -> STOP: it advances unconditionally",
                "capability schema validation fails -> STOP")),
        PreflightStage(
            stage=4, name="conditional tie-break on seed sc", blocking=False,
            purpose=("resolve finalists that finished inside the preregistered "
                     "equivalence interval after two seeds — and only those"),
            produces=("probes on seed sc for the tied finalists only",
                      "pooled aggregates over sa, sb and sc",
                      "the resolved winner, or unresolved_equivalence"),
            stop_conditions=(
                "the rung-2 selection did not request a tie-break -> this stage "
                "does not run, which is not a failure",
                "no fourth seed is requested under any outcome")),
        PreflightStage(
            stage=5, name="selection and report", blocking=False,
            purpose=("record which initializer won, that the canonical control "
                     "won, or that v1 resolved no unique behavioural winner"),
            produces=("the final selection record with every exclusion",
                      "feasibility and catastrophic-gate outcomes per candidate",
                      "behaviour and correctness reported on separate axes",
                      "the Phase-A result artifact"),
            stop_conditions=(
                "a tie surviving seed sc is reported as unresolved_equivalence "
                "and is a RESULT, not a failure",
                "no follow-on experiment starts from this session")),
    ))


@dataclass(frozen=True)
class PhaseAAuthorization:
    """What a named maintainer permitted for a Phase-A search, bound by hash.

    Deliberately not a subclass of `SpendAuthorization`. Inheriting would make a
    Phase-A grant substitutable wherever the narrower type is expected, and the
    preflight's `allows_phase_a is False` assertion would then depend on which
    object happened to be passed rather than on the type system.
    """

    authorization_id: str
    granted_utc: str
    granted_by: str
    plan_id: str
    plan_hash: str
    #: The frozen *science* plan. Distinct from `plan_hash`, which is the session
    #: plan; both must match or the run is measuring under rules that moved.
    science_plan_hash: str
    expected_usd: float
    hard_cap_usd: float
    authorized_stages: tuple[int, ...]
    stage_conditions: dict[str, str]
    scope_note: str
    authorized_session_commit: str | None = None
    harness_source_digest: str | None = None
    harness_source_files: tuple[str, ...] = PHASE_A_HARNESS_SOURCE_FILES_V1
    per_launch_hard_usd: float | None = None
    provenance_commit: str | None = None
    version: int = 1

    #: This is the one artifact type that can say yes — and it still cannot say
    #: yes to anything *after* Phase A.
    @property
    def allows_phase_a(self) -> bool:
        return True

    @property
    def automatic_followon_start(self) -> bool:
        """Not a field. Phase A stops for review; nothing chains off it."""
        return False

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA,
            "authorization_id": self.authorization_id,
            "version": self.version,
            "granted_utc": self.granted_utc,
            "granted_by": self.granted_by,
            "plan_id": self.plan_id,
            "phase_a_session_plan_hash": self.plan_hash,
            "phase_a_science_plan_hash": self.science_plan_hash,
            "expected_usd": self.expected_usd,
            "hard_cap_usd": self.hard_cap_usd,
            "authorized_stages": list(self.authorized_stages),
            "stage_conditions": dict(self.stage_conditions),
            "scope_note": self.scope_note,
            "phase_a_authorized": self.allows_phase_a,
            "automatic_followon_start": self.automatic_followon_start,
            "authorized_session_commit": self.authorized_session_commit,
            "harness_source_digest": self.harness_source_digest,
            "harness_source_files": list(self.harness_source_files),
            "per_launch_hard_usd": self.per_launch_hard_usd,
            "provenance_commit": self.provenance_commit,
            "enforcement": (
                "the launcher loads this artifact and refuses to create a pod "
                "whose priced hard threshold exceeds hard_cap_usd or "
                "per_launch_hard_usd, refuses a stage not in authorized_stages, "
                "refuses a harness that does not digest to the authorized value, "
                "and has no code path to any experiment after Phase A"),
        }
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    # -- the checks the launcher and driver call --------------------------
    def require_plan(self, plan_hash: str) -> None:
        if plan_hash != self.plan_hash:
            raise AuthorizationError(
                f"this authorization is bound to Phase-A session plan "
                f"{self.plan_hash} but the plan about to run hashes to "
                f"{plan_hash}. An authorization does not transfer to a plan "
                "that changed.")

    def require_science_plan(self, science_plan_hash: str) -> None:
        """The session plan and the science plan move independently."""
        if science_plan_hash != self.science_plan_hash:
            raise AuthorizationError(
                f"this authorization is bound to science plan "
                f"{self.science_plan_hash} but the successive-halving plan about "
                f"to run hashes to {science_plan_hash}. A threshold, seed, "
                "survivor count or selection rule moved after the grant.")

    def require_stage(self, stage: int) -> None:
        if stage not in self.authorized_stages:
            raise AuthorizationError(
                f"stage {stage} is not in the authorized set "
                f"{list(self.authorized_stages)}")

    def require_within_cap(self, projected_usd: float, *, what: str = "") -> None:
        if projected_usd > self.hard_cap_usd:
            raise AuthorizationError(
                f"{what or 'projected spend'} ${projected_usd:.4f} exceeds the "
                f"authorized hard cap ${self.hard_cap_usd:.4f}")

    def require_within_launch_limit(self, hard_usd: float, *, what: str = "") -> None:
        if self.per_launch_hard_usd is None:
            return
        if hard_usd > self.per_launch_hard_usd:
            raise AuthorizationError(
                f"{what or 'planned hard threshold'} ${hard_usd:.4f} exceeds the "
                f"per-launch limit ${self.per_launch_hard_usd:.4f}")

    def require_harness(self, repo_root: str | Path = ".") -> dict[str, Any]:
        observed = phase_a_harness_digest(repo_root, files=self.harness_source_files)
        if self.harness_source_digest is None:
            raise AuthorizationError(
                "this authorization declares no harness_source_digest, so it "
                "cannot authorize any executable. Re-issue it against the "
                f"rehearsed harness (observed {observed['digest']}).")
        if observed["digest"] != self.harness_source_digest:
            raise AuthorizationError(
                f"the Phase-A harness on disk digests to {observed['digest']} but "
                f"this authorization was granted against "
                f"{self.harness_source_digest}. The rehearsed harness and the "
                "executable harness differ; re-rehearse and re-issue rather than "
                "running an unrehearsed harness against a paid authorization.")
        return observed

    def refuse_followon(self, what: str = "a follow-on experiment") -> None:
        raise AuthorizationError(
            f"{what} is separately unauthorized and is not reachable from Phase A. "
            "Collect the evidence, tear down, report, and STOP for review.")

    @classmethod
    def load(cls, path: str | Path) -> "PhaseAAuthorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has "
                "been edited since it was granted")
        # Refused by SCHEMA, not by convention: this is what stops a narrow
        # `SpendAuthorization` artifact being pressed into service as a Phase-A
        # grant simply because someone added a `phase_a_authorized` key to it.
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. "
                "Only an artifact issued as a Phase-A authorization can permit "
                "Phase A; a spend authorization cannot be reinterpreted as one.")
        if not raw.get("phase_a_authorized"):
            raise AuthorizationError(
                f"{path} carries the Phase-A schema but does not assert "
                "phase_a_authorized; refusing to infer permission from a schema "
                "name alone")
        if raw.get("automatic_followon_start"):
            raise AuthorizationError(
                "this artifact claims an automatic follow-on start, which it "
                "cannot grant; Phase A stops for review")
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
                                       or PHASE_A_HARNESS_SOURCE_FILES_V1),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


#: The grant this session issues. Figures come from the launcher's own
#: `make_plan`, which is what `plan_session` computes before a pod can exist —
#: NOT from `logs/autoinit_phase_a_repricing.md`, which priced search and probes
#: and never priced the session around them.
#:
#: `authorized_session_commit` and `harness_source_digest` are filled at issue
#: time by `scripts/autoinit/issue_phase_a_authorization.py`, against the
#: committed tree. Editing any declared harness file invalidates it by design:
#: re-rehearse, re-commit, re-issue.
PHASE_A_AUTHORIZATION = PhaseAAuthorization(
    authorization_id="autoinit.phase_a.PLACEHOLDER",
    granted_utc="PLACEHOLDER",
    granted_by=(
        "PLACEHOLDER. The figures below are the launcher's priced plan, not a "
        "transcription, and they require a cumulative project cap of at least "
        "$216.2266 against a spend of $193.1783. The maintainer RECOMMENDED "
        "$217.00 on 2026-08-17; that cap has NOT been formally approved or "
        "recorded in BUDGET_LEDGER.md, and until it is, this template must not "
        "be issued. The superseded $20.0126 grant basis (project cap $213.00) "
        "is recorded in autoinit_phase_a_fallback_audit.json."),
    plan_id=PHASE_A_PLAN_V1.plan_id,
    plan_hash=PHASE_A_PLAN_V1.plan_hash,
    # Filled at issue time from the frozen plan on disk, so a threshold that
    # moved after freezing cannot be authorized by a stale constant.
    science_plan_hash="PLACEHOLDER",
    #: `PhaseA.make_plan`: 12 priced probes (rung 1's 6, rung 2's 3, and headroom
    #: for the conditional seed-sc rung so the watchdog cannot kill a legitimate
    #: one), 1023 steps each at the measured 61.55 min end-to-end, plus setup,
    #: attestation, selection, collection, a 10% contingency and a 20-minute
    #: artifact-recovery reserve.
    #: plus two named SOFT-STOP reserves: 147.7683 min for the reference-cache
    #: fallback and 36.2158 min for the beam-6 search pricing correction. They
    #: sit before the soft stop, not after it, because the fallback is consumed
    #: inside stage 1 and a hard-only reserve would leave `afford()` refusing the
    #: conditional seed-sc rung. Derivation: autoinit_phase_a_fallback_audit.json.
    #: $23.048325 is the priced figure; the cap is the 4-dp CEILING of it,
    #: because `require_within_cap` refuses `projected > cap` and a cap rounded
    #: down would make the launcher refuse its own plan by 2.5e-5 dollars.
    expected_usd=17.8933,
    hard_cap_usd=23.0484,
    #: One session, so the per-launch limit IS the cap. Stated anyway: it is the
    #: check that stopped a single continuation run from spending the cumulative
    #: allowance of five.
    per_launch_hard_usd=23.0484,
    authorized_stages=(0, 1, 2, 3, 4, 5),
    stage_conditions={
        "0": "attestation; frozen assets; assert_preregistered against the "
             "frozen science plan; both thresholds must be materialized",
        "1": "beam search; 5 admissible leaves or STOP and report the shortfall",
        "2": "recovery rung 1 on seed sa: 5 searched leaves + the injected "
             "canonical control, then the survivor selection and the leaf "
             "retention record",
        "3": "recovery rung 2 on seed sb: 2 survivors + the control",
        "4": "conditional tie-break on seed sc, ONLY for finalists inside the "
             "preregistered equivalence interval; no fourth seed",
        "5": "selection and report; unresolved_equivalence is a RESULT",
        "teardown": "collect, fetch the finalists, delete the pod, confirm from "
                    "the provider, STOP for review",
    },
    scope_note=(
        "Phase A only, one launcher invocation. This artifact authorizes the "
        "SPEND and the STAGES; it is not by itself an instruction to launch. The "
        "maintainer's 2026-08-15 message that requested it also says 'Do not "
        "launch Phase A yet', so the run waits for a separate explicit go. "
        "Nothing here permits retraining the permanent controls, changing the "
        "frozen search or recovery design, a fourth seed, or any follow-on "
        "experiment: Phase A is a terminus and stops for review on every path, "
        "including unresolved_equivalence."),
    harness_source_files=PHASE_A_HARNESS_SOURCE_FILES_V1,
)


@dataclass(frozen=True)
class PhaseAScope:
    """What a Phase-A session is permitted to do, stated once."""

    trains_anything: bool = True
    retrains_permanent_controls: bool = False
    reaches_any_followon: bool = False
    #: The permanent controls are INPUTS. Phase A probes the retained canonical
    #: control checkpoint; it does not re-execute the recipe that made it.
    control_is_injected_by_hash: bool = True
    searched_leaves: int = 5
    survivors: int = 2
    seeds: tuple[int, ...] = (SEED_SA, SEED_SB)
    conditional_tie_break_seed: int = SEED_SC
    battery: str = "recovery_search_v2"
    state_eval_suite: str = "state_eval_v1"
    notes: dict[str, str] = field(default_factory=lambda: {
        "what_training_means_here": (
            "Phase A trains 9 recovery probes (12 if the tie-break rung runs) at "
            "1023 steps each. It does NOT retrain the two permanent controls, "
            "which were trained once and are injected by frozen hash"),
        "control_may_win": (
            "'the incumbent won, AutoInitializer v1 did not improve recovered "
            "behaviour' is a reachable conclusion, and the selection is "
            "symmetric so that it stays reachable"),
        "no_winner_is_a_result": (
            "unresolved_equivalence after seed sc is the finding, not a failure "
            "to be resolved by a fourth seed or a lexicographic tie-break"),
        "terminus": (
            "Phase A stops for review. No follow-on experiment, full recovery of "
            "a winner, or Phase B starts from this session"),
    })

    def as_dict(self) -> dict[str, Any]:
        return {"trains_anything": self.trains_anything,
                "retrains_permanent_controls": self.retrains_permanent_controls,
                "reaches_any_followon": self.reaches_any_followon,
                "control_is_injected_by_hash": self.control_is_injected_by_hash,
                "searched_leaves": self.searched_leaves,
                "survivors": self.survivors,
                "seeds": list(self.seeds),
                "conditional_tie_break_seed": self.conditional_tie_break_seed,
                "battery": self.battery,
                "state_eval_suite": self.state_eval_suite,
                "notes": dict(self.notes),
                "session_plan_hash": PHASE_A_PLAN_V1.plan_hash}


PHASE_A_SCOPE = PhaseAScope()


def phase_a_manifest() -> dict[str, Any]:
    """The Phase-A session plan and its scope, as one hashable record."""
    payload = {"plan": PHASE_A_PLAN_V1.as_dict(),
               "scope": PHASE_A_SCOPE.as_dict(),
               "harness_source_files": list(PHASE_A_HARNESS_SOURCE_FILES_V1)}
    payload["manifest_sha256"] = sha256_json(payload)
    return payload
