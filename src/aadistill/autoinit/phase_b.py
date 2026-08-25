"""Phase B: the calibration-distribution experiment's session plan and grant type.

Phase A asked whether operator *order* matters under one fixed mixture. Phase B
asks a different question — does the preferred composition change when the
**calibration distribution** changes — by searching `calib.domain_balanced@v1`
and `calib.reasoning_heavy@v2` jointly at P=2.

Three things here are deliberately not inherited from Phase A.

**A separate executable-source identity.** `PHASE_A_HARNESS_SOURCE_FILES_V1` is
historical fact about a completed experiment and is not widened. Phase B declares
its own set, covering what a paid P=2 *search* executes. The probe path is not
duplicated into it: training, generation and scoring already have bound source
identities, and `PHASE_B_DELEGATED_IDENTITIES` names which covers what. The goal
is provenance closure, not a maximal file list.

**Comparability is a precondition, not a reuse convenience.**
`generation_runtime_comparability@v2` is what lets Phase B's behavioural results
be judged against the *frozen* Stage-3 feasibility floor and equivalence interval.
If Stage 0 finds the runtime not comparable, those thresholds do not apply to
anything this session could produce, so the session **terminates before any
search or probe**. It does not respond by re-running eight candidates: that would
be a differently-thresholded experiment wearing Phase B's name, and it would
still not restore the frozen interval.

**The candidate set is closed before results exist.** Phase A's three
non-finalists already hold paid `sa` probes, so admitting them would cost nothing
at `sa`. That is not an admission criterion — they lost the Phase-A behavioural
admission step under the same primary metric and gates, and a set that can grow
once results are visible is not a preregistered set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json
from .authorization import AuthorizationError
from .phase_a import sha256_file
from .recovery import PreflightPlan, PreflightStage, SEED_SA, SEED_SB, SEED_SC

SCHEMA = "aadistill.autoinit.phase_b_authorization/v1"

#: What a paid Phase-B SEARCH executes and can change its result through.
#:
#: Derived from the import closure of the search entry point, then curated: every
#: module here can alter search state, ranking, materialization, measurement or
#: the terminal output. `adapters/` is present because the package `__init__`
#: registers it and the adapter decides shapes, projections and statistics
#: collection — the AST closure of `search.py` alone does not reach it.
#:
#: Deliberately ABSENT, and why:
#:
#: * `reweight.py` and `scripts/data/build_reasoning_heavy_calibration.py` — the
#:   pod never runs them. It consumes an already-materialized mixture bound by
#:   `profile_hash` + `content_sha256` + `items_file_sha256`. They are artifact
#:   provenance, and putting them here would digest a builder that does not
#:   execute while implying that it does.
#: * the probe path — see `PHASE_B_DELEGATED_IDENTITIES`.
PHASE_B_EXECUTABLE_SOURCE_FILES_V1: tuple[str, ...] = (
    "scripts/autoinit/load_state_eval.py",
    "scripts/autoinit/phase_a_frozen.py",
    "scripts/autoinit/phase_a_search.py",
    "scripts/autoinit/verify_frozen_assets.py",
    "scripts/autoinit/write_preregistration.py",
    "scripts/pod/autoinit_engine_probe.py",
    "scripts/pod/autoinit_phase_a_driver.py",
    "scripts/pod/autoinit_phase_a_launch.py",
    "scripts/pod/autoinit_phase_b_driver.py",
    "scripts/pod/autoinit_phase_b_launch.py",
    "scripts/pod/autoinit_preflight_setup.sh",
    "scripts/pod/autoinit_science_inputs.py",
    "scripts/pod/collect_artifacts.py",
    "scripts/pod/watchdog.py",
    "src/aadistill/autoinit/__init__.py",
    "src/aadistill/autoinit/adapters/__init__.py",
    "src/aadistill/autoinit/adapters/qwen3.py",
    "src/aadistill/autoinit/arch.py",
    "src/aadistill/autoinit/artifact.py",
    "src/aadistill/autoinit/authorization.py",
    "src/aadistill/autoinit/calibration.py",
    "src/aadistill/autoinit/cost.py",
    "src/aadistill/autoinit/datasets.py",
    "src/aadistill/autoinit/device.py",
    "src/aadistill/autoinit/device_handoff.py",
    "src/aadistill/autoinit/generation.py",
    "src/aadistill/autoinit/generation_compat.py",
    "src/aadistill/autoinit/leaf_durability.py",
    "src/aadistill/autoinit/metrics.py",
    "src/aadistill/autoinit/operators/__init__.py",
    "src/aadistill/autoinit/operators/_common.py",
    "src/aadistill/autoinit/operators/attention.py",
    "src/aadistill/autoinit/operators/base.py",
    "src/aadistill/autoinit/operators/composite.py",
    "src/aadistill/autoinit/operators/depth.py",
    "src/aadistill/autoinit/operators/ffn.py",
    "src/aadistill/autoinit/operators/width.py",
    "src/aadistill/autoinit/phase_a.py",
    "src/aadistill/autoinit/phase_b.py",
    "src/aadistill/autoinit/ranking.py",
    "src/aadistill/autoinit/search.py",
    "src/aadistill/autoinit/state.py",
    "src/aadistill/autoinit/stats.py",
    "src/aadistill/data/extra_stream.py",
    "src/aadistill/infrastructure/artifact_gate.py",
    "src/aadistill/infrastructure/budget.py",
    "src/aadistill/infrastructure/log_relay.py",
    "src/aadistill/infrastructure/manifest.py",
    "src/aadistill/infrastructure/provider.py",
    "src/aadistill/infrastructure/remote.py",
    "src/aadistill/infrastructure/session.py",
    "src/aadistill/infrastructure/session_prechecks.py",
    "src/aadistill/infrastructure/session_runner.py",
    "src/aadistill/init/contribution.py",
    "src/aadistill/init/project.py",
    "src/aadistill/init/sandwich.py",
)
#: Bumped when the driver and launcher joined the set.
PHASE_B_SOURCE_SET_VERSION = 2

#: What covers the rest of the paid session, so "not in the digest" never means
#: "unaccounted for". Each is an existing, independently bound source identity.
PHASE_B_DELEGATED_IDENTITIES: dict[str, str] = {
    "probe training": "recovery.trainer_source_digest over TRAINER_SOURCE_FILES_V1",
    "probe generation": ("generation.generation_source_digest over "
                         "GENERATION_SOURCE_FILES_V1, inside the attested "
                         "evaluation protocol"),
    "probe scoring": ("recovery.recovery_scoring_contract over "
                      "RECOVERY_SCORING_FILES_V2, which includes recovery.py"),
    "the calibration mixtures": ("profile_hash for the spec and content_sha256 + "
                                 "items_file_sha256 for the sampled bytes; the "
                                 "builder is provenance, not runtime"),
    "the state-evaluation suite": "state_eval@v1 content and manifest hashes",
    "the recovery battery": "recovery_search_v2 content and manifest hashes",
    "the selection rules": "the frozen science plan hash",
}

#: Empty since 2026-08-26, when the driver and launcher were written and joined
#: the set above. It stays as a declared field rather than being deleted: the
#: launcher's precheck refuses to create a pod while anything is listed here, so
#: a future uncovered executable fails closed instead of being forgotten.
PHASE_B_UNCOVERED: tuple[str, ...] = ()


def phase_b_source_digest(repo_root: str | Path = ".", *,
                          files: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Digest the declared Phase-B executable source. Fails closed on a gap.

    Same rule and same failure mode as its Phase-A counterpart: a missing declared
    file raises rather than yielding a digest over a smaller executable than the
    one that runs.
    """
    root = Path(repo_root)
    declared = tuple(files) if files is not None else PHASE_B_EXECUTABLE_SOURCE_FILES_V1
    entries = []
    for rel in sorted(declared):
        path = root / rel
        if not path.is_file():
            raise AuthorizationError(
                f"declared Phase-B executable source {rel!r} is missing; refusing "
                "a digest over a smaller executable than the one that runs")
        entries.append({"path": rel, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "set_version": PHASE_B_SOURCE_SET_VERSION,
            "files": entries,
            "rule": ("sha256 over sorted 'path:sha256' lines of the declared "
                     "Phase-B executable source set"),
            "delegated": dict(PHASE_B_DELEGATED_IDENTITIES),
            "not_yet_covered": list(PHASE_B_UNCOVERED)}


# --- the candidate set, closed before any Phase-B result exists -------------

#: Admitted to the cross-phase behavioural comparison.
PHASE_B_SEARCHED_LEAVES = 5
PHASE_A_IMPORTED_FINALISTS: tuple[str, ...] = ("cca699c93f34", "85bde4ded2c3")
CANONICAL_CONTROL = "control-qwen"
SURVIVORS_AT_SB = 2

#: Excluded, and why — recorded here so it cannot be reopened once Phase-B
#: results are visible and one of them looks convenient.
PHASE_A_EXCLUDED_LEAVES: dict[str, str] = {
    "158b96cf651f": "lost the Phase-A rung-1 behavioural admission step",
    "281a02c3ac18": "lost the Phase-A rung-1 behavioural admission step",
    "4e429f7ed722": "lost the Phase-A rung-1 behavioural admission step",
}
PHASE_A_EXCLUSION_RULE = (
    "These three hold VERIFIED sa probes and are retained off-pod, so admitting "
    "them would cost nothing at sa. Zero marginal cost is not an admission "
    "criterion. They were excluded at Phase-A rung 1 under the same primary "
    "metric and the same feasibility and catastrophic gates that Phase B applies, "
    "and a candidate set that can grow after results are visible is not a "
    "preregistered set.")


PHASE_B_PLAN_V1 = PreflightPlan(
    plan_id="autoinit.v1.phase_b_session",
    version=1,
    stages=(
        PreflightStage(
            stage=0, name="attestation, comparability gate and evidence binding",
            blocking=True,
            purpose=("establish that this session measures under a runtime the "
                     "frozen Stage-3 thresholds still apply to, and that the "
                     "historical evidence it intends to cite is the evidence it "
                     "verified at $0"),
            produces=("runtime fingerprint including image digest",
                      "attested evaluation protocol over recovery_search_v2",
                      "comparable_generation_identity under "
                      "generation_runtime_comparability@v2",
                      "frozen-asset verification for state_eval_v1 and "
                      "recovery_search_v2",
                      "both calibration profiles resolved and hash-verified: "
                      "spec identity AND materialized content identity",
                      "the Phase-B executable-source digest",
                      "re-verification of the historical probe reconstruction "
                      "record against the probe bytes on the pod",
                      "assert_preregistered against the frozen Phase-B plan"),
            stop_conditions=(
                "the comparable identity differs from Phase A's -> STOP AND "
                "TERMINATE before any search or probe. The frozen feasibility "
                "floor and equivalence interval were materialized under Phase A's "
                "runtime and do not describe this one, so nothing this session "
                "could produce would be judgeable against them",
                "comparability failure is NOT answered by re-running all eight "
                "candidates: that is a differently-thresholded experiment, and it "
                "would not restore the frozen interval either",
                "Stage 3 controls are NOT rematerialized and thresholds are NOT "
                "redefined under any outcome of this stage",
                "a calibration profile's content hash does not re-derive from its "
                "items -> STOP: profile_hash identifies the spec, not the bytes",
                "the historical probe record does not verify -> STOP rather than "
                "citing evidence whose reconstruction was never proved",
                "a frozen asset does not match its preregistered constant -> STOP",
                "the executing science plan does not hash to the frozen one -> STOP")),
        PreflightStage(
            stage=1, name="joint P=2 beam search over both calibration profiles",
            blocking=True,
            purpose=("search the two calibration distributions JOINTLY, so the "
                     "domain-balanced arm competes for the same beam slots as the "
                     "reasoning-heavy arm and pruning is decided across both"),
            produces=("one measured, hash-bound state per expansion",
                      "each state's calibration profile recorded on its steps",
                      "level records including every prune reason",
                      "the epsilon-Pareto ranking at each pruned level",
                      "5 admissible complete leaves, the Phase-B Top-5",
                      "the search journal, resumable by content-derived state id"),
            stop_conditions=(
                "Phase-A leaves are NOT injected to restrict the space: P=2 "
                "changes pruning, so this is a fresh joint search",
                "fewer than 5 admissible complete leaves exist -> STOP and report "
                "the shortfall rather than shrinking N",
                "a materialized state fails canonical reload, spec match, "
                "parameter count or finiteness -> STOP",
                "the state evaluation suite hash differs from the attested one -> "
                "STOP: the beam would rank on a different suite's questions")),
        PreflightStage(
            stage=2, name="cross-phase rung 1 on seed sa", blocking=True,
            purpose=("probe the Phase-B Top-5 on seed sa, and admit the imported "
                     "Phase-A finalists and canonical control on their VERIFIED "
                     "historical sa evidence rather than re-buying it"),
            produces=("5 probes at 1023 steps each on seed sa, for the new leaves",
                      "3 imported sa results, cited by verified reconstruction",
                      "the global rung-1 selection over all 8 candidates",
                      "per-probe journal entries, resumable"),
            stop_conditions=(
                "the canonical control is absent -> STOP",
                "an imported result whose reconstruction is not verified is NOT "
                "cited; the candidate is re-probed or the session stops",
                "the three excluded Phase-A leaves are NOT admitted, whatever "
                "their retained sa evidence would have cost",
                "capability schema validation fails -> scoring defect, STOP")),
        PreflightStage(
            stage=3, name="cross-phase rung 2 on seed sb", blocking=True,
            purpose=("second seed for the two globally best searched candidates "
                     "and the control, because the behaviour metric's seed-only "
                     "spread is 0.1290"),
            produces=("probes on seed sb for those finalists that lack a verified "
                      "sb — between 0 and 2, plus the control if it lacks one",
                      "pooled_counts@v2 aggregates over sa and sb",
                      "the final selection under the frozen equivalence rule"),
            stop_conditions=(
                "the control did not advance -> STOP: it advances unconditionally",
                "the two survivors are selected GLOBALLY by correct_overall among "
                "feasible candidates, not within the Phase-B leaves alone",
                "capability schema validation fails -> STOP")),
        PreflightStage(
            stage=4, name="conditional tie-break on seed sc", blocking=False,
            purpose=("resolve finalists inside the preregistered equivalence "
                     "interval after two seeds — and only those, and only where "
                     "no verified sc already exists"),
            produces=("probes on seed sc for tied finalists that lack one",
                      "imported sc results where verified",
                      "pooled aggregates over sa, sb and sc",
                      "the resolved winner, or unresolved_equivalence"),
            stop_conditions=(
                "the rung-2 selection did not request a tie-break -> this stage "
                "does not run, which is not a failure",
                "no fourth seed is requested under any outcome")),
        PreflightStage(
            stage=5, name="selection and report", blocking=False,
            purpose=("record which initialization won across both phases, that "
                     "the canonical control won, or that the comparison resolved "
                     "no unique behavioural winner"),
            produces=("the final cross-phase selection record with every exclusion",
                      "which calibration profile each admitted leaf was searched "
                      "under, so the distribution-sensitivity question is answerable",
                      "feasibility and catastrophic-gate outcomes per candidate",
                      "behaviour and correctness reported on separate axes",
                      "the Phase-B result artifact"),
            stop_conditions=(
                "a tie surviving seed sc is reported as unresolved_equivalence "
                "and is a RESULT, not a failure",
                "search-side KL and NLL may NOT break a behavioural tie",
                "the canonical Stage-1 NLL diagnostic may NOT break a tie and is "
                "not run by this session",
                "no follow-on experiment starts from this session")),
    ))


@dataclass(frozen=True)
class PhaseBAuthorization:
    """What a named maintainer permitted for a Phase-B search, bound by hash.

    A distinct type, not a subclass of `PhaseAAuthorization` and not of
    `SpendAuthorization`. Phase A is complete; a Phase-A grant must not be
    substitutable for a Phase-B one, and `allows_phase_a` stays False by type so
    no Phase-B artifact can ever be pressed into service to restart Phase A.
    """

    authorization_id: str
    granted_utc: str
    granted_by: str
    plan_id: str
    plan_hash: str
    science_plan_hash: str
    #: Both calibration identities. `profile_hash` fixes the specification;
    #: `content_sha256` fixes the sampled bytes. They are separate fields because
    #: they answer different questions and one does not imply the other — on this
    #: pool the seed does not even reach the bytes.
    calibration_profile_hashes: dict[str, str]
    calibration_content_hashes: dict[str, str]
    #: NOT `expected_usd`. No expected-value assumption over survivor identity or
    #: tie-break probability is defined anywhere, so the low figure is a planning
    #: FLOOR — what the session costs if reuse holds, the Phase-A finalists
    #: survive sb and no tie-break fires. Calling it "expected" would invite
    #: budgeting against an outcome nobody estimated the probability of.
    planning_floor_usd: float
    hard_cap_usd: float
    authorized_stages: tuple[int, ...]
    stage_conditions: dict[str, str]
    scope_note: str
    authorized_session_commit: str | None = None
    source_digest: str | None = None
    source_files: tuple[str, ...] = PHASE_B_EXECUTABLE_SOURCE_FILES_V1
    per_launch_hard_usd: float | None = None
    provenance_commit: str | None = None
    version: int = 1

    @property
    def allows_phase_b(self) -> bool:
        return True

    @property
    def allows_phase_a(self) -> bool:
        """Phase A is complete. Nothing issued here can reopen it."""
        return False

    @property
    def automatic_followon_start(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA,
            "authorization_id": self.authorization_id,
            "version": self.version,
            "granted_utc": self.granted_utc,
            "granted_by": self.granted_by,
            "plan_id": self.plan_id,
            "phase_b_session_plan_hash": self.plan_hash,
            "phase_b_science_plan_hash": self.science_plan_hash,
            "calibration_profile_hashes": dict(sorted(
                self.calibration_profile_hashes.items())),
            "calibration_content_hashes": dict(sorted(
                self.calibration_content_hashes.items())),
            "planning_floor_usd": self.planning_floor_usd,
            "hard_cap_usd": self.hard_cap_usd,
            "authorized_stages": list(self.authorized_stages),
            "stage_conditions": dict(self.stage_conditions),
            "scope_note": self.scope_note,
            "phase_b_authorized": self.allows_phase_b,
            "phase_a_authorized": self.allows_phase_a,
            "automatic_followon_start": self.automatic_followon_start,
            "authorized_session_commit": self.authorized_session_commit,
            "source_digest": self.source_digest,
            "source_files": list(self.source_files),
            "per_launch_hard_usd": self.per_launch_hard_usd,
            "provenance_commit": self.provenance_commit,
            "enforcement": (
                "the launcher loads this artifact and refuses to create a pod "
                "whose priced hard threshold exceeds hard_cap_usd or "
                "per_launch_hard_usd, refuses a stage not in authorized_stages, "
                "refuses an executable that does not digest to the authorized "
                "value, refuses a calibration mixture whose spec OR content hash "
                "differs, and has no code path to Phase A or to any experiment "
                "after Phase B"),
        }
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    # -- the checks a launcher and driver call ----------------------------
    def require_plan(self, plan_hash: str) -> None:
        if plan_hash != self.plan_hash:
            raise AuthorizationError(
                f"this authorization is bound to Phase-B session plan "
                f"{self.plan_hash} but the plan about to run hashes to "
                f"{plan_hash}. An authorization does not transfer to a plan "
                "that changed.")

    def require_science_plan(self, science_plan_hash: str) -> None:
        if science_plan_hash != self.science_plan_hash:
            raise AuthorizationError(
                f"this authorization is bound to science plan "
                f"{self.science_plan_hash} but the plan about to run hashes to "
                f"{science_plan_hash}. A threshold, seed, survivor count or "
                "selection rule moved after the grant.")

    def require_calibration(self, profile) -> None:
        """Both identities, because neither implies the other.

        A mixture can satisfy its spec hash and be different bytes — a rebuild
        under a different rule version, a truncated file, a rendering change. The
        spec says what was intended; the content says what will actually be fed
        to the operators.
        """
        qid = profile.qualified_id
        expected_spec = self.calibration_profile_hashes.get(qid)
        expected_content = self.calibration_content_hashes.get(qid)
        if expected_spec is None or expected_content is None:
            raise AuthorizationError(
                f"{qid} is not one of the authorized calibration profiles "
                f"{sorted(self.calibration_profile_hashes)}")
        if profile.profile_hash != expected_spec:
            raise AuthorizationError(
                f"{qid} specifies {profile.profile_hash[:12]} but this "
                f"authorization was granted against {expected_spec[:12]}")
        if profile.content_sha256 != expected_content:
            raise AuthorizationError(
                f"{qid}'s materialized content is {str(profile.content_sha256)[:12]} "
                f"but this authorization was granted against {expected_content[:12]}. "
                "The spec matching is not enough: profile_hash does not identify "
                "the sampled bytes.")

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

    def require_source(self, repo_root: str | Path = ".") -> dict[str, Any]:
        observed = phase_b_source_digest(repo_root, files=self.source_files)
        if self.source_digest is None:
            raise AuthorizationError(
                "this authorization declares no source_digest, so it cannot "
                f"authorize any executable (observed {observed['digest']}).")
        if observed["digest"] != self.source_digest:
            raise AuthorizationError(
                f"the Phase-B executable on disk digests to {observed['digest']} "
                f"but this authorization was granted against {self.source_digest}. "
                "Re-rehearse and re-issue rather than running an unrehearsed "
                "executable against a paid authorization.")
        return observed

    def refuse_followon(self, what: str = "a follow-on experiment") -> None:
        raise AuthorizationError(
            f"{what} is separately unauthorized and is not reachable from Phase B. "
            "Collect the evidence, tear down, report, and STOP for review.")

    @classmethod
    def load(cls, path: str | Path) -> "PhaseBAuthorization":
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
                "Only an artifact issued as a Phase-B authorization can permit "
                "Phase B.")
        if not raw.get("phase_b_authorized"):
            raise AuthorizationError(
                f"{path} carries the Phase-B schema but does not assert "
                "phase_b_authorized")
        if raw.get("phase_a_authorized"):
            raise AuthorizationError(
                "this artifact claims to authorize Phase A, which is complete "
                "and which no Phase-B grant may reopen")
        if raw.get("automatic_followon_start"):
            raise AuthorizationError(
                "this artifact claims an automatic follow-on start, which it "
                "cannot grant; Phase B stops for review")
        return cls(
            authorization_id=raw["authorization_id"],
            granted_utc=raw["granted_utc"], granted_by=raw["granted_by"],
            plan_id=raw["plan_id"],
            plan_hash=raw["phase_b_session_plan_hash"],
            science_plan_hash=raw["phase_b_science_plan_hash"],
            calibration_profile_hashes=dict(raw["calibration_profile_hashes"]),
            calibration_content_hashes=dict(raw["calibration_content_hashes"]),
            planning_floor_usd=float(raw["planning_floor_usd"]),
            hard_cap_usd=float(raw["hard_cap_usd"]),
            authorized_stages=tuple(raw["authorized_stages"]),
            stage_conditions=dict(raw["stage_conditions"]),
            scope_note=raw["scope_note"],
            authorized_session_commit=raw.get("authorized_session_commit"),
            source_digest=raw.get("source_digest"),
            source_files=tuple(raw.get("source_files")
                               or PHASE_B_EXECUTABLE_SOURCE_FILES_V1),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


#: Refused by the issuer, exactly as Phase A's is. A grant is a one-use maintainer
#: decision at a particular cumulative spend; this module is the durable schema.
GRANT_PROSE_REQUIRED = (
    "NO GRANT. This is the Phase-B authorization SCHEMA, not a grant. No Phase-B "
    "grant exists, no cumulative-budget increase has been requested, and issuing "
    "one requires a maintainer decision this module cannot make.")
