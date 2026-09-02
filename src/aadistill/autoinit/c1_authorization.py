"""The Phase-C1 authorization type and budget, both derived, neither transcribed.

Two properties make this a *type* rather than a policy comment.

**It cannot authorize anything else.** `allows_phase_a` and `allows_beam_search`
are hard `False`, and `load` refuses any artifact whose schema is not this one.
A Phase-A or continuation grant carries a ceiling derived for different work over
a different harness; accepting one here would certify code this session does not
run and price it for work it does not do. That refusal is by schema, at load.

**Its ceiling is derived from the accepted pricing record, not typed in.**
`c1_budget_spec()` reads `logs/phase_c1_pricing.json` and builds the `BudgetSpec`
from it, so there is exactly one place the enforceable ceiling comes from. A second
hand-maintained copy is how a session comes to be authorized for one figure and
priced at another.

The step time deserves its own note, because it trips the budget module's floor
guard on purpose. `plan_session` refuses a step time below the measured
`4.15 s/step` unless the caller says why. This workload's rate is **measured, on
this exact recipe, sixteen times**: every retained 0.86M probe ran 1023 steps in
60.98–64.91 minutes, i.e. 3.58–3.81 s/step. The 4.15 figure comes from E6b at a
different scale. So the reason is supplied and lands in the session record, which
is exactly what that mechanism is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infrastructure.budget import Phase
from ..infrastructure.manifest import sha256_json
from ..infrastructure.session import BudgetSpec
from .authorization import AuthorizationError
from .phase_a import PhaseAAuthorization

SCHEMA = "aadistill.autoinit.c1_authorization/v1"

#: Every file whose bytes decide what the paid C1 session executes. The
#: authorization measures THIS set; a grant that declares a different one is
#: certifying different code.
#:
#: The two artifact specs are in here for the same reason the collector is.
#: `collect_artifacts.py` is only a spec interpreter — what actually decides
#: which evidence survives teardown is the declared pattern list, and a session
#: whose evidence contract can be edited without moving the harness digest has an
#: unmeasured mutable input at exactly the point where loss is irreversible. They
#: are configuration by file type and executable by consequence.
C1_HARNESS_SOURCE_FILES_V1: tuple[str, ...] = (
    # the session, end to end
    "scripts/pod/autoinit_c1_launch.py",
    "scripts/pod/autoinit_c1_driver.py",
    "scripts/pod/setup.sh",
    "scripts/pod/start_job.py",
    "scripts/pod/watchdog.py",
    "scripts/pod/collect_artifacts.py",
    "scripts/pod/autoinit_science_inputs.py",
    # transport: whether the pod can OBTAIN the authorized commit at all. Attempt
    # 1 passed every content gate and died at SETUP_RC=1 on a bundle that did not
    # exist. The staging tool is measured too: it decides what the pod checks out.
    "src/aadistill/autoinit/c1_bundle.py",
    "scripts/autoinit/stage_c1_bundle.py",
    # what the collector is told to save, and what it may skip on failure
    "configs/autoinit/c1_artifacts.json",
    "configs/autoinit/c1_artifacts_failed.json",
    # the C1 scoring binding, and every file that can move a C1 number.
    # `recovery_search_scoring@v2` cannot run on this battery, so C1 declares
    # `c1_confirmation_scoring@v1`; the three files V2 omits and this set does not
    # are audit_tool_scoring, data/tools and data/verify.
    "scripts/autoinit/score_c1_confirmation.py",
    "src/aadistill/autoinit/c1_scoring.py",
    "scripts/autoinit/audit_tool_scoring.py",
    "src/aadistill/data/tools.py",
    "src/aadistill/data/verify.py",
    # the evaluation packaging and the six-probe aggregation
    "src/aadistill/autoinit/c1_packaging.py",
    "src/aadistill/autoinit/c1_probe_results.py",
    "src/aadistill/autoinit/device_handoff.py",
    "src/aadistill/models/teacher.py",
    "scripts/pod/autoinit_engine_probe.py",
    "scripts/training/train_stage3.py",
    # the C1 science
    "src/aadistill/autoinit/c1_session.py",
    "src/aadistill/autoinit/c1_isolation.py",
    "src/aadistill/autoinit/c1_authorization.py",
    "src/aadistill/autoinit/authorization.py",
    # both package __init__ files execute on import, and the operators one
    # decides what `attention_activation` resolves to
    "src/aadistill/autoinit/__init__.py",
    "src/aadistill/autoinit/operators/__init__.py",
    "src/aadistill/autoinit/fixed_path.py",
    "src/aadistill/autoinit/operators/attention_activation.py",
    "src/aadistill/init/attention_stats.py",
    # the operators the fixed path actually applies
    "src/aadistill/autoinit/operators/attention.py",
    "src/aadistill/autoinit/operators/base.py",
    "src/aadistill/autoinit/operators/depth.py",
    "src/aadistill/autoinit/operators/ffn.py",
    "src/aadistill/autoinit/operators/width.py",
    "src/aadistill/autoinit/operators/_common.py",
    "src/aadistill/init/contribution.py",
    "src/aadistill/init/project.py",
    "src/aadistill/init/sandwich.py",
    "src/aadistill/init/collect.py",
    # identity, calibration and artifact machinery the gates depend on
    "src/aadistill/autoinit/adapters/qwen3.py",
    "src/aadistill/autoinit/arch.py",
    "src/aadistill/autoinit/artifact.py",
    "src/aadistill/autoinit/calibration.py",
    "src/aadistill/autoinit/device.py",
    "src/aadistill/autoinit/metrics.py",
    "src/aadistill/autoinit/stats.py",
    # the recovery-probe and scoring paths the six probes invoke
    "src/aadistill/autoinit/recovery.py",
    "src/aadistill/autoinit/generation.py",
    "scripts/autoinit/score_recovery_search.py",
    "scripts/evaluation/uncapped_eval.py",
    "src/aadistill/evaluation/usable_rollout.py",
    "src/aadistill/evaluation/strict_answer.py",
    "src/aadistill/evaluation/behavior.py",
    "src/aadistill/evaluation/capability.py",
    "src/aadistill/evaluation/degeneration.py",
    # session infrastructure this session runs through
    "src/aadistill/infrastructure/session.py",
    "src/aadistill/infrastructure/session_runner.py",
    "src/aadistill/infrastructure/session_prechecks.py",
    "src/aadistill/infrastructure/budget.py",
    "src/aadistill/infrastructure/manifest.py",
    "src/aadistill/infrastructure/provider.py",
    "src/aadistill/infrastructure/artifact_gate.py",
)


def c1_harness_digest(repo_root: str | Path = ".",
                      files: tuple[str, ...] = C1_HARNESS_SOURCE_FILES_V1,
                      ) -> dict[str, Any]:
    """The digest a C1 grant binds. A missing declared file raises.

    Same rule and same failure mode as every other source-identity set in this
    project: refusing is the point, because a digest computed over a smaller set
    silently describes less code than will run.
    """
    from ..infrastructure.manifest import sha256_file

    root = Path(repo_root)
    entries = []
    for rel in sorted(files):
        p = root / rel
        if not p.is_file():
            raise AuthorizationError(
                f"declared C1 harness source {rel!r} is missing; refusing to "
                "produce a digest that describes a smaller harness than the one "
                "that would run")
        entries.append({"path": rel, "sha256": sha256_file(p),
                        "bytes": p.stat().st_size})
    import hashlib

    digest = hashlib.sha256(
        "".join(f"{e['path']}:{e['sha256']}\n" for e in entries).encode()).hexdigest()
    return {"digest": digest, "files": entries, "n_files": len(entries),
            "rule": "sha256 over sorted 'path:sha256' lines of the declared set"}


@dataclass(frozen=True)
class C1Authorization(PhaseAAuthorization):
    """Permits exactly the Phase-C1 fixed-path ATTENTION isolation.

    Structurally a `PhaseAAuthorization` — same commit binding, same harness
    rule, same hash-of-itself check — and a different **type**, so neither can
    stand in for the other.
    """

    harness_source_files: tuple[str, ...] = C1_HARNESS_SOURCE_FILES_V1

    @property
    def allows_phase_a(self) -> bool:
        """Never. C1 replays one frozen path; it cannot start Phase A."""
        return False

    @property
    def allows_beam_search(self) -> bool:
        """Never. There is no search anywhere in this session."""
        return False

    @property
    def authorizes_c1_isolation(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        payload = dict(super().as_dict())
        payload["schema"] = SCHEMA
        # From the properties, never a literal: a document that disagreed with
        # the object that wrote it would be worse than no document.
        payload["allows_phase_a"] = self.allows_phase_a
        payload["allows_beam_search"] = self.allows_beam_search
        payload["authorizes_c1_isolation"] = self.authorizes_c1_isolation
        payload["scope"] = (
            "2 arms x 3 fresh seeds = 6 fixed 0.86M recovery probes on one frozen "
            "path, evaluated once each on c1_confirmation_v1. No search, no "
            "ranking, no successive halving, no tie-breaking, no arm elimination, "
            "and no formal Stage-2/3 recovery.")
        payload.pop("authorization_sha256", None)
        payload["authorization_sha256"] = sha256_json(payload)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "C1Authorization":
        raw = json.loads(Path(path).read_text())
        stated = raw.get("authorization_sha256")
        check = dict(raw)
        check.pop("authorization_sha256", None)
        if stated != sha256_json(check):
            raise AuthorizationError(
                f"{path} does not match its own authorization_sha256; it has been "
                "edited since it was granted")
        if raw.get("schema") != SCHEMA:
            raise AuthorizationError(
                f"{path} declares schema {raw.get('schema')!r}, not {SCHEMA!r}. A "
                "Phase-A, Phase-B or continuation grant measures a different "
                "harness and carries a ceiling derived for different work; it "
                "cannot authorize the C1 isolation.")
        for forbidden in ("allows_phase_a", "allows_beam_search"):
            if raw.get(forbidden):
                raise AuthorizationError(
                    f"{path} claims {forbidden}. C1 replays one frozen path and "
                    "runs no search; an artifact claiming otherwise is not a C1 "
                    "authorization.")
        # Mapped explicitly, not by field name. `as_dict` serialises `plan_hash`
        # as `phase_a_session_plan_hash` and `science_plan_hash` as
        # `phase_a_science_plan_hash`; a by-name filter silently drops both and
        # the constructor then fails on a required argument. Every sibling
        # loader maps them by hand for the same reason.
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
                                       or C1_HARNESS_SOURCE_FILES_V1),
            per_launch_hard_usd=raw.get("per_launch_hard_usd"),
            provenance_commit=raw.get("provenance_commit"),
            version=int(raw.get("version", 1)))


# ---------------------------------------------------------------------------
# the budget, derived from the accepted pricing record
# ---------------------------------------------------------------------------

PRICING_PATH = "logs/phase_c1_pricing.json"

#: 1023 steps at the rate every retained probe of this recipe actually ran.
PROBE_STEPS = 1023


def load_pricing(repo_root: str | Path = ".") -> dict[str, Any]:
    p = Path(repo_root) / PRICING_PATH
    if not p.is_file():
        raise AuthorizationError(
            f"{PRICING_PATH} is missing; the C1 budget is derived from the "
            "accepted pricing record and must not be typed in a second time")
    doc = json.loads(p.read_text())
    stated = doc.get("pricing_sha256")
    check = {k: v for k, v in doc.items() if k != "pricing_sha256"}
    if stated != sha256_json(check):
        raise AuthorizationError(
            f"{PRICING_PATH} does not match its own pricing_sha256; it has been "
            "edited since it was accepted")
    return doc


def _minutes(doc: dict[str, Any], substring: str) -> float:
    for item in doc["line_items"]:
        if substring in item["item"]:
            return float(item["minutes"])
    raise AuthorizationError(f"pricing record has no line item matching {substring!r}")


def c1_budget_spec(repo_root: str | Path = ".") -> BudgetSpec:
    """The `BudgetSpec` whose plan reproduces the accepted pricing record.

    Every number is read out of that record. The step time is back-derived from
    its measured per-probe training minutes rather than restated, so the two
    cannot drift apart.
    """
    doc = load_pricing(repo_root)
    n_probes = int(doc["session_shape"]["probes"])
    train_total = _minutes(doc, "recovery training")
    eval_total = _minutes(doc, "950-prompt evaluation")
    step_seconds = train_total * 60.0 / (n_probes * PROBE_STEPS)

    return BudgetSpec(
        arms=n_probes,
        steps_per_arm=PROBE_STEPS,
        step_seconds=step_seconds,
        step_source=("measured: 16 retained 0.86M probes of this exact recipe ran "
                     "1023 steps in 60.98-64.91 min (3.58-3.81 s/step)"),
        below_floor_reason=(
            "the 4.15 s/step floor is E6b's rate at a different scale. This "
            "workload's rate is measured on this exact recipe sixteen times, in "
            "the retained Phase-A/B probe records, and never exceeded 3.81 s/step"),
        setup_minutes=_minutes(doc, "session setup"),
        eval_minutes_per_arm=eval_total / n_probes,
        transfer_minutes=_minutes(doc, "evidence collection"),
        other_phases=(
            Phase("teacher_fetch_verify", _minutes(doc, "teacher fetch")),
            Phase("staging", _minutes(doc, "staging")),
            Phase("fixed_parent_replay", _minutes(doc, "fixed-parent replay")),
            Phase("incumbent_replay_gate", _minutes(doc, "incumbent replay gate")),
            Phase("attention_stats", _minutes(doc, "attention statistics pass")),
        ),
        contingency_fraction=float(doc["totals"]["contingency_fraction"]),
        artifact_recovery_reserve_minutes=float(
            doc["totals"]["artifact_recovery_reserve_minutes"]),
    )


def c1_hard_ceiling_usd(repo_root: str | Path = ".") -> float:
    """The one place the enforceable ceiling comes from."""
    return float(load_pricing(repo_root)["totals"]["hard_ceiling_usd"])
