"""Governance for the Stage-0 calibration-volume programme: one type, one envelope.

The programme grant (`PROGRAMME_GRANT`) is the maintainer's 2026-10-18 decision:
start Stage-0 calibration-data experiments, funded inside the existing project
cap and the per-session hard cap, with no D-series grant carried over. Every
paid session is a RUN of this experiment with its own one-use authorization,
bound to that session's plan hash and harness digest; the launcher refuses a
pod when any of them disagrees.

What this module adds, and the failure each piece prevents:

* `CalVolAuthorization` -- an action vocabulary of this experiment's session
  kinds, so a battery authorization cannot fund a search;
* `programme_spent` -- the programme's cumulative spend read from its own run
  closeouts, so a retry cannot silently reset the envelope (P12.1);
* `plan_hash` -- the hash of the frozen session plan document, so an edited
  plan cannot run on an old authorization.

Nothing here is reusable core: the type, the envelope and the paths are this
experiment's (P3).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aadistill.governance.authorization import (ActionPolicy, AuthorizationError,
                                                SpendAuthorization,
                                                harness_source_digest)
from aadistill.infrastructure.manifest import sha256_json

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT_ID = "calibration_volume"
STAGE_ID = "0"
EXPERIMENT_LOGS = Path("logs/stages/stage-0/calibration_volume")
PROGRAMME_GRANT = Path("logs/budget/approvals/stage0_calibration_volume_programme_grant.json")
PLANS = Path("configs/stages/stage-0/calibration_volume")

SESSION_KINDS = ("battery", "search", "probes")

#: One policy per session kind: an artifact grants exactly its own kind, and
#: loading it as another kind refuses on the claim it carries.
POLICIES = {k: ActionPolicy(
    policy_id=f"stage0.calibration_volume.{k}",
    allowed=frozenset({k}),
    wire_claims={f"authorizes_{x}": x for x in SESSION_KINDS},
    wire_schema="aadistill.stage0.calibration_volume.authorization/v1",
    plan_hash_key="session_plan_hash",
    enforcement=("the launcher refuses a pod unless this artifact is bound to the "
                 "session plan and the harness on disk, and its hard cap covers "
                 "the planned hard threshold; one artifact funds one run"),
) for k in SESSION_KINDS}


class CalVolAuthorization(SpendAuthorization):
    pass


def load_for(kind: str, path) -> CalVolAuthorization:
    auth = CalVolAuthorization.load(path, policy=POLICIES[kind])
    if not auth.allows(kind):
        raise AuthorizationError(f"{path} does not authorize a {kind} session")
    return auth


#: The executable a session runs, digested into its authorization. Shared
#: infrastructure the pod executes is included: an edited setup script or
#: watchdog is an unrehearsed harness as much as an edited driver.
COMMON_HARNESS = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "scripts/shared/pod/watchdog.py",
    "scripts/shared/pod/collect_artifacts.py",
    "scripts/stages/stage-0/calibration_volume/governance.py",
    "scripts/stages/stage-0/calibration_volume/launch.py",
)
HARNESS = {
    "battery": COMMON_HARNESS + (
        "scripts/stages/stage-0/calibration_volume/battery_driver.py",
        "scripts/stages/stage-0/calibration_volume/run_battery.py",
        "scripts/stages/stage-0/calibration_volume/score_battery.py",
        "scripts/stages/stage-0/calibration_volume/tasks/common.py",
        "scripts/stages/stage-0/calibration_volume/tasks/induction.py",
        "scripts/stages/stage-0/calibration_volume/tasks/composition.py",
        "scripts/stages/stage-0/calibration_volume/tasks/deduction.py",
        "scripts/stages/stage-0/calibration_volume/tasks/analogy.py",
        "scripts/stages/stage-0/calibration_volume/tasks/interactive.py",
        "scripts/stages/stage-0/calibration_volume/tasks/registry.py",
        "scripts/shared/evaluation/uncapped_eval.py",
        "src/aadistill/evaluation/behavior.py",
        "src/aadistill/evaluation/degeneration.py",
        "src/aadistill/evaluation/strict_answer.py",
        "src/aadistill/evaluation/usable_rollout.py",
        "src/aadistill/data/verify.py",
    ),
}


def run_dir(run_id: str) -> Path:
    return EXPERIMENT_LOGS / "runs" / run_id


def authorization_path(run_id: str) -> Path:
    return run_dir(run_id) / "governance" / "authorization.json"


def plan_doc(kind: str, repo_root: Path = REPO) -> dict:
    return json.loads((repo_root / PLANS / f"{kind}_session.json").read_text())


def plan_hash(kind: str, repo_root: Path = REPO) -> str:
    return sha256_json(plan_doc(kind, repo_root))


def grant(repo_root: Path = REPO) -> dict:
    return json.loads((repo_root / PROGRAMME_GRANT).read_text())


def programme_spent(repo_root: Path = REPO) -> dict:
    """Cumulative spend of every run of this experiment, from its closeouts.

    A run that created a provider resource but has no closeout cost is an
    UNKNOWN, and an unknown refuses: the envelope cannot be checked against a
    number nobody has.
    """
    total, runs, unknown = 0.0, [], []
    for d in sorted((repo_root / EXPERIMENT_LOGS / "runs").glob("*")):
        if not d.is_dir():
            continue
        out = d / "closeout" / "outcome.json"
        sess = d / "runtime" / "session.json"
        if out.is_file():
            doc = json.loads(out.read_text())
            usd = (doc.get("money") or {}).get("all_in_usd")
            if usd is None and doc.get("provider_resource_created") is False:
                usd = 0.0
            if usd is None:
                unknown.append(d.name)
                continue
            total += float(usd)
            runs.append({"run_id": d.name, "usd": float(usd)})
        elif sess.is_file() and json.loads(sess.read_text()).get("provider_resource_created"):
            unknown.append(d.name)
    return {"spent_usd": round(total, 4), "runs": runs, "unknown": unknown}


def remaining_envelope(phase: str, repo_root: Path = REPO) -> dict:
    g = grant(repo_root)
    spent = programme_spent(repo_root)
    if spent["unknown"]:
        raise AuthorizationError(
            f"runs {spent['unknown']} created provider resources and record no "
            "closeout cost; reconcile them before anything else is funded")
    phase_spent = sum(r["usd"] for r in spent["runs"]
                      if json.loads((repo_root / run_dir(r["run_id"]) /
                                     "closeout/outcome.json").read_text()
                                    ).get("phase") == phase)
    cap = float(g["phases"][phase]["ceiling_usd"])
    prog = float(g["programme_ceiling_usd"])
    return {"phase": phase, "phase_ceiling_usd": cap,
            "phase_spent_usd": round(phase_spent, 4),
            "phase_remaining_usd": round(cap - phase_spent, 4),
            "programme_ceiling_usd": prog, "programme_spent_usd": spent["spent_usd"],
            "programme_remaining_usd": round(prog - spent["spent_usd"], 4),
            "per_session_cap_usd": float(g["per_session_cap_usd"])}


def git_head(repo_root: Path = REPO) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                          capture_output=True, text=True, check=True).stdout.strip()


def issue(kind: str, run_id: str, *, phase: str, expected_usd: float,
          hard_cap_usd: float, repo_root: Path = REPO, note: str = "") -> Path:
    """Write a one-use authorization for `run_id`, after checking the envelope."""
    if kind not in SESSION_KINDS:
        raise AuthorizationError(f"unknown session kind {kind!r}")
    g = grant(repo_root)
    if kind not in g["phases"][phase]["session_kinds"]:
        raise AuthorizationError(f"phase {phase!r} does not fund {kind!r} sessions")
    if g["phases"][phase].get("requires_preregistration"):
        prereg = repo_root / g["phases"][phase]["requires_preregistration"]
        if not prereg.is_file():
            raise AuthorizationError(
                f"phase {phase!r} opens only after {prereg.relative_to(repo_root)} "
                "is committed")
    env = remaining_envelope(phase, repo_root)
    if hard_cap_usd > env["per_session_cap_usd"]:
        raise AuthorizationError(f"${hard_cap_usd} exceeds the per-session cap")
    if hard_cap_usd > env["phase_remaining_usd"] or hard_cap_usd > env["programme_remaining_usd"]:
        raise AuthorizationError(f"${hard_cap_usd} does not fit the envelope {env}")
    path = repo_root / authorization_path(run_id)
    if path.exists():
        raise AuthorizationError(f"{path.relative_to(repo_root)} exists; one artifact "
                                 "per run, never re-issued")
    files = HARNESS[kind]
    harness = harness_source_digest(repo_root, files=files)
    auth = CalVolAuthorization(
        authorization_id=f"stage0.calibration_volume.{kind}.{run_id}",
        granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        granted_by=g["granted_by"],
        plan_id=f"stage0.calibration_volume.{kind}",
        plan_hash=plan_hash(kind, repo_root),
        expected_usd=round(expected_usd, 4), hard_cap_usd=round(hard_cap_usd, 4),
        authorized_stages=(), stage_conditions={"phase": phase, "envelope": json.dumps(env)},
        scope_note=note or g["phases"][phase]["scope"],
        authorized_session_commit=None,
        harness_source_digest=harness["digest"], harness_source_files=files,
        per_launch_hard_usd=round(hard_cap_usd, 4),
        provenance_commit=git_head(repo_root), action_policy=POLICIES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.as_dict(), indent=1) + "\n")
    load_for(kind, path).require_plan(plan_hash(kind, repo_root))
    return path
