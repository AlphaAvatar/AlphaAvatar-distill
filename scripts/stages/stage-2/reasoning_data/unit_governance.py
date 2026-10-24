"""What binds an H1 unit session (P1g GPU validation, P2 formal units), and the one way to issue it.

The maintainer's phase grant (`UNITS_GRANT`) funds P1g and P2 out of ONE
cumulative envelope: every attempt, every resource and the network volume
together (2026-10-23). Every paid session is a run of this experiment with its
own one-use authorization, loaded under the policy of its KIND -- ``validation``
(P1g: never a formal result) or ``formal`` (P2) -- so a validation artifact can
never fund a formal unit or the reverse.

An artifact binds:

* the UNITS IDENTITY (`units_identity`): the sha256 of the units config, the
  comparison design, the accounting record that owns the schedule, the
  initialization record and the frozen recipe, and the pack identities the
  config records; recomputed on the pod by setup's own branch;
* the HARNESS (`harness_files`): every module of the reusable core plus the
  experiment scripts the session runs (a coarse closure, chosen so that no
  imported module can be missed);
* a hard cap that must fit what the envelope has left (`remaining_envelope`):
  the ceiling less the closeout cost of every earlier unit run, less every
  open run's hard cap, less the storage reservation. A run that created a
  provider resource and records no closeout cost is an unknown, and refuses.

Dollar amounts live in the grant document, never here (P3).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aadistill.governance.authorization import (ActionPolicy, AuthorizationError,
                                                SpendAuthorization, harness_source_digest)
from aadistill.infrastructure.manifest import sha256_file, sha256_json

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT_LOGS = Path("logs/stages/stage-2/reasoning_data")
UNITS_GRANT = Path("logs/budget/approvals/h1_units_phase_grant.json")
UNITS_CONFIG = Path("configs/stages/stage-2/reasoning_data/h1_units.json")
#: the storage this phase pays for, booked at its reservation until retired
STORAGE = EXPERIMENT_LOGS / "storage"
#: kind -> the phase that funds it
KINDS = {"validation": "P1g", "formal": "P2"}
POLICIES = {k: ActionPolicy(
    policy_id=f"stage2.reasoning_data.h1_units.{k}", allowed=frozenset({k, "training", "evaluation"}),
    wire_claims={"authorizes_validation": "validation", "authorizes_formal": "formal",
                 "authorizes_training": "training", "authorizes_evaluation": "evaluation",
                 "authorizes_teacher_generation": "teacher_generation"},
    wire_schema="aadistill.stage2.reasoning_data.unit_authorization/v1",
    plan_hash_key="units_identity_sha256",
    enforcement=("the launcher refuses a pod unless this artifact is bound to the units identity and the "
                 "harness on disk and its hard cap covers the planned hard threshold; setup re-checks both "
                 "on the pod; one artifact funds one run, inside the phase grant's cumulative envelope"),
) for k in KINDS}
#: what a unit session executes beyond the reusable core
SCRIPTS = (
    "scripts/shared/pod/autoinit_preflight_setup.sh",
    "scripts/shared/pod/watchdog.py",
    "scripts/shared/pod/collect_artifacts.py",
    "scripts/shared/evaluation/uncapped_eval.py",
    "scripts/shared/evaluation/eval_set_scoring.py",
    "scripts/shared/eval_sets.py",
    "scripts/shared/training/train_stage3.py",
    "scripts/stages/stage-0/calibration_volume/dose_driver.py",
    "scripts/stages/stage-0/calibration_volume/run_battery.py",
    "scripts/stages/stage-0/calibration_volume/score_battery.py",
    "scripts/stages/stage-0/calibration_volume/volume_sync.py",
    "scripts/stages/stage-1/phase_c1/autoinit_c1_driver.py",
    "scripts/stages/stage-1/phase_c1/packaging.py",
    "scripts/stages/stage-2/reasoning_data/h1_unit_launch.py",
    "scripts/stages/stage-2/reasoning_data/p1g_driver.py",
    "scripts/stages/stage-2/reasoning_data/persist_unit.py",
    "scripts/stages/stage-2/reasoning_data/restore_initialization.py",
    "scripts/stages/stage-2/reasoning_data/unit_driver.py",
    "scripts/stages/stage-2/reasoning_data/unit_governance.py",
)


def harness_files(repo_root: Path = REPO) -> tuple[str, ...]:
    root = Path(repo_root)
    core = sorted(p.relative_to(root).as_posix() for p in (root / "src/aadistill").rglob("*.py")
                  if "__pycache__" not in p.parts)
    tasks = sorted(p.relative_to(root).as_posix()
                   for p in (root / "scripts/stages/stage-0/calibration_volume/tasks").glob("*.py"))
    return tuple(core) + tuple(tasks) + SCRIPTS


class H1UnitAuthorization(SpendAuthorization):
    """The unit-session artifact type. `load` reads the kind the artifact
    declares and loads it under that kind's policy only."""

    @classmethod
    def load(cls, path, *, policy=None):
        path = Path(path)
        if not path.is_file():
            raise AuthorizationError(f"no authorization at {path}: nothing was issued for this run")
        kind = (json.loads(path.read_text()).get("stage_conditions") or {}).get("kind")
        if kind not in KINDS:
            raise AuthorizationError(f"{path} names kind {kind!r}, not one of {sorted(KINDS)}")
        auth = super().load(path, policy=policy or POLICIES[kind])
        if not (auth.allows(kind) and auth.allows("training") and auth.allows("evaluation")):
            raise AuthorizationError(f"{path} does not authorize a {kind} unit session")
        return auth


def load_for(kind: str, path) -> H1UnitAuthorization:
    auth = H1UnitAuthorization.load(path)
    if auth.stage_conditions.get("kind") != kind:
        raise AuthorizationError(f"{path} is a {auth.stage_conditions.get('kind')} authorization, not {kind}")
    return auth


def units_identity(repo_root: Path = REPO) -> dict:
    root = Path(repo_root)
    cfg = json.loads((root / UNITS_CONFIG).read_text())
    files = [UNITS_CONFIG, EXPERIMENT_LOGS / "plans/comparison_design.json",
             EXPERIMENT_LOGS / "plans/h1_accounting.json", Path(cfg["initialization"]["record"]),
             Path(cfg["recipe"]["frozen"])]
    return {"files": [{"path": str(f), "sha256": sha256_file(root / f)} for f in files],
            "packs": {arm: {k: p.get(k) for k in ("blocks_sha256", "ladder_sha256", "expected")}
                      for arm, p in sorted(cfg["packs"].items()) if arm in ("A", "B")}}


def units_hash(repo_root: Path = REPO) -> str:
    return sha256_json(units_identity(repo_root))


def run_dir(run_id: str) -> Path:
    return EXPERIMENT_LOGS / "runs" / run_id


def authorization_path(run_id: str) -> Path:
    return run_dir(run_id) / "governance" / "authorization.json"


def grant(repo_root: Path = REPO) -> dict:
    return json.loads((Path(repo_root) / UNITS_GRANT).read_text())


def spent(repo_root: Path = REPO) -> dict:
    """Every unit run's closeout cost, every open run, and the storage held.

    A unit run is a run directory whose authorization is a unit authorization.
    One that created a provider resource without a closeout cost is UNKNOWN."""
    root = Path(repo_root)
    total, runs, unknown, open_ = 0.0, [], [], []
    for d in sorted((root / EXPERIMENT_LOGS / "runs").glob("*")):
        auth = d / "governance/authorization.json"
        if not auth.is_file() or json.loads(auth.read_text()).get("plan_id") != "stage2.reasoning_data.h1_units":
            continue
        out, sess = d / "closeout/outcome.json", d / "runtime/session.json"
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
            open_.append(d.name)
    held = []
    for v in sorted((root / STORAGE).glob("*.json")) if (root / STORAGE).is_dir() else []:
        doc = json.loads(v.read_text())
        held.append({"volume": doc["volume_id"], "usd": float(doc.get("actual_usd") or doc["reserved_usd"]),
                     "basis": "actual" if doc.get("actual_usd") is not None else "reserved"})
    return {"sessions_usd": round(total, 4), "runs": runs, "unknown": unknown, "open": open_,
            "storage_usd": round(sum(h["usd"] for h in held), 4), "storage": held}


def remaining_envelope(repo_root: Path = REPO) -> dict:
    g, s = grant(repo_root), spent(repo_root)
    if s["unknown"]:
        raise AuthorizationError(f"runs {s['unknown']} created provider resources and record no closeout "
                                 "cost; reconcile them before anything else is funded")
    used = s["sessions_usd"] + s["storage_usd"]
    ceiling = float(g["ceiling_usd"])
    return {"ceiling_usd": ceiling, "sessions_usd": s["sessions_usd"], "storage_usd": s["storage_usd"],
            "remaining_usd": round(ceiling - used, 4),
            "project_cap_usd": float(g["project_cumulative_cap_usd"]),
            "project_committed_usd": round(float(g["occupancy_basis_usd"]) + used, 4),
            "project_remaining_usd": round(float(g["project_cumulative_cap_usd"])
                                           - float(g["occupancy_basis_usd"]) - used, 4),
            "runs": s["runs"], "storage": s["storage"]}


def git_head(repo_root: Path = REPO) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                          capture_output=True, text=True, check=True).stdout.strip()


def issue(kind: str, run_id: str, *, expected_usd: float, hard_cap_usd: float,
          repo_root: Path = REPO, note: str = "", out: Path | None = None) -> Path:
    """Write the one-use authorization for `run_id`, after checking the envelope."""
    if kind not in KINDS:
        raise AuthorizationError(f"unknown kind {kind!r}")
    g = grant(repo_root)
    if g.get("closed"):
        raise AuthorizationError(f"the H1 units grant is closed ({g['closed']['decision']}); its remainder "
                                 "funds nothing, a later study included")
    phase = KINDS[kind]
    if phase not in g["phases"]:
        raise AuthorizationError(f"the grant funds no phase {phase!r}")
    if kind == "formal":
        cfg = json.loads((Path(repo_root) / UNITS_CONFIG).read_text())
        if cfg.get("frozen") is not True:
            raise AuthorizationError("a formal unit session needs the frozen units config")
    env = remaining_envelope(repo_root)
    if hard_cap_usd > env["remaining_usd"] or hard_cap_usd > env["project_remaining_usd"]:
        raise AuthorizationError(f"${hard_cap_usd} does not fit the envelope {env}")
    path = Path(out) if out is not None else Path(repo_root) / authorization_path(run_id)
    if path.exists():
        raise AuthorizationError(f"{path} exists; one artifact per run, never re-issued")
    files = harness_files(repo_root)
    harness = harness_source_digest(repo_root, files=files)
    ident = units_hash(repo_root)
    auth = H1UnitAuthorization(
        authorization_id=f"stage2.reasoning_data.h1_units.{kind}.{run_id}",
        granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        granted_by=g["granted_by"], plan_id="stage2.reasoning_data.h1_units", plan_hash=ident,
        expected_usd=round(expected_usd, 4), hard_cap_usd=round(hard_cap_usd, 4), authorized_stages=(),
        stage_conditions={"kind": kind, "phase": phase, "grant_sha256": sha256_file(Path(repo_root) / UNITS_GRANT),
                          "envelope": json.dumps(env)},
        scope_note=note or g["phases"][phase]["scope"], authorized_session_commit=None,
        harness_source_digest=harness["digest"], harness_source_files=files,
        per_launch_hard_usd=round(hard_cap_usd, 4), provenance_commit=git_head(repo_root),
        action_policy=POLICIES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.as_dict(), indent=1) + "\n")
    load_for(kind, path).require_plan(ident)
    return path
