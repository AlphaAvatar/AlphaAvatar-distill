"""What binds an H1-PA student session (PA2v GPU validation, PA2 formal units), and the one way to issue it.

The sibling of `unit_governance` (H1's, closed): the same artifact type, the
same kinds and the same envelope arithmetic, bound to H1-PA's grant, study and
preregistration instead of H1's. H1's grant is closed and funds nothing here.

The maintainer's phase grant (`GRANT`, written only from the maintainer's
decision) funds PA2v and PA2 out of ONE cumulative envelope: every attempt,
every resource and the study's network volume together. Each paid session has
its own one-use authorization, loaded under the policy of its KIND --
``validation`` (PA2v: never a formal result) or ``formal`` (PA2).

An artifact binds:

* the STUDY IDENTITY (`study_identity`): the sha256 of the study config, the
  preregistration, the decoding protocol, the initialization record and both
  E1 configs, plus the packs' and the PCA-Original checkpoints' recorded
  identities;
* the HARNESS (`harness_files`): the reusable core plus every script the
  session runs;
* a hard cap inside what the envelope has left (`remaining_envelope`): the
  ceiling less every earlier H1-PA session's closeout cost, every open
  session's hard cap, and the study's storage (actual once retired, its
  reservation until then). A session that created a provider resource and
  records no closeout cost is an unknown, and refuses.

Dollar amounts live in the grant document, never here (P3).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from aadistill.governance.authorization import (ActionPolicy, AuthorizationError,
                                                SpendAuthorization, harness_source_digest)
from aadistill.infrastructure.manifest import sha256_file, sha256_json
from stages.reasoning_data import unit_governance as UG

REPO = UG.REPO
EXPERIMENT_LOGS = UG.EXPERIMENT_LOGS
GRANT = Path("logs/budget/approvals/h1pa_training_grant.json")
STUDY = Path("configs/stages/stage-2/reasoning_data/h1pa.json")
PREREG = EXPERIMENT_LOGS / "plans/h1pa_preregistration.json"
PLAN_ID = "stage2.reasoning_data.h1pa"
#: the study's own storage records; H1's retired volume record is not one of them
STORAGE_GLOB = "h1pa_*.json"
KINDS = {"validation": "PA2v", "formal": "PA2"}
POLICIES = {k: ActionPolicy(
    policy_id=f"stage2.reasoning_data.h1pa.{k}", allowed=frozenset({k, "training", "evaluation"}),
    wire_claims={"authorizes_validation": "validation", "authorizes_formal": "formal",
                 "authorizes_training": "training", "authorizes_evaluation": "evaluation",
                 "authorizes_teacher_generation": "teacher_generation"},
    wire_schema="aadistill.stage2.reasoning_data.h1pa_authorization/v1",
    plan_hash_key="study_identity_sha256",
    enforcement=("the launcher refuses a pod unless this artifact is bound to the study identity and the harness "
                 "on disk and its hard cap covers the planned hard threshold; setup re-checks both on the pod; one "
                 "artifact funds one run, inside the phase grant's cumulative envelope"),
) for k in KINDS}
#: H1's unit-session scripts this session reuses (its driver helpers, launcher pieces, persistence), less
#: P1g's driver, plus H1-PA's own
SCRIPTS = tuple(s for s in UG.SCRIPTS if not s.endswith("p1g_driver.py")) + (
    "scripts/stages/stage-2/reasoning_data/h1_generation_launch.py",
    "scripts/stages/stage-2/reasoning_data/governance.py",
    "scripts/stages/stage-2/reasoning_data/h1pa_driver.py",
    "scripts/stages/stage-2/reasoning_data/h1pa_governance.py",
    "scripts/stages/stage-2/reasoning_data/h1pa_launch.py",
    "scripts/shared/evaluation/decoding.py",
)


def harness_files(repo_root: Path = REPO) -> tuple[str, ...]:
    root = Path(repo_root)
    core = sorted(p.relative_to(root).as_posix() for p in (root / "src/aadistill").rglob("*.py")
                  if "__pycache__" not in p.parts)
    tasks = sorted(p.relative_to(root).as_posix()
                   for p in (root / "scripts/stages/stage-0/calibration_volume/tasks").glob("*.py"))
    return tuple(core) + tuple(tasks) + SCRIPTS


class H1PAAuthorization(SpendAuthorization):
    """Loads an artifact under the policy of the kind it declares, and only that one."""

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
            raise AuthorizationError(f"{path} does not authorize a {kind} H1-PA session")
        return auth


def load_for(kind: str, path) -> H1PAAuthorization:
    auth = H1PAAuthorization.load(path)
    if auth.stage_conditions.get("kind") != kind:
        raise AuthorizationError(f"{path} is a {auth.stage_conditions.get('kind')} authorization, not {kind}")
    return auth


def study_identity(repo_root: Path = REPO) -> dict:
    root = Path(repo_root)
    st = json.loads((root / STUDY).read_text())
    files = [STUDY, PREREG, Path(st["evaluation"]["decoding"]), Path(st["initializations"]["d4_top1"]["record"])]
    files += [Path(c["path"]) for _, c in sorted(st["recipe"]["e1_configs"].items())]
    return {"files": [{"path": str(f), "sha256": sha256_file(root / f)} for f in files],
            "packs": {k: {f: st["datasets"][k][f] for f in ("blocks_sha256", "ladder_sha256")}
                      for k in ("e1_2p96M", "e1_2p96M_plus_fluid")},
            "pca_original": {s: c["model_safetensors_sha256"]
                             for s, c in sorted(st["arms"]["PCA-Original"]["checkpoints"].items())}}


def study_hash(repo_root: Path = REPO) -> str:
    return sha256_json(study_identity(repo_root))


run_dir, authorization_path, git_head = UG.run_dir, UG.authorization_path, UG.git_head


def grant(repo_root: Path = REPO) -> dict:
    p = Path(repo_root) / GRANT
    if not p.is_file():
        raise AuthorizationError(f"no grant at {p}: H1-PA's student training and evaluation are not authorized")
    return json.loads(p.read_text())


def spent(repo_root: Path = REPO) -> dict:
    """Every H1-PA session's closeout cost, every open session, and the storage held."""
    root = Path(repo_root)
    total, runs, unknown, open_ = 0.0, [], [], []
    for d in sorted((root / EXPERIMENT_LOGS / "runs").glob("*")):
        auth = d / "governance/authorization.json"
        if not auth.is_file() or json.loads(auth.read_text()).get("plan_id") != PLAN_ID:
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
    store = root / EXPERIMENT_LOGS / "storage"
    for v in sorted(store.glob(STORAGE_GLOB)) if store.is_dir() else []:
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


def issue(kind: str, run_id: str, *, expected_usd: float, hard_cap_usd: float,
          repo_root: Path = REPO, note: str = "", out: Path | None = None) -> Path:
    """Write the one-use authorization for `run_id`, after checking the envelope."""
    if kind not in KINDS:
        raise AuthorizationError(f"unknown kind {kind!r}")
    g = grant(repo_root)
    if g.get("closed"):
        raise AuthorizationError(f"the H1-PA training grant is closed ({g['closed']['decision']})")
    phase = KINDS[kind]
    if phase not in g["phases"]:
        raise AuthorizationError(f"the grant funds no phase {phase!r}")
    st = json.loads((Path(repo_root) / STUDY).read_text())
    if kind == "formal" and st.get("frozen") is not True:
        raise AuthorizationError("a formal H1-PA session needs the frozen study")
    ident = study_hash(repo_root)
    if g.get("study_identity_sha256") not in (None, ident):
        raise AuthorizationError(f"the grant names study identity {g['study_identity_sha256'][:12]}, not {ident[:12]}")
    env = remaining_envelope(repo_root)
    if hard_cap_usd > env["remaining_usd"] or hard_cap_usd > env["project_remaining_usd"]:
        raise AuthorizationError(f"${hard_cap_usd} does not fit the envelope {env}")
    path = Path(out) if out is not None else Path(repo_root) / authorization_path(run_id)
    if path.exists():
        raise AuthorizationError(f"{path} exists; one artifact per run, never re-issued")
    files = harness_files(repo_root)
    harness = harness_source_digest(repo_root, files=files)
    auth = H1PAAuthorization(
        authorization_id=f"{PLAN_ID}.{kind}.{run_id}",
        granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        granted_by=g["granted_by"], plan_id=PLAN_ID, plan_hash=ident,
        expected_usd=round(expected_usd, 4), hard_cap_usd=round(hard_cap_usd, 4), authorized_stages=(),
        stage_conditions={"kind": kind, "phase": phase, "grant_sha256": sha256_file(Path(repo_root) / GRANT),
                          "envelope": json.dumps(env)},
        scope_note=note or g["phases"][phase]["scope"], authorized_session_commit=None,
        harness_source_digest=harness["digest"], harness_source_files=files,
        per_launch_hard_usd=round(hard_cap_usd, 4), provenance_commit=git_head(repo_root),
        action_policy=POLICIES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.as_dict(), indent=1) + "\n")
    load_for(kind, path).require_plan(ident)
    return path
