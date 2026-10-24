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
  records no closeout cost is an unknown, and refuses;
* a hard cap inside the PROJECT cap as the provider bills it NOW
  (`live_position`): every issue re-reads the provider's billing and account,
  adds each closed session the provider has not posted yet and the study's
  unretired storage, and refuses when the cap does not cover it, or when the
  account holds anything but the study's volume. Nothing ever raises a cap;
* for a FORMAL session, a passed PA2v gate (`pa2v_gate`), re-derived from
  PA2v's secured evidence at issue time and bound to the same study identity
  and harness as the formal session.

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
#: H1's unit-session scripts this session reuses (its driver helpers, launcher pieces, persistence,
#: P1g's resume mechanism), plus H1-PA's own
SCRIPTS = tuple(s for s in UG.SCRIPTS if not s.endswith("p1g_driver.py")) + (
    "scripts/stages/stage-2/reasoning_data/h1_generation_launch.py",
    "scripts/stages/stage-2/reasoning_data/governance.py",
    "scripts/stages/stage-2/reasoning_data/h1pa_driver.py",
    "scripts/stages/stage-2/reasoning_data/h1pa_validation.py",
    "scripts/stages/stage-2/reasoning_data/p1g_driver.py",
    "scripts/maintenance/consolidation/provider_billing.py",
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


def provider_snapshot(start: str = "2026-06-01") -> dict:
    """The provider's billing and account, read now (read-only, $0)."""
    from aadistill.infrastructure.provider import read_api_key
    from maintenance.consolidation import provider_billing as PB
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return PB.fetch(read_api_key(str(Path("~/.runpod/config.toml").expanduser())), start, end)


def live_position(repo_root: Path = REPO, *, snapshot: dict | None = None) -> dict:
    """The project's position as the PROVIDER bills it now: everything billed,
    plus every closed H1-PA session whose pod the provider has not posted yet,
    plus the study's unretired storage at its reservation. Refuses when the
    account holds a pod or a volume this study does not own."""
    g = grant(repo_root)
    snap = snapshot if snapshot is not None else provider_snapshot()
    billed = round(sum(float(r["amount"]) for r in snap["pods"]) + sum(float(r["amount"]) for r in snap["networkvolumes"]), 4)
    posted = {r.get("podId") for r in snap["pods"]}
    s, root = spent(repo_root), Path(repo_root)
    unposted = []
    for r in s["runs"]:
        sess = root / run_dir(r["run_id"]) / "runtime/session.json"
        pod = json.loads(sess.read_text()).get("pod_id") if sess.is_file() else None
        if pod and pod not in posted:
            unposted.append({"run_id": r["run_id"], "pod_id": pod, "usd": r["usd"]})
    storage = round(sum(h["usd"] for h in s["storage"] if h["basis"] == "reserved"), 4)
    acct = snap["account"]
    owned = {h["volume"] for h in s["storage"]}
    foreign = [p.get("id") for p in acct.get("pods") or []] + \
              [v.get("id") for v in acct.get("network_volumes") or [] if v.get("id") not in owned]
    if foreign:
        raise AuthorizationError(f"the provider account holds {foreign}, which this study does not own: reconcile first")
    committed = round(billed + sum(u["usd"] for u in unposted) + storage, 4)
    cap = float(g["project_cumulative_cap_usd"])
    return {"fetched_utc": snap["fetched_utc"], "provider_billed_usd": billed, "unposted_sessions": unposted,
            "storage_reserved_usd": storage, "committed_usd": committed, "project_cap_usd": cap,
            "project_remaining_usd": round(cap - committed, 4),
            "spend_per_hour": acct.get("current_spend_per_hr"), "balance_usd": acct.get("client_balance_usd")}


PA2V_EVIDENCE = "pa2v.evidence.tgz"


def pa2v_gate(run_id: str, *, repo_root: Path = REPO, write: bool = True) -> dict:
    """PA2v's verdict, re-derived on the dev box from its secured evidence:
    the run's authorization (a validation artifact of this study and grant),
    its closeout (COMPLETE, the pod gone), the evidence archive's hash against
    the closeout, and every check of `h1pa_validation.checks` run again on the
    extracted files. Writes ``runs/<run_id>/closeout/pa2v_gate.json``."""
    import tarfile
    import tempfile
    from stages.reasoning_data import h1pa_driver as D
    from stages.reasoning_data import h1pa_validation as V
    root = Path(repo_root)
    auth = H1PAAuthorization.load(root / authorization_path(run_id))
    out = json.loads((root / run_dir(run_id) / "closeout/outcome.json").read_text())
    store = Path(out["store"]) if out.get("store") else None
    tgz = store / PA2V_EVIDENCE if store else None
    problems = []
    if auth.stage_conditions.get("kind") != "validation" or auth.plan_id != PLAN_ID:
        problems.append("the run's authorization is not an H1-PA validation artifact")
    if auth.stage_conditions.get("grant_sha256") != sha256_file(root / GRANT):
        problems.append("the run was authorized under another grant")
    if out.get("status") != "COMPLETE" or not out.get("provider_confirms_gone"):
        problems.append(f"the closeout is {out.get('status')}, provider gone {out.get('provider_confirms_gone')}")
    rec = (out.get("evidence") or {}).get(PA2V_EVIDENCE) or {}
    if tgz is None or not tgz.is_file() or sha256_file(tgz) != rec.get("sha256") or not rec.get("verified"):
        problems.append("the secured PA2v evidence is missing or not the archive the closeout hashed")
        result = {"checks": {}, "measured": {}, "passed": False}
    else:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(tgz) as t:
                t.extractall(tmp, filter="data")
            pod = json.loads((Path(tmp) / V.REPORT).read_text()) if (Path(tmp) / V.REPORT).is_file() else {}
            result = V.checks(Path(tmp) / "evidence", D.load_plan())
            if not pod.get("passed"):
                problems.append("the pod's own report did not pass")
    gate = {"schema": "aadistill.stage2.reasoning_data.pa2v_gate/v1", "run_id": run_id,
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "study_identity_sha256": auth.plan_hash, "harness_digest": auth.harness_source_digest,
            "evidence_sha256": rec.get("sha256"), "problems": problems, "checks": result["checks"],
            "measured": result["measured"], "passed": bool(result["passed"] and not problems), "authorizes": "nothing"}
    if write:
        p = root / run_dir(run_id) / "closeout/pa2v_gate.json"
        p.write_text(json.dumps(gate, indent=1, default=str) + "\n")
    return gate


def verify_gate(path: Path | None, *, harness: str, study: str) -> dict:
    """A formal session's precondition: the PA2v gate at `path`, RE-DERIVED now
    from its evidence (a recorded gate is never trusted as written), passed,
    and bound to the same study identity and harness as the formal session."""
    if path is None or not Path(path).is_file():
        raise AuthorizationError("no PA2v gate: a formal H1-PA session needs PA2v's passed GPU validation first")
    recorded = json.loads(Path(path).read_text())
    #: the gate lives at <root>/runs/<run>/closeout/pa2v_gate.json; its run's records are under that same root
    rel = run_dir(recorded["run_id"]) / "closeout/pa2v_gate.json"
    root = Path(path).resolve().parents[len(rel.parts) - 1]
    if root / rel != Path(path).resolve():
        raise AuthorizationError(f"{path} is not run {recorded['run_id']}'s gate record")
    gate = pa2v_gate(recorded["run_id"], repo_root=root, write=False)
    bad = [k for k in ("passed", "evidence_sha256", "study_identity_sha256", "harness_digest")
           if gate.get(k) != recorded.get(k)]
    if bad:
        raise AuthorizationError(f"the recorded PA2v gate disagrees with its re-derivation on {bad}")
    if not gate["passed"]:
        raise AuthorizationError(f"PA2v did not pass: {gate['problems'] or [k for k, v in gate['checks'].items() if not v.get('ok')]}")
    if gate["study_identity_sha256"] != study:
        raise AuthorizationError(f"PA2v validated study {str(gate['study_identity_sha256'])[:12]}, not {study[:12]}")
    if gate["harness_digest"] != harness:
        raise AuthorizationError(f"PA2v validated harness {str(gate['harness_digest'])[:12]}, not {harness[:12]}: "
                                 "the code changed since the validation")
    return {"path": str(Path(path)), "sha256": sha256_file(Path(path)), "run_id": gate["run_id"],
            "measured": gate["measured"]}


def issue(kind: str, run_id: str, *, expected_usd: float, hard_cap_usd: float,
          repo_root: Path = REPO, note: str = "", out: Path | None = None,
          pa2v: Path | None = None, snapshot: dict | None = None) -> Path:
    """Write the one-use authorization for `run_id`, after checking the
    envelope, the project cap against the provider's billing now, and -- for a
    formal session -- PA2v's gate (`pa2v`: the gate record), re-derived."""
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
    live = live_position(repo_root, snapshot=snapshot)
    if hard_cap_usd > live["project_remaining_usd"]:
        raise AuthorizationError(f"${hard_cap_usd} does not fit the project cap as the provider bills it now: {live}")
    files = harness_files(repo_root)
    harness = harness_source_digest(repo_root, files=files)
    gate = None
    if kind == "formal":
        gate = verify_gate(pa2v, harness=harness["digest"], study=ident)
    path = Path(out) if out is not None else Path(repo_root) / authorization_path(run_id)
    if path.exists():
        raise AuthorizationError(f"{path} exists; one artifact per run, never re-issued")
    auth = H1PAAuthorization(
        authorization_id=f"{PLAN_ID}.{kind}.{run_id}",
        granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        granted_by=g["granted_by"], plan_id=PLAN_ID, plan_hash=ident,
        expected_usd=round(expected_usd, 4), hard_cap_usd=round(hard_cap_usd, 4), authorized_stages=(),
        stage_conditions={"kind": kind, "phase": phase, "grant_sha256": sha256_file(Path(repo_root) / GRANT),
                          "envelope": json.dumps(env), "live_position": json.dumps(live),
                          "pa2v_gate": json.dumps(gate) if gate else None},
        scope_note=note or g["phases"][phase]["scope"], authorized_session_commit=None,
        harness_source_digest=harness["digest"], harness_source_files=files,
        per_launch_hard_usd=round(hard_cap_usd, 4), provenance_commit=git_head(repo_root),
        action_policy=POLICIES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.as_dict(), indent=1) + "\n")
    load_for(kind, path).require_plan(ident)
    return path
