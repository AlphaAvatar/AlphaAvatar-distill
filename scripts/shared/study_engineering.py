"""A research study's engineering validation: grant, cumulative envelope, one-use authorization, live position, session and closeout.

One mechanism for every study's engineering validation (Research A's A-E,
Research B's B-E). The instance -- which grant, which files bind the study,
which scripts run, where its runs live, which driver and inputs the pod gets --
is a `Study` and a `Launch` its own module declares. Dollar amounts live in the
grant document, never here (P3).

* GRANT. Written only from the maintainer's decision, at ``Study.grant`` under
  ``logs/budget/approvals/``, ``status: "granted"``. Until then the study holds
  a PROPOSAL (``Study.proposal``, ``status: "proposed"``) beside its plans. The
  issuer refuses a proposal, except for a dry run: that artifact is written
  outside the run directory, carries ``stage_conditions.dry_run = "true"``, and
  every launch check that could lead to a provider resource refuses it. A
  grant funds its own study only -- never another grant's remainder, never the
  account balance.
* ENVELOPE. The grant's ceiling less every session of this study: its
  closeout's all-in cost; a session that created a provider resource and
  records no closeout cost is an unknown, and refuses.
* LIVE POSITION. `shared.project_position.position` on the provider read NOW,
  which fails closed: an unknown or errored provider part, a pod in the
  account, a volume no live record owns, an unposted session or a campaign
  subrun whose teardown is unconfirmed. The hard cap must fit what the
  project cap leaves and what the envelope leaves; the account balance must
  cover it, as a necessary condition and never as headroom.
* ONE USE. One artifact per run id, never re-issued; a launch refuses an
  artifact whose run has already created a resource or closed out.

`session_spec` builds the session on the shared runner
(`aadistill.infrastructure.session_runner`), the same chain H1-PA's sessions
ran: the provider reconciliation, the prechecks below, the watchdog and the
hard deadline derived from the accepted price, setup (the pinned train and
vLLM environments, the teacher at its revision), the driver, the evidence
relay on every poll, collection, teardown and the provider's confirmation.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from aadistill.governance.authorization import (ActionPolicy, AuthorizationError,
                                                SpendAuthorization, harness_source_digest)
from aadistill.infrastructure.manifest import sha256_file, sha256_json

REPO_ROOT = Path(__file__).resolve().parents[2]
POD_REPO = "/workspace/aad"
#: container disk rate (the provider's pricing record; `h1_generation_launch.DISK_USD_PER_GB_MONTH`)
DISK_USD_PER_GB_MONTH = 0.10
#: the watchdog's time to act on the hard deadline
TERMINATION_MINUTES = 1.25
EVIDENCE_ARCHIVE = "evidence.tgz"
REPORT = "report.json"
CLAIMS = {"authorizes_engineering_validation": "engineering_validation", "authorizes_formal": "formal",
          "authorizes_training": "training", "authorizes_evaluation": "evaluation",
          "authorizes_teacher_generation": "teacher_generation", "authorizes_paid_storage": "paid_storage"}


@dataclass(frozen=True)
class Study:
    """What binds one study's engineering validation."""
    key: str                                  #: "A-E", "B-E"
    plan_id: str
    logs: Path                                #: the experiment's log area (runs live under it)
    grant: Path                               #: logs/budget/approvals/<...>.json, the maintainer's
    proposal: Path                            #: the proposal beside the study's plans
    identity_files: tuple[str, ...]           #: the bound files: design, preregistration, check config...
    scripts: tuple[str, ...]                  #: every script the session runs, beyond src/aadistill
    wire_schema: str

    def policy(self) -> ActionPolicy:
        return ActionPolicy(
            policy_id=self.plan_id, allowed=frozenset({"engineering_validation"}), wire_claims=CLAIMS,
            wire_schema=self.wire_schema, plan_hash_key="study_identity_sha256",
            enforcement=("the launcher refuses a pod unless this artifact is bound to the study identity and the "
                         "harness on disk, is not a dry run, has not been used, and its hard cap covers the planned "
                         "all-in bound and fits the project cap as the provider bills it at launch; setup re-checks "
                         "the binding on the pod; one artifact funds one run inside the grant's envelope"))

    def run_dir(self, run_id: str) -> Path:
        if "/" in run_id or run_id in ("", ".", ".."):
            raise AuthorizationError(f"run id {run_id!r} must be one path segment")
        return self.logs / "runs" / run_id

    def authorization_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "governance/authorization.json"


class EngineeringAuthorization(SpendAuthorization):
    """A study's engineering-validation artifact type. Each study declares its own subclass naming
    ``STUDY``, so setup's branch for that study's session kind loads exactly that type
    (`tests/integration/test_session_kind_dispatch.py`)."""
    STUDY = None

    @classmethod
    def load(cls, path, *, policy=None, allow_dry_run: bool = False):
        if cls.STUDY is None:
            raise AuthorizationError(f"{cls.__name__} names no study; declare a subclass with STUDY")
        return load_authorization(cls.STUDY, path, allow_dry_run=allow_dry_run, cls=cls)


def require_on_pod(auth: SpendAuthorization, study: Study, session_plan_hash: str, root: str = ".") -> dict:
    """Setup's check on the pod, from the pod's own files: the plan and study identity, the harness,
    an engineering validation and nothing formal, never a dry run."""
    ident = identity_hash(study, root)
    auth.require_plan(session_plan_hash)
    auth.require_plan(ident)
    h = auth.require_harness(root)
    if not auth.allows("engineering_validation") or any(
            auth.allows(x) for x in ("formal", "training", "evaluation", "teacher_generation", "paid_storage")):
        raise AuthorizationError("not an engineering-validation artifact, or one that claims something formal")
    if (auth.stage_conditions or {}).get("dry_run") != "false":
        raise AuthorizationError("a dry-run artifact funds nothing")
    return {"identity": ident, "harness": h["digest"]}


def load_authorization(study: Study, path, *, allow_dry_run: bool = False, cls=SpendAuthorization):
    """The artifact under the study's policy: an engineering validation and nothing else; a dry-run
    artifact only where the caller says it is reading one."""
    path = Path(path)
    if not path.is_file():
        raise AuthorizationError(f"no authorization at {path}: nothing was issued for this run")
    #: the base loader's checks (its own sha256, the policy's claims), building an instance of `cls`
    auth = SpendAuthorization.load.__func__(cls, path, policy=study.policy())
    if auth.plan_id != study.plan_id:
        raise AuthorizationError(f"{path} is for {auth.plan_id}, not {study.plan_id}")
    if not auth.allows("engineering_validation") or any(auth.allows(a) for a in ("formal", "teacher_generation")):
        raise AuthorizationError(f"{path} is not an engineering-validation authorization of {study.key}")
    if (auth.stage_conditions or {}).get("dry_run") == "true" and not allow_dry_run:
        raise AuthorizationError(f"{path} is a DRY-RUN artifact; it funds nothing and creates nothing")
    return auth


def identity(study: Study, root: Path = REPO_ROOT) -> dict:
    root = Path(root)
    return {"study": study.key, "files": [{"path": f, "sha256": sha256_file(root / f)} for f in study.identity_files]}


def identity_hash(study: Study, root: Path = REPO_ROOT) -> str:
    return sha256_json(identity(study, root))


def harness_files(study: Study, root: Path = REPO_ROOT) -> tuple[str, ...]:
    root = Path(root)
    core = sorted(p.relative_to(root).as_posix() for p in (root / "src/aadistill").rglob("*.py")
                  if "__pycache__" not in p.parts)
    return tuple(core) + tuple(s for s in study.scripts if s not in core)


def git_head(root: Path = REPO_ROOT) -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return r.stdout.strip() or None


# ---- grant and envelope ------------------------------------------------------------------------------------

GRANT_FIELDS = ("study", "ceiling_usd", "per_session_hard_cap_usd", "project_cumulative_cap_usd", "gpu",
                "max_price_usd_per_hour", "granted_by", "decision")


def grant(study: Study, root: Path = REPO_ROOT, *, dry_run: bool = False) -> dict:
    """The maintainer's grant; for a dry run only, the study's proposal in its place."""
    root = Path(root)
    p = root / study.grant
    if p.is_file():
        g = json.loads(p.read_text())
        if g.get("status") != "granted":
            raise AuthorizationError(f"{study.grant} is {g.get('status')!r}, not granted")
        if g.get("closed"):
            raise AuthorizationError(f"{study.grant} is closed ({g['closed']})")
        g["_source"] = str(study.grant)
    elif dry_run and (root / study.proposal).is_file():
        g = json.loads((root / study.proposal).read_text())
        if g.get("status") != "proposed":
            raise AuthorizationError(f"{study.proposal} is {g.get('status')!r}, not a proposal")
        g["_source"] = str(study.proposal)
    else:
        raise AuthorizationError(f"no grant at {study.grant}: {study.key} is not authorized "
                                 f"(a proposal{' exists' if (root / study.proposal).is_file() else ' is missing'})")
    missing = [k for k in GRANT_FIELDS if g.get(k) in (None, "")]
    if missing or g["study"] != study.key:
        raise AuthorizationError(f"{g['_source']} is not a grant of {study.key} (missing {missing})")
    return g


def spent(study: Study, root: Path = REPO_ROOT) -> dict:
    """Every session of this study: its closeout's all-in cost, or an unknown."""
    root = Path(root)
    total, runs, unknown = 0.0, [], []
    for d in sorted((root / study.logs / "runs").glob("*")):
        a = d / "governance/authorization.json"
        if not a.is_file() or json.loads(a.read_text()).get("plan_id") != study.plan_id:
            continue
        out, sess = d / "closeout/outcome.json", d / "runtime/session.json"
        doc = json.loads(out.read_text()) if out.is_file() else {}
        usd = (doc.get("money") or {}).get("all_in_usd")
        created = (doc.get("provider_resource_created") if out.is_file() else
                   json.loads(sess.read_text()).get("provider_resource_created") if sess.is_file() else False)
        if usd is None and not created:
            usd = 0.0
        if usd is None:
            unknown.append(d.name)
            continue
        total += float(usd)
        runs.append({"run_id": d.name, "usd": float(usd)})
    return {"sessions_usd": round(total, 4), "runs": runs, "unknown": unknown}


def envelope(study: Study, g: dict, root: Path = REPO_ROOT) -> dict:
    s = spent(study, root)
    if s["unknown"]:
        raise AuthorizationError(f"runs {s['unknown']} created provider resources and record no closeout cost; "
                                 "reconcile them before anything else is funded")
    ceiling = float(g["ceiling_usd"])
    return {"ceiling_usd": ceiling, "spent_usd": s["sessions_usd"], "remaining_usd": round(ceiling - s["sessions_usd"], 4),
            "runs": s["runs"]}


def provider_snapshot() -> dict:
    """The provider's billing and account, read now (read-only, $0); an unreadable one refuses."""
    from stages.calibration_volume import mixed_governance as MG
    try:
        return MG.provider_snapshot()
    except Exception as exc:  # noqa: BLE001 -- any failed read is an unknown position
        raise AuthorizationError(f"the provider could not be read ({type(exc).__name__}: {exc}); "
                                 "an unknown position funds nothing") from exc


def live_position(root: Path = REPO_ROOT, *, snapshot: dict | None = None) -> dict:
    """`project_position.position` on the provider now (fail closed; module docstring)."""
    from shared import project_position as PP
    snap = snapshot if snapshot is not None else provider_snapshot()
    pos = PP.position(Path(root), snapshot=snap)
    keep = ("fetched_utc", "provider_billed_usd", "committed_usd", "project_cap_usd", "project_remaining_usd",
            "account_pods", "account_volumes", "balance_usd", "spend_per_hour", "rule")
    return {**{k: pos.get(k) for k in keep}, "unposted_sessions": len(pos["unposted_sessions"]),
            "campaign_unposted": len(pos["campaign_subruns"]["unposted"]),
            "campaign_teardown_confirmed_by": pos["campaign_subruns"].get("teardown_confirmed_by")}


def funding_problems(hard_cap_usd: float, g: dict, env: dict, pos: dict) -> list[str]:
    """Why a session at `hard_cap_usd` cannot be funded; empty when it can."""
    bad = []
    if hard_cap_usd > float(g["per_session_hard_cap_usd"]) + 1e-9:
        bad.append(f"${hard_cap_usd} exceeds the grant's per-session hard cap ${g['per_session_hard_cap_usd']}")
    if hard_cap_usd > env["remaining_usd"] + 1e-9:
        bad.append(f"${hard_cap_usd} exceeds the ${env['remaining_usd']} the grant's envelope leaves")
    if abs(float(pos["project_cap_usd"]) - float(g["project_cumulative_cap_usd"])) > 1e-9:
        bad.append(f"the project cap is ${pos['project_cap_usd']}, the grant was written under "
                   f"${g['project_cumulative_cap_usd']}: a new maintainer decision, not an inference")
    if hard_cap_usd > float(pos["project_remaining_usd"]) + 1e-9:
        bad.append(f"${hard_cap_usd} does not fit the ${pos['project_remaining_usd']} the ${pos['project_cap_usd']} "
                   "project cap leaves as the provider bills it now")
    if float(pos.get("balance_usd") or 0.0) < hard_cap_usd:
        bad.append(f"the provider account holds ${pos.get('balance_usd')}, less than the ${hard_cap_usd} hard cap")
    return bad


def issue(study: Study, run_id: str, *, expected_usd: float, hard_cap_usd: float, root: Path = REPO_ROOT,
          snapshot: dict | None = None, dry_run: bool = False, out: Path | None = None, note: str = "") -> Path:
    """Write the one-use authorization for `run_id` after every funding check (module docstring).
    A dry run reads the proposal, writes to `out` only (never the run directory) and marks the artifact."""
    root = Path(root)
    g = grant(study, root, dry_run=dry_run)
    ident = identity_hash(study, root)
    if g.get("study_identity_sha256") not in (None, ident):
        raise AuthorizationError(f"the grant names study identity {g['study_identity_sha256'][:12]}, not {ident[:12]}")
    env = envelope(study, g, root)
    pos = live_position(root, snapshot=snapshot)
    bad = funding_problems(hard_cap_usd, g, env, pos)
    if bad:
        raise AuthorizationError(f"{study.key} at ${hard_cap_usd} is not fundable: {bad}")
    if dry_run:
        if out is None:
            raise AuthorizationError("a dry-run artifact needs an explicit out path outside the run directory")
        path = Path(out)
        if (root / study.logs / "runs") in path.resolve().parents:
            raise AuthorizationError("a dry-run artifact never lands in the study's run directory")
    else:
        path = Path(out) if out is not None else root / study.authorization_path(run_id)
        if (root / study.run_dir(run_id)).exists():
            raise AuthorizationError(f"{study.run_dir(run_id)} exists; one artifact per run, never re-issued")
    if path.exists():
        raise AuthorizationError(f"{path} exists; one artifact per run, never re-issued")
    files = harness_files(study, root)
    harness = harness_source_digest(root, files=files)
    auth = SpendAuthorization(
        authorization_id=f"{study.plan_id}.{run_id}", granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        granted_by=g["granted_by"], plan_id=study.plan_id, plan_hash=ident,
        expected_usd=round(expected_usd, 4), hard_cap_usd=round(hard_cap_usd, 4), authorized_stages=(),
        stage_conditions={"kind": "engineering_validation", "study": study.key, "dry_run": "true" if dry_run else "false",
                          "grant": g["_source"], "grant_sha256": sha256_file(root / g["_source"]),
                          "gpu": g["gpu"], "max_price_usd_per_hour": str(g["max_price_usd_per_hour"]),
                          "envelope": json.dumps(env), "live_position": json.dumps(pos, default=str)},
        scope_note=note or g.get("scope", ""), authorized_session_commit=None,
        harness_source_digest=harness["digest"], harness_source_files=files,
        per_launch_hard_usd=round(hard_cap_usd, 4), provenance_commit=git_head(root), action_policy=study.policy())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.as_dict(), indent=1) + "\n")
    load_authorization(study, path, allow_dry_run=dry_run).require_plan(ident)
    return path


# ---- the session ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Launch:
    """What one study's engineering session runs, and on what."""
    session_kind: str                         #: setup's authorization branch (autoinit_preflight_setup.sh)
    authorization: type                       #: the study's EngineeringAuthorization subclass, which that branch loads
    teacher_model: str
    teacher_revision: str
    driver: str                               #: the pod-side driver, repo-relative
    validation_minutes: float                 #: the driver's bound (the forecast's conservative hours)
    setup_minutes: float = 25.0
    disk_gb: int = 120
    driver_args: Callable[[Any], str] = lambda args: ""
    relay_inputs: tuple = ()
    local_assets: tuple = ()
    contingency_fraction: float = 0.25
    verdict: Callable[[Path], dict] | None = None   #: re-derives the driver's report from its evidence
    artifact_spec: str = ""                   #: what the collector archives at teardown (configs/...)

    def audit(self) -> str:
        return f"{self.session_kind}_engineering"

    def status(self) -> str:
        return f"/workspace/{self.session_kind}.status"


def evidence_dir(launch: Launch, run_id: str) -> str:
    return f"{POD_REPO}/artifacts/audit/{launch.audit()}/{run_id}"


def store(ctx) -> Path:
    return Path(ctx.args.scr) / "store"


def _scp(ctx, remote: str, dest: Path, *, limit_s: float = 900) -> int:
    cmd = ["timeout", f"{int(limit_s)}s", "scp", "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", f"root@{ctx.host}:{remote}", str(dest)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=limit_s + 30).returncode
    except subprocess.TimeoutExpired:
        return 124


def secure_evidence(launch: Launch):
    """on_poll: once the driver's report exists on the pod, its evidence directory is tarred there,
    hashed, pulled into ``<scr>/store`` and re-hashed here. Idempotent; never raises."""
    def on_poll(ctx) -> None:
        try:
            if ctx.target is None:
                return
            store(ctx).mkdir(parents=True, exist_ok=True)
            if (ctx.evidence.get("evidence_secured") or {}).get("ok"):
                _scp(ctx, launch.status(), store(ctx) / "driver.status", limit_s=120)
                return
            ev = evidence_dir(launch, ctx.args.run_id)
            tgz = f"{ev}.{EVIDENCE_ARCHIVE}"
            res = ctx.target.run(f"test -f {ev}/{REPORT} && tar czf {tgz} -C {ev} . && sha256sum {tgz} | cut -d' ' -f1",
                                 timeout=600)
            digest = (getattr(res, "stdout", "") or "").strip().splitlines()
            _scp(ctx, launch.status(), store(ctx) / "driver.status", limit_s=120)
            if getattr(res, "returncode", 1) != 0 or not digest or len(digest[-1]) != 64:
                return
            dest = store(ctx) / EVIDENCE_ARCHIVE
            rc = _scp(ctx, tgz, dest)
            ok = dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest[-1]
            if ok:
                dest.with_suffix(".sha256").write_text(digest[-1] + "\n")
            ctx.evidence["evidence_secured"] = {"ok": ok, "sha256": digest[-1], "rc": rc}
            ctx.say(f"evidence {'SECURED' if ok else 'fetch incomplete'} ({digest[-1][:12]})")
        except Exception as exc:  # noqa: BLE001
            ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])
    return on_poll


def evidence_secured(launch: Launch):
    def products_secured(ctx, *_a, **_k) -> tuple[bool, str]:
        secure_evidence(launch)(ctx)
        sec = ctx.evidence.get("evidence_secured") or {}
        return (bool(sec.get("ok")), "the validation's evidence is secured off-pod" if sec.get("ok")
                else "the validation's evidence is not secured off-pod")
    return products_secured


def all_in(plan, price_per_hour: float, disk_gb: float, cap_usd: float | None = None) -> dict:
    """The most a session can bill: GPU and container disk from the create to the hard deadline plus
    the watchdog's time to act, ceiled to the closeout's quantum (`h1_generation_launch.all_in`)."""
    minutes = plan.hard_terminate_minutes + TERMINATION_MINUTES
    disk_per_hour = disk_gb * DISK_USD_PER_GB_MONTH / (24 * 30.4375)
    out = {"billable_minutes": round(minutes, 3), "price_per_hour": price_per_hour, "container_disk_gb": disk_gb,
           "gpu_usd": round(minutes / 60 * price_per_hour, 4), "disk_usd": round(minutes / 60 * disk_per_hour, 4),
           "all_in_usd": math.ceil(minutes / 60 * (price_per_hour + disk_per_hour) * 1e4) / 1e4}
    if cap_usd is not None:
        out["highest_price_inside_cap"] = math.floor((cap_usd * 60 / minutes - disk_per_hour) * 100) / 100
    return out


def prechecks(study: Study, launch: Launch, root: Path = REPO_ROOT) -> tuple:
    """The session's own gates, each read at $0 before the create."""
    def the_artifact_is_this_studys_and_unused(ctx) -> tuple[bool, str]:
        sc = ctx.auth.stage_conditions or {}
        if ctx.auth.plan_id != study.plan_id or sc.get("study") != study.key:
            return False, f"the artifact funds {sc.get('study')} / {ctx.auth.plan_id}, not {study.key}"
        if sc.get("dry_run") != "false" and not getattr(ctx.args, "dry_run", False):
            return False, "a dry-run artifact funds nothing and creates nothing"
        #: ONE USE: any earlier launch of this run -- one that created a pod, or one refused at $0 -- left a
        #: session record, and consumed the artifact (P12.1); a retry is a new run id with a fresh artifact
        rd = Path(root) / study.run_dir(ctx.args.run_id)
        if not getattr(ctx.args, "dry_run", False) and (
                (rd / "closeout/outcome.json").is_file() or (rd / "runtime/session.json").is_file()):
            return False, f"run {ctx.args.run_id} has already used its artifact; a retry is a new run and a new artifact"
        return True, f"a {study.key} engineering-validation artifact, unused"

    def the_gpu_and_price_are_the_grants(ctx) -> tuple[bool, str]:
        sc = ctx.auth.stage_conditions or {}
        if ctx.args.gpu != sc.get("gpu"):
            return False, f"--gpu {ctx.args.gpu!r} is not the granted {sc.get('gpu')!r}"
        if float(ctx.args.max_price) > float(sc.get("max_price_usd_per_hour") or 0) + 1e-9:
            return False, f"--max-price ${ctx.args.max_price}/h exceeds the granted ${sc.get('max_price_usd_per_hour')}/h"
        return True, f"{ctx.args.gpu} at no more than ${ctx.args.max_price}/h, as granted"

    def the_study_identity_is_the_authorized_one(ctx) -> tuple[bool, str]:
        got = identity_hash(study, root)
        return (got == ctx.auth.plan_hash, f"study identity {got[:12]} is the authorized one" if got == ctx.auth.plan_hash
                else f"the study identity on disk is {got[:12]}, not the authorized {ctx.auth.plan_hash[:12]}")

    def bound_files_are_in_the_session_commit(ctx) -> tuple[bool, str]:
        bad = []
        for f in identity(study, root)["files"]:
            blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{f['path']}"], cwd=root,
                                  capture_output=True)
            if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != f["sha256"]:
                bad.append(f["path"])
        return (not bad, "every bound file is in the session commit" if not bad
                else f"not in the session commit as bound: {bad[:4]}")

    def all_in_within_cap(ctx) -> tuple[bool, str]:
        if ctx.plan is None:
            return False, "no priced plan to bound"
        b = all_in(ctx.plan, ctx.args.max_price, ctx.args.disk_gb, ctx.auth.hard_cap_usd)
        ctx.evidence["all_in_bound"] = b
        if b["all_in_usd"] > ctx.auth.hard_cap_usd:
            return False, (f"all-in ${b['all_in_usd']:.4f} over {b['billable_minutes']} min at ${ctx.args.max_price}/h "
                           f"exceeds the authorized ${ctx.auth.hard_cap_usd:.2f}; it fits at "
                           f"${b['highest_price_inside_cap']}/h or less. The cap is not raised and the run not shrunk")
        return True, f"all-in ${b['all_in_usd']:.4f} over {b['billable_minutes']} min, inside ${ctx.auth.hard_cap_usd:.2f}"

    def the_project_cap_holds_now(ctx, snapshot: dict | None = None) -> tuple[bool, str]:
        try:
            g = grant(study, root, dry_run=getattr(ctx.args, "dry_run", False))
            env = envelope(study, g, root)
            pos = live_position(root, snapshot=snapshot)
        except AuthorizationError as exc:
            return False, str(exc)
        ctx.evidence["live_position"] = pos
        ctx.evidence["envelope"] = env
        bad = funding_problems(ctx.auth.hard_cap_usd, g, env, pos)
        return (not bad, f"${ctx.auth.hard_cap_usd:.2f} inside the envelope (${env['remaining_usd']}) and the "
                f"${pos['project_remaining_usd']} the project cap leaves now" if not bad else "; ".join(bad))

    return (the_artifact_is_this_studys_and_unused, the_gpu_and_price_are_the_grants, all_in_within_cap,
            the_study_identity_is_the_authorized_one, bound_files_are_in_the_session_commit, the_project_cap_holds_now)


def owned_volumes(root: Path = REPO_ROOT) -> tuple[str, ...]:
    """The network volumes live records own: the only resources the account may hold beside this
    session's pod (the project position refuses any other)."""
    from stages.calibration_volume import mixed_governance as MG
    return tuple(v["volume"] for v in MG.volume_records(Path(root)) if v["deleted"] is None)


def session_spec(study: Study, launch: Launch, args, root: Path = REPO_ROOT):
    """The session on the shared runner (module docstring)."""
    from aadistill.infrastructure.budget import Phase
    from aadistill.infrastructure.session import (ArtifactPolicy, BudgetSpec, ExecutionCommands, MarkerPolicy,
                                                  SessionSpec, SetupManifest, TeardownPolicy)
    from shared.deployment import deployment_commands

    status, ev = launch.status(), evidence_dir(launch, args.run_id)
    work = f"/workspace/{launch.session_kind}_work"

    def driver_command(ctx, plan) -> str:
        py = (f"cd {POD_REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={POD_REPO}/src:{POD_REPO}/scripts:"
              f"{POD_REPO}/scripts/stages/stage-1 /opt/train/bin/python")
        cmd = (f"{launch.driver} --run-id {ctx.args.run_id} --out {ev} --work {work} --status {status} "
               f"--device cuda:0 --vllm-python /opt/vllm/bin/python {launch.driver_args(ctx.args)}").strip()
        #: the driver writes its own terminal marker; one that dies before it gets RUN_FAILED
        return (f"mkdir -p {ev} {work} && {py} {cmd}; "
                f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
                f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")

    f = 1.0 + launch.contingency_fraction
    return SessionSpec(
        session_id=f"{launch.session_kind}-engineering",
        schema=f"{study.wire_schema.rsplit('/', 1)[0]}_session/v1",
        description=f"{study.key}: the engineering validation (never formal)",
        authorization_path=str(study.authorization_path(args.run_id)),
        #: the bound classmethod itself for a launch (its class is what setup's branch must load); a dry run
        #: alone may read a dry-run artifact
        authorization_loader=(partial(launch.authorization.load, allow_dry_run=True) if getattr(args, "dry_run", False)
                              else launch.authorization.load),
        commands=ExecutionCommands(watchdog="scripts/shared/pod/watchdog.py",
                                   setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
                                   artifact_collector="scripts/shared/pod/collect_artifacts.py", **deployment_commands()),
        plan_id=study.plan_id, plan_hash=identity_hash(study, root),
        budget=BudgetSpec(
            arms=0, steps_per_arm=0, step_seconds=1.0, step_source="unused: the validation is one phase (arms=0)",
            below_floor_reason="no arm-level steps; the validation is priced as one phase",
            setup_minutes=launch.setup_minutes, transfer_minutes=5.0,
            other_phases=(Phase(f"{study.key}_validation", launch.validation_minutes),),
            contingency_fraction=launch.contingency_fraction,
            soft_stop_reserves=(Phase("closeout_and_teardown", round(f * 10.0, 2)),),
            artifact_recovery_reserve_minutes=10.0,
            account_balance_required_usd=lambda plan: plan.hard_terminate_usd,
            accounted_provider_resources=lambda: owned_volumes(root)),
        setup=SetupManifest(
            env={"SESSION_KIND": launch.session_kind, "TEACHER_MODEL": launch.teacher_model},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS", "SESSION_KIND", "TEACHER_REVISION",
                          "TEACHER_MODEL"),
            #: no TESTS_OK: no repository suite on the paid pod (P8.2.1); no ROPE_OK: it checks C1-lineage
            #: checkpoints' RoPE base, which no input of this session is
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV", "VLLM_READY", "TEACHER_READY",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            local_assets=launch.local_assets, relay_inputs=launch.relay_inputs,
            teacher_revision=launch.teacher_revision, uv_max_seconds=1500, tests_max_seconds=600),
        driver_command=driver_command, driver_job_id=f"{launch.session_kind}_driver", status_path=status,
        run_log_path=f"/workspace/{launch.session_kind}_run.log",
        markers=MarkerPolicy(success="ALL_DONE", failure=("RUN_FAILED",), products_eligible=lambda t, s: True,
                             failure_note="the validation keeps no weights; its evidence is pulled on every poll"),
        artifacts=ArtifactPolicy(audit_dirname=launch.audit(), evidence_filename=REPORT,
                                 archive_basename=f"{launch.session_kind}_engineering",
                                 spec_success=launch.artifact_spec, spec_failed=launch.artifact_spec,
                                 report_names=(), on_poll=secure_evidence(launch),
                                 products_secured=evidence_secured(launch)),
        teardown=TeardownPolicy(note="delete the pod, verify from the provider that it is gone, STOP; nothing is kept "
                                     "on the provider: the validation's weights are scratch"),
        precheck=prechecks(study, launch, root),
    )


# ---- closeout --------------------------------------------------------------------------------------------------

def closeout(study: Study, launch: Launch, run_id: str, scr: Path, *, root: Path = REPO_ROOT, reconcile=None,
             disk_gb: float | None = None) -> dict:
    """The run's outcome on the dev box: the driver's terminal marker, the secured evidence archive
    re-hashed and its verdict RE-DERIVED from the files it holds (`Launch.verdict`), money (GPU from
    the session record plus container disk) and a fresh provider reconciliation. Writes
    ``runs/<run_id>/closeout/outcome.json``, whose ``money.all_in_usd`` the envelope reads."""
    root, st = Path(root), Path(scr) / "store"
    disk_gb = launch.disk_gb if disk_gb is None else disk_gb
    tgz = st / EVIDENCE_ARCHIVE
    evidence, rederived = None, None
    if tgz.is_file():
        got = hashlib.sha256(tgz.read_bytes()).hexdigest()
        want = tgz.with_suffix(".sha256")
        evidence = {"path": str(tgz), "sha256": got, "bytes": tgz.stat().st_size,
                    "verified": want.is_file() and want.read_text().strip() == got}
        if evidence["verified"] and launch.verdict is not None:
            with tempfile.TemporaryDirectory() as tmp:
                with tarfile.open(tgz) as t:
                    t.extractall(tmp, filter="data")
                pod = json.loads((Path(tmp) / REPORT).read_text()) if (Path(tmp) / REPORT).is_file() else {}
                again = launch.verdict(Path(tmp))
                rederived = {"passed": bool(again.get("passed")),
                             "checks": {k: v.get("status") for k, v in (again.get("checks") or {}).items()},
                             "agrees_with_the_pod": (again.get("passed") == pod.get("passed") and
                                                     {k: v.get("status") for k, v in (again.get("checks") or {}).items()}
                                                     == {k: v.get("status") for k, v in (pod.get("checks") or {}).items()})}
    status = (st / "driver.status").read_text() if (st / "driver.status").is_file() else ""
    terminal = "ALL_DONE" if "MARKER:ALL_DONE" in status else "RUN_FAILED" if "MARKER:RUN_FAILED" in status else None
    sess_p = root / study.run_dir(run_id) / "runtime/session.json"
    sess = json.loads(sess_p.read_text()) if sess_p.is_file() else {}
    if reconcile is None:
        from aadistill.infrastructure.provider import RunPodProvider, read_api_key, reconcile_account
        reconcile = lambda: reconcile_account(  # noqa: E731
            RunPodProvider(read_api_key(str(Path("~/.runpod/config.toml").expanduser()))), owned_volumes(root))
    rec = reconcile()
    cost = sess.get("cost") or {}
    gpu, minutes = float(cost.get("actual_usd") or 0.0), float(cost.get("elapsed_minutes") or 0.0)
    disk = disk_gb * DISK_USD_PER_GB_MONTH * minutes / (60 * 24 * 30.4375)
    created = bool(sess.get("provider_resource_created"))
    passed = bool(terminal == "ALL_DONE" and rederived and rederived["passed"] and rederived["agrees_with_the_pod"])
    out = {"schema": f"{study.wire_schema.rsplit('/', 1)[0]}_outcome/v1", "run_id": run_id, "study": study.key,
           "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "status": "COMPLETE" if evidence and evidence["verified"] and sess.get("provider_confirms_gone") else "INCOMPLETE",
           "validation_passed": passed, "terminal": terminal, "runner_terminal": sess.get("terminal"),
           "provider_resource_created": created, "evidence": evidence, "verdict_rederived": rederived,
           "money": {"all_in_usd": math.ceil((gpu + disk) * 1e4) / 1e4 if created else 0.0, "gpu_usd": gpu,
                     "disk_usd": round(disk, 4), "elapsed_minutes": minutes, "price_per_hour": cost.get("price_per_hour"),
                     "authorized_usd": cost.get("authorized_usd"),
                     "_basis": (f"GPU actual from runtime/session.json :: cost plus container disk at "
                                f"${DISK_USD_PER_GB_MONTH}/GB-month over the elapsed minutes ({disk_gb:g} GB)")},
           "provider_confirms_gone": sess.get("provider_confirms_gone"), "provider_reconciliation_after": rec,
           "scientific_use": False, "authorizes": "nothing"}
    dest = root / study.run_dir(run_id) / "closeout/outcome.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


# ---- the launcher's command line --------------------------------------------------------------------------------

def build_parser(description: str, *, gpu: str):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default=gpu)
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--token-src", default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--runpod-config", default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--country-code", default=None)
    ap.add_argument("--image-digest", default=None)
    ap.add_argument("--setup-timeout-s", type=float, default=3600.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--poll-limit-min", type=float, default=600.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="every $0 step (authorization, prechecks, the live provider read, the plan); no create")
    ap.add_argument("--authorization", default=None,
                    help="dry run only: the dry-run artifact `issue --dry-run` wrote outside the run directory")
    return ap


def launch_main(study: Study, launch: Launch, argv=None, *, root: Path = REPO_ROOT, description: str = "",
                gpu: str = "") -> int:
    """`h1pa_launch.main`'s shape: refuse at $0 with a reason, or run the session and close it out.

    ``--dry-run`` runs the runner's own pre-provider gates for real -- the plan at the given price,
    the live GPU quote, the account's funds and resources, every precheck (the live project
    position among them) and the relay listing -- on the dry-run artifact ``--authorization``
    names, records them in scratch (never the run directory) and stops before any create."""
    from dataclasses import replace

    from aadistill.infrastructure.session_runner import SessionRunner, run_session
    from stages.reasoning_data import h1_generation_launch as GL

    args = build_parser(description, gpu=gpu).parse_args(argv)
    if args.max_price is None:
        raise SystemExit("--max-price is required: read the live price first")
    if args.dry_run:
        if not args.authorization:
            raise SystemExit("--dry-run reads the dry-run artifact `issue --dry-run` wrote (--authorization)")
        args.out = args.out or str(Path(args.scr).resolve() / "dry_run_session.json")
    else:
        if args.authorization:
            raise SystemExit("--authorization names a dry-run artifact; a real launch reads the run's own")
        args.out = args.out or str(study.run_dir(args.run_id) / "runtime/session.json")
    try:
        session = session_spec(study, launch, args, root)
        if args.dry_run:
            session = replace(session, authorization_path=str(Path(args.authorization).resolve()))
        #: the artifact, its plan and its harness, read here so a missing or stale one refuses with its
        #: reason before anything else (the runner reads them again when it is built)
        auth = session.authorization_loader(Path(root) / session.authorization_path)
        auth.require_plan(session.plan_hash)
        auth.require_harness(root)
        runner = SessionRunner(session, args, root) if args.dry_run else None
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11
    ok, why = GL.bundle_on_relay(args)
    print(f"{'bundle OK' if ok else 'REFUSED at $0'}: {why}")
    if not ok and not args.dry_run:
        print("No provider resource was created.")
        return 13
    if args.dry_run:
        runner.ev["bundle_on_relay"] = {"ok": ok, "why": why}
        runner.run()
        ev = runner.ev
        print(json.dumps({"terminal": ev.get("terminal"), "provider_resource_created": ev.get("provider_resource_created"),
                          "bundle_on_relay": ev["bundle_on_relay"], "budget_plan": ev.get("budget_plan"),
                          "quoted_price_per_hour": ev.get("quoted_price_per_hour"), "stock": ev.get("stock_status"),
                          "account_balance": ev.get("account_balance"), "account_resources": ev.get("account_resources"),
                          "prechecks": ev.get("prechecks"), "precheck": ev.get("precheck"),
                          "harness": runner.harness["digest"], "study_identity": session.plan_hash,
                          "record": args.out}, indent=1, default=str)[:15000])
        return 0 if ev.get("terminal") == "DRY_RUN_GATES_PASSED" and ok else 12
    rc = run_session(session, args, root)
    print(json.dumps(closeout(study, launch, args.run_id, Path(args.scr), root=root)["money"]))
    return rc


# ---- the grant, written only from the maintainer's decision ---------------------------------------------------

def write_grant(study: Study, *, decision: str, granted_by: str, root: Path = REPO_ROOT) -> Path:
    """The maintainer's grant from the study's proposal, unchanged but for the decision: run only on the
    maintainer's explicit approval, which `decision` quotes. Binds the study identity at this commit."""
    root = Path(root)
    prop, dest = root / study.proposal, root / study.grant
    if dest.exists():
        raise AuthorizationError(f"{study.grant} exists; a grant is written once")
    doc = json.loads(prop.read_text())
    if doc.get("status") != "proposed" or doc.get("study") != study.key:
        raise AuthorizationError(f"{study.proposal} is not {study.key}'s proposal")
    if not decision.strip() or not granted_by.strip():
        raise AuthorizationError("a grant quotes the maintainer's decision and names who granted it")
    app = doc.get("application")
    if isinstance(app, str) and (root / app).is_file():
        doc["application"] = {"path": app, "sha256": sha256_file(root / app)}
    doc.update(status="granted", granted_by=granted_by, decision=decision,
               granted_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
               proposal={"path": str(study.proposal), "sha256": sha256_file(prop)},
               study_identity_sha256=identity_hash(study, root), granted_at_commit=git_head(root))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1) + "\n")
    return dest


def cli(study: Study, launch_spec: Callable[[], Launch], argv=None, *, gpu: str, description: str) -> int:
    """``issue | launch | closeout | position | grant`` for one study (its module's docstring)."""
    import argparse
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = ("issue", "launch", "closeout", "position", "grant")
    if not argv or argv[0] not in cmds:
        raise SystemExit(f"usage: {{{','.join(cmds)}}} ...  ({description})")
    cmd, rest = argv[0], argv[1:]
    if cmd == "launch":
        return launch_main(study, launch_spec(), rest, description=description, gpu=gpu)
    ap = argparse.ArgumentParser(prog=f"{study.key} {cmd}")
    if cmd == "issue":
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--expected-usd", type=float, required=True)
        ap.add_argument("--hard-cap-usd", type=float, required=True)
        ap.add_argument("--dry-run", action="store_true", help="read the proposal; write --out, never the run directory")
        ap.add_argument("--out", default=None)
        ap.add_argument("--note", default="")
        a = ap.parse_args(rest)
        try:
            path = issue(study, a.run_id, expected_usd=a.expected_usd, hard_cap_usd=a.hard_cap_usd,
                         dry_run=a.dry_run, out=Path(a.out) if a.out else None, note=a.note)
        except AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}")
            return 11
        print(path)
        return 0
    if cmd == "closeout":
        ap.add_argument("--run-id", required=True)
        ap.add_argument("--scr", required=True)
        a = ap.parse_args(rest)
        print(json.dumps(closeout(study, launch_spec(), a.run_id, Path(a.scr)), indent=1, default=str))
        return 0
    if cmd == "position":
        ap.add_argument("--hard-cap-usd", type=float, required=True)
        a = ap.parse_args(rest)
        out: dict = {"study": study.key}
        try:
            g = grant(study, dry_run=True)
            env = envelope(study, g)
            pos = live_position()
            out.update(grant=g["_source"], grant_status=g["status"], envelope=env, live_position=pos,
                       funding_problems=funding_problems(a.hard_cap_usd, g, env, pos))
        except AuthorizationError as exc:
            out["refused"] = str(exc)
        print(json.dumps(out, indent=1, default=str))
        return 0 if not out.get("refused") and not out.get("funding_problems") else 11
    ap.add_argument("--decision", required=True, help="the maintainer's approval, quoted")
    ap.add_argument("--granted-by", required=True)
    a = ap.parse_args(rest)
    print(write_grant(study, decision=a.decision, granted_by=a.granted_by))
    return 0
