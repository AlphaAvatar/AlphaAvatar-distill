"""What binds a mixed-calibration session (MC1 search, MC2 recovery), and the one way to issue it.

The maintainer's grant (`GRANT`, written only from the maintainer's decision of
2026-10-26) funds MC1 and MC2 out of ONE cumulative envelope: every session,
retry and recovery, and the MC1 network volume. Each paid session has its own
one-use authorization, loaded under the policy of its KIND -- ``search`` (MC1)
or ``recovery`` (MC2) -- and bound to:

* the session's IDENTITY (`identity_hash`): for MC1 the search plan
  (`configs/.../mixed_search_session.json`, which pins the profile, the
  design, the preregistration and the search identity); for MC2 the recovery
  study, the preregistration, the decoding protocol, H1-PA's study record it
  takes D4-Augmented's recipe and data from, E1's configs, the MC1 Top-1's
  read-back record, and the merged set's pack identity;
* the HARNESS (`harness_files`): the reusable core plus every script the
  session runs;
* a hard cap inside what the envelope has left (`remaining_envelope`) and
  inside the PROJECT cap as the provider bills it now (`position`).

`position` is the project's committed spend, recomputed from the provider at
every issue and before every launch (the maintainer's corrections of
2026-10-26):

1. everything the provider has billed (pods and network volumes);
2. every session whose pod the provider has not posted yet, within the
   posting window (`POSTING_WINDOW_HOURS`); a pod absent from the bill for
   longer is listed, not counted -- the provider's bill is the charge;
3. every retained volume's booked obligation LESS what the provider has
   already posted for it, and posted rows are attributed CONSERVATIVELY
   (`netted_posted`): the provider's volume rows carry no volume id, so a
   day's row is netted against a volume only when that volume was the sole
   recorded volume on the account that whole UTC day, and never beyond the
   volume's own size x the hours it existed that day. Rows on shared days and
   any excess (2026-10-25: 870 GB-h posted where one 30 GB volume can accrue
   720) stay in the bill and reduce nothing;
4. a volume deleted inside the posting window: its whole actual cost again,
   beside whatever of it the bill already shows.

It refuses -- nothing is issued or created -- when any provider read failed
or is unknown, when the account holds a pod (at most one billing pod: the one
about to be created), when it holds a volume no record owns, or when a
session created a resource and records no cost.

Dollar amounts live in the grant, never here (P3).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aadistill.governance.authorization import (ActionPolicy, AuthorizationError,
                                                SpendAuthorization, harness_source_digest)
from aadistill.infrastructure.manifest import sha256_file, sha256_json
from stages.calibration_volume import governance as G

REPO = G.REPO
EXPERIMENT_LOGS = G.EXPERIMENT_LOGS
GRANT = Path("logs/budget/approvals/mixed_calibration_grant.json")
SEARCH_PLAN = "mixed_search"
RECOVERY_STUDY = Path("configs/stages/stage-0/calibration_volume/mixed_recovery.json")
PREREG = EXPERIMENT_LOGS / "plans/mixed_preregistration.json"
#: every run of this study lives in the experiment's runs/ under this prefix
#: (mixs_ = MC1 session, mixr_ = MC2 session, mixvol_ = the MC1 volume)
RUN_PREFIX = "mix"
KINDS = {"search": "MC1", "recovery": "MC2"}
PLAN_IDS = {"search": "stage0.calibration_volume.mixed.search", "recovery": "stage0.calibration_volume.mixed.recovery"}
#: a closed session's pod normally appears in the provider's bill within hours
#: (H1-PA's kqfz2osp8ps8jv, closed 2026-10-25 ~16:00Z, was posted by 10-26 09:28Z);
#: a pod still absent after this window was never charged (a3_attempt20's three
#: cold-host pods, absent since September) and is listed, not counted
POSTING_WINDOW_HOURS = 72.0
POLICIES = {k: ActionPolicy(
    policy_id=f"stage0.calibration_volume.mixed.{k}", allowed=frozenset({k}),
    wire_claims={"authorizes_search": "search", "authorizes_recovery": "recovery",
                 "authorizes_teacher_generation": "teacher_generation", "authorizes_4x": "4x"},
    wire_schema="aadistill.stage0.calibration_volume.mixed_authorization/v1",
    plan_hash_key="session_identity_sha256",
    enforcement=("the launcher refuses a pod unless this artifact is bound to the session identity and the harness "
                 "on disk and its hard cap covers the planned hard threshold; the driver re-checks both on the pod; "
                 "one artifact funds one run, inside the grant's cumulative envelope and the project cap as billed"),
) for k in KINDS}


class MixedAuthorization(SpendAuthorization):
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
        if not auth.allows(kind):
            raise AuthorizationError(f"{path} does not authorize a mixed-calibration {kind} session")
        return auth


def load_for(kind: str, path) -> MixedAuthorization:
    auth = MixedAuthorization.load(path)
    if auth.stage_conditions.get("kind") != kind:
        raise AuthorizationError(f"{path} is a {auth.stage_conditions.get('kind')} authorization, not {kind}")
    return auth


def run_dir(run_id: str) -> Path:
    if not run_id.startswith(RUN_PREFIX):
        raise AuthorizationError(f"{run_id!r}: a mixed-calibration run id starts with {RUN_PREFIX!r}")
    return G.run_dir(run_id)


def authorization_path(run_id: str) -> Path:
    return run_dir(run_id) / "governance" / "authorization.json"


def grant(repo_root: Path = REPO) -> dict:
    p = Path(repo_root) / GRANT
    if not p.is_file():
        raise AuthorizationError(f"no grant at {p}: the mixed-calibration study is not authorized")
    return json.loads(p.read_text())


# ---- identities --------------------------------------------------------------------------------------

def recovery_identity(repo_root: Path = REPO) -> dict:
    root = Path(repo_root)
    st = json.loads((root / RECOVERY_STUDY).read_text())
    base = json.loads((root / st["base_study"]["path"]).read_text())
    files = [RECOVERY_STUDY, PREREG, Path(base["evaluation"]["decoding"]), Path(st["base_study"]["path"]),
             Path(st["initialization"]["record"])]
    files += [Path(c["path"]) for _, c in sorted(base["recipe"]["e1_configs"].items())]
    key = base["arms"][st["control_arm"]]["dataset"]
    return {"files": [{"path": str(f), "sha256": sha256_file(root / f)} for f in files],
            "pack": {key: {f: base["datasets"][key][f] for f in ("blocks_sha256", "ladder_sha256")}}}


def identity_hash(kind: str, repo_root: Path = REPO) -> str:
    if kind == "search":
        return G.plan_hash(SEARCH_PLAN, repo_root)
    if kind == "recovery":
        return sha256_json(recovery_identity(repo_root))
    raise AuthorizationError(f"unknown kind {kind!r}")


SCRIPTS = {
    "search": (
        "scripts/shared/pod/autoinit_preflight_setup.sh",
        "scripts/shared/pod/watchdog.py",
        "scripts/shared/pod/collect_artifacts.py",
        "scripts/shared/forward_baseline.py",
        "scripts/stages/stage-0/calibration_volume/governance.py",
        "scripts/stages/stage-0/calibration_volume/launch.py",
        "scripts/stages/stage-0/calibration_volume/search_driver.py",
        "scripts/stages/stage-0/calibration_volume/search_session.py",
        "scripts/stages/stage-0/calibration_volume/volume_sync.py",
        "scripts/stages/stage-0/calibration_volume/finalists.py",
        "scripts/stages/stage-0/calibration_volume/mixed_search_driver.py",
    ),
    "recovery": (
        "scripts/stages/stage-0/calibration_volume/dose_driver.py",
        "scripts/stages/stage-0/calibration_volume/mixed_recovery_driver.py",
        "scripts/stages/stage-0/calibration_volume/run_battery.py",
        "scripts/stages/stage-0/calibration_volume/score_battery.py",
        "scripts/stages/stage-0/calibration_volume/volume_sync.py",
        "scripts/shared/evaluation/eval_set_scoring.py",
        "scripts/shared/evaluation/uncapped_eval.py",
        "scripts/shared/eval_sets.py",
        "scripts/shared/training/train_stage3.py",
        "scripts/stages/stage-1/phase_c1/autoinit_c1_driver.py",
        "scripts/stages/stage-1/phase_c1/packaging.py",
    ),
}
COMMON_SCRIPTS = (
    "scripts/maintenance/consolidation/provider_billing.py",
    "scripts/stages/stage-0/calibration_volume/governance.py",
    "scripts/stages/stage-0/calibration_volume/launch.py",
    "scripts/stages/stage-0/calibration_volume/mixed_governance.py",
    "scripts/stages/stage-0/calibration_volume/mixed_launch.py",
)


def harness_files(kind: str, repo_root: Path = REPO) -> tuple[str, ...]:
    """The reusable core, every task module, and the scripts the session runs."""
    root = Path(repo_root)
    core = sorted(p.relative_to(root).as_posix() for p in (root / "src/aadistill").rglob("*.py")
                  if "__pycache__" not in p.parts)
    tasks = sorted(p.relative_to(root).as_posix()
                   for p in (root / "scripts/stages/stage-0/calibration_volume/tasks").glob("*.py"))
    extra: tuple[str, ...] = ()
    if kind == "recovery":
        from stages.reasoning_data import h1pa_governance as HG
        extra = tuple(s for s in HG.SCRIPTS if s not in SCRIPTS[kind])
    return tuple(dict.fromkeys(tuple(core) + tuple(tasks) + SCRIPTS[kind] + extra + COMMON_SCRIPTS))


# ---- the envelope --------------------------------------------------------------------------------------

def _utc(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)


def spent(repo_root: Path = REPO) -> dict:
    """Every run of this study: sessions by their closeout, the MC1 volume at its
    reservation until a closeout states its actual cost. A run that created a
    resource and records no cost is an unknown."""
    root = Path(repo_root)
    sessions, storage, unknown, open_ = [], [], [], []
    for d in sorted((root / EXPERIMENT_LOGS / "runs").glob(f"{RUN_PREFIX}*")):
        sess_p, out_p = d / "runtime/session.json", d / "closeout/outcome.json"
        sess = json.loads(sess_p.read_text()) if sess_p.is_file() else {}
        out = json.loads(out_p.read_text()) if out_p.is_file() else None
        if sess.get("kind") == "network_volume":
            deleted = (out or {}).get("status") == "VOLUME_DELETED"
            usd = (out or {}).get("money", {}).get("all_in_usd") if deleted else sess["reserved_usd"]
            storage.append({"run_id": d.name, "volume": sess["volume_id"], "phase": sess.get("phase"),
                            "usd": float(usd), "basis": "actual" if deleted else "reserved"})
            continue
        auth_p = d / "governance/authorization.json"
        phase = (json.loads(auth_p.read_text()).get("stage_conditions") or {}).get("phase") if auth_p.is_file() else None
        if out is not None:
            usd = (out.get("money") or {}).get("all_in_usd")
            if usd is None and out.get("provider_resource_created") is False:
                usd = 0.0
            if usd is None:
                unknown.append(d.name)
                continue
            sessions.append({"run_id": d.name, "phase": phase, "usd": float(usd)})
        elif sess.get("provider_resource_created"):
            unknown.append(d.name)
            open_.append(d.name)
    return {"sessions_usd": round(sum(s["usd"] for s in sessions), 4), "sessions": sessions,
            "storage_usd": round(sum(s["usd"] for s in storage), 4), "storage": storage,
            "unknown": unknown, "open": open_}


def remaining_envelope(repo_root: Path = REPO) -> dict:
    g, s = grant(repo_root), spent(repo_root)
    if s["unknown"]:
        raise AuthorizationError(f"runs {s['unknown']} created provider resources and record no closeout cost; "
                                 "reconcile them before anything else is funded")
    used = s["sessions_usd"] + s["storage_usd"]
    by_phase = {p: round(sum(x["usd"] for x in s["sessions"] if x["phase"] == p), 4) for p in g["phases"]}
    return {"ceiling_usd": float(g["ceiling_usd"]), "sessions_usd": s["sessions_usd"],
            "storage_usd": s["storage_usd"], "used_usd": round(used, 4),
            "remaining_usd": round(float(g["ceiling_usd"]) - used, 4), "by_phase": by_phase,
            "sessions": s["sessions"], "storage": s["storage"]}


# ---- the project position, from the provider -------------------------------------------------------------

def provider_snapshot(start: str = "2026-06-01") -> dict:
    """The provider's billing, balance and resources, read now (read-only, $0)."""
    from stages.reasoning_data import h1pa_governance as HG
    return HG.provider_snapshot(start)


def snapshot_problems(snap: dict) -> list[str]:
    from stages.reasoning_data import h1pa_governance as HG
    return HG.snapshot_problems(snap)


def volume_records(repo_root: Path = REPO) -> list[dict]:
    """Every network volume a record in this repository owns or owned, with its
    lifetime and its booked cost: the stage storage records and every volume
    run (`runtime/session.json` of kind network_volume, deleted per its
    tombstone or closeout)."""
    root, out = Path(repo_root), []
    for p in sorted(root.glob("logs/stages/*/*/storage/*.json")):
        d = json.loads(p.read_text())
        if not d.get("volume_id"):
            continue
        ret = d.get("retirement") or {}
        out.append({"volume": d["volume_id"], "size_gb": float(d["size_gb"]), "created": _utc(d["created_utc"]),
                    "deleted": _utc(ret.get("deleted_utc")) if ret.get("status") == "DELETED" else None,
                    "reserved_usd": float(d["reserved_usd"]), "actual_usd": d.get("actual_usd"),
                    "record": str(p.relative_to(root))})
    for p in sorted(root.glob("logs/stages/*/*/runs/*/runtime/session.json")):
        d = json.loads(p.read_text())
        if d.get("kind") != "network_volume" or not d.get("volume_id"):
            continue
        run = p.parents[1]
        tomb = run / "closeout/tombstone.json"
        out_p = run / "closeout/outcome.json"
        deleted = json.loads(tomb.read_text()).get("deleted_utc") if tomb.is_file() else None
        oc = json.loads(out_p.read_text()) if out_p.is_file() else {}
        actual = (oc.get("money") or {}).get("all_in_usd") if oc.get("status") == "VOLUME_DELETED" else None
        out.append({"volume": d["volume_id"], "size_gb": float(d["size_gb"]), "created": _utc(d["created_utc"]),
                    "deleted": _utc(deleted), "reserved_usd": float(d.get("reserved_usd") or 0.0),
                    "actual_usd": actual, "record": str(p.relative_to(root))})
    return out


def _hours_in_day(v: dict, day: datetime, now: datetime) -> float:
    """Whole hours (ceiled, as the provider bills) volume `v` existed in UTC day `day`, up to `now`."""
    start = max(v["created"], day)
    end = min(v["deleted"] or now, day + timedelta(days=1), now)
    return float(math.ceil(max((end - start).total_seconds(), 0.0) / 3600 - 1e-9))


def netted_posted(v: dict, rows: list[dict], records: list[dict], now: datetime) -> dict:
    """The provider-posted volume charge attributable to `v`, conservatively: a
    day's row only on a UTC day `v` was the sole recorded volume on the account,
    and never more than `v`'s own size x hours that day. Everything else stays
    unattributed (it is in the bill either way)."""
    days, gbh, usd = [], 0.0, 0.0
    for r in rows:
        day = datetime.fromisoformat(r["startDate"][:10]).replace(tzinfo=timezone.utc)
        if day + timedelta(days=1) <= v["created"] or (v["deleted"] and day > v["deleted"]):
            continue
        others = [w["volume"] for w in records if w["volume"] != v["volume"]
                  and w["created"] < day + timedelta(days=1) and (w["deleted"] is None or w["deleted"] > day)]
        own = v["size_gb"] * _hours_in_day(v, day, now)
        row_gbh = float(r.get("diskSpaceBilledGb") or 0.0)
        take = 0.0 if others or row_gbh <= 0 else min(row_gbh, own)
        amount = float(r["amount"]) * take / row_gbh if take else 0.0
        days.append({"day": r["startDate"][:10], "row_gb_h": row_gbh, "own_max_gb_h": own,
                     "netted_gb_h": take, "netted_usd": round(amount, 6), "shared_with": others})
        gbh += take
        usd += amount
    return {"netted_gb_h": gbh, "netted_usd": round(usd, 6), "days": days}


def position(repo_root: Path = REPO, *, snapshot: dict | None = None, now: datetime | None = None,
             expect_pods: tuple[str, ...] = ()) -> dict:
    """The project's committed spend as the PROVIDER bills it now (module
    docstring). `expect_pods`: pods the caller owns and expects to be live (a
    launch precheck before create passes none)."""
    root = Path(repo_root)
    g = grant(root)
    snap = snapshot if snapshot is not None else provider_snapshot()
    now = now or _utc(snap.get("fetched_utc")) or datetime.now(timezone.utc)
    bad = snapshot_problems(snap)
    if bad:
        raise AuthorizationError(f"the provider position cannot be confirmed: {bad}; nothing is authorized or created")
    billed_pods = round(sum(float(r["amount"]) for r in snap["pods"]), 4)
    billed_vols = round(sum(float(r["amount"]) for r in snap["networkvolumes"]), 4)
    posted = {r.get("podId") for r in snap["pods"]}
    unposted, never_posted = [], []
    for p in sorted(root.glob("logs/stages/*/*/runs/*/runtime/session.json")):
        sess = json.loads(p.read_text())
        pod = sess.get("pod_id")
        if not pod or pod in posted or not sess.get("provider_resource_created"):
            continue
        run = p.parents[1]
        out_p = run / "closeout/outcome.json"
        out = json.loads(out_p.read_text()) if out_p.is_file() else None
        usd = ((out or {}).get("money") or {}).get("all_in_usd")
        usd = usd if usd is not None else (sess.get("cost") or {}).get("actual_usd")
        ended = _utc((out or {}).get("recorded_utc")) or datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        row = {"run": str(run.relative_to(root)), "pod_id": pod, "usd": usd, "ended_utc": ended.isoformat()}
        if pod in expect_pods:
            continue
        if not sess.get("provider_confirms_gone"):
            raise AuthorizationError(f"{row['run']}: pod {pod} is not confirmed gone and not posted; reconcile first")
        if usd is None:
            raise AuthorizationError(f"{row['run']}: pod {pod} is unposted and its session records no cost")
        (unposted if (now - ended).total_seconds() / 3600 < POSTING_WINDOW_HOURS else never_posted).append(row)
    acct = snap["account"]
    records = volume_records(root)
    by_id = {v["volume"]: v for v in records}
    live_pods = [x.get("id") for x in acct.get("pods") or [] if x.get("id") not in expect_pods]
    if live_pods:
        raise AuthorizationError(f"the provider account holds pods {live_pods}: at most one billing pod, the one "
                                 "being created; reconcile first")
    live_vols = [x.get("id") for x in acct.get("network_volumes") or []]
    foreign = [x for x in live_vols if x not in by_id or by_id[x]["deleted"] is not None]
    if foreign:
        raise AuthorizationError(f"the provider account holds network volumes {foreign} no live record owns")
    vol_rows = sorted(snap["networkvolumes"], key=lambda r: r["startDate"])
    storage = []
    for vid in live_vols:
        v = by_id[vid]
        n = netted_posted(v, vol_rows, records, now)
        future = max(0.0, v["reserved_usd"] - n["netted_usd"])
        storage.append({"volume": vid, "record": v["record"], "basis": "reserved", "obligation_usd": v["reserved_usd"],
                        "netted_posted_usd": n["netted_usd"], "future_usd": round(future, 4), "netting": n["days"]})
    for v in records:
        if v["deleted"] is not None and (now - v["deleted"]).total_seconds() / 3600 < POSTING_WINDOW_HOURS:
            usd = float(v["actual_usd"] if v["actual_usd"] is not None else v["reserved_usd"])
            storage.append({"volume": v["volume"], "record": v["record"], "basis": "deleted inside the posting window",
                            "obligation_usd": usd, "netted_posted_usd": 0.0, "future_usd": round(usd, 4)})
    committed = round(billed_pods + billed_vols + sum(float(u["usd"]) for u in unposted)
                      + sum(s["future_usd"] for s in storage), 4)
    cap = float(g["project_cumulative_cap_usd"])
    return {"fetched_utc": snap.get("fetched_utc"), "provider_billed_usd": round(billed_pods + billed_vols, 4),
            "provider_billed_pods_usd": billed_pods, "provider_billed_volumes_usd": billed_vols,
            "unposted_sessions": unposted, "never_posted_sessions_not_counted": never_posted,
            "storage": storage, "committed_usd": committed, "project_cap_usd": cap,
            "project_remaining_usd": round(cap - committed, 4), "account_pods": live_pods,
            "account_volumes": live_vols, "spend_per_hour": acct.get("current_spend_per_hr"),
            "balance_usd": acct.get("client_balance_usd"), "rule": "mixed_governance.position"}


def write_position(path: Path, repo_root: Path = REPO, *, snapshot: dict | None = None) -> dict:
    """The position as a record (the snapshot it was derived from saved beside it)."""
    from maintenance.consolidation import provider_billing as PB  # noqa: F401 -- the fetcher's owner
    snap = snapshot if snapshot is not None else provider_snapshot()
    pos = position(repo_root, snapshot=snap)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).with_suffix(".snapshot.json").write_text(json.dumps(snap, indent=1) + "\n")
    Path(path).write_text(json.dumps(pos, indent=1, default=str) + "\n")
    return pos


# ---- issue -----------------------------------------------------------------------------------------------

def mc1_top1_record(repo_root: Path = REPO) -> Path:
    st = json.loads((Path(repo_root) / RECOVERY_STUDY).read_text())
    return Path(repo_root) / st["initialization"]["record"]


def funding_problems(kind: str, hard_cap_usd: float, env: dict, pos: dict, g: dict) -> tuple[list[str], bool]:
    """Why a session of `kind` at `hard_cap_usd` cannot be funded (empty when it can), and
    whether this is the first FUNDED issue. The envelope must hold this cap -- plus, for an
    MC1 issue before any MC2 session, MC2's whole cap, and the MC1 volume's reservation
    while no volume record exists -- and the project position must hold the whole
    remaining envelope at the first funded issue, this session's cap afterwards."""
    out = []
    reserve_for_mc2 = (float(g["phases"]["MC2"]["session_hard_cap_usd"])
                       if kind == "search" and not any(s["phase"] == "MC2" for s in env["sessions"]) else 0.0)
    planned_volume = (float(g["phases"]["MC1"]["network_volume"]["reserved_usd"])
                      if kind == "search" and not env["storage"] else 0.0)
    if hard_cap_usd + reserve_for_mc2 + planned_volume > env["remaining_usd"] + 1e-9:
        out.append(f"${hard_cap_usd} (+ ${reserve_for_mc2} kept for MC2, + ${planned_volume} for the MC1 volume) "
                   f"does not fit the ${env['remaining_usd']} left in the envelope")
    #: until a session has been paid for (a $0 dry run does not count) the whole remaining
    #: envelope must fit; the MC1 volume's reservation is already in both `committed` and
    #: `used`, so the sum is the same before and after the volume exists
    first = not any(x["usd"] > 0 for x in env["sessions"])
    need = env["remaining_usd"] if first else hard_cap_usd
    if pos["committed_usd"] + need > pos["project_cap_usd"] + 1e-9:
        out.append(f"HARD BOUNDARY: committed ${pos['committed_usd']} + ${need} "
                   f"({'the whole remaining envelope' if first else 'this session'}) exceeds the "
                   f"${pos['project_cap_usd']} project cap as the provider bills it now")
    if float(pos.get("balance_usd") or 0.0) < hard_cap_usd:
        out.append(f"the provider account holds ${pos.get('balance_usd')}, less than the ${hard_cap_usd} hard cap")
    return out, first


def issue(kind: str, run_id: str, *, expected_usd: float, hard_cap_usd: float, repo_root: Path = REPO,
          note: str = "", out: Path | None = None, snapshot: dict | None = None) -> Path:
    """Write the one-use authorization for `run_id` after checking, in order:
    the grant, the session's phase cap, the cumulative envelope (an MC1 retry
    must leave MC2 fundable), the project cap against the provider's billing
    now (at the first issue the WHOLE remaining envelope must fit, later this
    session's cap), and the account balance. MC2 also needs the MC1 Top-1's
    read-back record."""
    if kind not in KINDS:
        raise AuthorizationError(f"unknown kind {kind!r}")
    root = Path(repo_root)
    g = grant(root)
    if g.get("closed"):
        raise AuthorizationError(f"the mixed-calibration grant is closed ({g['closed']})")
    phase = KINDS[kind]
    cap = float(g["phases"][phase]["session_hard_cap_usd"])
    if hard_cap_usd > cap:
        raise AuthorizationError(f"${hard_cap_usd} exceeds {phase}'s session cap ${cap}")
    if kind == "recovery":
        rec = mc1_top1_record(root)
        if not rec.is_file() or json.loads(rec.read_text()).get("read_back", {}).get("verified") is not True:
            raise AuthorizationError(f"{rec}: no verified MC1 Top-1 read-back; MC2 starts only after it")
    env = remaining_envelope(root)
    pos = position(root, snapshot=snapshot)
    problems, first = funding_problems(kind, hard_cap_usd, env, pos, g)
    if problems:
        raise AuthorizationError("; ".join(problems) + "; nothing is issued")
    files = harness_files(kind, root)
    harness = harness_source_digest(root, files=files)
    ident = identity_hash(kind, root)
    path = Path(out) if out is not None else root / authorization_path(run_id)
    if path.exists():
        raise AuthorizationError(f"{path} exists; one artifact per run, never re-issued")
    auth = MixedAuthorization(
        authorization_id=f"{PLAN_IDS[kind]}.{run_id}",
        granted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        granted_by=g["granted_by"], plan_id=PLAN_IDS[kind], plan_hash=ident,
        expected_usd=round(expected_usd, 4), hard_cap_usd=round(hard_cap_usd, 4), authorized_stages=(),
        stage_conditions={"kind": kind, "phase": phase, "grant_sha256": sha256_file(root / GRANT),
                          "envelope": json.dumps(env), "position": json.dumps(pos, default=str),
                          "first_issue_checked_whole_envelope": first},
        scope_note=note or g["phases"][phase]["scope"], authorized_session_commit=None,
        harness_source_digest=harness["digest"], harness_source_files=files,
        per_launch_hard_usd=round(hard_cap_usd, 4), provenance_commit=G.git_head(root),
        action_policy=POLICIES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth.as_dict(), indent=1) + "\n")
    load_for(kind, path).require_plan(ident)
    return path
